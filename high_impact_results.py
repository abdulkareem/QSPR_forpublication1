"""High-impact results generator for manuscript submission.

Produces advanced validation and robustness outputs:
- temporal split validation
- bootstrap confidence intervals (R2/RMSE)
- ablation study
- permutation importance
- partial dependence plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="high_impact_outputs")
    return p.parse_args()


def build_pipeline(num, cat):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_estimators=260, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])


def main():
    args = parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    target = "Specific_Capacitance_F_g" if "Specific_Capacitance_F_g" in df.columns else "Specific_Capacitance_Fg"
    candidates = ["Polymer_Type", "Polymer_Weight_pct", "Polymer_wt_percent", "Specific_Surface_Area_m2g", "Graphene_Surface_Area_m2_g", "Pore_Size_nm", "Electrolyte_Type", "Scan_Rate_mVs", "Doping_Level"]
    features = [c for c in candidates if c in df.columns]
    df = df.dropna(subset=[target] + features).copy()

    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]

    # temporal split if year exists
    if "Publication_Year" in df.columns:
        cut = int(df["Publication_Year"].quantile(0.8))
        train_df = df[df["Publication_Year"] <= cut]
        test_df = df[df["Publication_Year"] > cut]
        if len(test_df) < 10:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    Xtr, ytr = train_df[features], train_df[target]
    Xte, yte = test_df[features], test_df[target]

    pipe = build_pipeline(num, cat)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    r2 = r2_score(yte, pred); rmse = np.sqrt(mean_squared_error(yte, pred))

    with open(out / "metrics.txt", "w") as f:
        f.write(f"R2={r2:.4f}\nRMSE={rmse:.4f}\nN_train={len(Xtr)}\nN_test={len(Xte)}\n")

    # bootstrap CI
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(100):
        idx = rng.choice(len(Xtr), len(Xtr), replace=True)
        Xb, yb = Xtr.iloc[idx], ytr.iloc[idx]
        pipe_b = build_pipeline(num, cat)
        pipe_b.fit(Xb, yb)
        pb = pipe_b.predict(Xte)
        boots.append((r2_score(yte, pb), np.sqrt(mean_squared_error(yte, pb))))
    bdf = pd.DataFrame(boots, columns=["R2", "RMSE"])
    bdf.to_csv(out / "bootstrap_metrics.csv", index=False)

    # ablation (drop one feature at a time)
    rows = []
    for fdrop in [None] + features:
        feats = [f for f in features if f != fdrop]
        n2 = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        c2 = [c for c in feats if c not in n2]
        p2 = build_pipeline(n2, c2)
        p2.fit(Xtr[feats], ytr)
        pp = p2.predict(Xte[feats])
        rows.append({"Dropped": fdrop or "None", "R2": r2_score(yte, pp), "RMSE": np.sqrt(mean_squared_error(yte, pp))})
    adf = pd.DataFrame(rows).sort_values("R2", ascending=False)
    adf.to_csv(out / "ablation_study.csv", index=False)

    # permutation importance
    perm = permutation_importance(pipe, Xte, yte, n_repeats=20, random_state=42, scoring="r2")
    imp = pd.DataFrame({"Feature": features, "ImportanceMean": perm.importances_mean, "ImportanceStd": perm.importances_std}).sort_values("ImportanceMean", ascending=False)
    imp.to_csv(out / "permutation_importance.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8,5.8), dpi=160)
    sns.barplot(data=imp, x="ImportanceMean", y="Feature", palette="magma")
    plt.title("Permutation Importance (R2 drop)")
    plt.tight_layout(); plt.savefig(out / "fig_perm_importance.png", bbox_inches="tight")

    # partial dependence for top 2 numeric features
    top_num = [f for f in imp["Feature"].tolist() if f in num][:2]
    if top_num:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        PartialDependenceDisplay.from_estimator(pipe, Xte, top_num, ax=ax)
        plt.tight_layout(); plt.savefig(out / "fig_partial_dependence.png", bbox_inches="tight")

    print(f"High-impact outputs saved in {out.resolve()}")


if __name__ == "__main__":
    main()
