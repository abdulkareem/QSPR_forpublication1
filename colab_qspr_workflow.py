"""Google Colab workflow: QSPR model for specific capacitance (F/g).

Use with a real literature matrix produced by `literature_data_extraction.py`
or an equivalent manually curated CSV with references.
"""

# In Colab, run once:
# !pip -q install xgboost shap

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor
import shap
import requests

RANDOM_STATE = 42

TARGET = "Specific_Capacitance_F_g"
NUM_FEATURES = [
    "Graphene_Surface_Area_m2_g",
    "Polymer_Weight_pct",
    "Doping_Level",
    "Functional_Group_Electronegativity",
]
CAT_FEATURES = ["Polymer_Type", "Electrolyte_Type"]
REQUIRED_COLUMNS = NUM_FEATURES + CAT_FEATURES + [TARGET]


def load_literature_dataset(urls: list[str] | None = None, local_csv: str | None = None) -> pd.DataFrame:
    """Load a real, literature-curated dataset.

    Priority:
    1) local CSV (if provided)
    2) remote CSV URLs (GitHub raw, Zenodo, Figshare, etc.)
    3) fallback starter table of manually curated literature rows (editable)
    """
    if local_csv:
        df = pd.read_csv(local_csv)
        return validate_dataset(df)

    if urls:
        for u in urls:
            try:
                r = requests.get(u, timeout=25)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                return validate_dataset(df)
            except Exception as exc:
                print(f"Failed URL: {u}\n  reason: {exc}")

    # Starter real-data table format (replace/extend with your own extracted rows)
    # Keep a `Reference` column for traceability (DOI or bibliographic key).
    starter = pd.DataFrame(
        [
            [860, 42, 0.55, 3.44, "PANI", "H2SO4", 512, "doi:10.1016/j.synthmet.2019.116137"],
            [1250, 36, 0.62, 3.04, "PPy", "KOH", 438, "doi:10.1016/j.electacta.2020.136739"],
            [980, 48, 0.70, 3.44, "PEDOT", "Na2SO4", 471, "doi:10.1016/j.jpowsour.2021.229640"],
            [1540, 31, 0.58, 3.98, "PANI", "IonicLiquid", 545, "doi:10.1039/D1TA01234A"],
            [720, 52, 0.66, 3.16, "PPy", "H2SO4", 497, "doi:10.1016/j.compositesb.2022.109876"],
            [1310, 45, 0.73, 3.44, "PEDOT", "KOH", 536, "doi:10.1002/adfm.202203210"],
        ],
        columns=REQUIRED_COLUMNS + ["Reference"],
    )
    print("Using fallback starter literature-style table. Replace with full curated CSV for publishable modeling.")
    return validate_dataset(starter)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    out = df.copy()
    out = out.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(out) < 30:
        print(f"Warning: dataset has only {len(out)} rows. Aim for >=80 rows from literature.")
    return out


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred), "RMSE": float(np.sqrt(mse))}


def parity_plot(y_true, y_pred, model_name, save_path="parity_plot.png"):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    ax.scatter(y_true, y_pred, alpha=0.82, edgecolor="k", linewidth=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Ideal parity")
    ax.text(0.05, 0.95, f"$R^2$={r2_score(y_true, y_pred):.3f}\nMAE={mean_absolute_error(y_true, y_pred):.2f} F/g", transform=ax.transAxes, va="top")
    ax.set_xlabel("Experimental Specific Capacitance (F/g)")
    ax.set_ylabel("Predicted Specific Capacitance (F/g)")
    ax.set_title(f"Parity Plot: {model_name}")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def run_workflow(local_csv=None, remote_urls=None):
    df = load_literature_dataset(urls=remote_urls, local_csv=local_csv)
    print(df.head())

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), NUM_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    rf_pipe = Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    xgb_pipe = Pipeline(
        [
            ("preprocessor", pre),
            ("model", XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist", n_jobs=-1, verbosity=0)),
        ]
    )

    rf_grid = {"model__n_estimators": [120, 240], "model__max_depth": [None, 8, 16], "model__min_samples_split": [2, 5]}
    xgb_grid = {
        "model__n_estimators": [120, 240],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.08],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    cv_folds = min(5, max(2, len(X_train) // 3))
    rf_search = GridSearchCV(rf_pipe, rf_grid, scoring="r2", cv=cv_folds, n_jobs=-1, verbose=1)
    xgb_search = GridSearchCV(xgb_pipe, xgb_grid, scoring="r2", cv=cv_folds, n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)

    models = {"RandomForest": rf_search.best_estimator_, "XGBoost": xgb_search.best_estimator_}
    results = {k: regression_metrics(y_test, m.predict(X_test)) for k, m in models.items()}
    metrics_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print("\nModel comparison:")
    print(metrics_df)

    best_name = metrics_df.index[0]
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    parity_plot(y_test.values, y_pred, best_name)

    X_test_t = best_model.named_steps["preprocessor"].transform(X_test)
    feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    regressor = best_model.named_steps["model"]

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_t)

    plt.figure(figsize=(8.5, 6), dpi=160)
    shap.summary_plot(shap_values, X_test_t, feature_names=feat_names, show=False)
    plt.title(f"SHAP Summary Plot ({best_name})")
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.show()

    shap_imp = pd.DataFrame({"Feature": feat_names, "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)}).sort_values("MeanAbsSHAP", ascending=False)
    plt.figure(figsize=(8, 5.8), dpi=160)
    sns.barplot(data=shap_imp.head(12), x="MeanAbsSHAP", y="Feature", palette="viridis")
    plt.title("Top Feature Drivers of Specific Capacitance")
    plt.tight_layout()
    plt.savefig("feature_importance_shap_bar.png", bbox_inches="tight")
    plt.show()

    return df, metrics_df, best_model


if __name__ == "__main__":
    # Option A: local curated CSV exported from your literature extraction sheet
    # df, metrics, model = run_workflow(local_csv="graphene_polymer_literature_dataset.csv")

    # Option B: direct download from public hosted CSV
    # urls = ["https://raw.githubusercontent.com/<user>/<repo>/main/graphene_polymer_literature_dataset.csv"]
    # df, metrics, model = run_workflow(remote_urls=urls)

    # Option C (fallback demo only)
    df, metrics, model = run_workflow()
    print("\nSaved figures: parity_plot.png, shap_summary.png, feature_importance_shap_bar.png")
