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
MIN_ROWS_FOR_MODELING = 80

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
    if local_csv:
        return validate_dataset(pd.read_csv(local_csv))
    if urls:
        for u in urls:
            try:
                r = requests.get(u, timeout=25)
                r.raise_for_status()
                return validate_dataset(pd.read_csv(io.StringIO(r.text)))
            except Exception as exc:
                print(f"Failed URL: {u}\n  reason: {exc}")
    raise ValueError("No real curated dataset was loaded. Provide --local_csv or remote URLs.")


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    out = df.copy().dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(out) < MIN_ROWS_FOR_MODELING:
        raise ValueError(f"Dataset has only {len(out)} rows; require >= {MIN_ROWS_FOR_MODELING} rows for publication workflow.")
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
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    pre = ColumnTransformer(transformers=[("num", StandardScaler(), NUM_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    rf_pipe = Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    xgb_pipe = Pipeline([("preprocessor", pre), ("model", XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist", n_jobs=-1, verbosity=0))])

    rf_grid = {"model__n_estimators": [120, 240], "model__max_depth": [None, 8, 16], "model__min_samples_split": [2, 5]}
    xgb_grid = {"model__n_estimators": [120, 240], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.08], "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0]}

    cv_folds = min(5, max(3, len(X_train) // 20))
    rf_search = GridSearchCV(rf_pipe, rf_grid, scoring="r2", cv=cv_folds, n_jobs=-1, verbose=1)
    xgb_search = GridSearchCV(xgb_pipe, xgb_grid, scoring="r2", cv=cv_folds, n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)

    models = {"RandomForest": rf_search.best_estimator_, "XGBoost": xgb_search.best_estimator_}
    metrics_df = pd.DataFrame({k: regression_metrics(y_test, m.predict(X_test)) for k, m in models.items()}).T.sort_values("R2", ascending=False)
    print(metrics_df)

    best_name = metrics_df.index[0]
    best_model = models[best_name]
    y_pred = best_model.predict(X_test)
    parity_plot(y_test.values, y_pred, best_name)

    X_test_t = best_model.named_steps["preprocessor"].transform(X_test)
    feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    shap_values = shap.TreeExplainer(best_model.named_steps["model"]).shap_values(X_test_t)

    plt.figure(figsize=(8.5, 6), dpi=160)
    shap.summary_plot(shap_values, X_test_t, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig("shap_summary.png", bbox_inches="tight"); plt.show()

    shap_imp = pd.DataFrame({"Feature": feat_names, "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)}).sort_values("MeanAbsSHAP", ascending=False)
    plt.figure(figsize=(8, 5.8), dpi=160)
    sns.barplot(data=shap_imp.head(12), x="MeanAbsSHAP", y="Feature", hue="Feature", dodge=False, palette="viridis", legend=False)
    plt.tight_layout(); plt.savefig("feature_importance_shap_bar.png", bbox_inches="tight"); plt.show()

    return df, metrics_df, best_model
