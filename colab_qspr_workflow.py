"""
Google Colab-ready workflow for ML-aided QSPR prediction of specific capacitance
in graphene-conductive polymer nanocomposites.

Target system: Graphene + (PANI, PPy, PEDOT) composites.
"""

# =============================
# 0) Optional: install deps in Colab
# =============================
# In a new Colab notebook cell, uncomment and run:
# !pip -q install xgboost shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =============================
# 1) Simulate a CSV-style dataset
# =============================
def simulate_dataset(n_samples: int = 280) -> pd.DataFrame:
    """Generate a physically plausible synthetic dataset for demonstration."""
    polymers = np.random.choice(["PANI", "PPy", "PEDOT"], size=n_samples, p=[0.4, 0.35, 0.25])
    electrolytes = np.random.choice(["H2SO4", "KOH", "Na2SO4", "IonicLiquid"], size=n_samples)

    graphene_surface_area = np.random.uniform(120, 1800, size=n_samples)  # m^2/g
    polymer_weight_pct = np.random.uniform(10, 85, size=n_samples)         # wt%
    doping_level = np.random.uniform(0.0, 1.0, size=n_samples)             # fraction
    functional_group_en = np.random.uniform(2.5, 4.0, size=n_samples)      # Pauling scale proxy

    polymer_factor = pd.Series(polymers).map({"PANI": 1.08, "PPy": 1.00, "PEDOT": 1.12}).values
    electrolyte_factor = pd.Series(electrolytes).map({"H2SO4": 1.10, "KOH": 1.03, "Na2SO4": 0.96, "IonicLiquid": 1.05}).values

    # Physics-informed trend with optimum around mid polymer loading
    loading_term = -0.085 * (polymer_weight_pct - 48) ** 2 + 220
    area_term = 0.075 * graphene_surface_area
    doping_term = 140 * doping_level
    en_term = 70 * (functional_group_en - 2.5)
    interaction = 0.025 * graphene_surface_area * doping_level

    noise = np.random.normal(0, 25, size=n_samples)
    specific_capacitance = (loading_term + area_term + doping_term + en_term + interaction) * polymer_factor * electrolyte_factor + noise
    specific_capacitance = np.clip(specific_capacitance, 20, None)

    return pd.DataFrame(
        {
            "Graphene_Surface_Area_m2_g": graphene_surface_area,
            "Polymer_Weight_pct": polymer_weight_pct,
            "Doping_Level": doping_level,
            "Functional_Group_Electronegativity": functional_group_en,
            "Polymer_Type": polymers,
            "Electrolyte_Type": electrolytes,
            "Specific_Capacitance_F_g": specific_capacitance,
        }
    )


df = simulate_dataset()
df.to_csv("graphene_polymer_qspr_dataset.csv", index=False)
print("Saved dataset: graphene_polymer_qspr_dataset.csv")
print(df.head())


# =============================
# 2) Load dataset (as in Colab)
# =============================
df = pd.read_csv("graphene_polymer_qspr_dataset.csv")

X = df.drop(columns=["Specific_Capacitance_F_g"])
y = df["Specific_Capacitance_F_g"]

num_features = [
    "Graphene_Surface_Area_m2_g",
    "Polymer_Weight_pct",
    "Doping_Level",
    "Functional_Group_Electronegativity",
]
cat_features = ["Polymer_Type", "Electrolyte_Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# =============================
# 3) Model definitions + tuning
# =============================
rf_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ]
)

xgb_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                tree_method="hist",  # low-memory efficient
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    ]
)

rf_grid = {
    "model__n_estimators": [150, 300],
    "model__max_depth": [None, 8, 16],
    "model__min_samples_split": [2, 5],
}

xgb_grid = {
    "model__n_estimators": [150, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.08, 0.15],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

rf_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

xgb_search = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=xgb_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

rf_search.fit(X_train, y_train)
xgb_search.fit(X_train, y_train)

models = {
    "RandomForest": rf_search.best_estimator_,
    "XGBoost": xgb_search.best_estimator_,
}


# =============================
# 4) Evaluate models
# =============================
def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
    }

results = {}
for name, model in models.items():
    pred = model.predict(X_test)
    results[name] = regression_metrics(y_test, pred)

metrics_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print("\nModel comparison (test set):")
print(metrics_df)

best_name = metrics_df.index[0]
best_model = models[best_name]
print(f"\nBest model: {best_name}")


# =============================
# 5) Publication-quality parity plot
# =============================
def parity_plot(y_true, y_pred, model_name, save_path="parity_plot.png"):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)

    ax.scatter(y_true, y_pred, alpha=0.8, edgecolor="k", linewidth=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Ideal parity")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f} F/g", transform=ax.transAxes, va="top")

    ax.set_xlabel("Experimental Specific Capacitance (F/g)")
    ax.set_ylabel("Predicted Specific Capacitance (F/g)")
    ax.set_title(f"Parity Plot: {model_name}")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()

best_pred = best_model.predict(X_test)
parity_plot(y_test.values, best_pred, best_name)


# =============================
# 6) Feature importance + SHAP
# =============================
# Transform features once for explainability
X_train_t = best_model.named_steps["preprocessor"].transform(X_train)
X_test_t = best_model.named_steps["preprocessor"].transform(X_test)

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
regressor = best_model.named_steps["model"]

# SHAP explainer for tree models
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_test_t)

# SHAP summary (global impact)
plt.figure(figsize=(8.5, 6), dpi=160)
shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False)
plt.title(f"SHAP Summary Plot ({best_name})")
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()

# Mean absolute SHAP bar chart (publication-friendly)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_imp = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_shap}).sort_values(
    "MeanAbsSHAP", ascending=False
)

plt.figure(figsize=(8, 5.8), dpi=160)
sns.barplot(data=shap_imp.head(12), x="MeanAbsSHAP", y="Feature", palette="viridis")
plt.xlabel("Mean |SHAP value| (impact on prediction)")
plt.ylabel("Descriptor")
plt.title("Top Feature Drivers of Specific Capacitance")
plt.tight_layout()
plt.savefig("feature_importance_shap_bar.png", bbox_inches="tight")
plt.show()

print("\nSaved figures: parity_plot.png, shap_summary.png, feature_importance_shap_bar.png")
