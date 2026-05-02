from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42

FEATURES = ["Graphene_Surface_Area_m2_g","Polymer_Weight_pct","Doping_Level","Functional_Group_Electronegativity","Polymer_Type","Electrolyte_Type"]
TARGET = "Specific_Capacitance_F_g"


def _metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_publication_workflow(local_csv: str) -> dict:
    df = pd.read_csv(local_csv).dropna(subset=FEATURES+[TARGET]).reset_index(drop=True)
    if len(df) < 80:
        raise ValueError(f"Need >=80 curated rows, got {len(df)}")

    if "Publication_Year" in df.columns:
        df = df.sort_values("Publication_Year").reset_index(drop=True)
    split = int(len(df)*0.8)
    train_df, ext_df = df.iloc[:split], df.iloc[split:]

    num = ["Graphene_Surface_Area_m2_g","Polymer_Weight_pct","Doping_Level","Functional_Group_Electronegativity"]
    cat = ["Polymer_Type","Electrolyte_Type"]
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([("preprocessor", pre), ("model", XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1))])
    grid = {"model__n_estimators":[200,400],"model__max_depth":[3,5],"model__learning_rate":[0.03,0.08],"model__subsample":[0.8,1.0],"model__colsample_bytree":[0.8,1.0]}

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, scoring="r2", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(train_df[FEATURES], train_df[TARGET])

    pred_ext = gs.best_estimator_.predict(ext_df[FEATURES])
    ext_metrics = _metrics(ext_df[TARGET], pred_ext)

    preds = np.vstack([est.predict(ext_df[FEATURES]) for est in gs.best_estimator_.named_steps['model'].estimators_]) if hasattr(gs.best_estimator_.named_steps['model'], 'estimators_') else np.vstack([pred_ext])
    uncertainty = float(np.mean(np.std(preds, axis=0)))

    report = {
        "rows_total": int(len(df)),
        "rows_train_temporal": int(len(train_df)),
        "rows_external_holdout": int(len(ext_df)),
        "best_params": gs.best_params_,
        "external_holdout": ext_metrics,
        "mean_prediction_std": uncertainty,
    }
    print("Publication evaluation report:", report)
    return report


if __name__ == "__main__":
    raise SystemExit("Use run_publication_workflow(local_csv=...) from colab_single_cell_runner.py")
