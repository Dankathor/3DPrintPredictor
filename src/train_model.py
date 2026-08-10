"""
train_model.py
--------------
Trains and evaluates a Random Forest regression model that predicts FDM
surface roughness (roughness_avg_um) from six pre-print process settings.

Run from the project root:
    python src/train_model.py
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CLEANED_PATH = pathlib.Path("data/cleaned/cleaned_3d_printing_data.csv")
MODELS_DIR = pathlib.Path("models")
MODEL_PATH = MODELS_DIR / "roughness_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["printer_type", "material", "infill_pattern"]
NUMERIC_FEATURES = ["layer_thickness_mm", "infill_density_pct", "speed_mm_s"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "roughness_avg_um"


def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
    }


def print_metrics(metrics: dict) -> None:
    """Print the three primary regression metrics."""
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categorical features and pass numeric features through."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


df_full = pd.read_csv(CLEANED_PATH)
total_cleaned = len(df_full)
print(f"Rows in cleaned CSV                  : {total_cleaned}")

# The dataset contains zero roughness values whose meaning is not explained by
# the source documentation. Those records are retained in the cleaned dataset
# but excluded from roughness model development.
df = df_full[df_full["roughness_valid"].eq(True)].copy()
excluded = total_cleaned - len(df)

print(f"Rows excluded (roughness_valid=False): {excluded}")
print(f"Rows available for modeling          : {len(df)}")

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)
print(f"\nTraining rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

print("\n" + "=" * 60)
print("BASELINE (DummyRegressor — predict mean)")
print("=" * 60)

baseline_pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor()),
        ("model", DummyRegressor(strategy="mean")),
    ]
)
baseline_pipeline.fit(X_train, y_train)
baseline_predictions = baseline_pipeline.predict(X_test)
baseline_metrics = evaluate(y_test, baseline_predictions)
print_metrics(baseline_metrics)

print("\n" + "=" * 60)
print("RANDOM FOREST (n_estimators=200)")
print("=" * 60)

rf_pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)
rf_metrics = evaluate(y_test, rf_predictions)
print_metrics(rf_metrics)

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION (training set)")
print("=" * 60)

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

cv_results = cross_validate(
    cv_pipeline,
    X_train,
    y_train,
    cv=kfold,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
    return_train_score=False,
    n_jobs=-1,
)

cv_mae = float(-cv_results["test_mae"].mean())
cv_rmse = float(-cv_results["test_rmse"].mean())
cv_r2 = float(cv_results["test_r2"].mean())

print(f"  CV MAE  (avg): {cv_mae:.4f}")
print(f"  CV RMSE (avg): {cv_rmse:.4f}")
print(f"  CV R²   (avg): {cv_r2:.4f}")

cv_metrics = {
    "mae": round(cv_mae, 6),
    "rmse": round(cv_rmse, 6),
    "r2": round(cv_r2, 6),
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(rf_pipeline, MODEL_PATH)
print(f"\nModel saved to : {MODEL_PATH.resolve()}")

metrics_payload = {
    "model_type": "RandomForestRegressor",
    "random_state": RANDOM_STATE,
    "features": ALL_FEATURES,
    "target": TARGET,
    "total_cleaned_records": total_cleaned,
    "records_used_for_modeling": int(len(df)),
    "excluded_invalid_roughness": int(excluded),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "baseline": {
        "strategy": "mean",
        **baseline_metrics,
    },
    "random_forest": {
        "n_estimators": 200,
        **rf_metrics,
    },
    "cross_validation": {
        "n_splits": 5,
        "shuffle": True,
        **cv_metrics,
    },
}

METRICS_PATH.write_text(
    json.dumps(metrics_payload, indent=2),
    encoding="utf-8",
)
print(f"Metrics saved to: {METRICS_PATH.resolve()}")

print("\n" + "=" * 60)
print("Training complete.")
print("=" * 60)
