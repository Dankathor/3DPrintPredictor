"""
evaluate_model.py
-----------------
Generates reproducible evaluation artifacts for the trained Random Forest
regression model that predicts FDM surface roughness.

This script does NOT retrain or tune the model.  It loads the already-fitted
pipeline from disk, recreates the same test split used during training, and
evaluates the model against that previously unseen test data.

Run from the project root:
    python src/evaluate_model.py
"""

import pathlib

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# Use the non-interactive Agg backend so plots are written to disk without
# attempting to open a display window.
matplotlib.use("Agg")

# Paths

CLEANED_PATH  = pathlib.Path("data/cleaned/cleaned_3d_printing_data.csv")
MODEL_PATH    = pathlib.Path("models/roughness_model.joblib")
PREDICTIONS_PATH = pathlib.Path("models/test_predictions.csv")
DOCS_DIR      = pathlib.Path("documentation")

RANDOM_STATE  = 42

# Feature and target definitions — must match train_model.py exactly

CATEGORICAL_FEATURES = ["printer_type", "material", "infill_pattern"]
NUMERIC_FEATURES     = ["layer_thickness_mm", "infill_density_pct", "speed_mm_s"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = "roughness_avg_um"

# Load data and recreate the same 80/20 test split used during training
#
# The random_state and test_size must be identical to train_model.py so that
# the rows returned as X_test here are exactly the rows the model has never
# seen before.

df_full = pd.read_csv(CLEANED_PATH)

# Exclude invalid roughness records for the same reason as in training:
# zero/negative roughness values indicate failed measurements and are not
# physically meaningful.
df = df_full[df_full["roughness_valid"] == True].copy()

X = df[ALL_FEATURES]
y = df[TARGET]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"Total modelling rows : {len(df)}")
print(f"Test set rows        : {len(X_test)}")

# Load the trained pipeline and generate predictions

pipeline = joblib.load(MODEL_PATH)
y_pred   = pipeline.predict(X_test)

# Build evaluation table

eval_df = X_test.copy().reset_index(drop=True)
eval_df["actual_roughness_um"]    = y_test.values
eval_df["predicted_roughness_um"] = y_pred
eval_df["residual_um"]            = eval_df["actual_roughness_um"] - eval_df["predicted_roughness_um"]
eval_df["absolute_error_um"]      = eval_df["residual_um"].abs()

eval_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"\nPredictions saved to: {PREDICTIONS_PATH.resolve()}")

# Accuracy metrics

mae        = mean_absolute_error(y_test, y_pred)
rmse       = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2         = r2_score(y_test, y_pred)
mean_resid = float(eval_df["residual_um"].mean())
med_abs_err = median_absolute_error(y_test, y_pred)
max_abs_err = float(eval_df["absolute_error_um"].max())

print("\n" + "=" * 60)
print("ACCURACY METRICS")
print("=" * 60)
print(f"  MAE                    : {mae:.4f} µm")
print(f"  RMSE                   : {rmse:.4f} µm")
print(f"  R²                     : {r2:.4f}")
print(f"  Mean residual          : {mean_resid:.4f} µm")
print(f"  Median absolute error  : {med_abs_err:.4f} µm")
print(f"  Max absolute error     : {max_abs_err:.4f} µm")

# Worst predictions

print("\n" + "=" * 60)
print("FIVE LARGEST ABSOLUTE ERRORS")
print("=" * 60)
worst = eval_df.nlargest(5, "absolute_error_um")[
    ALL_FEATURES + ["actual_roughness_um", "predicted_roughness_um",
                    "residual_um", "absolute_error_um"]
]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print(worst.to_string(index=False))

# Prediction accuracy bands

n = len(eval_df)
bands = [0.5, 1.0, 1.5, 2.0]

print("\n" + "=" * 60)
print("PREDICTION ACCURACY BANDS")
print("=" * 60)
for band in bands:
    within = (eval_df["absolute_error_um"] <= band).sum()
    pct    = 100.0 * within / n
    print(f"  Within ±{band:.1f} µm : {within:3d} / {n}  ({pct:.1f}%)")

# Figure 1: Predicted vs Actual scatter

DOCS_DIR.mkdir(parents=True, exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(7, 6))

ax1.scatter(
    eval_df["actual_roughness_um"],
    eval_df["predicted_roughness_um"],
    alpha=0.65,
    edgecolors="steelblue",
    facecolors="lightsteelblue",
    linewidths=0.6,
    s=60,
    label="Test predictions",
)

# Perfect-prediction reference line
all_vals = pd.concat([eval_df["actual_roughness_um"],
                       eval_df["predicted_roughness_um"]])
lo, hi   = all_vals.min(), all_vals.max()
ax1.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.5,
         linestyle="--", label="Perfect prediction")

ax1.set_xlabel("Actual Surface Roughness (µm)", fontsize=12)
ax1.set_ylabel("Predicted Surface Roughness (µm)", fontsize=12)
ax1.set_title("Random Forest: Predicted vs Actual Surface Roughness\n"
              f"(R² = {r2:.3f}, MAE = {mae:.3f} µm)", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.5)
fig1.tight_layout()

pva_path = DOCS_DIR / "predicted_vs_actual.png"
fig1.savefig(pva_path, dpi=150)
plt.close(fig1)
print(f"\nFigure saved : {pva_path.resolve()}")

# Figure 2: Residual distribution histogram

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.hist(
    eval_df["residual_um"],
    bins=20,
    color="steelblue",
    edgecolor="white",
    linewidth=0.5,
    alpha=0.85,
)

ax2.axvline(0, color="crimson", linewidth=1.8, linestyle="--",
            label="Zero residual")

ax2.set_xlabel("Residual (Actual − Predicted) µm", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Random Forest: Distribution of Residuals\n"
              f"(Mean residual = {mean_resid:.3f} µm)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.5)
fig2.tight_layout()

resid_path = DOCS_DIR / "residual_distribution.png"
fig2.savefig(resid_path, dpi=150)
plt.close(fig2)
print(f"Figure saved : {resid_path.resolve()}")

print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)
