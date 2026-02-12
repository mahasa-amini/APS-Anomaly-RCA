import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path

# -----------------------------
# Resolve project root
# -----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../APS_project
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# -----------------------------
# Load X and y from Phase 1
# -----------------------------
X_path = DATA_PROCESSED / "X_clean.npy"
y_path = DATA_PROCESSED / "y_clean.npy"

print("Loading:", X_path)
print("Loading:", y_path)

X = np.load(X_path)
y = np.load(y_path)

num_features = X.shape[1]
feature_names = np.array([f"f_{i}" for i in range(num_features)])

X = pd.DataFrame(X, columns=feature_names)

# Select only anomalies
X_anom = X[y == 1]
print("Anomaly samples for SHAP:", X_anom.shape)

# -----------------------------
# Load trained model dictionary
# -----------------------------
model_path = PROJECT_ROOT / "xgboost_aps_model.pkl"
print("Loading XGBoost model from:", model_path)

loaded_obj = joblib.load(model_path)

# Extract real XGBoost model
model = loaded_obj["model"]
print("Model extracted from dictionary!")

# -----------------------------
# Compute SHAP values
# -----------------------------
print("\nComputing SHAP values for anomaly samples...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_anom)

print("SHAP values computed!")

# -----------------------------
# Save SHAP outputs
# -----------------------------
shap_values_path = DATA_PROCESSED / "shap_values_anom.npy"
feature_names_path = DATA_PROCESSED / "shap_feature_names.npy"

np.save(shap_values_path, shap_values)
np.save(feature_names_path, feature_names)

print("\nSaved:")
print(" -", shap_values_path)
print(" -", feature_names_path)

print("\nAll done!")
