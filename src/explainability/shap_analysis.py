import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.training.feature_engineering import create_features


# =========================
# LOAD PIPELINE MODEL
# =========================
pipeline = joblib.load("models/best_model.pkl")

# Extract parts
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/loan_train.csv")
df = create_features(df)

X = df.drop(columns=["loan_id", "application_date", "default_flag"])

# =========================
# SAMPLE DATA
# =========================
X_sample = X.sample(500, random_state=42)

# =========================
# PREPROCESS DATA
# =========================
X_transformed = preprocessor.transform(X_sample)

# Convert to dense if sparse
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# =========================
# FEATURE NAMES (IMPORTANT)
# =========================
feature_names = preprocessor.get_feature_names_out()

X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# =========================
# SHAP EXPLAINER (MODEL ONLY)
# =========================
explainer = shap.Explainer(model, X_transformed_df)

shap_values = explainer(X_transformed_df)

# =========================
# GLOBAL IMPORTANCE
# =========================
shap.plots.bar(shap_values, max_display=10)
plt.show()

# =========================
# SUMMARY PLOT
# =========================
shap.plots.beeswarm(shap_values, max_display=10)
plt.show()

# =========================
# LOCAL EXPLANATIONS
# =========================
for i in range(2):
    shap.plots.waterfall(shap_values[i])
    plt.show()