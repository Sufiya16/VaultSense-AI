import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from src.training.feature_engineering import create_features


# =====================================
# LOAD MODEL
# =====================================

model_pipeline = joblib.load(
    "models/best_model.pkl"
)

print("Model Loaded Successfully")

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/raw/loan_train.csv"
)

print("Dataset Loaded")

# =====================================
# FEATURE ENGINEERING
# =====================================

df = create_features(df)

# =====================================
# REMOVE UNUSED COLUMNS
# =====================================

X = df.drop(
    columns=[
        'loan_id',
        'application_date',
        'default_flag'
    ]
)

y = df['default_flag']

print("Feature Engineering Complete")

# =====================================
# PREPROCESS DATA
# =====================================

preprocessor = model_pipeline.named_steps['preprocessor']

model = model_pipeline.named_steps['model']

X_processed = preprocessor.transform(X)

print("Preprocessing Complete")

# =====================================
# GET FEATURE NAMES
# =====================================

categorical_features = preprocessor.named_transformers_[
    'cat'
].named_steps['encoder'].get_feature_names_out()

numeric_features = preprocessor.transformers_[0][2]

all_features = np.concatenate([
    numeric_features,
    categorical_features
])

print("Feature Names Extracted")

# =====================================
# CREATE SHAP EXPLAINER
# =====================================

explainer = shap.Explainer(
    model,
    X_processed
)

print("SHAP Explainer Created")

# =====================================
# COMPUTE SHAP VALUES
# =====================================

shap_values = explainer(X_processed)

print("SHAP Values Computed")

# =====================================
# SUMMARY PLOT
# =====================================

print("Generating Summary Plot...")

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=all_features,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/shap_summary.png",
    bbox_inches='tight'
)

plt.close()

print("Summary Plot Saved")

# =====================================
# BAR PLOT
# =====================================

print("Generating Feature Importance Plot...")

shap.plots.bar(
    shap_values,
    max_display=10,
    show=False
)

plt.savefig(
    "reports/shap_bar.png",
    bbox_inches='tight'
)

plt.close()

print("Feature Importance Plot Saved")

# =====================================
# WATERFALL PLOTS
# =====================================

defaulted_indices = np.where(y == 1)[0][:2]

non_defaulted_indices = np.where(y == 0)[0][:2]

# DEFAULTED CASES

for i, idx in enumerate(defaulted_indices):

    shap.plots.waterfall(
        shap_values[idx],
        max_display=10,
        show=False
    )

    plt.savefig(
        f"reports/default_case_{i+1}.png",
        bbox_inches='tight'
    )

    plt.close()

# NON DEFAULTED CASES

for i, idx in enumerate(non_defaulted_indices):

    shap.plots.waterfall(
        shap_values[idx],
        max_display=10,
        show=False
    )

    plt.savefig(
        f"reports/non_default_case_{i+1}.png",
        bbox_inches='tight'
    )

    plt.close()

print("Waterfall Plots Saved")

print("SHAP Analysis Complete")