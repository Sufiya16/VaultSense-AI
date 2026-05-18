import shap
import pandas as pd

def get_shap_values(model, X_sample):

    # Extract trained model inside pipeline
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]

    # Transform data
    X_transformed = preprocessor.transform(X_sample)

    # SHAP explainer (tree/linear safe fallback)
    explainer = shap.Explainer(clf, X_transformed)

    shap_values = explainer(X_transformed)

    return shap_values, X_transformed