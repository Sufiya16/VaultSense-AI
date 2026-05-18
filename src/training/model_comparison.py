import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.training.feature_engineering import create_features
from src.training.preprocessor import build_preprocessor
from src.training.metrics import ks_statistic
from src.training.threshold_optimization import find_best_threshold


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/loan_train.csv")
df = create_features(df)

X = df.drop(columns=["loan_id", "application_date", "default_flag"])
y = df["default_flag"]

# =========================
# PREPROCESSOR
# =========================
preprocessor = build_preprocessor(X)

# =========================
# MODELS (PIPELINE SAFE)
# =========================
models = {
    "LogisticRegression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        ))
    ]),

    "LightGBM": Pipeline([
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31
        ))
    ])
}

# =========================
# CROSS VALIDATION SETUP
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# =========================
# TRAIN + EVALUATE MODELS
# =========================
for name, model in models.items():

    auc_scores = []
    pr_auc_scores = []
    ks_scores = []

    all_probs = []
    all_labels = []

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]

        # ROC-AUC
        auc_scores.append(roc_auc_score(y_val, preds))

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_val, preds)
        pr_auc_scores.append(auc(recall, precision))

        # KS Statistic
        ks_scores.append(ks_statistic(y_val, preds))

        # Collect for threshold tuning
        all_probs.extend(preds)
        all_labels.extend(y_val)

    # =========================
    # THRESHOLD OPTIMIZATION
    # =========================
    best_threshold = find_best_threshold(
        np.array(all_labels),
        np.array(all_probs)
    )

    print(f"\nMODEL: {name}")
    print("Best Threshold:", round(best_threshold, 4))

    # =========================
    # STORE RESULTS
    # =========================
    results.append({
        "model": name,
        "roc_auc_mean": np.mean(auc_scores),
        "pr_auc_mean": np.mean(pr_auc_scores),
        "ks_mean": np.mean(ks_scores),
        "best_threshold": best_threshold
    })

# =========================
# FINAL RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["ks_mean", "roc_auc_mean"],
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================")
print(results_df)

print("\nBEST MODEL:", results_df.iloc[0]["model"])
print("BEST THRESHOLD:", results_df.iloc[0]["best_threshold"])