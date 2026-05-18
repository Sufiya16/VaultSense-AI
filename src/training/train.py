import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    classification_report
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from src.training.feature_engineering import create_features


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/raw/loan_train.csv")

print("Dataset Loaded")
print(df.shape)

# =====================================
# FEATURE ENGINEERING
# =====================================

df = create_features(df)

print("Feature Engineering Complete")

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop(
    columns=[
        'loan_id',
        'application_date',
        'default_flag'
    ]
)

y = df['default_flag']

# =====================================
# COLUMN TYPES
# =====================================

categorical_cols = X.select_dtypes(
    include=['object']
).columns.tolist()

numeric_cols = X.select_dtypes(
    exclude=['object']
).columns.tolist()

# =====================================
# NUMERIC PIPELINE
# =====================================

numeric_transformer = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='median')
    ),
    (
        'scaler',
        StandardScaler()
    )
])

# =====================================
# CATEGORICAL PIPELINE
# =====================================

categorical_transformer = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='most_frequent')
    ),
    (
        'encoder',
        OneHotEncoder(handle_unknown='ignore')
    )
])

# =====================================
# PREPROCESSOR
# =====================================

preprocessor = ColumnTransformer([
    (
        'num',
        numeric_transformer,
        numeric_cols
    ),
    (
        'cat',
        categorical_transformer,
        categorical_cols
    )
])

# =====================================
# MODELS
# =====================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
}

# =====================================
# CROSS VALIDATION
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

best_auc = 0
best_model_name = None
best_pipeline = None

# =====================================
# MODEL TRAINING LOOP
# =====================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        (
            'preprocessor',
            preprocessor
        ),
        (
            'model',
            model
        )
    ])

    scores = cross_validate(

        pipeline,

        X,
        y,

        cv=cv,

        scoring={

            'roc_auc': 'roc_auc',

            'pr_auc': 'average_precision',

            'f1': 'f1'

        },

        return_train_score=False
    )

    auc_mean = np.mean(scores['test_roc_auc'])

    pr_auc_mean = np.mean(scores['test_pr_auc'])

    f1_mean = np.mean(scores['test_f1'])

    results[name] = {

        "AUC_ROC": round(auc_mean, 4),

        "PR_AUC": round(pr_auc_mean, 4),

        "F1_SCORE": round(f1_mean, 4)
    }

    print(results[name])

    # BEST MODEL SELECTION

    if auc_mean > best_auc:

        best_auc = auc_mean

        best_model_name = name

        best_pipeline = pipeline

# =====================================
# FINAL BEST MODEL TRAINING
# =====================================

print("\n==============================")
print(f"BEST MODEL: {best_model_name}")
print("==============================")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

best_pipeline.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================

pred_probs = best_pipeline.predict_proba(X_test)[:, 1]

# =====================================
# THRESHOLD OPTIMIZATION
# =====================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    pred_probs
)

f1_scores = (
    2 * precision * recall
    /
    (precision + recall + 1e-9)
)

best_threshold_index = np.argmax(f1_scores)

optimal_threshold = thresholds[best_threshold_index]

print("\nOptimal Threshold:")
print(round(optimal_threshold, 4))

# =====================================
# FINAL PREDICTIONS
# =====================================

final_preds = (
    pred_probs >= optimal_threshold
).astype(int)

# =====================================
# FINAL METRICS
# =====================================

final_auc = roc_auc_score(
    y_test,
    pred_probs
)

final_pr_auc = average_precision_score(
    y_test,
    pred_probs
)

final_f1 = f1_score(
    y_test,
    final_preds
)

print("\nFINAL TEST METRICS")

print("AUC ROC:", round(final_auc, 4))

print("PR AUC:", round(final_pr_auc, 4))

print("F1 SCORE:", round(final_f1, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    final_preds
))

# =====================================
# SAVE MODEL
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/best_model.pkl"
)

joblib.dump(
    optimal_threshold,
    "models/threshold.pkl"
)

# =====================================
# SAVE METRICS
# =====================================

os.makedirs("reports", exist_ok=True)

with open(
    "reports/model_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nModel Saved Successfully")

print("Threshold Saved Successfully")

print("Metrics Saved Successfully")