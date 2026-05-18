import numpy as np
from sklearn.metrics import roc_curve


# =========================
# KS STATISTIC
# =========================
def ks_statistic(y_true, y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    ks_value = max(tpr - fpr)

    return ks_value