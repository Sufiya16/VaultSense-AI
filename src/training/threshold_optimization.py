import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve


def find_best_threshold(y_true, y_probs):

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = []

    for p, r in zip(precision[:-1], recall[:-1]):
        if (p + r) == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * (p * r) / (p + r))

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    return best_threshold