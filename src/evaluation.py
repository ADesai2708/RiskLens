"""
Model evaluation utilities.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(model, X, y):
    """
    Evaluate a binary classification model.

    Returns:
        dict: Evaluation metrics.
    """

    # Class predictions
    y_pred = model.predict(X)

    # Probability of positive class
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "pr_auc": average_precision_score(y, y_proba),
    }

    return metrics


def get_confusion_matrix(model, X, y):
    """
    Return the confusion matrix for a binary classifier.
    """

    y_pred = model.predict(X)

    return confusion_matrix(y, y_pred)