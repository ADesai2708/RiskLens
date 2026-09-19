"""
Cross-validation utilities.
"""

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def run_stratified_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    Run stratified K-fold cross-validation.

    Evaluates:
        - F1
        - Average Precision (PR-AUC)
        - Precision
        - Recall
        - Accuracy
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "pr_auc": "average_precision",
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "accuracy_mean": results["test_accuracy"].mean(),
        "precision_mean": results["test_precision"].mean(),
        "recall_mean": results["test_recall"].mean(),
        "f1_mean": results["test_f1"].mean(),
        "pr_auc_mean": results["test_pr_auc"].mean(),
    }