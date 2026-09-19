"""
Utilities for building the machine learning pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.preprocessing import build_preprocessor


def build_logistic_regression_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build a preprocessing + Logistic Regression pipeline.
    """

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
def build_xgboost_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build a preprocessing + XGBoost pipeline.
    """

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline