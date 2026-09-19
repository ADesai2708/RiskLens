"""
Utilities for building the machine learning pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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