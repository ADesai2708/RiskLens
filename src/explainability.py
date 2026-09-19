"""
SHAP-based explainability utilities.
"""

import pandas as pd
import shap


def create_shap_explainer(model):
    """
    Create a SHAP TreeExplainer for a tree-based model.
    """

    return shap.TreeExplainer(model)


def calculate_shap_values(
    explainer,
    X,
):
    """
    Calculate SHAP values.
    """

    return explainer(X)


def get_feature_importance(
    shap_values,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP feature importance.
    """

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": abs(
                shap_values.values
            ).mean(axis=0),
        }
    )

    return importance.sort_values(
        by="importance",
        ascending=False,
    )


def explain_single_prediction(
    explainer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explain a single prediction using SHAP.

    Returns a dataframe containing:
        feature
        shap_value
        direction
    """

    shap_values = explainer(X)

    values = shap_values.values[0]

    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": values,
        }
    )

    explanation["direction"] = explanation[
        "shap_value"
    ].apply(
        lambda value: (
            "toward churn"
            if value > 0
            else "away from churn"
        )
    )

    explanation["impact"] = explanation[
        "shap_value"
    ].abs()

    explanation = explanation.sort_values(
        by="impact",
        ascending=False,
    )

    return explanation