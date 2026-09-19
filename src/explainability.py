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
def format_feature_name(feature_name):
    """
    Convert transformed feature names into
    human-readable names for the dashboard.
    """

    feature_name = feature_name.replace(
        "numerical__",
        ""
    )

    feature_name = feature_name.replace(
        "categorical__",
        ""
    )

    replacements = {
        "IsMonthToMonth": "Month-to-month contract",
        "UsesElectronicCheck": "Electronic check",
        "AvgMonthlySpend": "Average monthly spend",
        "ServiceCount": "Number of services",
        "SeniorCitizen": "Senior citizen",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "tenure": "Customer tenure",
        "InternetService_Fiber optic": "Fiber optic internet",
        "InternetService_DSL": "DSL internet",
        "InternetService_No": "No internet service",
        "OnlineSecurity_No": "No online security",
        "OnlineSecurity_Yes": "Online security",
        "TechSupport_No": "No tech support",
        "TechSupport_Yes": "Tech support",
        "OnlineBackup_No": "No online backup",
        "OnlineBackup_Yes": "Online backup",
        "DeviceProtection_No": "No device protection",
        "DeviceProtection_Yes": "Device protection",
        "StreamingTV_Yes": "Streaming TV",
        "StreamingMovies_Yes": "Streaming movies",
        "Contract_Month-to-month": "Month-to-month contract",
        "Contract_One year": "One-year contract",
        "Contract_Two year": "Two-year contract",
        "PaperlessBilling_Yes": "Paperless billing",
        "PaperlessBilling_No": "No paperless billing",
        "MultipleLines_Yes": "Multiple phone lines",
        "MultipleLines_No": "No multiple phone lines",
        "gender_Male": "Male customer",
        "gender_Female": "Female customer",
        "Dependents_Yes": "Has dependents",
        "Dependents_No": "No dependents",
    }

    return replacements.get(feature_name, feature_name)