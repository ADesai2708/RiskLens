"""
Feature engineering utilities for the customer churn project.
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from the raw Telco churn dataset.
    """

    df = df.copy()

    # 1. Average monthly spending over the customer's tenure
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )

    # 2. Number of additional services used by the customer
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["ServiceCount"] = (
        df[service_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    # 3. Whether the customer is on a month-to-month contract
    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    # 4. Whether the customer uses electronic check
    df["UsesElectronicCheck"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    return df