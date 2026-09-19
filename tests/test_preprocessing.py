import pandas as pd

from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
    build_preprocessor,
)

from src.feature_engineering import create_features


def test_split_features_target():
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "tenure": [1, 12],
        "MonthlyCharges": [29.85, 70.00],
        "Churn": ["No", "Yes"],
    })

    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns

    assert len(X) == 2
    assert len(y) == 2


def test_encode_target():
    y = pd.Series(["No", "Yes", "No", "Yes"])

    encoded = encode_target(y)

    assert encoded.tolist() == [0, 1, 0, 1]


def test_feature_engineering():
    df = pd.DataFrame({
        "TotalCharges": [100.0, 200.0],
        "tenure": [10, 20],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["Yes", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
        ],
    })

    result = create_features(df)

    assert "AvgMonthlySpend" in result.columns
    assert "ServiceCount" in result.columns
    assert "IsMonthToMonth" in result.columns
    assert "UsesElectronicCheck" in result.columns

    assert result["AvgMonthlySpend"].tolist() == [10.0, 10.0]

    assert result["ServiceCount"].tolist() == [3, 3]

    assert result["IsMonthToMonth"].tolist() == [1, 0]

    assert result["UsesElectronicCheck"].tolist() == [1, 0]


def test_feature_types():
    df = pd.DataFrame({
        "tenure": [1, 12],
        "MonthlyCharges": [29.85, 70.00],
        "Contract": ["Month-to-month", "Two year"],
        "InternetService": ["DSL", "Fiber optic"],
    })

    numerical, categorical = get_feature_types(df)

    assert "tenure" in numerical
    assert "MonthlyCharges" in numerical

    assert "Contract" in categorical
    assert "InternetService" in categorical


def test_preprocessor():
    numerical_features = ["tenure", "MonthlyCharges"]
    categorical_features = ["Contract", "InternetService"]

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    transformer_names = [
        name
        for name, _, _ in preprocessor.transformers
    ]

    assert "numerical" in transformer_names
    assert "categorical" in transformer_names