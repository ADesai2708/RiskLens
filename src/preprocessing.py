"""
Preprocessing pipeline for the customer churn prediction project.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the input features from the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer churn dataset.

    Returns
    -------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    """

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y
def encode_target(y: pd.Series) -> pd.Series:
    """
    Convert the churn target from Yes/No to 1/0.

    Parameters
    ----------
    y : pd.Series
        Raw target values.

    Returns
    -------
    pd.Series
        Binary target values.
    """

    return y.map({"No": 0, "Yes": 1})
def get_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.

    Returns
    -------
    numerical_features : list[str]
        Numerical feature names.
    categorical_features : list[str]
        Categorical feature names.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return numerical_features, categorical_features
def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor
