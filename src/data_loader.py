"""
Dataset loading utilities for the churn prediction project.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_data() -> pd.DataFrame:
    """
    Load and perform basic type correction on the raw
    customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Loaded customer churn data.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # TotalCharges contains numeric values but may be
    # represented as strings in the raw dataset.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df