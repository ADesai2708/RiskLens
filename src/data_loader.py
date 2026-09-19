"""
Dataset loading utilities for the churn prediction project.
"""

from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_data() -> pd.DataFrame:
    """
    Load the raw customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Raw customer churn data.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)