"""
Utilities for saving and loading trained model artifacts.
"""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_churn_pipeline.joblib"


def save_model(model) -> None:
    """
    Save a trained model pipeline.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


def load_model():
    """
    Load the saved model pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)