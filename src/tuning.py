"""
Hyperparameter tuning utilities for XGBoost.
"""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def tune_xgboost(
    pipeline: Pipeline,
    X_train,
    y_train,
) -> RandomizedSearchCV:
    """
    Tune an XGBoost pipeline using stratified cross-validation.
    """

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return search