"""
Model training utilities.
"""

from sklearn.pipeline import Pipeline


def train_model(
    pipeline: Pipeline,
    X_train,
    y_train,
) -> Pipeline:
    """
    Train a model pipeline using the training data.
    """

    pipeline.fit(X_train, y_train)

    return pipeline