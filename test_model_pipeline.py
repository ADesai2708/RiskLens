from src.data_loader import load_data
from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
)
from src.split_data import split_data
from src.model_pipeline import build_logistic_regression_pipeline


# Load dataset
df = load_data()

# Separate features and target
X, y = split_features_target(df)

# Encode target
y = encode_target(y)

# Get feature types
numerical_features, categorical_features = get_feature_types(X)

# Split dataset
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(X, y)

# Build pipeline
pipeline = build_logistic_regression_pipeline(
    numerical_features,
    categorical_features,
)

# Train only on training data
pipeline.fit(X_train, y_train)

print("Pipeline trained successfully!")

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nValidation predictions:")
print(pipeline.predict(X_val)[:20])