from src.data_loader import load_data
from src.feature_engineering import create_features
from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
)
from src.split_data import split_data
from src.model_pipeline import (
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
)
from src.evaluation import evaluate_model, get_confusion_matrix


# -----------------------------
# 1. Load data
# -----------------------------

df = load_data()

# Feature engineering
df = create_features(df)

# -----------------------------
# 2. Prepare X and y
# -----------------------------

X, y = split_features_target(df)

y = encode_target(y)


# -----------------------------
# 3. Identify feature types
# -----------------------------

numerical_features, categorical_features = get_feature_types(X)


# -----------------------------
# 4. Split data
# -----------------------------

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(X, y)


# -----------------------------
# 5. Build model pipeline
# -----------------------------

pipeline = build_logistic_regression_pipeline(
    numerical_features,
    categorical_features,
)
# -----------------------------
# XGBoost model
# -----------------------------

xgb_pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)

xgb_pipeline.fit(X_train, y_train)

xgb_metrics = evaluate_model(
    xgb_pipeline,
    X_val,
    y_val,
)

print("\nXGBoost Validation Metrics")
print("------------------------------")

for name, value in xgb_metrics.items():
    print(f"{name.upper():10s}: {value:.4f}")

print("\nXGBoost Confusion Matrix")
print("------------------------------")

xgb_cm = get_confusion_matrix(
    xgb_pipeline,
    X_val,
    y_val,
)

print(xgb_cm)

# -----------------------------
# 6. Train model
# -----------------------------

pipeline.fit(X_train, y_train)


# -----------------------------
# 7. Evaluate on validation
# -----------------------------

metrics = evaluate_model(
    pipeline,
    X_val,
    y_val,
)


# -----------------------------
# 8. Print metrics
# -----------------------------

print("\nValidation Metrics")
print("------------------------------")

for name, value in metrics.items():
    print(f"{name.upper():10s}: {value:.4f}")


# -----------------------------
# 9. Confusion matrix
# -----------------------------

cm = get_confusion_matrix(
    pipeline,
    X_val,
    y_val,
)

print("\nConfusion Matrix")
print("------------------------------")
print(cm)