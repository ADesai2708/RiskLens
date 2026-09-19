"""
Final model training and untouched test-set evaluation.
"""

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
)
from src.split_data import split_data
from src.model_pipeline import build_xgboost_pipeline
from src.evaluation import evaluate_model, get_confusion_matrix
from src.model_artifact import save_model


# --------------------------------
# 1. Load data
# --------------------------------

df = load_data()

df = create_features(df)


# --------------------------------
# 2. Prepare X and y
# --------------------------------

X, y = split_features_target(df)

y = encode_target(y)


# --------------------------------
# 3. Feature types
# --------------------------------

numerical_features, categorical_features = get_feature_types(X)


# --------------------------------
# 4. Split data
# --------------------------------

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(X, y)


# --------------------------------
# 5. Combine train + validation
# --------------------------------

X_train_final = X_train._append(X_val)

y_train_final = y_train._append(y_val)


print("Final training data:")
print(f"X shape: {X_train_final.shape}")
print(f"y shape: {y_train_final.shape}")

print("\nTest data:")
print(f"X shape: {X_test.shape}")
print(f"y shape: {y_test.shape}")


# --------------------------------
# 6. Build pipeline
# --------------------------------

final_pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)


# --------------------------------
# 7. Apply tuned parameters
# --------------------------------

final_pipeline.set_params(
    model__subsample=0.8,
    model__n_estimators=200,
    model__min_child_weight=1,
    model__max_depth=4,
    model__learning_rate=0.03,
    model__colsample_bytree=0.8,
)


# --------------------------------
# 8. Train final model
# --------------------------------

print("\nTraining final model...")
print("=" * 50)

final_pipeline.fit(
    X_train_final,
    y_train_final,
)


# --------------------------------
# 9. Evaluate on untouched test
# --------------------------------

print("\nFinal Test Metrics")
print("=" * 50)

test_metrics = evaluate_model(
    final_pipeline,
    X_test,
    y_test,
)

for metric, value in test_metrics.items():
    print(f"{metric.upper():10s}: {value:.4f}")


# --------------------------------
# 10. Confusion matrix
# --------------------------------

test_cm = get_confusion_matrix(
    final_pipeline,
    X_test,
    y_test,
)

print("\nFinal Test Confusion Matrix")
print("=" * 50)

print(test_cm)


# --------------------------------
# 11. Save final model
# --------------------------------

print("\nSaving final model...")
save_model(final_pipeline)