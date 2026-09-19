import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_data
from src.feature_engineering import create_features
from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
)
from src.split_data import split_data
from src.model_pipeline import build_xgboost_pipeline
from src.tuning import tune_xgboost
from src.evaluation import evaluate_model


# --------------------------------
# 1. Load data
# --------------------------------

df = load_data()


# --------------------------------
# 2. Feature engineering
# --------------------------------

df = create_features(df)


# --------------------------------
# 3. Separate features and target
# --------------------------------

X, y = split_features_target(df)

y = encode_target(y)


# --------------------------------
# 4. Identify feature types
# --------------------------------

numerical_features, categorical_features = get_feature_types(X)


# --------------------------------
# 5. Split data
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
# 6. Build XGBoost pipeline
# --------------------------------

xgb_pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)


# --------------------------------
# 7. Tune XGBoost
# --------------------------------

print("\nStarting XGBoost hyperparameter tuning...")
print("=" * 50)

search = tune_xgboost(
    xgb_pipeline,
    X_train,
    y_train,
)


# --------------------------------
# 8. Best parameters
# --------------------------------

print("\nBest Parameters")
print("=" * 50)

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# --------------------------------
# 9. Best CV score
# --------------------------------

print("\nBest Cross-Validation PR-AUC")
print("=" * 50)

print(f"{search.best_score_:.4f}")


# --------------------------------
# 10. Validation evaluation
# --------------------------------

print("\nValidation Metrics")
print("=" * 50)

validation_metrics = evaluate_model(
    search.best_estimator_,
    X_val,
    y_val,
)

for metric, value in validation_metrics.items():
    print(f"{metric.upper():10s}: {value:.4f}")