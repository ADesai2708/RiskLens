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
from src.model_pipeline import (
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
)
from src.cross_validation import run_stratified_cv


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
# 5. Create train/validation/test
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
# 6. Build models
# --------------------------------

logistic_pipeline = build_logistic_regression_pipeline(
    numerical_features,
    categorical_features,
)

xgb_pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)


# --------------------------------
# 7. Logistic Regression CV
# --------------------------------

print("\nLogistic Regression")
print("==============================")

logistic_results = run_stratified_cv(
    logistic_pipeline,
    X_train,
    y_train,
)

for metric, value in logistic_results.items():
    print(f"{metric:20s}: {value:.4f}")


# --------------------------------
# 8. XGBoost CV
# --------------------------------

print("\nXGBoost")
print("==============================")

xgb_results = run_stratified_cv(
    xgb_pipeline,
    X_train,
    y_train,
)

for metric, value in xgb_results.items():
    print(f"{metric:20s}: {value:.4f}")