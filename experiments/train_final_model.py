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
from src.model_artifact import save_model


# --------------------------------
# 1. Load data
# --------------------------------

df = load_data()


# --------------------------------
# 2. Feature engineering
# --------------------------------

df = create_features(df)


# --------------------------------
# 3. Separate X and y
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

pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)


# --------------------------------
# 7. Tune XGBoost
# --------------------------------

print("Tuning XGBoost...")
print("=" * 50)

search = tune_xgboost(
    pipeline,
    X_train,
    y_train,
)


# --------------------------------
# 8. Get best pipeline
# --------------------------------

best_pipeline = search.best_estimator_


print("\nBest parameters")
print("=" * 50)

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


print("\nBest CV PR-AUC:")
print(f"{search.best_score_:.4f}")


# --------------------------------
# 9. Save model
# --------------------------------

print("\nSaving model...")
save_model(best_pipeline)