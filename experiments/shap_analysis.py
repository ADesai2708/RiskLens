import sys
from pathlib import Path
import matplotlib.pyplot as plt
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
from src.explainability import (
    create_shap_explainer,
    calculate_shap_values,
    explain_single_prediction,
    get_feature_importance,
)
import matplotlib.pyplot as plt
import pandas as pd
import shap

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
# 4. Train / validation / test
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
# 5. Build XGBoost pipeline
# --------------------------------

pipeline = build_xgboost_pipeline(
    numerical_features,
    categorical_features,
)


# --------------------------------
# 6. Tune model
# --------------------------------

print("Tuning XGBoost...")
search = tune_xgboost(
    pipeline,
    X_train,
    y_train,
)

best_pipeline = search.best_estimator_

print("\nBest parameters:")
for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# --------------------------------
# 7. Transform validation data
# --------------------------------

preprocessor = best_pipeline.named_steps[
    "preprocessor"
]

model = best_pipeline.named_steps[
    "model"
]

X_val_transformed = preprocessor.transform(X_val)


# --------------------------------
# 8. Get transformed feature names
# --------------------------------

feature_names = preprocessor.get_feature_names_out()

X_val_transformed_df = pd.DataFrame(
    X_val_transformed.toarray()
    if hasattr(X_val_transformed, "toarray")
    else X_val_transformed,
    columns=feature_names,
    index=X_val.index,
)


# --------------------------------
# 9. Create SHAP explainer
# --------------------------------

explainer = create_shap_explainer(model)

shap_values = calculate_shap_values(
    explainer,
    X_val_transformed_df,
)


# --------------------------------
# 10. Feature importance
# --------------------------------

importance = get_feature_importance(
    shap_values,
    X_val_transformed_df,
)

print("\nTop 20 SHAP Features")
print("=" * 50)

print(
    importance.head(20).to_string(
        index=False
    )
)
# --------------------------------
# 11. SHAP summary plot
# --------------------------------

shap.plots.beeswarm(
    shap_values,
    max_display=20,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
# --------------------------------
# 12. Explain one customer
# --------------------------------

customer_index = 0

X_customer = X_val_transformed_df.iloc[
    [customer_index]
]

local_explanation = explain_single_prediction(
    explainer,
    X_customer,
)

print("\nLocal SHAP Explanation")
print("=" * 60)

print(
    local_explanation[
        [
            "feature",
            "shap_value",
            "direction",
        ]
    ]
    .head(10)
    .to_string(index=False)
)