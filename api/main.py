from fastapi import FastAPI
import pandas as pd
import shap

from api.schemas import CustomerData, PredictionResponse
from src.model_artifact import load_model
from src.feature_engineering import create_features
from src.explainability import format_feature_name

app = FastAPI(
    title="Explainable Customer Churn API",
    description="API for customer churn prediction using XGBoost.",
    version="1.0.0",
)


# Load the trained model once when the API starts
model = load_model()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    # --------------------------------
    # 1. Convert request to DataFrame
    # --------------------------------

    input_data = pd.DataFrame([customer.model_dump()])

    # --------------------------------
    # 2. Apply feature engineering
    # --------------------------------

    input_data = create_features(input_data)

    # --------------------------------
    # 3. Get prediction
    # --------------------------------

    churn_probability = model.predict_proba(input_data)[0][1]

    prediction = (
        "Yes"
        if churn_probability >= 0.5
        else "No"
    )

    # --------------------------------
    # 4. Transform input for SHAP
    # --------------------------------

    preprocessor = model.named_steps["preprocessor"]
    xgboost_model = model.named_steps["model"]

    transformed_input = preprocessor.transform(input_data)

    # --------------------------------
    # 5. Calculate SHAP values
    # --------------------------------

    explainer = shap.TreeExplainer(xgboost_model)

    shap_values = explainer(transformed_input)

    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values.values[0],
    })

    # Absolute impact determines importance
    explanation["impact"] = explanation["shap_value"].abs()

    # Sort most influential features first
    explanation = explanation.sort_values(
        by="impact",
        ascending=False,
    )

    # Keep top 5 explanations
    top_features = explanation.head(5)

    explanations = []

    for _, row in top_features.iterrows():

        direction = (
            "increases"
            if row["shap_value"] > 0
            else "decreases"
        )

        explanations.append({
    "feature": format_feature_name(row["feature"]),
    "shap_value": float(row["shap_value"]),
    "impact": float(row["impact"]),
    "direction": direction,
})

    return {
        "churn_probability": float(churn_probability),
        "prediction": prediction,
        "explanations": explanations,
    }