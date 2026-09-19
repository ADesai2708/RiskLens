import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to predict the probability of churn."
)

st.divider()


# -----------------------------
# Customer Information
# -----------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
    )


with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"],
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
    )


with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )


st.divider()


# -----------------------------
# Billing Information
# -----------------------------

st.subheader("Billing Information")

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=0.01,
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0,
        step=0.01,
    )


st.divider()


# -----------------------------
# Prediction Button
# -----------------------------

if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True,
):

    if tenure < 0:
        st.error("Tenure must not be negative.")
        st.stop()

    if monthly_charges < 0:
        st.error("Monthly charges must not be negative.")
        st.stop()

    if total_charges < 0:
        st.error("Total charges must not be negative.")
        st.stop()

    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:

        # Send customer data to FastAPI
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:

            result = response.json()

            churn_probability = result["churn_probability"]
            prediction = result["prediction"]
            explanations = result["explanations"]

            # -----------------------------
            # Prediction Result
            # -----------------------------

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}",
                )

            with col2:
                st.metric(
                    "Prediction",
                    prediction,
                )
                st.progress(
    churn_probability,
    text=f"Churn probability: {churn_probability:.2%}",
)

            if prediction == "Yes":
                st.warning(
                    "This customer has been classified as likely to churn."
                )
            else:
                st.success(
                    "This customer has been classified as unlikely to churn."
                )

            st.caption(
                "The prediction is generated by the trained XGBoost model. "
                "The factors below show which features influenced this "
                "individual prediction according to SHAP."
            )

            # -----------------------------
            # SHAP Explanation
            # -----------------------------

            st.subheader("Why did the model make this prediction?")

            st.write(
                "These are the strongest factors influencing this "
                "individual prediction."
            )

            for explanation in explanations:

                feature = explanation["feature"]
                shap_value = explanation["shap_value"]
                impact = explanation["impact"]
                direction = explanation["direction"]

                if direction == "increases":
                    indicator = "↑"
                else:
                    indicator = "↓"

                col1, col2, col3 = st.columns([5, 2, 2])

                with col1:
                    st.write(f"**{indicator} {feature}**")

                with col2:
                    st.write(f"SHAP: `{shap_value:.4f}`")

                with col3:
                    st.write(f"Impact: `{impact:.4f}`")

        else:

            try:
                error_detail = response.json().get(
                    "detail",
                    "Unknown API error",
                )
            except ValueError:
                error_detail = response.text

            st.error(f"Prediction failed: {error_detail}")

    except requests.exceptions.RequestException as error:

        st.error(
            f"Could not connect to the prediction API: {error}"
        )