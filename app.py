import streamlit as st
import pandas as pd
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

st.set_page_config(page_title="Churn Predictor", page_icon="🎯")
st.title("🎯 Customer Churn Predictor")
st.write("Enter customer details to predict churn risk")


@st.cache_resource
def load_models():
    try:
        with open("rf_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("subscription_encoder.pkl", "rb") as f:
            le_subscription = pickle.load(f)
        with open("contract_length_encoder.pkl", "rb") as f:
            le_contract = pickle.load(f)
        return model, le_subscription, le_contract
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, le_subscription, le_contract = load_models()
st.success("✅ Model loaded successfully!")
st.markdown("---")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def explain_prediction_with_groq(customer_data, prediction, churn_prob, stay_prob):
    try:
        client = Groq(api_key=groq_api_key)

        prompt = f"""
You are an AI explainability expert.

Explain clearly and simply why the model predicted this outcome.

Customer Data:
{customer_data}

Prediction: {prediction}   (1 = High Churn Risk, 0 = Low Churn Risk)
Churn Probability: {churn_prob:.2f}%
Stay Probability: {stay_prob:.2f}%

Explain:
- Which features contributed the most
- Why the score is high or low
- Keep it simple and non-technical
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error from Groq API: {e}"
    
# ------------------------------------------------------
# User Input Section
st.subheader("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)
    usage_frequency = st.number_input("Usage Frequency", min_value=0, value=15)
    support_calls = st.number_input("Support Calls", min_value=0, value=3)

with col2:
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=5)
    subscription_type = st.selectbox("Subscription Type", le_subscription.classes_)
    contract_length = st.selectbox("Contract Length", le_contract.classes_)
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=500.0)
    last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, value=30)


if st.button("🔮 Predict Churn", use_container_width=True):

    # Create customer data dictionary
    customer_data = {
        'Age': age,
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': subscription_type,
        'Contract Length': contract_length,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }

    df = pd.DataFrame([customer_data])

    # Step 1: Show original data
    st.markdown("---")
    st.subheader("📋 Step 1: Original Data")
    st.dataframe(df, use_container_width=True)

    # Step 2: Encoding
    st.subheader("🔄 Step 2: Encoding Data")
    df['Gender'] = 1 if gender == 'Female' else 0
    df['Subscription Type'] = le_subscription.transform([subscription_type])[0]
    df['Contract Length'] = le_contract.transform([contract_length])[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Gender", f"{gender} → {df['Gender'].values[0]}")
    col2.metric("Subscription", f"{subscription_type} → {df['Subscription Type'].values[0]}")
    col3.metric("Contract", f"{contract_length} → {df['Contract Length'].values[0]}")

    st.dataframe(df, use_container_width=True)

    # Step 3: Prediction
    st.subheader("🎯 Step 3: Prediction Result")

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    churn_prob = probabilities[1] * 100
    stay_prob = probabilities[0] * 100

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK**")
            st.metric("Churn Probability", f"{churn_prob:.1f}%")
        else:
            st.success("✅ **LOW CHURN RISK**")
            st.metric("Stay Probability", f"{stay_prob:.1f}%")

    with col2:
        chart_data = pd.DataFrame({'Probability': [churn_prob, stay_prob]},
                                  index=['Churn', 'Stay'])
        st.bar_chart(chart_data)

    # Step 4: Explanation using Groq
    st.markdown("---")
    st.subheader("🧠 Step 4: AI Explanation (Groq LLM)")

    if not groq_api_key:
        st.warning("⚠ Please enter your Groq API Key to generate explanation.")
    else:
        explanation = explain_prediction_with_groq(customer_data, prediction, churn_prob, stay_prob)
        st.write(explanation)

    # Optional Recommendation
    st.markdown("---")
    if prediction == 1:
        st.warning("💡 Recommendation: Contact customer for retention")
    else:
        st.info("💡 Recommendation: Customer is stable")

# Footer
st.markdown("---")
st.caption("🤖 Model Accuracy: 99.42%")
