import streamlit as st
import pandas as pd
from utils import load_models, explain_prediction_with_groq
from dotenv import load_dotenv
import os

# Page configuration
st.set_page_config(page_title="Churn Predictor", page_icon="🎯")
st.title("🎯 Customer Churn Predictor")
st.write("Enter customer details to predict churn risk")

# Load models
model, le_subscription, le_contract = load_models()
st.success("✅ Model loaded successfully!")
st.markdown("---")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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

# Prediction Button
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
    
    # Calculate churn probability as decimal for categorization
    churn_prob_decimal = probabilities[1]
    
    # Determine risk category
    if churn_prob_decimal < 0.3:
        risk_category = "Low Churn Risk"
        risk_color = "🟢"
        risk_style = "success"
    elif churn_prob_decimal <= 0.7:
        risk_category = "Medium Churn Risk"
        risk_color = "🟡"
        risk_style = "warning"
    else:
        risk_category = "High Churn Risk"
        risk_color = "🔴"
        risk_style = "error"

    # Display risk category prominently
    if risk_style == "success":
        st.success(f"{risk_color} **{risk_category.upper()}**")
    elif risk_style == "warning":
        st.warning(f"{risk_color} **{risk_category.upper()}**")
    else:
        st.error(f"{risk_color} **{risk_category.upper()}**")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        st.metric("Stay Probability", f"{stay_prob:.1f}%")
        
        # Risk breakdown
        st.markdown("**Risk Categories:**")
        st.markdown("🟢 Low: < 30%")
        st.markdown("🟡 Medium: 30-70%")
        st.markdown("🔴 High: > 70%")

    with col2:
        # Probability bar chart
        chart_data = pd.DataFrame({'Probability': [churn_prob, stay_prob]},
                                  index=['Churn', 'Stay'])
        st.bar_chart(chart_data)
        
        # Risk gauge visualization - Single progress bar
        st.markdown("**Churn Risk Meter**")
        st.progress(churn_prob_decimal)
        
        # Visual risk zones indicator
        st.markdown("**Current Risk Level**")
        
        # Create colored boxes to show risk zones
        risk_zones = f"""
        <div style='display: flex; gap: 10px; margin-top: 10px;'>
            <div style='flex: 3; background-color: {"#90EE90" if churn_prob_decimal < 0.3 else "#E8E8E8"}; 
                        padding: 15px; border-radius: 5px; text-align: center; 
                        border: {"3px solid #228B22" if churn_prob_decimal < 0.3 else "1px solid #CCC"};'>
                <strong>🟢 LOW</strong><br/>
                <small>0-30%</small>
            </div>
            <div style='flex: 4; background-color: {"#FFD700" if 0.3 <= churn_prob_decimal <= 0.7 else "#E8E8E8"}; 
                        padding: 15px; border-radius: 5px; text-align: center;
                        border: {"3px solid #FFA500" if 0.3 <= churn_prob_decimal <= 0.7 else "1px solid #CCC"};'>
                <strong>🟡 MEDIUM</strong><br/>
                <small>30-70%</small>
            </div>
            <div style='flex: 3; background-color: {"#FF6B6B" if churn_prob_decimal > 0.7 else "#E8E8E8"}; 
                        padding: 15px; border-radius: 5px; text-align: center;
                        border: {"3px solid #DC143C" if churn_prob_decimal > 0.7 else "1px solid #CCC"};'>
                <strong>🔴 HIGH</strong><br/>
                <small>70-100%</small>
            </div>
        </div>
        """
        st.markdown(risk_zones, unsafe_allow_html=True)

    # Step 4: Explanation using Groq
    st.markdown("---")
    st.subheader("🧠 Step 4: AI Explanation (Groq LLM)")

    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API Key to generate explanation.")
    else:
        explanation = explain_prediction_with_groq(
            groq_api_key, 
            customer_data, 
            prediction, 
            churn_prob, 
            stay_prob
        )
        st.write(explanation)

    # Optional Recommendation
    st.markdown("---")
    if risk_category == "High Churn Risk":
        st.error("💡 **Urgent Action Required**: Immediate customer retention intervention needed")
    elif risk_category == "Medium Churn Risk":
        st.warning("💡 **Recommendation**: Monitor customer and consider proactive engagement")
    else:
        st.success("💡 **Recommendation**: Customer is stable, maintain regular service quality")

# Footer
st.markdown("---")
st.caption("🤖 Model Accuracy: 99.42%")