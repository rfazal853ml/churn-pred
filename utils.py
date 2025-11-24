import pickle
import streamlit as st
from groq import Groq
from prompts import get_explanation_prompt


@st.cache_resource
def load_models():
    """
    Load the trained model and encoders from disk.
    
    Returns:
        tuple: (model, le_subscription, le_contract)
    """
    try:
        with open("models/rf_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("encoders/subscription_encoder.pkl", "rb") as f:
            le_subscription = pickle.load(f)
        with open("encoders/contract_length_encoder.pkl", "rb") as f:
            le_contract = pickle.load(f)
        return model, le_subscription, le_contract
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


def explain_prediction_with_groq(groq_api_key, customer_data, prediction, churn_prob, stay_prob):
    """
    Generate an explanation for the churn prediction using Groq LLM.
    
    Args:
        groq_api_key (str): Groq API key
        customer_data (dict): Customer information dictionary
        prediction (int): Model prediction (0 or 1)
        churn_prob (float): Probability of churn (percentage)
        stay_prob (float): Probability of staying (percentage)
    
    Returns:
        str: Explanation text from the LLM
    """
    try:
        client = Groq(api_key=groq_api_key)
        
        prompt = get_explanation_prompt(customer_data, prediction, churn_prob, stay_prob)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error from Groq API: {e}"