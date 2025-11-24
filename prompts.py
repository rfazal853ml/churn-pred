def get_explanation_prompt(customer_data, prediction, churn_prob, stay_prob):
    """
    Generate the prompt for explaining churn predictions.
    
    Args:
        customer_data (dict): Customer information dictionary
        prediction (int): Model prediction (0 or 1)
        churn_prob (float): Probability of churn (percentage)
        stay_prob (float): Probability of staying (percentage)
    
    Returns:
        str: Formatted prompt for the LLM
    """
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
    return prompt