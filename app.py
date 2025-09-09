import streamlit as st
import pandas as pd
from joblib import load
import project_config as config

# Page Configuration
st.set_page_config(
    page_title="Project Nova Credit Scoring",
    page_icon="✨",
    layout="centered"
)

# Caching the model loading for performance
@st.cache_resource
def load_model():
    """Loads the saved model and scaler from the /models directory."""
    try:
        model = load(config.MODEL_DIR + config.FINAL_MODEL_NAME)
        scaler = load(config.MODEL_DIR + 'scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

# UI Layout
st.title("✨ Project Nova: Fair Credit Scoring Engine")
st.write(
    "This interactive app demonstrates the Project Nova model. "
    "Adjust the sliders in the sidebar to describe a Grab partner, then click 'Get Nova Score' "
    "to see the model's prediction."
)

st.sidebar.header("Partner Input Features")

def user_input_features():
    """Creates sliders in the sidebar for user input and returns a DataFrame."""
    weekly_earnings = st.sidebar.slider('Weekly Earnings ($)', 100, 1000, 600)
    trip_frequency = st.sidebar.slider('Weekly Trip Frequency', 10, 100, 70)
    customer_ratings = st.sidebar.slider('Average Customer Rating', 3.5, 5.0, 4.9, 0.01)
    driving_score = st.sidebar.slider('Driving Score (0-100, lower is better)', 0, 100, 8)
    earnings_consistency_std = st.sidebar.slider('Earnings Consistency (Std. Dev.)', 10, 200, 30)
    rating_trend = st.sidebar.slider('Customer Rating Trend', -0.2, 0.2, 0.08, 0.01)
    high_demand_acceptance_rate = st.sidebar.slider('High Demand Acceptance Rate', 0.0, 1.0, 0.95, 0.01)

    # Derived features are calculated automatically
    data = {
        'weekly_earnings': weekly_earnings, 'trip_frequency': trip_frequency,
        'customer_ratings': customer_ratings, 'driving_score': driving_score,
        'earnings_per_trip': weekly_earnings / (trip_frequency + 1),
        'rating_to_driving_ratio': customer_ratings / (driving_score + 1e-6),
        'earnings_x_rating': weekly_earnings * customer_ratings,
        'earnings_consistency_std': earnings_consistency_std,
        'rating_trend': rating_trend,
        'high_demand_acceptance_rate': high_demand_acceptance_rate
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('Partner Profile Summary')
st.write(input_df)

# Prediction Logic
if st.sidebar.button('Get Nova Score'):
    if model is not None and scaler is not None:
        is_override_deny = False
        if (input_df['customer_ratings'].iloc[0] < 4.0 and 
            input_df['trip_frequency'].iloc[0] < 20):
            is_override_deny = True
            reason = "Very low customer ratings and trip frequency."
        elif input_df['weekly_earnings'].iloc[0] < 150 and input_df['high_demand_acceptance_rate'].iloc[0] < 0.2:
             is_override_deny = True
             reason = "Very low earnings and poor reliability on high-demand trips."

        st.subheader('Prediction Result')

        if is_override_deny:
            st.error(f'**Decision: Denied (Business Rule Override)** - {reason}')
            st.metric(label="Nova Score (Creditworthiness Probability)", value="< 1%")
            st.info("This profile was denied based on a pre-defined business rule for extreme cases, ensuring system reliability.")
        
        else:
            # If the guardrail is not triggered, proceed with the ML model prediction.
            input_df = input_df[config.FEATURES]
            input_scaled = scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            nova_score = prediction_proba[0][1]
            st.metric(label="Nova Score (Creditworthiness Probability)", value=f"{nova_score:.2%}")

            if prediction[0] == 1:
                st.success('**Decision: Approved** - This partner is deemed creditworthy based on the ML model.')
            else:
                st.error('**Decision: Denied** - This partner is not deemed creditworthy based on the ML model.')
    else:
        st.error(
            "**Model not found!** Please run the training pipeline first. "
            "In your terminal, run: `python -m src.train`"
        )