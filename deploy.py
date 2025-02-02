import streamlit as st
import pandas as pd
import joblib

# Load the trained models
linear_model = joblib.load('linear_regression_model.joblib')
random_forest_model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')  # Load the scaler used during training

# Title and description
st.title("Sleep and Academic Performance")
st.write("""
This application predicts cumulative GPA based on sleep patterns and related features.
Choose the model you want to use for predictions.
""")

# Sidebar for selecting the model
model_choice = st.sidebar.selectbox(
    "Select a Model",
    ("Linear Regression", "Random Forest")
)

# Input fields for user data
st.header("Enter Input Data")
TotalSleepTime = st.number_input("Total Sleep Time (hours)", min_value=0.0, max_value=24.0, step=0.1)
midpoint_sleep = st.number_input("Midpoint Sleep Time (hours)", min_value=0.0, max_value=24.0, step=0.1)
daytime_sleep = st.number_input("Daytime Sleep (hours)", min_value=0.0, max_value=24.0, step=0.1)
term_units = st.number_input("Term Units", min_value=0.0, max_value=10.0, step=0.1)
frac_nights_with_data = st.number_input("Fraction of Nights with Data", min_value=0.0, max_value=1.0, step=0.01)

# Prediction
if st.button("Predict"):
    # Organize input data into a dataframe
    input_data = pd.DataFrame({
        'TotalSleepTime': [TotalSleepTime],
        'midpoint_sleep': [midpoint_sleep],
        'daytime_sleep': [daytime_sleep],
        'term_units': [term_units],
        'frac_nights_with_data': [frac_nights_with_data]
    })

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Choose model for prediction
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(scaled_data)[0]
    elif model_choice == "Random Forest":
        prediction = random_forest_model.predict(scaled_data)[0]

    st.success(f"The predicted cumulative GPA is: {prediction:.2f}")

