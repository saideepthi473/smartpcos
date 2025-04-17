
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf

# Load saved model, scaler, and column names
model = tf.keras.models.load_model('pcos_model.h5')
scaler = joblib.load('scaler.pkl')

with open('columns.json', 'r') as f:
    feature_columns = json.load(f)

# App title and description
st.title("👩‍⚕️ PCOS Prediction App")
st.markdown("""
This app uses a machine learning model to predict whether a person may have Polycystic Ovary Syndrome (PCOS), based on medical inputs.
Please enter your values below:
""")

# Collect user input for each feature
user_input = []
for col in feature_columns:
    val = st.number_input(f"{col}", min_value=0.0, step=0.1)
    user_input.append(val)

# Prediction button
if st.button("Predict PCOS"):
    # Convert input to numpy array and reshape
    input_array = np.array([user_input])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]
    result = "⚠️ PCOS Likely" if prediction > 0.5 else "✅ No PCOS Detected"

    # Show result
    st.subheader("Prediction Result:")
    st.success(result)
    st.write(f"**Confidence Score:** {prediction:.2f}")
