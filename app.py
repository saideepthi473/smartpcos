import streamlit as st
import numpy as np
import joblib
import json

st.title("PCOS Prediction App")

model = joblib.load("pcos_model.h5")
scaler = joblib.load("scaler.save")

with open("columns.json") as f:
    columns = json.load(f)

def predict(data):
    arr = np.array(data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]
    return prediction

inputs = []
for col in columns:
    value = st.number_input(f"{col}", step=1.0 if "yrs" in col or "days" in col else 0.1)
    inputs.append(value)

if st.button("Predict PCOS"):
    result = predict(inputs)
    st.success("PCOS Positive" if result == 1 else "PCOS Negative")

    # Show result
    st.subheader("Prediction Result:")
    st.success(result)
    st.write(f"**Confidence Score:** {prediction:.2f}")
