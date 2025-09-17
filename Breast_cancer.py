import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

# Load model and scaler
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('bc_scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'logistic_regression_model.joblib' and 'bc_scaler.joblib' are in the same folder.")
    st.stop()

# Load feature names
data = load_breast_cancer()
features = data.feature_names

st.title("🔬 Breast Cancer Prediction App")
st.write("Adjust the sliders to input tumor measurements and predict malignancy.")

# Collect user input
user_input = []
for i, feature in enumerate(features):
    min_val = float(np.min(data.data[:, i]))
    max_val = float(np.max(data.data[:, i]))
    default_val = float(np.mean(data.data[:, i]))
    value = st.slider(f"{feature}", min_val, max_val, default_val)
    user_input.append(value)

# Predict
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 0:
        st.error(f"Prediction: Malignant ({confidence:.2%} confidence)")
    else:
        st.success(f"Prediction: Benign ({confidence:.2%} confidence)")