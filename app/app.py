# app/app.py
import streamlit as st
import pickle
import numpy as np

st.title("Credit Scoring App with Model")

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Input form
income = st.number_input("Monthly income (Toman):", min_value=0)
age = st.slider("Age:", 18, 100, 30)

# Prediction
if st.button("Check creditworthiness"):
    input_data = np.array([[income, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Credit score: {probability:.2f}")
    if prediction == 1:
        st.success("You are likely creditworthy.")
    else:
        st.error("You may not be creditworthy.")
