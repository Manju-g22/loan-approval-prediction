
import streamlit as st
import pandas as pd
import joblib
from urllib.request import urlretrieve
import os

# Load trained model
MODEL_URL = "https://github.com/Manju-g22/loan-approval-prediction/releases/download/v1.0/loan_approval_model.1.pkl"

MODEL_PATH = "loan_approval_model.pkl"

if not os.path.exists(MODEL_PATH):
    urlretrieve(MODEL_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval.")

# Applicant details
age = st.number_input("Age", 18, 100, 25)
gender = st.selectbox("Gender", ["female", "male"])
education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "Associate", "Doctorate"]
)

income = st.number_input("Person Income", min_value=0, value=30000)
experience = st.number_input("Employee Experience", min_value=0, value=2)

home = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)

intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
     "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

interest = st.number_input("Loan Interest Rate", min_value=0.0, value=10.0)
percentage = st.number_input("Loan Percentage", min_value=0.0, value=0.2)
history = st.number_input("Credit History", min_value=0, value=5)
credit_score = st.number_input("Credit Score", min_value=0, value=650)

previous_loan = st.selectbox(
    "Previous Loan",
    ["Yes", "No"]
)

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education": education,
        "Person Income": income,
        "Employee Experience": experience,
        "Home Ownership": home,
        "Loan Amount": loan_amount,
        "Loan Intent": intent,
        "Loan interest Rate": interest,
        "Loan percentage": percentage,
        "Credit History": history,
        "Credit Score": credit_score,
        "Previous Loan": previous_loan
    }])

    # Match training-time encoding
    input_data = pd.get_dummies(input_data, drop_first=True, dtype=int)
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan is likely to be APPROVED! ✅")
    else:
        st.error("Loan is likely to be REJECTED! ❌")

    st.caption("This is an educational prediction, not an actual bank decision.")
