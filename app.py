
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table
)
from reportlab.lib.styles import getSampleStyleSheet


# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# Load trained model
model = joblib.load("/content/loan_approval_model.pkl")


# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []


# Premium dashboard design
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f4f5ff, #eef5ff);
}

h1 {
    color: #25245c;
    font-weight: 800;
}

h2, h3 {
    color: #393477;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #f0efff);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #d9d5ff;
    box-shadow: 0 5px 15px rgba(45,40,110,0.08);
}

div[data-testid="stMetricValue"] {
    color: #5145a8;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(90deg, #5145a8, #7655d9);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 12px 22px;
}

.stButton > button:hover {
    background: #392f86;
    color: white;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #25245c, #393477);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: white;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🏦 Loan Predictor")
    st.write("Smart Banking Dashboard")
    st.divider()

    st.subheader("Project Details")
    st.write("Project: Loan Approval Prediction")
    st.write("Algorithm: Random Forest")
    st.write("Technology: Python & Streamlit")
    st.write("Platform: Google Colab")

    st.divider()
    st.info("Enter applicant details to predict loan approval.")
    st.caption("MCA Mini Project | 2026")


# Main dashboard
st.title("🏦 Loan Approval Prediction System")

st.write(
    "An intelligent machine learning system "
    "to predict loan approval status."
)

st.divider()


# Applicant information
st.header("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)

    income = st.number_input(
        "Person Income", 0, value=100000
    )

    experience = st.number_input(
        "Employee Experience", 0, value=8
    )

    gender = st.selectbox(
        "Gender", ["female", "male"]
    )

    education = st.selectbox(
        "Education",
        ["Bachelor", "Doctorate", "High School", "Master", "Other"]
    )

    home = st.selectbox(
        "Home Ownership",
        ["OTHER", "OWN", "RENT", "MORTGAGE"]
    )


with col2:
    loan_amount = st.number_input(
        "Loan Amount", 0, value=5000
    )

    intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )

    interest = st.number_input(
        "Loan Interest Rate", 0.0, value=8.0
    )

    percentage = st.number_input(
        "Loan Percentage", 0.0, value=0.05
    )

    credit_history = st.number_input(
        "Credit History", 0, value=10
    )

    credit_score = st.number_input(
        "Credit Score", 0, value=800
    )

    previous_loan = st.selectbox(
        "Previous Loan", ["No", "Yes"]
    )


st.caption(
    "Educational prediction only. "
    "This is not an actual bank decision."
)

st.divider()


# Prediction
if st.button("🔍 Predict Loan Status", type="primary"):

    input_data = pd.DataFrame([{

        "Age": age,
        "Person Income": income,
        "Employee Experience": experience,
        "Loan Amount": loan_amount,
        "Loan interest Rate": interest,
        "Loan percentage": percentage,
        "Credit History": credit_history,
        "Credit Score": credit_score,

        "Gender_male": int(gender == "male"),

        "Education_Bachelor": int(education == "Bachelor"),
        "Education_Doctorate": int(education == "Doctorate"),
        "Education_High School": int(education == "High School"),
        "Education_Master": int(education == "Master"),

        "Home Onwership_OTHER": int(home == "OTHER"),
        "Home Onwership_OWN": int(home == "OWN"),
        "Home Onwership_RENT": int(home == "RENT"),

        "Loan Intent_EDUCATION": int(intent == "EDUCATION"),
        "Loan Intent_HOMEIMPROVEMENT": int(intent == "HOMEIMPROVEMENT"),
        "Loan Intent_MEDICAL": int(intent == "MEDICAL"),
        "Loan Intent_PERSONAL": int(intent == "PERSONAL"),
        "Loan Intent_VENTURE": int(intent == "VENTURE"),

        "Previous Loan_Yes": int(previous_loan == "Yes")

    }])

    # Match training feature order
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Model prediction
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = probabilities[
        list(model.classes_).index(prediction)
    ] * 100

    # Display result
    if prediction == 1:
        st.success("🎉 Loan is Likely Approved!")
    else:
        st.error("❌ Loan is Likely Rejected.")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # Save history
    st.session_state.history.append({

        "Age": age,
        "Person Income": income,
        "Loan Amount": loan_amount,
        "Credit Score": credit_score,
        "Loan Intent": intent,
        "Prediction": "Approved" if prediction == 1 else "Rejected",
        "Confidence": round(confidence, 2)

    })

    # Create PDF
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Loan Approval Prediction Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    report_data = [
        ["Applicant Information", "Details"],
        ["Age", str(age)],
        ["Person Income", str(income)],
        ["Employee Experience", str(experience)],
        ["Loan Amount", str(loan_amount)],
        ["Credit Score", str(credit_score)],
        ["Loan Intent", intent],
        ["Prediction", "Approved" if prediction == 1 else "Rejected"],
        ["Confidence", f"{confidence:.2f}%"]
    ]

    elements.append(Table(report_data))

    doc.build(elements)

    pdf_buffer.seek(0)

    st.download_button(
        "📄 Download Loan Report (PDF)",
        data=pdf_buffer.getvalue(),
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )


# Prediction history
st.divider()

st.header("📋 Loan Application History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # CSV download
    st.download_button(
        "⬇️ Download Prediction History (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="loan_prediction_history.csv",
        mime="text/csv"
    )

    # Statistics
    st.subheader("📊 Approval vs Rejection Statistics")

    approved = sum(
        item["Prediction"] == "Approved"
        for item in st.session_state.history
    )

    rejected = sum(
        item["Prediction"] == "Rejected"
        for item in st.session_state.history
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Applications", len(history_df))
    c2.metric("Approved", approved)
    c3.metric("Rejected", rejected)

    # Graph
    chart_df = pd.DataFrame({
        "Status": ["Approved", "Rejected"],
        "Applications": [approved, rejected]
    }).set_index("Status")

    st.bar_chart(chart_df)

else:
    st.info("Make a prediction to see history and statistics.")


st.divider()

st.caption(
    "Loan Approval Prediction System | MCA Mini Project | 2026"
)
