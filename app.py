import streamlit as st
import pandas as pd
import joblib
import os

model = joblib.load("models/fraud_model.pkl")
df = pd.read_csv("data/transactions.csv")


st.set_page_config(
    page_title="Mobile Money Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Mobile Money Fraud Detector")
st.write(
    "An AI-powered machine learning system for detecting "
    "suspicious mobile money transactions."
)

st.divider()
# Model performance
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "94.90%")

with col2:
    st.metric("Precision", "40.91%")

with col3:
    st.metric("Recall", "85.14%")

with col4:
    st.metric("F1 Score", "55.26%")

st.caption(
    "The model prioritizes fraud detection, with an emphasis on identifying "
    "as many fraudulent transactions as possible."
)

st.divider()
# Sidebar
st.sidebar.header("Transaction Information")

amount = st.sidebar.number_input(
    "Transaction Amount (₦)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN"]
)

old_balance = st.sidebar.number_input(
    "Account Balance Before Transaction (₦)",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

frequency = st.sidebar.number_input(
    "Transactions in Recent Period",
    min_value=0,
    value=3,
    step=1
)

account_age = st.sidebar.number_input(
    "Account Age (Days)",
    min_value=1,
    value=365,
    step=1
)

hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=14
)

device_change = st.sidebar.selectbox(
    "New/Changed Device?",
    ["No", "Yes"]
)

location_change = st.sidebar.selectbox(
    "Unusual Location?",
    ["No", "Yes"]
)

# Convert Yes/No to 1/0
device_change_value = 1 if device_change == "Yes" else 0
location_change_value = 1 if location_change == "Yes" else 0

# Prediction button
if st.sidebar.button("🔍 Analyze Transaction"):

    transaction = pd.DataFrame({
        "transaction_amount": [amount],
        "transaction_type": [transaction_type],
        "old_balance": [old_balance],
        "transaction_frequency": [frequency],
        "account_age_days": [account_age],
        "hour": [hour],
        "device_change": [device_change_value],
        "location_change": [location_change_value]
    })

    # Prediction
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    risk_percentage = probability * 100

    st.subheader("Fraud Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Transaction Amount",
            f"₦{amount:,.2f}"
        )

    with col2:
        st.metric(
            "Fraud Risk",
            f"{risk_percentage:.1f}%"
        )

    with col3:
        if prediction == 1:
            st.metric("Status", "🚨 FRAUD")
        else:
            st.metric("Status", "✅ LEGITIMATE")

    st.divider()

    if prediction == 1:

        st.error(
            "🚨 HIGH RISK: This transaction has been classified "
            "as potentially fraudulent."
        )

        st.warning(
            "Recommended Action: Flag this transaction for "
            "additional verification."
        )

    else:

        st.success(
            "✅ LOW RISK: This transaction appears legitimate."
        )

    # Risk gauge
    st.subheader("Fraud Risk Assessment")

    st.progress(min(int(risk_percentage), 100))

    if risk_percentage >= 70:
        risk_level = "HIGH"
    elif risk_percentage >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    st.write(f"**Risk Level:** {risk_level}")

    # Transaction details
    st.subheader("Transaction Details")

    display_data = {
        "Transaction Amount": f"₦{amount:,.2f}",
        "Transaction Type": transaction_type,
        "Previous Balance": f"₦{old_balance:,.2f}",
        "Recent Transaction Frequency": frequency,
        "Account Age": f"{account_age} days",
        "Transaction Hour": f"{hour}:00",
        "Device Changed": device_change,
        "Location Changed": location_change
    }

    details_df = pd.DataFrame(
        display_data.items(),
        columns=["Feature", "Value"]
    )

    st.table(details_df)

else:

    st.info(
        "👈 Enter the transaction information in the sidebar "
        "and click **Analyze Transaction**."
    )

    st.subheader("How the System Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Input")
        st.write(
            "Transaction and account information is provided "
            "to the system."
        )

    with col2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write(
            "The trained Random Forest machine-learning model "
            "analyzes the transaction."
        )

    with col3:
        st.markdown("### 3️⃣ Detection")
        st.write(
            "The system produces a fraud probability and "
            "classifies the transaction."
        )