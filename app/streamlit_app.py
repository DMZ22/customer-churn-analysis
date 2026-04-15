"""Streamlit Dashboard for Customer Churn Prediction."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ─── Auto-bootstrap: run main.py if artifacts missing (for fresh cloud deploys)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")) or \
   not os.path.exists(os.path.join(DATA_DIR, "customer_churn_db.sqlite")):
    with st.spinner("🔧 First-run setup: training models and initializing database..."):
        import subprocess
        subprocess.run(["python", os.path.join(BASE_DIR, "main.py")], check=True, cwd=BASE_DIR)
        st.success("Setup complete!")
        st.rerun()

from src.model import predict_churn, load_model, load_artifacts, get_feature_importance
from src.data_loader import load_data_from_db, DB_PATH
from src.preprocessing import prepare_data

st.title("📊 Customer Churn Analysis Dashboard")
st.markdown("---")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", [
    "Overview", "Data Explorer", "Predict Churn", "Model Performance", "Upload & Predict"
])


@st.cache_data
def get_data():
    """Load data with caching."""
    if os.path.exists(DB_PATH):
        return load_data_from_db()
    return None


# ─── OVERVIEW PAGE ───────────────────────────────────────────────────────────
if page == "Overview":
    st.header("Dataset Overview")
    df = get_data()

    if df is None:
        st.error("Database not found. Run `python main.py` first to initialize the project.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{(df['churn'] == 'Yes').mean():.1%}")
    col3.metric("Avg Monthly Charges", f"${df['monthly_charges'].mean():.2f}")
    col4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_counts = df["churn"].value_counts()
        ax.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
               colors=["#2ecc71", "#e74c3c"], startangle=90)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_contract = df.groupby("contract_type")["churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        churn_by_contract.columns = ["contract_type", "churn_rate"]
        sns.barplot(data=churn_by_contract, x="contract_type", y="churn_rate",
                    palette="rocket", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close()

    st.subheader("Key Insights")
    st.markdown("""
    - **Month-to-month** contracts have the highest churn rate
    - Customers with **shorter tenure** are more likely to churn
    - **Higher monthly charges** correlate with higher churn
    - **Electronic check** payment method shows higher churn
    """)


# ─── DATA EXPLORER PAGE ─────────────────────────────────────────────────────
elif page == "Data Explorer":
    st.header("Data Explorer")
    df = get_data()

    if df is None:
        st.error("Database not found.")
        st.stop()

    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        contract_filter = st.multiselect("Contract Type", df["contract_type"].unique(), default=df["contract_type"].unique())
    with col2:
        churn_filter = st.multiselect("Churn", df["churn"].unique(), default=df["churn"].unique())
    with col3:
        service_filter = st.multiselect("Internet Service", df["internet_service"].unique(), default=df["internet_service"].unique())

    filtered = df[
        (df["contract_type"].isin(contract_filter)) &
        (df["churn"].isin(churn_filter)) &
        (df["internet_service"].isin(service_filter))
    ]

    st.write(f"Showing {len(filtered):,} of {len(df):,} customers")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=filtered, x="monthly_charges", hue="churn", kde=True,
                     palette=["#2ecc71", "#e74c3c"], ax=ax)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Tenure Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=filtered, x="tenure", hue="churn", kde=True,
                     palette=["#2ecc71", "#e74c3c"], ax=ax)
        st.pyplot(fig)
        plt.close()


# ─── PREDICT CHURN PAGE ─────────────────────────────────────────────────────
elif page == "Predict Churn":
    st.header("Predict Customer Churn")
    st.markdown("Enter customer details to predict churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 80, 35)
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col3:
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("🔮 Predict Churn", type="primary"):
        customer = {
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
        }

        try:
            result = predict_churn(customer)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result["prediction"],
                        delta="High Risk" if result["prediction"] == "Yes" else "Low Risk",
                        delta_color="inverse")
            col2.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            col3.metric("Retention Probability", f"{result['retention_probability']:.1%}")

            # Risk gauge
            prob = result["churn_probability"]
            if prob > 0.7:
                st.error(f"⚠️ HIGH RISK: This customer has a {prob:.1%} probability of churning!")
            elif prob > 0.4:
                st.warning(f"⚡ MEDIUM RISK: This customer has a {prob:.1%} probability of churning.")
            else:
                st.success(f"✅ LOW RISK: This customer has a {prob:.1%} probability of churning.")

        except FileNotFoundError:
            st.error("Model not found. Run `python main.py` first to train models.")


# ─── MODEL PERFORMANCE PAGE ─────────────────────────────────────────────────
elif page == "Model Performance":
    st.header("Model Performance")

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

    roc_path = os.path.join(plots_dir, "roc_curves.png")
    cm_path = os.path.join(plots_dir, "confusion_matrices.png")
    fi_path = os.path.join(plots_dir, "feature_importance.png")

    if os.path.exists(roc_path):
        st.subheader("ROC Curves")
        st.image(roc_path)
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrices")
        st.image(cm_path)
    if os.path.exists(fi_path):
        st.subheader("Feature Importance")
        st.image(fi_path)

    if not any(os.path.exists(p) for p in [roc_path, cm_path, fi_path]):
        st.warning("No plots found. Run `python main.py` first to generate model evaluation plots.")


# ─── UPLOAD & PREDICT PAGE ───────────────────────────────────────────────────
elif page == "Upload & Predict":
    st.header("Batch Prediction - Upload CSV")
    st.markdown("Upload a CSV file with customer data to get batch predictions.")

    st.markdown("""
    **Required columns:** gender, age, tenure, monthly_charges, total_charges,
    contract_type, payment_method, internet_service
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(df_upload)} records")
            st.dataframe(df_upload.head(), use_container_width=True)

            required_cols = ["gender", "age", "tenure", "monthly_charges",
                            "total_charges", "contract_type", "payment_method", "internet_service"]
            missing = [c for c in required_cols if c not in df_upload.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    predictions = []
                    progress = st.progress(0)

                    for i, row in df_upload.iterrows():
                        customer = row[required_cols].to_dict()
                        result = predict_churn(customer)
                        predictions.append(result)
                        progress.progress((i + 1) / len(df_upload))

                    pred_df = pd.DataFrame(predictions)
                    result_df = pd.concat([df_upload, pred_df], axis=1)

                    st.success(f"Predictions complete!")
                    st.dataframe(result_df, use_container_width=True)

                    # Summary
                    churn_count = (pred_df["prediction"] == "Yes").sum()
                    st.metric("Predicted Churners", f"{churn_count} / {len(pred_df)}")

                    # Download
                    csv = result_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "churn_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
