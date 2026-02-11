import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🏦 Bank Term Deposit Predictor",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. LOAD MODEL (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("rf_pipeline.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------------------------------------------------------
# 3. LOAD DATASET (STATIC)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank-additional/bank-additional.csv", sep=";")

df = load_data()

# -----------------------------------------------------------------------------
# 4. APP TITLE
# -----------------------------------------------------------------------------
st.title("🏦 Bank Term Deposit Prediction System")
st.markdown(
    """
This application provides:
- 📊 **Exploratory Data Analysis (EDA)**
- 🔮 **Production-level prediction** using a pre-trained Random Forest pipeline  
"""
)

# -----------------------------------------------------------------------------
# 5. TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Customer Prediction"])

# =============================================================================
# TAB 1 — EDA
# =============================================================================
with tab1:
    st.header("Exploratory Data Analysis")

    # --- Target Distribution ---
    st.subheader("Target Variable Distribution")
    fig1, ax1 = plt.subplots()
    df["y"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Subscription")
    ax1.set_ylabel("Count")
    ax1.set_title("Target Class Distribution (y)")
    st.pyplot(fig1)

    st.markdown("""
- The dataset is **imbalanced**.
- This justifies the use of **SMOTE** during training.
""")

    # --- Duration Leakage ---
    if "duration" in df.columns:
        st.subheader("Call Duration vs Target")
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x="duration", hue="y", bins=40, ax=ax2)
        ax2.set_title("Duration vs Subscription Outcome")
        st.pyplot(fig2)

        st.warning("""
⚠ **Data Leakage Notice**

`duration` is only known **after** the call is completed.
It was therefore **excluded from model training** to ensure realistic predictions.
""")

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap (Numerical Features)")
    num_df = df.select_dtypes(include=["number"])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax3)
    ax3.set_title("Feature Correlation Matrix")
    st.pyplot(fig3)

    # --- Raw Data Preview ---
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# =============================================================================
# TAB 2 — PRODUCTION PREDICTION
# =============================================================================
with tab2:
    st.header("Customer Simulation & Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
            )
        job = st.selectbox("Job", [
            "admin.","technician","management","blue-collar",
            "services","housemaid","retired","self-employed",
            "entrepreneur","unemployed","student"
        ])
        education = st.selectbox("Education", [
            "basic.4y","basic.6y","basic.9y",
            "high.school","professional.course",
            "university.degree","illiterate"
        ])
        campaign = st.number_input(
            "Campaign Calls",
            min_value=1,
            max_value=50,
            value=2,
            step=1
            )
        housing = st.selectbox("Housing Loan", ["yes", "no"])

    with col2:
        loan = st.selectbox("Personal Loan", ["yes", "no"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Month", [
            "jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"
        ])
        poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "unknown"])
        emp_var_rate = st.selectbox("Employment Variation Rate",
                                    [-3.4, -2.9, -2.1, -1.8, -1.1, 0.1, 1.1])

    with col3:
        cons_price_idx = st.selectbox("Consumer Price Index",
                                      [92.2, 92.8, 93.2, 93.9, 94.4])
        cons_conf_idx = st.selectbox("Consumer Confidence Index",
                                     [-50.8, -46.2, -42.0, -36.4])
        euribor3m = st.selectbox("Euribor 3M Rate",
                                 [0.7, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0])
        nr_employed = st.selectbox("Number of Employees",
                                   [4963.6, 5008.7, 5076.2, 5099.1, 5228.1])
        pdays = st.number_input(
            "Days Since Last Contact (999 = never)",
            min_value=0,
            max_value=999,
            value=999,
            step=1
            )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
            )


    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING (IDENTICAL TO TRAINING)
    # -------------------------------------------------------------------------
    input_df = pd.DataFrame([{
        "age": age,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "education": education,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "job": job,

        "has_university_degree": int(education == "university.degree"),
        "married": 0,
        "is_high_campaign": int(campaign > 3),
        "white_collar": int(job in ["admin.","technician","management"]),
        "blue_collar": int(job in ["blue-collar","services","housemaid"]),
        "other_collar": int(job in ["retired","self-employed","entrepreneur","unemployed","student"]),
        "econ_interact": euribor3m * emp_var_rate,
        "age_euribor_interact": age * euribor3m,
        "a_age_euribor_interact": age * euribor3m,
        "m_age_euribor_interact": age * euribor3m,
        "r_age_euribor_interact": age * euribor3m,
        "campaign_conf_interact": campaign * cons_conf_idx,
        "cpi_euribor_diff": cons_price_idx - euribor3m,
        "previous_contact": int(pdays != 999),
        "has_multiple_loans": int(housing == "yes" and loan == "yes"),
        "economic_stress": int(emp_var_rate < 0)
    }])

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    st.markdown("---")

    if st.button("🔮 Predict"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"✅ **YES** — High chance of subscription (**{prob:.2%}**)")
        else:
         st.error(f"❌ **NO** — Low chance of subscription (**{prob:.2%}**)")


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("© ADA 442 – Machine Learning Project by Anıl Doni, Emre Beldağ, Volkan Sarı")
