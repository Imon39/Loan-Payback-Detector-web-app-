import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_models():
    lgb = joblib.load('pipe_lgb.pkl')
    rgb = joblib.load('pipe_rgb.pkl')
    features = joblib.load('features_list.pkl')
    return lgb, rgb, features


pipe_lgb, pipe_rgb, features_list = load_models()

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Payback Probability Predictor")

with st.expander("ℹ️ How your loan risk is calculated (Feature Info)"):
    st.write("""
    We use advanced metrics to analyze your application:
    * **Loan to Income Ratio:** Total loan divided by your yearly income.
    * **Interest Burden:** The actual cost of interest based on your loan amount.
    * **Credit Utilization:** How your loan amount relates to your credit score.
    * **Income per Debt:** Your remaining income after considering existing debts.
    """)

with st.form("loan_input_form"):
    st.subheader("Enter Applicant Details")
    col1, col2 = st.columns(2)

    input_data = {}


    cat_options = {
        'gender': ['Female', 'Male', 'Other'],
        'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
        'education_level': ["High School", "Master's", "Bachelor's", 'PhD', 'Other'],
        'employment_status': ['Self-employed', 'Employed', 'Unemployed', 'Retired', 'Student'],
        'loan_purpose': ['Other', 'Debt consolidation', 'Home', 'Education', 'Vacation', 'Car', 'Medical', 'Business'],
        'grade_subgrade': ['C3', 'D3', 'C5', 'F1', 'D1', 'D5', 'C2', 'C1', 'F5', 'D4', 'C4', 'D2', 'E5', 'B1', 'B2',
                           'F4', 'A4', 'E1', 'F2', 'B4', 'E4', 'B3', 'E3', 'B5', 'E2', 'F3', 'A5', 'A3', 'A1', 'A2']
    }


    for i, feature in enumerate(features_list):

        if feature in ["loan_to_income_ratio", "interest_burden", "credit_utilization",
                       "income_per_debt_ratio", "income_minus_loan",
                       "loan_income_interaction", "credit_burden"]:
            continue

        with col1 if i % 2 == 0 else col2:
            if feature in cat_options:
                input_data[feature] = st.selectbox(f" Select {feature}", cat_options[feature])
            else:
                if 'annual_income' in feature.lower() or 'loan_amount' in feature.lower():
                    input_data[feature] = st.number_input(f"Enter {feature}", value=5000.0,
                                                          step=500.0)
                elif 'interest_rate' in feature.lower():
                    input_data[feature] = st.number_input(f"Enter {feature}", value=10.0, step=0.1)
                elif 'debt_to_income_ratio' in feature.lower():
                    input_data[feature] = st.number_input(f"Enter {feature}", value=0.1, step=0.01)
                elif 'credit_score' in feature.lower():
                    input_data[feature] = st.number_input(f"Enter {feature}", value=600.0,
                                                          step=10.0)
                else:
                    input_data[feature] = st.number_input(f"Enter {feature}", value=1.0, step=1.0)

    submit = st.form_submit_button("Calculate Risk")

if submit:
    df_input = pd.DataFrame([input_data])

    df_input["loan_to_income_ratio"] = df_input["loan_amount"] / (df_input["annual_income"] + 1)
    df_input["interest_burden"] = df_input["loan_amount"] * df_input["interest_rate"] / 100
    df_input["credit_utilization"] = df_input["loan_amount"] / (df_input["credit_score"] + 1)
    df_input["income_per_debt_ratio"] = df_input["annual_income"] * (1 - df_input["debt_to_income_ratio"])
    df_input["income_minus_loan"] = df_input["annual_income"] - df_input["loan_amount"]
    df_input["loan_income_interaction"] = df_input["loan_amount"] * df_input["loan_to_income_ratio"]
    df_input["credit_burden"] = df_input["credit_score"] / (1 + df_input["debt_to_income_ratio"])

    df_input = df_input[features_list]

    try:
        pred_lgb = pipe_lgb.predict_proba(df_input)[:, 1]
        pred_rgb = pipe_rgb.predict_proba(df_input)[:, 1]

        final_pred = 0.65 * pred_lgb + 0.35 * pred_rgb
        probability = np.clip(final_pred, 0, 1)[0]

        st.markdown("---")
        st.write(f"### Payback Probability: **{probability:.2%}**")
        st.progress(float(probability))

        if probability > 0.8:
            st.success("✅ **Verdict:** Strong candidate for loan approval.")
        elif probability > 0.5:
            st.warning("🟡 **Verdict:** Moderate risk. Further review needed.")
        else:
            st.error("❌ **Verdict:** High risk of default. Loan rejection recommended.")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

st.markdown("---")
st.caption("Developed by **IMON HOSSAIN** | Kaggle Swag Participant")