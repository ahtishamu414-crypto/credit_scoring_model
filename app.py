import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CreditWise AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: #0f172a;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #111827;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #60a5fa;
}

.subtitle {
    color: #94a3b8;
    font-size: 1rem;
}

.card {
    background: #1e293b;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #334155;
    text-align: center;
}

.approved {
    background: #065f46;
    padding: 1rem;
    border-radius: 12px;
    border-left: 5px solid #22c55e;
}

.rejected {
    background: #7f1d1d;
    padding: 1rem;
    border-radius: 12px;
    border-left: 5px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>💳 CreditWise AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Credit Risk Prediction System</div>", unsafe_allow_html=True)
st.divider()

# =========================
# LOAD MODELS (YOU MUST HAVE THESE FILES)
# =========================
@st.cache_resource
def load_models():
    return {
        'lr': joblib.load('lr_model.pkl'),
        'dt': joblib.load('dt_model.pkl'),
        'rf': joblib.load('rf_model.pkl'),
        'scaler': joblib.load('scaler.pkl'),
        'features': joblib.load('feature_names.pkl')
    }

assets = load_models()

# =========================
# BUILD INPUT FUNCTION
# (YOU MUST ALREADY HAVE THIS FROM YOUR PROJECT)
# =========================
def build_input(age, gender, education, emp_exp, income, loan_amt,
                int_rate, intent, credit_score, cred_hist, prev_default):

    raw = {
        'person_age': age,
        'person_gender': 1 if gender == "Male" else 0,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'loan_amnt': loan_amt,
        'loan_int_rate': int_rate,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': 1 if prev_default == "Yes" else 0
    }

    features = assets['features']
    row = {f: 0 for f in features}
    row.update(raw)

    return pd.DataFrame([row])[features]

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=70)
    st.markdown("### 👤 Applicant Profile")

    age = st.slider("Age", 18, 75, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["High School","Associate","Bachelor","Master","Doctorate"])
    emp_exp = st.slider("Experience", 0, 40, 5)

    st.markdown("### 💰 Financials")
    income = st.number_input("Income", 10000, 500000, 60000)
    loan_amt = st.number_input("Loan Amount", 500, 50000, 10000)
    int_rate = st.slider("Interest Rate", 5.0, 25.0, 12.0)

    intent = st.selectbox("Loan Purpose",
        ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])

    st.markdown("### 📊 Credit Profile")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    cred_hist = st.slider("Credit History", 0, 30, 5)
    prev_default = st.radio("Default History", ["No","Yes"])

    model_choice = st.selectbox("Model",
        ["Random Forest","Logistic Regression","Decision Tree"])

    predict_btn = st.button("🚀 Predict Risk")

# =========================
# MAIN LOGIC (IMPORTANT FIX)
# =========================
model_map = {
    'Random Forest': assets['rf'],
    'Logistic Regression': assets['lr'],
    'Decision Tree': assets['dt']
}

if predict_btn:

    input_df = build_input(
        age, gender, education, emp_exp, income, loan_amt,
        int_rate, intent, credit_score, cred_hist, prev_default
    )

    model = model_map[model_choice]

    if model_choice == "Logistic Regression":
        prob = model.predict_proba(assets['scaler'].transform(input_df))[0][1]
    else:
        prob = model.predict_proba(input_df)[0][1]

    approved = prob >= 0.5

    # =========================
    # RESULT
    # =========================
    st.markdown("## 🎯 Decision Output")

    col1, col2 = st.columns([2,1])

    with col1:
        if approved:
            st.markdown(f"""
            <div class="approved">
                <h2>✅ APPROVED</h2>
                <h3>Probability: {prob*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rejected">
                <h2>❌ REJECTED</h2>
                <h3>Probability: {prob*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Key Stats")
        st.metric("Credit Score", credit_score)
        st.metric("Income", income)
        st.metric("Loan", loan_amt)

    # =========================
    # MODEL COMPARISON
    # =========================
    st.markdown("## 🤖 Model Comparison")

    cols = st.columns(3)

    for col, (name, mdl) in zip(cols, model_map.items()):

        if name == "Logistic Regression":
            p = mdl.predict_proba(assets['scaler'].transform(input_df))[0][1]
        else:
            p = mdl.predict_proba(input_df)[0][1]

        with col:
            st.markdown(f"""
            <div class="card">
                <h4>{name}</h4>
                <h2>{p*100:.1f}%</h2>
                <p>{"APPROVED" if p>=0.5 else "REJECTED"}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👈 Enter details in sidebar and click Predict")