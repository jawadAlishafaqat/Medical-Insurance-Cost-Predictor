import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide"
)

# ================= LOAD =================
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= CSS =================
st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.metric-value {
    font-size: 32px;
    color: #00ffcc;
    font-weight: bold;
}

.metric-label {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("💰 Insurance Cost Prediction Dashboard")

st.markdown(
    "Predict medical insurance charges using Lasso Regression"
)

# ================= SIDEBAR =================
st.sidebar.title("Enter Customer Information")

age = st.sidebar.slider("Age", 18, 65, 30)

bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)

children = st.sidebar.slider("Children", 0, 5, 0)

sex_male = st.sidebar.selectbox(
    "Gender",
    [0,1]
)

smoker_yes = st.sidebar.selectbox(
    "Smoker",
    [0,1]
)

region_northwest = st.sidebar.selectbox(
    "Northwest",
    [0,1]
)

region_southeast = st.sidebar.selectbox(
    "Southeast",
    [0,1]
)

region_southwest = st.sidebar.selectbox(
    "Southwest",
    [0,1]
)

# ================= FEATURE ENGINEERING =================
high_bmi = 1 if bmi > 30 else 0

age_bmi = age * bmi

# ================= INPUT =================
features = np.array([[
    age,
    bmi,
    children,
    sex_male,
    smoker_yes,
    region_northwest,
    region_southeast,
    region_southwest,
    high_bmi,
    age_bmi
]])

scaled_features = scaler.transform(features)

prediction = model.predict(scaled_features)[0]

# ================= CARDS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Cost</div>
        <div class="metric-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BMI</div>
        <div class="metric-value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Age</div>
        <div class="metric-value">{age}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= CHART =================
st.markdown("## 📊 Health Analysis")

chart_data = pd.DataFrame({
    "Feature": ["BMI", "Children"],
    "Value": [bmi, children]
})

fig, ax = plt.subplots()

ax.bar(chart_data["Feature"], chart_data["Value"])

st.pyplot(fig)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "Developed using Streamlit + Lasso Regression 🚀"
)
