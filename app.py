import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Insurance AI Dashboard",
    page_icon="💎",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1520607162513-77705c0f0d4a");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* GLASS EFFECT CONTAINER */
.glass {
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(31,38,135,0.25);
    border: 1px solid rgba(255,255,255,0.25);
}

/* TITLE */
.title-text {
    font-size: 52px;
    font-weight: bold;
    color: white;
    text-align: center;
}

/* SUBTITLE */
.sub-text {
    text-align: center;
    color: white;
    font-size: 20px;
    margin-bottom: 30px;
}

/* METRIC CARD */
.metric-card {
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* METRIC VALUE */
.metric-value {
    font-size: 36px;
    font-weight: bold;
    color: #00ffcc;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
}

/* CHART CONTAINER */
.chart-box {
    background: rgba(255,255,255,0.18);
    padding: 20px;
    border-radius: 18px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(
    '<div class="title-text">💎 Insurance Cost Prediction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">Industry-Level Machine Learning Dashboard using Lasso Regression</div>',
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.title("⚙ Customer Information")

age = st.sidebar.slider("Age", 18, 65, 30)

bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)

children = st.sidebar.slider("Children", 0, 5, 0)

sex_male = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoker_yes = st.sidebar.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region_northwest = st.sidebar.selectbox(
    "Northwest Region",
    [0,1]
)

region_southeast = st.sidebar.selectbox(
    "Southeast Region",
    [0,1]
)

region_southwest = st.sidebar.selectbox(
    "Southwest Region",
    [0,1]
)

# ================= CONVERT =================
sex_male = 1 if sex_male == "Male" else 0
smoker_yes = 1 if smoker_yes == "Yes" else 0

# ================= FEATURE ENGINEERING =================
high_bmi = 1 if bmi > 30 else 0

age_bmi = age * bmi

# ================= INPUT ARRAY =================
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

# ================= SCALING =================
scaled_features = scaler.transform(features)

# ================= PREDICTION =================
prediction = model.predict(scaled_features)[0]

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>💰 Predicted Cost</h3>
        <div class="metric-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 BMI</h3>
        <div class="metric-value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎂 Age</h3>
        <div class="metric-value">{age}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= CHART =================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="glass"><h2 style="color:white;">📈 Health Risk Analysis</h2></div>',
    unsafe_allow_html=True
)

chart_data = pd.DataFrame({
    "Feature": ["BMI", "Children"],
    "Value": [bmi, children]
})

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(
    chart_data["Feature"],
    chart_data["Value"]
)

ax.set_title("Health Factors")

st.pyplot(fig)

# ================= INSIGHT BOX =================
st.markdown(f"""
<div class="glass">
<h2 style="color:white;">🧠 AI Insight</h2>

<p style="color:white;font-size:18px;">
This customer is predicted to have an estimated insurance cost of 
<b>${prediction:,.0f}</b>.
</p>

<p style="color:white;">
Higher BMI and smoking status significantly influence insurance charges.
</p>

</div>
""", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center style="color:white;">
Built with ❤️ using Streamlit + Lasso Regression
</center>
""", unsafe_allow_html=True)
