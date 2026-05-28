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
    background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* DARK OVERLAY */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(5, 10, 20, 0.70);
    z-index: -1;
}

/* TITLE */
.title-text {
    font-size: 58px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-top: 10px;
    letter-spacing: 1px;
}

/* SUBTITLE */
.sub-text {
    text-align: center;
    color: #dbe4ee;
    font-size: 21px;
    margin-bottom: 40px;
}

/* GLASS EFFECT */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    border-radius: 24px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

/* METRIC CARD */
.metric-card {
    background: linear-gradient(
        135deg,
        rgba(15,23,42,0.95),
        rgba(30,41,59,0.88)
    );
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}

/* METRIC TITLE */
.metric-title {
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 12px;
}

/* METRIC VALUE */
.metric-value {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(15,23,42,0.97),
        rgba(30,41,59,0.95)
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    height: 3.3em;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #2563eb
    );
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* TEXT */
h1,h2,h3,h4,p,label {
    color: white !important;
}

/* INFO BOX */
.info-box {
    background: rgba(15,23,42,0.88);
    border-left: 5px solid #38bdf8;
    padding: 22px;
    border-radius: 18px;
    margin-top: 20px;
    color: white;
    font-size: 17px;
}

/* RISK BADGES */
.low-risk {
    color: #22c55e;
    font-size: 22px;
    font-weight: bold;
}

.medium-risk {
    color: #facc15;
    font-size: 22px;
    font-weight: bold;
}

.high-risk {
    color: #ef4444;
    font-size: 22px;
    font-weight: bold;
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

age = st.sidebar.slider(
    "Age",
    18,
    65,
    30
)

bmi = st.sidebar.slider(
    "BMI",
    15.0,
    50.0,
    25.0
)

children = st.sidebar.slider(
    "Children",
    0,
    5,
    0
)

sex = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoker = st.sidebar.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

# ================= ENCODING =================
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

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

# ================= SCALE =================
scaled_features = scaler.transform(features)

# ================= PREDICTION =================
prediction = model.predict(scaled_features)[0]

prediction = max(prediction, 0)

# ================= RISK LEVEL =================
if prediction < 10000:
    risk_level = "LOW RISK"
    risk_class = "low-risk"

elif prediction < 30000:
    risk_level = "MEDIUM RISK"
    risk_class = "medium-risk"

else:
    risk_level = "HIGH RISK"
    risk_class = "high-risk"

# ================= METRIC CARDS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💰 Predicted Insurance Cost</div>
        <div class="metric-value">${prediction:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📊 BMI Score</div>
        <div class="metric-value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🩺 Risk Category</div>
        <div class="{risk_class}">
            {risk_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= CHART SECTION =================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="glass"><h2>📈 Health Analytics</h2></div>',
    unsafe_allow_html=True
)

chart_data = pd.DataFrame({
    "Feature": ["BMI", "Children", "Age"],
    "Value": [bmi, children, age]
})

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    chart_data["Feature"],
    chart_data["Value"]
)

ax.set_title("Customer Health Indicators")

st.pyplot(fig)

# ================= PIE CHART =================
st.markdown(
    '<div class="glass"><h2>🥧 Lifestyle Overview</h2></div>',
    unsafe_allow_html=True
)

pie_data = [bmi, age, children + 1]
pie_labels = ["BMI", "Age", "Family"]

fig2, ax2 = plt.subplots(figsize=(5,5))

ax2.pie(
    pie_data,
    labels=pie_labels,
    autopct='%1.1f%%'
)

st.pyplot(fig2)

# ================= AI INSIGHT =================
st.markdown(f"""
<div class="info-box">

<h2>🧠 AI Insight</h2>

<p>
The estimated insurance cost for this customer is 
<b>${prediction:,.0f}</b>.
</p>

<p>
Factors such as smoking habits, BMI, age, and family size
strongly influence the insurance premium.
</p>

<p>
Customers with higher BMI and smoking status generally
show increased medical risk and insurance charges.
</p>

</div>
""", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center style="color:white;font-size:18px;">
Built with ❤️ using Streamlit + Lasso Regression
</center>
""", unsafe_allow_html=True)
