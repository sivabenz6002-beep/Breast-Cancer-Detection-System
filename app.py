import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.markdown("""
<style>

/* ---------- Light Blue Background ---------- */
.stApp {
    background-color: #eaf4ff;
}

/* ---------- Header Box ---------- */
.header-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

/* ---------- Titles ---------- */
h1, h2, h3, h4 {
    color: #000000 !important;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
    text-align: center;
}

/* ---------- Metric Title ---------- */
.metric-title {
    font-size: 15px;
    font-weight: 600;
    color: #333333;
}

/* ---------- Metric Value (Medical Blue) ---------- */
.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #1565c0;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #d6eaff;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background-color: #1976d2;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: none;
}

/* ---------- Improve Text Visibility ---------- */
p, span, label {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2 = st.columns([4,1])

with col1:
    st.markdown("""
    <div class="header-box">
        <h2 style="color:#2c3e50;">🩺 Breast Cancer Detection System</h2>
        <p style="color:gray;">AI-powered medical diagnosis dashboard</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=100)

st.write("")

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model, feature_names, _, X_test, y_test = pickle.load(f)

# ---------------- METRICS ----------------
y_pred = model.predict(X_test)

accuracy_val = accuracy_score(y_test, y_pred)
precision_val = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1_val = f1_score(y_test, y_pred)

st.subheader("📊 Model Performance Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Accuracy</div>
    <div class="metric-value">{accuracy_val*100:.2f}%</div>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Precision</div>
    <div class="metric-value">{precision_val:.2f}</div>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="metric-card">
    <div class="metric-title">Recall</div>
    <div class="metric-value">{recall_val:.2f}</div>
</div>
""", unsafe_allow_html=True)

m4.markdown(f"""
<div class="metric-card">
    <div class="metric-title">F1 Score</div>
    <div class="metric-value">{f1_val:.2f}</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Patient Input Parameters")

selected_features = feature_names[:10]
input_values = []

for feature in selected_features:
    val = st.sidebar.slider(feature, 0.0, 20.0, 1.0)
    input_values.append(val)

while len(input_values) < len(feature_names):
    input_values.append(0)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Diagnosis"):

    input_array = np.array([input_values])
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]

    st.subheader("🧾 Diagnosis Result")

    if prediction == 1:
        st.success("Benign (Non-Cancerous)")
    else:
        st.error("Malignant (Cancerous)")

    # Probability Chart
    st.subheader("📊 Prediction Probability")

    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(["Malignant", "Benign"], probability)
    ax_prob.set_ylabel("Probability")
    st.pyplot(fig_prob)

# ---------------- MODEL VISUALS ----------------
st.subheader("📈 Model Evaluation")

colA, colB = st.columns(2)

# ROC Curve
with colA:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0,1],[0,1],'--')
    ax_roc.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    st.pyplot(fig_roc)

# Confusion Matrix
with colB:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

importance = np.abs(model.coef_[0])

fig_imp, ax_imp = plt.subplots()
ax_imp.barh(feature_names[:10], importance[:10])
ax_imp.set_xlabel("Importance Score")
st.pyplot(fig_imp)