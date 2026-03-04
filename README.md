# 🩺 Breast Cancer Detection System (Streamlit ML App)

An interactive **Machine Learning Web Application** built using **Streamlit** that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on medical diagnostic features.

The system uses a trained **Logistic Regression model** and provides a clean medical-style dashboard with performance metrics and visual analytics to help understand model behavior.

---

# 🚀 Features

* 🧠 Machine Learning model for cancer diagnosis
* 📊 Real-time prediction using patient feature inputs
* 📈 Model performance dashboard
* 🔢 Accuracy, Precision, Recall, and F1-score metrics
* 📉 ROC Curve visualization
* 📊 Confusion Matrix
* 📈 Feature Importance plot
* 📊 Prediction probability chart
* 🎨 Clean medical-style UI with light blue theme
* 📱 Interactive sidebar input controls

---

# 🧠 Machine Learning Model

The application uses the **Breast Cancer Wisconsin dataset** available in **Scikit-learn**.

### Algorithm Used

* Logistic Regression

### Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

These metrics help evaluate how well the model detects malignant tumors.

---

# 🖥 Application Interface

### 📊 Model Performance Metrics

Displays important evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score

### 🔍 Patient Input Panel

Users can adjust diagnostic feature values using **interactive sliders** in the sidebar.

### 🧾 Diagnosis Prediction

The model predicts whether the tumor is:

* **Benign (Non-Cancerous)**
* **Malignant (Cancerous)**

The system also displays prediction probability.

### 📈 Model Evaluation Visualizations

The application includes several visualizations:

* ROC Curve
* Confusion Matrix
* Feature Importance Chart
* Prediction Probability Bar Chart

---

# 📁 Project Structure

```
breast-cancer-detection-app
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

| File             | Description                    |
| ---------------- | ------------------------------ |
| app.py           | Streamlit web application      |
| model.pkl        | Trained machine learning model |
| requirements.txt | Python dependencies            |
| README.md        | Project documentation          |

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/breast-cancer-detection-app.git
cd breast-cancer-detection-app
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

Then open the browser:

```
http://localhost:8501
```

---

# 🎨 User Interface

The application uses a **clean healthcare-style dashboard design**:

* Light blue background
* White metric cards
* Medical blue metric highlights
* Clean readable typography
* Structured layout with sidebar controls

---

# 🌍 Deployment (Streamlit Cloud)

1. Push your project to **GitHub**
2. Visit

```
https://share.streamlit.io
```

3. Connect your repository
4. Select:

```
app.py
```

5. Click **Deploy**

Your app will be live within seconds.

---

# 🧪 Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Matplotlib

---

# 📌 Future Improvements

Possible future upgrades:

* SHAP model explainability
* Multiple ML model comparison
* CSV batch prediction
* Dark mode UI
* Feature correlation heatmap
* Patient data report export

---

# 👨‍💻 Author

**Siva Balan G**
