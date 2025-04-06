# MLPilot 🧠

**MLPilot** is an interactive web-based dashboard built using **Streamlit** that allows users to explore, preprocess, train, evaluate, and visualize machine learning models on their own datasets without writing any code.

---

## 🔍 Features

- **Upload and Preview Data**
- **Handle Missing Values** (Simple & KNN Imputation)
- **Encode Categorical Data**
- **Scale Numerical Features** (Standard, MinMax, Robust)
- **Train-Test Split** with slider control
- **Auto ML Task Detection** (Classification / Regression)
- **Select & Train Models** (Scikit-learn + optional XGBoost)
- **Evaluate Models** with metrics like accuracy, precision, recall, F1, MAE, MSE, R2, and more
- **Confusion Matrix & ROC Curve (for classification)**
- **Actual vs Predicted Plots (for regression)**
- **Download Trained Model and Evaluation Report**

---

## 📚 Requirements

Install all necessary packages using:

```bash
pip install -r requirements.txt
```

---

## 🎓 How to Run

```bash
streamlit run app.py
```

---

## 🎓 Example Use Cases
- Teaching basic machine learning concepts interactively
- Quickly evaluating models on new datasets
- Sharing with non-technical stakeholders for ML experimentation

---

## 🌟 Models Supported

### Classification:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Naive Bayes
- Gradient Boosting
- XGBoost *(if installed)*

### Regression:
- Linear Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- Ridge & Lasso
- Gradient Boosting Regressor
- XGBoost Regressor *(if installed)*

---

## 📄 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgements
Built with ❤️ by Maharshi Jani to make machine learning more accessible and interactive.

