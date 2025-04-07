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
- **Model Comparison** to compare multiple models side-by-side with evaluation metrics along with the download comparison button
- **Select & Train Models** (Scikit-learn + optional XGBoost)
- **Hyperparameter Tuning** – both automatic (GridSearchCV) and manual options 
- **Evaluate Models** with metrics like accuracy, precision, recall, F1, MAE, MSE, R2, and more
- **Confusion Matrix & ROC Curve (for classification)**
- **Actual vs Predicted Plots (for regression)**
- **Download Trained Model and Evaluation Report**
- **Reset Button** after dataset upload to start fresh (refresh app state)
- **Prediction Interface** to upload new data or enter manually
- **View Predictions** (and probabilities for classification)

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

## 🛠️ Hyperparameter Tuning

MLPilot supports two powerful ways to fine-tune models:

🔧 Manual Tuning  
Customize hyperparameters directly before training.

🤖 Automatic Tuning  
Enable GridSearchCV to automatically find the best parameters with cross-validation.

## 📄 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgements
Built with ❤️ by Maharshi Jani to make machine learning more accessible and interactive.

