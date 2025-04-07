import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import io
import pickle

# Page Config
st.set_page_config(page_title="📊 MLPilot: ML Dashboard", layout="wide")

st.title("📊 MLPilot: Machine Learning Dashboard")
st.markdown("Upload your dataset and explore interactive ML tools!")

# Upload Section
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# Session State Initialization
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_copy" not in st.session_state:
    st.session_state.df_copy = None

if uploaded_file:
    if st.session_state.df_original is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_original = df.copy()
        st.session_state.df_copy = df.copy()

    df = st.session_state.df_original
    df_copy = st.session_state.df_copy

    st.success("✅ File uploaded successfully!")

    # Preview
    st.markdown("### 👀 Preview of Dataset")
    st.dataframe(df.head())
    st.markdown("---")

    # --- EDA Section ---
    st.header("🔍 Explore & Understand Your Data (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔽 Select a Column to Explore")
        selected_col = st.selectbox("Choose a column", df.columns)

        st.markdown("📊 **Visual Options**")
        show_corr = st.checkbox("Show Correlation Matrix (Numeric Only)")
        show_missing = st.checkbox("Show Missing Value Heatmap")

    with col2:
        st.subheader("📈 Data Insights")

        if selected_col:
            if df[selected_col].dtype in ["float64", "int64"]:
                st.markdown(f"**📊 Numeric Distribution: `{selected_col}`**")
                st.markdown("### 📈 Histogram")
                fig1, ax1 = plt.subplots()
                sns.histplot(df[selected_col].dropna(), kde=True, ax=ax1)
                ax1.set_xlabel(selected_col)
                ax1.set_title(f"Histogram of {selected_col}")
                st.pyplot(fig1)

                st.markdown("### 📦 Boxplot")
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=df[selected_col], ax=ax2)
                ax2.set_xlabel(selected_col)
                ax2.set_title(f"Boxplot of {selected_col}")
                st.pyplot(fig2)
            else:
                st.markdown(f"**📊 Categorical Value Counts: `{selected_col}`**")
                st.markdown("### 📈 Bar Chart")
                value_counts = df[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, "count"]
                fig = px.bar(value_counts, x=selected_col, y="count",
                             labels={selected_col: selected_col, "count": "Count"})
                fig.update_layout(title=f"Bar Chart of {selected_col}")
                st.plotly_chart(fig)

        if show_corr:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.markdown("### 🧮 Correlation Matrix")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
                ax3.set_title("Correlation Matrix")
                st.pyplot(fig3)
                st.caption("🔗 Correlation Heatmap for numeric columns")
            else:
                st.warning("⚠️ No numeric columns found to compute correlation matrix.")

        if show_missing:
            st.markdown("### 🕳️ Missing Values Heatmap")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.heatmap(df.isnull(), cbar=False, cmap="Blues", ax=ax4)
            ax4.set_title("Missing Value Heatmap")
            st.pyplot(fig4)
            st.caption("🧩 Heatmap of missing values in the dataset")

    st.markdown("---")
    st.info("✅ You’ve completed the EDA section. Now you can move to preprocessing!")

    # ---Feature Extraction ---
    st.markdown("## 🧠 Feature Selection")
    with st.expander("🎯 Drop Unwanted Columns"):
        drop_cols = st.multiselect("Select columns to drop from dataset", st.session_state.df_copy.columns.tolist(), key="drop_cols")
        if st.button("🗑️ Drop Selected Columns"):
            st.session_state.df_copy.drop(columns=drop_cols, inplace=True)
            st.success(f"✅ Dropped columns: {', '.join(drop_cols)}")
            st.dataframe(st.session_state.df_copy.head())

    # with st.expander("📈 Correlation Heatmap (Numerical Features Only)"):
    num_cols = st.session_state.df_copy.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        if st.button("📊 Show Correlation Heatmap"):
            corr_matrix = st.session_state.df_copy[num_cols].corr()
            # fig9, ax9 = plt.subplots(figsize=(6, 4))
            # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax9)
            plt.figure(figsize=(6,4))
            sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",cbar=True,square=True)
            st.pyplot(plt.gcf())
    else:
        st.info("ℹ️ No numerical features to generate correlation heatmap.")
    
    st.markdown("---")

    # --- Identify Missing Columns ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_missing_cols = [col for col in numeric_cols if col in df_copy.columns and df_copy[col].isnull().sum() > 0]
    categorical_missing_cols = [col for col in categorical_cols if col in df_copy.columns and df_copy[col].isnull().sum() > 0]

    # --- Handle Missing Numerical Data ---
    st.markdown("## 🧩 Handle Missing Data")
    with st.expander("🧮 Handle Missing Numerical Data"):
        if numeric_missing_cols:
            selected_num_cols = st.multiselect("Select Numeric Columns to Impute", numeric_missing_cols)

            imputer_type = st.selectbox(
                "Choose Imputation Strategy",
                ("Mean", "Median", "Most Frequent", "KNN Imputer")
            )

            if imputer_type != "KNN Imputer":
                strategy_map = {
                    "Mean": "mean",
                    "Median": "median",
                    "Most Frequent": "most_frequent"
                }
                strategy = strategy_map[imputer_type]
                if st.button("🚀 Impute Numeric Data"):
                    try:
                        imp = SimpleImputer(strategy=strategy)
                        st.session_state.df_copy[selected_num_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_num_cols])
                        st.success(f"✅ Missing values imputed using '{strategy}' strategy.")
                        st.dataframe(st.session_state.df_copy[selected_num_cols].head())
                    except Exception as e:
                        st.warning(f"⚠️ Error: {str(e)}")
            else:
                neighbors = st.slider("Select number of neighbors for KNN", min_value=1, max_value=10, value=3)
                if st.button("🚀 Impute using KNN"):
                    try:
                        imp = KNNImputer(n_neighbors=neighbors)
                        st.session_state.df_copy[selected_num_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_num_cols])
                        st.success(f"✅ Missing values imputed using KNN with {neighbors} neighbors.")
                        st.dataframe(st.session_state.df_copy[selected_num_cols].head())
                    except Exception as e:
                        st.warning(f"⚠️ Error: {str(e)}")
        else:
            st.info("✅ No missing values found in numeric columns.")

    # --- Handle Missing Categorical Data ---
    with st.expander("🔤 Handle Missing Categorical Data"):
        if categorical_missing_cols:
            selected_cat_cols = st.multiselect("Select Categorical Columns to Impute", categorical_missing_cols)

            cat_strategy = st.selectbox(
                "Choose Imputation Strategy for Categorical Columns",
                ("Most Frequent", "Constant Value")
            )

            if cat_strategy == "Most Frequent":
                if st.button("🚀 Impute Categorical (Most Frequent)"):
                    try:
                        imp = SimpleImputer(strategy='most_frequent')
                        st.session_state.df_copy[selected_cat_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_cat_cols])
                        st.success("✅ Missing categorical values filled with most frequent value.")
                        st.dataframe(st.session_state.df_copy[selected_cat_cols].head())
                    except Exception as e:
                        st.warning(f"⚠️ Error: {str(e)}")
            else:
                constant_val = st.text_input("Enter constant value to replace missing data", value="Unknown")
                if st.button("🚀 Impute Categorical (Constant Value)"):
                    try:
                        imp = SimpleImputer(strategy='constant', fill_value=constant_val)
                        st.session_state.df_copy[selected_cat_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_cat_cols])
                        st.success(f"✅ Missing values replaced with constant: '{constant_val}'.")
                        st.dataframe(st.session_state.df_copy[selected_cat_cols].head())
                    except Exception as e:
                        st.warning(f"⚠️ Error: {str(e)}")
        else:
            st.info("✅ No missing values found in categorical columns.")

    # --- Final Preview of Cleaned Dataset ---
    st.markdown("---")
    st.header("🧾 Updated Dataset Preview (After Imputation)")
    st.dataframe(st.session_state.df_copy.head(20))
    st.success("✅ This is your cleaned dataset after all missing values have been handled.")

    # --- Reset Button ---
    if st.button("🔄 Reset Changes"):
        st.session_state.df_copy = st.session_state.df_original.copy()
        st.success("✅ Dataset has been reset to original state.")

    # --- Scaling & Encoding ---
    st.markdown("---")
    st.header("🧮 Scaling & Encoding")

    # Create copy for X (without target for now)
    X = st.session_state.df_copy.copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    with st.expander("📐 Scale Numerical Features"):
        if num_cols:
            selected_scale_cols = st.multiselect("Select Numeric Columns to Scale", num_cols, key="scale_cols")

            scaler_option = st.selectbox("Choose a Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

            if st.button("🚀 Scale Selected Columns"):
                if selected_scale_cols:
                    scaler_map = {
                        "StandardScaler": StandardScaler(),
                        "MinMaxScaler": MinMaxScaler(),
                        "RobustScaler": RobustScaler()
                    }

                    try:
                        scaler = scaler_map[scaler_option]
                        scaled_values = scaler.fit_transform(X[selected_scale_cols])
                        st.session_state.df_copy[selected_scale_cols] = scaler.fit_transform(X[selected_scale_cols])
                        st.success(f"✅ Columns scaled using {scaler_option}.")
                        st.dataframe(st.session_state.df_copy[selected_scale_cols].head())
                    except Exception as e:
                        st.warning(f"⚠️ Scaling Error: {str(e)}")
                else:
                    st.warning("⚠️ Please select at least one column to scale.")
        else:
            st.warning("ℹ️ No numeric columns to scale.")

    with st.expander("🔡 Encode Categorical Features"):
        if cat_cols:
            selected_encode_cols = st.multiselect("Select Categorical Columns to Encode", cat_cols, key="encode_cols")

            encoding_type = st.selectbox("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])

            if st.button("🚀 Encode Selected Columns"):
                if selected_encode_cols:
                    try:
                        if encoding_type == "Label Encoding":
                            for col in selected_encode_cols:
                                le = LabelEncoder()
                                st.session_state.df_copy[col] = le.fit_transform(st.session_state.df_copy[col].astype(str))
                            st.success("✅ Label Encoding applied.")
                            st.dataframe(st.session_state.df_copy[selected_encode_cols].head())

                        elif encoding_type == "One-Hot Encoding":
                            try:
                                original_columns = st.session_state.df_copy.columns.tolist()
                                st.session_state.df_copy = pd.get_dummies(
                                    st.session_state.df_copy, columns=selected_encode_cols, drop_first=True
                                )

                            # Update cat_cols in session state if needed
                                new_columns = [col for col in st.session_state.df_copy.columns if col not in original_columns]
                                st.session_state.cat_cols = [
                                    col for col in cat_cols if col not in selected_encode_cols
                                ] + new_columns

                                st.success("✅ One-Hot Encoding applied.")
                                st.dataframe(st.session_state.df_copy.head())
                            except Exception as e:
                                st.warning(f"⚠️ Encoding Error: {str(e)}")
                    except Exception as e:
                        st.warning(f"⚠️ Encoding Error: {str(e)}")
                else:
                    st.warning("⚠️ Please select at least one column to encode.")
        else:
            st.info("ℹ️ No categorical columns to encode.")


    # --- Train-Test Split ---
    st.markdown("---")
    with st.expander("🧪 Train-Test Split"):
        st.subheader("Split Your Dataset")
        test_size = st.slider("Choose test set size (%)", min_value=10, max_value=50, value=20, step=5)
        target_col = st.selectbox("🎯 Select target column", options=st.session_state.df_copy.columns)

        if st.button("🚀 Split Dataset"):
            try:
                X = st.session_state.df_copy.drop(columns=[target_col])
                y = st.session_state.df_copy[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
                st.success(f"✅ Dataset split completed ({100 - test_size}% train / {test_size}% test)")

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.subheader("🧾 X_train Preview")
                st.dataframe(X_train.head())
                
                st.subheader("🧾 X_test Preview")
                st.dataframe(X_test.head())
                
                st.subheader("🎯 y_train Preview")
                st.dataframe(y_train.head())
                
                st.subheader("🎯 y_test Preview")
                st.dataframe(y_test.head())


            except Exception as e:
                st.warning(f"⚠️ Error during split: {str(e)}")

    # Model Building Section
    st.markdown("---")
    st.subheader("📊 Model Selection")

    # Check if train-test has been done
    if 'y_train' not in st.session_state:
        st.warning("⚠️ Please perform Train-Test Split before building the model.")
    else:
        y_train = st.session_state.y_train
        try:
            from xgboost import XGBClassifier, XGBRegressor
            xgboost_available = True
        except ImportError:
            xgboost_available = False

        # 1. Define models
        classification_models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine (SVM)": SVC(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

        regression_models = {
            "Linear Regression": LinearRegression(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor (SVR)": SVR(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
        }

        if xgboost_available:
            classification_models["XGBoost"] = XGBClassifier()
            regression_models["XGBoost Regressor"] = XGBRegressor()

        # 2. Infer task from y_train
        if y_train.dtype == 'object' or y_train.nunique() < 10:
            inferred_task = 'Classification'
        else:
            inferred_task = 'Regression'

        # 3. Let user choose category
        model_type_choice = st.radio("Select the type of model you want to build:", ["Classification", "Regression"])

        # 4. Dropdown based on selected category
        if model_type_choice == "Classification":
            model_name = st.selectbox("Select a classification model:", list(classification_models.keys()))
        else:
            model_name = st.selectbox("Select a regression model:", list(regression_models.keys()))

        # 5. Validate model type vs inferred task
        if st.button("Build Model"):
            if "X_train" not in st.session_state or "y_train" not in st.session_state:
                st.error("🚫 Please perform Train-Test Split before building the model.")
            else:
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train

                if inferred_task != model_type_choice:
                    st.error(f"❌ The selected model type '{model_type_choice}' does not match the target column type ({inferred_task}). Please choose a {inferred_task} model.")
                else:
                    selected_model = classification_models[model_name] if model_type_choice == "Classification" else regression_models[model_name]

                    try:
                        selected_model.fit(X_train, y_train)
                        st.success(f"✅ Model '{model_name}' has been successfully trained!")
                        
                        # Save the trained model and type to session state
                        st.session_state["selected_model"] = selected_model
                        st.session_state["model_type"] = model_type_choice
                        st.session_state["model_name"] = model_name

                    except Exception as e:
                        st.error(f"🚫 Error during model training: {str(e)}")

    # ---------------------- MODEL EVALUATION SECTION -----------------------
    st.markdown("---")
    st.subheader("📈 Model Evaluation")

    if "selected_model" in st.session_state and "X_test" in st.session_state and "y_test" in st.session_state:
        model = st.session_state["selected_model"]
        model_type = st.session_state["model_type"]

        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        try:
            # Make predictions
            y_pred = model.predict(X_test)
            report_text = ""
            if model_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.write("📌 **Metrics**")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write(f"**Precision:** {precision:.4f}")
                st.write(f"**Recall:** {recall:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")

                # Confusion matrix
                st.write("📊## Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                report_text = classification_report(y_test, y_pred)

                # Classification Report
                from sklearn.metrics import classification_report
                report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                st.write("### Classification Report")
                st.dataframe(report)

                # ROC Curve (only for binary)
                if len(np.unique(y_test)) == 2:
                    from sklearn.metrics import roc_curve, auc

                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)

                    st.write("### ROC Curve")
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax_roc.plot([0, 1], [0, 1], linestyle="--")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
                    ax_roc.legend()
                    st.pyplot(fig_roc)

            elif model_type == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.write("📌 **Metrics**")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
                st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")

                residuals = y_test - y_pred
                col1, col2 = st.columns(2)

                # 1. Actual vs Predicted
                with col1:
                    st.markdown("**Actual vs Predicted**")
                    fig1, ax1 = plt.subplots()
                    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
                    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax1.set_xlabel("Actual")
                    ax1.set_ylabel("Predicted")
                    ax1.set_title("Actual vs Predicted")
                    st.pyplot(fig1)

                # 2. Residuals vs Predicted
                with col2:
                    st.markdown("**Residuals vs Predicted**")
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
                    ax2.axhline(0, color='red', linestyle='--')
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Residuals")
                    ax2.set_title("Residuals vs Predicted")
                    st.pyplot(fig2)

                # 3. Distribution of Residuals (below)
                st.markdown("**Distribution of Residuals**")
                fig3, ax3 = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax3)
                ax3.set_xlabel("Residuals")
                ax3.set_title("Distribution of Residuals")
                st.pyplot(fig3)

                report_text = f"""
                Mean Absolute Error: {mae:.4f}
                Mean Squared Error: {mse:.4f}
                Root Mean Squared Error: {rmse:.4f}
                R2 Score: {r2:.4f}
                """

            model_buffer = io.BytesIO()
            pickle.dump(st.session_state["selected_model"], model_buffer)

            # Create 2 columns
            col1, col2 = st.columns([1,1],gap="small")

            # Left column: Download model
            with col1:
                st.download_button(
                    label="💾 Download Trained Model",
                    data=model_buffer.getvalue(),
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )

            # Right column: Download evaluation report
            with col2:
                st.download_button(
                    label="📥 Download Evaluation Report",
                    data=report_text,  # This should be a string (your evaluation summary)
                    file_name="model_evaluation_report.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
    else:
        st.warning("⚠️ Please build a model and perform train-test split first.")

    st.markdown("---")
    st.header("🧪 Prediction Interface")

    if "selected_model" in st.session_state and "X_train" in st.session_state:
        model = st.session_state.selected_model
        X_train = st.session_state.X_train

        pred_option = st.radio("Choose Prediction Input Mode", ["Upload CSV", "Manual Input"])

        if pred_option == "Upload CSV":
            pred_file = st.file_uploader("Upload new data for prediction", type=["csv"], key="pred_upload")
            if pred_file is not None:
                try:
                    input_df = pd.read_csv(pred_file)
                    st.write("📄 Uploaded Data Preview:")
                    st.dataframe(input_df.head())

                    pred_result = model.predict(input_df)
                    st.subheader("🔮 Predictions")
                    st.write(pred_result)

                    if st.session_state["model_type"] == "Classification" and hasattr(model, "predict_proba"):
                        st.subheader("📊 Prediction Probabilities")
                        proba = model.predict_proba(input_df)
                        st.dataframe(pd.DataFrame(proba, columns=[f"Class {i}" for i in range(proba.shape[1])]))

                except Exception as e:
                    st.error(f"❌ Error in prediction: {str(e)}")

        elif pred_option == "Manual Input":
            st.subheader("📝 Enter input values")
            input_data = []
            for col in X_train.columns:
                val = st.text_input(f"{col}", value=str(X_train[col].iloc[0]))
                input_data.append(val)

            if st.button("🔮 Predict"):
                try:
                    input_array = np.array(input_data).reshape(1, -1).astype(float)
                    prediction = model.predict(input_array)
                    st.success(f"🎯 Prediction: {prediction[0]}")

                    if st.session_state["model_type"] == "Classification" and hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_array)
                        st.write("📊 Prediction Probabilities:")
                        st.dataframe(pd.DataFrame(proba, columns=[f"Class {i}" for i in range(proba.shape[1])]))

                except Exception as e:
                    st.error(f"⚠️ Manual input prediction error: {str(e)}")
    else:
        st.info("ℹ️ Train a model first to use prediction interface.")


else:
    st.info("📌 Please upload a CSV file to get started.")
