import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="General ML Model Evaluation App", layout="wide")
st.title("General ML Model Evaluation App")
st.write("Upload any CSV dataset (up to 100 MB) and explore how different machine learning models perform in predicting your target variable.")

# Sidebar settings
st.sidebar.header(" Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100
model_options = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}
selected_models = st.sidebar.multiselect("Select Models to Run", list(model_options.keys()), default=list(model_options.keys()))
show_detailed = st.sidebar.checkbox("Show Detailed Reports", value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance (Tree Models)", value=True)

uploaded_file = st.file_uploader(
    "Upload CSV file (up to 100 MB supported)",
    type=["csv"]
)

if uploaded_file is not None:
    # Check file size (100 MB = 100 * 1024 * 1024 bytes)
    uploaded_file.seek(0, 2)  # move to end of file
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)     # reset to start
    if file_size > 100 * 1024 * 1024:
        st.error("Uploaded file is too large. Please upload a file up to 100 MB.")
    else:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Overview")
        st.write(data.head())
        with st.expander("Dataset Summary"):
            st.write(data.describe(include='all'))
        # Ask user to select target column dynamically
        target_column = st.selectbox(" Select Target Column", options=data.columns)
        # Encoding categorical columns automatically
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        if target_column not in data.columns:
            st.error(f" Selected target column '{target_column}' not found in the dataset after encoding.")
        else:
            X = data.drop(columns=[target_column])
            y = data[target_column]

            X = X.select_dtypes(include=[np.number])
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            results = []
            st.subheader(" Model Results")
            for name in selected_models:
                model = model_options[name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append((name, accuracy))
                with st.expander(f" {name} - Accuracy: {accuracy:.2f}"):
                    if show_detailed:
                        st.text("Classification Report")
                        st.text(classification_report(y_test, y_pred, zero_division=0))
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
                        ax.set_title(f"Confusion Matrix: {name}")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
                    # Show feature importance for tree-based models
                    if show_feature_importance and hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        feature_names = X.columns
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)
                        st.subheader(f"🔎 Feature Importance for {name}")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
                        ax.set_title(f"Top Features Impacting {name}")
                        st.pyplot(fig)

            # Compare model performances
            results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

            st.subheader(" Model Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette("crest", len(results_df))
            ax.barh(results_df["Model"], results_df["Accuracy"], color=colors)
            ax.set_xlabel("Accuracy")
            ax.set_title("Model Performance")
            st.pyplot(fig)

            # Download results
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=" Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="model_accuracy_results.csv",
                mime="text/csv"
            )
else:
    st.info(" Upload a CSV file to begin.")
