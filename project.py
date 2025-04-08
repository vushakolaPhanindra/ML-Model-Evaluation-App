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

st.set_page_config(page_title="Mental Health Prediction", layout="wide")
st.title("🧠 Mental Health Prediction App")
st.write("Upload your dataset and analyze how different machine learning models perform in predicting mental health conditions.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

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

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Overview")
    st.write(data.head())

    with st.expander("📊 Dataset Summary"):
        st.write(data.describe(include='all'))

    # Encoding categorical columns
    if {'Gender', 'Course', 'YearOfStudy'}.issubset(data.columns):
        data = pd.get_dummies(data, columns=['Gender', 'Course', 'YearOfStudy'], drop_first=True)

    if 'Depression' not in data.columns:
        st.error("⚠️ 'Depression' column not found in the dataset.")
    else:
        X = data.drop(columns=['Depression'])
        y = data['Depression']

        X = X.select_dtypes(include=[np.number])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        results = []

        st.subheader("📈 Model Results")

        for name in selected_models:
            model = model_options[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results.append((name, accuracy))

            with st.expander(f"🧪 {name} - Accuracy: {accuracy:.2f}"):
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

        # Compare model performances
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

        st.subheader("🏆 Model Accuracy Comparison")
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
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="model_accuracy_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a CSV file to begin.")
