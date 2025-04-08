# 🧠 Mental Health Prediction App

A Streamlit web application that predicts the likelihood of mental health issues (specifically **depression**) using various machine learning models. This tool is built for educational and research purposes, enabling quick model comparison, visualization, and reporting from CSV datasets.

---

## 🚀 Features

- 📁 Upload your own CSV dataset
- 🧼 Auto preprocessing (One-Hot Encoding, Standardization)
- ⚙️ Sidebar to choose:
  - Models to run (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
  - Test size split
  - Detailed report visibility
- 📊 Model performance visualization using accuracy bar chart
- 🧾 Classification report and confusion matrix per model
- 📥 Downloadable CSV report for all model accuracies

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Language**: Python 3.x

---

## 📂 Dataset Requirements

Your uploaded CSV must contain at least the following:

| Column Name     | Type         | Description                         |
|------------------|--------------|-------------------------------------|
| `Depression`     | Binary (0/1) | Target column (Depressed or not)    |
| `Gender`         | Categorical  | Gender of the individual            |
| `Course`         | Categorical  | Course enrolled (optional)          |
| `YearOfStudy`    | Categorical  | Academic year (optional)            |
| Other features   | Numerical    | Any relevant numeric features       |

---

## 💡 How to Use

1. Clone the repository
```bash
git clone https://github.com/<your-username>/mental-health-prediction.git
cd mental-health-prediction

Install the dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run streamlit_mental_health_ui.py
Open in browser and upload your dataset!

📁 Files Structure
bash
Copy
Edit
mental-health-prediction/
│
├── streamlit_mental_health_ui.py   # Main app
├── mentalhealth_dataset.csv        # Sample dataset (optional)
├── requirements.txt                # Required packages
├── screenshot.png                  # UI image 
└── README.md
📊 Sample Output
python-repl
Copy
Edit
Model: Random Forest
Accuracy: 0.87
Classification Report:
...
Confusion Matrix:
...
📌 Future Scope
Add prediction from user input form

Export trained model

Model explainability using SHAP

Deployment on Streamlit Cloud

🙌 Acknowledgments
Inspired by real-world mental health prediction research

Built with ❤️ using Python and Streamlit