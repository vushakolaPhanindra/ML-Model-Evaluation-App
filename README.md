General ML Model Evaluation App
This project is a Streamlit web application that allows users to upload any CSV dataset and quickly evaluate the performance of several popular machine learning models. The app is designed to help anyone, even without coding experience, understand how different models perform on their data by providing visual reports, feature analysis, and comparison charts.

About the project
Once a user uploads a dataset, the app guides them to select the target column they want to predict. It automatically handles encoding of categorical variables and scales the data to prepare it for modeling. The application supports several classifiers, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K‑Nearest Neighbors (KNN).

For each selected model, the app displays detailed results including accuracy, a confusion matrix, and a classification report. Additionally, for tree‑based models, it shows the feature importance scores so users can understand which features have the most impact on the model’s predictions. Finally, the app compares all selected models in a single chart and allows users to download the results as a CSV file.

Features
Upload any CSV file, with support for larger files up to about 1.2 GB if the server is configured correctly

Automatically detect and encode categorical columns so the data can be used for modeling

Choose the target column dynamically based on the uploaded dataset

Run multiple machine learning models and view their accuracy

See detailed classification reports and confusion matrices for each model

Analyze which features are most important for tree‑based models

Compare the performance of models side by side using visual charts

Download the final accuracy comparison as a CSV file for future reference

Installation
Before running the app, make sure you have Python 3.8 or higher installed on your machine. Then, install the required Python libraries using pip. Open a terminal or command prompt and run the following command:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn

Configuring large file uploads
By default, Streamlit limits file uploads to about 200 MB. If you plan to upload larger datasets, you can increase this limit. In your project directory, create a folder named .streamlit and inside it create a file called config.toml with the following content:

[server]
maxUploadSize = 1200

This will increase the maximum upload size to around 1.2 GB. Note that uploading very large files also depends on your computer’s available memory.

Running the app
After setting everything up, navigate to your project directory in the terminal and run:

streamlit run project.py

This will start the Streamlit app and open it in your default web browser. From there, you can upload a CSV file and begin exploring the models.

How to use the app
Upload your dataset as a CSV file. After uploading, select the target column that you want to predict. Adjust test size and choose which machine learning models you want to run from the sidebar options. View the results, including accuracy scores, detailed reports, and feature importance for tree models. Compare models in a visual chart and download the summary of results as a CSV file.

License
This project is open source and available under the Vushakola Phanindra License, so you are free to use, modify, and share it.

Author
This app was developed to make machine learning more accessible by providing an interactive, no-code interface for testing and comparing popular classification models on any dataset.

The idea is to help students, data enthusiasts, and professionals quickly explore how different models perform, understand feature importance, and visualize results without writing a single line of code.

Built using Python, Streamlit, and popular machine learning libraries like scikit-learn, it aims to turn raw data into insight in just a few clicks.

If you have suggestions, find issues, or want to contribute new features, your feedback is welcome. Feel free to fork the repository, open pull requests, or reach out.

Developed by Phanindra V in 2025.

