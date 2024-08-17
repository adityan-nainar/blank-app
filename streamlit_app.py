import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

# Helper functions for model training and prediction
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

file_uploaded = False
df = None

# Sidebar for navigation
with st.sidebar:
    st.title('Steps in Running the Model')
    choice = st.radio("Navigation", ['Home Page', 'Upload File', 'Data Profiling', 'Model Building', 'Download'])
    st.info('This app will do something')

if os.path.exists("/workspaces/blank-app/dataset.csv"): 
    df = pd.read_csv('/workspaces/blank-app/dataset.csv', index_col=None)

# Main content based on navigation choice
if choice == 'Home Page':
    st.title('Auto ML Application')

if choice == "Upload File":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload your dataset")

    if file is not None:  # Corrected 'none' to 'None'
        st.write('worked!')
        df = pd.read_csv(file)  # Removed 'index_col=None'
        df.to_csv('/workspaces/blank-app/dataset.csv', index=False)  # Changed 'index=None' to 'index=False'
        file_uploaded = True  # Corrected 'true' to 'True'
        st.dataframe(df)

# if choice == "Upload":
#     st.title("Upload Your Dataset")
#     file = st.file_uploader("Upload Your Dataset")
#     if file: 
#         df = pd.read_csv(file, index_col=None)
#         df.to_csv('dataset.csv', index=None)
#         st.dataframe(df)

if choice == "Data Profiling": 
    if os.path.exists("/workspaces/blank-app/dataset.csv"):
        df = pd.read_csv('/workspaces/blank-app/dataset.csv')
        pr = df.profile_report()
        st.title("Profiling in Streamlit")
        st.write(df)
        st_profile_report(pr)
    else:
        st.write('No data')

if choice == 'Model Building':
    if os.path.exists("/workspaces/blank-app/dataset.csv"):
        st.title("Auto Machine Learning - Model Building")
        data = pd.read_csv('/workspaces/blank-app/dataset.csv', index_col=None)

        # Selecting target variable
        target = st.selectbox("Select target variable", data.columns)
        if target:
            features = [col for col in data.columns if col != target]
            st.write("Features:", features)

            # Selecting model type
            model_type = st.selectbox("Select model type", ["Classification", "Regression"])

            # Splitting data
            X = data[features]
            y = data[target]

            # Encoding for classification tasks
            if model_type == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model selection
            model_selection = st.selectbox("Select Model", [
                "Random Forest", "Logistic Regression", "Support Vector Machine", 
                "Random Forest (Regression)", "Linear Regression", "Support Vector Machine (Regression)"])

            if model_selection.startswith("Random Forest"):
                model = RandomForestClassifier() if model_type == "Classification" else RandomForestRegressor()
            elif model_selection == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_selection == "Support Vector Machine":
                model = SVC() if model_type == "Classification" else SVR()
            elif model_selection == "Linear Regression":
                model = LinearRegression()

            # Train and predict
            if st.button("Run Model"):
                trained_model = train_model(model, X_train, y_train)
                predictions = predict(trained_model, X_test)

                # Evaluate model
                if model_type == "Classification":
                    accuracy = accuracy_score(y_test, predictions)
                    st.write("Accuracy:", accuracy)
                else:
                    mse = mean_squared_error(y_test, predictions)
                    st.write("Mean Squared Error:", mse)

                # Visualizations
                st.subheader("Predictions vs Actuals")
                plt.figure(figsize=(10, 6))
                if model_type == "Classification":
                    sns.countplot(x=predictions, palette="Set2")
                    plt.title("Predictions")
                    st.pyplot(plt)
                else:
                    plt.scatter(y_test, predictions)
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title("Actual vs Predicted")
                    st.pyplot(plt)
    else:
        st.warning("Please upload a file first.")

if choice == 'Download':
    st.title("Download Page")
    st.write("This page could be used for downloading results or model outputs.")