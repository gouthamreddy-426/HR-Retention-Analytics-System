# HR Retention Analytics System

## Overview

Employee attrition is a major challenge for organizations because losing
skilled employees increases recruitment cost, reduces productivity, and
affects organizational knowledge.

This project develops an **end-to-end Machine Learning system** that
predicts whether an employee is likely to leave the company. The system
includes a **complete ML pipeline, experiment tracking using MLflow, and
an interactive analytics dashboard built with Streamlit**.

The goal is to help HR teams **identify high-risk employees early and
make data-driven retention decisions**.

------------------------------------------------------------------------

# Key Features

-   End-to-end Machine Learning pipeline\
-   Multiple model training and comparison\
-   Experiment tracking using **MLflow**\
-   Best model selection based on **F1 Score**\
-   Interactive **Streamlit dashboard**\
-   Employee attrition prediction\
-   Attrition risk probability score\
-   HR analytics visualizations\
-   Feature importance analysis

------------------------------------------------------------------------

# Machine Learning Workflow

    Dataset
       ↓
    Data Ingestion
       ↓
    Data Transformation
       ↓
    Model Training
       ↓
    MLflow Experiment Tracking
       ↓
    Model Comparison
       ↓
    Best Model Selection (Random Forest)
       ↓
    Deployment (Streamlit Dashboard)

------------------------------------------------------------------------

# Models Used

The following machine learning models were trained and evaluated:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Gradient Boosting

Using MLflow experiment tracking, **Random Forest achieved the highest
F1 Score (\~0.97)** and was selected as the final deployed model.

------------------------------------------------------------------------

# Tech Stack

### Programming Language

-   Python

### Machine Learning

-   Scikit-learn\
-   NumPy\
-   Pandas

### Experiment Tracking

-   MLflow

### Data Visualization

-   Matplotlib\
-   Seaborn

### Web Application

-   Streamlit

------------------------------------------------------------------------

# Project Structure

    HR-Retention-Analytics-System
    │
    ├── notebook
    │   └── data
    │       └── HR.csv
    │
    ├── src
    │   ├── components
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   └── model_trainer.py
    │   │
    │   ├── pipeline
    │   │   ├── train_pipeline.py
    │   │   └── predict_pipeline.py
    │   │
    │   ├── exception.py
    │   ├── logger.py
    │   └── utils.py
    │
    ├── app.py
    ├── train.py
    ├── requirements.txt
    ├── setup.py
    └── README.md

------------------------------------------------------------------------

# Installation

Clone the repository:

    git clone https://github.com/your-username/hr-retention-analytics-system.git

Navigate to the project directory:

    cd hr-retention-analytics-system

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

# Training the Model

Run the training pipeline:

    python train.py

This process will:

-   Load the dataset\
-   Perform data preprocessing\
-   Train multiple machine learning models\
-   Track experiments using MLflow\
-   Select the best performing model\
-   Save the trained model

------------------------------------------------------------------------

# MLflow Experiment Tracking

Start the MLflow UI:

    mlflow ui

Then open in your browser:

    http://localhost:5000

You can compare models using:

-   Accuracy\
-   Precision\
-   Recall\
-   F1 Score

------------------------------------------------------------------------

# Running the Application

Launch the Streamlit dashboard:

    streamlit run app.py

The application will open automatically in your browser.

------------------------------------------------------------------------

# Dashboard Features

### HR Analytics Overview

-   Total employees\
-   Employees who left\
-   Attrition rate

### Data Insights

-   Attrition by salary level\
-   Department vs salary heatmap

### Prediction System

Users can input employee details and receive:

-   Attrition prediction\
-   Attrition risk probability\
-   Feature importance insights

------------------------------------------------------------------------

# Example Use Case

HR managers can use this system to:

-   Identify employees at risk of leaving\
-   Understand factors influencing attrition\
-   Implement proactive retention strategies

------------------------------------------------------------------------

# Future Improvements

-   Deploy the application on cloud platforms\
-   Integrate real-time HR databases\
-   Add advanced models such as XGBoost\
-   Improve explainability using SHAP

------------------------------------------------------------------------

# Author

Developed as a Machine Learning project demonstrating:

-   ML pipeline design\
-   experiment tracking\
-   model comparison\
-   model deployment\
-   data analytics visualization
