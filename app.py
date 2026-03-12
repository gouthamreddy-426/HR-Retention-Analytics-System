import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title="HR Retention Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("HR Retention Analytics Dashboard")

# ---------------------------
# Employee Prediction Section (TOP)
# ---------------------------

st.header("Employee Attrition Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    Work_accident = st.selectbox("Work Accident", [0, 1])

with col2:
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6)
    average_montly_hours = st.number_input("Average Monthly Hours", 80, 350, 160)
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])

with col3:
    time_spend_company = st.number_input("Years at Company", 1, 15, 3)

    Department = st.selectbox(
        "Department",
        ["sales","technical","support","IT","product_mng",
         "marketing","RandD","accounting","hr","management"]
    )

    salary = st.selectbox("Salary Level", ["low","medium","high"])

if st.button("Predict Attrition Risk"):

    input_data = CustomData(
        satisfaction_level,
        last_evaluation,
        number_project,
        average_montly_hours,
        time_spend_company,
        Work_accident,
        promotion_last_5years,
        Department,
        salary
    )

    pred_df = input_data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()

    result, probability = predict_pipeline.predict(pred_df)

    risk = probability[0][1] * 100

    st.subheader("Prediction Result")

    st.write(f"Attrition Risk Score: **{risk:.2f}%**")

    if result[0] == 1:
        st.error("Employee is likely to leave the company.")
    else:
        st.success("Employee is likely to stay in the company.")

st.divider()

# ---------------------------
# Load Dataset
# ---------------------------

hr_data = pd.read_csv("notebook/data/HR.csv")

# ---------------------------
# HR Overview Metrics
# ---------------------------

st.header("HR Analytics Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(hr_data))
col2.metric("Employees Left", int(hr_data["left"].sum()))
col3.metric("Attrition Rate", f"{(hr_data['left'].mean()*100):.2f}%")

st.divider()

# ---------------------------
# Salary Attrition Chart
# ---------------------------

st.subheader("Attrition by Salary Level")

salary_chart = hr_data.groupby("salary")["left"].mean()

st.bar_chart(salary_chart)

# ---------------------------
# Smaller Heatmap
# ---------------------------

st.subheader("Attrition Heatmap (Department vs Salary)")

pivot_table = hr_data.pivot_table(
    values="left",
    index="Department",
    columns="salary",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(6,4))  # smaller size
sns.heatmap(pivot_table, annot=True, cmap="Reds", ax=ax)

st.pyplot(fig)

st.divider()

# ---------------------------
# Feature Importance
# ---------------------------

st.subheader("Top Factors Influencing Attrition")

model = joblib.load("artifacts/model.pkl")

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    features = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company"
    ]

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance[:len(features)]
    })

    st.bar_chart(df.set_index("Feature"))