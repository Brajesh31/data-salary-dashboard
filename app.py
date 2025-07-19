import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


@st.cache_resource
def train_model():
    df = pd.read_csv("ds_salaries.csv")
    
    X = df[["job_title", "experience_level", "employment_type", "company_location", "remote_ratio"]]
    y = df["salary_in_usd"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["job_title", "experience_level", "employment_type", "company_location"])
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model

# Load the model
model = train_model()

# Streamlit UI
st.title("💼 Data Job Salary Estimator")

job_title = st.selectbox("Job Title", ["Data Scientist", "Data Analyst", "ML Engineer", "Data Engineer"])
experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
company_location = st.selectbox("Company Location", ["US", "GB", "IN", "CA", "DE"])
remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 50)

input_df = pd.DataFrame({
    "job_title": [job_title],
    "experience_level": [experience_level],
    "employment_type": [employment_type],
    "company_location": [company_location],
    "remote_ratio": [remote_ratio]
})

st.subheader("🔍 Input Summary")
st.dataframe(input_df)

# Predict
salary_pred = model.predict(input_df)[0]
st.subheader(f"💰 Predicted Salary: ${salary_pred:,.0f}")
