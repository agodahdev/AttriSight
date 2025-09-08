import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

MODEL_PATH = Path("artifacts/v1/rf_pipeline.joblib")
FEATS_PATH = Path("artifacts/v1/features.json")

def load():
    try:
        return joblib.load(MODEL_PATH), json.loads(FEATS_PATH.read_text())
    except Exception:
        return None, None


def run():
    st.title("Attrition Predictor (ML)")
    st.warning("Model not found yet. Train & export via Notebook 03 before using this page.")

    # inputs
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("MonthlyIncome", 1000, 25000, 5000, step=500)
    dist = st.number_input("DistanceFromHome", 0, 50, 5)
    twy = st.number_input("TotalWorkingYears", 0, 40, 5)
    yac = st.number_input("YearsAtCompany", 0, 40, 3)
    ncomp = st.number_input("NumCompaniesWorked", 0, 20, 2)
    hike = st.number_input("PercentSalaryHike", 0, 100, 15)
    overtime = st.selectbox("OverTime", ["Yes","No"])
    jobrole = st.selectbox("JobRole", ["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"])
    mstat = st.selectbox("MaritalStatus", ["Single","Married","Divorced"])
    btravel = st.selectbox("BusinessTravel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
    dept = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
    efield = st.selectbox("EducationField", ["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"])
    gender = st.selectbox("Gender", ["Male","Female"])
    jlevel = st.selectbox("JobLevel", [1,2,3,4,5])

    row = {"Age":age,"MonthlyIncome":income,"DistanceFromHome":dist,
           "TotalWorkingYears":twy,"YearsAtCompany":yac,"NumCompaniesWorked":ncomp,
           "PercentSalaryHike":hike,"OverTime":overtime,"JobRole":jobrole,
           "MaritalStatus":mstat,"BusinessTravel":btravel,"Department":dept,
           "EducationField":efield,"Gender":gender,"JobLevel":jlevel}
        
    X = pd.DataFrame([row])[feats]
    if st.button("Predict"):
        prob = pipe.predict_proba(X)[0,1]
        band = "High risk" if prob >= 0.60 else "Medium risk" if prob >= 0.35 else "Low risk"
        st.metric("Attrition probability", f"{prob:.2%}")
        st.success(f"Risk category: **{band}**")