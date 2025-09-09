import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Try to import shared paths. If that fails, use local fallbacks.
try:
    from src.config import (
        ROOT, READY_PARQUET, PROCESSED_PARQUET, RAW_CSV,
        MODEL_FILE, FEATURES_FILE
    )
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    READY_PARQUET     = ROOT / "data" / "processed" / "hr_attrition_ready.parquet"
    PROCESSED_PARQUET = ROOT / "data" / "processed" / "hr_attrition.parquet"
    RAW_CSV           = ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    MODEL_FILE        = ROOT / "artifacts" / "v1" / "rf_pipeline.joblib"
    FEATURES_FILE     = ROOT / "artifacts" / "v1" / "features.json"
    
# ---- Cache helpers ----
@st.cache_resource
def _load_artifacts():
    """Load the trained model and the feature list once."""
    pipe = joblib.load(MODEL_FILE)
    feats = json.loads(Path(FEATURES_FILE).read_text())
    return pipe, feats

@st.cache_data
def _load_dataset():
    """Load a dataset (prefer ready → processed → raw)."""
    for p in (READY_PARQUET, PROCESSED_PARQUET, RAW_CSV):
        if Path(p).exists():
            if Path(p).suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            # Make sure we have a 0/1 target
            if "target" not in df.columns and "Attrition" in df.columns:
                df = df.copy()
                df["target"] = df["Attrition"].map({"Yes": 1, "No": 0})
            return df, Path(p)
    return None, None

def _band(prob: float):
    """Turn a probability into a simple risk label and icon."""
    if prob < 0.35:
        return "Low", "✅"
    elif prob < 0.60:
        return "Medium", "⚠️"
    else:
        return "High", "🔴"

def run():
    st.title("Attrition Predictor (ML)")
    # 1) Load model + feature names
    try:
        pipe, feats = _load_artifacts()
    except Exception as e:
        st.error("Model not found yet. Train & export via Notebook 03 before using this page.")
        st.caption(f"Expected: {MODEL_FILE.relative_to(ROOT)} and {FEATURES_FILE.relative_to(ROOT)}")
        st.caption(f"Error: {e}")
        return

    # 2) Load data (used to build sensible input widgets)
    df, src = _load_dataset()
    if df is None:
        st.warning("No data found in data/processed or data/raw. Run Notebook 01–02.")
        return
    st.caption(f"Using dataset: `{src.relative_to(ROOT)}`")


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