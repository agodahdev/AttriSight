import streamlit as st
import pandas as pd
import numpy as np
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
    st.markdown("### Business Requirement #2: Predict Employee Attrition")
    st.info("""
    **Purpose:** This page allows HR to enter an employee profile and receive a prediction 
    of their attrition risk. This directly addresses **BR#2** - providing ML-based predictions 
    for proactive retention planning.
    """)
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

    # Check that all training features exist in the current data
    missing = [c for c in feats if c not in df.columns]
    if missing:
        st.error(f"Missing columns in data: {missing}")
        st.caption("Recreate the ready parquet in Notebook 02 so it matches the training features.")
        return
    # 3) Simple form: one input per feature (numbers → number input, text/categorical → dropdown)
    st.subheader("Enter an employee profile")
    st.caption("Dropdowns show values seen in the dataset.")
    user_vals = {}

    with st.form("ml_form", clear_on_submit=False):
        cols = st.columns(2)  # nicer layout
        for i, feat in enumerate(feats):
            col = cols[i % 2]
            ser = df[feat]

            if pd.api.types.is_numeric_dtype(ser):
                # Basic numeric bounds and default
                vmin = float(np.nanmin(ser)) if ser.notna().any() else 0.0
                vmax = float(np.nanmax(ser)) if ser.notna().any() else 100.0
                vdef = float(np.nanmedian(ser)) if ser.notna().any() else 0.0
                step = 1.0 if ser.dtype.kind in "iu" else 0.1
                with col:
                    user_vals[feat] = st.number_input(
                        feat, min_value=vmin, max_value=vmax, value=vdef, step=step,
                        key=f"in_num_{feat}"
                    )
            else:
                # Categorical options from the data
                opts = sorted([str(x) for x in ser.dropna().unique().tolist()])
                default_idx = 0 if opts else None
                with col:
                    user_vals[feat] = st.selectbox(
                        feat, options=opts, index=default_idx if opts else None,
                        key=f"in_cat_{feat}"
                    )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        st.info("Fill the form and click **Predict**.")
        
    
    # 4) Predict
    X = pd.DataFrame([user_vals])[feats]  # keep exact training feature order
    try:
        prob = float(pipe.predict_proba(X)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # 5) Show result
    band, icon = _band(prob)
    st.subheader("Result")
    st.metric(label="Attrition probability", value=f"{prob:.2%}")
    st.markdown(f"**Risk band:** {icon} **{band}**")
    st.caption("Note: thresholds are examples (Low < 0.35, Medium 0.35–0.59, High ≥ 0.60). Adjust with stakeholders.")
    
    