import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "data/processed/hr_attrition_ready.parquet"
PROCESSED = ROOT / "data/processed/hr_attrition.parquet"

def _load_df():
    """Load the processed dataset from an absolute path."""
    for p in (READY, PROCESSED):
        if p.exists():
            return pd.read_parquet(p)
    return None

def run():
    st.title("Project Hypotheses & Validation")
    st.markdown("""
    **H1:** Overtime workers leave more.  
    **H2:** Lower job satisfaction increases attrition.  
    **H3:** Younger employees (≤30) leave more often.
    """)

    path_opts = ["data/processed/hr_attrition_ready.parquet", "data/processed/hr_attrition.parquet"]
    df = None
    for p in path_opts:
        try:
            df = pd.read_parquet(p)
            break
        except Exception:
            continue
    if df is None:
        st.warning("Processed data not found. Run Notebook 02.")
        return

    # Ensure target exists
    if "target" not in df.columns and "Attrition" in df.columns:
        df["target"] = df["Attrition"].map({"Yes":1,"No":0})

    
    # H1 details (rate + count + bar)
    h1 = (df.groupby("OverTime")["target"]
        .agg(rate="mean", n="size")
        .reset_index())
    h1["rate_pct"] = (100*h1["rate"]).round(1)
    st.dataframe(h1, use_container_width=True)
    st.bar_chart(h1.set_index("OverTime")["rate"])

    # H2 details (if present)
    if "JobSatisfaction" in df.columns:
        h2 = (df.groupby("JobSatisfaction")["target"]
            .agg(rate="mean", n="size")
            .reset_index()
            .sort_values("JobSatisfaction"))
        h2["rate_pct"] = (100*h2["rate"]).round(1)
        st.dataframe(h2, use_container_width=True)
        st.bar_chart(h2.set_index("JobSatisfaction")["rate"])

    # H3 details (age groups)
    age_group = (df["Age"] <= 30).map({True:"<=30", False:">30"})
    h3 = (df.assign(AgeGroup=age_group).groupby("AgeGroup")["target"]
        .agg(rate="mean", n="size")
        .reset_index()
        .sort_values("AgeGroup"))
    h3["rate_pct"] = (100*h3["rate"]).round(1)
    st.dataframe(h3, use_container_width=True)
    st.bar_chart(h3.set_index("AgeGroup")["rate"])