import streamlit as st
import pandas as pd

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

    # H1: OverTime proportions
    st.subheader("H1: OverTime vs Attrition")
    h1 = df.groupby("OverTime")["target"].mean().rename("attrition_rate").to_frame().reset_index()
    st.dataframe(h1, use_container_width=True)

    # H2: JobSatisfaction (if available)
    st.subheader("H2: JobSatisfaction vs Attrition")
    if "JobSatisfaction" in df.columns:
        h2 = df.groupby("JobSatisfaction")["target"].mean().rename("attrition_rate").to_frame().reset_index()
        st.dataframe(h2, use_container_width=True)
    else:
        st.info("JobSatisfaction not present in dataset.")
    
    # H3: Age group
    st.subheader("H3: Age group (≤30 vs >30) vs Attrition")
    age_group = (df["Age"] <= 30).map({True:"<=30", False:">30"})
    h3 = df.assign(AgeGroup=age_group).groupby("AgeGroup")["target"].mean().rename("attrition_rate").to_frame().reset_index()
    st.dataframe(h3, use_container_width=True)
    
    st.info("Add results from notebooks here (proportions and simple stats).")