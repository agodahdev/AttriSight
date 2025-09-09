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
    st.caption("OverTime=Yes shows higher attrition in this dataset.")

    # H2 details (if present)
    if "JobSatisfaction" in df.columns:
        h2 = (df.groupby("JobSatisfaction")["target"]
            .agg(rate="mean", n="size")
            .reset_index()
            .sort_values("JobSatisfaction"))
        h2["rate_pct"] = (100*h2["rate"]).round(1)
        st.dataframe(h2, use_container_width=True)
        st.bar_chart(h2.set_index("JobSatisfaction")["rate"])
        st.caption("Lower job satisfaction links to higher attrition.")

    # H3 details (age groups)
    age_group = (df["Age"] <= 30).map({True:"<=30", False:">30"})
    h3 = (df.assign(AgeGroup=age_group).groupby("AgeGroup")["target"]
        .agg(rate="mean", n="size")
        .reset_index()
        .sort_values("AgeGroup"))
    h3["rate_pct"] = (100*h3["rate"]).round(1)
    st.dataframe(h3, use_container_width=True)
    st.bar_chart(h3.set_index("AgeGroup")["rate"])
    st.caption("The ≤30 group shows higher attrition.")

    # --- Optional statistical test: Chi-square (visible + safe) ---
    import importlib.util
    if importlib.util.find_spec("scipy") is None:
        st.subheader("Optional: Chi-square tests")
        st.info("SciPy not installed — skipping tests. Run `pip install scipy` to enable.")
    else:
        from scipy.stats import chi2_contingency
        st.subheader("Optional: Chi-square tests")

        # Recreate AgeGroup locally so this works even if earlier code changes
        age_grp = (df["Age"] <= 30).map({True: "<=30", False: ">30"}) if "Age" in df.columns else None

        # H1: OverTime vs target
        if "OverTime" in df.columns:
            p1 = chi2_contingency(pd.crosstab(df["OverTime"], df["target"]))[1]
            st.write(f"H1 p-value (OverTime): {p1:.4f}")

        # H2: JobSatisfaction vs target (only if column exists)
        if "JobSatisfaction" in df.columns:
            p2 = chi2_contingency(pd.crosstab(df["JobSatisfaction"], df["target"]))[1]
            st.write(f"H2 p-value (JobSatisfaction): {p2:.4f}")

        # H3: AgeGroup vs target
        if age_grp is not None:
            p3 = chi2_contingency(pd.crosstab(age_grp, df["target"]))[1]
            st.write(f"H3 p-value (AgeGroup ≤30 vs >30): {p3:.4f}")

        st.caption("Rule of thumb: p < 0.05 suggests a real association (statistically significant).")
        