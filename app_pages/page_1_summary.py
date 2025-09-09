import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import ROOT, RAW_CSV, PROCESSED_PARQUET, READY_PARQUET

def _load_preview():
    """
    Try these files in order and return a small preview:
    1) ready parquet (best for the app)
    2) processed parquet
    3) raw CSV
    """
    for p in (READY_PARQUET, PROCESSED_PARQUET, RAW_CSV):
        try:
            if p.exists():
                if p.suffix == ".parquet":
                    return pd.read_parquet(p), p
                else:
                    return pd.read_csv(p), p
        except Exception:
            # If one file fails to load, try the next
            continue
    return None, None


def run():
    st.title("Project Summary")
    st.markdown("""
    **Goal:** show what drives employee attrition (charts) and **predict** attrition risk (ML).  
    **Audience:** HR / People Analytics.
    """)

    st.markdown("### Business Requirements")
    st.write("""
    **Client requirements**
     - **BR#1 (Analysis):** Simple charts to explain what affects attrition.  
     - **BR#2 (ML):** Predict the probability an employee will leave.
    """)
    
    # Quick dataset glance (first 5 rows)
    df, src = _load_preview()
    st.markdown("### Dataset at a glance")
    if df is not None:
        rel = Path(src).relative_to(ROOT)
        st.caption(f"Loaded from: `{rel}` • Rows: {len(df):,} • Columns: {len(df.columns)}")
        st.dataframe(df.head(), use_container_width=True)
        st.caption("Tip: see **Workforce Analysis** for charts, and **Attrition Predictor (ML)** for predictions.")
    else:
        st.info("No dataset found yet. Run Notebook 01–02 or place the CSV in `data/raw/`.")

    #Simple site map so users know where to go
    st.markdown("### Navigation")
    st.markdown(""" 
    - **Project Summary** (this page)
    - **Workforce Analysis** — filters, category plots, box plot, correlation heatmap 
    - **Project Hypotheses** — H1–H3 with simple validation
    - **Attrition Predictor (ML)** — form ➜ probability + risk band  
    - **Technical: Model & Evaluation** — ROC/CM, threshold table, live CM, pipeline steps
    """)