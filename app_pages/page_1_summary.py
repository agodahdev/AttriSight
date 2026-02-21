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
            continue
    return None, None


def run():
    st.title("Project Summary")
    st.markdown("""
    **Goal:** Show what drives employee attrition (charts) and **predict**
    attrition risk (ML).  
    **Audience:** HR / People Analytics.
    """)

    # -- Business Requirements --
    st.markdown("### Business Requirements")
    st.markdown("""
    - **BR#1 (Analysis):** Simple charts to explain what affects attrition.  
    - **BR#2 (ML):** Predict the probability an employee will leave.
    """)

    # -- Dataset overview --
    df, src = _load_preview()
    st.markdown("### Dataset at a Glance")

    if df is not None:
        rel = Path(src).relative_to(ROOT)
        st.caption(
            f"Loaded from: `{rel}` · Rows: {len(df):,} · Columns: {len(df.columns)}"
        )
        st.dataframe(df.head(), use_container_width=True)

      
        # Dataset and split details 
        st.markdown("### Dataset & Train/Test Split")
        st.markdown(f"""
        - **Source:** IBM HR Analytics Employee Attrition & Performance (Kaggle)
        - **Total samples:** {len(df):,} employee records
        - **Features used:** 7 numeric + 8 categorical = **15 features**
        - **Target:** Attrition (Yes/No → 1/0)
        - **Training set:** 1,176 samples (80%) — used to train the model
        - **Test set:** 294 samples (20%) — used to evaluate performance
        - **Split strategy:** Stratified by target to keep the ~16% attrition
          rate balanced in both sets
        """)

        st.caption(
            "Tip: see **Workforce Analysis** for charts, "
            "and **Attrition Predictor (ML)** for predictions."
        )
    else:
        st.info("No dataset found yet. Run Notebook 01–02 or "
                "place the CSV in `data/raw/`.")

    # -- Navigation / site map --
    st.markdown("### Navigation")
    st.markdown("""
    - **Project Summary** (this page)
    - **Workforce Analysis** — filters, category plots, box plot, 
      correlation heatmap, interactive sunburst
    - **Project Hypotheses** — H1–H3 with data, charts, and clear verdicts
    - **Attrition Predictor (ML)** — form → probability + risk band
    - **Technical: Model & Evaluation** — ROC-AUC vs goal, metrics,
      actual vs predicted, threshold table, live CM slider, pipeline steps
    """)
