import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Import from src modules for consistency
from src.config import ROOT, READY_PARQUET, PROCESSED_PARQUET, RAW_CSV
from src.features import NUM_FEATURES, CAT_FEATURES

# Use paths from config
READY = READY_PARQUET
PROCESSED = PROCESSED_PARQUET

# Use features from src.features
SUGGESTED_CAT = CAT_FEATURES
SUGGESTED_NUM = NUM_FEATURES

def _load_df():
    """Load dataset from preferred locations; return (DataFrame, path) or (None, None)."""
    for p in (READY, PROCESSED, RAW_CSV):
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p), p
            return pd.read_csv(p), p
    return None, None

def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target column if missing and 'Attrition' exists."""
    if "target" not in df.columns and "Attrition" in df.columns:
        df = df.copy()
        df["target"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df

def run():
    st.title("Workforce Analysis (Conventional)")
    df, src = _load_df()
    if df is None:
        st.warning("No data found in data/processed/ or data/raw/. Run Notebook 01–02.")
        return

    df = _ensure_target(df)
    st.caption(f"Loaded from: `{src.relative_to(ROOT)}` • Rows: {len(df):,} • Columns: {len(df.columns)}")

    
    # Simple Filters (affect charts)
    with st.expander("Filters", expanded=False):
        dept_opts = df["Department"].dropna().unique().tolist() if "Department" in df.columns else []
        ot_opts   = df["OverTime"].dropna().unique().tolist()   if "OverTime"   in df.columns else []
        min_age   = int(df["Age"].min()) if "Age" in df.columns else 18
        max_age   = int(df["Age"].max()) if "Age" in df.columns else 70

        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            sel_dept = st.multiselect(
                "Department", options=dept_opts,
                default=dept_opts[:2] if dept_opts else [],
                key="filter_dept"
            )
        with colf2:
            sel_ot = st.multiselect(
                "OverTime", options=ot_opts,
                default=ot_opts if ot_opts else [],
                key="filter_ot"
            )
        with colf3:
            age_range = st.slider(
                "Age range", min_age, max_age, (min_age, max_age),
                key="filter_age_range"
            )

    # apply filters if columns available
    mask = pd.Series(True, index=df.index)
    if "Department" in df.columns and sel_dept:
        mask &= df["Department"].isin(sel_dept)
    if "OverTime" in df.columns and sel_ot:
        mask &= df["OverTime"].isin(sel_ot)
    if "Age" in df.columns:
        mask &= df["Age"].between(age_range[0], age_range[1])

    dff = df[mask].copy()
    st.caption(f"Filtered rows: {len(dff):,}")

    
    # Categorical comparison (histogram)
    st.subheader("Compare attrition by category")
    cat_choices = [c for c in SUGGESTED_CAT if c in dff.columns]
    if not cat_choices:
        st.info("No suggested categorical columns found in data.")
    else:
        cat = st.selectbox("Categorical", cat_choices, index=0, key="cat_select")
        hue = "Attrition" if "Attrition" in dff.columns else "target"
        fig_cat = px.histogram(dff, x=cat, color=hue, barmode="group")
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption("Bars higher for 'Attrition=Yes' suggest a stronger link with leaving.")

    # Numeric distribution (box plot)

    st.subheader("Numeric distribution by attrition")
    num_choices = [c for c in SUGGESTED_NUM if c in dff.columns]
    if not num_choices:
        st.info("No suggested numeric columns found in data.")
    else:
        num = st.selectbox("Numeric", num_choices, index=0, key="num_select")
        hue = "Attrition" if "Attrition" in dff.columns else "target"
        fig_box = px.box(dff, x=hue, y=num, points="all")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("If box plots differ a lot between Yes/No, that feature may matter.")

   
    # Correlation heatmap (numeric + 'target')
    st.divider()
    st.subheader("Correlation (numeric features)")
    if "target" not in dff.columns and "Attrition" in dff.columns:
        dff["target"] = dff["Attrition"].map({"Yes": 1, "No": 0})

    num_cols = [c for c in SUGGESTED_NUM if c in dff.columns]
    cols_for_corr = [*num_cols, "target"] if "target" in dff.columns else num_cols
    if len(cols_for_corr) >= 2:
        corr = dff[cols_for_corr].corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Stronger absolute correlation with 'target' can indicate predictive power (watch for leakage).")
    else:
        st.info("Not enough numeric columns found to compute a correlation heatmap.")

 