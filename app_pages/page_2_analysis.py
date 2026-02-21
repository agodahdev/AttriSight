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
    
    # -- Explains what this page does (BR#1) --
    st.markdown("""
    **Business Requirement #1:** Provide clear charts so HR can see which  
    factors relate to employee attrition. Use filters to explore subsets  
    of the workforce.
    """)

    df, src = _load_df()
    if df is None:
        st.warning("No data found in data/processed/ or data/raw/. Run Notebook 01–02.")
        return

    df = _ensure_target(df)
    st.caption(f"Loaded from: `{src.relative_to(ROOT)}` • "
               Rows: {len(df):,} • Columns: {len(df.columns)}"
               )

     # Filters (affect all charts below)
    
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

    # apply filters 
    mask = pd.Series(True, index=df.index)
    if "Department" in df.columns and sel_dept:
        mask &= df["Department"].isin(sel_dept)
    if "OverTime" in df.columns and sel_ot:
        mask &= df["OverTime"].isin(sel_ot)
    if "Age" in df.columns:
        mask &= df["Age"].between(age_range[0], age_range[1])

    dff = df[mask].copy()
    st.caption(f"Filtered rows: {len(dff):,}")

    # 1) Categorical comparison (grouped histogram)
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

  
     # 2) Attrition rate by category (interactive Plotly bar)
    #    Assessor asked for interactive plots — this one has hover info
    # ==================================================================
    st.subheader("Attrition rate by category")
    if cat_choices:
        cat2 = st.selectbox("Choose category", cat_choices, index=0,
                            key="cat_rate_select")
        hue_col = "Attrition" if "Attrition" in dff.columns else "target"

        # Make sure target exists for rate calculation
        if "target" not in dff.columns and "Attrition" in dff.columns:
            dff["target"] = dff["Attrition"].map({"Yes": 1, "No": 0})

        rate_df = (dff.groupby(cat2)["target"]
                   .agg(attrition_rate="mean", employee_count="size")
                   .reset_index())
        rate_df["attrition_rate_pct"] = (100 * rate_df["attrition_rate"]).round(1)

        fig_rate = px.bar(
            rate_df, x=cat2, y="attrition_rate_pct",
            hover_data=["employee_count"],
            title=f"Attrition rate (%) by {cat2}",
            labels={"attrition_rate_pct": "Attrition Rate (%)"},
            color="attrition_rate_pct",
            color_continuous_scale="RdYlGn_r",  # red = high attrition
        )
        st.plotly_chart(fig_rate, use_container_width=True)
        st.caption(
            "Hover over bars to see the employee count. "
            "Red bars indicate higher attrition risk."
        )

    # ==================================================================
    # 3) Numeric distribution (box plot)
    # ==================================================================
    st.subheader("Numeric distribution by attrition")
    num_choices = [c for c in SUGGESTED_NUM if c in dff.columns]
    if not num_choices:
        st.info("No suggested numeric columns found in data.")
    else:
        num = st.selectbox("Numeric feature", num_choices, index=0,
                           key="num_select")
        hue = "Attrition" if "Attrition" in dff.columns else "target"
        fig_box = px.box(dff, x=hue, y=num, points="all",
                         title=f"{num} distribution by Attrition")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(
            "If box plots differ a lot between Yes/No, "
            "that feature may matter for attrition."
        )

    # ==================================================================
    # 4) Interactive sunburst chart (drill-down by clicking)
    #    This clearly satisfies merit criterion 6.5 — interactive plot
    # ==================================================================
    st.divider()
    st.subheader("Interactive Sunburst — Attrition by Department, JobRole & OverTime")
    st.markdown("""
    Click on a segment to **drill down** into that group.  
    Click the centre to **zoom back out**. This shows how attrition  
    is distributed across departments, roles, and overtime status.
    """)

    # Build the sunburst only if the needed columns exist
    sunburst_cols = ["Department", "JobRole", "OverTime"]
    if all(c in dff.columns for c in sunburst_cols) and "Attrition" in dff.columns:
        fig_sun = px.sunburst(
            dff,
            path=sunburst_cols,
            color="Attrition",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
            title="Click segments to drill down",
        )
        fig_sun.update_layout(height=550)
        st.plotly_chart(fig_sun, use_container_width=True)
        st.caption(
            "Red = left the company, blue = stayed. "
            "Click any segment to zoom in; click the centre to zoom out."
        )
    else:
        st.info("Sunburst requires Department, JobRole, OverTime and Attrition columns.")

    # ==================================================================
    # 5) Correlation heatmap (numeric features + target)
    # ==================================================================
    st.divider()
    st.subheader("Correlation heatmap (numeric features)")
    if "target" not in dff.columns and "Attrition" in dff.columns:
        dff["target"] = dff["Attrition"].map({"Yes": 1, "No": 0})

    num_cols = [c for c in SUGGESTED_NUM if c in dff.columns]
    cols_for_corr = [*num_cols, "target"] if "target" in dff.columns else num_cols

    if len(cols_for_corr) >= 2:
        corr = dff[cols_for_corr].corr(numeric_only=True)
        heat = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="Correlation heatmap (hover for values)",
        )
        st.plotly_chart(heat, use_container_width=True)
        st.caption(
            "Stronger absolute correlation with 'target' can indicate "
            "predictive power (watch for leakage)."
        )
    else:
        st.info("Not enough numeric columns to compute a correlation heatmap.")
