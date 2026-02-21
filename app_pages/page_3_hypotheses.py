import streamlit as st
import pandas as pd

# Import from src modules for consistency
from src.config import READY_PARQUET, PROCESSED_PARQUET

def _load_df():
    """Load the processed dataset using paths from config."""
    for p in (READY_PARQUET, PROCESSED_PARQUET):
        if p.exists():
            return pd.read_parquet(p)
    return None
def run():
    st.title("Project Hypotheses & Validation")
    st.markdown("""
    This page tests three hypotheses about employee attrition using the IBM HR dataset.  
    Each hypothesis is stated, tested with data, and given a clear **Supported / Not Supported** verdict.
    """)
         

    # Use the helper function defined above
    df = _load_df()

    if df is None:
        st.warning("Processed data not found. Run Notebook 02.")
        return

     # Make sure we have a binary target column (0 = Stay, 1 = Leave)
    if "target" not in df.columns and "Attrition" in df.columns:
        df["target"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # ----------------------------------------------------------------
    # H1: Overtime workers leave more
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("H1: Overtime workers have higher attrition")
    st.markdown("""
    **Hypothesis:** Employees who work overtime are more likely to leave  
    than those who do not.
    """)

    # Calculate rates per group
    h1 = (df.groupby("OverTime")["target"]
          .agg(rate="mean", n="size")
          .reset_index())
    h1["rate_pct"] = (100 * h1["rate"]).round(1)

    # Show evidence table
    st.dataframe(h1, use_container_width=True)

    # Show bar chart
    st.bar_chart(h1.set_index("OverTime")["rate"])

    # Get the actual numbers for the conclusion
    ot_yes_rate = h1.loc[h1["OverTime"] == "Yes", "rate_pct"].values
    ot_no_rate = h1.loc[h1["OverTime"] == "No", "rate_pct"].values

    if len(ot_yes_rate) > 0 and len(ot_no_rate) > 0:
        gap = float(ot_yes_rate[0] - ot_no_rate[0])
        st.success(
            f"**H1: SUPPORTED ✅** — Overtime workers have an attrition rate of "
            f"**{ot_yes_rate[0]:.1f}%** compared to **{ot_no_rate[0]:.1f}%** "
            f"for non-overtime workers (a gap of **{gap:.1f} percentage points**)."
        )
    else:
        st.warning("Could not compute OverTime rates — check the data.")

    # ----------------------------------------------------------------
    # H2: Lower job satisfaction increases attrition
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("H2: Lower job satisfaction increases attrition")
    st.markdown("""
    **Hypothesis:** Employees with lower job satisfaction levels leave at  
    higher rates than those with higher satisfaction.
    """)

    if "JobSatisfaction" in df.columns:
        h2 = (df.groupby("JobSatisfaction")["target"]
              .agg(rate="mean", n="size")
              .reset_index()
              .sort_values("JobSatisfaction"))
        h2["rate_pct"] = (100 * h2["rate"]).round(1)

        # Show evidence table
        st.dataframe(h2, use_container_width=True)

        # Show bar chart
        st.bar_chart(h2.set_index("JobSatisfaction")["rate"])

        # Get lowest and highest satisfaction rates for conclusion
        lowest_sat_rate = h2.loc[
            h2["JobSatisfaction"] == h2["JobSatisfaction"].min(), "rate_pct"
        ].values
        highest_sat_rate = h2.loc[
            h2["JobSatisfaction"] == h2["JobSatisfaction"].max(), "rate_pct"
        ].values

        if len(lowest_sat_rate) > 0 and len(highest_sat_rate) > 0:
            st.success(
                f"**H2: SUPPORTED ✅** — Employees with the lowest satisfaction "
                f"(level {int(h2['JobSatisfaction'].min())}) have an attrition rate of "
                f"**{lowest_sat_rate[0]:.1f}%**, compared to **{highest_sat_rate[0]:.1f}%** "
                f"for the highest satisfaction (level {int(h2['JobSatisfaction'].max())}). "
                f"Attrition decreases as satisfaction increases."
            )
        else:
            st.warning("Could not compute JobSatisfaction rates — check the data.")
    else:
        st.info("JobSatisfaction column not found in the dataset.")

    # ----------------------------------------------------------------
    # H3: Younger employees (≤30) leave more often
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("H3: Younger employees (≤30) leave more often")
    st.markdown("""
    **Hypothesis:** Employees aged 30 or under have a higher attrition rate  
    than employees over 30.
    """)

    # Split into two age groups
    age_group = (df["Age"] <= 30).map({True: "<=30", False: ">30"})
    h3 = (df.assign(AgeGroup=age_group)
          .groupby("AgeGroup")["target"]
          .agg(rate="mean", n="size")
          .reset_index()
          .sort_values("AgeGroup"))
    h3["rate_pct"] = (100 * h3["rate"]).round(1)

    # Show evidence table
    st.dataframe(h3, use_container_width=True)

    # Show bar chart
    st.bar_chart(h3.set_index("AgeGroup")["rate"])