import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import DATA_PROCESSED

def run():
    st.title("Workforce Analysis (Conventional)")
    if not DATA_PROCESSED.exists():
        st.warning("No processed data yet. Run notebooks 01–02.")
        return
    df = pd.read_parquet(DATA_PROCESSED)

    st.write("Compare attrition by category:")
    cat = st.selectbox("Categorical", ["OverTime","JobRole","MaritalStatus","BusinessTravel","Department","EducationField","Gender","JobLevel"])
    st.caption("Bars for 'Yes' higher than 'No' → stronger association with attrition.")
    st.plotly_chart(px.histogram(df, x=cat, color="Attrition", barmode="group"), use_container_width=True)

    st.write("Numeric distribution by attrition:")
    num = st.selectbox("Numeric", ["Age","MonthlyIncome","DistanceFromHome","TotalWorkingYears","YearsAtCompany","NumCompaniesWorked","PercentSalaryHike"])
    st.caption("If distributions differ across 'Yes'/'No', the variable may relate to attrition.")
    st.plotly_chart(px.box(df, x="Attrition", y=num, points="all"), use_container_width=True)

    st.divider()
    st.subheader("Correlation (numeric features)")
    num_cols = ["Age","MonthlyIncome","DistanceFromHome","TotalWorkingYears","YearsAtCompany","NumCompaniesWorked","PercentSalaryHike"]
    corr = df[num_cols + ["target"]].corr(numeric_only=True)
    import plotly.express as px
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap"), use_container_width=True)

 