import streamlit as st
import pandas as pd
from src.config import DATA_PROCESSED    

def run():
    st.title("Project Summary")
    st.write("""
    **Client requirements**
     - **BR#1 (EDA):** Understand factors linked with employee attrition.
     - **BR#2 (ML):** Predict attrition risk for an employee profile in a dashboard.
    """)
    
    if DATA_PROCESSED.exists():
        df = pd.read_parquet(DATA_PROCESSED)
        st.subheader("Dataset at a glance")
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        st.dataframe(df.head(10))
    else:
        st.info("Processed dataset not found yet. Run notebooks 01–02 first.")