import streamlit as st
from app_pages.page_1_summary import run as page_summary
from app_pages.page_2_analysis import run as page_analysis
from app_pages.page_3_hypotheses import run as page_hypotheses
from app_pages.page_4_ml import run as page_ml
from app_pages.page_5_technical import run as page_technical

st.set_page_config(page_title="AttriSight", layout="wide")
PAGES = {
    "Project Summary": page_summary,
    "Workforce Analysis": page_analysis,
    "Project Hypotheses": page_hypotheses,
    "Attrition Predictor (ML)": page_ml,
    "Technical: Model & Evaluation": page_technical
}
st.sidebar.title("AttriSight")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()