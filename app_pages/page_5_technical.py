import streamlit as st
import pandas as pd
import json, joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics  import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Paths
ROOT = Path(__file__).resolve().parents[1]
try:
    from src.config import READY_PARQUET as DATA_READY, ARTIFACTS_DIR, ASSETS_DIR
    MODEL_PATH = ARTIFACTS_DIR / "rf_pipeline.joblib"
    FEATS_PATH = ARTIFACTS_DIR / "features.json"
except Exception:
    DATA_READY = ROOT / "data" / "processed" / "hr_attrition_ready.parquet"
    ARTIFACTS_DIR = ROOT / "artifacts" / "v1"
    ASSETS_DIR = ROOT / "assets"
    MODEL_PATH = ARTIFACTS_DIR / "rf_pipeline.joblib"
    FEATS_PATH = ARTIFACTS_DIR / "features.json"

ROC_PATH = ASSETS_DIR / "roc_curve.png"
CM_PATH  = ASSETS_DIR / "confusion_matrix_050.png"
THR_CSV  = ASSETS_DIR / "threshold_metrics.csv"

# Helpers (cached) 
@st.cache_resource
def _load_artifacts():
    """Load trained model feature list once"""
    model = joblib.load(MODEL_PATH)
    feats = json.loads(FEATS_PATH.read_text())
    return model, feats

@st.cache_data
def _load_ready_df():
    """ Load processed dataset once"""
    return pd.read_parquet(DATA_READY) if DATA_READY.exists() else None

@st.cache_data
def _load_threshold_table():
     return pd.read_csv(THR_CSV) if THR_CSV.exists() else None

def run():
    st.title("Technical: Model & Evaluation")

     # Load artifacts & data 
    try:
        pipe, feats = _load_artifacts()
    except Exception as e:
        st.error("Model artifacts not found. Train/export them in Notebook 03.")
        st.caption(f"Expected: {MODEL_PATH.relative_to(ROOT)} and {FEATS_PATH.relative_to(ROOT)}")
        st.caption(f"Error: {e}")
        return

    df = _load_ready_df()
    if df is None:
        st.error("Processed data not found. Create it in Notebook 02.")
        st.caption(f"Expected: {DATA_READY.relative_to(ROOT)}")
        return

    #  AUC on full data + clear goal line
    try:
        X = df[feats]
        y_true = df["target"].to_numpy()
        y_prob = pipe.predict_proba(X)[:, 1]

        auc = roc_auc_score(y_true, y_prob)
        st.markdown(f"**ROC-AUC:** {auc:.3f}")

        target_auc = 0.75
        st.markdown(f"**Performance goal:** ROC-AUC ≥ {target_auc:.2f}")
        
        # Clear objective statement on model success
        st.markdown("---")
        st.subheader("Model Performance Evaluation")
        if auc >= target_auc:
            st.success(f"✅ **MODEL SUCCESS**: The model achieves ROC-AUC of {auc:.3f}, which **MEETS** the business requirement of ≥ {target_auc:.2f}")
            st.markdown("""
            **Conclusion:** The Random Forest classification model successfully meets the performance criteria 
            and is suitable for deployment to predict employee attrition risk.
            """)
        else:
            st.error(f"❌ **MODEL FAILURE**: The model achieves ROC-AUC of {auc:.3f}, which **DOES NOT MEET** the business requirement of ≥ {target_auc:.2f}")
            st.markdown("""
            **Conclusion:** The model requires further tuning or a different modeling approach before deployment. 
            Consider: more hyperparameter optimization, feature engineering, or alternative algorithms.
            """)
        st.markdown("---")
    except Exception as e:
        st.warning(f"Could not compute AUC: {e}")
        y_true, y_prob, auc = None, None, None

    st.divider()

    # Saved figures (ROC + Confusion Matrix @ 0.50)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curve (saved)")
        if ROC_PATH.exists():
            st.image(str(ROC_PATH), use_column_width=True)
        else:
            st.info(f"Missing: {ROC_PATH.relative_to(ROOT)}. Run Notebook 04 to create it.")
    with c2:
        st.subheader("Confusion Matrix @ 0.50")
        if CM_PATH.exists():
            st.image(str(CM_PATH), use_column_width=True)
        else:
            st.info(f"Missing: {CM_PATH.relative_to(ROOT)}. Run Notebook 04 to generate it.")

    st.divider()

    # Threshold table + live confusion matrix
    st.subheader("Threshold metrics")
    thr_df = _load_threshold_table()
    if thr_df is not None:
        st.dataframe(
            thr_df.style.highlight_max(axis=0, subset=["f1"]).format(
                {"accuracy":"{:.3f}","precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}
            ),
            use_container_width=True
        )
        st.download_button("Download CSV", THR_CSV.read_bytes(), file_name="threshold_metrics.csv")
    else:
        st.info(f"Missing table: {THR_CSV.relative_to(ROOT)}. Run Notebook 04 to create it.")

        # Live confusion matrix at chosen threshold 
    if (y_true is not None) and (y_prob is not None):
        st.subheader("Confusion Matrix (live)")
        thr = st.slider("Choose threshold", 0.0, 1.0, 0.50, 0.01)
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Stay (0)", "Leave (1)"]).plot(values_format="d", ax=ax)
        ax.set_title(f"Confusion Matrix @ threshold = {thr:.2f}")
        st.pyplot(fig, use_container_width=True)
        st.caption("Tip: move the slider to see how the threshold changes the confusion matrix.")
    else:
        st.caption("Live confusion matrix unavailable because AUC could not be computed.")
    
    st.divider()

    # Pipeline details 
    st.subheader("Pipeline details")
    st.markdown("**Steps (in order):**")
    for name, step in pipe.named_steps.items():
        st.write(f"- **{name}**: `{type(step).__name__}`")

    if feats:
        with st.expander("Show feature list used during training"):
            st.code("\n".join(map(str, feats)), language="text")

    st.caption("Preprocessing: numeric = impute(median) + scale; "
     "categorical = impute(mostt_frequent) + one-hot (ignore unknowns)."
    )