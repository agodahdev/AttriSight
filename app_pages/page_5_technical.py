import streamlit as st
import pandas as pd
import json, joblib
from pathlib import Path

# Paths
ASSETS_DIR = Path("assets")
ROC_PATH = ASSETS_DIR / "roc_curve.png"
CM_PATH  = ASSETS_DIR / "confusion_matrix_050.png"
THR_CSV  = ASSETS_DIR / "threshold_metrics.csv"

MODEL_PATH = Path("artifacts/v1/rf_pipeline.joblib")
FEATS_PATH = Path("artifacts/v1/features.json")
READY_DATA_PATH = Path("data/processed/hr_attrition_ready.parquet")

# --- Helpers (cached) ---
@st.cache_resource
def _load_artifacts():
    """Return (pipeline, features_list) or (None, None) if missing."""
    pipe = feats = None
    try:
        pipe = joblib.load(MODEL_PATH)
    except Exception:
        pass
    try:
        feats = json.loads(FEATS_PATH.read_text())
    except Exception:
        pass
    return pipe, feats

@st.cache_data
def _load_threshold_csv():
    if THR_CSV.exists():
        return pd.read_csv(THR_CSV)
    return None

def run():
    st.title("Technical: Model & Evaluation")

    # Saved figures
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curve")
        if ROC_PATH.exists():
            st.image(str(ROC_PATH), use_column_width=True)
        else:
            st.info("ROC figure not found. Run Notebook 04 to generate `assets/roc_curve.png`.")
    with c2:
        st.subheader("Confusion Matrix @ 0.50")
        if CM_PATH.exists():
            st.image(str(CM_PATH), use_column_width=True)
        else:
            st.info("Confusion matrix image not found. Run Notebook 04 to generate it.")

    st.divider()

    # Threshold sweep table + live confusion matrix
    if THR_CSV.exists():
        st.subheader("Threshold metrics")
        thr_df = pd.read_csv(THR_CSV)
        st.dataframe(
            thr_df.style.highlight_max(axis=0, subset=["f1"]).format(
                {"accuracy":"{:.3f}","precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}
            ),
            use_container_width=True
        )
        st.download_button("Download CSV", THR_CSV.read_bytes(), file_name="threshold_metrics.csv")

        # Live confusion matrix at chosen threshold 
        st.subheader("Confusion Matrix (live)")
        thr = st.slider("Choose threshold", 0.0, 1.0, 0.50, 0.01)

        # Load artifacts and data for live CM
        pipe, feats = _load_artifacts()
        if pipe and feats:
            import numpy as np
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt

            if READY_DATA_PATH.exists():
                df_ready = pd.read_parquet(READY_DATA_PATH)
                y_true = df_ready["target"].to_numpy()
                y_prob = pipe.predict_proba(df_ready[feats])[:,1]
                y_pred = (y_prob >= thr).astype(int)
                cm = confusion_matrix(y_true, y_pred, labels=[0,1])

                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm, display_labels=["Stay (0)","Leave (1)"]).plot(values_format="d", ax=ax)
                ax.set_title(f"Confusion Matrix @ threshold = {thr:.2f}")
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("`data/processed/hr_attrition_ready.parquet` not found. Run Notebook 02.")
        else:
            st.warning("Model artifacts not found. Train/export in Notebook 03.")
    else:
        st.info("`assets/threshold_metrics.csv` not found. Run Notebook 04 to create it.")

    st.divider()

    # Pipeline details 
    st.subheader("Pipeline details")
    pipe, feats = _load_artifacts()
    if pipe is None:
        st.warning("Model artifact not found at `artifacts/v1/rf_pipeline.joblib`.")
        return

    st.markdown("**Steps (in order):**")
    for name, step in pipe.named_steps.items():
        st.write(f"- **{name}**: `{type(step).__name__}`")

    if feats:
        with st.expander("Show feature list used during training"):
            st.code("\n".join(feats), language="text")

    st.caption("Preprocessing: numeric = impute+scale; categorical = impute+one-hot (ColumnTransformer).")