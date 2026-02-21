import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# -- Paths (use src.config, fall back to local if import fails) --
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
CM_PATH = ASSETS_DIR / "confusion_matrix_050.png"
THR_CSV = ASSETS_DIR / "threshold_metrics.csv"


# -- Cached loaders (run once, then reuse) --

@st.cache_resource
def _load_artifacts():
    """Load the trained model and feature list once."""
    model = joblib.load(MODEL_PATH)
    feats = json.loads(FEATS_PATH.read_text())
    return model, feats


@st.cache_data
def _load_ready_df():
    """Load the processed dataset once."""
    return pd.read_parquet(DATA_READY) if DATA_READY.exists() else None


@st.cache_data
def _load_threshold_table():
    """Load the pre-computed threshold metrics table."""
    return pd.read_csv(THR_CSV) if THR_CSV.exists() else None


def run():
    st.title("Technical: Model & Evaluation")
    st.markdown("""
    This page shows whether the ML model **meets the business goal** (ROC-AUC ≥ 0.75),  
    displays evaluation charts, and lets you explore different thresholds interactively.
    """)


    # Load model + data

    try:
        pipe, feats = _load_artifacts()
    except Exception as e:
        st.error("Model artifacts not found. Train/export them in Notebook 03.")
        st.caption(f"Expected: {MODEL_PATH.relative_to(ROOT)} and "
                   f"{FEATS_PATH.relative_to(ROOT)}")
        st.caption(f"Error: {e}")
        return

    df = _load_ready_df()
    if df is None:
        st.error("Processed data not found. Create it in Notebook 02.")
        st.caption(f"Expected: {DATA_READY.relative_to(ROOT)}")
        return

  
    # Compute predictions on the full dataset
   
    try:
        X = df[feats]
        y_true = df["target"].to_numpy()
        y_prob = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        st.warning(f"Could not compute predictions: {e}")
        y_true, y_prob, auc = None, None, None


    # 1) MODEL PERFORMANCE — clear success / failure verdict
    
    st.subheader("Model Performance Evaluation")
    target_auc = 0.75

    if auc is not None:
        # Show the numbers
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("ROC-AUC achieved", f"{auc:.3f}")
        with col_b:
            st.metric("Target ROC-AUC", f"≥ {target_auc:.2f}")

        # Clear pass / fail verdict
        if auc >= target_auc:
            st.success(
                f"**MODEL MEETS THE BUSINESS REQUIREMENT ✅**  \n"
                f"The Random Forest model achieves ROC-AUC of **{auc:.3f}**, "
                f"which exceeds the goal of ≥ {target_auc:.2f}.  \n"
                f"**Conclusion:** The model is suitable for deployment to "
                f"predict employee attrition risk."
            )
        else:
            st.error(
                f"**MODEL DOES NOT MEET THE BUSINESS REQUIREMENT ❌**  \n"
                f"ROC-AUC of **{auc:.3f}** is below the goal of "
                f"≥ {target_auc:.2f}.  \n"
                f"**Conclusion:** Further tuning or a different algorithm "
                f"is needed before deployment."
            )
    else:
        st.warning("Could not evaluate the model — check data and artifacts.")

    st.divider()


    # 2) RESULTS METRICS TABLE

    st.subheader("Results Metrics (threshold = 0.50)")

    if y_true is not None and y_prob is not None:
        y_pred_50 = (y_prob >= 0.50).astype(int)

        # Key metrics in a row of columns
        acc = accuracy_score(y_true, y_pred_50)
        prec = precision_score(y_true, y_pred_50, zero_division=0)
        rec = recall_score(y_true, y_pred_50)
        f1 = f1_score(y_true, y_pred_50)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision (Leave)", f"{prec:.3f}")
        m3.metric("Recall (Leave)", f"{rec:.3f}")
        m4.metric("F1-Score (Leave)", f"{f1:.3f}")

        # Full classification report as a table
        report_dict = classification_report(y_true, y_pred_50, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(
            report_df.style
            .format("{:.3f}")
            .background_gradient(cmap="RdYlGn", subset=["f1-score"]),
            use_container_width=True,
        )
        st.caption("""
        **How to read this:**  
        - **Precision (class 1):** Of employees predicted to leave, what % actually left.  
        - **Recall (class 1):** Of employees who actually left, what % did we catch.  
        - **F1-score:** Balance of precision and recall — higher is better.  
        - Focus on class **1 (Leave)** for business decisions.
        """)
    else:
        st.info("Metrics unavailable — check that data and model loaded correctly.")

    st.divider()

    # 3) ACTUAL vs PREDICTED PLOT
    
    st.subheader("Actual vs Predicted (threshold = 0.50)")

    if y_true is not None and y_prob is not None:
        y_pred_50 = (y_prob >= 0.50).astype(int)

        comparison_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred_50,
            "Probability": y_prob,
        })
        comparison_df["Correct"] = comparison_df["Actual"] == comparison_df["Predicted"]

        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        correct = comparison_df[comparison_df["Correct"]]
        incorrect = comparison_df[~comparison_df["Correct"]]

        ax.scatter(correct["Actual"], correct["Predicted"],
                   alpha=0.6, c="green",
                   label=f"Correct ({len(correct)})", s=50)
        ax.scatter(incorrect["Actual"], incorrect["Predicted"],
                   alpha=0.6, c="red", marker="x",
                   label=f"Incorrect ({len(incorrect)})", s=50)

        ax.set_xlabel("Actual Class (0=Stay, 1=Leave)", fontsize=12)
        ax.set_ylabel("Predicted Class (0=Stay, 1=Leave)", fontsize=12)
        ax.set_title("Actual vs Predicted Classification", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)

        # Accuracy breakdown
        accuracy = (comparison_df["Correct"].sum() / len(comparison_df)) * 100
        st.markdown(
            f"**Prediction Accuracy:** {accuracy:.1f}%  \n"
            f"- ✅ Correct: {len(correct):,} "
            f"({len(correct)/len(comparison_df)*100:.1f}%)  \n"
            f"- ❌ Incorrect: {len(incorrect):,} "
            f"({len(incorrect)/len(comparison_df)*100:.1f}%)"
        )

        # Show a sample of misclassified cases
        if len(incorrect) > 0:
            with st.expander("View sample of misclassified cases"):
                st.dataframe(
                    incorrect.head(10).style.format({"Probability": "{:.3f}"}),
                    use_container_width=True,
                )
                st.caption("Review these cases to understand model limitations.")
    else:
        st.info("Actual vs Predicted plot unavailable — "
                "check that data and model loaded correctly.")

    st.divider()

    
    # 4) SAVED FIGURES (ROC curve + Confusion Matrix @ 0.50)
    
    st.subheader("Saved Evaluation Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC Curve**")
        if ROC_PATH.exists():
            st.image(str(ROC_PATH), use_column_width=True)
        else:
            st.info(f"Missing: {ROC_PATH.relative_to(ROOT)}. "
                    "Run Notebook 04 to create it.")
    with c2:
        st.markdown("**Confusion Matrix @ 0.50**")
        if CM_PATH.exists():
            st.image(str(CM_PATH), use_column_width=True)
        else:
            st.info(f"Missing: {CM_PATH.relative_to(ROOT)}. "
                    "Run Notebook 04 to generate it.")

    st.divider()

    # 5) THRESHOLD TABLE (pre-computed from Notebook 04)
    
    st.subheader("Threshold Metrics")
    thr_df = _load_threshold_table()
    if thr_df is not None:
        st.dataframe(
            thr_df.style
            .highlight_max(axis=0, subset=["f1"])
            .format({"accuracy": "{:.3f}", "precision": "{:.3f}",
                      "recall": "{:.3f}", "f1": "{:.3f}"}),
            use_container_width=True,
        )
        st.download_button("Download CSV", THR_CSV.read_bytes(),
                           file_name="threshold_metrics.csv")
    else:
        st.info(f"Missing: {THR_CSV.relative_to(ROOT)}. "
                "Run Notebook 04 to create it.")

    st.divider()

    # 6) LIVE CONFUSION MATRIX — interactive slider

    st.subheader("Live Confusion Matrix (interactive)")
    st.markdown("**Move the slider** to see how changing the threshold "
                "affects predictions.")

    if y_true is not None and y_prob is not None:
        thr = st.slider("Choose threshold", 0.0, 1.0, 0.50, 0.01)
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            cm, display_labels=["Stay (0)", "Leave (1)"]
        ).plot(values_format="d", ax=ax)
        ax.set_title(f"Confusion Matrix @ threshold = {thr:.2f}")
        st.pyplot(fig, use_container_width=True)

        # Show what this threshold means in plain English
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"At threshold {thr:.2f}: "
            f"**{tp}** correctly flagged as leaving, "
            f"**{fn}** missed (actually left but predicted stay), "
            f"**{fp}** false alarms (predicted leave but stayed)."
        )
    else:
        st.caption("Live confusion matrix unavailable — "
                   "AUC could not be computed.")

    st.divider()

   
    # 7) PIPELINE DETAILS
 
    st.subheader("Pipeline Details")
    st.markdown("**Steps (in order):**")
    for name, step in pipe.named_steps.items():
        st.write(f"- **{name}**: `{type(step).__name__}`")

    if feats:
        with st.expander("Show feature list used during training"):
            st.code("\n".join(map(str, feats)), language="text")

    st.caption(
        "Preprocessing: numeric = impute(median) + scale; "
        "categorical = impute(most_frequent) + one-hot (ignore unknowns)."
    )
