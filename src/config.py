from pathlib import Path

DATA_PROCESSED = Path("data/processed/hr_attrition.parquet")
DATA_READY = Path("data/processed/hr_attrition_ready.parquet")

ARTIFACT_DIR = Path("artifacts") / "v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "rf_pipeline.joblib"
FEATURES_JSON = ARTIFACT_DIR / "features.json"