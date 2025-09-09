from pathlib import Path

# Project root (the folder that contains app.py)
ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR        = ROOT / "data"
DATA_RAW        = DATA_DIR / "raw"
DATA_PROCESSED  = DATA_DIR / "processed"

# Data files
RAW_CSV           = DATA_RAW / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_PARQUET = DATA_PROCESSED / "hr_attrition.parquet"
READY_PARQUET     = DATA_PROCESSED / "hr_attrition_ready.parquet"

# Artifacts and assets
ARTIFACTS_DIR = ROOT / "artifacts" / "v1"
ASSETS_DIR   = ROOT / "assets"

# Model files
MODEL_FILE    = ARTIFACTS_DIR / "rf_pipeline.joblib"
FEATURES_FILE = ARTIFACTS_DIR / "features.json"

DATA_READY = READY_PARQUET
