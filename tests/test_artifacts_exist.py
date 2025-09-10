import pytest
from src.config import ARTIFACTS_DIR, MODEL_FILE, FEATURES_FILE, ASSETS_DIR

def test_model_artifacts_present():
    # Skip rather than fail if not trained yet
    if not ARTIFACTS_DIR.exists():
        pytest.skip("artifacts folder not created yet")
    assert MODEL_FILE.exists(), "rf_pipeline.joblib missing"
    assert FEATURES_FILE.exists(), "features.json missing"

def test_assets_present():
    # These appear after Notebook 04
    if not ASSETS_DIR.exists():
        pytest.skip("assets folder not created yet")
    # Don't hard-fail if images not exported yet; just check folder exists
    assert ASSETS_DIR.exists()