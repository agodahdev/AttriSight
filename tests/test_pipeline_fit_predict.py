import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_READY
from src.features import NUM_FEATURES, CAT_FEATURES
from src.pipeline import make_logreg_pipeline  # or make_rf_pipeline

@pytest.mark.skipif(not DATA_READY.exists(), reason="ready parquet not found")
def test_pipeline_fit_predict():
    df = pd.read_parquet(DATA_READY)

    # Use only features that actually exist
    num = [c for c in NUM_FEATURES if c in df.columns]
    cat = [c for c in CAT_FEATURES if c in df.columns]
    assert num or cat, "No matching features found in dataframe"

    X = df[num + cat].head(400)             # small sample = fast
    y = df["target"].head(400).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42, stratify=y)
    pipe = make_logreg_pipeline(num, cat)   # or make_rf_pipeline(num, cat)
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    # predictions must be 0 or 1
    assert set(preds) <= {0, 1}