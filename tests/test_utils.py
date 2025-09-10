import pandas as pd
from src.utils import yes_no_to_binary

def test_yes_no_to_binary():
    s = pd.Series(["Yes", "No", "Yes"])
    out = yes_no_to_binary(s)
    assert list(out) == [1, 0, 1]