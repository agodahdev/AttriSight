# test_pages_import.py — check that every app page can be imported
# This catches syntax errors and broken imports before deployment.

def test_import_page_1():
    """Page 1 (Summary) imports without error."""
    from app_pages.page_1_summary import run
    assert callable(run)


def test_import_page_2():
    """Page 2 (Workforce Analysis) imports without error."""
    from app_pages.page_2_analysis import run
    assert callable(run)


def test_import_page_3():
    """Page 3 (Hypotheses) imports without error."""
    from app_pages.page_3_hypotheses import run
    assert callable(run)


def test_import_page_4():
    """Page 4 (ML Predictor) imports without error."""
    from app_pages.page_4_ml import run
    assert callable(run)


def test_import_page_5():
    """Page 5 (Technical) imports without error."""
    from app_pages.page_5_technical import run
    assert callable(run)