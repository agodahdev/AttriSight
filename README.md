# AttriSight — HR Attrition Analytics & Predictor

A web app that helps HR **understand why people leave** and **predict who might leave next**.

## Overview & Goals

- **Explain (BR#1):** Show clear charts (bars, box plots, heatmap, interactive sunburst) so HR can see which factors relate to attrition.
- **Predict (BR#2):** Enter an employee profile and get an attrition probability plus a risk band (Low / Medium / High).
- **Audience:** HR, People Analytics, and managers.

## Business Case (in ML Terms)

- **Problem:** Reduce employee attrition by identifying risk early.
- **Users:** HR analysts and managers.
- **Dataset:** IBM HR Analytics — 1,470 employee records from Kaggle.
- **Features:** 7 numeric and 8 categorical (15 total), listed below.
- **Target:** Attrition (Yes / No mapped to 1 / 0).
- **Primary metric:** ROC-AUC (goal: ≥ 0.75).
- **Output:** Probability that the employee will leave (binary classification: 0 / 1).

### Dataset & Train/Test Split

| Detail         | Value                                                                |
| -------------- | -------------------------------------------------------------------- |
| Total samples  | 1,470                                                                |
| Training set   | 1,176 (80%)                                                          |
| Test set       | 294 (20%)                                                            |
| Split strategy | Stratified by target to maintain ~16% attrition rate in both sets    |
| Validation     | 5-fold cross-validation during grid search (no separate holdout set) |

### Features Used

**Numeric (7):** Age, MonthlyIncome, DistanceFromHome, TotalWorkingYears, YearsAtCompany, NumCompaniesWorked, PercentSalaryHike

**Categorical (8):** OverTime, JobRole, MaritalStatus, BusinessTravel, Department, EducationField, Gender, JobLevel

### Decision Support

Risk band thresholds (default values, adjustable with stakeholders):

- **Low** — probability below 0.35
- **Medium** — probability 0.35 to 0.59
- **High** — probability 0.60 and above

### Success Criteria

The model reaches or beats the ROC-AUC goal of 0.75, and insights are understandable enough to guide action (for example, focus on overtime and low-satisfaction groups).

## Map Business Requirements to Tasks

### BR#1 (Explain)

- **User story:** "As HR, I want clear charts so I can see which factors relate to attrition."
- **Tasks:** Grouped bar charts, box plots, correlation heatmap, interactive sunburst, attrition rate by category, short captions under each plot.
- **Pages:** Workforce Analysis, Project Hypotheses.

### BR#2 (Predict)

- **User story:** "As HR, I want to enter a profile and see the attrition probability."
- **Tasks:** Build preprocessing and model pipeline, train and tune, export artifacts, Streamlit form for inputs, risk band output.
- **Pages:** Attrition Predictor (ML), Technical: Model & Evaluation.

## Hypothesis Validation Results

### H1: Overtime Workers Leave More — SUPPORTED

- Attrition rate for OverTime = Yes: approximately 30%
- Attrition rate for OverTime = No: approximately 10%
- Gap: roughly 20 percentage points higher attrition for overtime workers
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** There is strong statistical evidence that employees working overtime have significantly higher attrition rates. **Recommended action:** Review overtime policies and workload distribution.

### H2: Lower Job Satisfaction Increases Attrition — SUPPORTED

- JobSatisfaction level 1 (lowest): approximately 23% attrition rate
- JobSatisfaction level 4 (highest): approximately 11% attrition rate
- Pattern: Progressive decrease in attrition as satisfaction increases
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** There is a clear inverse relationship between job satisfaction and attrition. **Recommended action:** Implement regular satisfaction surveys and develop targeted retention programmes for low-satisfaction teams.

### H3: Younger Employees (age 30 or under) Leave More Often — SUPPORTED

- Age 30 or under: approximately 25% attrition rate
- Age over 30: approximately 13% attrition rate
- Gap: roughly 12 percentage points higher attrition for younger employees
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** Younger employees demonstrate significantly higher attrition rates. **Recommended action:** Develop early-career retention programmes including mentorship, clear career progression paths, and competitive compensation reviews.

### Statistical Validation Summary

All three hypotheses achieved statistical significance (p < 0.05) using chi-square tests of independence. The observed differences in attrition rates are very unlikely to be caused by random chance. The business should prioritise interventions addressing overtime practices, job satisfaction, and early-career retention.

## Modelling Summary

### Algorithms Evaluated

Two classification algorithms were compared to establish a strong baseline before hyperparameter tuning:

**1. Logistic Regression (Baseline)**

- Type: Linear model with L2 regularisation (max_iter=1000)
- ROC-AUC on test set: approximately 0.76 to 0.78
- Strengths: Fast training, interpretable coefficients, good baseline
- Limitations: Assumes linear relationships between features and log-odds

**2. Random Forest (Final Model — Selected)**

- Type: Ensemble of decision trees
- ROC-AUC on test set: approximately 0.82 to 0.85 (after tuning)
- Strengths: Handles non-linear relationships, captures feature interactions, robust to outliers, provides feature importance rankings
- Trade-off: Longer training time, less interpretable than Logistic Regression

**Decision:** Random Forest was selected as the final model due to superior ROC-AUC performance and ability to capture complex patterns in employee behaviour.

### Pipeline Architecture

Binary classification using Random Forest with a ColumnTransformer:

- Numeric preprocessing: impute with median, then standard scale
- Categorical preprocessing: impute with most frequent, then one-hot encode (ignore unknowns)

### Hyperparameter Tuning

Grid search over 6 hyperparameters (each with 3 or more values), optimising ROC-AUC with 5-fold cross-validation:

- `n_estimators` — number of trees in the forest
- `max_depth` — maximum depth of each tree
- `min_samples_split` — minimum samples required to split a node
- `min_samples_leaf` — minimum samples required at a leaf node
- `max_features` — number of features to consider for the best split
- `criterion` — function to measure the quality of a split

These parameters control model complexity (depth, leaves, split thresholds), ensemble size, feature sampling, and split measure.

## Model Performance

### Primary Metric

ROC-AUC with a goal of 0.75 or above.

### Where to See It

The Technical: Model & Evaluation page in the dashboard shows the AUC value and a clear pass/fail verdict against the goal.

### Evaluation Outputs

- ROC curve (saved image)
- Confusion matrix at threshold 0.50 (saved image)
- Classification report with precision, recall, and F1 for each class
- Actual vs predicted comparison plot
- Threshold sweep table (accuracy, precision, recall, F1 at multiple thresholds)
- Live interactive confusion matrix with a threshold slider

## Dashboard Design (Pages & Content)

### Project Summary

Project goal, client requirements, dataset preview, train/test split details, and navigation guide.

### Workforce Analysis (BR#1)

Filters (Department, OverTime, Age range), grouped bar charts, box plots, attrition rate by category with colour coding, interactive sunburst chart (click to drill down by Department, JobRole, OverTime), and correlation heatmap.

### Project Hypotheses

H1 (OverTime), H2 (JobSatisfaction), and H3 (Age group), each with: hypothesis statement, evidence table with rates, bar chart, explicit SUPPORTED/NOT SUPPORTED verdict with specific percentages, chi-square p-values, and a summary of findings with recommended actions.

### Attrition Predictor (BR#2)

Form with one input per feature (numeric inputs and categorical dropdowns). Returns attrition probability and risk band (Low, Medium, High).

### Technical: Model & Evaluation

ROC-AUC value with pass/fail verdict against the 0.75 goal, results metrics cards (accuracy, precision, recall, F1), classification report table, actual vs predicted plot, saved ROC and confusion matrix images, threshold metrics table with F1 highlight, live interactive confusion matrix with slider, and pipeline step details.

## How to Use the Attrition Predictor

### Step-by-Step Guide

1. Open the dashboard and click **Attrition Predictor (ML)** in the sidebar.
2. Enter employee information in the form. Numeric fields use number inputs; categorical fields use dropdowns showing values from the dataset.
3. Click **Predict** to see the attrition probability and risk band.

### Input Fields

**Numeric fields:** Age, MonthlyIncome, DistanceFromHome, TotalWorkingYears, YearsAtCompany, NumCompaniesWorked, PercentSalaryHike

**Categorical fields:** OverTime (Yes/No), JobRole, MaritalStatus (Single/Married/Divorced), BusinessTravel (Non-Travel/Travel_Rarely/Travel_Frequently), Department, EducationField, Gender (Male/Female), JobLevel (1-5)

### Understanding the Results

- **Low Risk** (below 35%) — Employee unlikely to leave. Maintain current engagement.
- **Medium Risk** (35% to 59%) — Monitor closely. Consider retention interventions.
- **High Risk** (60% and above) — Immediate action needed. Schedule a retention discussion.

### Tips for Best Results

- Ensure all employee information is current and correct.
- Try adjusting factors (for example, reduce OverTime or increase salary) to see the impact on risk.
- Re-run predictions quarterly or after major organisational changes.
- Use the Workforce Analysis page alongside the predictor to understand which factors matter most.

### Privacy Note

Employee profile data entered in this tool is not stored. Predictions are generated in real time and are not saved to any database.

## Reproduce the Project (Notebooks)

Run in order:

1. **01_data_collection.ipynb** — Pull from Kaggle into `data/raw/` and save `data/processed/hr_attrition.parquet`.
2. **02_clean_and_target.ipynb** — Create `target` (Yes to 1, No to 0), quick EDA, save `data/processed/hr_attrition_ready.parquet`.
3. **03_train_tune_export.ipynb** — Train baseline (Logistic Regression and Random Forest), grid search Random Forest, show train and test metrics, export model and features to `artifacts/v1/`.
4. **04_evaluate_and_release.ipynb** — Save ROC and confusion matrix images plus `threshold_metrics.csv` to `assets/`.

## Versioned Artifacts

- Trained model and feature list: `artifacts/v1/`
- Evaluation images and tables: `assets/`
- Processed data: `data/processed/`

All artifacts are committed to the repository so the deployed app works without re-training.

## Testing

### How to Run Tests

```bash
pytest -q
```

### What Is Included

**test_smoke.py** — Checks pytest runs at all (simple smoke test).

**test_pages_import.py** — Imports each Streamlit page module to catch syntax and import errors early.

**test_utils.py** — Tests the helper that converts Yes/No to 1/0.

**test_pipeline_fit_predict.py** — Loads a small sample, fits a Logistic Regression pipeline, checks it predicts 0 or 1 without error. Skips automatically if the ready file has not been created yet.

**test_artifacts_exist.py** — Checks that model and asset files exist after training.

### Test Configuration

`pytest.ini` limits tests to the `tests/` folder.

## Manual Testing

### A. App Starts

Run `streamlit run app.py`. The app loads with 5 pages in the sidebar.

### B. Summary Page

Shows a table preview of the dataset, train/test split details, and which file was loaded.

### C. Workforce Analysis (BR#1)

Filters work without conflicts. Charts render: grouped bars by category, box plot, attrition rate by category, interactive sunburst (click to drill down), and correlation heatmap. Each plot has a caption explaining what to look for.

### D. Project Hypotheses

Each hypothesis (H1, H2, H3) shows rates, a bar chart, and a bold SUPPORTED verdict with specific percentages. Chi-square p-values are shown if SciPy is installed. A summary box at the bottom gives all three verdicts and recommended actions.

### E. Attrition Predictor (BR#2)

The form shows one input per feature. Clicking Predict returns the probability and risk band with an icon.

### F. Technical: Model & Evaluation

Shows ROC-AUC with a clear pass/fail verdict against the 0.75 goal. Shows results metrics (accuracy, precision, recall, F1), classification report, actual vs predicted plot, saved ROC and confusion matrix images, threshold table, and live confusion matrix with a slider. Shows pipeline steps and feature list.

## Limitations & Next Steps

### Limitations

- The dataset is small (1,470 records) and synthetic. It may not reflect your organisation.
- Features are limited to structured data. No text or time-series features are included.
- The model was evaluated on the full dataset rather than a held-out test set in the dashboard, so displayed metrics may be optimistic.

### Next Steps

- Align risk band thresholds with HR stakeholders based on the costs of false positives and false negatives.
- Try other models such as XGBoost or calibrated probabilities.
- Add monitoring for data drift and weekly AUC tracking.
- Add feature importance explanations (for example, SHAP values) in the app.

## Error & Bug Fix Log

### Kaggle Download Error

**Symptom:** `CalledProcessError` when running `kaggle datasets download`.
**Cause:** Missing Kaggle login or wrong path.
**Fix:** Confirm `~/.kaggle/kaggle.json` exists, re-run the install and download cells, use `-p ../data/raw --unzip`.

### File Not Found for Processed Data

**Symptom:** `FileNotFoundError: ../data/processed/hr_attrition_ready.parquet`.
**Cause:** Notebook 02 not run yet.
**Fix:** Run Notebook 02 to create the ready parquet.

### Import Error for Config Variables

**Symptom:** `ImportError: cannot import name 'RAW_CSV' from src.config`.
**Cause:** Config variable names did not match.
**Fix:** Standardised names in `src/config.py` and updated pages to import them.

### NameError in Notebook 03 Grid Search

**Symptom:** `NameError: name 'Pipeline' is not defined` in grid search cells.
**Cause:** Cells used `Pipeline` and `preproc` without importing them.
**Fix:** Replaced with `make_rf_pipeline(NUM, CAT)` from `src.pipeline`.

### Float Rounding Error

**Symptom:** `AttributeError: 'float' object has no attribute 'round'`.
**Cause:** Used `.round(3)` on a Python float.
**Fix:** Use `round(value, 3)` or f-strings like `f"{value:.3f}"`.

### Grid Search Taking Too Long

**Symptom:** Grid search runs for a very long time.
**Cause:** Large parameter grid.
**Fix:** Two options — keep the big grid (thorough) and accept the wait, or use the small grid (faster).

### Model Not Found on Deployed App

**Symptom:** Dashboard shows "Model not found yet" and "Model artifacts not found".
**Cause:** `.gitignore` excluded `artifacts/*` and `data/processed/*`, so these files were not pushed to GitHub.
**Fix:** Removed those lines from `.gitignore` and committed the artifact files.

### Pipeline Test Skipped

**Symptom:** Pipeline test is skipped when running pytest.
**Cause:** Ready parquet not created yet.
**Fix:** Run Notebook 02 to create `hr_attrition_ready.parquet`.

## Credits & Resources

### Machine Learning

- [Scikit-learn Documentation](https://scikit-learn.org/stable/) — ML library documentation
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest) — Ensemble methods guide
- [ROC-AUC Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) — ROC curve documentation

### Data Science

- [Pandas Documentation](https://pandas.pydata.org/docs/) — Data manipulation and analysis
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) — Statistical data visualisation
- [Kaggle Learn](https://www.kaggle.com/learn) — Free micro-courses on data science

### Streamlit & Deployment

- [Streamlit Documentation](https://docs.streamlit.io/) — Interactive web app framework
- [Streamlit Gallery](https://streamlit.io/gallery) — Community app examples

### Dataset

- [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — Kaggle dataset used in this project

### Acknowledgements

- Code Institute — for the Predictive Analytics and Machine Learning course
- Kaggle Community — for providing high-quality datasets and notebooks
- Open Source Contributors — for maintaining the Python data science ecosystem
