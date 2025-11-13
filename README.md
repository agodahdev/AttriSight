# AttriSight — HR Attrition Analytics & Predictor

A simple web app that helps HR **understand why people leave** and **predict who might leave next**.

## Overview & Goals

- **Explain (BR#1):** Show clear charts (bars / box / heatmap) to see which factors relate to attrition.
- **Predict (BR#2):** Enter an employee profile and get an **attrition probability** + **risk band** (Low/Medium/High).
- **Audience:** HR / People Analytics / managers.

## Business Case (in ML terms)

- **Problem:** Reduce employee attrition by identifying risk early.
- **Users:** HR analysts and managers.
- **Inputs (features):** e.g., `Age`, `MonthlyIncome`, `DistanceFromHome`, `TotalWorkingYears`, `YearsAtCompany`, `NumCompaniesWorked`, `PercentSalaryHike`, plus categorical fields such as `OverTime`, `JobRole`, `MaritalStatus`, `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobLevel`.
- **Dataset:** 1,470 employee records from IBM HR Analytics
  - **Training set:** 1,176 samples (80%)
  - **Test set:** 294 samples (20%)
  - **Split strategy:** Stratified by target to maintain class balance (~16% attrition rate in both sets)
- **Output:** Probability that the employee will leave (classification: 0/1).
- **Primary metric:** **ROC-AUC** (goal: **≥ 0.75**).
- **Decision support:** Risk band thresholds (default: Low < 0.35, Medium 0.35–0.59, High ≥ 0.60). These can be adjusted with stakeholders.
- **Success:** Model reaches or beats the ROC-AUC goal, and insights are understandable enough to guide action (e.g., focus on overtime and low satisfaction groups).

## Reproduce the Project (Notebooks)

Run in order:

1. **01_data_collection.ipynb**

   - Pull from Kaggle into `data/raw/` and save `data/processed/hr_attrition.parquet`.

2. **02_clean_target.ipynb**

   - Create `target` (Yes→1 / No→0), quick EDA, save `data/processed/hr_attrition_ready.parquet`.

3. **03_train_tune_export.ipynb**

   - Train baseline (LogReg, RF), **grid search RF**, show train+test metrics, export model & features to `artifacts/v1/`.

4. **04_evaluate_and_release.ipynb**
   - Save **ROC** and **Confusion Matrix** images + **threshold_metrics.csv** to `assets/`.

## Dashboard Design (pages & content)

### **Project Summary**

- Project goal, client requirements, dataset preview + where files were loaded from.

### **Workforce Analysis (BR#1)**

- Filters, bar charts (e.g., OverTime), box plot (e.g., Age), correlation heatmap (numeric).
- Caption under each plot explaining what to look for.

### **Project Hypotheses**

- **H1:** OverTime → higher attrition (shows rates, clear KPIs, gap, and callout).
- **H2:** Low JobSatisfaction → higher attrition.
- **H3:** ≤30 yrs → higher attrition.
- (Optional) Chi-square p-values if SciPy installed.

### **Attrition Predictor (BR#2)**

- Form with one input per feature → probability + risk band.

### **Technical**

- **ROC-AUC** vs goal line, saved **ROC** and **CM**, threshold table with F1 highlight, live CM slider, and pipeline steps.

## Map Business Requirements → Tasks

### **BR#1 (Explain)**

- **User story:** "As HR, I want clear charts so I can see which factors relate to attrition."
- **Tasks:** grouped bar charts, box plot, correlation heatmap, short captions.
- **Pages:** Workforce Analysis, Hypotheses.

### **BR#2 (Predict)**

- **User story:** "As HR, I want to enter a profile and see the attrition probability."
- **Tasks:** build preprocessing + model pipeline, Train/Tune, export, Streamlit form for inputs, risk band.
- **Pages:** Attrition Predictor (ML), Technical.

## How to Use the Attrition Predictor

The **Attrition Predictor (ML)** page allows you to predict whether an employee is at risk of leaving by entering their profile information.

### **Step-by-Step Guide:**

1. **Navigate to the Predictor**

   - Open the dashboard
   - Click **"Attrition Predictor (ML)"** in the sidebar

2. **Enter Employee Information**
   The form has inputs for 15 features across two columns:

   **Numeric Fields** (use number inputs):

   - **Age**: Employee's age in years (e.g., 35)
   - **MonthlyIncome**: Monthly salary in currency units (e.g., 5000)
   - **DistanceFromHome**: Distance from home to work in miles/km (e.g., 10)
   - **TotalWorkingYears**: Total years of professional experience (e.g., 12)
   - **YearsAtCompany**: Years worked at current company (e.g., 5)
   - **NumCompaniesWorked**: Number of previous employers (e.g., 3)
   - **PercentSalaryHike**: Last salary increase percentage (e.g., 15)

   **Categorical Fields** (use dropdowns):

   - **OverTime**: Does the employee work overtime? (Yes/No)
   - **JobRole**: Current position (e.g., Sales Executive, Research Scientist)
   - **MaritalStatus**: Marital status (Single/Married/Divorced)
   - **BusinessTravel**: Travel frequency (Non-Travel/Travel_Rarely/Travel_Frequently)
   - **Department**: Working department (Sales/Research & Development/Human Resources)
   - **EducationField**: Field of education (e.g., Life Sciences, Medical, Technical Degree)
   - **Gender**: Employee gender (Male/Female)
   - **JobLevel**: Job level ranking (1-5, where 5 is highest)

3. **Get Prediction**
   - Click the **"Predict"** button at the bottom of the form
   - The system will display:
     - **Attrition Probability**: Percentage likelihood of leaving (e.g., 45.2%)
     - **Risk Band**: Classification into Low/Medium/High risk with visual indicator

### **Understanding the Results:**

**Risk Bands:**

- 🟢 **Low Risk** (< 35%): Employee unlikely to leave - maintain current engagement
- 🟡 **Medium Risk** (35-59%): Monitor closely - consider retention interventions
- 🔴 **High Risk** (≥ 60%): Immediate action needed - schedule retention discussion

**Example Interpretation:**

```
Attrition Probability: 62.5%
Risk Band: 🔴 High

Action: This employee shows strong indicators of leaving.
Consider: one-on-one meeting, career development discussion,
workload review, or compensation adjustment.
```

### **Tips for Best Results:**

- **Accurate Data**: Ensure all employee information is current and correct
- **Multiple Scenarios**: Try adjusting factors (e.g., reduce OverTime, increase salary) to see impact on risk
- **Regular Monitoring**: Re-run predictions quarterly or after major organizational changes
- **Combine with Insights**: Use the **Workforce Analysis** page to understand which factors matter most

## Hypothesis Validation Results

### **H1: OverTime workers leave more - SUPPORTED **

**Evidence:**

- Attrition rate for OverTime=Yes: ~30-31%
- Attrition rate for OverTime=No: ~10-11%
- **Gap: ~20 percentage points higher attrition for overtime workers**
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** There is strong statistical evidence that employees working overtime have significantly higher attrition rates. This hypothesis is **SUPPORTED**.

**Recommended Action:** Review overtime policies and workload distribution. Consider limiting mandatory overtime or providing additional compensation/time-off for overtime work.

---

### **H2: Lower JobSatisfaction increases attrition - SUPPORTED **

**Evidence:**

- JobSatisfaction level 1 (lowest): ~23% attrition rate
- JobSatisfaction level 4 (highest): ~11% attrition rate
- **Progressive decrease in attrition as satisfaction increases**
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** There is a clear inverse relationship between job satisfaction and attrition. Lower satisfaction levels correlate with higher attrition rates. This hypothesis is **SUPPORTED**.

**Recommended Action:** Implement regular satisfaction surveys, identify low-satisfaction teams, and develop targeted retention programs focusing on career development and work environment improvements.

---

### **H3: Younger employees (≤30) leave more often - SUPPORTED **

**Evidence:**

- Age ≤30: ~25% attrition rate
- Age >30: ~13% attrition rate
- **Gap: ~12 percentage points higher attrition for younger employees**
- Chi-square test: p < 0.001 (statistically significant)

**Conclusion:** Younger employees demonstrate significantly higher attrition rates compared to their older colleagues. This hypothesis is **SUPPORTED**.

**Recommended Action:** Develop early-career retention programs including mentorship, clear career progression paths, competitive compensation reviews, and engagement initiatives targeted at employees under 30.

---

### **Statistical Validation Summary**

All three hypotheses achieved statistical significance (p < 0.05) using chi-square tests of independence, providing strong evidence that the observed patterns are not due to random chance. The business should prioritize interventions addressing overtime practices, job satisfaction, and early-career retention.

## Modeling Summary

### **Approach**

Binary classification using **Random Forest** with a clean **ColumnTransformer**:

- **Numeric:** impute median + scale
- **Categorical:** impute most_frequent + one-hot (ignore unknowns)

### **Algorithms Evaluated**

We compared two classification algorithms to establish a strong baseline before hyperparameter tuning:

1. **Logistic Regression** (Baseline)

   - **Type:** Linear model with L2 regularization (max_iter=1000)
   - **ROC-AUC on test set:** ~0.76-0.78 (typical range)
   - **Strengths:** Fast training, interpretable coefficients, good baseline
   - **Limitations:** Assumes linear relationships between features and log-odds

2. **Random Forest** (Final Model - Selected)
   - **Type:** Ensemble of decision trees
   - **ROC-AUC on test set:** ~0.82-0.85 (after tuning)
   - **Why chosen:**
     - Better handles non-linear relationships
     - Captures complex feature interactions (e.g., Age × OverTime)
     - More robust to outliers
     - Provides feature importance rankings
   - **Trade-off:** Longer training time, less interpretable than LogReg

**Decision:** Random Forest selected as final model due to superior ROC-AUC performance and ability to capture complex patterns in employee behavior.

### **Tuning**

Grid search over ≥6 hyperparameters (each with ≥3 values) for Distinction:

- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`

### **Why these params**

They control model complexity (depth/leaves/split), ensemble size (`n_estimators`), feature sampling (`max_features`), and split measure (`criterion`).

## Model Performance

### **Primary metric**

ROC-AUC (goal ≥ 0.75).

### **Where to see it**

Technical page shows AUC and whether the goal is met.

### **Also provided**

- Saved ROC curve
- Saved confusion matrix @0.50
- Threshold sweep table (accuracy/precision/recall/F1)
- Live CM slider

## Versioned Artifacts

- Trained model and feature list live under `artifacts/v1/`.
- Images and tables for the app live under `assets/`.
- This keeps releases tidy and reproducible.

## Limitations & Next Steps

### **Limitations:**

- Dataset is small and generic; may not reflect your company.
- Features are limited; no text or time-series data.

### **Next:**

- Align thresholds with HR stakeholders (costs of FP/FN).
- Try other models (XGBoost, calibrated probabilities).
- Add monitoring (drift, weekly AUC).
- Add feature importance/explanations in the app.

## Manual Testing

### A. App starts

- **Run:** `streamlit run app.py`
- **Expect:** App loads with 5 pages in the sidebar (Summary, Analysis, Hypotheses, ML, Technical).

### B. Summary page

- Shows a small table preview of the dataset.
- Caption tells you which file was loaded (ready → processed → raw).
- If no files exist, it shows a friendly message telling you to run Notebook 01–02.

### C. Workforce Analysis (BR#1)

- Filters work (dropdowns don't clash).
- Charts render:
  - Bars by category (e.g., OverTime).
  - Box plot (e.g., Age vs Attrition).
  - Correlation heatmap (numeric).
- Each plot has a short caption explaining what to look for.

### D. Hypotheses

- H1 (OverTime) shows KPIs (rates + gap) and a clear callout.
- H2 (JobSatisfaction) and H3 (Age group) show rates and charts.
- If SciPy is installed: chi-square p-values are shown. If not, a friendly note appears.

### E. ML Predictor (BR#2)

- The form shows one input per feature (numbers → number input, categories → dropdowns).
- Clicking Predict returns:
  - Probability (e.g., 0.62 → 62%),
  - Risk band (Low/Medium/High) with an icon.

### F. Technical

- Shows ROC-AUC and whether the goal (≥ 0.75) is met.
- Shows saved ROC and Confusion Matrix images.
- Shows threshold table (with F1 highlight) and live confusion matrix with a slider.
- Shows pipeline steps and feature list.

## How to run:

"pytest -q"

- What's included:

1 - tests/test_smoke.py:

- Checks pytest runs at all (simple smoke test).

2 - tests/test_pages_import.py:
Imports each Streamlit page module to catch syntax/import errors early.

3 - tests/test_utils.py:
Tests the small helper that converts Yes/No → 1/0.

4 - tests/test_pipeline_fit_predict.py:
If data/processed/hr_attrition_ready.parquet exists: loads 200 rows, fits a tiny Logistic Regression pipeline, and checks it predicts 0/1 without error. (Skips automatically if the ready file isn't there yet.)

5: Test config
pytest.ini limits tests to the tests/ folder

## Error & Bug Fix Log (what went wrong and how we fixed it)

Below are the common issues we hit during build, with quick fixes.

### Kaggle

#### Symptom: `CalledProcessError` when running `kaggle datasets download`

- **Cause:** Missing Kaggle login or wrong path.
- **Fix:** Confirm `~/.kaggle/kaggle.json` exists, re-run the install + download cells, use `-p ../data/raw --unzip`.

### File paths & imports

#### Symptom: `FileNotFoundError: ../data/processed/hr_attrition_ready.parquet`

- **Cause:** Notebook 02 not run yet.
- **Fix:** Run NB02 to create the ready parquet.

#### Symptom: `NameError: Path is not defined`

- **Cause:** Missing `from pathlib import Path` in that cell.
- **Fix:** Import Path at the top of the cell.

#### Symptom: `ImportError: cannot import name 'RAW_CSV' from src.config`

- **Cause:** Config variable names didn't match.
- **Fix:** Standardized names in `src/config.py` and updated pages to import them.

#### Symptom: `ARTIFACTS_DIR` or `DATA_READY` not defined

- **Cause:** Inconsistent config during refactor.
- **Fix:** Use a single `config.py` with `ROOT`, `RAW_CSV`, `PROCESSED_PARQUET`, `READY_PARQUET`, `ARTIFACTS/V1` paths.

### Modeling & metrics

#### Symptom: `AttributeError: 'float' object has no attribute 'round'`

- **Cause:** Used `.round(3)` on a Python float.
- **Fix:** Use `round(value, 3)` or f-strings like `f"{value:.3f}"`.

#### Symptom: Grid search taking too long

- **Cause:** Big parameter grid.
- **Fix:** Two options:
  - Keep **big grid** (Distinction-level) and accept the wait.
  - Or use the **small grid** (faster) and mention it in README.

#### Symptom: AUC or confusion matrix not showing live

- **Cause:** Missing artifacts or data.
- **Fix:** Run **NB03** (exports model) and **NB02** (creates ready parquet), then reload the Technical page.

### Testing hiccups

#### Symptom: Pipeline test skipped

- **Cause:** Ready parquet not created yet.
- **Fix:** Run **NB02** to create `hr_attrition_ready.parquet`.

## Credits & Resources

### Machine Learning & Mathematics

- **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) - Official ML library documentation
- **Random Forest Algorithm:** [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) - Ensemble methods guide
- **ROC-AUC Explained:** [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

### Data Science Resources

- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) - Data manipulation and analysis
- **Seaborn Gallery:** [https://seaborn.pydata.org/examples/index.html](https://seaborn.pydata.org/examples/index.html) - Statistical data visualization
- **Kaggle Learn:** [https://www.kaggle.com/learn](https://www.kaggle.com/learn) - Free micro-courses on data science

### Streamlit & Deployment

- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/) - Interactive web app framework
- **Streamlit Gallery:** [https://streamlit.io/gallery](https://streamlit.io/gallery) - Community app examples

### Dataset

- **Original Dataset:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) - Kaggle dataset used in this project

### Special Thanks

- **Code Institute** - For the comprehensive Predictive Analytics and Machine Learning course
- **Kaggle Community** - For providing high-quality datasets and notebooks
- **Open Source Contributors** - For maintaining the amazing Python data science ecosystem
