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

## Hypotheses & Validation

### **H1**
- **Hypothesis:** OverTime=Yes shows higher attrition.
- **Evidence:** higher rate in table/plot; positive gap KPI; (optional) chi-square p < 0.05.

### **H2**
- **Hypothesis:** Lower JobSatisfaction relates to higher attrition.
- **Evidence:** rising rates at low satisfaction levels; (optional) chi-square p < 0.05.

### **H3**
- **Hypothesis:** Younger group (≤30) shows higher attrition.
- **Evidence:** higher rate for ≤30 group; (optional) chi-square p < 0.05.

*(Exact p-values appear on the Hypotheses page if SciPy is installed.)*

## Modeling Summary

### **Approach**
Binary classification using **Random Forest** with a clean **ColumnTransformer**:
- **Numeric:** impute median + scale
- **Categorical:** impute most_frequent + one-hot (ignore unknowns)

### **Baselines**
Compared Logistic Regression vs Random Forest.

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
* Trained model and feature list live under `artifacts/v1/`.
* Images and tables for the app live under `assets/`.
* This keeps releases tidy and reproducible.

## Limitations & Next Steps

### **Limitations:**
* Dataset is small and generic; may not reflect your company.
* Features are limited; no text or time-series data.

### **Next:**
* Align thresholds with HR stakeholders (costs of FP/FN).
* Try other models (XGBoost, calibrated probabilities).
* Add monitoring (drift, weekly AUC).
* Add feature importance/explanations in the app.

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
* **Cause:** Missing Kaggle login or wrong path.
* **Fix:** Confirm `~/.kaggle/kaggle.json` exists, re-run the install + download cells, use `-p ../data/raw --unzip`.

### File paths & imports

#### Symptom: `FileNotFoundError: ../data/processed/hr_attrition_ready.parquet`
* **Cause:** Notebook 02 not run yet.
* **Fix:** Run NB02 to create the ready parquet.

#### Symptom: `NameError: Path is not defined`
* **Cause:** Missing `from pathlib import Path` in that cell.
* **Fix:** Import Path at the top of the cell.

#### Symptom: `ImportError: cannot import name 'RAW_CSV' from src.config`
* **Cause:** Config variable names didn't match.
* **Fix:** Standardized names in `src/config.py` and updated pages to import them.

#### Symptom: `ARTIFACTS_DIR` or `DATA_READY` not defined
* **Cause:** Inconsistent config during refactor.
* **Fix:** Use a single `config.py` with `ROOT`, `RAW_CSV`, `PROCESSED_PARQUET`, `READY_PARQUET`, `ARTIFACTS/V1` paths.

### Modeling & metrics

#### Symptom: `AttributeError: 'float' object has no attribute 'round'`
* **Cause:** Used `.round(3)` on a Python float.
* **Fix:** Use `round(value, 3)` or f-strings like `f"{value:.3f}"`.

#### Symptom: Grid search taking too long
* **Cause:** Big parameter grid.
* **Fix:** Two options:
  * Keep **big grid** (Distinction-level) and accept the wait.
  * Or use the **small grid** (faster) and mention it in README.

#### Symptom: AUC or confusion matrix not showing live
* **Cause:** Missing artifacts or data.
* **Fix:** Run **NB03** (exports model) and **NB02** (creates ready parquet), then reload the Technical page.

### Testing hiccups

#### Symptom: Pipeline test skipped
* **Cause:** Ready parquet not created yet.
* **Fix:** Run **NB02** to create `hr_attrition_ready.parquet`.

## Credits & Resources


### Machine Learning & Mathematics
* **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) - Official ML library documentation
* **Random Forest Algorithm:** [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) - Ensemble methods guide
* **ROC-AUC Explained:** [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) 

### Data Science Resources
* **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) - Data manipulation and analysis
* **Seaborn Gallery:** [https://seaborn.pydata.org/examples/index.html](https://seaborn.pydata.org/examples/index.html) - Statistical data visualization
* **Kaggle Learn:** [https://www.kaggle.com/learn](https://www.kaggle.com/learn) - Free micro-courses on data science

### Streamlit & Deployment
* **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/) - Interactive web app framework
* **Streamlit Gallery:** [https://streamlit.io/gallery](https://streamlit.io/gallery) - Community app examples

### Dataset
* **Original Dataset:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) - Kaggle dataset used in this project


### Special Thanks
* **Code Institute** - For the comprehensive Predictive Analytics and Machine Learning course
* **Kaggle Community** - For providing high-quality datasets and notebooks
* **Open Source Contributors** - For maintaining the amazing Python data science ecosystem


## How to use this repo

1. Use this template to create your GitHub project repo

1. In your newly created repo click on the green Code button. 

1. Then, from the Codespaces tab, click Create codespace on main.

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Open the jupyter_notebooks directory, and click on the notebook you want to open.

1. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.12.1 as it inherits from the workspace, so it will be Python-3.12.1 as installed by Codespaces. To confirm this, you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked


You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


## Dataset Content
* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size and to have a shorter model training time. If you are doing an image recognition project, we suggest you consider using an image shape that is 100px × 100px or 50px × 50px, to ensure the model meets the performance requirement but is smaller than 100Mb for a smoother push to GitHub. A reasonably sized image set is ~5000 images, but you can choose ~10000 lines for numeric or textual data. 


## Business Requirements
* Describe your business requirements


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them) 


## The rationale to map the business requirements to the Data Visualizations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks


## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://attrisight-404abf5f2e34.herokuapp.com/

### 1. Login to Heroku:

heroku login
heroku create app

### 3. Connect GitHub:
* Go to Heroku Dashboard
* Select your app
* Deploy tab → Connect to GitHub
* Search for your repository
* Click Connect

### 4. Deploy:
* Select branch: main
* Click "Deploy Branch"
* Watch build logs for errors




