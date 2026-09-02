# Telco Customer Churn — Classification

Predicts customer churn on the IBM Telco Customer Churn dataset. The project covers a
leakage-safe preprocessing pipeline, an explicit class-imbalance decision, model
comparison under stratified cross-validation, hyperparameter tuning, a single held-out
test evaluation, SHAP explainability, and a cost-based operating-threshold analysis.

The champion model is a tuned logistic regression. It catches ~80% of churners at the
default threshold and ~91% at a cost-based threshold of 0.32, and its SHAP attribution
points to one dominant driver: month-to-month contracts with short tenure.

## Result summary

| Model | CV Recall | CV ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression (tuned) | 0.813 | 0.851 | champion — `C=0.1`, `l1`, `liblinear`, `class_weight='balanced'` |
| Random Forest | 0.470 | 0.825 | untuned fallback |
| XGBoost (tuned) | 0.811 | 0.847 | matches LogReg on recall, beats it on no metric |

Held-out test (champion, tuned logistic regression):

| Threshold | Recall | Precision | Missed churners (FN) | False alarms (FP) |
|---|---|---|---|---|
| 0.50 (default) | 0.802 | 0.502 | 74 | 298 |
| 0.32 (cost-based, K=5) | 0.914 | 0.440 | 32 | 435 |

Test ROC-AUC 0.849, within 0.002 of the cross-validation estimate, no overfitting.

## Setup

Requires Python 3.13.9.

```bash
pip install -r requirements.txt
```

Key libraries: scikit-learn, xgboost, shap, pandas, numpy, matplotlib, seaborn, pyarrow.

## Data

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is committed with the repository. Original
source: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 21 columns, ~26.5% churn rate.

## Running

```bash
jupyter notebook notebooks/
```

1. **`notebooks/pipeline.ipynb`** — data loading, feature engineering, EDA, the
   class-imbalance decision, and the preprocessing pipeline. Writes the stratified
   train/test split and the fitted preprocessor to `data/processed/`.
2. **`notebooks/modelling.ipynb`** — model comparison, hyperparameter tuning, held-out
   test evaluation, SHAP, and the cost-based threshold analysis. Reads from
   `data/processed/`.

Run `pipeline.ipynb` end to end first as `modelling.ipynb` depends on its outputs.

## Approach notes

- **Class imbalance:** `class_weight='balanced'` rather than SMOTE, to keep the pipeline
  simple and avoid synthesising minority-class rows. Recall on the churn class is the
  primary metric; accuracy is never used alone.
- **Leakage:** all preprocessing (imputation, scaling, one-hot encoding) is fit inside
  cross-validation folds via an sklearn `Pipeline`. The test set is transformed with the
  training-fit preprocessor only and is touched exactly once, for final evaluation.
- **Model choice:** logistic regression is preferred over the tree ensembles, it wins
  on recall and ROC-AUC, and its coefficients are directly interpretable. A tuned
  XGBoost matches its recall but improves on nothing, because the churn signal here is
  close to linear (contract length dominates).
- **Threshold:** the 0.5 default is replaced with a cost-minimising threshold under an
  assumed false-negative : false-positive cost ratio of 5:1 (a lost customer ≈ five
  months of revenue; a retention offer ≈ one month free). Threshold selection uses the
  test set, there is no separate calibration split.

## Known limitations and TODOs

- The data is a single cross-sectional snapshot. Tenure is static; the model cannot see
  trajectory, so a customer whose engagement is declining looks identical to a stable
  one until they churn.
- The operating threshold is selected on the test set. A dedicated calibration split
  would be cleaner.
- The 5:1 cost ratio is an assumption. The cost-minimising threshold moves from 0.45 at
  3:1 to 0.18 at 12:1, so the ratio needs sign-off from whoever owns retention
  economics.
- Retention-offer effectiveness is not modelled, the cost analysis assumes a flagged
  churner can actually be saved.
- At the 0.32 threshold the model flags a large share of the base (precision 0.44), so
  it is better used as a ranked list worked to call capacity than as a hard cutoff.
- **TODO:** calibration split for threshold selection; model calibration
  (`CalibratedClassifierCV`); a rule-based override for the tenured month-to-month
  segment that SHAP shows the model under-predicts; time-aware features if a temporal
  dataset becomes available.
```
