# Insurance Premium Risk Classification and Customer Segmentation

End-to-end tabular machine-learning pipeline that classifies insurance applicants into three premium-risk classes (Low / Standard / High), with ablations, interpretability and unsupervised clustering. Individual coursework for XJTLU DTS304TC (Machine Learning), April-May 2026. 保险风险三分类与客户聚类（课程个人作业）。

All work is in one notebook: [`insurance_risk_classification.ipynb`](insurance_risk_classification.ipynb) (outputs stripped; re-run to regenerate figures).

## Data
Course-provided dataset split as 74,375 train / 13,125 validation / 12,500 unlabeled test rows (`dataset_final/`). **The data is not included.** Class balance on the training set: Standard 53.4%, High 29.0%, Low 17.6%.

## What the notebook does
1. **Cleaning and leakage control.** Drops administrative / identifier columns and noise features; mutual information flags `bureau_risk_index` (MI about 1.0) as a leakage feature and it is removed; garbled spending-profile strings are set to `Unknown`, account tenure is parsed from free text, product lists are converted to counts, negative financial values are clipped.
2. **Baselines.** Logistic regression, Random Forest and XGBoost with a shared preprocessing pipeline.
3. **Hyper-parameter search.** Optuna TPE, 100 trials, 8-dimensional XGBoost space.
4. **Ablations.** Feature engineering (log transforms, interactions, smoothed target encoding); imbalance handling (class weights, SMOTE, threshold tuning).
5. **Interpretability and error analysis.** SHAP (global and per-class), per-class error rates, SHAP differences for misclassified samples.
6. **Unsupervised exploration.** PCA to 10 components (59.2% variance) followed by K-Means and GMM.

## Results (validation set, macro-F1)

| Stage | Macro-F1 |
|---|---|
| Logistic regression baseline | 0.6013 |
| Random Forest | 0.7985 |
| XGBoost (default) | 0.8384 |
| XGBoost + Optuna | 0.8843 |
| + log + interaction + target encoding (best feature config) | 0.8913 |
| + threshold tuning (best overall) | 0.8917 |

- Top SHAP features: late-payment count, total outstanding debt, monthly instalment total, gross annual income.
- 1,351 of 13,125 validation samples misclassified (10.3%), mostly Low <-> Standard confusion.
- SMOTE did not help (it lowered Low-class F1); class weights and threshold tuning gave only marginal changes.
- K-Means and GMM agree on k = 3; clusters only partly align with the risk labels, so supervised learning is needed.

The validation set was also used to choose the final configuration, so these are validation (not held-out test) figures.

## Run
```bash
pip install pandas numpy scikit-learn xgboost optuna shap imbalanced-learn matplotlib seaborn
# place train.csv, val.csv, test_features.csv in dataset_final/
jupyter notebook insurance_risk_classification.ipynb
```
Not re-run when preparing this repository; the XGBoost cells request `device='cuda'` and can be switched to CPU.

## Tech stack
Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn, Matplotlib / Seaborn.
