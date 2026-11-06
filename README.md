# Explainable Customer Churn Prediction

Predicts which telecom customers are likely to cancel, explains *why*, and turns the
predictions into a cost-optimised retention decision.

**Covers:** complex tabular data · leakage-safe preprocessing · class imbalance ·
model comparison with cross-validation and a paired statistical test · ROC-AUC / PR-AUC /
F1 / calibration · business-cost threshold · SHAP + glass-box EBM · MLflow · FastAPI · pytest + GitHub Actions CI.

## Data
IBM Telco Customer Churn (7,043 customers, 19 features, ~26% churn).
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or
[IBM's GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data),
and save it as `data/telco_churn.csv`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --data data/telco_churn.csv     # ~2-5 min
pytest -q                                           # tests
mlflow ui                                           # optional: browse runs at http://127.0.0.1:5000
uvicorn src.api:app --reload                        # optional: API at http://127.0.0.1:8000/docs
```
Results: `reports/metrics.json`, `reports/confusion_matrix.png`,
`reports/calibration_curve.png`, `reports/shap_summary.png`.

## Method
1. **Cleaning**: `TotalCharges` text → numeric (blanks for tenure-0 customers become NaN);
   `customerID` dropped (an identifier invites memorisation).
2. **Leakage prevention**: an 80/20 stratified split is made first. Imputation, scaling,
   one-hot encoding and SMOTE all live *inside* the sklearn/imblearn pipeline, so they are
   fitted only on training folds.
3. **Imbalance**: class weights (`class_weight="balanced"`, XGBoost `scale_pos_weight`)
   and a SMOTE variant for comparison.
4. **Model comparison**: logistic regression, random forest, XGBoost and LR+SMOTE with stratified
   5-fold CV, plus a paired t-test on per-fold ROC-AUC against the best model.
5. **Quality assessment on the held-out test set**: ROC-AUC, PR-AUC, Brier score, F1,
   confusion matrix, calibration curve.
6. **Business threshold**: chooses the cut-off that minimises
   `500 € × missed churners + 50 € × unnecessary offers` (assumptions in `src/train.py`).
7. **Explainability**: SHAP beeswarm (global + per-customer) and a glass-box Explainable
   Boosting Machine as an interpretable benchmark.

## Results (fill in from reports/metrics.json after your run)
| Metric | Value |
|---|---|
| Best model (CV) | `best_model` |
| CV ROC-AUC | `cv.<best>.roc_auc_mean` ± `roc_auc_std` |
| Test ROC-AUC | `test.roc_auc` |
| Test F1 @ business threshold | `test.f1_at_business_threshold` |
| Cost saving vs. no model | `test.cost_saving_pct_vs_no_model` % |
| EBM (glass-box) ROC-AUC | `glassbox_ebm_test_roc_auc` |
| Top churn drivers | `top_features` |
