"""Train, compare, evaluate and explain churn models.

Run:  python -m src.train --data data/telco_churn.csv
Outputs go to reports/ (metrics.json, plots) and models/ (best_model.joblib).

Optional libraries are used when installed (xgboost, shap, interpret,
imbalanced-learn, mlflow); the script still runs without them.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_data

SEED = 42

# Business assumptions for threshold tuning (change them and say so in the README):
COST_FALSE_NEGATIVE = 500.0  # a churner we missed: lost customer value (EUR)
COST_FALSE_POSITIVE = 50.0   # retention offer sent to a customer who would have stayed (EUR)


def optional(name):
    try:
        return __import__(name)
    except ImportError:
        return None


xgboost = optional("xgboost")
shap = optional("shap")
mlflow = optional("mlflow")
imblearn = optional("imblearn")


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_COLS),
                              ("cat", categorical, CATEGORICAL_COLS)])


def make_models(pos_weight: float) -> dict:
    """Class imbalance is handled with class weights (no rows invented)."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
    }
    if xgboost is not None:
        models["xgboost"] = xgboost.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=SEED)
    pipelines = {name: Pipeline([("prep", make_preprocessor()), ("model", m)])
                 for name, m in models.items()}

    # SMOTE variant: resampling sits INSIDE the pipeline, so it only touches
    # training folds and never the validation/test data (avoids leakage).
    if imblearn is not None:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipelines["logistic_regression_smote"] = ImbPipeline([
            ("prep", make_preprocessor()), ("smote", SMOTE(random_state=SEED)),
            ("model", LogisticRegression(max_iter=2000))])
    return pipelines


def best_threshold(y_true, proba) -> tuple[float, float]:
    """Pick the probability cut-off that minimises expected business cost."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        pred = proba >= t
        fn = np.sum((pred == 0) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def main(data_path: str, out_dir: str = "reports", model_dir: str = "models") -> dict:
    out, mdir = Path(out_dir), Path(model_dir)
    out.mkdir(exist_ok=True)
    mdir.mkdir(exist_ok=True)

    X, y = load_data(data_path)
    # Held-out test set, stratified so the churn rate is the same in both parts.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    results = {"n_rows": int(len(X)), "n_features_raw": int(X.shape[1]),
               "churn_rate": round(float(y.mean()), 4), "cv": {}}

    # ---- 1. Model comparison with stratified 5-fold cross-validation ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_auc = {}
    for name, pipe in make_models(pos_weight).items():
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv,
                                scoring=["roc_auc", "f1", "average_precision"])
        fold_auc[name] = scores["test_roc_auc"]
        results["cv"][name] = {
            "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
            "f1_mean": round(float(scores["test_f1"].mean()), 4),
            "pr_auc_mean": round(float(scores["test_average_precision"].mean()), 4),
        }
        print(f"{name:28s} CV ROC-AUC {scores['test_roc_auc'].mean():.4f} "
              f"± {scores['test_roc_auc'].std():.4f}")

    best_name = max(results["cv"], key=lambda n: results["cv"][n]["roc_auc_mean"])
    results["best_model"] = best_name

    # ---- 2. Paired statistical test: is the best model really better? ----
    # Paired t-test on per-fold AUCs (same folds for every model). Note: CV folds
    # overlap, so treat p-values as indicative, not exact.
    results["paired_ttest_vs_best"] = {}
    for name, aucs in fold_auc.items():
        if name == best_name:
            continue
        t, p = stats.ttest_rel(fold_auc[best_name], aucs)
        results["paired_ttest_vs_best"][name] = {"t": round(float(t), 3),
                                                 "p_value": round(float(p), 4)}

    # ---- 3. Final fit + held-out test evaluation ----
    final = make_models(pos_weight)[best_name].fit(X_tr, y_tr)
    proba = final.predict_proba(X_te)[:, 1]
    thr, cost = best_threshold(y_te.to_numpy(), proba)
    pred = (proba >= thr).astype(int)

    cost_contact_nobody = float(y_te.sum() * COST_FALSE_NEGATIVE)
    results["test"] = {
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "pr_auc": round(average_precision_score(y_te, proba), 4),
        "brier_score": round(brier_score_loss(y_te, proba), 4),
        "f1_at_0.5": round(f1_score(y_te, proba >= 0.5), 4),
        "business_threshold": round(thr, 2),
        "f1_at_business_threshold": round(f1_score(y_te, pred), 4),
        "precision_at_business_threshold": round(precision_score(y_te, pred), 4),
        "recall_at_business_threshold": round(recall_score(y_te, pred), 4),
        "expected_cost_model": cost,
        "expected_cost_no_model": cost_contact_nobody,
        "cost_saving_pct_vs_no_model": round(100 * (1 - cost / cost_contact_nobody), 1),
    }

    # Plots: confusion matrix + calibration curve
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Stay", "Churn"])
    plt.title(f"{best_name} @ threshold {thr:.2f}")
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    CalibrationDisplay.from_predictions(y_te, proba, n_bins=10, name=best_name)
    plt.savefig(out / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 4. Explainability ----
    prep, model = final.named_steps["prep"], final.named_steps["model"]
    feat_names = prep.get_feature_names_out()
    X_te_t = prep.transform(X_te)
    X_te_t = X_te_t.toarray() if hasattr(X_te_t, "toarray") else X_te_t

    if shap is not None:
        explainer = shap.Explainer(model, X_te_t, feature_names=feat_names)
        sv = explainer(X_te_t)
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
        shap.plots.beeswarm(sv, max_display=15, show=False)
        plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        imp = np.abs(sv.values).mean(axis=0)
        source = "mean_abs_shap"
    else:  # fallback so the script still works without shap
        from sklearn.inspection import permutation_importance
        r = permutation_importance(final, X_te, y_te, scoring="roc_auc",
                                   n_repeats=5, random_state=SEED)
        feat_names, imp, source = X_te.columns.to_numpy(), r.importances_mean, "permutation"
    top = np.argsort(imp)[::-1][:10]
    results["top_features"] = {"method": source,
                               "features": [str(feat_names[i]) for i in top]}

    # Glass-box benchmark: Explainable Boosting Machine (interpretable by design)
    if optional("interpret") is not None:
        from interpret.glassbox import ExplainableBoostingClassifier
        ebm = ExplainableBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
        results["glassbox_ebm_test_roc_auc"] = round(
            roc_auc_score(y_te, ebm.predict_proba(X_te)[:, 1]), 4)

    # ---- 5. Save + MLflow tracking ----
    joblib.dump({"pipeline": final, "threshold": thr}, mdir / "best_model.joblib")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    if mlflow is not None:
        mlflow.set_experiment("telco-churn")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_params({"model": best_name, "threshold": thr,
                               "cost_fn": COST_FALSE_NEGATIVE, "cost_fp": COST_FALSE_POSITIVE})
            mlflow.log_metrics({k: v for k, v in results["test"].items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(out))
            try:
                mlflow.sklearn.log_model(final, name='model', serialization_format='cloudpickle')
            except Exception as e:
                print('MLflow model logging skipped:', e)

    print(json.dumps(results["test"], indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    args = ap.parse_args()
    main(args.data)
