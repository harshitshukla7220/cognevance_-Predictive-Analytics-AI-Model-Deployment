from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.pipeline import add_features, build_preprocessor, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/customer_churn.csv"
MODEL_PATH = ROOT / "models/best_model.joblib"
REPORTS = ROOT / "reports"

def make_candidates(seed=42, fast=False):
    trees = 100 if fast else 250
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100 if fast else 180, learning_rate=0.05, max_depth=2, random_state=seed),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=150 if fast else 250, random_state=seed)
    }

def train(data_path=DATA, fast=False, cv_folds=3):
    REPORTS.mkdir(exist_ok=True, parents=True); MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET]).copy()
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    X_raw = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    X = add_features(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Fit preprocessing inside each CV fold to avoid leakage.
    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    fitted = {}
    for name, estimator in make_candidates(fast=fast).items():
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        started = time.time()
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            row = {
                "model": name, "cv_pr_auc_mean": float(scores.mean()), "cv_pr_auc_std": float(scores.std()),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "pr_auc": float(average_precision_score(y_test, proba)),
                "mcc": float(matthews_corrcoef(y_test, pred)),
                "training_seconds": round(time.time()-started, 3)
            }
            results.append(row); fitted[name] = (pipe, proba)
            print(f"{name}: accuracy={row['accuracy']:.4f}, PR-AUC={row['pr_auc']:.4f}")
        except Exception as exc:
            print(f"WARNING: {name} skipped: {exc}")
    if not results:
        raise RuntimeError("No model trained successfully.")
    results.sort(key=lambda r: r["cv_pr_auc_mean"], reverse=True)
    best_name = results[0]["model"]
    best_pipe, best_proba = fitted[best_name]
    # Threshold is selected on the training set only using out-of-fold probabilities.
    from sklearn.model_selection import cross_val_predict
    best_estimator = make_candidates(fast=fast)[best_name]
    cv_pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", best_estimator)])
    train_oof = cross_val_predict(cv_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thresholds = np.linspace(0.2, 0.8, 61)
    threshold = float(max(thresholds, key=lambda t: f1_score(y_train, train_oof >= t, zero_division=0)))
    final_pred = (best_proba >= threshold).astype(int)
    final_metrics = {
        "accuracy": float(accuracy_score(y_test, final_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, final_pred)),
        "precision": float(precision_score(y_test, final_pred, zero_division=0)),
        "recall": float(recall_score(y_test, final_pred, zero_division=0)),
        "f1": float(f1_score(y_test, final_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, best_proba)),
        "pr_auc": float(average_precision_score(y_test, best_proba)),
        "mcc": float(matthews_corrcoef(y_test, final_pred)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, final_pred).tolist(),
        "classification_report": classification_report(y_test, final_pred, target_names=["No Churn", "Churn"], output_dict=True, zero_division=0)
    }
    bundle = {
        "pipeline": best_pipe, "model_name": best_name, "threshold": threshold,
        "feature_columns": X_raw.columns.tolist(), "target": TARGET,
        "positive_label": "Yes", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    }
    joblib.dump(bundle, MODEL_PATH)
    payload = {
        "project": "Customer Churn Prediction - Predictive Analytics & AI Model Deployment",
        "dataset": str(Path(data_path).name), "rows": int(len(df)), "features_before_engineering": int(X_raw.shape[1]),
        "positive_rate": float(y.mean()), "selection_metric": "mean cross-validation PR-AUC",
        "best_model": best_name, "final_metrics": final_metrics, "models": results,
        "feature_engineering": ["tenure_years", "tenure_group", "avg_charge_per_month", "number_of_services",
                                "is_month_to_month", "uses_electronic_check", "has_internet"],
        "dataset_sha256": bundle["dataset_sha256"]
    }
    (REPORTS/"model_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = ["# Customer Churn Model Evaluation Report", "",
        f"- Dataset rows: **{len(df):,}**",
        f"- Input columns before engineering: **{X_raw.shape[1]}**",
        f"- Best model: **{best_name}**",
        f"- Selection: **{payload['selection_metric']}**",
        f"- Decision threshold: **{threshold:.3f}**", "",
        "## Model comparison", "",
        "| Model | CV PR-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['model']} | {r['cv_pr_auc_mean']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |")
    report += ["", "## Best model held-out test metrics", "", "```json", json.dumps(final_metrics, indent=2), "```",
               "", "## Notes", "", "The hold-out test set is used only for final evaluation. Model selection uses cross-validation on the training split. Metrics describe this dataset and split; they are not a guarantee of production performance."]
    (REPORTS/"model_report.md").write_text("\n".join(report), encoding="utf-8")
    # Confusion matrix image
    cm = np.array(final_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(cm)
    ax.set_title("Customer Churn - Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1], ["No Churn","Churn"]); ax.set_yticks([0,1], ["No Churn","Churn"])
    for (i,j), v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); fig.savefig(ROOT/"images/confusion_matrix.png", dpi=150); plt.close(fig)
    print(f"\nBest model: {best_name}; threshold={threshold:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved reports: {REPORTS}")
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    train(args.data, args.fast, args.cv_folds)
