"""
Supervised Classification
==========================
Predicts bird attributes (sex, breeding status, network role) from
graph metrics and/or node2vec embeddings.

Models
------
    Logistic Regression  — fast baseline, interpretable coefficients
    Random Forest        — handles non-linear structure, feature importance
    SVM (RBF kernel)     — strong on high-dim embedding features

Evaluation
----------
    Stratified k-fold CV, ROC-AUC (binary & macro-OvR), confusion matrix,
    per-class precision/recall/F1.

Typical usage
-------------
    from ml.features import build_feature_matrix
    from ml.classification import train_and_evaluate

    X, names, meta = build_feature_matrix(2016, "SPRA", "daytime")
    results = train_and_evaluate(X, meta["sex"], feature_names=names)
    print(results["summary"])
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE, CV_FOLDS


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_models(random_state: int = RANDOM_STATE) -> Dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def _evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    label_names: List[str],
) -> Dict:
    """
    Evaluate a model via cross-validated predictions.
    Returns accuracy, ROC-AUC, confusion matrix, classification report.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    # ROC-AUC — needs probability estimates
    try:
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        if len(label_names) == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        auc = round(float(auc), 4)
    except Exception:
        auc = np.nan

    report = classification_report(
        y, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "roc_auc":              auc,
        "accuracy":             round(report["accuracy"], 4),
        "macro_f1":             round(report["macro avg"]["f1-score"], 4),
        "classification_report": report,
        "confusion_matrix":     cm,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y_series: pd.Series,
    feature_names: Optional[List[str]] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    models: Optional[Dict] = None,
) -> Dict:
    """
    Run stratified k-fold CV for all models on a single target.

    Args:
        X:            feature matrix (n_birds, n_features)
        y_series:     target Series (bird_id-aligned); NaN rows are dropped
        feature_names: used for feature importance output
        cv_folds:     number of CV folds
        models:       dict of {name: sklearn estimator}; defaults to all three

    Returns:
        dict with keys per model name + a 'summary' DataFrame row
    """
    # Drop rows where label is missing
    mask  = y_series.notna()
    X_    = X[mask.values]
    y_raw = y_series[mask]

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y_raw)
    labels = list(le.classes_)

    n_samples = len(y_enc)
    n_classes = len(labels)

    if n_samples < cv_folds * n_classes:
        raise ValueError(
            f"Too few labelled samples ({n_samples}) for {cv_folds}-fold CV "
            f"with {n_classes} classes."
        )

    cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models = models or _get_models(random_state)

    results = {"label_names": labels, "n_samples": n_samples}
    rows    = []

    for name, model in models.items():
        print(f"  [{name}] fitting…")
        try:
            r = _evaluate_model(model, X_, y_enc, cv, labels)
            results[name] = r
            rows.append({
                "model":     name,
                "roc_auc":   r["roc_auc"],
                "accuracy":  r["accuracy"],
                "macro_f1":  r["macro_f1"],
            })
        except Exception as e:
            print(f"    [error] {name}: {e}")

    results["summary"] = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    # Feature importance (Random Forest only)
    if "random_forest" in models and feature_names:
        rf = models["random_forest"]
        rf.fit(X_, y_enc)
        fi = pd.DataFrame({
            "feature":   feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        results["feature_importance"] = fi

    return results


# ── Multi-target convenience ───────────────────────────────────────────────────

def evaluate_all_targets(
    X: np.ndarray,
    meta: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Run train_and_evaluate for each available target column in meta.
    Returns a dict keyed by target name.
    """
    if target_cols is None:
        target_cols = [c for c in ("sex", "breeding_status", "network_position")
                       if c in meta.columns]

    all_results = {}
    for col in target_cols:
        if meta[col].notna().sum() < 10:
            print(f"  [skip] {col}: too few labelled samples")
            continue
        print(f"\n── Target: {col} ──")
        try:
            all_results[col] = train_and_evaluate(
                X, meta[col], feature_names=feature_names
            )
        except Exception as e:
            print(f"  [error] {col}: {e}")

    return all_results
