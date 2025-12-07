# Models/models_svm.py
import os
import json
import time
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# Helper: evaluate metrics safely
# ============================================================

def evaluate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all classification metrics with correct handling
    for string labels ('nasal', 'oral').
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError("Binary classification expected.")

    pos_label = classes[1]  # 'oral'
    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }

    # Probability-based metrics
    if y_prob is not None:
        if isinstance(y_prob, np.ndarray) and y_prob.ndim == 2:
            y_score = y_prob[:, 1]  # positive class
        else:
            y_score = np.asarray(y_prob)

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_score))
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_bin, y_score))
        except Exception:
            metrics["pr_auc"] = None

    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


# ============================================================
# Generic SVM grid search wrapper
# ============================================================

def run_grid_svm(X_train, y_train, X_test, y_test,
                 param_grid: dict,
                 svc_ctor,
                 model_name: str,
                 cv_inner: int = 3,
                 scoring: str = "balanced_accuracy",
                 save_folder: str = None) -> Dict[str, Any]:

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    estimator = svc_ctor()

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_inner,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    start = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - start

    best = gs.best_estimator_
    best_params = gs.best_params_
    best_cv_score = gs.best_score_

    y_pred = best.predict(X_test)

    # probabilities
    y_prob = None
    if hasattr(best, "predict_proba"):
        try:
            y_prob = best.predict_proba(X_test)
        except:
            pass
    elif hasattr(best, "decision_function"):
        dec = best.decision_function(X_test)
        if dec.ndim == 1:
            # min-max normalisation as fallback
            y_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
        else:
            y_prob = dec

    metrics = evaluate_metrics(y_test, y_pred, y_prob)

    result = {
        "model_name": model_name,
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "test_metrics": metrics,
        "fit_time_sec": float(elapsed),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": None if y_prob is None else y_prob.tolist(),
    }

    if save_folder is not None:
        json_path = os.path.join(save_folder, f"{model_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        joblib.dump(best, os.path.join(save_folder, f"{model_name}_best_model.joblib"))
        joblib.dump(gs, os.path.join(save_folder, f"{model_name}_gridsearch.joblib"))

    return result


# ============================================================
# Specific SVM tuners
# ============================================================

def svm_linear_tuned(X_train, y_train, X_test, y_test, cv_inner=3, save_folder=None):
    grid = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
    return run_grid_svm(X_train, y_train, X_test, y_test,
                        param_grid=grid,
                        svc_ctor=lambda: SVC(kernel="linear", probability=True),
                        model_name="svm_linear",
                        cv_inner=cv_inner,
                        save_folder=save_folder)


def svm_rbf_tuned(X_train, y_train, X_test, y_test, cv_inner=3, save_folder=None):
    grid = {
        "C": [0.01, 0.1, 1, 10, 100, 1000],
        "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]
    }
    return run_grid_svm(X_train, y_train, X_test, y_test,
                        param_grid=grid,
                        svc_ctor=lambda: SVC(kernel="rbf", probability=True),
                        model_name="svm_rbf",
                        cv_inner=cv_inner,
                        save_folder=save_folder)


def svm_poly_tuned(X_train, y_train, X_test, y_test, cv_inner=3, save_folder=None):
    grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4, 5]
    }
    return run_grid_svm(X_train, y_train, X_test, y_test,
                        param_grid=grid,
                        svc_ctor=lambda: SVC(kernel="poly", probability=True),
                        model_name="svm_poly",
                        cv_inner=cv_inner,
                        save_folder=save_folder)
