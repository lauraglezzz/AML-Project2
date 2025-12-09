import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time   # ⏱️ TIMER ADDED HERE

# Start timer
start_time = time.time()

# Add project root
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# Import modules
from Preprocessing.preprocess import load_phoneme_data, prepare_fold_data
from Models.dimensionality import run_pca, run_fda, run_kpca_tuned, run_kfda
from Models.models_svm import (
    svm_linear_tuned, svm_rbf_tuned, svm_poly_tuned
)

# ===============================================================
# Settings
# ===============================================================

RESULTS_DIR = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = load_phoneme_data()
print("\n=== Nested CV with MULTIPLE SVM kernels ===\n")

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

summary_records = {"pca": [], "fda": [], "kpca": [], "kfda": []}

SVM_MODELS = {
    "linear": svm_linear_tuned,
    "rbf": svm_rbf_tuned,
    "poly": svm_poly_tuned
}

# ===============================================================
# Helper to evaluate all kernels
# ===============================================================

def evaluate_all_svms(Xtr, ytr, Xte, yte, base_folder):
    results = {}
    for name, tuner in SVM_MODELS.items():
        print(f"   → Running SVM {name.upper()}")
        folder = os.path.join(base_folder, f"SVM_{name.upper()}")
        os.makedirs(folder, exist_ok=True)
        try:
            res = tuner(Xtr, ytr, Xte, yte, cv_inner=3, save_folder=folder)
            results[name] = res
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = {"error": str(e)}
    return results


# ===============================================================
# LOOP OVER FOLDS — we store ALL kernels temporarily
# ===============================================================

all_results = { "pca": [], "fda": [], "kpca": [], "kfda": [] }

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
    print(f"\n----- Fold {fold_idx}/{K} -----")

    fold_folder = os.path.join(RESULTS_DIR, f"Fold{fold_idx}")
    os.makedirs(fold_folder, exist_ok=True)

    # preprocessing
    fold_data = prepare_fold_data(df, train_idx, test_idx)
    Xtrain, ytrain = fold_data["X_train"], fold_data["y_train"]
    Xtest,  ytest = fold_data["X_test"],  fold_data["y_test"]

    # ---------------- PCA ----------------
    Xtr_pca, Xte_pca, _ = run_pca(Xtrain, Xtest, n_components=2)
    pca_folder = os.path.join(fold_folder, "PCA")
    pca_results = evaluate_all_svms(Xtr_pca, ytrain, Xte_pca, ytest, pca_folder)
    pca_results["X_test_proj"] = Xte_pca.tolist()
    all_results["pca"].append(pca_results)

    # ---------------- FDA ----------------
    Xtr_fda, Xte_fda, _ = run_fda(Xtrain, ytrain, Xtest, n_components=2)
    fda_folder = os.path.join(fold_folder, "FDA")
    fda_results = evaluate_all_svms(Xtr_fda, ytrain, Xte_fda, ytest, fda_folder)
    fda_results["X_test_proj"] = Xte_fda.tolist()
    all_results["fda"].append(fda_results)

    # ---------------- kPCA ----------------
    Xtr_kpca, Xte_kpca, kpca_model, best_gamma, score = run_kpca_tuned(
        Xtrain, ytrain, Xtest, gamma_grid=[0.001,0.01,0.1,1,10], n_components=2
    )
    kpca_folder = os.path.join(fold_folder, "KPCA")
    kpca_results = evaluate_all_svms(Xtr_kpca, ytrain, Xte_kpca, ytest, kpca_folder)
    kpca_results["X_test_proj"] = Xte_kpca.tolist()
    all_results["kpca"].append(kpca_results)

    # ---------------- kFDA ----------------
    Xtr_kfda, Xte_kfda, _ = run_kfda(Xtrain, ytrain, Xtest, n_components=2)
    kfda_folder = os.path.join(fold_folder, "KFDA")
    kfda_results = evaluate_all_svms(Xtr_kfda, ytrain, Xte_kfda, ytest, kfda_folder)
    kfda_results["X_test_proj"] = Xte_kfda.tolist()
    all_results["kfda"].append(kfda_results)



# ===============================================================
# NOW SELECT THE BEST GLOBAL SVM PER TECHNIQUE
# ===============================================================

def pick_best_global(results_list):
    """
    results_list = list of dicts:
       { "linear": {...}, "rbf": {...}, "poly": {...}, "X_test_proj": ... }
    """
    kernel_scores = { "linear": [], "rbf": [], "poly": [] }

    # collect BA per fold for each kernel
    for fold in results_list:
        for k in ["linear", "rbf", "poly"]:
            res = fold.get(k, None)
            if res and "test_metrics" in res:
                ba = res["test_metrics"].get("balanced_accuracy")
                kernel_scores[k].append(ba)

    # compute global mean
    means = { k: np.nanmean(v) for k, v in kernel_scores.items() }

    # pick best
    best_kernel = max(means, key=lambda x: means[x])
    print(f"\nBEST SVM for this technique = {best_kernel.upper()} with mean BA = {means[best_kernel]:.4f}")
    return best_kernel


# build final summary_records
final_summary = { "pca": [], "fda": [], "kpca": [], "kfda": [] }

for technique in ["pca", "fda", "kpca", "kfda"]:
    print(f"\nSelecting best kernel for {technique.upper()}")
    best_k = pick_best_global(all_results[technique])

    for fold in all_results[technique]:
        best_result = fold[best_k]
        final_summary[technique].append({
            "best_kernel": best_k,
            "test_metrics": best_result["test_metrics"],
            "y_true": best_result["y_true"],
            "y_pred": best_result["y_pred"],
            "y_prob": best_result["y_prob"],
            "X_test_proj": fold["X_test_proj"]
        })

# save summary
summary_dir = os.path.join(RESULTS_DIR, "Summary")
os.makedirs(summary_dir, exist_ok=True)

with open(os.path.join(summary_dir, "aggregated_results.json"), "w") as f:
    json.dump(final_summary, f, indent=2)

print("\n=== DONE — best SVM per technique selected and saved ===")

# ============================
# PRINT TOTAL EXECUTION TIME
# ============================

total_time = time.time() - start_time
print(f"=== Total execution time: {total_time/60:.2f} minutes ({total_time:.1f} seconds) ===")
