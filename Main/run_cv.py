# Main/run_cv.py
import os
import json
import pandas as pd
from sklearn.model_selection import KFold

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocessing.preprocess import load_phoneme_data, prepare_fold_data
from Models.dimensionality import run_pca, run_fda, run_kpca_tuned, run_kfda
from Models.models_svm import svm_rbf_tuned, svm_linear_tuned, svm_poly_tuned


# ===============================================================
# Create results folder
# ===============================================================

RESULTS_DIR = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)



# ===============================================================
# Load dataset
# ===============================================================

df = load_phoneme_data()
print("\n=== Starting Nested CV (preprocessing + dimensionality reduction + SVM) ===\n")


# ===============================================================
# Outer CV
# ===============================================================

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

summary_records = {
    "pca": [],
    "fda": [],
    "kpca": [],
    "kfda": []
}


# ===============================================================
# Loop over folds
# ===============================================================

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), start=1):

    print(f"\n----- Fold {fold_idx}/{K} -----")

    fold_folder = os.path.join(RESULTS_DIR, f"Fold{fold_idx}")
    os.makedirs(fold_folder, exist_ok=True)

    # ---------------- PREPROCESSING -----------------
    fold_data = prepare_fold_data(df, train_idx, test_idx)
    Xtrain, ytrain = fold_data["X_train"], fold_data["y_train"]
    Xtest, ytest   = fold_data["X_test"], fold_data["y_test"]

    print(f"Train: {Xtrain.shape}   Test: {Xtest.shape}")


    # ======================================================
    # PCA → SVM
    # ======================================================
    Xtr_pca, Xte_pca, _ = run_pca(Xtrain, Xtest, n_components=2)

    pca_folder = os.path.join(fold_folder, "PCA")
    res_pca = svm_rbf_tuned(Xtr_pca, ytrain, Xte_pca, ytest,
                            save_folder=pca_folder)

    summary_records["pca"].append(res_pca)


    # ======================================================
    # FDA → SVM
    # ======================================================
    Xtr_fda, Xte_fda, _ = run_fda(Xtrain, ytrain, Xtest)

    fda_folder = os.path.join(fold_folder, "FDA")
    res_fda = svm_rbf_tuned(Xtr_fda, ytrain, Xte_fda, ytest,
                            save_folder=fda_folder)

    summary_records["fda"].append(res_fda)


    # ======================================================
    # kPCA (tuned) → SVM
    # ======================================================
    Xtr_kpca, Xte_kpca, kpca_model, best_gamma, score = run_kpca_tuned(
        Xtrain, ytrain, Xtest,
        gamma_grid=[0.001, 0.01, 0.1, 1, 10],
        n_components=2
    )

    print(f"Best gamma (kPCA tuned): {best_gamma}   Score={score:.3f}")

    kpca_folder = os.path.join(fold_folder, "KPCA")
    res_kpca = svm_rbf_tuned(Xtr_kpca, ytrain, Xte_kpca, ytest,
                             save_folder=kpca_folder)

    res_kpca["best_gamma"] = best_gamma
    summary_records["kpca"].append(res_kpca)


    # ======================================================
    # kFDA → SVM
    # ======================================================
    Xtr_kfda, Xte_kfda, _ = run_kfda(Xtrain, ytrain, Xtest)

    kfda_folder = os.path.join(fold_folder, "KFDA")
    res_kfda = svm_rbf_tuned(Xtr_kfda, ytrain, Xte_kfda, ytest,
                             save_folder=kfda_folder)

    summary_records["kfda"].append(res_kfda)



# ===============================================================
# SAVE SUMMARY (GLOBAL)
# ===============================================================

summary_dir = os.path.join(RESULTS_DIR, "Summary")
os.makedirs(summary_dir, exist_ok=True)

with open(os.path.join(summary_dir, "aggregated_results.json"), "w") as f:
    json.dump(summary_records, f, indent=2)

print("\n=== Nested CV pipeline completed successfully. ===")
