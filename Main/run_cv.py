# ============================================================
# Main script for nested cross-validation
# ============================================================

import pandas as pd
from sklearn.model_selection import KFold

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocessing.preprocess import load_phoneme_data, prepare_fold_data


# ------------------------------------------------------------
# 1. Load dataset cleaned in R
# ------------------------------------------------------------

df = load_phoneme_data()
print("\n=== Starting Nested Cross-Validation Pipeline ===\n")


# ------------------------------------------------------------
# 2. Define outer-fold CV
# ------------------------------------------------------------

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# later we will store results here
fold_results = []


# ------------------------------------------------------------
# 3. Loop through outer folds (preprocessing only for now)
# ------------------------------------------------------------

for fold_number, (train_idx, test_idx) in enumerate(kf.split(df), start=1):

    print(f"\n----- Fold {fold_number}/{K} -----")

    # Preprocess inside the fold (equivalent to preprocess_split in R)
    fold_data = prepare_fold_data(df, train_idx, test_idx)

    Xtrain = fold_data["X_train"]
    ytrain = fold_data["y_train"]
    Xtest  = fold_data["X_test"]
    ytest  = fold_data["y_test"]

    print("Train shape:", Xtrain.shape)
    print("Test shape :", Xtest.shape)

    # --------------------------------------------------------
    # 4. PLACEHOLDER for PCA / FDA / kPCA / kFDA
    # --------------------------------------------------------
    # Example (once dimensionality.py is created):
    # from dimensionality import run_pca
    # Xtrain_pca, Xtest_pca, pca_model = run_pca(Xtrain, Xtest, n_components=2)

    # --------------------------------------------------------
    # 5. PLACEHOLDER for SVM models
    # --------------------------------------------------------
    # Example (once models.py is created):
    # from models import svm_rbf
    # acc = svm_rbf(Xtrain_pca, ytrain, Xtest_pca, ytest)
    # fold_results.append({"fold": fold_number, "acc": acc})

    # --------------------------------------------------------
    # For now, we only perform preprocessing.
    # --------------------------------------------------------


print("\n=== Nested CV pipeline (preprocessing only) completed. ===")
