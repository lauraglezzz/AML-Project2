# Models/dimensionality.py
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ============================================================
# 1. PCA
# ============================================================

def run_pca(X_train, X_test, n_components=2):
    Xtr = np.asarray(X_train)
    Xte = np.asarray(X_test)

    pca = PCA(n_components=n_components)
    return pca.fit_transform(Xtr), pca.transform(Xte), pca


# ============================================================
# 2. FDA (stable version)
# ============================================================

def run_fda(X_train, y_train, X_test, n_components=None):
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train)

    classes = np.unique(ytr)
    n_features = Xtr.shape[1]

    # means
    overall_mean = np.mean(Xtr, axis=0)
    mean_vectors = {c: np.mean(Xtr[ytr == c], axis=0) for c in classes}

    Sb = np.zeros((n_features, n_features))
    Sw = np.zeros((n_features, n_features))

    for c in classes:
        Xc = Xtr[ytr == c]
        mean_c = mean_vectors[c]
        diff = (mean_c - overall_mean).reshape(-1, 1)
        Sb += Xc.shape[0] * (diff @ diff.T)
        Sw += (Xc - mean_c).T @ (Xc - mean_c)

    # regularize Sw
    Sw += 1e-6 * np.eye(n_features)

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)

    # only take real part
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # sort eigenvalues
    idx = np.argsort(eigvals)[::-1]

    if n_components is None:
        n_components = len(classes) - 1

    W = eigvecs[:, idx[:n_components]]

    return Xtr @ W, Xte @ W, W


# ============================================================
# 3. Kernel PCA (supervised tuning)
# ============================================================

def run_kpca_tuned(X_train, y_train, X_test,
                   gamma_grid=(0.001, 0.01, 0.1, 1, 10),
                   n_components=2):
    
    Xtr = np.asarray(X_train)
    Xte = np.asarray(X_test)
    ytr = np.asarray(y_train)

    best_gamma, best_score = None, -np.inf
    best_kpca = None

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    for gamma in gamma_grid:
        kpca = KernelPCA(
            kernel="rbf",
            gamma=gamma,
            n_components=n_components,
            remove_zero_eig=True
        )

        try:
            Xtr_k = kpca.fit_transform(Xtr)
            Xtr_k = np.real(Xtr_k)
        except Exception:
            continue

        clf = SVC(kernel='linear')

        try:
            scores = cross_val_score(clf, Xtr_k, ytr, cv=skf, scoring='balanced_accuracy')
            mean_score = np.mean(scores)
        except Exception:
            mean_score = -np.inf

        if mean_score > best_score:
            best_score = mean_score
            best_gamma = gamma
            best_kpca = kpca

    # final fit
    Xtr_final = best_kpca.fit_transform(Xtr)
    Xte_final = best_kpca.transform(Xte)

    return np.real(Xtr_final), np.real(Xte_final), best_kpca, best_gamma, best_score


# ============================================================
# 4. Kernel FDA (stable version)
# ============================================================

def run_kfda(X_train, y_train, X_test, gamma=1.0, n_components=3):
    Xtr = np.asarray(X_train)
    Xte = np.asarray(X_test)
    ytr = np.asarray(y_train)

    # KPCA mapping (more components improves FDA stability)
    kpca = KernelPCA(
        kernel="rbf",
        gamma=gamma,
        n_components=n_components,
        remove_zero_eig=True
    )

    Xtr_k = np.real(kpca.fit_transform(Xtr))
    Xte_k = np.real(kpca.transform(Xte))

    # FDA in mapped space
    Xtr_fda, Xte_fda, W = run_fda(Xtr_k, ytr, Xte_k, n_components=1)

    return Xtr_fda, Xte_fda, {"kpca": kpca, "W": W}
