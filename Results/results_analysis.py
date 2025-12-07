import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# ------------------------------------------------------------
# 1. Load aggregated results from run_cv.py
# ------------------------------------------------------------

RESULTS_PATH = "Summary/aggregated_results.json"

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

methods = ["pca", "fda", "kpca", "kfda"]
metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]

# Convert nested JSON → dataframe for easier analysis
rows = []
for m in methods:
    for fold, fold_data in enumerate(results[m]["folds"], start=1):
        row = {"method": m, "fold": fold}
        row.update(fold_data["test_metrics"])
        rows.append(row)

df = pd.DataFrame(rows)


# ------------------------------------------------------------
# 2. Summary statistics: mean ± std per method
# ------------------------------------------------------------

summary = df.groupby("method")[metrics].agg(["mean", "std"])
summary.to_csv("../Results/summary_metrics.csv")

print("\n===== SUMMARY METRICS (mean ± std) =====\n")
print(summary)


# ------------------------------------------------------------
# 3. Wilcoxon tests between methods (pairwise)
# ------------------------------------------------------------

def wilcoxon_pair(metric, m1, m2):
    x = df[df["method"] == m1][metric]
    y = df[df["method"] == m2][metric]
    stat, p = wilcoxon(x, y)
    return stat, p

pairs = [
    ("pca", "fda"),
    ("pca", "kpca"),
    ("fda", "kpca"),
    ("kpca", "kfda"),
    ("fda", "kfda"),
]

wilcoxon_results = {}

for metric in metrics:
    wilcoxon_results[metric] = {}
    for a, b in pairs:
        stat, p = wilcoxon_pair(metric, a, b)
        wilcoxon_results[metric][f"{a}_vs_{b}"] = {"stat": stat, "p": p}

with open("../Results/wilcoxon_tests.json", "w") as f:
    json.dump(wilcoxon_results, f, indent=2)

print("\n===== WILCOXON TESTS =====\n")
print(json.dumps(wilcoxon_results, indent=2))


# ------------------------------------------------------------
# 4. Plot: barplot with std (mean ± std)
# ------------------------------------------------------------

os.makedirs("../Results/plots", exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x="method", y=metric, ci="sd")
    plt.title(f"{metric} (mean ± std over folds)")
    plt.savefig(f"../Results/plots/bar_{metric}.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# 5. Plot: 2D scatter projections (PCA/FDA/kPCA/kFDA)
# ------------------------------------------------------------

# Load projections saved by run_cv
PROJ_PATH = "../Results/projections/"
proj_files = {
    m: os.path.join(PROJ_PATH, f"{m}_projections_fold1.npz")
    for m in methods
}

for m, fpath in proj_files.items():
    if not os.path.exists(fpath):
        continue
    data = np.load(fpath)
    X = data["Xtrain"]
    y = data["ytrain"]

    plt.figure(figsize=(5,4))
    plt.scatter(X[:,0], X[:,1], c=(y=="oral"), cmap="coolwarm", alpha=0.6)
    plt.title(f"{m.upper()} projection (Fold 1)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f"../Results/plots/scatter_{m}.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# 6. 1D density plots of discriminant axis (FDA / kFDA)
# ------------------------------------------------------------

for m in ["fda", "kfda"]:
    fpath = proj_files[m]
    if not os.path.exists(fpath):
        continue

    data = np.load(fpath)
    X = data["Xtrain"][:,0]   # first discriminant direction
    y = data["ytrain"]

    plt.figure(figsize=(6,4))
    sns.kdeplot(X[y=="nasal"], label="nasal")
    sns.kdeplot(X[y=="oral"], label="oral")
    plt.title(f"Density on discriminant axis ({m.upper()})")
    plt.legend()
    plt.savefig(f"../Results/plots/density_{m}.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# 7. ROC and PR curves (averaged over folds)
# ------------------------------------------------------------

# run_cv saved per-fold predictions → load them
if "predictions" in results:
    preds = results["predictions"]

    for m in methods:
        if m not in preds:
            continue

        fprs = []
        tprs = []
        prs = []
        recs = []

        for fold in preds[m]:
            y_true = np.array(fold["y_true"])
            y_score = np.array(fold["y_prob"])

            fpr, tpr, _ = roc_curve((y_true=="oral").astype(int), y_score)
            pre, rec, _ = precision_recall_curve((y_true=="oral").astype(int), y_score)

            fprs.append(fpr)
            tprs.append(tpr)
            prs.append(pre)
            recs.append(rec)

        # average curves
        mean_fpr = np.linspace(0,1,100)
        mean_tpr = np.mean([np.interp(mean_fpr, fprs[i], tprs[i]) for i in range(len(fprs))], axis=0)

        plt.figure(figsize=(5,4))
        plt.plot(mean_fpr, mean_tpr)
        plt.title(f"ROC Curve (Averaged) — {m.upper()}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"../Results/plots/roc_{m}.png", dpi=200)
        plt.close()


print("\n=== Analysis complete: plots + summaries generated ===\n")
