"""
Aggregates results, computes summary statistics + CI, runs Wilcoxon tests,
plots ROC mean+CI, boxplots and 2D scatter + decision boundary (if available).
Assumes there's a JSON at: ../Summary/aggregated_results.json  (relative to Results/)
Saves outputs under: ./SummaryTables and ./FinalPlots (inside Results folder).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon
from scipy.stats import sem
from scipy.stats import gaussian_kde
import joblib

# ---------- Config ----------
HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "Summary" / "aggregated_results.json"
OUT_TABLES = HERE / "SummaryTables"
OUT_PLOTS = HERE / "FinalPlots"

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# FPR grid for ROC interpolation
FPR_GRID = np.linspace(0, 1, 101)
Z95 = 1.96


# ---------- Helpers ----------
def load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Reading results from: {path}\nFile not found.")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected aggregated_results.json to be a dict with method keys.")
    return data


def mean_ci95(arr: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    se = sem(arr, nan_policy="omit")
    lo = m - Z95 * se
    hi = m + Z95 * se
    return float(m), float(lo), float(hi)


def interp_tprs(tprs: List[np.ndarray], fprs: List[np.ndarray], base_fpr=FPR_GRID):
    tprs_interp = []
    for tpr, fpr in zip(tprs, fprs):
        fpr = np.asarray(fpr)
        tpr = np.asarray(tpr)

        if fpr[0] > 0:
            fpr = np.concatenate(([0.0], fpr))
            tpr = np.concatenate(([0.0], tpr))
        if fpr[-1] < 1:
            fpr = np.concatenate((fpr, [1.0]))
            tpr = np.concatenate((tpr, [1.0]))

        tpr_i = np.interp(base_fpr, fpr, tpr)
        tprs_interp.append(tpr_i)

    return np.vstack(tprs_interp) if len(tprs_interp) else np.empty((0, len(base_fpr)))


def safe_get_fold_field(fold: dict, field: str):
    if field in fold:
        return fold[field]
    for alt in (field.lower(), field.upper(), field + "_prob", field + "_proba", field + "s"):
        if alt in fold:
            return fold[alt]
    return None


# ---------- Main ----------
def main():
    print(f"Reading results from: {RESULTS_JSON}")
    results = load_results(RESULTS_JSON)

    methods = sorted(results.keys())
    print("Methods found:", methods)

    metrics_list = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]

    summary_tables = {}
    per_method_folds = {}

    for m in methods:
        folds = results[m]
        per_method_folds[m] = folds

        # Build arrays for metrics
        metric_vals = {met: [] for met in metrics_list}
        fold_y_trues = []
        fold_y_scores = []
        available_roc = True

        # Determine kernel used
        method_kernel = folds[0].get("best_kernel", "UNKNOWN").upper()

        for idx, fold in enumerate(folds):
            tm = fold.get("test_metrics", {})
            for met in metrics_list:
                metric_vals[met].append(tm.get(met, np.nan))

            # probs
            y_true = safe_get_fold_field(fold, "y_true")
            y_prob = safe_get_fold_field(fold, "y_prob")
            if y_prob is None:
                y_prob = safe_get_fold_field(fold, "y_score")

            if y_prob is not None:
                y_prob_arr = np.asarray(y_prob)
                if y_prob_arr.ndim == 2:
                    if y_prob_arr.shape[1] >= 2:
                        y_score_pos = y_prob_arr[:, 1]
                    else:
                        y_score_pos = y_prob_arr[:, 0]
                else:
                    y_score_pos = y_prob_arr
            else:
                y_score_pos = None

            fold_y_trues.append(np.asarray(y_true) if y_true is not None else None)
            fold_y_scores.append(y_score_pos)
            if y_true is None or y_score_pos is None:
                available_roc = False

        # Summary table
        rows = []
        for met in metrics_list:
            arr = np.array(metric_vals[met], dtype=float)
            mval, lo, hi = mean_ci95(arr)
            rows.append({"metric": met, "mean": mval, "ci95_lo": lo, "ci95_hi": hi, "values": arr.tolist()})

        df_summary = pd.DataFrame(rows)
        summary_tables[m] = df_summary
        df_summary.to_csv(OUT_TABLES / f"{m}_summary.csv", index=False)

        with open(OUT_TABLES / f"{m}_summary.tex", "w") as f:
            f.write(df_summary.to_latex(index=False, float_format="%.4f"))

        print(f"Saved summary for {m} to {OUT_TABLES}")

        # ROC plotting
        if available_roc:
            tprs, fprs, aucs = [], [], []
            for y_true, y_score in zip(fold_y_trues, fold_y_scores):
                if y_true is None or y_score is None:
                    continue
                y_bin = (np.asarray(y_true) == "oral").astype(int)
                fpr, tpr, _ = roc_curve(y_bin, np.asarray(y_score))
                tprs.append(tpr)
                fprs.append(fpr)
                aucs.append(auc(fpr, tpr))

            if len(tprs) > 0:
                tprs_interp = interp_tprs(tprs, fprs)
                mean_tpr = np.nanmean(tprs_interp, axis=0)
                std_tpr = np.nanstd(tprs_interp, axis=0)

                lo_tpr = np.maximum(0, mean_tpr - Z95 * std_tpr / np.sqrt(tprs_interp.shape[0]))
                hi_tpr = np.minimum(1, mean_tpr + Z95 * std_tpr / np.sqrt(tprs_interp.shape[0]))

                plt.figure(figsize=(6, 5))
                plt.plot(FPR_GRID, mean_tpr, label=f"AUC={np.mean(aucs):.3f}")
                plt.fill_between(FPR_GRID, lo_tpr, hi_tpr, alpha=0.25)

                plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")

                # 🔥 NEW: include kernel name
                plt.title(f"Mean ROC ± CI95 — {m.upper()} — SVM {method_kernel}")

                plt.legend(loc="lower right")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                plt.savefig(OUT_PLOTS / f"roc_mean_ci_{m}.png", dpi=150)
                plt.close()
                print(f"Saved ROC mean+CI for {m}")

    # OVERALL SUMMARY
    all_summary_rows = []
    for m, df in summary_tables.items():
        ba_row = df[df.metric == "balanced_accuracy"].iloc[0]
        all_summary_rows.append({
            "method": m,
            "balanced_accuracy_mean": ba_row["mean"],
            "balanced_accuracy_ci_lo": ba_row["ci95_lo"],
            "balanced_accuracy_ci_hi": ba_row["ci95_hi"],
            "n_folds": len(per_method_folds[m])
        })
    pd.DataFrame(all_summary_rows).to_csv(OUT_TABLES / "methods_overview.csv", index=False)

    print("Saved summary tables.")

    # WILCOXON TESTS
    print("Computing Wilcoxon pairwise tests...")
    methods_present = list(summary_tables.keys())
    wilc_rows = []
    for i in range(len(methods_present)):
        for j in range(i + 1, len(methods_present)):
            m1, m2 = methods_present[i], methods_present[j]
            v1 = np.array(summary_tables[m1].loc[summary_tables[m1].metric == "balanced_accuracy", "values"].iloc[0])
            v2 = np.array(summary_tables[m2].loc[summary_tables[m2].metric == "balanced_accuracy", "values"].iloc[0])
            n = min(len(v1), len(v2))
            stat, p = wilcoxon(v1[:n], v2[:n])
            wilc_rows.append({"method1": m1, "method2": m2, "stat": stat, "pvalue": p})

    pd.DataFrame(wilc_rows).to_csv(OUT_TABLES / "wilcoxon_balanced_accuracy.csv", index=False)


    # BOXPLOTS
    for met in ["balanced_accuracy", "accuracy", "f1"]:
        data, names = [], []
        for m in methods_present:
            values = summary_tables[m].loc[summary_tables[m].metric == met, "values"].iloc[0]
            data.append(values)
            names.append(m.upper())

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=names, showmeans=True)

        plt.ylabel(met)

        # 🔥 Add kernel in title
        plt.title(f"Boxplot — {met.replace('_',' ').title()}")

        plt.tight_layout()
        plt.savefig(OUT_PLOTS / f"boxplot_{met}.png", dpi=150)
        plt.close()


    # SCATTER PLOTS
    print("Creating scatterplots for best fold per method...")
    for m in methods_present:
        folds = per_method_folds[m]
        kernel = folds[0].get("best_kernel", "UNKNOWN").upper()

        best_idx = 0
        best_ba = -np.inf
        for idx, fold in enumerate(folds):
            ba = fold.get("test_metrics", {}).get("balanced_accuracy", None)
            if ba is not None and ba > best_ba:
                best_ba = ba
                best_idx = idx

        fold = folds[best_idx]

        X_test_proj = None
        for key in ("X_test_proj", "X_test_2d", "Xtest_2d"):
            v = safe_get_fold_field(fold, key)
            if v is not None and np.asarray(v).ndim == 2:
                X_test_proj = np.asarray(v)
                break

        y_true = safe_get_fold_field(fold, "y_true")
        if X_test_proj is None or y_true is None:
            print(f"Skipping scatter for {m}")
            continue

        y_true = np.asarray(y_true)

        plt.figure(figsize=(6, 5))
        colors = {"nasal": "#1f77b4", "oral": "#ff7f0e"}
        for cls in np.unique(y_true):
            mask = (y_true == cls)
            plt.scatter(X_test_proj[mask, 0], X_test_proj[mask, 1],
                        s=12, alpha=0.7, label=str(cls), color=colors.get(cls))

        # Title with kernel
        plt.title(f"{m.upper()} — SVM {kernel} — Test Projection (Fold {best_idx}, BA={best_ba:.3f})")
        plt.xlabel("dim1")
        plt.ylabel("dim2")
        plt.legend()

        plt.tight_layout()
        plt.savefig(OUT_PLOTS / f"scatter_{m}_best.png", dpi=150)
        plt.close()

    print("All done! Plots and tables saved.")


if __name__ == "__main__":
    main()
