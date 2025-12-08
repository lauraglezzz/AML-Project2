# Results/results_analysis.py
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
RESULTS_JSON = HERE / "Summary" / "aggregated_results.json"   # fixed path as requested
OUT_TABLES = HERE / "SummaryTables"
OUT_PLOTS = HERE / "FinalPlots"

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# FPR grid for ROC interpolation
FPR_GRID = np.linspace(0, 1, 101)
ALPHA = 0.05
Z95 = 1.96  # approximate

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
    """Return mean, lower, upper (95% CI using normal approx)"""
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    se = sem(arr, nan_policy="omit")
    lo = m - Z95 * se
    hi = m + Z95 * se
    return float(m), float(lo), float(hi)

def interp_tprs(tprs: List[np.ndarray], fprs: List[np.ndarray], base_fpr=FPR_GRID):
    """Interpolate tprs to base_fpr and return array (n_folds, len(base_fpr))"""
    tprs_interp = []
    for tpr, fpr in zip(tprs, fprs):
        # clip monotonic
        fpr = np.asarray(fpr)
        tpr = np.asarray(tpr)
        # ensure first point 0 and last 1
        if fpr[0] > 0:
            fpr = np.concatenate(([0.0], fpr))
            tpr = np.concatenate(([0.0], tpr))
        if fpr[-1] < 1:
            fpr = np.concatenate((fpr, [1.0]))
            tpr = np.concatenate((tpr, [1.0]))
        tpr_i = np.interp(base_fpr, fpr, tpr)
        tprs_interp.append(tpr_i)
    return np.vstack(tprs_interp) if len(tprs_interp) > 0 else np.empty((0, len(base_fpr)))

def safe_get_fold_field(fold: dict, field: str):
    """Try several common names for probabilities/scores"""
    if field in fold:
        return fold[field]
    # Common alternates
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

    # Metrics we'll summarize (if present)
    metrics_list = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]

    # Collect per-method metric arrays and folds data
    summary_tables = {}
    per_method_folds = {}

    for m in methods:
        folds = results[m]
        if not isinstance(folds, list) or len(folds) == 0:
            print(f"Warning: method {m} has no fold list or it's empty. Skipping.")
            continue

        per_method_folds[m] = folds

        # Build arrays for metrics
        metric_vals = {met: [] for met in metrics_list}
        # Store per-fold y_true and y_score if present for ROC
        fold_y_trues = []
        fold_y_scores = []  # positive-class scores (1d)
        available_roc = True

        for idx, fold in enumerate(folds):
            tm = fold.get("test_metrics", {})
            for met in metrics_list:
                metric_vals[met].append(tm.get(met, np.nan))
            # get probabilities / scores
            y_true = safe_get_fold_field(fold, "y_true")
            y_prob = safe_get_fold_field(fold, "y_prob")
            # some JSONs store 'y_score' or 'y_scores' or 'prob'
            if y_prob is None:
                y_prob = safe_get_fold_field(fold, "y_score")
            # If y_prob is 2D list, extract positive class if present
            if y_prob is not None:
                y_prob_arr = np.asarray(y_prob)
                if y_prob_arr.ndim == 2:
                    # assume second column is positive
                    if y_prob_arr.shape[1] >= 2:
                        y_score_pos = y_prob_arr[:, 1]
                    else:
                        # fallback to column 0
                        y_score_pos = y_prob_arr[:, 0]
                else:
                    y_score_pos = y_prob_arr
            else:
                y_score_pos = None

            fold_y_trues.append(np.asarray(y_true) if y_true is not None else None)
            fold_y_scores.append(y_score_pos)
            if y_true is None or y_score_pos is None:
                available_roc = False

        # Summary table for method m
        rows = []
        for met in metrics_list:
            arr = np.array(metric_vals[met], dtype=float)
            mval, lo, hi = mean_ci95(arr)
            rows.append({"metric": met, "mean": mval, "ci95_lo": lo, "ci95_hi": hi, "values": arr.tolist()})

        df_summary = pd.DataFrame(rows)
        summary_tables[m] = df_summary

        # Save per-method table CSV & LaTeX
        df_summary.to_csv(OUT_TABLES / f"{m}_summary.csv", index=False)
        with open(OUT_TABLES / f"{m}_summary.tex", "w") as f:
            f.write(df_summary.to_latex(index=False, float_format="%.4f"))

        print(f"Saved summary for {m} to {OUT_TABLES}")

        # ROC plotting preparation: compute per-fold ROC curves if possible
        if available_roc:
            tprs = []
            fprs = []
            aucs = []
            for y_true, y_score in zip(fold_y_trues, fold_y_scores):
                #  y_true: array of labels strings; positive label is 'oral' in your dataset
                if y_true is None or y_score is None:
                    continue
                y_true_arr = np.asarray(y_true)
                # define positive label: method consistent assumption: 'oral' is positive
                y_bin = (y_true_arr == "oral").astype(int)
                try:
                    fpr, tpr, _ = roc_curve(y_bin, np.asarray(y_score))
                    tprs.append(tpr)
                    fprs.append(fpr)
                    aucs.append(auc(fpr, tpr))
                except Exception as e:
                    print(f"Skipping ROC for fold due to: {e}")

            if len(tprs) > 0:
                tprs_interp = interp_tprs(tprs, fprs, base_fpr=FPR_GRID)
                mean_tpr = np.nanmean(tprs_interp, axis=0)
                std_tpr = np.nanstd(tprs_interp, axis=0, ddof=1)
                lo_tpr = np.maximum(0, mean_tpr - Z95 * std_tpr / np.sqrt(tprs_interp.shape[0]))
                hi_tpr = np.minimum(1, mean_tpr + Z95 * std_tpr / np.sqrt(tprs_interp.shape[0]))

                # Plot ROC mean+CI
                plt.figure(figsize=(6, 5))
                plt.plot(FPR_GRID, mean_tpr, label=f"{m} mean ROC (AUC~{np.mean(aucs):.3f})")
                plt.fill_between(FPR_GRID, lo_tpr, hi_tpr, alpha=0.25)
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Mean ROC ± CI95 — {m}")
                plt.legend(loc="lower right")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                plt.savefig(OUT_PLOTS / f"roc_mean_ci_{m}.png", dpi=150)
                plt.close()
                print(f"Saved ROC mean+CI for {m} to {OUT_PLOTS}")
            else:
                print(f"No valid ROC curves computed for {m} (no y_prob or ROC computation failed).")
        else:
            print(f"Skipping ROC for {m}: missing per-fold y_true or y_prob.")

    # ----------------------------
    # Save an overall aggregated summary CSV (metrics per method)
    all_summary_rows = []
    for m, df in summary_tables.items():
        # pick balanced_accuracy row
        ba_row = df[df.metric == "balanced_accuracy"].iloc[0]
        all_summary_rows.append({
            "method": m,
            "balanced_accuracy_mean": ba_row["mean"],
            "balanced_accuracy_ci_lo": ba_row["ci95_lo"],
            "balanced_accuracy_ci_hi": ba_row["ci95_hi"],
            "n_folds": len(per_method_folds[m])
        })
    df_overall = pd.DataFrame(all_summary_rows)
    df_overall.to_csv(OUT_TABLES / "methods_overview.csv", index=False)
    print("Saved summary tables.")

    # ----------------------------
    # Pairwise Wilcoxon on balanced_accuracy (method vs method)
    print("Computing Wilcoxon pairwise tests on balanced_accuracy...")
    methods_present = list(summary_tables.keys())
    wilc_rows = []
    for i in range(len(methods_present)):
        for j in range(i + 1, len(methods_present)):
            m1 = methods_present[i]
            m2 = methods_present[j]
            v1 = np.array(summary_tables[m1].loc[summary_tables[m1].metric == "balanced_accuracy", "values"].iloc[0], dtype=float)
            v2 = np.array(summary_tables[m2].loc[summary_tables[m2].metric == "balanced_accuracy", "values"].iloc[0], dtype=float)
            # ensure equal length
            n = min(len(v1), len(v2))
            try:
                stat, p = wilcoxon(v1[:n], v2[:n])
            except Exception:
                stat, p = np.nan, np.nan
            wilc_rows.append({"method1": m1, "method2": m2, "stat": float(stat) if not np.isnan(stat) else None, "pvalue": float(p) if not np.isnan(p) else None})
    pd.DataFrame(wilc_rows).to_csv(OUT_TABLES / "wilcoxon_balanced_accuracy.csv", index=False)
    print("Saved Wilcoxon results.")

    # ----------------------------
    # Boxplots for the metrics (balanced_accuracy by default; also accuracy/f1)
    for met in ["balanced_accuracy", "accuracy", "f1"]:
        data = []
        names = []
        for m in methods_present:
            values = summary_tables[m].loc[summary_tables[m].metric == met, "values"].iloc[0]
            data.append(values)
            names.append(m)
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.boxplot(data, labels=names, showmeans=True)
        ax.set_title(f"Boxplot — {met}")
        ax.set_ylabel(met)
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / f"boxplot_{met}.png", dpi=150)
        plt.close()
    print("Saved boxplots.")

    # ----------------------------
    # Scatterplots for 2D projections on TEST for the best fold (by balanced_accuracy) per method
    print("Creating scatterplots on TEST for best-performing fold per method (balanced_accuracy).")
    for m in methods_present:
        folds = per_method_folds[m]
        # find best fold index by balanced_accuracy in the fold entries (if present) else fallback to first fold
        best_idx = 0
        best_ba = -np.inf
        for idx, fold in enumerate(folds):
            tm = fold.get("test_metrics", {})
            ba = tm.get("balanced_accuracy", None)
            if ba is not None and ba > best_ba:
                best_ba = ba
                best_idx = idx

        fold = folds[best_idx]
        # try load projections: common keys to try
        X_test_proj = None
        for key in ("X_test_2d", "X_test_proj", "X_test", "Xtest_2d"):
            v = safe_get_fold_field(fold, key)
            if v is not None:
                arr = np.asarray(v)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    X_test_proj = arr[:, :2]
                    break
        y_true = safe_get_fold_field(fold, "y_true")
        if X_test_proj is None:
            print(f" - {m}: No 2D test projections found for best fold (idx {best_idx}). Skipping scatter.")
            continue
        if y_true is None:
            print(f" - {m}: No y_true in fold; cannot color scatter. Skipping.")
            continue

        y_true = np.asarray(y_true)
        # prepare plot
        plt.figure(figsize=(6, 5))
        colors = {"nasal": "#1f77b4", "oral": "#ff7f0e"}
        for cls in np.unique(y_true):
            mask = (y_true == cls)
            plt.scatter(X_test_proj[mask, 0], X_test_proj[mask, 1], s=12, alpha=0.7, label=str(cls), color=colors.get(cls, None))
            # density contours via kde
            try:
                xy = X_test_proj[mask].T
                kde = gaussian_kde(xy)
                # grid
                xmin, ymin = X_test_proj[:,0].min(), X_test_proj[:,1].min()
                xmax, ymax = X_test_proj[:,0].max(), X_test_proj[:,1].max()
                xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                # plot contour
                plt.contour(xi, yi, zi.reshape(xi.shape), levels=3, alpha=0.4, colors=[colors.get(cls, "k")])
            except Exception:
                pass

        plt.legend()
        plt.title(f"{m.upper()} — Test projection (best fold idx {best_idx}, BA={best_ba:.3f})")
        plt.xlabel("dim1")
        plt.ylabel("dim2")

        # try to overlay decision boundary if model path present
        model_path = safe_get_fold_field(fold, "model_path") or safe_get_fold_field(fold, "model")
        if model_path:
            try:
                # if model_path is string, try load; if dict-like, skip
                if isinstance(model_path, str) and Path(model_path).exists():
                    clf = joblib.load(model_path)
                    # build grid and predict
                    xmin, ymin = X_test_proj[:,0].min(), X_test_proj[:,1].min()
                    xmax, ymax = X_test_proj[:,0].max(), X_test_proj[:,1].max()
                    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
                    grid = np.c_[xx.ravel(), yy.ravel()]
                    try:
                        Z = clf.predict(grid)
                        Zm = np.array([1 if z == "oral" or z == 1 else 0 for z in Z])
                        Zm = Zm.reshape(xx.shape)
                        plt.contour(xx, yy, Zm, levels=[0.5], colors="k", linewidths=1.2, linestyles="--")
                        plt.title(plt.gca().get_title() + " + SVM boundary")
                    except Exception:
                        pass
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(OUT_PLOTS / f"scatter_{m}_bestfold.png", dpi=150)
        plt.close()
        print(f"Saved scatter for {m} best fold.")

    print("All done. Plots and tables saved under:", OUT_PLOTS, "and", OUT_TABLES)


if __name__ == "__main__":
    main()
