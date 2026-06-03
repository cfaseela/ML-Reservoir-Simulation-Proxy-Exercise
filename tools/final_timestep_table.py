"""Quantitative per-case table of PRESSURE and SWAT differences at the
FINAL timestep. Outputs a CSV and a pretty-printed text table.

For each of the 9 representative cases, computes:
  - mean(truth), mean(pred)           — single-number summary per case
  - mean(pred - truth)                — bias (signed mean error)
  - mean(|pred - truth|)              — MAE across all active cells
  - RMSE                              — sqrt(mean((pred-truth)^2))
  - %_of_truth_mean                   — MAE / |truth_mean| * 100 (relative scale)
"""
import json
import os
import csv

import numpy as np
from resdata.resfile import ResdataFile

CASES_JSON = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/representative_cases.json"
TRUTH_BASE = "/mnt/e/NORNE/cases"
PRED_BASE = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/predictions_unrst"
OUT_CSV = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/final_timestep_table.csv"
OUT_TXT = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/final_timestep_table.txt"


def load_final_p_and_s(unrst_path):
    """Single pass through the UNRST: return the LAST PRESSURE and LAST SWAT."""
    f = ResdataFile(unrst_path)
    last_p, last_s = None, None
    for kw in f:
        n = kw.getName().strip()
        if n == "PRESSURE":
            last_p = kw.numpy_view().copy()
        elif n == "SWAT":
            last_s = kw.numpy_view().copy()
    return last_p, last_s


def stats(truth, pred):
    truth = truth.astype(np.float64)
    pred = pred.astype(np.float64)
    diff = pred - truth
    return {
        "truth_mean": float(truth.mean()),
        "pred_mean": float(pred.mean()),
        "bias": float(diff.mean()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "pct_of_truth": float(np.mean(np.abs(diff)) / abs(truth.mean()) * 100),
    }


def main():
    with open(CASES_JSON) as f:
        spec = json.load(f)
    cases = spec["cases"]
    ann = spec.get("annotations", {})
    order = {"BEST": 0, "MEDIAN": 1, "WORST": 2}
    cases.sort(key=lambda c: (order.get(ann.get(c, ""), 9), c))

    rows = []
    for case in cases:
        truth = os.path.join(TRUTH_BASE, case, f"{case}.UNRST")
        pred = os.path.join(PRED_BASE, f"{case}.UNRST")
        if not (os.path.exists(truth) and os.path.exists(pred)):
            print(f"  skip {case}: missing file")
            continue
        print(f"  loading {case} ...", flush=True)
        t_p, t_s = load_final_p_and_s(truth)
        p_p, p_s = load_final_p_and_s(pred)
        ps = stats(t_p, p_p)
        ss = stats(t_s, p_s)
        rows.append({
            "case": case,
            "label": ann.get(case, ""),
            "p_truth_mean": ps["truth_mean"],
            "p_pred_mean": ps["pred_mean"],
            "p_bias": ps["bias"],
            "p_mae": ps["mae"],
            "p_rmse": ps["rmse"],
            "p_pct": ps["pct_of_truth"],
            "s_truth_mean": ss["truth_mean"],
            "s_pred_mean": ss["pred_mean"],
            "s_bias": ss["bias"],
            "s_mae": ss["mae"],
            "s_rmse": ss["rmse"],
            "s_pct": ss["pct_of_truth"],
        })

    # CSV output
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {OUT_CSV}")

    # Pretty text table
    lines = []
    lines.append("=" * 110)
    lines.append("FINAL TIMESTEP — Eclipse truth vs X-MGN proxy, 9 representative cases")
    lines.append("Aggregated over ~44,431 active cells per case")
    lines.append("=" * 110)
    lines.append("")
    lines.append("PRESSURE (bar):")
    lines.append(f"{'case':<12} {'label':<7} {'truth_mean':>11} {'pred_mean':>11} {'bias':>8} {'MAE':>8} {'RMSE':>8} {'MAE % of truth':>16}")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['case']:<12} {r['label']:<7} "
            f"{r['p_truth_mean']:>11.3f} {r['p_pred_mean']:>11.3f} "
            f"{r['p_bias']:>+8.3f} {r['p_mae']:>8.3f} {r['p_rmse']:>8.3f} "
            f"{r['p_pct']:>15.2f}%"
        )
    lines.append("")
    lines.append("SWAT (fraction):")
    lines.append(f"{'case':<12} {'label':<7} {'truth_mean':>11} {'pred_mean':>11} {'bias':>8} {'MAE':>8} {'RMSE':>8} {'MAE % of truth':>16}")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['case']:<12} {r['label']:<7} "
            f"{r['s_truth_mean']:>11.5f} {r['s_pred_mean']:>11.5f} "
            f"{r['s_bias']:>+8.5f} {r['s_mae']:>8.5f} {r['s_rmse']:>8.5f} "
            f"{r['s_pct']:>15.2f}%"
        )
    lines.append("=" * 110)
    txt = "\n".join(lines)
    print()
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
