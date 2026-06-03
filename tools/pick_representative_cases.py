"""
Pick a small, informative set of cases from per_case_metrics.csv to generate
GRDECL files for. Strategy:
  * 3 BEST  (lowest PRESSURE RMSE — model handles these well)
  * 3 WORST (highest PRESSURE RMSE — failure-mode candidates)
  * 3 MEDIAN-ish (around P50 — typical case)
The 9-case set covers the full distribution for the writeup figures without
running GRDECL generation for all 432 cases (~7 h on this hardware).

Outputs a JSON file the run_post_only script can consume.
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/mnt/e/NVIDIA/reservoir_simulation/eval_results/per_case_metrics.csv",
    )
    ap.add_argument(
        "--out",
        default="/mnt/e/NVIDIA/reservoir_simulation/eval_results/representative_cases.json",
    )
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["PRESSURE_rmse"] = float(r["PRESSURE_rmse"])
            r["SWAT_rmse"] = float(r["SWAT_rmse"])
            rows.append(r)

    rows.sort(key=lambda r: r["PRESSURE_rmse"])
    n = len(rows)
    mid = n // 2

    picked = []
    picked.extend([("BEST", r) for r in rows[:3]])
    picked.extend([("MEDIAN", r) for r in rows[mid - 1 : mid + 2]])
    picked.extend([("WORST", r) for r in rows[-3:]])

    cases_out = [r["case"] for _, r in picked]
    print(f"Picked {len(cases_out)} representative cases from {n} total:")
    for label, r in picked:
        print(
            f"  {label:7s}  {r['case']:12s}  "
            f"PRESSURE_rmse={r['PRESSURE_rmse']:.3f}  "
            f"SWAT_rmse={r['SWAT_rmse']:.5f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"cases": cases_out, "annotations": {r["case"]: label for label, r in picked}},
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
