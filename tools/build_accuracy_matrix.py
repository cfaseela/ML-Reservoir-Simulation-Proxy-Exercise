"""
Parse xmgn inference HDF5 outputs into an at-a-glance accuracy matrix comparing
PRESSURE and SWAT prediction error across all evaluated cases.

Each <case>.hdf5 has:
  predictions/<VAR>/timestep_NNNN  (denormalized vector, one value per active cell)
  targets/<VAR>/timestep_NNNN

Outputs (to --out_dir):
  per_case_metrics.csv     one row per case: RMSE/MAE for PRESSURE and SWAT
  error_vs_timestep.csv    RMSE/MAE per timestep (shows autoregressive drift)
  summary_matrix.csv       headline distribution stats (mean/median/P10/P90/min/max)
  summary_matrix.txt       same, pretty-printed for at-a-glance reading
  accuracy_heatmap.png     cases x timesteps error heatmap (if matplotlib present)

Usage:
  python build_accuracy_matrix.py --inference_dir <dir> --out_dir <dir>
"""

import argparse, os, json, glob
import numpy as np
import h5py

VARS = ["PRESSURE", "SWAT"]
UNITS = {"PRESSURE": "bar", "SWAT": "fraction"}


def per_timestep_errors(h5path):
    """Return {var: {timestep:int -> (rmse, mae, n_cells)}} for one case file."""
    out = {v: {} for v in VARS}
    with h5py.File(h5path, "r") as f:
        if "predictions" not in f or "targets" not in f:
            return out
        for v in VARS:
            if v not in f["predictions"]:
                continue
            for ts_name in f["predictions"][v]:
                pred = f["predictions"][v][ts_name][:]
                if ts_name not in f["targets"][v]:
                    continue
                tgt = f["targets"][v][ts_name][:]
                diff = pred.astype(np.float64) - tgt.astype(np.float64)
                ts = int(ts_name.split("_")[-1])
                out[v][ts] = (float(np.sqrt(np.mean(diff**2))),
                              float(np.mean(np.abs(diff))),
                              diff.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.inference_dir, "*.hdf5")))
    print(f"Found {len(files)} case HDF5 files in {args.inference_dir}")
    if not files:
        raise SystemExit("No HDF5 files — did inference produce outputs?")

    # Accumulators
    per_case = []                       # rows for per_case_metrics.csv
    # squared-error & abs-error sums per timestep per var (for cross-case aggregation)
    ts_se = {v: {} for v in VARS}       # ts -> [sum_sq_err, sum_abs_err, n]
    # for heatmap: case -> var -> {ts: rmse}
    heat = {}

    for fp in files:
        case = os.path.splitext(os.path.basename(fp))[0]
        errs = per_timestep_errors(fp)
        heat[case] = errs
        row = {"case": case}
        for v in VARS:
            tsmap = errs[v]
            if not tsmap:
                row[f"{v}_rmse"] = np.nan; row[f"{v}_mae"] = np.nan
                continue
            # per-case RMSE/MAE = aggregate across that case's timesteps (cell-weighted)
            sse = sum(r**2 * n for (r, _, n) in tsmap.values())
            sae = sum(m * n for (_, m, n) in tsmap.values())
            ntot = sum(n for (_, _, n) in tsmap.values())
            row[f"{v}_rmse"] = float(np.sqrt(sse / ntot))
            row[f"{v}_mae"] = float(sae / ntot)
            # accumulate per-timestep across cases
            for ts, (r, m, n) in tsmap.items():
                acc = ts_se[v].setdefault(ts, [0.0, 0.0, 0])
                acc[0] += r**2 * n; acc[1] += m * n; acc[2] += n
        per_case.append(row)

    # ---- per_case_metrics.csv ----
    pc_path = os.path.join(args.out_dir, "per_case_metrics.csv")
    with open(pc_path, "w") as f:
        f.write("case," + ",".join(f"{v}_rmse,{v}_mae" for v in VARS) + "\n")
        for row in per_case:
            f.write(row["case"] + "," +
                    ",".join(f"{row[f'{v}_rmse']:.6g},{row[f'{v}_mae']:.6g}" for v in VARS) + "\n")
    print(f"Wrote {pc_path}")

    # ---- error_vs_timestep.csv ----
    ev_path = os.path.join(args.out_dir, "error_vs_timestep.csv")
    all_ts = sorted(set().union(*[set(ts_se[v].keys()) for v in VARS]))
    with open(ev_path, "w") as f:
        f.write("timestep," + ",".join(f"{v}_rmse,{v}_mae" for v in VARS) + "\n")
        for ts in all_ts:
            cells = []
            for v in VARS:
                if ts in ts_se[v]:
                    sse, sae, n = ts_se[v][ts]
                    cells.append(f"{np.sqrt(sse/n):.6g},{sae/n:.6g}")
                else:
                    cells.append("nan,nan")
            f.write(f"{ts}," + ",".join(cells) + "\n")
    print(f"Wrote {ev_path}")

    # ---- summary_matrix (distribution of per-case RMSE/MAE) ----
    def stats(arr):
        a = np.array([x for x in arr if not np.isnan(x)])
        return dict(mean=a.mean(), p50=np.percentile(a,50), p10=np.percentile(a,10),
                    p90=np.percentile(a,90), min=a.min(), max=a.max())

    summary = {}
    for v in VARS:
        summary[v] = {
            "rmse": stats([r[f"{v}_rmse"] for r in per_case]),
            "mae":  stats([r[f"{v}_mae"]  for r in per_case]),
        }

    sm_csv = os.path.join(args.out_dir, "summary_matrix.csv")
    with open(sm_csv, "w") as f:
        f.write("variable,metric,mean,P50,P10,P90,min,max\n")
        for v in VARS:
            for m in ("rmse", "mae"):
                s = summary[v][m]
                f.write(f"{v} ({UNITS[v]}),{m.upper()},{s['mean']:.6g},{s['p50']:.6g},"
                        f"{s['p10']:.6g},{s['p90']:.6g},{s['min']:.6g},{s['max']:.6g}\n")
    print(f"Wrote {sm_csv}")

    # Pretty text version (the at-a-glance matrix)
    sm_txt = os.path.join(args.out_dir, "summary_matrix.txt")
    lines = []
    lines.append("=" * 78)
    lines.append(f"ACCURACY MATRIX — {len(per_case)} unseen cases (NORNE_061..500)")
    lines.append("Per-case error distribution (each case aggregated over ~62 timesteps)")
    lines.append("=" * 78)
    hdr = f"{'Variable':<18}{'Metric':<6}{'mean':>10}{'P50':>10}{'P10':>10}{'P90':>10}{'max(worst)':>12}"
    lines.append(hdr)
    lines.append("-" * 78)
    for v in VARS:
        for m in ("rmse", "mae"):
            s = summary[v][m]
            lines.append(f"{v+' ('+UNITS[v]+')':<18}{m.upper():<6}"
                         f"{s['mean']:>10.4g}{s['p50']:>10.4g}{s['p10']:>10.4g}{s['p90']:>10.4g}{s['max']:>12.4g}")
    lines.append("=" * 78)
    lines.append("\nError vs. rollout timestep (autoregressive drift):")
    lines.append(f"{'timestep':>10}{'PRES_RMSE(bar)':>16}{'SWAT_RMSE':>12}")
    for ts in all_ts[::max(1, len(all_ts)//12)]:   # ~12 rows
        pr = np.sqrt(ts_se['PRESSURE'][ts][0]/ts_se['PRESSURE'][ts][2]) if ts in ts_se['PRESSURE'] else float('nan')
        sw = np.sqrt(ts_se['SWAT'][ts][0]/ts_se['SWAT'][ts][2]) if ts in ts_se['SWAT'] else float('nan')
        lines.append(f"{ts:>10}{pr:>16.4g}{sw:>12.4g}")
    txt = "\n".join(lines)
    with open(sm_txt, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\nWrote {sm_txt}")

    # ---- heatmap (optional) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cases = [r["case"] for r in per_case]
        for v in VARS:
            M = np.full((len(cases), len(all_ts)), np.nan)
            for ci, case in enumerate(cases):
                for ti, ts in enumerate(all_ts):
                    if ts in heat[case][v]:
                        M[ci, ti] = heat[case][v][ts][0]  # rmse
            plt.figure(figsize=(12, max(4, len(cases)*0.04)))
            plt.imshow(M, aspect="auto", cmap="viridis")
            plt.colorbar(label=f"{v} RMSE ({UNITS[v]})")
            plt.xlabel("rollout timestep"); plt.ylabel("case")
            plt.title(f"{v} prediction error — {len(cases)} unseen cases x timesteps")
            png = os.path.join(args.out_dir, f"heatmap_{v}.png")
            plt.tight_layout(); plt.savefig(png, dpi=110); plt.close()
            print(f"Wrote {png}")
    except Exception as e:
        print(f"(heatmap skipped: {e})")

    # machine-readable summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"num_cases": len(per_case), "summary": summary}, f, indent=2, default=float)
    print("\nDone.")


if __name__ == "__main__":
    main()
