"""
Batch runner for Eclipse simulations on the LHS-sampled Norne cases.

Iterates through every NORNE_<n>/ subfolder under <dataset_dir>, runs Eclipse via eclrun,
and logs per-case status. Resumable: cases that already produced a .UNRST file are skipped.

Logs go to <dataset_dir>/run_status.csv. Per-case stdout/stderr lands in <case>/run.log.

Usage:
    python run_eclipse_batch.py
    python run_eclipse_batch.py --dataset_dir ../dataset/NORNE_LHS
    python run_eclipse_batch.py --start 1 --end 50          # run only cases 1-50
    python run_eclipse_batch.py --rerun_failed               # retry cases that previously failed
    python run_eclipse_batch.py --parallel 2                 # 2 concurrent eclrun processes
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ECLRUN = r"C:\ecl\macros\eclrun.exe"
CASE_RE = re.compile(r"^NORNE_(\d+)$")


def discover_cases(dataset_dir: Path):
    """Return sorted [(case_id, case_dir), ...] for all NORNE_<n>/ subfolders."""
    cases = []
    for entry in dataset_dir.iterdir():
        if not entry.is_dir():
            continue
        m = CASE_RE.match(entry.name)
        if m:
            cases.append((int(m.group(1)), entry))
    cases.sort(key=lambda x: x[0])
    return cases


def case_state(case_dir: Path) -> str:
    """Return 'done' (.UNRST exists and >0 bytes), 'failed' (.PRT exists but no .UNRST),
    or 'pending' (nothing has run)."""
    base = case_dir / case_dir.name
    unrst = Path(str(base) + ".UNRST")
    prt = Path(str(base) + ".PRT")
    if unrst.exists() and unrst.stat().st_size > 0:
        return "done"
    if prt.exists() and prt.stat().st_size > 0:
        return "failed"
    return "pending"


def run_one_case(case_id: int, case_dir: Path, timeout_min: int) -> dict:
    """Run eclrun on a single case. Returns a status dict."""
    case_name = case_dir.name
    log_file = case_dir / "run.log"
    started = time.time()

    try:
        with open(log_file, "w") as f:
            proc = subprocess.run(
                [ECLRUN, "eclipse", case_name],
                cwd=str(case_dir),
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout_min * 60,
            )
            exit_code = proc.returncode
            timed_out = False
    except subprocess.TimeoutExpired:
        exit_code = -1
        timed_out = True

    elapsed = time.time() - started
    state = case_state(case_dir)

    return {
        "case_id": case_id,
        "case_name": case_name,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_sec": int(elapsed),
        "final_state": state,
    }


def append_status(status_csv: Path, row: dict):
    """Append one row to the status CSV (creating header if needed)."""
    new_file = not status_csv.exists()
    with open(status_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "case_name", "exit_code", "timed_out", "elapsed_sec", "final_state", "ts"])
        if new_file:
            w.writeheader()
        row = dict(row)
        row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="../dataset/NORNE_LHS")
    p.add_argument("--start", type=int, default=None, help="Inclusive lower case_id bound")
    p.add_argument("--end", type=int, default=None, help="Inclusive upper case_id bound")
    p.add_argument("--rerun_failed", action="store_true", help="Retry cases that previously failed")
    p.add_argument("--timeout_min", type=int, default=60, help="Per-case timeout in minutes")
    p.add_argument("--parallel", type=int, default=1, help="Number of concurrent eclrun processes")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = (script_dir / args.dataset_dir).resolve()
    if not dataset_dir.exists():
        sys.exit(f"ERROR: dataset dir does not exist: {dataset_dir}")

    cases = discover_cases(dataset_dir)
    if args.start is not None:
        cases = [c for c in cases if c[0] >= args.start]
    if args.end is not None:
        cases = [c for c in cases if c[0] <= args.end]

    # Filter by state
    pending = []
    skipped = 0
    for cid, cdir in cases:
        st = case_state(cdir)
        if st == "done":
            skipped += 1
            continue
        if st == "failed" and not args.rerun_failed:
            skipped += 1
            continue
        pending.append((cid, cdir))

    status_csv = dataset_dir / "run_status.csv"
    print(f"Dataset:          {dataset_dir}")
    print(f"Total cases:      {len(cases)}")
    print(f"Already complete: {skipped}")
    print(f"To run:           {len(pending)}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Per-case timeout: {args.timeout_min} minutes")
    print(f"Status CSV:       {status_csv}")
    print()

    if not pending:
        print("Nothing to do.")
        return

    started_all = time.time()
    completed = 0

    if args.parallel <= 1:
        for cid, cdir in pending:
            print(f"[case {cid:03d}] starting {cdir.name}...")
            result = run_one_case(cid, cdir, args.timeout_min)
            append_status(status_csv, result)
            completed += 1
            elapsed_total = time.time() - started_all
            avg = elapsed_total / completed
            remaining_est = avg * (len(pending) - completed) / 60
            print(f"[case {cid:03d}] {result['final_state']:8s}  rc={result['exit_code']:>3d}  {result['elapsed_sec']}s   (avg {int(avg)}s/case, ~{remaining_est:.1f} min remaining)")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one_case, cid, cdir, args.timeout_min): (cid, cdir) for cid, cdir in pending}
            for fut in as_completed(futures):
                result = fut.result()
                append_status(status_csv, result)
                completed += 1
                cid = result["case_id"]
                print(f"[case {cid:03d}] {result['final_state']:8s}  rc={result['exit_code']:>3d}  {result['elapsed_sec']}s  ({completed}/{len(pending)} done)")

    total = time.time() - started_all
    print()
    print(f"All done in {total/60:.1f} minutes ({total:.0f} seconds)")


if __name__ == "__main__":
    main()
