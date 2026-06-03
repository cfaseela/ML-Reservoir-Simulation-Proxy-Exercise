"""
Generate Latin Hypercube Sampled cases for the Norne deck.

For each of N cases, creates a folder containing:
- NORNE_<n>.DATA            (hard link to base deck — identical content, different name)
- INCLUDE/                  (hard-linked tree to base, EXCEPT...)
  - INCLUDE/FAULT/FAULTMULT_AUG-2006.INC   (per-case modified multipliers)

The 47 fault multipliers from the base FAULTMULT_AUG-2006.INC are each scaled
by a factor sampled log-uniformly in [10^-1, 10^+1] = [0.1, 10] via Latin
Hypercube Sampling, so each case explores the fault-multiplier space around
the geological baseline values.

Outputs:
- One case folder per sample under <output_dir>/NORNE_<n>/
- lhs_design.csv at <output_dir>/ — the full design matrix (rows = cases, cols = fault names)

Usage:
    python generate_lhs_cases.py --n 500 --output_dir ../dataset/NORNE_LHS --base norne_base [--seed 42]
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np


FAULT_LINE_RE = re.compile(
    r"^\s*'([A-Za-z0-9_]+)'\s+([\-+]?\d*\.?\d+(?:[eE][\-+]?\d+)?)\s*/"
)


def parse_faultmult(path: Path):
    """Return (header_lines, list of (fault_name, baseline_multiplier), footer_lines)."""
    text = path.read_text().splitlines()
    header, faults, footer = [], [], []
    seen_multflt = False
    seen_terminator = False
    for line in text:
        if not seen_multflt:
            header.append(line)
            if line.strip() == "MULTFLT":
                seen_multflt = True
            continue
        if seen_terminator:
            footer.append(line)
            continue
        m = FAULT_LINE_RE.match(line)
        if m:
            faults.append((m.group(1), float(m.group(2))))
        else:
            stripped = line.strip()
            if stripped == "/" and faults:
                seen_terminator = True
                footer.append(line)
            else:
                header.append(line)
    return header, faults, footer


def latin_hypercube(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """Return n_samples × n_dims matrix of values in [0, 1) via LHS."""
    rng = np.random.default_rng(seed)
    cuts = np.arange(n_samples) / n_samples
    out = np.empty((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        jitter = rng.uniform(0.0, 1.0 / n_samples, n_samples)
        out[:, j] = cuts[perm] + jitter
    return out


def replicate_include_tree(src_include: Path, dst_include: Path, faultmult_relpath: Path):
    """Mirror src_include into dst_include using hard links for every file
    except the per-case faultmult file (which is skipped here and written by caller)."""
    for root, dirs, files in os.walk(src_include):
        rel = Path(root).relative_to(src_include)
        target_dir = dst_include / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            rel_file = (rel / fname).as_posix()
            if rel_file == faultmult_relpath.as_posix():
                continue  # skip; caller writes the per-case version
            src = Path(root) / fname
            dst = target_dir / fname
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def write_faultmult(path: Path, header, faults_scaled, footer):
    lines = list(header)
    for name, value in faults_scaled:
        lines.append(f"  '{name:s}'  {value:.6g}  /")
    lines.extend(footer)
    path.write_text("\n".join(lines) + "\n")


def build_case(
    case_id: int,
    base_dir: Path,
    output_dir: Path,
    base_data_file: Path,
    base_faultmult: Path,
    faultmult_relpath: Path,
    header, base_faults, footer,
    lhs_row: np.ndarray,
    scale_log_range: tuple,
):
    """Create one case folder with hard-linked INCLUDEs and a fresh FAULTMULT file."""
    case_name = f"NORNE_{case_id:03d}"
    case_dir = output_dir / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    # Hard-link the .DATA file with the per-case name
    case_data = case_dir / f"{case_name}.DATA"
    try:
        os.link(base_data_file, case_data)
    except OSError:
        shutil.copy2(base_data_file, case_data)

    # Mirror the INCLUDE tree (hard links for everything except faultmult)
    base_include = base_dir / "INCLUDE"
    case_include = case_dir / "INCLUDE"
    replicate_include_tree(base_include, case_include, faultmult_relpath)

    # Write the per-case fault multipliers
    log_lo, log_hi = scale_log_range
    log_factors = log_lo + (log_hi - log_lo) * lhs_row
    factors = 10.0 ** log_factors
    scaled = [(name, baseline * factor) for (name, baseline), factor in zip(base_faults, factors)]
    write_faultmult(case_include / faultmult_relpath, header, scaled, footer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="Number of LHS cases")
    p.add_argument("--seed", type=int, default=42, help="LHS random seed")
    p.add_argument("--output_dir", default="../dataset/NORNE_LHS", help="Where to write cases")
    p.add_argument("--base", default="../norne_base", help="Path to base Norne deck folder")
    p.add_argument("--log_lo", type=float, default=-1.0, help="Lower bound of log10(multiplier scale)")
    p.add_argument("--log_hi", type=float, default=1.0, help="Upper bound of log10(multiplier scale)")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = (script_dir / args.base).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    if not base_dir.exists():
        sys.exit(f"ERROR: base deck dir does not exist: {base_dir}")

    base_data_file = base_dir / "NORNE_ATW2013.DATA"
    if not base_data_file.exists():
        sys.exit(f"ERROR: base .DATA file missing: {base_data_file}")

    faultmult_relpath = Path("FAULT") / "FAULTMULT_AUG-2006.INC"
    base_faultmult = base_dir / "INCLUDE" / faultmult_relpath
    if not base_faultmult.exists():
        sys.exit(f"ERROR: base FAULTMULT file missing: {base_faultmult}")

    header, base_faults, footer = parse_faultmult(base_faultmult)
    n_faults = len(base_faults)
    print(f"Base deck:        {base_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Faults found:     {n_faults}")
    print(f"Fault range:      log10(scale) in [{args.log_lo}, {args.log_hi}]  ->  factor in [{10**args.log_lo:.4g}, {10**args.log_hi:.4g}]")
    print(f"Cases to build:   {args.n}")
    print()

    if n_faults == 0:
        sys.exit("ERROR: parsed zero faults from base FAULTMULT — check the regex / file format")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LHS in [0, 1)
    lhs = latin_hypercube(args.n, n_faults, args.seed)

    # Save the design matrix as CSV for reproducibility / inspection
    csv_path = output_dir / "lhs_design.csv"
    with open(csv_path, "w") as f:
        f.write("case_id," + ",".join(name for name, _ in base_faults) + "\n")
        for i in range(args.n):
            log_factors = args.log_lo + (args.log_hi - args.log_lo) * lhs[i]
            factors = 10.0 ** log_factors
            multipliers = [baseline * factor for (_, baseline), factor in zip(base_faults, factors)]
            f.write(f"{i+1:03d}," + ",".join(f"{m:.6g}" for m in multipliers) + "\n")
    print(f"Wrote LHS design matrix: {csv_path}")
    print()

    # Build cases
    for i in range(args.n):
        build_case(
            case_id=i + 1,
            base_dir=base_dir,
            output_dir=output_dir,
            base_data_file=base_data_file,
            base_faultmult=base_faultmult,
            faultmult_relpath=faultmult_relpath,
            header=header, base_faults=base_faults, footer=footer,
            lhs_row=lhs[i],
            scale_log_range=(args.log_lo, args.log_hi),
        )
        if (i + 1) % 50 == 0 or (i + 1) == args.n:
            print(f"  built case {i+1}/{args.n}")
    print()
    print(f"Done. {args.n} case folders under {output_dir}")


if __name__ == "__main__":
    main()
