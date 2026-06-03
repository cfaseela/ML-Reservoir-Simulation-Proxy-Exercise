"""
Hard-link the case-level output files of every completed Eclipse run into a
clean "completed" view that xmgn can ingest without seeing INCLUDE files that
contain .DATA-suffixed sub-files (which would confuse xmgn's recursive glob).

A case is considered "completed" if its .UNRST is > 400 MB.
For each completed case, we hard-link ONLY the case-level output files:
  .DATA, .EGRID, .INIT, .UNRST, .UNSMRY, .SMSPEC
"""

import os
from pathlib import Path

SRC = Path(r"D:\NORNE\cases")
DST = Path(r"D:\NORNE\completed")
MIN_UNRST_MB = 400
EXTS = ["DATA", "EGRID", "INIT", "UNRST", "UNSMRY", "SMSPEC"]

case_dirs = sorted([d for d in SRC.iterdir() if d.is_dir() and d.name.startswith("NORNE_")])
done = []
for cd in case_dirs:
    u = cd / f"{cd.name}.UNRST"
    if u.exists() and u.stat().st_size > MIN_UNRST_MB * 1024 * 1024:
        done.append(cd)

print(f"Source root: {SRC}")
print(f"Dest root:   {DST}")
print(f"Completed cases found: {len(done)}")
if done:
    print(f"  range: {done[0].name} ... {done[-1].name}")
print()

DST.mkdir(parents=True, exist_ok=True)

linked = 0
errors = 0
missing = 0
for case in done:
    case_dst = DST / case.name
    case_dst.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        src_f = case / f"{case.name}.{ext}"
        dst_f = case_dst / f"{case.name}.{ext}"
        if dst_f.exists():
            continue
        if not src_f.exists():
            print(f"  MISSING: {src_f}")
            missing += 1
            continue
        try:
            os.link(src_f, dst_f)
            linked += 1
        except OSError as e:
            print(f"  LINK FAILED {dst_f}: {e}")
            errors += 1

print(f"Linked {linked} files; {missing} missing files; {errors} link errors")
print(f"Hard-link copy uses near-zero disk space.")
