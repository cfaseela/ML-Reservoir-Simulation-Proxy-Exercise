"""
Convert X-MGN prediction HDF5 files into synthetic Eclipse-format .UNRST
restart files so ResInsight can load them on the original .EGRID grid.

Workflow for each case:
  1. Open the original simulation's UNRST (the "truth" template)
  2. Clone its keyword stream verbatim, EXCEPT for PRESSURE and SWAT
  3. For each report timestep, substitute model predictions in those two
  4. Write to <case>_PRED.UNRST

After this runs, ResInsight can load NORNE_xxx.EGRID + NORNE_xxx_PRED.UNRST
as a single case showing the model's predictions over time. Loading the
original NORNE_xxx.UNRST separately gives the ground-truth comparison.

Usage:
  python hdf5_to_unrst.py \
    --cases_json /mnt/e/.../representative_cases.json \
    --hdf5_dir /mnt/e/.../inference \
    --cases_dir /mnt/e/NORNE/cases \
    --out_dir   /mnt/e/.../predictions_unrst
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

try:
    from resdata.resfile import ResdataFile, FortIO
    from resdata.resfile import ResdataKW
    from resdata import ResDataType
except ImportError as e:
    sys.exit(f"resdata not installed: {e}\n  pip install resdata")

try:
    import torch
except ImportError as e:
    sys.exit(f"torch not installed: {e}")


def build_natural_order_index(partitions_dir, case_name):
    """Return permutation array P of length nact such that
    `pred_natural[P] = pred_partition_order` puts each METIS-ordered
    prediction at its EGRID-natural-order cell index.

    The HDF5 stores predictions concatenated across partitions in the order
    the inference loop processed them (partition 0, partition 1, partition 2,
    for num_partitions=3 in this config). For each partition, only INNER
    cells contribute. The inner cells' GLOBAL graph-node indices come from
    `partition.part_node[partition.inner_node]`. Concatenating these gives
    the permutation: position i in HDF5 -> natural-order index P[i].

    We load partitions_<case>_002.pt (the first ordinal that has predictions)
    and use ITS partition layout — every timestep in the case shares the
    same spatial partitioning, so one .pt suffices.
    """
    import glob
    # Try organized subdirs first (train/val/test), then top-level
    candidates = (
        glob.glob(os.path.join(partitions_dir, "test", f"partitions_{case_name}_002.pt"))
        + glob.glob(os.path.join(partitions_dir, "*", f"partitions_{case_name}_002.pt"))
        + glob.glob(os.path.join(partitions_dir, f"partitions_{case_name}_002.pt"))
    )
    if not candidates:
        # Fall back to ANY partition file for this case
        candidates = (
            glob.glob(os.path.join(partitions_dir, "test", f"partitions_{case_name}_*.pt"))
            + glob.glob(os.path.join(partitions_dir, "*", f"partitions_{case_name}_*.pt"))
            + glob.glob(os.path.join(partitions_dir, f"partitions_{case_name}_*.pt"))
        )
    if not candidates:
        raise FileNotFoundError(
            f"No partition .pt found for {case_name} under {partitions_dir}"
        )
    pt_path = candidates[0]
    partitions_list = torch.load(pt_path, weights_only=False, map_location="cpu")

    global_indices = []
    for part in partitions_list:
        # part.inner_node: local-to-partition indices of inner cells
        # part.part_node:  global graph indices for ALL nodes in this partition
        # part_node[inner_node] -> global natural-order indices of this partition's inner cells
        inner_global = part.part_node[part.inner_node].cpu().numpy()
        global_indices.append(inner_global)
    return np.concatenate(global_indices).astype(np.int64)


# UNRST stores per-cell values for ACTIVE cells in natural order.
# Our predictions are also in active-cell order (same as INIT/UNRST).
# So we replace values 1:1, no padding needed.

# Keywords we'll replace with predictions (others copied verbatim from truth):
SUBSTITUTE_KEYWORDS = {"PRESSURE", "SWAT"}


def case_predictions(hdf5_path):
    """Return (predictions_dict, ordering) where:
      predictions_dict: {ordinal_report_index: {'PRESSURE': np.array, 'SWAT': np.array}}
      ordering: 'natural' if HDF5 declares attrs["ordering"]=='natural' (patched
                inference.py, EGRID active-cell order); 'partition' otherwise
                (legacy NVIDIA output, must be reordered before mapping back
                to UNRST cell slots).

    HDF5 layout (from inference.py _save_case_results_hdf5):
        predictions/<VAR>/timestep_NNNN    array of denormalized predictions

    NOTE: NNNN is the preprocessor's ORDINAL report-step counter (0..64 for
    Norne — one entry per SEQNUM block in the source UNRST, in order), NOT
    the Eclipse global SEQNUM *value* (which is sparse, e.g. 0,1,2,3,4,5,10,
    19,27,33,...,241 for Norne_074). Predictions are FOR the next ordinal,
    so the dataset named timestep_0003 contains predictions for the 3rd
    SEQNUM block (0-indexed) in the source UNRST.
    """
    out = {}
    with h5py.File(hdf5_path, "r") as f:
        ordering = str(f.attrs.get("ordering", "partition"))
        if "predictions" not in f:
            return out, ordering
        pred = f["predictions"]
        for var in SUBSTITUTE_KEYWORDS:
            if var not in pred:
                continue
            for ts_name, dset in pred[var].items():
                # ts_name like "timestep_0007" -> ordinal report index 7
                ordinal = int(ts_name.split("_")[-1])
                out.setdefault(ordinal, {})[var] = dset[:]
    return out, ordering


def write_pred_unrst(truth_unrst_path, predictions_by_ts, out_unrst_path, natural_order_perm=None):
    """Clone truth_unrst_path, swapping PRESSURE/SWAT values in each report.

    If natural_order_perm is provided, the prediction arrays (which come from
    the HDF5 in METIS-partition order) are reordered to EGRID natural order
    before being written into the UNRST cell slots:
        natural[perm] = partition_order
    This is required because NVIDIA's inference HDF5 stores predictions in
    the order the inference loop processed partitions, NOT in EGRID's
    natural (i,j,k) active-cell order that UNRST expects.

    SOIL guard: SWAT predictions are clipped to [0, 1 - SGAS_truth] before
    writing, so the derived SOIL = 1 - SWAT - SGAS_truth stays in [0, 1].
    The clipping amount equals the model's unconstrained-output physics
    violation (typically ~0.001 per cell, 25-30% of cells per timestep).
    """
    # Pre-scan: collect SGAS arrays per ordinal so we know the truth gas
    # saturation when we get around to clipping each SWAT prediction.
    # Eclipse stores PRESSURE -> SWAT -> SGAS in that order within each
    # report, so a single-pass approach would not know SGAS yet at SWAT time.
    pre = ResdataFile(truth_unrst_path)
    sgas_by_ordinal = {}
    pre_ordinal = -1
    for kw in pre:
        n = kw.getName().strip()
        if n == "SEQNUM":
            pre_ordinal += 1
        elif n == "SGAS":
            sgas_by_ordinal[pre_ordinal] = kw.numpy_view().copy()
    del pre
    truth = ResdataFile(truth_unrst_path)
    os.makedirs(os.path.dirname(out_unrst_path), exist_ok=True)

    # Track the ORDINAL report index (0, 1, 2, ...) — incremented each time
    # we see a SEQNUM keyword. This is what the preprocessor used to number
    # graphs/predictions, NOT the SEQNUM *value* (which is the sparse Eclipse
    # global step ID).
    current_ordinal = -1  # incremented to 0 at first SEQNUM
    n_replaced = 0
    n_kw_total = 0
    n_reports = 0
    seqnum_values = []

    # FortIO doesn't implement __enter__ in this version of resdata — use try/finally
    out_fortio = FortIO(out_unrst_path, mode=FortIO.WRITE_MODE)
    try:
        for kw in truth:
            n_kw_total += 1
            name = kw.getName().strip()

            if name == "SEQNUM":
                current_ordinal += 1
                seqnum_values.append(int(kw[0]))
                n_reports += 1
                kw.fwrite(out_fortio)
                continue

            if name in SUBSTITUTE_KEYWORDS and current_ordinal >= 0:
                preds_for_step = predictions_by_ts.get(current_ordinal)
                if preds_for_step is not None and name in preds_for_step:
                    pred_values = preds_for_step[name]
                    truth_size = len(kw)
                    if pred_values.size != truth_size:
                        # Active-cell count mismatch — leave truth in place
                        # (better than corrupt UNRST). Log so we notice.
                        print(
                            f"  WARN ordinal={current_ordinal} {name}: "
                            f"truth={truth_size} pred={pred_values.size} — keeping truth"
                        )
                        kw.fwrite(out_fortio)
                        continue
                    # Reorder HDF5 predictions from METIS-partition-order
                    # to EGRID-natural-order. Without this, every cell's
                    # value lands at the wrong physical location and
                    # ResInsight shows scrambled spatial patterns.
                    if natural_order_perm is not None:
                        pred_natural = np.empty(truth_size, dtype=np.float32)
                        pred_natural[natural_order_perm] = pred_values.astype(np.float32)
                    else:
                        pred_natural = pred_values.astype(np.float32)

                    # SOIL guard: when writing SWAT, clip to [0, 1 - SGAS_truth]
                    # so the derived oil saturation stays in [0, 1]. Without
                    # this the unconstrained NN can output SWAT slightly > 1
                    # in ~30% of cells, making SOIL go negative in ResInsight.
                    # PRESSURE has no analogous physical [0, 1] constraint.
                    if name == "SWAT":
                        sgas = sgas_by_ordinal.get(current_ordinal)
                        if sgas is not None:
                            max_swat = 1.0 - sgas.astype(np.float32)
                            # also keep SWAT >= 0 (no negative saturations)
                            pred_natural = np.clip(pred_natural, 0.0, max_swat)
                        else:
                            # No SGAS for this report — just keep SWAT in [0,1]
                            pred_natural = np.clip(pred_natural, 0.0, 1.0)

                    # Mutate the kw's buffer in-place via numpy_view (a writable
                    # view into the C-level buffer). Older attempts used
                    # np.asarray(new_kw)[:] = ... which produces a COPY, not a
                    # view, so the kw's actual data stayed zero-initialized.
                    view = kw.numpy_view()
                    view[:] = pred_natural.astype(view.dtype)
                    kw.fwrite(out_fortio)
                    n_replaced += 1
                    continue

            kw.fwrite(out_fortio)
    finally:
        out_fortio.close()
    return {
        "kw_total": n_kw_total,
        "kw_replaced": n_replaced,
        "n_reports": n_reports,
        "seqnums": seqnum_values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_json", required=True)
    ap.add_argument("--hdf5_dir", required=True)
    ap.add_argument("--cases_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--partitions_dir",
        default="/root/NORNE/eval.dataset/XMGN_Norne_Eval440/partitions",
        help="Where the preprocessor's partition .pt files live — needed to "
             "recover the METIS-partition-order -> natural-order permutation",
    )
    args = ap.parse_args()

    with open(args.cases_json) as f:
        cases_spec = json.load(f)
    cases = cases_spec["cases"]
    annotations = cases_spec.get("annotations", {})

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Processing {len(cases)} cases:")

    summary = []
    for case_name in cases:
        label = annotations.get(case_name, "")
        hdf5_path = os.path.join(args.hdf5_dir, f"{case_name}.hdf5")
        truth_unrst = os.path.join(args.cases_dir, case_name, f"{case_name}.UNRST")
        out_unrst = os.path.join(args.out_dir, f"{case_name}_PRED.UNRST")

        if not os.path.exists(hdf5_path):
            print(f"  [{label:7s}] {case_name}: HDF5 missing at {hdf5_path}")
            continue
        if not os.path.exists(truth_unrst):
            print(f"  [{label:7s}] {case_name}: UNRST missing at {truth_unrst}")
            continue

        preds, ordering = case_predictions(hdf5_path)
        if not preds:
            print(f"  [{label:7s}] {case_name}: no predictions found in HDF5")
            continue

        # If the HDF5 self-reports natural order (patched inference.py wrote
        # attrs["ordering"]=="natural"), the predictions are already in the
        # right order for UNRST substitution. Skip the partition-file lookup.
        if ordering == "natural":
            perm = None
            print(f"  [{label:7s}] {case_name}: HDF5 in natural order — no reorder needed")
        else:
            # Legacy / unpatched HDF5 — reconstruct the permutation from the
            # partition .pt files for this case.
            try:
                perm = build_natural_order_index(args.partitions_dir, case_name)
            except Exception as e:
                print(f"  [{label:7s}] {case_name}: WARNING — could not load partition layout ({e})")
                print(f"                      writing in HDF5 order (will appear scrambled in ResInsight)")
                perm = None

        stats = write_pred_unrst(truth_unrst, preds, out_unrst, natural_order_perm=perm)
        rep_range = (
            f"{stats['seqnums'][0]}..{stats['seqnums'][-1]}"
            if stats["seqnums"]
            else "none"
        )
        print(
            f"  [{label:7s}] {case_name}: "
            f"{stats['n_reports']} reports ({rep_range}), "
            f"{stats['kw_replaced']} PRESSURE/SWAT kws replaced "
            f"of {stats['kw_total']} total kws  ->  {out_unrst}"
        )
        summary.append(
            {
                "case": case_name,
                "label": label,
                "out": out_unrst,
                **stats,
            }
        )

    # Copy EGRID alongside for ResInsight convenience.
    # IMPORTANT: must be a REAL FILE COPY, not a symlink. WSL's `ln -s` on /mnt
    # creates WSL-flavored reparse points whose targets are Linux paths like
    # /mnt/e/... — native Windows apps (ResInsight, Notepad) can't resolve
    # those. Real copies work cross-app. EGRID is only ~4 MB so the copy is
    # cheap.
    print("\nCopying companion EGRIDs to out_dir...")
    import shutil
    for case_name in cases:
        src_egrid = os.path.join(args.cases_dir, case_name, f"{case_name}.EGRID")
        dst_egrid = os.path.join(args.out_dir, f"{case_name}.EGRID")
        if os.path.exists(src_egrid) and not os.path.exists(dst_egrid):
            shutil.copy2(src_egrid, dst_egrid)

    print(f"\nDone. {len(summary)} _PRED.UNRST files in {args.out_dir}")


if __name__ == "__main__":
    main()
