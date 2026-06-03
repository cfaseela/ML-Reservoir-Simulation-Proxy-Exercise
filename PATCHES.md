# Patches Applied to the NVIDIA X-MeshGraphNet Example

This package ships NVIDIA's `examples/reservoir_simulation/xmgn` (Apache-2.0)
with a small number of fixes pre-applied so the pipeline runs end-to-end on a
standard Linux/WSL2 + CUDA setup. Each patch is documented below with **what**
changed, **why**, and **the engineering lesson** — because understanding these
is part of the assessment.

> The upstream example is genuinely excellent ML, but — like much research code —
> it was demoed at smaller scale on the authors' own (Linux, big-GPU) environment.
> The gaps below only surface when you run it at real scale on different hardware.

---

## Patch 1 — PhysicsNeMo ≥ 1.3.0 module relocation

**Files**: `xmgn/src/train.py`, `xmgn/src/inference.py`

**What changed**: import paths updated from the old `physicsnemo.utils.*`
hierarchy to the new `physicsnemo.launch.*` hierarchy:

```python
# Before (xmgn as shipped — targets an older physicsnemo):
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.mlflow import initialize_mlflow
from physicsnemo.utils import load_checkpoint, save_checkpoint

# After (works with nvidia-physicsnemo==1.3.0):
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.utils.checkpoint import load_checkpoint, save_checkpoint
```

**Why**: PhysicsNeMo was reorganized (post-Modulus-rebrand). The current pip
package `nvidia-physicsnemo==1.3.0` no longer has `physicsnemo.utils.logging`;
those modules moved under `physicsnemo.launch`. The xmgn example was written
against the older layout, so it fails at import with
`ModuleNotFoundError: No module named 'physicsnemo.utils.logging'`.

**Lesson**: framework version skew is a first-class hazard. Research examples
rarely pin the exact framework version they were written against, so an example
"that worked when published" breaks against the latest release. Always pin the
framework version (we pin `nvidia-physicsnemo==1.3.0` in `requirements.txt`).

---

## Patch 2 — Streaming statistics (memory scaling)

**File**: `xmgn/src/data/dataloader.py`, function `compute_global_statistics`

**What changed**: replaced the original implementation, which concatenated every
per-node feature from every graph into one giant tensor before computing
mean/std, with a **streaming Welford's algorithm** that maintains running
`(count, mean, M2)` accumulators and processes one graph at a time.

**Why**: the original does, in effect:

```python
all_node_features = [g.x for g in all_graphs]   # holds EVERYTHING in RAM
all_nodes = torch.cat(all_node_features, dim=0)  # one ~9 GB allocation for 60 cases
node_mean = all_nodes.mean(dim=0)
```

Memory scales **linearly with dataset size**:
- 60 cases  → ~9.2 GB  → OOMs a 16 GB box / WSL2 default RAM
- 500 cases → ~72 GB  → OOMs essentially any workstation

We observed a hard SIGKILL (OOM) at Step 3 on a 60-case dataset. The streaming
version uses O(num_features) memory (a few KB) regardless of dataset size, and
produces a **mathematically identical** result (Welford is exact, not an
approximation).

**Lesson**: the classic "research code doesn't scale" trap. Accumulating
everything in a list then concatenating is fine for a 10-case demo, catastrophic
at 500. Before scaling any pipeline, audit the memory complexity of each stage —
anything O(dataset_size) in RAM is a red flag. Welford / streaming reductions are
the standard fix.

---

## Patch 3 — Crash-resilient resume default

**File**: `xmgn/src/preprocessor.py`, function `check_existing_data`

**What changed**: in a non-interactive environment, the preprocessor now
defaults to **"use existing data, run only the missing steps"** instead of
**"overwrite everything from scratch"**.

```python
# Before: non-interactive auto-selects 'y' (overwrite all), redoing Steps 1-2
#         from scratch even after a crash in Step 3.
# After:  non-interactive auto-selects 'n' (keep completed steps), so re-running
#         after a crash resumes instead of restarting a ~40-minute job.
```

**Why**: preprocessing has 5 sequential steps (graph build → partition → split →
statistics → metadata). Steps 1-2 take ~40 min on a 60-case set. If Step 3
crashes (e.g., the OOM that Patch 2 fixes), the original would redo Steps 1-2 on
the next run. Defaulting to "use existing" lets the built-in resume logic skip
completed steps.

**Lesson**: long multi-stage pipelines should be idempotent and resumable.
"Start fresh" as the non-interactive default punishes every crash with a full
restart. Prefer "resume what's done" and make overwrite explicit.

---

## Patch 4 — Idempotent per-graph partition skip

**File**: `xmgn/src/preprocessor.py`, function `create_partitions_from_graphs`

**What changed**: the partition loop now skips graphs whose partition `.pt`
already exists on disk, provided the case's partition assignment is already
populated in memory:

```python
for graph_file in tqdm(graph_files):
    case_name = ...
    partition_file = os.path.join(self.partitions_dir, f"partitions_{basename}")
    if (case_name in partition_assignments_by_case
            and os.path.exists(partition_file)
            and os.path.getsize(partition_file) > 0):
        successful_partitions += 1
        continue
    # ... otherwise do the expensive METIS + halo + torch.save ...
```

**Why**: partitioning is the expensive preprocessing step — ~1 second per graph
on a modern CPU, so several hours for a few thousand graphs. Any crash midway
(OOM, full disk, WSL hiccup) used to mean restarting from zero. With this skip,
re-running picks up where it left off. The "case in dict" check ensures the
per-case JSON of partition assignments still gets written for cases whose first
timestep we DO process this run (otherwise the JSON-write loop has no data).

**Lesson**: idempotent stages aren't free — you need a way to skip what's done
*and* preserve any side-effects that depend on it. The naive "skip if output
exists" forgets the dict population step, breaking the JSON write. Per-file
existence + per-case membership is the right combination here.

---

## Patch 5 — Three-part case-name parsing

**File**: `xmgn/src/inference.py`, function `_extract_case_and_timestep`

**What changed**: added a branch for 3-part filenames like `NORNE_001_012`
(Norne dataset format) alongside the original 4-part `CASE_2D_1_000` format
(xaeronet/other datasets). Without this, the fallback branch was treating each
(case, timestep) pair as its own single-timestep "case":

```python
parts = filename.split("_")
if len(parts) >= 4:
    # Original: CASE_2D_1_000  -> case = "CASE_2D_1", ts = "000"
    case_name = "_".join(parts[:-1])
    timestep = parts[-1]
elif len(parts) == 3:
    # Added: NORNE_001_012  -> case = "NORNE_001", ts = "012"
    case_name = "_".join(parts[:-1])
    timestep = parts[-1]
else:
    case_name = filename
    timestep = "000"  # fallback (broken for autoregressive rollout)
```

**Why**: under the bug, the inference loop log shows `[N/26784] Processing
case: NORNE_xxx_yyy` instead of `[N/432] Processing case: NORNE_xxx
(62 timesteps)`. The 26784 cases × 1 timestep view disables autoregressive
rollout entirely — every prediction is a one-shot from true inputs, never
feeding back its own output. Aggregate numbers are still computable but the
*evolution* they're meant to capture isn't. Easy to miss because losses still
look reasonable.

**Lesson**: when a fallback path produces "almost right" output, it's harder to
catch than a crash. A parser that silently treats heterogeneous inputs as if
they were homogeneous corrupts the semantics. Always assert the parse instead
of falling through.

---

## Patch 6 — Per-case streaming HDF5 + running-sum aggregate metrics

**File**: `xmgn/src/inference.py`, function `run_inference`

**What changed**: the inference loop no longer accumulates per-case results
in memory until the end. Each case's HDF5 is written and freed as soon as the
rollout for that case finishes:

```python
# Before: case_results[case_name] grows for ALL cases; HDF5s saved in one batch
#         at the end; case_results held in memory throughout.
# After:  after each case completes, save its HDF5 immediately, then `del`
#         case_results[case_name] to free memory. Aggregate stats are tracked
#         as running sums (abs_err_sum_per_var, sq_err_sum_per_var,
#         cells_per_var) — exact, not approximate.
```

**Why**: each test case is ~50 MB of predictions + targets + losses
(`~62 timesteps × ~44K cells × 2 vars × 8 bytes × 2`). For a large eval set
that's tens of GB peak — past typical workstation memory caps → swap-thrash
→ inference hangs at 97% CPU but no progress. Streaming the per-case writes
caps memory at O(one case) regardless of how many cases. The running-sum
metrics produce identical values to "concatenate all and compute" but never
materialize the full tensor (same Welford trick as Patch 2, applied to
MAE/RMSE).

**Lesson**: scaling inference has the same memory pitfalls as scaling training
data. Anything `O(num_cases × per_case_size)` in RAM is a red flag at
production scale.

---

## Patch 7 — METIS-partition order → EGRID-natural order in HDF5 output

**File**: `xmgn/src/inference.py`, function `run_inference` (post-rollout reorder)

**What changed**: predictions are reordered to **EGRID natural (i, j, k)
active-cell order** before being written to HDF5, instead of being saved in
the order the inference loop's partition iteration produced them.

```python
# Before: predictions = [partition_0_inner, partition_1_inner, partition_2_inner]
#         concatenated and saved in HDF5 directly.
# After:  build the partition->natural permutation once per case from
#         partition.part_node[partition.inner_node], then
#             pred_natural[perm] = pred_partition_order
#         before saving. HDF5 root group also gets attrs["ordering"] = "natural"
#         so downstream tools can verify.
```

**Why**: the upstream inference iterates the model over METIS-partitioned
subgraphs, concatenates the resulting inner-node predictions, and saves the
concatenated vector as the HDF5 dataset. That concatenation is in
*partition* order — not the same as the EGRID's natural active-cell order
that `.UNRST` files use. Any downstream consumer that maps the prediction
vector back onto the grid (a GRDECL writer, our `hdf5_to_unrst.py`, or
anything that builds a 3D color map) will put each cell's value at the
*wrong physical location*. Aggregate metrics (RMSE/MAE) are robust to this
because they sum over all cells, but **spatial visualization is completely
scrambled** — and visually correct-looking, because the value distribution
is preserved.

We caught this when ResInsight showed an obviously-wrong water front at the
first prediction step for a test case. The diagnostic: HDF5 TARGETS and the
truth UNRST PRESSURE at the same timestep had max |diff| of ~45 bar
cell-by-cell but max |diff| of 0.0 after sorting both vectors — same values,
different order.

**Lesson**: research code often outputs data in whatever order the loop
produces it, without naming the convention or asserting it on read. A
permutation bug in spatial data is the most insidious kind: it doesn't
change any aggregate metric you'd look at, but it makes every per-cell
interpretation false. **Always store the ordering convention as an
attribute** so downstream consumers can refuse stale-format inputs.

---

## NOT applied here (Windows-only, intentionally excluded)

These fixes were needed when we briefly attempted native-Windows execution. They
are **not** in this package because the supported platform is Linux/WSL2, where
they're unnecessary — and one is actively harmful on Linux.

- **SimplePartition bypass** (`preprocessor.py`): on native Windows, PyG's
  `ClusterData`/METIS C-extension segfaults on Norne-scale graphs, so we forced
  the pure-Python `SimplePartition` fallback. On Linux this is **reverted** —
  Linux PyG's METIS works correctly and produces better-balanced partitions.
  (The upstream try/except METIS→SimplePartition fallback is left intact.)
- **`torch+cu121` explicit index** (install-time): PyPI's Windows torch wheel is
  CPU-only; Linux's is already CUDA. See `SETUP.md`.
- **`PYTHONUTF8=1`**: Windows stdio defaults to cp1252 and crashes on the `→`
  characters in xmgn's log messages. Linux defaults to UTF-8. See
  `TROUBLESHOOTING.md`.

---

## Summary table

| # | File | Change | Class | Applied? |
|---|------|--------|-------|----------|
| 1 | train.py, inference.py | physicsnemo.utils.* → physicsnemo.launch.* | Version compat | ✅ |
| 2 | dataloader.py | torch.cat → streaming Welford | Memory scaling | ✅ |
| 3 | preprocessor.py | non-interactive resume default | Pipeline robustness | ✅ |
| 4 | preprocessor.py | idempotent per-graph partition skip | Pipeline robustness | ✅ |
| 5 | inference.py | 3-part case-name parsing (NORNE_xxx_yyy) | Dataset compat | ✅ |
| 6 | inference.py | per-case streaming HDF5 + running-sum metrics | Memory scaling | ✅ |
| 7 | inference.py | METIS→natural cell-ordering in HDF5 output | Spatial correctness | ✅ |
| — | preprocessor.py | SimplePartition bypass | Windows-only | ❌ (reverted on Linux) |
