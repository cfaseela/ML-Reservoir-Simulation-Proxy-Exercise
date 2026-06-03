# Reservoir Surrogate Modeling — Technical Assessment

A take-home assessment built around training a neural surrogate for reservoir
simulation, using NVIDIA's [X-MeshGraphNet](https://arxiv.org/pdf/2411.17164)
(X-MGN) on the **Norne field** benchmark.

The candidate trains a graph-neural-network surrogate that predicts the
spatio-temporal evolution of pressure and fluid saturation across a faulted 3D
reservoir — the same class of workflow Stone Ridge Technology runs commercially
on PhysicsNeMo ([NVIDIA spotlight](https://developer.nvidia.com/blog/spotlight-stone-ridge-technology-accelerates-reservoir-simulation-workflows-with-nvidia-physicsnemo-on-aws/)).

## What's in here

| Path | Contents |
|---|---|
| `xmgn/` | NVIDIA's X-MGN example (Apache-2.0), with fixes pre-applied (see `PATCHES.md`). Its `README.md` is the framework deep-dive + Norne visualizations |
| `docs/img/` | Bundled visualization images so `xmgn/README.md` renders offline |
| `sim_utils/` | Eclipse-format binary reader (xmgn dependency) |
| `tools/` | Helper scripts: graph inspection, accuracy matrix builder, HDF5→UNRST converter for ResInsight, representative-case picker, plus operator-only LHS/Eclipse-batch tools |
| `requirements.txt` | Pinned Python dependencies |
| **`SETUP.md`** | Environment setup (WSL2/Linux + Python 3.10 + CUDA + physicsnemo) |
| **`TASK.md`** | The assessment description — start here |
| **`PATCHES.md`** | Every fix we pre-applied, why, and the lesson |
| **`TROUBLESHOOTING.md`** | Symptom → fix for environment gotchas |
| `data/README.md` | How to obtain the dataset (not shipped in git) |

## Quick start

1. Read `TASK.md` (what you're asked to do)
2. Read **`xmgn/README.md`** — NVIDIA's original X-MGN documentation: the
   architecture, dataset-format expectations, the paper reference, and
   **example Norne visualizations** (pressure/saturation TRUE vs PRED vs ERROR
   at multiple timesteps) showing what good surrogate output looks like.
   Images are bundled under `docs/img/` so they render offline.
3. Follow `SETUP.md` (get the environment running)
4. Obtain the dataset per `data/README.md`
5. Run preprocess → train → inference (commands in `SETUP.md` §6)

## Platform note

PhysicsNeMo supports **Linux (Ubuntu 24.04)** and **Windows via WSL2 only** —
native Windows is unsupported. A CUDA-capable NVIDIA GPU is required.

## Attribution & licenses

- **X-MGN example & PhysicsNeMo**: © NVIDIA, Apache-2.0
- **Norne deck**: © Statoil, Open Database License (ODbL) via the
  [OPM project](https://github.com/OPM/opm-data)
- Patches and assessment scaffolding in this repo are documented in `PATCHES.md`.
