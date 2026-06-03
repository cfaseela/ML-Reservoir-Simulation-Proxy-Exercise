# Dataset

The simulation dataset is **not** stored in git (the binary outputs are tens to
hundreds of GB). It's delivered separately.

## What the dataset is

Norne field simulations (grid 46×112×22, ~44K active cells), each a distinct
realization with **fault transmissibility multipliers** varied via Latin
Hypercube Sampling. Each case was run through the Eclipse simulator and provides:

```
NORNE_<id>/
├── NORNE_<id>.DATA      # simulation deck
├── NORNE_<id>.EGRID     # grid geometry (binary)
├── NORNE_<id>.INIT      # static properties: PERMX, PORV, TRAN... (binary)
├── NORNE_<id>.UNRST     # dynamic state per timestep: PRESSURE, SWAT... (~470 MB)
├── NORNE_<id>.UNSMRY    # well/field summary time series (binary)
└── NORNE_<id>.SMSPEC    # summary metadata (binary)
```

~65 timesteps per case spanning ~9 years of simulated production.

## How to obtain it

**Download:**
[**60-case Norne LHS bundle (SharePoint)**](https://hbkuedu-my.sharepoint.com/:f:/g/personal/aabd_hbku_edu_qa/IgDVHMaIsK_bTa89Gqufe13oATPpgR80ITBv_hlmME7QUkI?e=cZtipD)

The bundle is **60 cases** (`NORNE_001` through `NORNE_060`), ~30 GB
compressed, ~270 GB uncompressed (~470 MB per `.UNRST` is the heavy item).
It already includes all the file types listed above.

Use these 60 cases for preprocessing, training, validation, and your test
split. See `TASK.md` §4 for the mandated split (seeded 80/10/10).

If the SharePoint link prompts for sign-in and your institution doesn't have
access, email us and we'll grant or share via a different channel.

If you can't download (corporate firewall, etc.), email us. Don't try to
regenerate the dataset from scratch unless you have an Eclipse or OPM Flow
license — see the "Regenerating" section below.

## Placement (important for performance)

Put the cases on a **native Linux filesystem**, not a Windows-mounted `/mnt/`
path. Then point your config at them:

```yaml
# conf/<your-config>.yaml
dataset:
  sim_dir: /home/<you>/data/NORNE_LHS    # native ext4, NOT /mnt/d/...
```

If you accidentally extracted to a Windows mount, see `TROUBLESHOOTING.md`
for the migration steps — the difference is roughly 100× I/O throughput
during preprocessing.

For a flat, glob-safe layout (one folder per case, no nested `INCLUDE/`), use
`tools/link_completed.py` as a template. See `TROUBLESHOOTING.md` for why the
flat layout matters for preprocessing.

## Regenerating the dataset from scratch

If you have an Eclipse (or OPM Flow) license, you can generate cases yourself:

```bash
# 0. Get the Norne base deck (not shipped in this repo — fetch from OPM):
#    https://github.com/OPM/opm-data/tree/master/norne
#    Place it at e.g. ./norne_base/

# 1. Generate N LHS-sampled cases from the Norne base deck
python tools/generate_lhs_cases.py --n 500 --output_dir ../dataset/NORNE_LHS \
    --base ./norne_base --seed 42

# 2. Run them through Eclipse (resumable, parallel)
python tools/run_eclipse_batch.py --dataset_dir ../dataset/NORNE_LHS --parallel 4
```

The LHS design (which fault multipliers per case) is written to
`lhs_design.csv` for reproducibility — fixed seed gives identical samples.

> Generating 500 cases takes ~1-2 days of Eclipse compute on a 4-parallel
> workstation and ~250 GB of disk. The 60-case subset is enough for a meaningful
> pipeline test.
