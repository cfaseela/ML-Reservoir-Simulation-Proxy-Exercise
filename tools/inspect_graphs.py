"""Inspect a few xmgn-built graphs to see what's actually in them."""

import sys
from pathlib import Path
import torch
import numpy as np

GRAPHS_DIR = Path(r"D:\NORNE\completed.dataset\XMGN_Norne_PipelineTest_60\graphs")

# Pick a handful of files from different cases/timesteps
all_files = sorted(GRAPHS_DIR.glob("*.pt"))
print(f"Total graph files in dir: {len(all_files)}")

if not all_files:
    sys.exit("No .pt graph files found")

# Sample 3 graphs: first, middle, last (from arbitrary cases/timesteps)
sample_indices = [0, len(all_files)//2, len(all_files)-1]

for idx in sample_indices:
    fp = all_files[idx]
    print("\n" + "=" * 75)
    print(f"GRAPH FILE: {fp.name}")
    print("=" * 75)
    g = torch.load(fp, weights_only=False)
    print(f"PyG Data type:        {type(g).__name__}")
    print(f"Number of nodes:      {g.num_nodes:,}")
    print(f"Number of edges:      {g.num_edges:,}")
    print(f"Avg degree:           {g.num_edges / g.num_nodes:.2f}")

    print(f"\nAll attributes stored on this graph:")
    for k, v in g.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}  shape={tuple(v.shape):<18}  dtype={str(v.dtype):14s}  is_node_attr={g.is_node_attr(k)}  is_edge_attr={g.is_edge_attr(k)}")
        else:
            print(f"  {k:25s}  type={type(v).__name__}  value={v}")

    # Show sample static feature values for 5 active cells
    print(f"\nFirst 5 cells (node) feature values:")
    static_attrs = []
    dynamic_attrs = []
    for k in g.keys():
        if k in ("edge_index", "edge_attr"):
            continue
        v = g[k] if isinstance(g[k], torch.Tensor) else None
        if v is None or not g.is_node_attr(k):
            continue
        if v.ndim == 1:
            static_attrs.append((k, v))
        else:
            # likely dynamic with shape [N, T] or similar
            dynamic_attrs.append((k, v))

    if static_attrs:
        names = [n for n, _ in static_attrs]
        print(f"  {'cell':<6}  " + "  ".join(f"{n:>12s}" for n in names))
        for i in range(min(5, g.num_nodes)):
            vals = [v[i].item() for _, v in static_attrs]
            print(f"  {i:<6d}  " + "  ".join(f"{x:>12.4g}" for x in vals))

    # Show edge connectivity + edge attrs for a few edges
    if hasattr(g, 'edge_index') and g.edge_index is not None:
        print(f"\nFirst 5 edges (src -> dst) + edge_attr sample:")
        ei = g.edge_index
        ea = g.edge_attr if hasattr(g, 'edge_attr') and g.edge_attr is not None else None
        print(f"  edge_index shape: {tuple(ei.shape)}")
        if ea is not None:
            print(f"  edge_attr shape:  {tuple(ea.shape)}")
        for i in range(min(5, ei.shape[1])):
            src, dst = ei[0, i].item(), ei[1, i].item()
            attr_str = ""
            if ea is not None:
                attr_str = "  attrs=[" + ", ".join(f"{v:.4g}" for v in ea[i].tolist()) + "]"
            print(f"  edge {i}: {src} -> {dst}{attr_str}")
