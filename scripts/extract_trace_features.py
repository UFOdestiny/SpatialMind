#!/usr/bin/env python3
"""One-time: extract only `trace_constraint_features` (small vectors) from the
heavy cache chunks and save them as compact .npz per (subdir, dataset, split),
so the fusion combiner can iterate without re-loading multi-GB activation chunks.

Writes: {out}/{subdir}/{dataset}/{split}.npz  with arrays `feats` (N x F) and
`sample_ids` (N,). sample_id is the row order in the cache (0..N-1), matching how
the evaluation predictions index samples.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_constraints.analysis import TRACE_FEATURE_NAMES  # noqa: E402

CACHE = "spatialmind/cache/cached_features"
OUT = "spatialmind/cache/trace_features_small"

# (cache_subdir, model_dir)
CELLS = [
    ("constraint_guided_v10", "Llama-3.1-8B-Instruct"),
    ("constraint_guided_v10_mistral7b", "Mistral-7B-Instruct-v0.3"),
    ("constraint_guided_v10_gemma2", "gemma-2-9b-it"),
    ("constraint_guided_v10_phi4reason", "Phi-4-reasoning"),
]
DATASETS = ["StepGame", "spartqa", "babi", "SpaRTUN", "SpaceNLI", "SpartQA_YN"]
SPLITS = ["validation", "test"]
F = len(TRACE_FEATURE_NAMES)


def main():
    for sub, model in CELLS:
        for ds in DATASETS:
            for sp in SPLITS:
                cdir = f"{CACHE}/{sub}/{ds}/{model}/{sp}"
                chunks = sorted(glob.glob(os.path.join(cdir, "chunk_*.pt")))
                if not chunks:
                    continue
                outp = f"{OUT}/{sub}/{ds}/{sp}.npz"
                if os.path.exists(outp):
                    print(f"skip {outp}"); continue
                feats = []
                for c in chunks:
                    rows = torch.load(c, map_location="cpu", weights_only=False)
                    for row in rows:
                        tf = row.get("trace_constraint_features")
                        feats.append(np.asarray(tf, float) if tf is not None
                                     else np.full(F, np.nan))
                    del rows
                arr = np.stack(feats) if feats else np.zeros((0, F))
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                np.savez_compressed(outp, feats=arr,
                                    sample_ids=np.arange(len(arr), dtype=int))
                print(f"wrote {outp}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
