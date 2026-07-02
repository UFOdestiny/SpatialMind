#!/usr/bin/env python3
"""
rechunk_cache.py - Re-chunk an existing Phase-1 feature cache into smaller chunks.

Phase-1 caches store per-trace feature dicts in chunk_*.pt files. Very large
chunks (e.g. 10k traces => 20-45 GB each at feature_dim~4.4k) force training to
either serialize data loading (num_workers=0, slow) or duplicate a huge chunk per
DataLoader worker (OOM). This tool rewrites a cache split into smaller chunks
WITHOUT re-running the LLM: it only redistributes the already-generated sample
dicts, so every field (features, claims, verified, label, ...) is preserved
byte-for-byte. The rewritten cache is a drop-in replacement consumed identically
by CachedFeatureDataset.

Memory-safe: reads ONE source chunk at a time, flushes small target chunks as it
fills them, and never holds the whole split in RAM.

Usage:
    # re-chunk all splits of a dataset/model cache to 2500 traces/chunk, in place
    python utils/rechunk_cache.py \
        --cache_dir spatialmind/cache/cached_features/StepGame/Llama-3.1-8B-Instruct \
        --chunk_size 2500

    # single split, write to a new dir (keep the original)
    python utils/rechunk_cache.py --cache_dir <src> --split train \
        --chunk_size 2500 --out_dir <dst>
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def _list_splits(cache_dir: Path) -> List[str]:
    return sorted(p.name for p in cache_dir.iterdir()
                  if p.is_dir() and (p / "manifest.json").exists())


def rechunk_split(src_split_dir: Path, dst_split_dir: Path, chunk_size: int) -> dict:
    """Rewrite one split's chunks into chunk_size-trace chunks. Returns new manifest."""
    manifest_path = src_split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    src_chunks = sorted(src_split_dir.glob("chunk_*.pt"))
    if not src_chunks:
        raise FileNotFoundError(f"No chunk_*.pt under {src_split_dir}")

    in_place = src_split_dir.resolve() == dst_split_dir.resolve()
    # When rewriting in place, stage to a temp dir then swap, so a crash never
    # leaves a half-rewritten split.
    work_dir = dst_split_dir.parent / (dst_split_dir.name + ".rechunk_tmp") if in_place else dst_split_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    buf: List[dict] = []
    out_idx = 0
    chunk_counts: List[int] = []
    total = 0

    def flush():
        nonlocal buf, out_idx
        if not buf:
            return
        torch.save(buf, work_dir / f"chunk_{out_idx}.pt")
        chunk_counts.append(len(buf))
        log.info("  wrote chunk_%d.pt (%d traces)", out_idx, len(buf))
        out_idx += 1
        buf = []

    for sc in src_chunks:
        data = torch.load(sc, map_location="cpu", weights_only=False)
        for sample in data:
            buf.append(sample)
            total += 1
            if len(buf) >= chunk_size:
                flush()
        del data
        gc.collect()
    flush()

    new_manifest = dict(manifest)
    new_manifest["chunk_size"] = chunk_size
    new_manifest["num_chunks"] = out_idx
    new_manifest["chunk_sample_counts"] = chunk_counts
    new_manifest["total_count"] = total
    (work_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=1))

    if in_place:
        # Atomic-ish swap: remove old chunk_*.pt + manifest, move new files in.
        for old in src_split_dir.glob("chunk_*.pt"):
            old.unlink()
        for f in work_dir.iterdir():
            shutil.move(str(f), str(src_split_dir / f.name))
        shutil.rmtree(work_dir)

    log.info("  split done: %d traces -> %d chunks of <=%d", total, out_idx, chunk_size)
    return new_manifest


def main():
    ap = argparse.ArgumentParser(description="Re-chunk a Phase-1 feature cache into smaller chunks")
    ap.add_argument("--cache_dir", required=True, help="…/<dataset>/<model> cache dir")
    ap.add_argument("--chunk_size", type=int, default=2500)
    ap.add_argument("--split", default=None, help="Single split; default = all splits found")
    ap.add_argument("--out_dir", default=None, help="Write to a new cache dir (default: in place)")
    ap.add_argument("--skip_if_smaller", action="store_true",
                    help="Skip a split whose current chunk_size <= target")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        log.error("cache_dir not found: %s", cache_dir); sys.exit(1)
    out_root = Path(args.out_dir) if args.out_dir else cache_dir

    splits = [args.split] if args.split else _list_splits(cache_dir)
    if not splits:
        log.error("no splits with manifest.json under %s", cache_dir); sys.exit(1)

    log.info("Re-chunking %s -> chunk_size=%d | splits=%s | out=%s",
             cache_dir, args.chunk_size, splits, out_root)
    for split in splits:
        src = cache_dir / split
        cur = json.loads((src / "manifest.json").read_text()).get("chunk_size", 0)
        if args.skip_if_smaller and 0 < cur <= args.chunk_size:
            log.info("[SKIP] %s: chunk_size=%d already <= %d", split, cur, args.chunk_size)
            continue
        log.info("Split '%s' (current chunk_size=%d)…", split, cur)
        rechunk_split(src, out_root / split, args.chunk_size)
    log.info("Done.")


if __name__ == "__main__":
    main()
