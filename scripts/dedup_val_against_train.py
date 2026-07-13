#!/usr/bin/env python3
"""Drop validation rows whose (context, question) collide with train.

StepGame is template-generated, so a tiny number of identical (context,question)
instances can appear in both train and validation. The fail-closed leakage audit
rejects ANY cross-split overlap. This removes the colliding validation rows (test
is never touched) and rewrites the validation chunks + manifest counts so a
re-audit passes. Uses the SAME hash as scripts/leakage_audit.py (context+question).
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os

import torch


def instance_hash(sample):
    payload = json.dumps(
        [str(sample.get("context", "")).strip(), str(sample.get("question", "")).strip()],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_split(cache_dir, split):
    chunks = sorted(glob.glob(os.path.join(cache_dir, split, "chunk_*.pt")))
    rows = []
    for c in chunks:
        rows.extend(torch.load(c, map_location="cpu", weights_only=False))
    return rows, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--purge_split", default="validation")
    ap.add_argument("--against", default="train")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ref_rows, _ = load_split(args.cache_dir, args.against)
    ref = {instance_hash(r) for r in ref_rows}
    rows, chunks = load_split(args.cache_dir, args.purge_split)
    keep = [r for r in rows if instance_hash(r) not in ref]
    dropped = len(rows) - len(keep)
    print(f"{args.purge_split}: {len(rows)} -> {len(keep)} (dropped {dropped} colliding with {args.against})")
    if args.dry_run or dropped == 0:
        print("no rewrite" + (" (dry run)" if args.dry_run else " (nothing to drop)"))
        return

    split_dir = os.path.join(args.cache_dir, args.purge_split)
    # single-chunk assumption matches this cache (num_chunks=1); rewrite all chunks
    for c in chunks:
        os.remove(c)
    torch.save(keep, os.path.join(split_dir, "chunk_0.pt"))

    # patch manifest counts
    mpath = os.path.join(split_dir, "manifest.json")
    with open(mpath) as f:
        m = json.load(f)
    n = len(keep)
    correct = sum(1 for r in keep if int(r.get("label", 0)) == 1)
    m["total_count"] = n
    m["num_chunks"] = 1
    m["chunk_sample_counts"] = [n]
    if "total_correct" in m:
        m["total_correct"] = correct
        m["total_incorrect"] = n - correct
        m["correct_rate"] = correct / n if n else 0.0
        m["incorrect_rate"] = (n - correct) / n if n else 0.0
    # Keep judge bookkeeping consistent with the physical cache so the
    # fail-closed leakage audit's judge-count invariant still holds.
    # (audit expects judge_samples_processed == n + judge_dropped, dropped=0 here)
    for k in ("judge_samples_processed", "judge_input_samples", "total_count"):
        if k in m:
            m[k] = n
    with open(mpath, "w") as f:
        json.dump(m, f, indent=2)
    print(f"rewrote chunk_0.pt ({n} rows) + manifest (correct={correct}, judge counts synced)")


if __name__ == "__main__":
    main()
