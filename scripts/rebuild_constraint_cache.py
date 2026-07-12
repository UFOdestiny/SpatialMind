#!/usr/bin/env python3
"""Recompute the native constraint view from stored context/question/claims.

Generation features and judge labels are left untouched.  This command makes
parser/solver changes reproducible without another expensive backbone forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_constraints import CLAIM_CONSTRAINT_DIM, TRACE_CONSTRAINT_DIM, analyze_trace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--split", default="train,validation,test")
    return p.parse_args()


def rebuild_split(cache_dir: Path, split: str):
    split_dir = cache_dir / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    counts = Counter()
    chunk_counts = []

    for path in sorted(split_dir.glob("chunk_*.pt")):
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        for sample in chunk:
            context = str(sample.get("context", ""))
            question = str(sample.get("question", ""))
            claims = sample.get("claims", []) or []
            source = analyze_trace(context, "", [])
            analysis = analyze_trace(context, question, claims)
            rows = [x.features for x in analysis.claims]
            sample["claim_constraint_features"] = torch.tensor(
                rows, dtype=torch.float32
            ).reshape(len(rows), CLAIM_CONSTRAINT_DIM)
            sample["trace_constraint_features"] = torch.tensor(
                analysis.trace_features, dtype=torch.float32
            )
            sample["constraint_analysis"] = analysis.to_dict()
            counts["samples"] += 1
            counts["source_infeasible"] += int(not source.full_trace_feasible)
            counts["context_available"] += int(bool(analysis.context_relations))
            for claim in analysis.claims:
                counts["claims"] += 1
                counts["parseable"] += int(bool(claim.parsed))
                counts[claim.status] += int(bool(claim.parsed))
                counts["first_conflicts"] += int(claim.first_conflict)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(chunk, tmp)
        os.replace(tmp, path)
        chunk_counts.append(len(chunk))

    manifest.update({
        "cache_schema_version": 2,
        "constraint_method": "explicit_spatial_relation_algebra_v2",
        "claim_constraint_dim": CLAIM_CONSTRAINT_DIM,
        "trace_constraint_dim": TRACE_CONSTRAINT_DIM,
        "chunk_sample_counts": chunk_counts,
        "num_chunks": len(chunk_counts),
        "total_count": counts["samples"],
        "constraint_diagnostics": {
            "claims": counts["claims"],
            "parse_rate": counts["parseable"] / max(counts["claims"], 1),
            "contradiction_rate": counts["contradicted"] / max(counts["parseable"], 1),
            "entailment_rate": counts["entailed"] / max(counts["parseable"], 1),
            "unknown_rate": counts["unknown"] / max(counts["parseable"], 1),
            "first_conflict_rate": counts["first_conflicts"] / max(counts["samples"], 1),
            "context_coverage": counts["context_available"] / max(counts["samples"], 1),
            "source_infeasible_rate": counts["source_infeasible"] / max(counts["samples"], 1),
        },
    })
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, manifest_path)
    print(json.dumps({"split": split, **manifest["constraint_diagnostics"]}, indent=2))


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    for split in [x.strip() for x in args.split.split(",") if x.strip()]:
        rebuild_split(cache_dir, split)


if __name__ == "__main__":
    main()
