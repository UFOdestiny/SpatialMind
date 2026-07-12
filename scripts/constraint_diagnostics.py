#!/usr/bin/env python3
"""Audit parser coverage and standalone structural signal in a native cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metrics import compute_all_metrics
from spatial_constraints import CLAIM_FEATURE_NAMES, TRACE_FEATURE_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def _rule_trace_score(features):
    x = dict(zip(TRACE_FEATURE_NAMES, map(float, features)))
    score = (
        0.45
        + 0.20 * x["full_trace_feasible"]
        + 0.20 * x["conclusion_entailed"]
        - 0.30 * x["conclusion_contradicted"]
        - 0.25 * x["contradiction_rate"]
        + 0.05 * x["parse_rate"]
    )
    return float(np.clip(score, 0.0, 1.0))


def main():
    args = parse_args()
    split_dir = Path(args.cache_dir) / args.split
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if int(manifest.get("cache_schema_version", 0)) < 2:
        raise ValueError("constraint diagnostics require cache_schema_version >= 2")

    n = parseable = context_available = first_conflicts = 0
    statuses = Counter()
    parser_by_label = Counter()
    labels, scores, trace_rows = [], [], []
    claim_labels, claim_rows = [], []
    processed = 0
    for chunk_path in sorted(split_dir.glob("chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for sample in chunk:
            analysis = sample.get("constraint_analysis") or {}
            context_available += int(bool(analysis.get("context_relations")))
            for ci, claim in enumerate(analysis.get("claims", [])):
                n += 1
                is_parsed = bool(claim.get("parsed"))
                parseable += int(is_parsed)
                statuses[str(claim.get("status", "unknown"))] += 1
                first_conflicts += int(bool(claim.get("first_conflict")))
                verified = sample.get("verified", [])
                if ci < len(verified) and int(verified[ci]) in (0, 1):
                    parser_by_label[f"parsed={int(is_parsed)},label={int(verified[ci])}"] += 1
                    claim_labels.append(int(verified[ci]))
                    claim_rows.append(claim.get("features", []))
            label = int(sample.get("label", -1))
            tf = sample.get("trace_constraint_features")
            if label in (0, 1) and tf is not None:
                labels.append(label)
                scores.append(_rule_trace_score(torch.as_tensor(tf).tolist()))
                trace_rows.append(torch.as_tensor(tf).tolist())
            processed += 1
            if args.max_samples and processed >= args.max_samples:
                break
        if args.max_samples and processed >= args.max_samples:
            break

    univariate_trace = {}
    if trace_rows:
        X = np.asarray(trace_rows, dtype=float)
        y = np.asarray(labels)
        for j, name in enumerate(TRACE_FEATURE_NAMES):
            if np.unique(X[:, j]).size > 1:
                univariate_trace[name] = compute_all_metrics(y, X[:, j])["roc_auc"]
    univariate_claim = {}
    if claim_rows and all(len(x) == len(CLAIM_FEATURE_NAMES) for x in claim_rows):
        X = np.asarray(claim_rows, dtype=float)
        y = np.asarray(claim_labels)
        for j, name in enumerate(CLAIM_FEATURE_NAMES):
            if np.unique(X[:, j]).size > 1:
                univariate_claim[name] = compute_all_metrics(y, X[:, j])["roc_auc"]

    report = {
        "cache_dir": str(args.cache_dir), "split": args.split, "samples": processed,
        "claim_feature_names": list(CLAIM_FEATURE_NAMES),
        "trace_feature_names": list(TRACE_FEATURE_NAMES),
        "claim_parse_rate": parseable / max(n, 1),
        "context_coverage": context_available / max(processed, 1),
        "first_conflicts_per_trace": first_conflicts / max(processed, 1),
        "status_counts": dict(statuses),
        "parser_by_claim_label": dict(parser_by_label),
        "constraint_rule_metrics": compute_all_metrics(np.asarray(labels), np.asarray(scores)),
        "univariate_trace_auroc": univariate_trace,
        "univariate_claim_auroc": univariate_claim,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
