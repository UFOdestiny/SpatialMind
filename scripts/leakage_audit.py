#!/usr/bin/env python3
"""Fail-closed audit of feature provenance, split overlap, and label contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_constraints import analyze_trace


SUPERVISION_FIELDS = {"label", "verified", "ground_truth", "ground_truth_dir", "answer_correct"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--output", default=None)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def _instance_hash(sample):
    payload = json.dumps(
        [str(sample.get("context", "")).strip(), str(sample.get("question", "")).strip()],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    args = parse_args()
    root = Path(args.cache_dir)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    hashes = defaultdict(set)
    report = {"cache_dir": str(root), "splits": {}, "overlap": {}, "passed": True}

    for split in splits:
        split_dir = root / split
        manifest = json.loads((split_dir / "manifest.json").read_text())
        errors = []
        n = constraint_mismatch = 0
        chunk_sample_counts = []
        sample_indices = set()
        for path in sorted(split_dir.glob("chunk_*.pt")):
            chunk = torch.load(path, map_location="cpu", weights_only=False)
            chunk_sample_counts.append(len(chunk))
            for sample in chunk:
                n += 1
                hashes[split].add(_instance_hash(sample))
                sample_index = sample.get("sample_index")
                if sample_index is not None:
                    if sample_index in sample_indices:
                        errors.append(f"sample {n}: duplicate sample_index={sample_index}")
                    sample_indices.add(sample_index)
                claims = sample.get("claims", []) or []
                verified = sample.get("verified", []) or []
                try:
                    trace_label = int(sample.get("label", -1))
                except Exception:
                    trace_label = -1
                if trace_label not in (0, 1):
                    errors.append(f"sample {n}: invalid trace label")
                claim_types = [
                    str(c.get("claim_type", "")).strip().lower()
                    for c in claims if isinstance(c, dict)
                ]
                valid_claim_contract = (
                    len(claims) >= 2
                    and len(claim_types) == len(claims)
                    and claim_types[-1] == "conclusion"
                    and claim_types.count("conclusion") == 1
                    and any(t == "reasoning" for t in claim_types[:-1])
                    and all(t in {"reasoning", "conclusion"} for t in claim_types)
                )
                try:
                    complete_labels = (
                        isinstance(verified, list)
                        and len(verified) == len(claims)
                        and all(type(v) is int and v in (0, 1) for v in verified)
                    )
                except Exception:
                    complete_labels = False
                if not valid_claim_contract:
                    errors.append(f"sample {n}: invalid claim structure")
                if not complete_labels:
                    errors.append(f"sample {n}: incomplete claim-label contract")

                # Recompute using only deployment-visible inputs. Supervision
                # fields are deliberately neither read nor passed here.
                analysis = analyze_trace(
                    str(sample.get("context", "")), str(sample.get("question", "")), claims
                )
                cf = torch.tensor([x.features for x in analysis.claims], dtype=torch.float32)
                cached_cf = torch.as_tensor(sample.get("claim_constraint_features"), dtype=torch.float32)
                tf = torch.tensor(analysis.trace_features, dtype=torch.float32)
                cached_tf = torch.as_tensor(sample.get("trace_constraint_features"), dtype=torch.float32)
                if cf.shape != cached_cf.shape or not torch.equal(cf, cached_cf):
                    constraint_mismatch += 1
                if tf.shape != cached_tf.shape or not torch.equal(tf, cached_tf):
                    constraint_mismatch += 1
                if args.max_samples and n >= args.max_samples:
                    break
            if args.max_samples and n >= args.max_samples:
                break
        if constraint_mismatch:
            errors.append(f"{constraint_mismatch} recomputed constraint tensors differ")
        if int(manifest.get("total_count", -1)) != n:
            errors.append("manifest total_count does not match physical cache")
        if manifest.get("chunk_sample_counts") != chunk_sample_counts:
            errors.append("manifest chunk_sample_counts do not match physical cache")
        if int(manifest.get("total_pending", 0)) != 0 or int(manifest.get("total_claim_pending", 0)) != 0:
            errors.append("manifest contains pending supervision")
        excluded = 0
        exclusion_summary = {}
        for prefix in ("generation", "claim_extraction", "judge"):
            count = int(manifest.get(f"{prefix}_dropped_samples", 0) or 0)
            by_label = manifest.get(f"{prefix}_dropped_by_trace_label", {}) or {}
            if by_label and sum(int(v) for v in by_label.values()) != count:
                errors.append(f"{prefix} dropped-label distribution does not sum to count")
            excluded += count
            exclusion_summary[prefix] = {
                "count": count,
                "by_trace_label": by_label,
                "reasons": manifest.get(f"{prefix}_drop_reasons", {}) or {},
            }
        judge_processed = manifest.get("judge_samples_processed")
        judge_dropped = exclusion_summary["judge"]["count"]
        if judge_processed is not None:
            judge_processed = int(judge_processed)
            if judge_processed < judge_dropped or judge_processed > n + judge_dropped:
                errors.append("judge processed count is inconsistent with physical cache")
            if manifest.get("judge_full_split_pass") and judge_processed != n + judge_dropped:
                errors.append("full-split judge count is inconsistent with one-pass deletion")
        report["splits"][split] = {
            "samples_audited": n, "unique_input_hashes": len(hashes[split]), "errors": errors,
            "constraint_inputs": ["context", "question", "claims"],
            "excluded_supervision_fields": sorted(SUPERVISION_FIELDS),
            "exclusions": exclusion_summary,
            "source_samples_reconstructed": n + excluded,
            "retention_rate": n / max(n + excluded, 1),
        }
        report["passed"] = report["passed"] and not errors

    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = hashes[a] & hashes[b]
            report["overlap"][f"{a}__{b}"] = len(overlap)
            if overlap:
                report["passed"] = False

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
