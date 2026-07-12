#!/usr/bin/env python3
"""Paired intervention audit for the explicit spatial-consistency module.

For every usable trusted-context relation, construct an entailed textual claim
and a minimally changed contradictory claim over the same entities.  Both go
through the public parser + solver path.  This is a label-free mechanism check:
the perturbed claim must become contradicted while the original remains entailed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_constraints import analyze_trace
from spatial_constraints.relations import relation_family


OPPOSITE = {
    "upper-right": "lower-left", "lower-left": "upper-right",
    "upper-left": "lower-right", "lower-right": "upper-left",
    "above": "below", "below": "above", "left": "right", "right": "left",
    "overlap": "above", "inside": "not-inside", "not-inside": "inside",
    "near": "far", "far": "near", "touching": "disconnected",
    "disconnected": "touching",
}


def _surface(subject: str, relation: str, obj: str) -> str:
    phrase = {
        "upper-right": "upper right of", "lower-right": "lower right of",
        "upper-left": "upper left of", "lower-left": "lower left of",
        "above": "above", "below": "below", "left": "left of", "right": "right of",
        "overlap": "overlapping", "inside": "inside", "not-inside": "outside",
        "near": "near", "far": "far from", "touching": "touching",
        "disconnected": "disconnected from",
    }[relation]
    return f"{subject} is {phrase} {obj}."


def build_counterfactual_pair(relation: dict) -> tuple[str, str, str] | None:
    """Return (entailed text, contradictory text, family), if supported."""
    s, r, o = relation["subject"], relation["relation"], relation["object"]
    # Normalise active containment into the parser's robust inside form.
    if r == "contains":
        s, r, o = o, "inside", s
    elif r == "not-contains":
        s, r, o = o, "not-inside", s
    corrupt = OPPOSITE.get(r)
    if corrupt is None:
        return None
    return _surface(s, r, o), _surface(s, corrupt, o), relation_family(r)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_pairs_per_trace", type=int, default=8)
    return p.parse_args()


def _claim(text):
    return {"text": text, "claim_type": "conclusion", "claim_type_id": 2}


def main():
    args = parse_args()
    split_dir = Path(args.cache_dir) / args.split
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if int(manifest.get("cache_schema_version", 0)) < 2:
        raise ValueError("counterfactual audit requires cache_schema_version >= 2")

    total = paired_success = original_parsed = corrupt_parsed = 0
    eligible_samples = infeasible_contexts = 0
    status_pairs = Counter()
    by_family = defaultdict(lambda: Counter(total=0, success=0))
    feature_deltas = []
    examples = []
    seen_samples = 0

    for chunk_path in sorted(split_dir.glob("chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for sample in chunk:
            context = str(sample.get("context", ""))
            context_analysis = analyze_trace(context, "", [])
            if not context_analysis.full_trace_feasible:
                infeasible_contexts += 1
                seen_samples += 1
                if args.max_samples and seen_samples >= args.max_samples:
                    break
                continue
            eligible_samples += 1
            relations = context_analysis.to_dict()["context_relations"]
            used = 0
            for relation in relations:
                pair = build_counterfactual_pair(relation)
                if pair is None:
                    continue
                original_text, corrupt_text, family = pair
                original = analyze_trace(context, "", [_claim(original_text)])
                corrupt = analyze_trace(context, "", [_claim(corrupt_text)])
                os = original.claims[0].status
                cs = corrupt.claims[0].status
                op = bool(original.claims[0].parsed)
                cp = bool(corrupt.claims[0].parsed)
                success = op and cp and os == "entailed" and cs == "contradicted"
                total += 1
                original_parsed += int(op)
                corrupt_parsed += int(cp)
                paired_success += int(success)
                status_pairs[f"{os}->{cs}"] += 1
                by_family[family]["total"] += 1
                by_family[family]["success"] += int(success)
                feature_deltas.append(
                    np.asarray(corrupt.trace_features) - np.asarray(original.trace_features)
                )
                if len(examples) < 12:
                    examples.append({
                        "original": original_text, "perturbed": corrupt_text,
                        "family": family, "status": [os, cs], "success": success,
                    })
                used += 1
                if used >= args.max_pairs_per_trace:
                    break
            seen_samples += 1
            if args.max_samples and seen_samples >= args.max_samples:
                break
        if args.max_samples and seen_samples >= args.max_samples:
            break

    deltas = np.stack(feature_deltas) if feature_deltas else np.zeros((0, 16))
    report = {
        "cache_dir": str(args.cache_dir), "split": args.split,
        "samples": seen_samples, "eligible_samples": eligible_samples,
        "infeasible_context_rate": infeasible_contexts / max(seen_samples, 1), "pairs": total,
        "original_parse_rate": original_parsed / max(total, 1),
        "perturbed_parse_rate": corrupt_parsed / max(total, 1),
        "paired_detection_rate": paired_success / max(total, 1),
        "status_transitions": dict(status_pairs),
        "by_family": {
            k: {"pairs": v["total"], "paired_detection_rate": v["success"] / max(v["total"], 1)}
            for k, v in sorted(by_family.items())
        },
        "mean_feature_delta": deltas.mean(axis=0).tolist() if len(deltas) else [],
        "examples": examples,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
