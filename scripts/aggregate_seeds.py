#!/usr/bin/env python3
"""Aggregate repeated head runs and paired comparisons without cherry-picking."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


METRICS = ("roc_auc", "pr_auc", "aurc", "ece", "brier", "nll")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--reference", default="spatialmind")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = defaultdict(dict)
    for path in sorted(root.glob("seed_*/eval/*/evaluation_report.json")):
        seed = path.parts[-4].removeprefix("seed_")
        report = json.loads(path.read_text())
        head = str(report.get("head_type") or path.parent.name)
        rows[head][seed] = report.get("overall_metrics", {})

    summary = {}
    for head, by_seed in sorted(rows.items()):
        stats = {}
        for metric in METRICS:
            values = np.asarray([
                float(x[metric]) for x in by_seed.values() if x.get(metric) is not None
            ])
            if len(values):
                stats[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_half_width": float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
        summary[head] = {"n_seeds": len(by_seed), "seeds": sorted(by_seed), "metrics": stats}

    paired = {}
    ref = rows.get(args.reference, {})
    for head, by_seed in sorted(rows.items()):
        if head == args.reference:
            continue
        common = sorted(set(ref) & set(by_seed))
        if not common:
            continue
        paired[head] = {}
        for metric in METRICS:
            pairs = [(ref[s].get(metric), by_seed[s].get(metric)) for s in common]
            pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                continue
            a, b = map(np.asarray, zip(*pairs))
            delta = a - b
            entry = {"n": len(delta), "mean_delta_reference_minus_baseline": float(delta.mean())}
            if len(delta) >= 2 and not np.allclose(delta, delta[0]):
                entry["paired_t_p"] = float(ttest_rel(a, b).pvalue)
                try:
                    entry["wilcoxon_p"] = float(wilcoxon(delta).pvalue)
                except ValueError:
                    entry["wilcoxon_p"] = None
            paired[head][metric] = entry

    report = {"reference": args.reference, "summary": summary, "paired_tests": paired}
    text = json.dumps(report, indent=2)
    print(text)
    output = Path(args.output) if args.output else root / "multiseed_summary.json"
    output.write_text(text + "\n")


if __name__ == "__main__":
    main()
