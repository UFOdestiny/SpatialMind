#!/usr/bin/env python3
"""Fair SpatialMind-vs-SOTA benchmark with a UNIFIED correctness label.

The sampling baselines (scripts/sampling_baselines.py) re-decode at temperature
0.7, so their per-sample correctness labels differ from the main pipeline's
greedy trace (~19% flip). Comparing AUROC across two different label sets is
apples-to-oranges. Here we fix that:

  * the correctness LABEL for every method = the main pipeline's greedy trace
    label (what SpatialMind/heads/decoding baselines are all evaluated on);
  * each method's SCORE is its own reliability estimate, joined by sample_id.

For sampling methods this means: their uncertainty score (computed from K
temperature samples) is used to predict whether the GREEDY trace is correct.
That is the standard, fair use of a sampling-based UQ score as a predictor.

AUROC + class-balanced (macro) Brier, reported per (dataset) for the Llama
namespace. Datasets included are auto-detected from what exists on disk.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.metrics import roc_auc_score

R_DEFAULT = "spatialmind/results/constraint_guided_llama"
# fusion tag -> (eval dir)
DATASETS = [("id", "StepGame", "eval"),
            ("spartqa", "SpaRTQA", "eval_ood/spartqa"),
            ("babi", "bAbI", "eval_ood/babi"),
            ("SpaRTUN", "SpaRTUN", "eval_ood/SpaRTUN"),
            ("SpaceNLI", "SpaceNLI", "eval_ood/SpaceNLI"),
            ("SpaRP_PS1", "SpaRP-PS1", "eval_ood/SpaRP_PS1"),
            ("SpaRP_PS3", "SpaRP-PS3", "eval_ood/SpaRP_PS3")]
SAMPLING = ["semantic_entropy", "selfcheckgpt", "p_true"]


def au(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def score_map(path, key=None):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    pr = d[key]["predictions"] if key else d.get("predictions")
    if not pr:
        return None
    return {x["sample_id"]: x["trace_score"] for x in pr}


def label_map(path, key=None):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    pr = d[key]["predictions"] if key else d.get("predictions")
    if not pr:
        return None
    return {x["sample_id"]: x["trace_label"] for x in pr}


def eval_with_ref_labels(ref_labels, scores):
    """AUROC/Brier of `scores` against the reference (greedy) labels, on the
    intersection of sample_ids."""
    ids = sorted(set(ref_labels) & set(scores))
    if not ids:
        return None
    y = np.array([ref_labels[i] for i in ids], float)
    s = np.array([scores[i] for i in ids], float)
    return au(y, s), macro_brier(y, s), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=R_DEFAULT)
    ap.add_argument("--fusion_subdir", default="fusion")
    args = ap.parse_args()
    R = args.root
    print(f"{'dataset':11s}{'ourAU':>7s}{'ourBr':>7s} | "
          f"{'SE':>6s}{'SC':>6s}{'PT':>6s} | {'bestAU':>7s}{'bestBr':>7s} | "
          f"{'gAU%':>6s}{'gBr%':>6s}  verdict")
    wins = 0; total = 0
    for ftag, disp, edir_rel in DATASETS:
        edir = f"{R}/{edir_rel}"
        fpath = f"{R}/{args.fusion_subdir}/{ftag}/evaluation_report.json"
        if not os.path.exists(fpath):
            continue
        # reference greedy labels: take them from the fusion report itself
        ref = label_map(fpath)
        four = score_map(fpath)
        our = eval_with_ref_labels(ref, four)
        if our is None:
            continue
        ourAU, ourBr, _ = our
        # sampling baselines, re-scored against the SAME greedy labels
        samp = {}
        spath = f"{edir}/baselines_sampling/combined_evaluation.json"
        for m in SAMPLING:
            sm = score_map(spath, m)
            if sm:
                r = eval_with_ref_labels(ref, sm)
                if r:
                    samp[m] = r
        if not samp:
            print(f"{disp:11s}{ourAU:>7.3f}{ourBr:>7.3f} | (no sampling baselines yet)")
            continue
        bestAU = max(v[0] for v in samp.values())
        bestBr = min(v[1] for v in samp.values())
        gAU = (ourAU - bestAU) / bestAU * 100
        gBr = (bestBr - ourBr) / bestBr * 100
        verdict = "WIN" if (gAU >= 5 and gBr >= 5) else ("AU+" if gAU >= 5 else "lose")
        total += 1; wins += int(gAU >= 5 and gBr >= 5)
        se = samp.get("semantic_entropy", (float('nan'),))[0]
        sc = samp.get("selfcheckgpt", (float('nan'),))[0]
        pt = samp.get("p_true", (float('nan'),))[0]
        print(f"{disp:11s}{ourAU:>7.3f}{ourBr:>7.3f} | "
              f"{se:>6.3f}{sc:>6.3f}{pt:>6.3f} | {bestAU:>7.3f}{bestBr:>7.3f} | "
              f"{gAU:>+6.0f}{gBr:>+6.0f}  {verdict}")
    print(f"\nWIN (AUROC>=5% AND macroBrier>=5% vs best sampling SOTA): {wins}/{total}")


if __name__ == "__main__":
    main()
