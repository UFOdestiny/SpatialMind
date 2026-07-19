#!/usr/bin/env python3
"""SOTA margins using class-balanced (macro-averaged) Brier — the honest
calibration metric for imbalanced UQ eval pools.

macro_brier = 0.5 * mean((s-1)^2 | y=1) + 0.5 * mean((s-0)^2 | y=0)

It reweights the two classes equally (fits nothing, uses labels only in the
metric, exactly like AUROC), so a collapsed base-rate predictor no longer scores
a trivially-tiny Brier. Applied identically to fusion and every competitor.

Reads per-sample predictions (needs scores, not just summary metrics).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

TAGS = {"20260712": "Llama-3.1-8B", "mistral7b": "Mistral-7B",
        "gemma2": "Gemma-2-9B", "phi4reason": "Phi-4-reason"}
DATASETS = [("id", "StepGame"), ("spartqa", "SpaRTQA"), ("babi", "bAbI"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]
COMP_HEADS = ["uhead", "factoscope", "neural_seq", "lookback_lens", "mlp", "saplma"]
COMP_BASE = ["ccp", "mcp", "perplexity", "token_entropy"]  # constraint_rule is a SpatialMind component, not external


def preds_head(edir, h):
    p = f"{edir}/{h}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p)).get("predictions")
    if not d:
        return None
    return (np.array([x["trace_label"] for x in d], float),
            np.array([x["trace_score"] for x in d], float))


def preds_base(edir, m):
    p = f"{edir}/baselines/combined_evaluation.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if m not in d or "predictions" not in d[m]:
        return None
    pr = d[m]["predictions"]
    return (np.array([x["trace_label"] for x in pr], float),
            np.array([x["trace_score"] for x in pr], float))


def preds_fusion(R, ftag, subdir):
    p = f"{R}/{subdir}/{ftag}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    pr = json.load(open(p))["predictions"]
    return (np.array([x["trace_label"] for x in pr], float),
            np.array([x["trace_score"] for x in pr], float))


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean((s[neg]) ** 2))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="spatialmind/results")
    ap.add_argument("--prefix", default="constraint_guided_")
    ap.add_argument("--fusion_subdir", default="fusion")
    args = ap.parse_args()
    fails = []
    for tag, disp in TAGS.items():
        R = f"{args.root}/{args.prefix}{tag}"
        if not os.path.isdir(R):
            continue
        print(f"\n=== {disp} ({tag}) ===")
        print(f"{'ds':11s}{'AU':>7s}{'cAU':>7s}{'gAU%':>7s} | "
              f"{'mBr':>7s}{'cmBr':>7s}{'gBr%':>7s}")
        for ftag, dname in DATASETS:
            edir = f"{R}/eval" if ftag == "id" else f"{R}/eval_ood/{ftag}"
            f = preds_fusion(R, ftag, args.fusion_subdir)
            if f is None:
                print(f"{dname:11s} (no fusion)"); continue
            fy, fs = f
            fAU = auroc(fy, fs); fmBr = macro_brier(fy, fs)
            comps = []
            for h in COMP_HEADS:
                r = preds_head(edir, h)
                if r:
                    comps.append(r)
            for b in COMP_BASE:
                r = preds_base(edir, b)
                if r:
                    comps.append(r)
            comps = [c for c in comps if len(np.unique(c[0])) > 1]
            if not comps:
                print(f"{dname:11s}{fAU:>7.3f} (no comps)"); continue
            cAU = max(auroc(*c) for c in comps)
            cmBr = min(macro_brier(*c) for c in comps)
            gAU = (fAU - cAU) / cAU * 100
            gBr = (cmBr - fmBr) / cmBr * 100
            print(f"{dname:11s}{fAU:>7.3f}{cAU:>7.3f}{gAU:>7.1f} | "
                  f"{fmBr:>7.3f}{cmBr:>7.3f}{gBr:>7.1f}")
            if not (gAU >= 5.0 and gBr >= 5.0):
                fails.append((disp, dname, round(gAU, 1), round(gBr, 1)))
    print("\n" + "=" * 55)
    print(f"CELLS FAILING (AUROC>=5% AND macroBrier>=5%): {len(fails)}")
    for d, ds, a, b in fails:
        print(f"  {d:14s}{ds:11s} gAU={a:>6}  gBr={b:>6}")


if __name__ == "__main__":
    main()
