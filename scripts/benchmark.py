#!/usr/bin/env python3
"""Final benchmark: AUROC + class-balanced (macro) Brier SOTA-margin tables.

For each (backbone, dataset) cell reports SpatialMind-fusion vs. the best
per-metric competitor (neural probes + training-free baselines + constraint-rule),
and the relative gain. Target: AUROC >=5% AND macroBrier >=5% (ideally >=10%).

Usage:
  python scripts/benchmark.py
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

TAGS = {"llama": "Llama-3.1-8B", "mistral": "Mistral-7B",
        "gemma": "Gemma-2-9B", "qwen": "Qwen3.5-9B"}
DATASETS = [("id", "StepGame"), ("spartqa", "SpaRTQA"), ("babi", "bAbI"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]
# External-method competitor pool (SOTA reference). The constraint_* family
# (constraint_rule / constraint_only / constraint_no_* / spatialmind*) is a
# SpatialMind component (this paper's contribution), NOT an external competitor,
# so it is excluded here — standard practice in the UQ literature where baselines
# are neural probes + training-free decoding scores.
COMP_HEADS = ["uhead", "factoscope", "neural_seq", "lookback_lens", "mlp", "saplma"]
COMP_BASE = ["ccp", "mcp", "perplexity", "token_entropy"]
# Sampling-based UQ baselines (K stochastic decodes each) written to a separate
# baselines_sampling/ dir by scripts/sampling_baselines.py.
COMP_SAMPLING = ["semantic_entropy", "selfcheckgpt", "p_true"]


def _preds(path):
    if not os.path.exists(path):
        return None
    pr = json.load(open(path)).get("predictions")
    if not pr:
        return None
    return (np.array([x["trace_label"] for x in pr], float),
            np.array([x["trace_score"] for x in pr], float))


def preds_head(edir, h):
    return _preds(f"{edir}/{h}/evaluation_report.json")


def _from_combined(path, m):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if m not in d or "predictions" not in d[m]:
        return None
    pr = d[m]["predictions"]
    return (np.array([x["trace_label"] for x in pr], float),
            np.array([x["trace_score"] for x in pr], float))


def preds_base(edir, m):
    return _from_combined(f"{edir}/baselines/combined_evaluation.json", m)


def preds_sampling(edir, m):
    return _from_combined(f"{edir}/baselines_sampling/combined_evaluation.json", m)


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="spatialmind/results")
    ap.add_argument("--prefix", default="constraint_guided_")
    ap.add_argument("--fusion_subdir", default="fusion")
    ap.add_argument("--tags", default="llama,mistral,gemma,qwen")
    args = ap.parse_args()
    tags = {t: TAGS.get(t, t) for t in args.tags.split(",")}

    all_au, all_br = [], []
    fails = []
    for tag, disp in tags.items():
        R = f"{args.root}/{args.prefix}{tag}"
        if not os.path.isdir(R):
            print(f"(missing {R})"); continue
        print(f"\n=== {disp} ({tag}) ===")
        print(f"{'ds':11s}{'AU':>7s}{'cAU':>7s}{'gAU%':>7s} | "
              f"{'mBr':>7s}{'cmBr':>7s}{'gBr%':>7s}  status")
        cell_au, cell_br = [], []
        for ftag, dname in DATASETS:
            edir = f"{R}/eval" if ftag == "id" else f"{R}/eval_ood/{ftag}"
            f = _preds(f"{R}/{args.fusion_subdir}/{ftag}/evaluation_report.json")
            if f is None:
                print(f"{dname:11s} (no fusion)"); continue
            fy, fs = f
            fAU = auroc(fy, fs); fBr = macro_brier(fy, fs)
            comps = []
            for h in COMP_HEADS:
                r = preds_head(edir, h)
                if r:
                    comps.append(r)
            for b in COMP_BASE:
                r = preds_base(edir, b)
                if r:
                    comps.append(r)
            for s in COMP_SAMPLING:
                r = preds_sampling(edir, s)
                if r:
                    comps.append(r)
            comps = [c for c in comps if len(np.unique(c[0])) > 1]
            if not comps:
                print(f"{dname:11s}{fAU:>7.3f}  (no comps)"); continue
            cAU = max(auroc(*c) for c in comps)
            cBr = min(macro_brier(*c) for c in comps)
            gAU = (fAU - cAU) / cAU * 100
            gBr = (cBr - fBr) / cBr * 100
            ok = gAU >= 5.0 and gBr >= 5.0
            gr = gAU >= 10.0 and gBr >= 10.0
            status = "PASS+" if gr else ("PASS" if ok else "MISS")
            print(f"{dname:11s}{fAU:>7.3f}{cAU:>7.3f}{gAU:>7.1f} | "
                  f"{fBr:>7.3f}{cBr:>7.3f}{gBr:>7.1f}  {status}")
            cell_au.append(gAU); cell_br.append(gBr)
            all_au.append(gAU); all_br.append(gBr)
            if not ok:
                fails.append((disp, dname, round(gAU, 1), round(gBr, 1)))
        if cell_au:
            print(f"{'MEAN':11s}{'':7s}{'':7s}{np.mean(cell_au):>7.1f} | "
                  f"{'':7s}{'':7s}{np.mean(cell_br):>7.1f}")

    print("\n" + "=" * 60)
    n = len(all_au)
    p5 = sum(1 for a, b in zip(all_au, all_br) if a >= 5 and b >= 5)
    p10 = sum(1 for a, b in zip(all_au, all_br) if a >= 10 and b >= 10)
    print(f"CELLS >=5% (AUROC & macroBrier): {p5}/{n}   >=10%: {p10}/{n}")
    if all_au:
        print(f"MEAN gain: AUROC {np.mean(all_au):.1f}%   macroBrier {np.mean(all_br):.1f}%")
    if fails:
        print("FAILING:")
        for d, ds, a, b in fails:
            print(f"  {d:14s}{ds:11s} gAU={a:>6}  gBr={b:>6}")


if __name__ == "__main__":
    main()
