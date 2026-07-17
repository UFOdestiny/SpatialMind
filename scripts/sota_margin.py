#!/usr/bin/env python3
"""Report SpatialMind-fusion vs. best-competitor SOTA margins across the grid.

"Competitor" = every non-SpatialMind method available in a cell:
  neural probes (uhead, factoscope, and neural-seq/lookback if present),
  mlp, and the training-free baselines (constraint_rule, ccp, mcp, perplexity,
  token_entropy). This matches the competitor set in the paper's result table.

For each cell we report:
  AUROC : fusion vs. max competitor AUROC   -> relative gain %
  Brier : fusion vs. min competitor Brier   -> relative gain % (lower better)
  ECE   : fusion vs. min competitor ECE     -> relative gain %

Reads only the precomputed `overall_metrics` in each evaluation_report /
combined_evaluation json (fast; no heavy cache loads).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

DEFAULT_TAGS = {
    "20260712": "Llama-3.1-8B",
    "mistral7b": "Mistral-7B",
    "gemma2": "Gemma-2-9B",
    "phi4reason": "Phi-4-reason",
}
# (fusion_tag, display) ; SpartQA-YN intentionally excluded from headline
DEFAULT_DATASETS = [
    ("id", "StepGame"), ("spartqa", "SpaRTQA"), ("babi", "bAbI"),
    ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI"),
]
COMP_HEADS = ["uhead", "factoscope", "neural_seq", "lookback_lens", "mlp", "saplma"]
COMP_BASE = ["ccp", "mcp", "perplexity", "token_entropy"]  # constraint_rule is a SpatialMind component, not external


def head_metrics(edir, h):
    p = f"{edir}/{h}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("overall_metrics")


def base_metrics(edir, m):
    p = f"{edir}/baselines/combined_evaluation.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get(m, {}).get("overall_metrics")


def cell(results_root, fusion_tag, edir, fusion_subdir="fusion"):
    fp = f"{results_root}/{fusion_subdir}/{fusion_tag}/evaluation_report.json"
    if not os.path.exists(fp):
        return None
    fm = json.load(open(fp))["overall_metrics"]
    comps = []
    for h in COMP_HEADS:
        m = head_metrics(edir, h)
        if m:
            comps.append(m)
    for b in COMP_BASE:
        m = base_metrics(edir, b)
        if m:
            comps.append(m)
    comps = [m for m in comps if m and m.get("roc_auc") is not None]
    if not comps:
        return {"fusion": fm, "comp": None}
    bAU = max(m["roc_auc"] for m in comps)
    bBr = min(m["brier"] for m in comps)
    bECE = min(m["ece"] for m in comps)
    return {"fusion": fm, "comp": {"auroc": bAU, "brier": bBr, "ece": bECE}}


def rel(new, old, lower_better=False):
    if old is None or old == 0 or np.isnan(old):
        return float("nan")
    return (old - new) / old * 100 if lower_better else (new - old) / old * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="spatialmind/results")
    ap.add_argument("--prefix", default="constraint_guided_v10_")
    ap.add_argument("--llama_prefix_tag", default="20260712")
    ap.add_argument("--fusion_subdir", default="fusion")
    args = ap.parse_args()

    grid_fail = []
    for tag, disp in DEFAULT_TAGS.items():
        R = f"{args.root}/{args.prefix}{tag}"
        if not os.path.isdir(R):
            print(f"(missing {R})"); continue
        print(f"\n=== {disp} ({tag}) ===")
        print(f"{'ds':11s}{'AU':>7s}{'cAU':>7s}{'gAU%':>7s} | "
              f"{'Br':>7s}{'cBr':>7s}{'gBr%':>7s} | "
              f"{'ECE':>7s}{'cECE':>7s}{'gECE%':>7s}")
        for ftag, dname in DEFAULT_DATASETS:
            edir = f"{R}/eval" if ftag == "id" else f"{R}/eval_ood/{ftag}"
            c = cell(R, ftag, edir, args.fusion_subdir)
            if c is None:
                print(f"{dname:11s}  (no fusion)"); continue
            f = c["fusion"]; comp = c["comp"]
            if comp is None:
                print(f"{dname:11s}{f['roc_auc']:>7.3f}  (no competitors)"); continue
            gAU = rel(f["roc_auc"], comp["auroc"])
            gBr = rel(f["brier"], comp["brier"], lower_better=True)
            gECE = rel(f["ece"], comp["ece"], lower_better=True)
            print(f"{dname:11s}{f['roc_auc']:>7.3f}{comp['auroc']:>7.3f}{gAU:>7.1f} | "
                  f"{f['brier']:>7.3f}{comp['brier']:>7.3f}{gBr:>7.1f} | "
                  f"{f['ece']:>7.3f}{comp['ece']:>7.3f}{gECE:>7.1f}")
            # target: AUROC >=5% AND (Brier OR ECE) >=5%
            au_ok = gAU >= 5.0
            cal_ok = (gBr >= 5.0) or (gECE >= 5.0)
            if not (au_ok and cal_ok):
                grid_fail.append((disp, dname, round(gAU, 1), round(gBr, 1), round(gECE, 1)))

    print("\n" + "=" * 60)
    print(f"CELLS FAILING TARGET (AUROC>=5% AND (Brier|ECE)>=5%): {len(grid_fail)}")
    for d, ds, a, b, e in grid_fail:
        print(f"  {d:14s}{ds:11s} gAU={a:>6}  gBr={b:>6}  gECE={e:>6}")


if __name__ == "__main__":
    main()
