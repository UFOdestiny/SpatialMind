#!/usr/bin/env python3
"""Compute fusion improvement over the strongest external baseline, per backbone.
Reverse-engineers the definition used in the paper (Mistral 21.9%/26.9%,
Gemma 20.3%/31.7%). External baseline = every ROW except SpatialMind (fusion)
and Random. We try a few aggregation definitions and print all so we can match.
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score
import importlib.util

spec = importlib.util.spec_from_file_location(
    "pn", os.path.join(os.path.dirname(__file__), "paper_numbers.py"))
pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)

# rows that count as "external baselines" (exclude our fusion + Random + the
# three constraint scorers which are arguably "ours")
FUSION = "SpatialMind"


def collect(tag):
    R = f"{pn.ROOT}/{pn.PREFIX}{tag}"
    if not os.path.isdir(R):
        return None
    refs = {d: pn.ref_labelmap(R, ftag) for ftag, d, _ in pn.DATASETS}
    data = {}
    for grp, label, spc in pn.ROWS:
        row = {}
        for ftag, d, erel in pn.DATASETS:
            row[d] = pn.metrics(pn.get(R, f"{R}/{erel}", spc, ftag), refs[d])
        data[label] = (grp, row)
    return data


def report(tag, disp, exclude_constraints):
    data = collect(tag)
    if data is None:
        print(f"{disp}: MISSING"); return
    dcols = [d for _, d, _ in pn.DATASETS]
    excl = {FUSION, "Random"}
    if exclude_constraints:
        excl |= {"Constraint", "Constraint-Only", "Constraint-Rule"}
    # fusion means
    fau = [data[FUSION][1][d][0] for d in dcols if data[FUSION][1][d] and not np.isnan(data[FUSION][1][d][0])]
    fbr = [data[FUSION][1][d][1] for d in dcols if data[FUSION][1][d]]
    fusion_au, fusion_br = np.mean(fau), np.mean(fbr)

    # DEF A: pick single best-mean-AUC baseline overall, improvement of means
    base_means_au = {}
    base_means_br = {}
    for _, l, _ in pn.ROWS:
        if l in excl:
            continue
        aus = [data[l][1][d][0] for d in dcols if data[l][1][d] and not np.isnan(data[l][1][d][0])]
        brs = [data[l][1][d][1] for d in dcols if data[l][1][d]]
        if aus:
            base_means_au[l] = np.mean(aus)
        if brs:
            base_means_br[l] = np.mean(brs)
    best_au_method = max(base_means_au, key=base_means_au.get)
    best_br_method = min(base_means_br, key=base_means_br.get)
    a_au = (fusion_au - base_means_au[best_au_method]) / base_means_au[best_au_method] * 100
    a_br = (base_means_br[best_br_method] - fusion_br) / base_means_br[best_br_method] * 100

    # DEF B: per-dataset best baseline, then average the per-dataset improvements
    per_au, per_br = [], []
    for d in dcols:
        if not data[FUSION][1][d]:
            continue
        fau_d, fbr_d = data[FUSION][1][d]
        cand_au = [data[l][1][d][0] for _, l, _ in pn.ROWS if l not in excl and data[l][1][d] and not np.isnan(data[l][1][d][0])]
        cand_br = [data[l][1][d][1] for _, l, _ in pn.ROWS if l not in excl and data[l][1][d]]
        if cand_au and not np.isnan(fau_d):
            b = max(cand_au); per_au.append((fau_d - b) / b * 100)
        if cand_br:
            b = min(cand_br); per_br.append((b - fbr_d) / b * 100)
    b_au = np.mean(per_au); b_br = np.mean(per_br)

    print(f"\n### {disp} (exclude_constraints={exclude_constraints})")
    print(f"  fusion mean AUC={fusion_au:.3f} BS={fusion_br:.3f}")
    print(f"  best-AUC baseline: {best_au_method} ({base_means_au[best_au_method]:.3f}); best-BS baseline: {best_br_method} ({base_means_br[best_br_method]:.3f})")
    print(f"  DEF-A (mean-of-means): AUC +{a_au:.1f}%  BS +{a_br:.1f}%")
    print(f"  DEF-B (per-dataset-best, avg): AUC +{b_au:.1f}%  BS +{b_br:.1f}%")


if __name__ == "__main__":
    for tag, disp in pn.BACKBONES:
        for exc in (False, True):
            report(tag, disp, exc)
