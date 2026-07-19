#!/usr/bin/env python3
"""Ad-hoc: average % improvement of SpatialMind (fusion) over every other method,
using the STANDARD Brier score  BS = mean((s - y)^2)  instead of the reported
class-balanced macro-Brier. Pure post-processing over saved predictions.

Improvement % (lower Brier is better): (base_BS - fusion_BS) / base_BS * 100.
"""
from __future__ import annotations
import os, sys
import numpy as np
import importlib.util

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pn", os.path.join(here, "paper_numbers.py"))
pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)


def standard_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return float(np.mean((s - y) ** 2))


def metrics_std(scoremap, ref_labels):
    if scoremap is None or ref_labels is None:
        return None
    ids = sorted(set(scoremap) & set(ref_labels))
    if not ids:
        return None
    y = np.array([ref_labels[i] for i in ids], float)
    s = np.array([scoremap[i] for i in ids], float)
    return standard_brier(y, s)


FUSION = "SpatialMind"
EXCLUDE = {FUSION, "Random"}


def main():
    # cell_bs[label] = list of standard-Brier over all (backbone, dataset) cells
    per_method_cells = {}       # label -> list of BS across cells
    fusion_cells = []           # aligned fusion BS per cell key
    # per-cell improvement: for each cell, fusion vs each method
    per_cell_impr = {}          # label -> list of per-cell % improvements
    n_backbones = 0

    for tag, disp in pn.BACKBONES:
        R = f"{pn.ROOT}/{pn.PREFIX}{tag}"
        if not os.path.isdir(R):
            continue
        n_backbones += 1
        refs = {d: pn.ref_labelmap(R, ftag) for ftag, d, _ in pn.DATASETS}
        # gather BS per (label, dataset)
        cell = {}
        for grp, label, spec in pn.ROWS:
            cell[label] = {}
            for ftag, d, erel in pn.DATASETS:
                bs = metrics_std(pn.get(R, f"{R}/{erel}", spec, ftag), refs[d])
                cell[label][d] = bs
        # accumulate
        for ftag, d, erel in pn.DATASETS:
            f_bs = cell[FUSION][d]
            if f_bs is None:
                continue
            fusion_cells.append(f_bs)
            for grp, label, spec in pn.ROWS:
                if label in EXCLUDE:
                    continue
                b = cell[label][d]
                if b is None:
                    continue
                per_method_cells.setdefault(label, []).append(b)
                per_cell_impr.setdefault(label, []).append((b - f_bs) / b * 100)

    # diagnostic: base-rate per cell to spot degenerate/near-single-class pools
    print("=== per-cell base rate (pos fraction of greedy labels) ===")
    for tag, disp in pn.BACKBONES:
        R = f"{pn.ROOT}/{pn.PREFIX}{tag}"
        if not os.path.isdir(R):
            continue
        parts = []
        for ftag, d, erel in pn.DATASETS:
            rl = pn.ref_labelmap(R, ftag)
            if rl:
                y = np.array(list(rl.values()), float)
                parts.append(f"{d}={y.mean():.2f}")
            else:
                parts.append(f"{d}=--")
        print(f"  {disp:16s} " + "  ".join(parts))
    print()

    print(f"Backbones found: {n_backbones}   fusion cells: {len(fusion_cells)}")
    fusion_mean = np.mean(fusion_cells)
    print(f"\nSpatialMind (fusion) mean standard Brier over all cells: {fusion_mean:.4f}\n")

    print(f"{'method':18s}{'meanBS':>9s}{'meanImpr%':>11s}{'cellImpr%':>11s}{'nCells':>8s}")
    all_method_mean_bs = []
    all_pooled_impr = []       # every per-cell improvement across every method
    for grp, label, spec in pn.ROWS:
        if label in EXCLUDE:
            continue
        if label not in per_method_cells:
            continue
        bslist = per_method_cells[label]
        mean_bs = np.mean(bslist)
        # improvement of means: (mean_base - mean_fusion) / mean_base
        # NB use fusion mean over the SAME cells for fairness
        mean_impr = (mean_bs - fusion_mean) / mean_bs * 100
        cell_impr = np.mean(per_cell_impr[label])
        all_method_mean_bs.append(mean_bs)
        all_pooled_impr.extend(per_cell_impr[label])
        print(f"{label:18s}{mean_bs:>9.4f}{mean_impr:>11.1f}{cell_impr:>11.1f}{len(bslist):>8d}")

    print("\n=== SUMMARY (standard Brier) ===")
    grand_base = np.mean(all_method_mean_bs)
    print(f"Avg over all OTHER methods' mean-BS         : {grand_base:.4f}")
    print(f"SpatialMind fusion mean-BS                  : {fusion_mean:.4f}")
    print(f"Improvement vs avg-of-methods (mean-of-means): "
          f"{(grand_base - fusion_mean)/grand_base*100:+.1f}%")
    # average of each method's own mean-improvement
    per_method_mean_impr = [
        (np.mean(per_method_cells[l]) - fusion_mean) / np.mean(per_method_cells[l]) * 100
        for _, l, _ in pn.ROWS if l not in EXCLUDE and l in per_method_cells]
    print(f"Avg of per-method mean-improvements          : "
          f"{np.mean(per_method_mean_impr):+.1f}%")
    print(f"Pooled avg per-cell improvement (all methods): "
          f"{np.mean(all_pooled_impr):+.1f}%")


if __name__ == "__main__":
    main()
