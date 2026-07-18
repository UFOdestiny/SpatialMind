#!/usr/bin/env python3
"""Generate paper table numbers (AUROC / class-balanced macro-Brier) for every
method, all backbones, 5 headline datasets (StepGame, SpaRTQA, SpaRTUN,
SpaceNLI, SpaRP=SpaRP_PS3). Reads v11 namespaces. Pure post-processing.

macro-Brier = 0.5*mean((s-1)^2 | y=1) + 0.5*mean(s^2 | y=0)  (class-balanced;
plain Brier is base-rate-coupled on imbalanced OOD pools and is not reported).

Prints, per backbone: a raw table, a MEAN column, and marks best (**) and
second-best (_) per (dataset, metric) column across comparable rows.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = "spatialmind/results"
PREFIX = "constraint_guided_v11_"
BACKBONES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B"),
             ("gemma", "Gemma-2-9B"), ("phi", "Phi-4-reasoning"),
             ("qwen", "Qwen3-8B")]
DATASETS = [("id", "StepGame", "eval"),
            ("spartqa", "SpaRTQA", "eval_ood/spartqa"),
            ("SpaRTUN", "SpaRTUN", "eval_ood/SpaRTUN"),
            ("SpaceNLI", "SpaceNLI", "eval_ood/SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "eval_ood/SpaRP_PS3")]

# group, label -> spec
ROWS = [
    ("Unsup.", "Random",           ("base", "random")),
    ("Unsup.", "Perplexity",       ("base", "perplexity")),
    ("Unsup.", "Token Entropy",    ("base", "token_entropy")),
    ("Unsup.", "MCP",              ("base", "mcp")),
    ("Unsup.", "CCP",              ("base", "ccp")),
    ("Sampl.", "Semantic Entropy", ("samp", "semantic_entropy")),
    ("Sampl.", "SelfCheckGPT",     ("samp", "selfcheckgpt")),
    ("Sampl.", "P(True)",          ("samp", "p_true")),
    ("Neural", "Factoscope",       ("head", "factoscope")),
    ("Neural", "UHead",            ("head", "uhead")),
    ("Neural", "Neural-Seq",       ("head", "spatialmind_neural")),
    ("Neural", "MLP",              ("head", "mlp")),
    ("Symb.",  "Constraint-Rule",  ("base", "constraint_rule")),
    ("Symb.",  "Constraint-Only",  ("head", "constraint_only")),
    ("Symb.",  "Constraint",       ("head", "spatialmind")),
    ("Fusion", "SpatialMind",      ("fusion", None)),
]


def _scoremap(pr):
    return {x["sample_id"]: x["trace_score"] for x in pr}


def _labelmap(pr):
    return {x["sample_id"]: x["trace_label"] for x in pr}


def load_head(edir, name):
    p = f"{edir}/{name}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    pr = json.load(open(p)).get("predictions")
    return _scoremap(pr) if pr else None


def load_combined(edir, sub, key):
    p = f"{edir}/{sub}/combined_evaluation.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if key not in d or "predictions" not in d[key]:
        return None
    return _scoremap(d[key]["predictions"])


def load_fusion(R, tag):
    p = f"{R}/fusion/{tag}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    pr = json.load(open(p)).get("predictions")
    return _scoremap(pr) if pr else None


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def metrics(scoremap, ref_labels):
    """AUROC / macro-Brier of a method's scores against the shared reference
    (greedy) labels, on the intersection of sample_ids."""
    if scoremap is None or ref_labels is None:
        return None
    ids = sorted(set(scoremap) & set(ref_labels))
    if not ids:
        return None
    y = np.array([ref_labels[i] for i in ids], float)
    s = np.array([scoremap[i] for i in ids], float)
    au = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    return au, macro_brier(y, s)


def get(R, edir, spec, ftag):
    kind, key = spec
    if kind == "head":
        return load_head(edir, key)
    if kind == "base":
        return load_combined(edir, "baselines", key)
    if kind == "samp":
        return load_combined(edir, "baselines_sampling", key)
    if kind == "fusion":
        return load_fusion(R, ftag)
    return None


def ref_labelmap(R, tag):
    p = f"{R}/fusion/{tag}/evaluation_report.json"
    if not os.path.exists(p):
        return None
    pr = json.load(open(p)).get("predictions")
    return _labelmap(pr) if pr else None


def main():
    for tag, disp in BACKBONES:
        R = f"{ROOT}/{PREFIX}{tag}"
        print(f"\n########## {disp} ({tag})")
        if not os.path.isdir(R):
            print("  MISSING namespace"); continue
        # shared reference labels per dataset (greedy trace, from fusion report)
        refs = {d: ref_labelmap(R, ftag) for ftag, d, _ in DATASETS}
        # collect metrics: data[label][dcol] = (au, br)
        data = {}
        for grp, label, spec in ROWS:
            row = {}
            for ftag, d, erel in DATASETS:
                m = metrics(get(R, f"{R}/{erel}", spec, ftag), refs[d])
                row[d] = m
            data[label] = (grp, row)
        # per-dataset best/2nd for AU (max) and BR (min), excluding Random
        dcols = [d for _, d, _ in DATASETS]
        rank = {}
        for d in dcols:
            aus = [(data[l][1][d][0], l) for _, l, _ in ROWS
                   if l != "Random" and data[l][1][d] and not np.isnan(data[l][1][d][0])]
            brs = [(data[l][1][d][1], l) for _, l, _ in ROWS
                   if l != "Random" and data[l][1][d]]
            aus_sorted = sorted(aus, reverse=True)
            brs_sorted = sorted(brs)
            rank[d] = {
                "au1": aus_sorted[0][1] if aus_sorted else None,
                "au2": aus_sorted[1][1] if len(aus_sorted) > 1 else None,
                "br1": brs_sorted[0][1] if brs_sorted else None,
                "br2": brs_sorted[1][1] if len(brs_sorted) > 1 else None,
            }
        # print
        hdr = f"{'method':17s}" + "".join(f"{d:>16s}" for d in dcols) + f"{'MEAN-AU':>9s}"
        print(hdr)
        for grp, label, spec in ROWS:
            _, row = data[label]
            cells = []
            aus = []
            for d in dcols:
                m = row[d]
                if m is None:
                    cells.append(f"{'--':>16s}"); continue
                au, br = m
                aus.append(au)
                amark = "**" if rank[d]["au1"] == label else ("_" if rank[d]["au2"] == label else "")
                bmark = "**" if rank[d]["br1"] == label else ("_" if rank[d]["br2"] == label else "")
                cells.append(f"{amark}{au:.3f}{amark}/{bmark}{br:.3f}{bmark}".rjust(16))
            mean = f"{np.mean(aus):.3f}" if aus else "--"
            print(f"{label:17s}" + "".join(cells) + f"{mean:>9s}")


if __name__ == "__main__":
    main()
