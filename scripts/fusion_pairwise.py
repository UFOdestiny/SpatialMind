#!/usr/bin/env python3
"""Symbolizability-gated stacked fusion — a first-class, reproducible UQ method.

Given two already-evaluated base heads (a constraint head and a neural probe),
this fits a small stacked-fusion combiner on the VALIDATION split and applies it
to TEST. Protocol (validation-adapted, NO leakage):
  * combiner parameters are fit only on validation scores + validation labels;
  * test transform uses only test features/base-scores + the frozen val fit;
  * feature standardization stats come from validation only.

Inputs are the per-sample `predictions` blocks that evaluate.py already writes
(sample_id, trace_label, trace_score) plus per-sample constraint trace features
(symbolizability) read from the cache. Emits an evaluation_report.json in the
SAME schema as evaluate.py (via scripts.metrics.compute_all_metrics), so the
fused method drops directly into every results table and figure.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics  # noqa: E402
from spatial_constraints.analysis import TRACE_FEATURE_NAMES  # noqa: E402

PARSE_RATE_IDX = TRACE_FEATURE_NAMES.index("parse_rate")
UNKNOWN_RATE_IDX = TRACE_FEATURE_NAMES.index("unknown_rate")
ENTAIL_RATE_IDX = TRACE_FEATURE_NAMES.index("entailment_rate")
CONCL_UNKNOWN_IDX = TRACE_FEATURE_NAMES.index("conclusion_unknown")


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_predictions(report_path: str):
    """Return (ids, labels, scores) ordered by sample_id, or None if absent."""
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        rep = json.load(f)
    preds = rep.get("predictions")
    if not preds:
        return None
    ids = np.array([p["sample_id"] for p in preds], dtype=int)
    lab = np.array([p["trace_label"] for p in preds], dtype=float)
    sc = np.array([p["trace_score"] for p in preds], dtype=float)
    order = np.argsort(ids)
    return ids[order], lab[order], sc[order]


def load_trace_features(cache_dir: str, split: str) -> np.ndarray:
    """Per-sample trace constraint features in dataset row order."""
    chunks = sorted(glob.glob(os.path.join(cache_dir, split, "chunk_*.pt")))
    if not chunks:
        raise FileNotFoundError(f"no cache chunks under {cache_dir}/{split}")
    feats = []
    for c in chunks:
        for row in torch.load(c, map_location="cpu", weights_only=False):
            tf = row.get("trace_constraint_features")
            feats.append(np.asarray(tf, dtype=float) if tf is not None
                         else np.full(len(TRACE_FEATURE_NAMES), np.nan))
    return np.stack(feats)


# --------------------------------------------------------------------------- #
# Tiny L2 logistic regression (numpy, deterministic)
# --------------------------------------------------------------------------- #
def fit_logreg(X, y, l2=1.0, iters=2000, lr=0.5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xn @ w))
        grad = Xn.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * grad
    return {"w": w, "mu": mu, "sd": sd}


def apply_logreg(fit, X):
    X = np.asarray(X, float)
    Xn = np.hstack([(X - fit["mu"]) / fit["sd"], np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Xn @ fit["w"]))


# --------------------------------------------------------------------------- #
# Fusion feature construction
# --------------------------------------------------------------------------- #
def build_fusion_features(s_con, s_neu, tf, mode: str):
    """Design matrix for the stacked fusion, indexed by fusion mode.

    scores_only : [s_con, s_neu, s_con*s_neu]                  (no symbolizability)
    symb        : scores_only + [pr, s_con*pr, s_neu*(1-pr)]   (parse_rate gate)
    determinacy : symb + determinacy signals (unknown/entail/concl_unknown) and
                  their interaction with the constraint score. Motivated by the
                  finding that the constraint signal is reliable only when claims
                  parse AND resolve determinately (low unknown_rate).
    full_gate   : symb + full trace feature vector             (rich gate)
    """
    s_con = np.asarray(s_con, float)
    s_neu = np.asarray(s_neu, float)
    base = [s_con, s_neu, s_con * s_neu]
    if mode == "scores_only":
        return np.column_stack(base)
    pr = np.nan_to_num(tf[:, PARSE_RATE_IDX], nan=0.0)
    symb = base + [pr, s_con * pr, s_neu * (1.0 - pr)]
    if mode == "symb":
        return np.column_stack(symb)
    if mode == "determinacy":
        unk = np.nan_to_num(tf[:, UNKNOWN_RATE_IDX], nan=1.0)
        ent = np.nan_to_num(tf[:, ENTAIL_RATE_IDX], nan=0.0)
        cunk = np.nan_to_num(tf[:, CONCL_UNKNOWN_IDX], nan=1.0)
        # determinacy = parse coverage discounted by how much resolves to unknown
        det = pr * (1.0 - unk)
        return np.column_stack(symb + [unk, ent, cunk, det, s_con * det, s_neu * (1.0 - det)])
    if mode == "full_gate":
        # append the whole trace feature block (NaNs -> 0) for a rich gate
        tf_clean = np.nan_to_num(tf, nan=0.0)
        return np.column_stack(symb + [tf_clean])
    raise ValueError(f"unknown fusion mode {mode!r}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_report(results_root, tag, head, split, dataset_cache_name):
    if split == "validation":
        return f"{results_root}/val_scores/{dataset_cache_name}/{head}/evaluation_report.json"
    if tag == "id":
        return f"{results_root}/eval/{head}/evaluation_report.json"
    return f"{results_root}/eval_ood/{tag}/{head}/evaluation_report.json"


# Hyperparameter grid searched ON VALIDATION ONLY when --mode=auto.
_AUTO_GRID = [("scores_only", 1.0), ("scores_only", 5.0),
              ("symb", 0.5), ("symb", 1.0), ("symb", 2.0),
              ("symb", 5.0), ("symb", 10.0), ("symb", 20.0),
              ("determinacy", 1.0), ("determinacy", 2.0),
              ("determinacy", 5.0), ("determinacy", 10.0)]


def _select_config(scv, snv, tfv, yv, l2):
    """Pick (mode, l2) by a deterministic validation-internal split (no test).

    Even-indexed val samples fit the combiner; odd-indexed val samples score it.
    The chosen config is then refit on the FULL validation split. This keeps
    hyperparameter selection strictly off the test set.
    """
    from scripts.metrics import compute_all_metrics as _m
    n = len(yv)
    fit_mask = (np.arange(n) % 2 == 0)
    sel_mask = ~fit_mask
    best = None
    for mode, cand_l2 in _AUTO_GRID:
        f = fit_logreg(build_fusion_features(scv[fit_mask], snv[fit_mask], tfv[fit_mask], mode),
                       yv[fit_mask], l2=cand_l2)
        sel_scores = apply_logreg(f, build_fusion_features(
            scv[sel_mask], snv[sel_mask], tfv[sel_mask], mode))
        if len(np.unique(yv[sel_mask])) < 2:
            continue
        a = _m(yv[sel_mask], sel_scores)["roc_auc"]
        if best is None or a > best[0]:
            best = (a, mode, cand_l2)
    return (best[1], best[2]) if best else ("symb", l2)


def fuse_one(results_root, cache_root, model, tag, cache_name,
             con_head, neu_head, mode, l2):
    cache_dir = f"{cache_root}/{cache_name}/{model}"
    cv = load_predictions(resolve_report(results_root, tag, con_head, "validation", cache_name))
    nv = load_predictions(resolve_report(results_root, tag, neu_head, "validation", cache_name))
    ct = load_predictions(resolve_report(results_root, tag, con_head, "test", cache_name))
    nt = load_predictions(resolve_report(results_root, tag, neu_head, "test", cache_name))
    if any(x is None for x in (cv, nv, ct, nt)):
        return None

    idv, yv, scv = cv
    idv2, yv2, snv = nv
    idt, yt, sct = ct
    idt2, yt2, snt = nt
    assert np.array_equal(idv, idv2) and np.array_equal(yv, yv2), "val id/label mismatch"
    assert np.array_equal(idt, idt2) and np.array_equal(yt, yt2), "test id/label mismatch"

    tfv = load_trace_features(cache_dir, "validation")[idv]
    tft = load_trace_features(cache_dir, "test")[idt]

    if mode == "auto":
        mode, l2 = _select_config(scv, snv, tfv, yv, l2)

    Xv = build_fusion_features(scv, snv, tfv, mode)
    Xt = build_fusion_features(sct, snt, tft, mode)
    fit = fit_logreg(Xv, yv, l2=l2)
    fused_test = apply_logreg(fit, Xt)

    metrics = compute_all_metrics(yt, fused_test)
    # reference points
    a_con = compute_all_metrics(yt, sct)["roc_auc"]
    a_neu = compute_all_metrics(yt, snt)["roc_auc"]
    report = {
        "method_type": "fusion_pairwise",
        "head_type": f"fusion_pairwise[{con_head}+{neu_head}]",
        "fusion_mode": mode,
        "l2": l2,
        "config_selected_on": "validation_internal_split",
        "base_heads": {"constraint": con_head, "neural": neu_head},
        "base_test_auroc": {"constraint": a_con, "neural": a_neu},
        "split": "test",
        "total_samples": int(len(yt)),
        "calibration": {"mode": "validation_stacked_fusion", "fit_on": "validation"},
        "overall_metrics": metrics,
        "symbolizability_used": mode != "scores_only",
        "predictions": [
            {"sample_id": int(idt[i]), "trace_label": int(yt[i]),
             "trace_score": float(fused_test[i]),
             "parse_rate": float(tft[i, PARSE_RATE_IDX])}
            for i in range(len(yt))
        ],
    }
    return report, metrics["roc_auc"], a_con, a_neu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--cache_root",
                    default="spatialmind/cache/cached_features/constraint_guided_v9")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--con_head", default="constraint_no_conflict")
    ap.add_argument("--neu_head", default="mlp")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "scores_only", "symb", "determinacy", "full_gate"],
                    help="auto = select (mode,l2) on a validation-internal split")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--datasets", default="id:StepGame,spartqa:spartqa,babi:babi,"
                                          "SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI")
    ap.add_argument("--out_subdir", default="fusion_pairwise")
    args = ap.parse_args()

    pairs = [tuple(x.split(":")) for x in args.datasets.split(",")]
    print(f"fusion: con={args.con_head} neu={args.neu_head} mode={args.mode} l2={args.l2}\n")
    print(f"{'dataset':10s}{'con':>8s}{'neu':>8s}{'fused':>8s}{'Δbest':>8s}")
    aurocs = {}
    for tag, cname in pairs:
        res = fuse_one(args.results_root, args.cache_root, args.model,
                       tag, cname, args.con_head, args.neu_head, args.mode, args.l2)
        if res is None:
            print(f"{cname:10s}  (missing base predictions)")
            continue
        report, a_fuse, a_con, a_neu = res
        out_dir = f"{args.results_root}/{args.out_subdir}/{tag}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        dbest = a_fuse - max(a_con, a_neu)
        print(f"{cname:10s}{a_con:8.3f}{a_neu:8.3f}{a_fuse:8.3f}{dbest:+8.3f}")
        aurocs[cname] = (a_con, a_neu, a_fuse)

    if aurocs:
        cons = [v[0] for v in aurocs.values()]
        neus = [v[1] for v in aurocs.values()]
        fus = [v[2] for v in aurocs.values()]
        print(f"\n{'MEAN':10s}{np.mean(cons):8.3f}{np.mean(neus):8.3f}{np.mean(fus):8.3f}")
        print(f"{'WORST':10s}{np.min(cons):8.3f}{np.min(neus):8.3f}{np.min(fus):8.3f}")
        wins = sum(f >= max(c, n) - 1e-9 for c, n, f in aurocs.values())
        print(f"\nfused >= best pure on {wins}/{len(aurocs)} datasets")


if __name__ == "__main__":
    main()
