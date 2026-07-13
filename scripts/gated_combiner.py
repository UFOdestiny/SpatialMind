#!/usr/bin/env python3
"""Symbolizability-gated selective predictor.

Fits a tiny gate on the VALIDATION split of each dataset (validation-adapted
protocol: no test labels, no test statistics), then applies to test.

gate input  : per-sample constraint trace features (symbolizability signals)
gate output : a mixing weight w in [0,1]
combined     : w * s_constraint + (1-w) * s_neural

We compare three combiners of increasing capacity, all fit on validation only:
  A. global-w      : single scalar w (logistic-regressed on val to predict label)
  B. symb-linear   : w = sigmoid(a * parse_rate + b)   (1 feature)
  C. gate-mlp      : w = sigmoid(MLP(trace_features))   (16 features)
Baselines: pure constraint, pure neural, mean blend, best-of-val (oracle model pick).
"""
from __future__ import annotations
import json, os, glob, argparse
import numpy as np
import torch

RES = "spatialmind/results/constraint_guided_v9_20260712"
CACHE_ROOT = "spatialmind/cache/cached_features/constraint_guided_v9"
MODEL = "Llama-3.1-8B-Instruct"
PARSE_RATE_IDX = 3
DATASETS = [("id", "StepGame"), ("spartqa", "spartqa"), ("babi", "babi"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return (r_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def load_preds(rp):
    if not os.path.exists(rp):
        return None
    d = json.load(open(rp))
    preds = d.get("predictions")
    if not preds:
        return None
    ids = np.array([p["sample_id"] for p in preds])
    lab = np.array([p["trace_label"] for p in preds], float)
    sc = np.array([p["trace_score"] for p in preds], float)
    order = np.argsort(ids)
    return ids[order].astype(int), lab[order], sc[order]


def load_tf(cache_dir, split):
    chunks = sorted(glob.glob(os.path.join(cache_dir, split, "chunk_*.pt")))
    feats = []
    for c in chunks:
        for row in torch.load(c, map_location="cpu", weights_only=False):
            tf = row.get("trace_constraint_features")
            feats.append(np.asarray(tf, float) if tf is not None else np.full(16, np.nan))
    return np.stack(feats)


def scores_for(tag, cname, head, split):
    """(ids, y, score, trace_feats) for a head/dataset/split, aligned."""
    sub = "eval" if (tag == "id" and split == "test") else None
    if split == "validation":
        rp = f"{RES}/val_scores/{cname}/{head}/evaluation_report.json"
    elif tag == "id":
        rp = f"{RES}/eval/{head}/evaluation_report.json"
    else:
        rp = f"{RES}/eval_ood/{tag}/{head}/evaluation_report.json"
    p = load_preds(rp)
    if p is None:
        return None
    ids, y, sc = p
    tf = load_tf(f"{CACHE_ROOT}/{cname}/{MODEL}", split)[ids]
    return ids, y, sc, tf


def fit_logreg(X, y, l2=1.0, iters=500, lr=0.1):
    """Tiny logistic regression via full-batch gradient descent (numpy)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xn @ w))
        g = Xn.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0] / len(y)
        w -= lr * g
    return w, mu, sd


def apply_logreg(w, mu, sd, X):
    Xn = (np.asarray(X, float) - mu) / sd
    Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
    return 1 / (1 + np.exp(-Xn @ w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con", default="constraint_no_conflict")
    ap.add_argument("--neu", default="mlp")
    args = ap.parse_args()
    print(f"con={args.con} neu={args.neu}\n")
    hdr = ("dataset", "con", "neu", "meanblend", "gate", "stack", "oracle")
    print(f"{hdr[0]:10s}" + "".join(f"{h:>10s}" for h in hdr[1:]))

    summ = {k: [] for k in ["con", "neu", "meanblend", "combined"]}
    for tag, cname in DATASETS:
        cv = scores_for(tag, cname, args.con, "validation")
        nv = scores_for(tag, cname, args.neu, "validation")
        ct = scores_for(tag, cname, args.con, "test")
        nt = scores_for(tag, cname, args.neu, "test")
        if any(x is None for x in (cv, nv, ct, nt)):
            print(f"{cname:10s}  (missing)"); continue
        idv, yv, scv, tfv = cv
        _, yv2, snv, _ = nv
        idt, yt, sct, tft = ct
        _, yt2, snt, _ = nt
        assert np.array_equal(idv, nv[0]) and np.array_equal(idt, nt[0])

        # --- Combiner C (gate): logreg on trace feats predicting which score is
        #     more reliable; soft-mix by P(constraint better). ---
        gate_target = (np.abs(scv - yv) <= np.abs(snv - yv)).astype(float)
        wgt, mu, sd = fit_logreg(tfv, gate_target, l2=2.0)
        wtest = apply_logreg(wgt, mu, sd, tft)
        combined = wtest * sct + (1 - wtest) * snt

        # --- Combiner D (stack): logreg predicting the LABEL directly from both
        #     scores + symbolizability + their interactions (learned fusion). ---
        pr_v = tfv[:, PARSE_RATE_IDX]; pr_t = tft[:, PARSE_RATE_IDX]
        Xv = np.column_stack([scv, snv, pr_v, scv * pr_v, snv * (1 - pr_v), scv * snv])
        Xt = np.column_stack([sct, snt, pr_t, sct * pr_t, snt * (1 - pr_t), sct * snt])
        ws, mus, sds = fit_logreg(Xv, yv, l2=1.0)
        stack = apply_logreg(ws, mus, sds, Xt)

        a_con = auroc(yt, sct); a_neu = auroc(yt, snt)
        a_blend = auroc(yt, 0.5 * sct + 0.5 * snt)
        a_gateC = auroc(yt, combined)
        a_comb = auroc(yt, stack)
        s_or = np.where(np.abs(sct - yt) <= np.abs(snt - yt), sct, snt)
        a_or = auroc(yt, s_or)
        print(f"{cname:10s}{a_con:10.3f}{a_neu:10.3f}{a_blend:10.3f}"
              f"{a_gateC:10.3f}{a_comb:10.3f}{a_or:10.3f}")
        summ["con"].append(a_con); summ["neu"].append(a_neu)
        summ["meanblend"].append(a_blend); summ["combined"].append(a_comb)

    print("\n--- mean across 5 datasets ---")
    for k in ["con", "neu", "meanblend", "combined"]:
        print(f"{k:12s} {np.mean(summ[k]):.3f}")
    print("\n--- worst-case (min) across 5 datasets ---")
    for k in ["con", "neu", "meanblend", "combined"]:
        print(f"{k:12s} {np.min(summ[k]):.3f}")


if __name__ == "__main__":
    main()
