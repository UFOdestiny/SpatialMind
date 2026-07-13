#!/usr/bin/env python3
"""Offline exploration: symbolizability-gated selective prediction.

Uses ONLY already-saved artifacts (no GPU):
  - per-sample trace_score/trace_label from evaluation_report.json (each head)
  - per-sample trace_constraint_features from the cache (parse_rate = symbolizability)

Goal: test whether routing/blending a constraint head with a neural head by
per-sample symbolizability beats either pure method across the ID+OOD spectrum.
"""
from __future__ import annotations
import json, os, glob, argparse
import numpy as np
import torch

RES = "spatialmind/results/constraint_guided_v9_20260712"
CACHE_ROOT = "spatialmind/cache/cached_features/constraint_guided_v9"
MODEL = "Llama-3.1-8B-Instruct"

# trace feature index 3 == parse_rate (see spatial_constraints/analysis.py TRACE_FEATURE_NAMES)
PARSE_RATE_IDX = 3
FULL_FEASIBLE_IDX = 6
UNKNOWN_RATE_IDX = 9
CONCL_ENTAILED_IDX = 13


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def load_preds(report_path):
    d = json.load(open(report_path))
    preds = d.get("predictions")
    if not preds:
        return None
    n = len(preds)
    ids = np.array([p["sample_id"] for p in preds])
    lab = np.array([p["trace_label"] for p in preds], float)
    sc = np.array([p["trace_score"] for p in preds], float)
    kh = np.array([p.get("k_hop", 0) for p in preds])
    order = np.argsort(ids)
    return ids[order], lab[order], sc[order], kh[order]


def load_symbolizability(cache_dir, split):
    """Per-sample trace features from cache, kept in dataset ROW order.

    Predictions carry sample_id = row position (0..n-1), which matches the
    cache file/row order — so we align by position, not by sample_index.
    """
    chunks = sorted(glob.glob(os.path.join(cache_dir, split, "chunk_*.pt")))
    feats = []
    for c in chunks:
        data = torch.load(c, map_location="cpu", weights_only=False)
        for row in data:
            tf = row.get("trace_constraint_features")
            feats.append(np.asarray(tf, float) if tf is not None else np.full(16, np.nan))
    return np.stack(feats)


def eval_head(name, tag, cache_dir):
    """Return aligned (label, score, khop, tracefeats) for a head on a split dir."""
    if tag == "id":
        rp = f"{RES}/eval/{name}/evaluation_report.json"
    else:
        rp = f"{RES}/eval_ood/{tag}/{name}/evaluation_report.json"
    if not os.path.exists(rp):
        return None
    p = load_preds(rp)
    if p is None:
        return None
    return p


def gate_report(y, s_con, s_neu, symb, thr):
    """Hard route: use constraint score where symb>=thr else neural. Report AUROC."""
    route = symb >= thr
    s = np.where(route, s_con, s_neu)
    return auroc(y, s), route.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con_head", default="constraint_no_conflict")
    ap.add_argument("--neu_head", default="mlp")
    args = ap.parse_args()

    datasets = [("id", "StepGame"), ("spartqa", "spartqa"), ("babi", "babi"),
                ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]

    print(f"con_head={args.con_head}  neu_head={args.neu_head}\n")
    print(f"{'dataset':12s} {'symb':>6s} {'con':>6s} {'neu':>6s} "
          f"{'best-of':>7s} {'gate.5':>7s} {'oracle':>7s}")
    rows = {}
    for tag, cname in datasets:
        cache_dir = f"{CACHE_ROOT}/{cname}/{MODEL}"
        con = eval_head(args.con_head, tag, cache_dir)
        neu = eval_head(args.neu_head, tag, cache_dir)
        if con is None or neu is None:
            print(f"{cname:12s}  (missing preds)")
            continue
        ids_c, y, s_con, kh = con
        ids_n, y2, s_neu, _ = neu
        # align by sample id
        assert np.array_equal(ids_c, ids_n), f"id mismatch {cname}"
        assert np.array_equal(y, y2)
        tf = load_symbolizability(cache_dir, "test")
        # predictions' sample_id is the dataset row position -> index cache rows
        sel = ids_c.astype(int)
        tf = tf[sel]
        symb = tf[:, PARSE_RATE_IDX]

        a_con = auroc(y, s_con); a_neu = auroc(y, s_neu)
        a_gate, frac = gate_report(y, s_con, s_neu, symb, 0.5)
        # oracle upper bound: per-sample pick the score closer to label
        s_or = np.where(np.abs(s_con - y) <= np.abs(s_neu - y), s_con, s_neu)
        a_or = auroc(y, s_or)
        best = max(a_con, a_neu)
        print(f"{cname:12s} {symb.mean():6.3f} {a_con:6.3f} {a_neu:6.3f} "
              f"{best:7.3f} {a_gate:7.3f} {a_or:7.3f}")
        rows[cname] = dict(symb=symb, y=y, s_con=s_con, s_neu=s_neu, kh=kh,
                           tf=tf, a_con=a_con, a_neu=a_neu)

    # ---- symbolizability -> gain curve (pooled across datasets) ----
    print("\n--- symbolizability vs per-sample signal quality (pooled) ---")
    allsymb = np.concatenate([r["symb"] for r in rows.values()])
    ally = np.concatenate([r["y"] for r in rows.values()])
    allcon = np.concatenate([r["s_con"] for r in rows.values()])
    allneu = np.concatenate([r["s_neu"] for r in rows.values()])
    bins = [(-0.01, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.99), (0.99, 1.01)]
    print(f"{'symb bin':12s} {'n':>5s} {'AUROC_con':>10s} {'AUROC_neu':>10s} {'con-neu':>8s}")
    for lo, hi in bins:
        m = (allsymb > lo) & (allsymb <= hi)
        if m.sum() < 20:
            print(f"({lo:.2f},{hi:.2f}]  n={m.sum()} (too few)"); continue
        ac = auroc(ally[m], allcon[m]); an = auroc(ally[m], allneu[m])
        print(f"({lo:.2f},{hi:.2f}] {m.sum():6d} {ac:10.3f} {an:10.3f} {ac-an:8.3f}")

    return rows


if __name__ == "__main__":
    main()
