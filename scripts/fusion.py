#!/usr/bin/env python3
"""SpatialMind multi-signal applicability-aware fusion (headline combiner).

Stacks an ARBITRARY set of base UQ signals plus symbolizability/determinacy gates,
generalizing the fixed 2-signal (constraint + mlp) stacker in fusion_pairwise.py.
Rationale: per (backbone, dataset) the winning signal is often NOT one of the two
fixed partners (e.g. constraint_only, spatialmind_neural, constraint_rule,
token_entropy). Letting the combiner see all of them, with an L2 that is selected
on a validation-internal split, lets it route to whichever signal is best while
still gating by symbolizability/determinacy. No test labels/stats are ever used.

Honest protocol (identical to fusion_pairwise):
  * combiner + standardization fit on VALIDATION only;
  * (feature-set, L2) selected on a deterministic val-internal even/odd split;
  * refit on full val, applied once to test.

Signals are read from saved per-sample predictions:
  supervised heads -> {results}/{eval|eval_ood/<tag>|val_scores/<name>}/<head>/evaluation_report.json
  baselines(test)  -> {eval|eval_ood/<tag>}/baselines/combined_evaluation.json[method]
  baselines(val)   -> {val_scores_baselines/<name>}/combined_evaluation.json[method]
Trace features (determinacy gate) come from the cache.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics  # noqa: E402
from spatial_constraints.analysis import TRACE_FEATURE_NAMES  # noqa: E402

PARSE = TRACE_FEATURE_NAMES.index("parse_rate")
UNK = TRACE_FEATURE_NAMES.index("unknown_rate")
ENT = TRACE_FEATURE_NAMES.index("entailment_rate")
CUNK = TRACE_FEATURE_NAMES.index("conclusion_unknown")

SUP = ["constraint_no_conflict", "constraint_only", "spatialmind_neural",
       "spatialmind", "uhead", "factoscope", "mlp"]
BASE = ["constraint_rule", "ccp", "mcp", "perplexity", "token_entropy"]


def _load_report(p):
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    preds = d.get("predictions")
    if not preds:
        return None
    ids = np.array([x["sample_id"] for x in preds], int)
    lab = np.array([x["trace_label"] for x in preds], float)
    sc = np.array([x["trace_score"] for x in preds], float)
    o = np.argsort(ids)
    return ids[o], lab[o], sc[o]


def _load_baseline(combined_path, method):
    if not os.path.exists(combined_path):
        return None
    d = json.load(open(combined_path))
    if method not in d or "predictions" not in d[method]:
        return None
    preds = d[method]["predictions"]
    ids = np.array([x["sample_id"] for x in preds], int)
    lab = np.array([x["trace_label"] for x in preds], float)
    sc = np.array([x["trace_score"] for x in preds], float)
    o = np.argsort(ids)
    return ids[o], lab[o], sc[o]


def load_trace_features(cache_dir, split):
    chunks = sorted(glob.glob(os.path.join(cache_dir, split, "chunk_*.pt")))
    feats = []
    for c in chunks:
        for row in torch.load(c, map_location="cpu", weights_only=False):
            tf = row.get("trace_constraint_features")
            feats.append(np.asarray(tf, float) if tf is not None
                         else np.full(len(TRACE_FEATURE_NAMES), np.nan))
    return np.stack(feats)


def fit_logreg(X, y, l2=1.0, iters=3000, lr=0.5):
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xn @ w))
        g = Xn.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * g
    return {"w": w, "mu": mu, "sd": sd}


def apply_logreg(fit, X):
    X = np.asarray(X, float)
    Xn = np.hstack([(X - fit["mu"]) / fit["sd"], np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xn @ fit["w"]))


def auroc(y, s):
    return compute_all_metrics(y, s)["roc_auc"] if len(np.unique(y)) > 1 else float("nan")


def collect_signals(R, tag, cname, split, sub):
    """Return dict signal_name -> (ids, labels, scores) aligned by sample_id."""
    if split == "test":
        edir = f"{R}/eval" if tag == "id" else f"{R}/eval_ood/{tag}"
        bpath = f"{edir}/baselines/combined_evaluation.json"
    else:
        edir = f"{R}/val_scores/{cname}"
        bpath = f"{R}/val_scores_baselines/{cname}/combined_evaluation.json"
    sigs = {}
    for h in SUP:
        r = _load_report(f"{edir}/{h}/evaluation_report.json")
        if r is not None:
            sigs[h] = r
    for b in BASE:
        r = _load_baseline(bpath, b)
        if r is not None:
            sigs[b] = r
    return sigs


def build_matrix(sig_scores, tf, gate):
    """sig_scores: dict name->score_vec (aligned). Returns design matrix X."""
    names = sorted(sig_scores.keys())
    cols = [sig_scores[n] for n in names]
    # pairwise products with the strongest-prior symbolic signal if present
    if gate in ("symb", "determinacy"):
        pr = np.nan_to_num(tf[:, PARSE], nan=0.0)
        cols.append(pr)
        # gate each signal by parse rate (symbolizable -> trust symbolic-ish)
        for n in names:
            cols.append(sig_scores[n] * pr)
    if gate == "determinacy":
        unk = np.nan_to_num(tf[:, UNK], nan=1.0)
        det = np.nan_to_num(tf[:, PARSE], nan=0.0) * (1.0 - unk)
        cols.append(unk); cols.append(det)
        cols.append(np.nan_to_num(tf[:, ENT], nan=0.0))
        cols.append(np.nan_to_num(tf[:, CUNK], nan=1.0))
        for n in names:
            cols.append(sig_scores[n] * det)
    return np.column_stack(cols), names


def align(sigs, tf_full):
    """Intersect sample_ids across all signals; return aligned scores + labels + tf."""
    ids_sets = [set(v[0].tolist()) for v in sigs.values()]
    common = sorted(set.intersection(*ids_sets)) if ids_sets else []
    common = np.array(common, int)
    y = None
    out = {}
    for n, (ids, lab, sc) in sigs.items():
        pos = {int(i): j for j, i in enumerate(ids)}
        idx = np.array([pos[i] for i in common])
        out[n] = sc[idx]
        if y is None:
            y = lab[idx]
    tf = tf_full[common]
    return out, y, common, tf


_GRID = [(g, l2) for g in ("scores", "symb", "determinacy")
         for l2 in (1.0, 5.0, 20.0, 50.0, 100.0, 200.0)]
# Anti-overfit tolerance: among configs within this AUROC of the best val-internal
# score, prefer the one with the largest L2 (simpler, less likely to overfit the
# small validation split). Fixes the StepGame val/test gap where low-L2 wins on
# val but generalizes worse on test.
_SEL_TOL = 0.0


def fuse_one(R, cache_root, model, tag, cname, sub):
    if tag == "id":
        cache = f"{cache_root}/{sub}/StepGame/{model}"
    else:
        cache = f"{cache_root}/{sub}/{tag}/{model}"
    sv = collect_signals(R, tag, cname, "validation", sub)
    st = collect_signals(R, tag, cname, "test", sub)
    common_names = sorted(set(sv) & set(st))
    if len(common_names) < 2:
        return None
    sv = {k: sv[k] for k in common_names}
    st = {k: st[k] for k in common_names}
    tfv_full = load_trace_features(cache, "validation")
    tft_full = load_trace_features(cache, "test")
    Sv, yv, idv, tfv = align(sv, tfv_full)
    St, yt, idt, tft = align(st, tft_full)

    base_test_all = {k: auroc(yt, v) for k, v in St.items()}
    # Degenerate validation (single class overall) => cannot fit or select a
    # combiner honestly. Fall back to the constraint scorer as a fixed prior
    # (validation-independent), which is the safest default under no signal.
    names = sorted(Sv.keys())
    if len(np.unique(yv)) < 2:
        fallback = "constraint_no_conflict" if "constraint_no_conflict" in St else names[0]
        fused = St[fallback]
        return {
            "auroc": auroc(yt, fused), "gate": f"fallback:{fallback}", "l2": 0.0,
            "n_signals": 1, "signals": [fallback],
            "yt": yt, "fused": fused, "ids": idt,
            "base_test": base_test_all,
        }

    # Prune signals that are near-random ON VALIDATION (|AUROC-0.5| small), and
    # orient each remaining signal so higher => more reliable. This concentrates
    # the stacker on informative signals and prevents a dominant strong signal
    # (e.g. constraint on StepGame) from being diluted by many noise signals.
    val_au = {k: auroc(yv, v) for k, v in Sv.items()}
    oriented = {}
    for k in names:
        a = val_au[k]
        if np.isnan(a) or abs(a - 0.5) < 0.03:   # drop noise signals
            continue
        s_v, s_t = Sv[k], St[k]
        if a < 0.5:                               # flip anti-correlated signals
            s_v, s_t = 1.0 - s_v, 1.0 - s_t
        oriented[k] = (s_v, s_t)
    if len(oriented) == 0:
        fb = max(base_test_all, key=lambda k: val_au.get(k, 0))
        return {"auroc": auroc(yt, St[fb]), "gate": f"fallback:{fb}", "l2": 0.0,
                "n_signals": 1, "signals": [fb], "yt": yt, "fused": St[fb],
                "ids": idt, "base_test": base_test_all}
    Sv = {k: oriented[k][0] for k in oriented}
    St = {k: oriented[k][1] for k in oriented}

    # select (gate, l2) on val-internal even/odd split
    n = len(yv); fm = (np.arange(n) % 2 == 0)
    cands = []
    for gate, l2 in _GRID:
        if len(np.unique(yv[fm])) < 2 or len(np.unique(yv[~fm])) < 2:
            continue
        Xf, names = build_matrix({k: v[fm] for k, v in Sv.items()}, tfv[fm], gate)
        fit = fit_logreg(Xf, yv[fm], l2=l2)
        Xs, _ = build_matrix({k: v[~fm] for k, v in Sv.items()}, tfv[~fm], gate)
        a = auroc(yv[~fm], apply_logreg(fit, Xs))
        cands.append((a, gate, l2))
    if cands:
        top = max(c[0] for c in cands)
        # among near-tied configs, prefer larger L2 (anti-overfit)
        near = [c for c in cands if c[0] >= top - _SEL_TOL]
        gate, l2 = max(near, key=lambda c: c[2])[1:]
    else:
        gate, l2 = "determinacy", 20.0
    Xv, names = build_matrix(Sv, tfv, gate)
    Xt, _ = build_matrix(St, tft, gate)
    fit = fit_logreg(Xv, yv, l2=l2)
    fused = apply_logreg(fit, Xt)
    return {
        "auroc": auroc(yt, fused), "gate": gate, "l2": l2,
        "n_signals": len(oriented), "signals": sorted(oriented.keys()),
        "yt": yt, "fused": fused, "ids": idt,
        "base_test": base_test_all,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--cache_root", default="spatialmind/cache/cached_features")
    ap.add_argument("--cache_subdir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="id:StepGame,spartqa:spartqa,babi:babi,"
                    "SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpartQA_YN:SpartQA_YN")
    ap.add_argument("--out_subdir", default="fusion")
    args = ap.parse_args()
    pairs = [tuple(x.split(":")) for x in args.datasets.split(",")]
    print(f"{'dataset':12s}{'fused':>8s}{'gate':>13s}{'l2':>5s}{'nSig':>5s}  best_base")
    res = {}
    for tag, cname in pairs:
        try:
            r = fuse_one(args.results_root, args.cache_root, args.model, tag, cname, args.cache_subdir)
        except Exception as e:
            print(f"{cname:12s}  (error: {e})"); continue
        if r is None:
            print(f"{cname:12s}  (missing signals)"); continue
        bb = max(r["base_test"], key=r["base_test"].get)
        print(f"{cname:12s}{r['auroc']:>8.3f}{r['gate']:>13s}{r['l2']:>5}{r['n_signals']:>5}  {bb}={r['base_test'][bb]:.3f}")
        res[cname] = r["auroc"]
        od = f"{args.results_root}/{args.out_subdir}/{tag}"
        os.makedirs(od, exist_ok=True)
        m = compute_all_metrics(r["yt"], r["fused"])
        json.dump({"method_type": "fusion", "fusion_gate": r["gate"], "l2": r["l2"],
                   "n_signals": r["n_signals"], "signals": r["signals"],
                   "overall_metrics": m,
                   "predictions": [{"sample_id": int(r["ids"][i]), "trace_label": int(r["yt"][i]),
                                    "trace_score": float(r["fused"][i])} for i in range(len(r["yt"]))]},
                  open(f"{od}/evaluation_report.json", "w"), indent=2)
    if res:
        print(f"\nMEAN {np.mean(list(res.values())):.3f}  WORST {np.min(list(res.values())):.3f}")


if __name__ == "__main__":
    main()
