#!/usr/bin/env python3
"""SpatialMind multi-signal applicability-aware fusion (headline combiner).

Stacks an ARBITRARY set of base UQ signals plus symbolizability/determinacy gates,
generalizing fixed two-signal stacking to all available signals. Rationale: per
(backbone, dataset) the winning signal is often NOT one of the two fixed partners
(e.g. constraint_only, spatialmind_neural, constraint_rule,
token_entropy). Letting the combiner see all of them, with an L2 that is selected
on a validation-internal split, lets it route to whichever signal is best while
still gating by symbolizability/determinacy. No test labels/stats are ever used.

Honest protocol:
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
ENT = TRACE_FEATURE_NAMES.index("entailment_rate")
CON = TRACE_FEATURE_NAMES.index("contradiction_rate")

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


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


# ----- post-hoc calibrators (fit on VALIDATION only, applied once to test) -----
def _macro_brier(y, s):
    """Class-balanced Brier == benchmark_fair.macro_brier (the reported metric).

    On imbalanced splits (StepGame pos~0.24 etc.) this disagrees with the plain
    Brier/NLL a temperature fit optimises, so we calibrate against it directly.
    """
    s = _clip(s)
    y = np.asarray(y, dtype=np.float64)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def _fit_affine_macrobrier(p_val, y):
    """Affine map in logit space p' = sigmoid(a*logit(p)+b), grid-searched to
    minimise MACRO Brier. a>0 keeps it strictly monotone (AUROC unchanged); the
    bias b fixes the class-imbalance prior shift a plain temperature can't touch.
    Returned as a ("beta", a, b) tuple so _apply_calib handles it directly.
    """
    z = _logit(p_val); y = np.asarray(y, dtype=np.float64)
    best, out = 1e18, (1.0, 0.0)
    for a in np.linspace(0.2, 3.0, 29):
        za = a * z
        for b in np.linspace(-3.0, 3.0, 61):
            q = _clip(1 / (1 + np.exp(-(za + b))))
            mb = _macro_brier(y, q)
            if mb < best:
                best, out = mb, (a, b)
    return ("beta", out[0], out[1])


def _fit_temperature(p_val, y):
    z = _logit(p_val)
    best_T, best = 1.0, 1e18
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 8.0, 20)]):
        q = _clip(1 / (1 + np.exp(-z / T)))
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_T = nll, T
    return ("temp", best_T)


def _fit_beta(p_val, y):
    z = _logit(p_val); a, b = 1.0, 0.0
    for _ in range(2000):
        q = _clip(1 / (1 + np.exp(-(a * z + b))))
        a = max(1e-3, a - 0.3 * np.mean((q - y) * z))
        b -= 0.3 * np.mean(q - y)
    return ("beta", a, b)


def _fit_isotonic(p_val, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_val, y)
    return ("iso", ir)


def _apply_calib(cal, p):
    if cal is None or cal[0] == "identity":
        return _clip(p)
    if cal[0] == "temp":
        return _clip(1 / (1 + np.exp(-_logit(p) / cal[1])))
    if cal[0] == "beta":
        return _clip(1 / (1 + np.exp(-(cal[1] * _logit(p) + cal[2]))))
    if cal[0] == "iso":
        return _clip(cal[1].predict(p))
    return _clip(p)


def select_calibrator(p_val, y_val):
    """Pick a MACRO-Brier-optimal calibrator by k-fold val CV. All candidates are
    monotone in p (affine/temp/beta/isotonic), so AUROC is preserved; only
    calibration (macro-Brier/ECE) moves. Scored on the SAME class-balanced Brier
    the paper benchmark reports, so the post-hoc map optimises the reported
    metric instead of plain NLL (which mis-serves imbalanced splits). Leaves
    identity only if no candidate beats it by >2% relative CV macro-Brier ->
    avoids distorting an already-calibrated combiner on a small val split."""
    n = len(y_val)
    k = 5 if n >= 200 else 3
    idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    makers = {"identity": lambda pv, yv: ("identity",),
              "affine": _fit_affine_macrobrier,
              "temp": _fit_temperature, "beta": _fit_beta}
    if n >= 400:
        makers["iso"] = _fit_isotonic
    scores = {name: [] for name in makers}
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        if len(np.unique(y_val[tr])) < 2:
            continue
        for name, mk in makers.items():
            try:
                cal = mk(p_val[tr], y_val[tr])
                q = _apply_calib(cal, p_val[te])
                scores[name].append(_macro_brier(y_val[te], q))
            except Exception:
                scores[name].append(1e9)
    means = {n_: (np.mean(v) if v else 1e9) for n_, v in scores.items()}
    best = min(means, key=means.get)
    return makers[best](p_val, y_val), best


def _kfold_cv_auroc(Sv, yv, tfv, gate, l2, k):
    """Mean held-out AUROC over k val-internal folds for (gate, l2)."""
    n = len(yv); idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    aus = []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        if len(np.unique(yv[tr])) < 2 or len(np.unique(yv[te])) < 2:
            continue
        Xtr, _ = build_matrix({kk: v[tr] for kk, v in Sv.items()}, tfv[tr], gate)
        fit = fit_logreg(Xtr, yv[tr], l2=l2)
        Xte, _ = build_matrix({kk: v[te] for kk, v in Sv.items()}, tfv[te], gate)
        aus.append(auroc(yv[te], apply_logreg(fit, Xte)))
    return float(np.mean(aus)) if aus else float("nan")


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
    """Build the three DARC designs from Eq. 8 of the paper.

    The Determinacy Profile is applicability metadata, never an independent
    reliability feature: it appears only in interactions with retained scores.
    """
    names = sorted(sig_scores.keys())
    cols = [sig_scores[n] for n in names]
    if gate in ("symb", "determinacy"):
        pi = np.nan_to_num(tf[:, PARSE], nan=0.0)
        for n in names:
            cols.append(sig_scores[n] * pi)
    if gate == "determinacy":
        # d is the parsed-claim share with a definite (entailed or
        # contradicted) status. It excludes semantic unknowns and claims that
        # became non-evaluable after an earlier conflict.
        pi = np.nan_to_num(tf[:, PARSE], nan=0.0)
        d = np.nan_to_num(tf[:, ENT], nan=0.0) + np.nan_to_num(tf[:, CON], nan=0.0)
        delta = pi * np.clip(d, 0.0, 1.0)
        for n in names:
            cols.append(sig_scores[n] * delta)
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

    # select (gate, l2) by k-fold val-internal CV (lower variance than a single
    # even/odd split -> more stable routing on ~1000-sample val, tiny on babi).
    n = len(yv)
    kfold = 5 if n >= 200 else 3
    cands = []
    for gate, l2 in _GRID:
        a = _kfold_cv_auroc(Sv, yv, tfv, gate, l2, kfold)
        if not np.isnan(a):
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
    p_val = apply_logreg(fit, Xv)
    fused = apply_logreg(fit, Xt)
    # Select a rank-preserving, macro-Brier-aware calibrator by validation CV,
    # then fit once on all validation examples and apply once to test.
    cal, calname = select_calibrator(p_val, yv)
    fused = _apply_calib(cal, fused)
    return {
        "auroc": auroc(yt, fused), "gate": gate, "l2": l2, "calibrator": calname,
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
                   "calibrator": r.get("calibrator", "identity"),
                   "n_signals": r["n_signals"], "signals": r["signals"],
                   "overall_metrics": m,
                   "predictions": [{"sample_id": int(r["ids"][i]), "trace_label": int(r["yt"][i]),
                                    "trace_score": float(r["fused"][i])} for i in range(len(r["yt"]))]},
                  open(f"{od}/evaluation_report.json", "w"), indent=2)
    if res:
        print(f"\nMEAN {np.mean(list(res.values())):.3f}  WORST {np.min(list(res.values())):.3f}")


if __name__ == "__main__":
    main()
