#!/usr/bin/env python3
"""Experimental fusion combiner for Phase-0 iteration (reads small trace-feature
caches from scripts/extract_trace_features.py so it runs in milliseconds).

Extends fusion.py with:
  * k-fold val-internal CV for (gate, L2) selection (instead of a single even/odd
    split) -> more stable routing on ~1000-sample val, tiny on babi (~196).
  * Brier-aware post-hoc calibration fit ON VALIDATION and applied to test:
    candidates = {identity, temperature, isotonic, beta}; pick the one with the
    best validation Brier (rank-preserving except isotonic which is monotone).
    AUROC is (near-)unchanged; Brier/ECE improve.

Honest protocol preserved: combiner + calibrator fit on validation only; test
touched once. No test labels/stats used for any fit or selection.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

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

SMALL_TF = "spatialmind/cache/trace_features_small"


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


def load_tf_small(sub, ds, split):
    p = f"{SMALL_TF}/{sub}/{ds}/{split}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["feats"]


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


def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


# ----- calibrators fit on validation -----
def fit_temperature(p_val, y):
    """Scale logits by 1/T; fit T>0 by 1-D search minimizing val NLL."""
    z = _logit(p_val)
    best_T, best = 1.0, 1e18
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 8.0, 20)]):
        q = _clip(1 / (1 + np.exp(-z / T)))
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_T = nll, T
    return ("temp", best_T)


def fit_beta(p_val, y):
    """Beta calibration: logit(q) = a*logit(p)+b, fit a,b by gradient on val NLL."""
    z = _logit(p_val)
    a, b = 1.0, 0.0
    for _ in range(2000):
        q = _clip(1 / (1 + np.exp(-(a * z + b))))
        ga = np.mean((q - y) * z); gb = np.mean(q - y)
        a -= 0.3 * ga; b -= 0.3 * gb
    return ("beta", a, b)


def fit_isotonic(p_val, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_val, y)
    return ("iso", ir)


def apply_calib(cal, p):
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
    """Pick calibrator by k-fold val Brier (guards tiny val like babi ~196)."""
    n = len(y_val)
    k = 5 if n >= 200 else 3
    idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    cand_makers = {
        "identity": lambda pv, yv: ("identity",),
        "temp": fit_temperature,
        "beta": fit_beta,
    }
    if n >= 400:                       # isotonic is flexible -> needs more val
        cand_makers["iso"] = fit_isotonic
    scores = {name: [] for name in cand_makers}
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        if len(np.unique(y_val[tr])) < 2:
            continue
        for name, mk in cand_makers.items():
            try:
                cal = mk(p_val[tr], y_val[tr])
                q = apply_calib(cal, p_val[te])
                scores[name].append(np.mean((q - y_val[te]) ** 2))
            except Exception:
                scores[name].append(1e9)
    means = {n_: (np.mean(v) if v else 1e9) for n_, v in scores.items()}
    # Anti-overfit: only leave identity if a candidate beats it by >2% relative
    # CV-Brier. Otherwise identity (safest, rank-preserving, no distortion).
    id_score = means.get("identity", 1e9)
    best = min(means, key=means.get)
    if best != "identity" and means[best] > id_score * 0.98:
        best = "identity"
    return cand_makers[best](p_val, y_val), best


def auroc(y, s):
    return compute_all_metrics(y, s)["roc_auc"] if len(np.unique(y)) > 1 else float("nan")


def collect_signals(R, tag, cname, split):
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
    names = sorted(sig_scores.keys())
    cols = [sig_scores[n] for n in names]
    if gate in ("symb", "determinacy"):
        pr = np.nan_to_num(tf[:, PARSE], nan=0.0)
        cols.append(pr)
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
    ids_sets = [set(v[0].tolist()) for v in sigs.values()]
    common = sorted(set.intersection(*ids_sets)) if ids_sets else []
    common = np.array(common, int)
    y = None; out = {}
    for n, (ids, lab, sc) in sigs.items():
        pos = {int(i): j for j, i in enumerate(ids)}
        idx = np.array([pos[i] for i in common])
        out[n] = sc[idx]
        if y is None:
            y = lab[idx]
    tf = tf_full[common] if tf_full is not None else np.zeros((len(common), len(TRACE_FEATURE_NAMES)))
    return out, y, common, tf


_GRID = [(g, l2) for g in ("scores", "symb", "determinacy")
         for l2 in (1.0, 5.0, 20.0, 50.0, 100.0, 200.0)]


def kfold_cv_score(Sv, yv, tfv, gate, l2, k):
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
    return np.mean(aus) if aus else float("nan")


def fuse_one(R, sub, tag, cname):
    ds_dir = "StepGame" if tag == "id" else tag
    sv = collect_signals(R, tag, cname, "validation")
    st = collect_signals(R, tag, cname, "test")
    common_names = sorted(set(sv) & set(st))
    if len(common_names) < 2:
        return None
    sv = {k: sv[k] for k in common_names}; st = {k: st[k] for k in common_names}
    tfv_full = load_tf_small(sub, ds_dir, "validation")
    tft_full = load_tf_small(sub, ds_dir, "test")
    Sv, yv, idv, tfv = align(sv, tfv_full)
    St, yt, idt, tft = align(st, tft_full)
    base_test_all = {k: auroc(yt, v) for k, v in St.items()}

    names = sorted(Sv.keys())
    if len(np.unique(yv)) < 2:
        fb = "constraint_no_conflict" if "constraint_no_conflict" in St else names[0]
        m = compute_all_metrics(yt, St[fb])
        return {"gate": f"fallback:{fb}", "l2": 0.0, "cal": "identity",
                "metrics": m, "base_test": base_test_all, "yt": yt,
                "fused": _clip(St[fb]), "ids": idt}

    val_au = {k: auroc(yv, v) for k, v in Sv.items()}
    oriented = {}
    for k in names:
        a = val_au[k]
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        s_v, s_t = Sv[k], St[k]
        if a < 0.5:
            s_v, s_t = 1.0 - s_v, 1.0 - s_t
        oriented[k] = (s_v, s_t)
    if not oriented:
        fb = max(base_test_all, key=lambda k: val_au.get(k, 0))
        m = compute_all_metrics(yt, St[fb])
        return {"gate": f"fallback:{fb}", "l2": 0.0, "cal": "identity",
                "metrics": m, "base_test": base_test_all, "yt": yt,
                "fused": _clip(St[fb]), "ids": idt}
    Sv = {k: oriented[k][0] for k in oriented}
    St = {k: oriented[k][1] for k in oriented}

    k = 5 if len(yv) >= 200 else 3
    cands = []
    for gate, l2 in _GRID:
        a = kfold_cv_score(Sv, yv, tfv, gate, l2, k)
        if not np.isnan(a):
            cands.append((a, gate, l2))
    if cands:
        top = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= top - 1e-9]
        gate, l2 = max(near, key=lambda c: c[2])[1:]
    else:
        gate, l2 = "determinacy", 20.0

    Xv, _ = build_matrix(Sv, tfv, gate)
    Xt, _ = build_matrix(St, tft, gate)
    fit = fit_logreg(Xv, yv, l2=l2)
    p_val = apply_logreg(fit, Xv)
    p_test = apply_logreg(fit, Xt)
    cal, calname = select_calibrator(p_val, yv)
    fused = apply_calib(cal, p_test)
    m = compute_all_metrics(yt, fused)
    return {"gate": gate, "l2": l2, "cal": calname, "n_signals": len(oriented),
            "signals": sorted(oriented.keys()), "metrics": m,
            "base_test": base_test_all, "yt": yt, "fused": fused, "ids": idt}


DATASETS = [("id", "StepGame"), ("spartqa", "spartqa"), ("babi", "babi"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--cache_subdir", required=True)
    ap.add_argument("--write", action="store_true", help="write evaluation_report.json under fusion_lab/")
    args = ap.parse_args()
    print(f"{'dataset':11s}{'AUROC':>7s}{'Brier':>7s}{'ECE':>7s}{'gate':>13s}{'l2':>5s}{'cal':>9s}")
    for tag, cname in DATASETS:
        r = fuse_one(args.results_root, args.cache_subdir, tag, cname)
        if r is None:
            print(f"{cname:11s}  (missing)"); continue
        m = r["metrics"]
        print(f"{cname:11s}{m['roc_auc']:>7.3f}{m['brier']:>7.3f}{m['ece']:>7.3f}"
              f"{r['gate']:>13s}{str(r['l2']):>5s}{r['cal']:>9s}")
        if args.write:
            od = f"{args.results_root}/fusion_lab/{tag}"
            os.makedirs(od, exist_ok=True)
            json.dump({"method_type": "fusion_lab", "fusion_gate": r["gate"],
                       "l2": r["l2"], "calibrator": r["cal"],
                       "overall_metrics": m,
                       "predictions": [{"sample_id": int(r["ids"][i]),
                                        "trace_label": int(r["yt"][i]),
                                        "trace_score": float(r["fused"][i])}
                                       for i in range(len(r["yt"]))]},
                      open(f"{od}/evaluation_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
