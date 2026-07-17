#!/usr/bin/env python3
"""Post-hoc calibration exploration for the fusion combiner (no GPU).

Goal: beat the sampling-SOTA on macro-Brier / ECE WITHOUT sacrificing our
already-strong AUROC. Every calibrator here is fit on VALIDATION only and applied
once to test (no leakage), exactly like fusion.py. Monotonic maps leave AUROC
provably unchanged; only the isotonic/monotone-preserving ones are used so the
ranking (hence AUROC) is preserved.

Why the current fusion.py under-calibrates on macro-Brier: its select_calibrator
scores candidates with the PLAIN Brier / NLL, but the paper benchmark
(benchmark_fair.py) reports class-BALANCED (macro) Brier. On imbalanced splits
(StepGame pos~0.24, etc.) those objectives disagree -> the "AU+" cells (win
AUROC, lose Brier). Here we (a) select the calibrator on the SAME macro-Brier
objective via val-internal CV, and (b) add a couple of imbalance-aware maps.

Run:
  python scripts/calib_explore.py                 # all 3 backbones, table
  python scripts/calib_explore.py --backbone llama
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fusion as F          # noqa: E402
from scripts.metrics import compute_ece  # noqa: E402

BACKBONES = {
    "llama":   ("constraint_guided_v11_llama",   "Llama-3.1-8B-Instruct"),
    "mistral": ("constraint_guided_v11_mistral", "Mistral-7B-Instruct-v0.3"),
    "gemma":   ("constraint_guided_v11_gemma",   "gemma-2-9b-it"),
}
DATASETS = [("id", "StepGame"), ("spartqa", "spartqa"), ("SpaRTUN", "SpaRTUN"),
            ("SpaceNLI", "SpaceNLI"), ("SpaRP_PS3", "SpaRP_PS3")]
CACHE_ROOT = "spatialmind/cache/cached_features"

EPS = 1e-6


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def macro_brier(y, s):
    """Class-balanced Brier == benchmark_fair.macro_brier."""
    s = _clip(s)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


# --------------------------------------------------------------------------- #
# Calibrator family (all monotonic in p -> AUROC preserved)
# --------------------------------------------------------------------------- #
def fit_identity(p, y):
    return ("identity",)


def fit_temp_nll(p, y):
    """Temperature scaling, NLL objective (what fusion.py uses today)."""
    z = _logit(p); best_T, best = 1.0, 1e18
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 8.0, 20)]):
        q = _clip(1 / (1 + np.exp(-z / T)))
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_T = nll, T
    return ("temp", best_T)


def fit_temp_macrobrier(p, y):
    """Temperature scaling, but chosen to minimise MACRO Brier (imbalance-aware)."""
    z = _logit(p); best_T, best = 1.0, 1e18
    for T in np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 8.0, 20)]):
        q = _clip(1 / (1 + np.exp(-z / T)))
        mb = macro_brier(y, q)
        if mb < best:
            best, best_T = mb, T
    return ("temp", best_T)


def fit_beta(p, y):
    """Platt/beta in logit space, NLL gradient (fusion.py's variant)."""
    z = _logit(p); a, b = 1.0, 0.0
    for _ in range(2000):
        q = _clip(1 / (1 + np.exp(-(a * z + b))))
        a -= 0.3 * np.mean((q - y) * z); b -= 0.3 * np.mean(q - y)
    return ("beta", a, b)


def fit_affine_macrobrier(p, y):
    """Affine map in logit space (a*z+b) minimising MACRO Brier via grid search.

    Two knobs: slope a (sharpness) and bias b (prior shift). Grid keeps a>0 so
    the map is strictly monotone (AUROC unchanged). The bias term is what fixes
    the class-imbalance prior mismatch that plain temperature can't touch."""
    z = _logit(p); best, out = 1e18, (1.0, 0.0)
    for a in np.linspace(0.2, 3.0, 29):
        za = a * z
        for b in np.linspace(-3.0, 3.0, 61):
            q = _clip(1 / (1 + np.exp(-(za + b))))
            mb = macro_brier(y, q)
            if mb < best:
                best, out = mb, (a, b)
    return ("beta", out[0], out[1])


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, y)
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


CALIBRATORS = {
    "identity":       fit_identity,
    "temp_nll":       fit_temp_nll,
    "temp_macroBr":   fit_temp_macrobrier,
    "beta_nll":       fit_beta,
    "affine_macroBr": fit_affine_macrobrier,
    "isotonic":       fit_isotonic,
}


def cv_macro_brier(fit_fn, p, y, k=5):
    """Val-internal k-fold CV macro-Brier for one calibrator (honest selection)."""
    n = len(y); idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    scores = []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        if len(np.unique(y[tr])) < 2:
            continue
        try:
            cal = fit_fn(p[tr], y[tr])
            scores.append(macro_brier(y[te], apply_calib(cal, p[te])))
        except Exception:
            scores.append(1e9)
    return float(np.mean(scores)) if scores else 1e9


# --------------------------------------------------------------------------- #
# Rebuild uncalibrated fused val/test probs by re-running fusion internals.
# --------------------------------------------------------------------------- #
def rebuild_uncalibrated(R, model, tag, cname, sub):
    """Mirror fuse_one() up to (but not including) the post-hoc calibration, and
    return uncalibrated (p_val, y_val, p_test, y_test). None if degenerate."""
    cache = f"{CACHE_ROOT}/{sub}/StepGame/{model}" if tag == "id" \
        else f"{CACHE_ROOT}/{sub}/{tag}/{model}"
    sv = F.collect_signals(R, tag, cname, "validation", sub)
    st = F.collect_signals(R, tag, cname, "test", sub)
    common = sorted(set(sv) & set(st))
    if len(common) < 2:
        return None
    sv = {k: sv[k] for k in common}; st = {k: st[k] for k in common}
    tfv_full = F.load_trace_features(cache, "validation")
    tft_full = F.load_trace_features(cache, "test")
    Sv, yv, idv, tfv = F.align(sv, tfv_full)
    St, yt, idt, tft = F.align(st, tft_full)
    if len(np.unique(yv)) < 2:
        return None
    val_au = {k: F.auroc(yv, v) for k, v in Sv.items()}
    oriented = {}
    for k in sorted(Sv.keys()):
        a = val_au[k]
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        s_v, s_t = Sv[k], St[k]
        if a < 0.5:
            s_v, s_t = 1.0 - s_v, 1.0 - s_t
        oriented[k] = (s_v, s_t)
    if not oriented:
        return None
    Sv = {k: oriented[k][0] for k in oriented}
    St = {k: oriented[k][1] for k in oriented}
    n = len(yv); kfold = 5 if n >= 200 else 3
    cands = []
    for gate, l2 in F._GRID:
        a = F._kfold_cv_auroc(Sv, yv, tfv, gate, l2, kfold)
        if not np.isnan(a):
            cands.append((a, gate, l2))
    if cands:
        top = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= top - F._SEL_TOL]
        gate, l2 = max(near, key=lambda c: c[2])[1:]
    else:
        gate, l2 = "determinacy", 20.0
    Xv, _ = F.build_matrix(Sv, tfv, gate)
    Xt, _ = F.build_matrix(St, tft, gate)
    fit = F.fit_logreg(Xv, yv, l2=l2)
    p_val = F.apply_logreg(fit, Xv)
    p_test = F.apply_logreg(fit, Xt)
    return p_val, yv, p_test, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="all")
    args = ap.parse_args()
    bbs = list(BACKBONES) if args.backbone == "all" else [args.backbone]

    # accumulate per-calibrator test macro-Brier / ECE / AUROC across all cells
    agg = {c: {"mb": [], "ece": [], "au": []} for c in CALIBRATORS}
    agg_sel = {"mb": [], "ece": [], "au": []}      # our CV-macroBrier selector

    for bb in bbs:
        sub, model = BACKBONES[bb]
        R = f"spatialmind/results/{sub}"
        print(f"\n===== {bb} ({model}) =====")
        hdr = f"{'dataset':11s}{'metric':>8s}" + "".join(f"{c:>15s}" for c in CALIBRATORS) + f"{'CVsel':>9s}"
        print(hdr)
        for tag, cname in DATASETS:
            try:
                out = rebuild_uncalibrated(R, model, tag, cname, sub)
            except Exception as e:
                print(f"{cname:11s}  (error: {e})"); continue
            if out is None:
                print(f"{cname:11s}  (degenerate/missing)"); continue
            p_val, yv, p_test, yt = out
            row_mb, row_ece, row_au = {}, {}, {}
            for c, fn in CALIBRATORS.items():
                cal = fn(p_val, yv)
                q = apply_calib(cal, p_test)
                row_mb[c] = macro_brier(yt, q)
                row_ece[c] = compute_ece(yt.astype(int), q)
                row_au[c] = F.auroc(yt, q)
                agg[c]["mb"].append(row_mb[c]); agg[c]["ece"].append(row_ece[c]); agg[c]["au"].append(row_au[c])
            # CV-macroBrier selector: pick calibrator with best val-internal CV macroBrier
            cv = {c: cv_macro_brier(fn, p_val, yv) for c, fn in CALIBRATORS.items()}
            pick = min(cv, key=cv.get)
            calp = CALIBRATORS[pick](p_val, yv); qp = apply_calib(calp, p_test)
            sel_mb, sel_ece, sel_au = macro_brier(yt, qp), compute_ece(yt.astype(int), qp), F.auroc(yt, qp)
            agg_sel["mb"].append(sel_mb); agg_sel["ece"].append(sel_ece); agg_sel["au"].append(sel_au)

            print(f"{cname:11s}{'macroBr':>8s}" + "".join(f"{row_mb[c]:>15.3f}" for c in CALIBRATORS) + f"{sel_mb:>9.3f}")
            print(f"{'':11s}{'ECE':>8s}" + "".join(f"{row_ece[c]:>15.3f}" for c in CALIBRATORS) + f"{sel_ece:>9.3f}")
            print(f"{'':11s}{'AUROC':>8s}" + "".join(f"{row_au[c]:>15.3f}" for c in CALIBRATORS) + f"{sel_au:>9.3f}  [pick={pick}]")

    # overall means
    print("\n===== OVERALL MEAN across all cells =====")
    print(f"{'calibrator':>16s}{'macroBr':>10s}{'ECE':>10s}{'AUROC':>10s}")
    for c in CALIBRATORS:
        print(f"{c:>16s}{np.mean(agg[c]['mb']):>10.3f}{np.mean(agg[c]['ece']):>10.3f}{np.mean(agg[c]['au']):>10.3f}")
    print(f"{'CVsel(macroBr)':>16s}{np.mean(agg_sel['mb']):>10.3f}{np.mean(agg_sel['ece']):>10.3f}{np.mean(agg_sel['au']):>10.3f}")


if __name__ == "__main__":
    main()
