#!/usr/bin/env python3
"""Which calibration protocol should the headline table use? (concern #7)

Both candidate calibrators are MONOTONIC, so AUROC is unchanged for every method
under either protocol; only the class-balanced Brier column moves. We recompute
macro-Brier for EVERY method (baselines, heads, and SpatialMind) under two
UNIFORM protocols, each fit on validation and applied once to test:

  Protocol A (Platt for all) : StandardCalibrator, a*logit(s)+b, a>0
  Protocol B (DARC for all)  : select_calibrator, macro-Brier-aware, val-CV chosen

To recalibrate SpatialMind fairly we reproduce its combiner locally (screen +
orient + selected (gate,l2) via fuse_one's own logic) and take the RAW combined
probability on validation and test BEFORE any post-hoc calibration, then apply
each protocol uniformly. Baselines/heads are recalibrated from their own stored
validation/test scores. No test labels touch any fit.

Reports, per backbone/dataset and protocol: SpatialMind macro-Brier, the
best-baseline macro-Brier under the SAME protocol (fair moving opponent), and the
gap. Prints an aggregate "how many cells SM wins Brier" per protocol.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402
from models.calibration import StandardCalibrator  # noqa: E402

BBS = [("mistral", "Mistral-7B"), ("llama", "Llama-3.1-8B"),
       ("gemma", "Gemma-2-9B"), ("phi", "Phi-4"), ("qwen", "Qwen3-8B")]
MODELS = {"mistral": "Mistral-7B-Instruct-v0.3", "llama": "Llama-3.1-8B-Instruct",
          "gemma": "gemma-2-9b-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
CACHE_ROOT = "spatialmind/cache/cached_features"


def platt(sv, yv, st):
    c = StandardCalibrator().fit(np.clip(sv, 1e-6, 1 - 1e-6), yv)
    return c.transform(np.clip(st, 1e-6, 1 - 1e-6)) if c.fitted else np.clip(st, 1e-6, 1 - 1e-6)


def darc_cal(sv, yv, st):
    cal, _ = F.select_calibrator(np.clip(sv, 1e-6, 1 - 1e-6), yv)
    return F._apply_calib(cal, st)


def sm_raw_scores(R, sub, model, tag, cname):
    """Reproduce fuse_one's combiner and return RAW (pre-calibration) p_val, p_test,
    plus yv, yt. Mirrors fusion.fuse_one exactly up to the calibration step."""
    cache = (f"{CACHE_ROOT}/{sub}/StepGame/{model}" if tag == "id"
             else f"{CACHE_ROOT}/{sub}/{tag}/{model}")
    sv = F.collect_signals(R, tag, cname, "validation", sub)
    st = F.collect_signals(R, tag, cname, "test", sub)
    common = sorted(set(sv) & set(st))
    if len(common) < 2:
        return None
    sv = {k: sv[k] for k in common}; st = {k: st[k] for k in common}
    tfv = F.load_trace_features(cache, "validation")
    tft = F.load_trace_features(cache, "test")
    Sv, yv, _, tfv = F.align(sv, tfv)
    St, yt, _, tft = F.align(st, tft)
    if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
        return None
    # screen + orient (validation only)
    oriented = {}
    for k in sorted(Sv):
        a = F.auroc(yv, Sv[k])
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        s_v, s_t = Sv[k], St[k]
        if a < 0.5:
            s_v, s_t = 1 - s_v, 1 - s_t
        oriented[k] = (s_v, s_t)
    if not oriented:
        return None
    Ov = {k: oriented[k][0] for k in oriented}
    Ot = {k: oriented[k][1] for k in oriented}
    # select (gate, l2) by k-fold CV, same as fuse_one
    n = len(yv); kfold = 5 if n >= 200 else 3
    cands = []
    for gate, l2 in F._GRID:
        a = F._kfold_cv_auroc(Ov, yv, tfv, gate, l2, kfold)
        if not np.isnan(a):
            cands.append((a, gate, l2))
    if cands:
        top = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= top - F._SEL_TOL]
        gate, l2 = max(near, key=lambda c: c[2])[1:]
    else:
        gate, l2 = "determinacy", 20.0
    Xv, _ = F.build_matrix(Ov, tfv, gate)
    Xt, _ = F.build_matrix(Ot, tft, gate)
    fit = F.fit_logreg(Xv, yv, l2=l2)
    p_val = F.apply_logreg(fit, Xv)
    p_test = F.apply_logreg(fit, Xt)
    return yv, p_val, yt, p_test


def run_backbone(bb):
    R = f"spatialmind/results/constraint_guided_v11_{bb}"
    sub = f"constraint_guided_v11_{bb}"
    model = MODELS[bb]
    out = {}
    for tag, disp, cname in DATASETS:
        sv_sig = F.collect_signals(R, tag, cname, "validation", sub)
        st_sig = F.collect_signals(R, tag, cname, "test", sub)
        common = sorted(set(sv_sig) & set(st_sig))
        if len(common) < 2:
            out[disp] = {"error": "missing"}; continue
        baseA, baseB = {}, {}
        for k in common:
            idv, yv, sv = sv_sig[k]
            idt, yt, st = st_sig[k]
            if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                continue
            baseA[k] = F._macro_brier(yt, platt(sv, yv, st))
            baseB[k] = F._macro_brier(yt, darc_cal(sv, yv, st))
        if not baseA:
            out[disp] = {"error": "degenerate"}; continue
        rec = {"A_best_base": _best(baseA), "B_best_base": _best(baseB)}
        # SpatialMind under both protocols from its RAW combined score
        try:
            smr = sm_raw_scores(R, sub, model, tag, cname)
        except Exception as e:
            smr = None; rec["sm_err"] = str(e)
        if smr is not None:
            yv, pv, yt, pt = smr
            rec["auroc_sm"] = round(F.auroc(yt, pt), 4)
            rec["A_sm"] = round(F._macro_brier(yt, platt(pv, yv, pt)), 4)
            rec["B_sm"] = round(F._macro_brier(yt, darc_cal(pv, yv, pt)), 4)
        out[disp] = rec
    return out


def _best(md):
    if not md:
        return None
    k = min(md, key=md.get)
    return {"name": k, "macro_brier": round(md[k], 4)}


def main():
    print("AUROC is identical under both protocols (both calibrators are monotone);")
    print("only the class-balanced Brier column moves.\n")
    allout = {}
    winsA = winsB = cells = 0
    gapsA, gapsB = [], []
    for bb, disp in BBS:
        if not os.path.isdir(f"spatialmind/results/constraint_guided_v11_{bb}"):
            continue
        r = run_backbone(bb)
        allout[bb] = r
        print(f"===== {bb} =====")
        print(f"{'dataset':10s}{'SM(A)':>9s}{'base(A)':>9s}{'gapA':>8s}   |{'SM(B)':>9s}{'base(B)':>9s}{'gapB':>8s}")
        for ds, v in r.items():
            if "error" in v or "A_sm" not in v:
                print(f"{ds:10s}  {v.get('error','no SM')}"); continue
            aSM, aB = v["A_sm"], v["A_best_base"]["macro_brier"]
            bSM, bB = v["B_sm"], v["B_best_base"]["macro_brier"]
            gapA = aB - aSM; gapB = bB - bSM
            cells += 1; winsA += gapA > 0; winsB += gapB > 0
            gapsA.append(gapA); gapsB.append(gapB)
            print(f"{ds:10s}{aSM:>9.3f}{aB:>9.3f}{gapA:>+8.3f}   |{bSM:>9.3f}{bB:>9.3f}{gapB:>+8.3f}")
    print(f"\nProtocol A (Platt-for-all):  SM wins Brier on {winsA}/{cells} cells, "
          f"mean gap {np.mean(gapsA):+.3f}")
    print(f"Protocol B (DARC-for-all):   SM wins Brier on {winsB}/{cells} cells, "
          f"mean gap {np.mean(gapsB):+.3f}")
    json.dump(allout, open("/tmp/calib_protocol.json", "w"), indent=2)


if __name__ == "__main__":
    main()
