#!/usr/bin/env python3
"""FAIR DARC ablation (P1 in review): every variant uses the IDENTICAL post-hoc
calibration protocol, so the Brier gap is no longer confounded by calibration.
Also reports paired-bootstrap 95% CIs and a two-sided bootstrap p-value for the
central contrast requested by the reviewer:

    scores-only stacking   vs.   + determinacy gate      (isolates the gate)
    scores-only stacking   vs.   SpatialMind (selected)

Every stacked variant:
  * uses the SAME oriented/screened signal bank (validation-only),
  * uses the SAME reference L2 for the three fixed-gate rows,
  * fits the SAME macro-Brier calibrator (select_calibrator) on validation and
    applies it once to test.
The SpatialMind row additionally selects (gate, L2, calibrator) on validation.

Pure post-processing over saved predictions; no GPU, no test labels used for any
fitting decision. Writes JSON to spatialmind/results/<ns>/fusion/fair_ablation.json
for every backbone in BACKBONES and prints a combined summary.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402

CACHE_ROOT = "spatialmind/cache/cached_features"
_MODELS = {"llama": "Llama-3.1-8B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3",
           "gemma": "gemma-2-9b-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
BACKBONES = ["mistral", "llama", "gemma", "phi", "qwen"]
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
REF_L2 = 20.0
NBOOT = 2000
rng = np.random.RandomState(0)


def prep(R, sub, model, tag, cname):
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
    return Sv, yv, tfv, St, yt, tft


def orient(Sv, St, yv):
    """Screen near-random signals and flip anti-correlated ones (validation-only),
    exactly as fusion.py.fuse_one does."""
    ov, ot = {}, {}
    for k in sorted(Sv):
        a = F.auroc(yv, Sv[k])
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        sv, st = Sv[k], St[k]
        if a < 0.5:
            sv, st = 1 - sv, 1 - st
        ov[k], ot[k] = sv, st
    return ov, ot


def stacked_calibrated(Ov, yv, tfv, Ot, tft, gate, l2):
    """Fit logreg at (gate, l2) on val, fit the SAME macro-Brier calibrator on
    val, apply once to test. Returns calibrated test scores."""
    Xv, _ = F.build_matrix(Ov, tfv, gate)
    Xt, _ = F.build_matrix(Ot, tft, gate)
    fit = F.fit_logreg(Xv, yv, l2=l2)
    p_val = F.apply_logreg(fit, Xv)
    p_test = F.apply_logreg(fit, Xt)
    cal, _ = F.select_calibrator(p_val, yv)
    return F._apply_calib(cal, p_test)


def paired_bootstrap(yt, s_a, s_b):
    """Two-sided bootstrap on delta AUROC = AUROC(a) - AUROC(b), paired by sample.
    Returns (delta, lo, hi, p_two_sided)."""
    d0 = F.auroc(yt, s_a) - F.auroc(yt, s_b)
    diffs = []
    n = len(yt)
    for _ in range(NBOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        diffs.append((F.auroc(yt[idx], s_a[idx]) - F.auroc(yt[idx], s_b[idx])))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples on the opposite side of 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(p, 1.0))


def run_backbone(bb):
    R = f"spatialmind/results/constraint_guided_v11_{bb}"
    sub = f"constraint_guided_v11_{bb}"
    model = _MODELS[bb]
    out = {"backbone": bb, "datasets": {}}
    for tag, disp, cname in DATASETS:
        try:
            got = prep(R, sub, model, tag, cname)
        except Exception as e:
            out["datasets"][disp] = {"error": str(e)}; continue
        if got is None:
            out["datasets"][disp] = {"error": "missing signals"}; continue
        Sv, yv, tfv, St, yt, tft = got
        if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
            out["datasets"][disp] = {"error": "degenerate labels"}; continue
        Ov, Ot = orient(Sv, St, yv)
        if not Ov:
            out["datasets"][disp] = {"error": "no informative signal"}; continue
        # three fixed-gate rows, ALL calibrated identically at REF_L2
        s_scores = stacked_calibrated(Ov, yv, tfv, Ot, tft, "scores", REF_L2)
        s_symb = stacked_calibrated(Ov, yv, tfv, Ot, tft, "symb", REF_L2)
        s_det = stacked_calibrated(Ov, yv, tfv, Ot, tft, "determinacy", REF_L2)
        # SpatialMind: read the selected combiner written by fusion.py
        fp = f"{R}/fusion/{tag}/evaluation_report.json"
        s_sm = fy = None
        if os.path.exists(fp):
            pr = json.load(open(fp))["predictions"]
            fy = np.array([x["trace_label"] for x in pr], float)
            s_sm = np.array([x["trace_score"] for x in pr], float)

        def cell(s, y=yt):
            return {"auroc": round(F.auroc(y, s), 4),
                    "macro_brier": round(F._macro_brier(y, s), 4)}

        d = {"n_test": int(len(yt)),
             "scores": cell(s_scores), "symb": cell(s_symb),
             "determinacy": cell(s_det)}
        # paired significance vs scores-only (all calibrated, same L2)
        dd, lo, hi, p = paired_bootstrap(yt, s_det, s_scores)
        d["det_vs_scores"] = {"delta_auroc": round(dd, 4),
                              "ci95": [round(lo, 4), round(hi, 4)],
                              "p_two_sided": round(p, 4)}
        if s_sm is not None and np.array_equal(fy, yt):
            d["spatialmind"] = cell(s_sm)
            sd, slo, shi, sp = paired_bootstrap(yt, s_sm, s_scores)
            d["sm_vs_scores"] = {"delta_auroc": round(sd, 4),
                                 "ci95": [round(slo, 4), round(shi, 4)],
                                 "p_two_sided": round(sp, 4)}
        elif s_sm is not None:
            # id ordering may differ; align by recomputing on stored preds only
            d["spatialmind"] = cell(s_sm, fy)
        out["datasets"][disp] = d
    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/fair_ablation.json", "w"), indent=2)
    return out


def main():
    allres = {}
    for bb in BACKBONES:
        R = f"spatialmind/results/constraint_guided_v11_{bb}"
        if not os.path.isdir(R):
            continue
        allres[bb] = run_backbone(bb)
    # summary print
    for bb, out in allres.items():
        print(f"\n===== {bb} (fair, all calibrated) =====")
        print(f"{'dataset':10s}{'scores':>16s}{'+symb':>16s}{'+determ':>16s}{'SpatialMind':>16s}   det-vs-scores")
        for disp, d in out["datasets"].items():
            if "error" in d:
                print(f"{disp:10s}  {d['error']}"); continue
            def fmt(k):
                c = d.get(k)
                return f"{c['auroc']:.3f}/{c['macro_brier']:.3f}" if c else "--"
            dv = d.get("det_vs_scores", {})
            sig = ""
            if dv:
                star = "*" if dv["p_two_sided"] < 0.05 else ""
                sig = f"Δ{dv['delta_auroc']:+.3f} p={dv['p_two_sided']:.3f}{star}"
            print(f"{disp:10s}{fmt('scores'):>16s}{fmt('symb'):>16s}{fmt('determinacy'):>16s}{fmt('spatialmind'):>16s}   {sig}")


if __name__ == "__main__":
    main()
