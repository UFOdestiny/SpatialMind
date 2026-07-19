#!/usr/bin/env python3
"""Full metric x protocol comparison for the headline table (concern #7).

For every backbone x dataset, and every method (baselines/heads + SpatialMind),
we recompute macro-Brier AND AURC under two UNIFORM calibration protocols:
  A: monotonic Platt (StandardCalibrator) for all
  B: DARC macro-Brier calibrator (select_calibrator) for all
Both are fit on validation, applied once to test. SpatialMind is recalibrated
from its RAW combined score so the comparison is symmetric.

Then we report the relative improvement of SpatialMind over the strongest
baseline under two conventions:
  DEF-A: single baseline with the best MEAN metric across datasets
  DEF-B: per-dataset best baseline, then averaged
for macro-Brier (lower better) and AURC (lower better), per backbone + aggregate.

AURC note: the implementation uses confidence=max(p,1-p) at a 0.5 threshold, so
it is calibration-dependent; we therefore evaluate it under both protocols too.
AUROC is protocol-invariant (both calibrators monotone) and printed once.

Caches SM raw scores to /tmp/sm_raw_<bb>.json so re-runs are fast.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402
from models.calibration import StandardCalibrator  # noqa: E402
from scripts.metrics import compute_aurc  # noqa: E402
from scripts.paper_calib_protocol import sm_raw_scores, MODELS  # noqa: E402

BBS = [("mistral", "Mistral-7B"), ("llama", "Llama-3.1-8B"),
       ("gemma", "Gemma-2-9B"), ("phi", "Phi-4"), ("qwen", "Qwen3-8B")]
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]


def platt(sv, yv, st):
    c = StandardCalibrator().fit(np.clip(sv, 1e-6, 1 - 1e-6), yv)
    return c.transform(np.clip(st, 1e-6, 1 - 1e-6)) if c.fitted else np.clip(st, 1e-6, 1 - 1e-6)


def darc(sv, yv, st):
    cal, _ = F.select_calibrator(np.clip(sv, 1e-6, 1 - 1e-6), yv)
    return F._apply_calib(cal, st)


def metrics_both(sv, yv, st, yt):
    pA = platt(sv, yv, st); pB = darc(sv, yv, st)
    return {"A_brier": F._macro_brier(yt, pA), "A_aurc": compute_aurc(yt, pA),
            "B_brier": F._macro_brier(yt, pB), "B_aurc": compute_aurc(yt, pB),
            "auroc": F.auroc(yt, st)}


def collect(bb):
    R = f"spatialmind/results/constraint_guided_{bb}"
    sub = f"constraint_guided_{bb}"
    model = MODELS[bb]
    cache_p = f"/tmp/sm_raw_{bb}.json"
    sm_cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    out = {}
    for tag, disp, cname in DATASETS:
        sv_sig = F.collect_signals(R, tag, cname, "validation", sub)
        st_sig = F.collect_signals(R, tag, cname, "test", sub)
        methods = {}
        for k in sorted(set(sv_sig) & set(st_sig)):
            if k == "random":
                continue
            idv, yv, sv = sv_sig[k]; idt, yt, st = st_sig[k]
            if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                continue
            methods[k] = metrics_both(sv, yv, st, yt)
        # SpatialMind
        if disp not in sm_cache:
            try:
                smr = sm_raw_scores(R, sub, model, tag, cname)
            except Exception:
                smr = None
            sm_cache[disp] = ([x.tolist() for x in smr] if smr else None)
        smr = sm_cache[disp]
        sm = None
        if smr:
            yv, pv, yt, pt = [np.array(x) for x in smr]
            sm = metrics_both(pv, yv, pt, yt)
        out[disp] = {"methods": methods, "sm": sm}
    json.dump(sm_cache, open(cache_p, "w"))
    return out


def rel(base, sm):  # lower-is-better metric; positive = SM better
    return (base - sm) / base * 100 if base else float("nan")


def summarize(allout):
    import statistics as st
    for metric, akey, bkey in [("macro-Brier", "A_brier", "B_brier"),
                               ("AURC", "A_aurc", "B_aurc")]:
        print(f"\n############ {metric} ############")
        for proto, key in [("A: Platt-for-all", akey), ("B: DARC-for-all", bkey)]:
            print(f"\n--- Protocol {proto} ---")
            print(f"{'backbone':12s}{'SM mean':>9s}{'DEF-A%':>9s}{'DEF-B%':>9s}   best-mean-base")
            defA, defB = [], []
            for bb, disp in BBS:
                if bb not in allout:
                    continue
                o = allout[bb]
                sm_vals, perbest, permethod = [], [], {}
                for ds, rec in o.items():
                    if not rec["sm"]:
                        continue
                    sm_vals.append(rec["sm"][key])
                    bvals = {m: v[key] for m, v in rec["methods"].items()}
                    if bvals:
                        perbest.append(min(bvals.values()))
                        for m, v in bvals.items():
                            permethod.setdefault(m, []).append(v)
                if not sm_vals:
                    continue
                msm = st.mean(sm_vals)
                n = len(sm_vals)
                means = {m: st.mean(v) for m, v in permethod.items() if len(v) >= n - 1}
                bestk = min(means, key=means.get)
                ra = rel(means[bestk], msm)
                rb = rel(st.mean(perbest), msm)
                defA.append(ra); defB.append(rb)
                print(f"{disp:12s}{msm:>9.3f}{ra:>8.1f}%{rb:>8.1f}%   {bestk}={means[bestk]:.3f}")
            if defA:
                nonphi_a = [r for (bb, _), r in zip(BBS, defA) if bb != "phi"]
                print(f"{'AVG (4bb)':12s}{'':>9s}{st.mean(nonphi_a):>8.1f}%"
                      f"{st.mean([r for (bb,_),r in zip(BBS,defB) if bb!='phi']):>8.1f}%")


def main():
    allout = {}
    for bb, disp in BBS:
        if os.path.isdir(f"spatialmind/results/constraint_guided_{bb}"):
            allout[bb] = collect(bb)
            print(f"[{bb} done]", flush=True)
    json.dump(allout, open("/tmp/metric_protocol.json", "w"), indent=2)
    summarize(allout)


if __name__ == "__main__":
    main()
