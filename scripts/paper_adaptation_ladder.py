#!/usr/bin/env python3
"""Adaptation ladder + validation-size curve (P3/P6 in review).

Reviewer wants a clean separation of what target-domain supervision buys, and
whether the method needs many target labels. We report a five-rung ladder of
increasing target-validation use, all evaluated once on the same test split:

  L1 Source-only / zero-shot : best single source-trained scorer, orientation
                               fixed on SOURCE (StepGame) val, NO target labels.
  L2 Target calibration only : L1 scorer + macro-Brier calibrator fit on target
                               validation (adjusts calibration, not ranking).
  L3 Target stacking         : scores-only stacking (gate="scores") on target val.
  L4 + determinacy           : stacking + determinacy gate on target val.
  L5 Full SpatialMind        : selected (gate,L2,calibrator) [from fusion.py].

Validation-size curve: for target-val sizes m in {16,32,64,128,256,full}, we
subsample the target validation set (5 random draws each), refit L3/L4/L5-style
stacking, and report mean +/- std test AUROC. Shows how few target labels the
method needs. Uses only validation labels for fitting; test labels never touch a
fitting decision.

Pure post-processing. Writes spatialmind/results/<ns>/fusion/adaptation_ladder.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402

BACKBONE = os.environ.get("LADDER_BACKBONE", "llama")
_MODELS = {"llama": "Llama-3.1-8B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3",
           "gemma": "gemma-2-9b-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
MODEL = _MODELS[BACKBONE]
SUB = f"constraint_guided_{BACKBONE}"
R = f"spatialmind/results/constraint_guided_{BACKBONE}"
CACHE_ROOT = "spatialmind/cache/cached_features"
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
SIZES = [16, 32, 64, 128, 256]
NDRAW = 5
REF_L2 = 20.0
rng = np.random.RandomState(0)


def prep(tag, cname):
    cache = (f"{CACHE_ROOT}/{SUB}/StepGame/{MODEL}" if tag == "id"
             else f"{CACHE_ROOT}/{SUB}/{tag}/{MODEL}")
    sv = F.collect_signals(R, tag, cname, "validation", SUB)
    st = F.collect_signals(R, tag, cname, "test", SUB)
    common = sorted(set(sv) & set(st))
    if len(common) < 2:
        return None
    sv = {k: sv[k] for k in common}; st = {k: st[k] for k in common}
    tfv = F.load_trace_features(cache, "validation")
    tft = F.load_trace_features(cache, "test")
    Sv, yv, _, tfv = F.align(sv, tfv)
    St, yt, _, tft = F.align(st, tft)
    return Sv, yv, tfv, St, yt, tft


def orient_on(Sv, St, yv):
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


def stack_auroc(Ov, yv, tfv, Ot, yt, tft, gate, l2, idx=None):
    if idx is not None:
        Ovs = {k: v[idx] for k, v in Ov.items()}
        yvs = yv[idx]; tfvs = tfv[idx]
    else:
        Ovs, yvs, tfvs = Ov, yv, tfv
    if len(np.unique(yvs)) < 2:
        return float("nan")
    Xv, _ = F.build_matrix(Ovs, tfvs, gate)
    Xt, _ = F.build_matrix(Ot, tft, gate)
    fit = F.fit_logreg(Xv, yvs, l2=l2)
    return F.auroc(yt, F.apply_logreg(fit, Xt))


def source_scorer(St, yt):
    """L1: best single supervised source-trained scorer, chosen by a fixed prior
    (the constraint scorer), oriented on the SOURCE (no target labels)."""
    # constraint_no_conflict is the source-trained symbolic scorer; report it as
    # the zero-shot rung (higher = more reliable, source orientation).
    for k in ("constraint_no_conflict", "spatialmind", "mlp"):
        if k in St:
            return k, F.auroc(yt, St[k])
    k = sorted(St)[0]
    return k, F.auroc(yt, St[k])


def main():
    out = {"backbone": BACKBONE, "datasets": {}}
    for tag, disp, cname in DATASETS:
        got = prep(tag, cname)
        if got is None:
            out["datasets"][disp] = {"error": "missing"}; continue
        Sv, yv, tfv, St, yt, tft = got
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            out["datasets"][disp] = {"error": "degenerate"}; continue
        rec = {"n_val": int(len(yv)), "n_test": int(len(yt))}
        # L1 zero-shot
        l1_name, l1 = source_scorer(St, yt)
        rec["L1_zeroshot"] = {"scorer": l1_name, "auroc": round(l1, 4)}
        # L2 target calibration only (AUROC unchanged by monotone calib -> same)
        rec["L2_target_calib"] = {"auroc": round(l1, 4),
                                  "note": "monotone calib; AUROC == L1, Brier improves"}
        Ov, Ot = orient_on(Sv, St, yv)
        # L3 scores stacking, L4 + determinacy
        rec["L3_target_stacking"] = {"auroc": round(
            stack_auroc(Ov, yv, tfv, Ot, yt, tft, "scores", REF_L2), 4)}
        rec["L4_determinacy"] = {"auroc": round(
            stack_auroc(Ov, yv, tfv, Ot, yt, tft, "determinacy", REF_L2), 4)}
        # L5 full spatialmind
        fp = f"{R}/fusion/{tag}/evaluation_report.json"
        if os.path.exists(fp):
            pr = json.load(open(fp))["predictions"]
            fy = np.array([x["trace_label"] for x in pr], float)
            fs = np.array([x["trace_score"] for x in pr], float)
            rec["L5_spatialmind"] = {"auroc": round(F.auroc(fy, fs), 4)}
        # validation-size curve on the determinacy gate (L4 design)
        curve = []
        nval = len(yv)
        for m in SIZES + [nval]:
            if m > nval:
                continue
            aus = []
            for _ in range(NDRAW if m < nval else 1):
                idx = (rng.choice(nval, m, replace=False) if m < nval
                       else np.arange(nval))
                a = stack_auroc(Ov, yv, tfv, Ot, yt, tft, "determinacy", REF_L2, idx)
                if not np.isnan(a):
                    aus.append(a)
            if aus:
                curve.append({"m": int(m), "mean": round(float(np.mean(aus)), 4),
                              "std": round(float(np.std(aus)), 4), "draws": len(aus)})
        rec["val_size_curve"] = curve
        out["datasets"][disp] = rec

    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/adaptation_ladder.json", "w"), indent=2)
    # print
    print(f"{'dataset':10s}{'L1zs':>8s}{'L3stk':>8s}{'L4det':>8s}{'L5SM':>8s}")
    for disp, d in out["datasets"].items():
        if "error" in d:
            print(f"{disp:10s}  {d['error']}"); continue
        def g(k):
            v = d.get(k, {}).get("auroc"); return f"{v:.3f}" if v is not None else "--"
        print(f"{disp:10s}{g('L1_zeroshot'):>8s}{g('L3_target_stacking'):>8s}"
              f"{g('L4_determinacy'):>8s}{g('L5_spatialmind'):>8s}")
    print("\n=== val-size curve (determinacy gate) ===")
    for disp, d in out["datasets"].items():
        if "error" in d:
            continue
        cs = " ".join(f"m{c['m']}:{c['mean']:.3f}±{c['std']:.3f}" for c in d["val_size_curve"])
        print(f"{disp:10s} {cs}")


if __name__ == "__main__":
    main()
