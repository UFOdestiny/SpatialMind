#!/usr/bin/env python3
"""DARC ablation table (Llama), AUROC / class-balanced macro-Brier.

Variants (all reuse scripts/fusion.py internals; validation-only fitting):
  * Constraint (pure)          -- the spatialmind constraint scorer alone
  * MLP (pure)                 -- the mlp probe alone
  * Fixed 0.5 average          -- unweighted mean of constraint + mlp
  * Oracle per-dataset pure    -- pick constraint OR neural-seq by TEST AUROC
  * Stacking (scores only)     -- gate="scores"
  * + symbolizability gate     -- gate="symb"
  * + determinacy gate         -- gate="determinacy"
  * SpatialMind (val-selected) -- fusion.py's selected (gate,l2)+calibrator

Fixed-gate rows use a reference L2 (=20) and no post-hoc calibrator, matching the
"reference regularization" caption. The final SpatialMind row selects everything
on validation (reads fusion/<tag>/evaluation_report.json written by fusion.py).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402

BACKBONE = os.environ.get("ABLATION_BACKBONE", "llama")
_MODELS = {"llama": "Llama-3.1-8B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3",
           "gemma": "Gemma-2-9B-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
R = f"spatialmind/results/constraint_guided_{BACKBONE}"
CACHE_ROOT = "spatialmind/cache/cached_features"
SUB = f"constraint_guided_{BACKBONE}"
MODEL = _MODELS[BACKBONE]
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
REF_L2 = 20.0


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def prep(tag, cname):
    cache = (f"{CACHE_ROOT}/{SUB}/StepGame/{MODEL}" if tag == "id"
             else f"{CACHE_ROOT}/{SUB}/{tag}/{MODEL}")
    sv = F.collect_signals(R, tag, cname, "validation", SUB)
    st = F.collect_signals(R, tag, cname, "test", SUB)
    common = sorted(set(sv) & set(st))
    sv = {k: sv[k] for k in common}; st = {k: st[k] for k in common}
    tfv_full = F.load_trace_features(cache, "validation")
    tft_full = F.load_trace_features(cache, "test")
    Sv, yv, _, tfv = F.align(sv, tfv_full)
    St, yt, _, tft = F.align(st, tft_full)
    return Sv, yv, tfv, St, yt, tft


def orient(Sv, St, yv):
    val_au = {k: F.auroc(yv, v) for k, v in Sv.items()}
    ov, ot = {}, {}
    for k in sorted(Sv):
        a = val_au[k]
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        sv, st = Sv[k], St[k]
        if a < 0.5:
            sv, st = 1 - sv, 1 - st
        ov[k], ot[k] = sv, st
    return ov, ot


def fit_gate(Sv, yv, tfv, St, tft, gate, l2):
    Xv, _ = F.build_matrix(Sv, tfv, gate)
    Xt, _ = F.build_matrix(St, tft, gate)
    fit = F.fit_logreg(Xv, yv, l2=l2)
    return F.apply_logreg(fit, Xt)


def cell(y, s):
    return f"{F.auroc(y, s):.3f}/{macro_brier(y, s):.3f}"


def main():
    rows = {k: [] for k in ["cons", "mlp", "avg", "oracle", "scores", "symb", "det", "sm"]}
    for tag, disp, cname in DATASETS:
        Sv, yv, tfv, St, yt, tft = prep(tag, cname)
        # pure signals (from oriented? no -- pure = raw scorer, higher=more reliable already)
        cons = St.get("spatialmind"); mlp = St.get("mlp")
        rows["cons"].append(cell(yt, cons) if cons is not None else "--")
        rows["mlp"].append(cell(yt, mlp) if mlp is not None else "--")
        # fixed average of the two primary partners
        if cons is not None and mlp is not None:
            rows["avg"].append(cell(yt, 0.5 * cons + 0.5 * mlp))
        else:
            rows["avg"].append("--")
        # oracle per-dataset pure pick: best of constraint vs neural-seq by TEST auroc
        cand = {k: St[k] for k in ("spatialmind", "spatialmind_neural") if k in St}
        if cand:
            best = max(cand, key=lambda k: F.auroc(yt, St[k]))
            rows["oracle"].append(cell(yt, St[best]))
        else:
            rows["oracle"].append("--")
        # stacked variants on oriented signals
        Ov, Ot = orient(Sv, St, yv)
        if len(Ov) >= 1 and len(np.unique(yv)) > 1:
            for key, gate in [("scores", "scores"), ("symb", "symb"), ("det", "determinacy")]:
                s = fit_gate(Ov, yv, tfv, Ot, tft, gate, REF_L2)
                rows[key].append(cell(yt, s))
        else:
            for key in ("scores", "symb", "det"):
                rows[key].append("--")
        # final val-selected SpatialMind (read from fusion report)
        fp = f"{R}/fusion/{tag}/evaluation_report.json"
        if os.path.exists(fp):
            pr = json.load(open(fp))["predictions"]
            fy = np.array([x["trace_label"] for x in pr], float)
            fs = np.array([x["trace_score"] for x in pr], float)
            rows["sm"].append(cell(fy, fs))
        else:
            rows["sm"].append("--")

    labels = [("cons", "Constraint scorer (pure)"), ("mlp", "MLP probe (pure)"),
              ("avg", "Fixed 0.5 average"), ("oracle", "Oracle per-dataset pure pick"),
              ("scores", "Stacking (scores only)"), ("symb", "+ symbolizability gate"),
              ("det", "+ determinacy gate"), ("sm", "SpatialMind (val-selected)")]
    print(f"{'variant':30s}" + "".join(f"{d:>16s}" for _, d, _ in DATASETS))
    for key, lab in labels:
        print(f"{lab:30s}" + "".join(c.rjust(16) for c in rows[key]))

    # ---- LaTeX emitter with best/2nd marks (excluding the oracle row) ----
    def parse(c):
        if c == "--":
            return None, None
        a, b = c.split("/")
        return float(a), float(b)

    vals = {k: [parse(c) for c in rows[k]] for k in rows}
    deploy = ["cons", "mlp", "avg", "scores", "symb", "det", "sm"]  # exclude oracle
    ncol = len(DATASETS)
    au_rank = [[] for _ in range(ncol)]
    br_rank = [[] for _ in range(ncol)]
    for j in range(ncol):
        au = [(vals[k][j][0], k) for k in deploy if vals[k][j][0] is not None]
        br = [(vals[k][j][1], k) for k in deploy if vals[k][j][1] is not None]
        au_rank[j] = [k for _, k in sorted(au, key=lambda x: -x[0])]
        br_rank[j] = [k for _, k in sorted(br, key=lambda x: x[0])]

    def fmt(v, best2nd):
        b, s = best2nd
        t = f"{v:.3f}"
        if v == b:
            return f"\\textbf{{{t}}}"
        if v == s:
            return f"\\underline{{{t}}}"
        return t

    print("\n% ---- LaTeX rows ----")
    for key, lab in labels:
        cells = []
        for j in range(ncol):
            av, bv = vals[key][j]
            if av is None:
                cells += ["--", "--"]; continue
            if key == "oracle":
                cells += [f"{av:.3f}", f"{bv:.3f}"]
            else:
                a_best = vals[au_rank[j][0]][j][0]
                a_2nd = vals[au_rank[j][1]][j][0] if len(au_rank[j]) > 1 else None
                b_best = vals[br_rank[j][0]][j][1]
                b_2nd = vals[br_rank[j][1]][j][1] if len(br_rank[j]) > 1 else None
                cells += [fmt(av, (a_best, a_2nd)), fmt(bv, (b_best, b_2nd))]
        prefix = "\\rowcolor{gray!15}\n\\textbf{\\m}" if key == "sm" else lab
        if key == "oracle":
            prefix = lab + "$^{\\dagger}$"
        print(f"{prefix} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
