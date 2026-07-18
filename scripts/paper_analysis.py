#!/usr/bin/env python3
"""Offline paper analyses for SpatialMind v10 (all reproducible, single-seed).

Reads only saved base-head predictions + cached trace features. Produces:
  (1) fusion mode ablation (scores_only/symb/determinacy) per dataset;
  (2) same-protocol adaptation baselines: fixed-0.5 average, oracle per-dataset
      pick, plain scores-only stacking (isolates the symbolizability gate);
  (3) pooled symbolizability -> constraint-minus-neural gain curve (monotone);
  (4) determinacy table (parse_rate / unknown_rate vs constraint AUROC);
  (5) paired bootstrap 95% CIs for FUSION vs best-pure per dataset;
  (6) per-hop AUROC on StepGame for the depth figure.
Writes JSON to spatialmind/results/<ns>/fusion/paper_analysis.json.
"""
from __future__ import annotations
import json, os, glob, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics  # noqa: E402
from spatial_constraints.analysis import TRACE_FEATURE_NAMES  # noqa: E402
# gated_fusion was renamed to fusion_pairwise in the v2 refactor; the helper
# functions live there now.
from scripts.fusion_pairwise import (  # noqa: E402
    load_predictions, load_trace_features, build_fusion_features,
    fit_logreg, apply_logreg, resolve_report, _select_config,
    PARSE_RATE_IDX, UNKNOWN_RATE_IDX,
)

R = "spatialmind/results/constraint_guided_v11_llama"
CACHE = "spatialmind/cache/cached_features/constraint_guided_v11_llama"
MODEL = "Llama-3.1-8B-Instruct"

R = "spatialmind/results/constraint_guided_v11_mistral"
CACHE = "spatialmind/cache/cached_features/constraint_guided_v11_mistral"
MODEL = "Mistral-7B-Instruct-v0.3"

CON = "constraint_no_conflict"
NEU = "mlp"
DATASETS = [("id", "StepGame"), ("spartqa", "spartqa"), ("babi", "babi"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS1", "SpaRP_PS1"), ("SpaRP_PS3", "SpaRP_PS3")]

rng = np.random.RandomState(0)


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return compute_all_metrics(y, s)["roc_auc"]


def load_all(tag, cname):
    cache_dir = f"{CACHE}/{cname}/{MODEL}"
    cv = load_predictions(resolve_report(R, tag, CON, "validation", cname))
    nv = load_predictions(resolve_report(R, tag, NEU, "validation", cname))
    ct = load_predictions(resolve_report(R, tag, CON, "test", cname))
    nt = load_predictions(resolve_report(R, tag, NEU, "test", cname))
    if any(x is None for x in (cv, nv, ct, nt)):
        return None
    idv, yv, scv = cv
    _, _, snv = nv
    idt, yt, sct = ct
    _, _, snt = nt
    tfv = load_trace_features(cache_dir, "validation")[idv]
    tft = load_trace_features(cache_dir, "test")[idt]
    return dict(idv=idv, yv=yv, scv=scv, snv=snv, idt=idt, yt=yt,
                sct=sct, snt=snt, tfv=tfv, tft=tft)


def fit_apply(d, mode, l2):
    Xv = build_fusion_features(d["scv"], d["snv"], d["tfv"], mode)
    Xt = build_fusion_features(d["sct"], d["snt"], d["tft"], mode)
    fit = fit_logreg(Xv, d["yv"], l2=l2)
    return apply_logreg(fit, Xt)


def main():
    out = {"datasets": {}, "meta": {"con": CON, "neu": NEU}}

    # (3) pooled symbolizability -> gain curve accumulators
    pooled = []  # (parse_rate, y, s_con, s_neu)

    for tag, cname in DATASETS:
        d = load_all(tag, cname)
        if d is None:
            continue
        yt = d["yt"]
        a_con = auroc(yt, d["sct"])
        a_neu = auroc(yt, d["snt"])

        # (1) fusion mode ablation (each mode's own best-l2 chosen on val split)
        modes = {}
        for mode in ["scores_only", "symb", "determinacy"]:
            grid = [0.5, 1.0, 2.0, 5.0, 10.0]
            # select l2 on validation-internal even/odd split
            best = None
            n = len(d["yv"])
            fm = (np.arange(n) % 2 == 0)
            for l2 in grid:
                f = fit_logreg(build_fusion_features(d["scv"][fm], d["snv"][fm], d["tfv"][fm], mode),
                               d["yv"][fm], l2=l2)
                ss = apply_logreg(f, build_fusion_features(
                    d["scv"][~fm], d["snv"][~fm], d["tfv"][~fm], mode))
                if len(np.unique(d["yv"][~fm])) < 2:
                    continue
                a = auroc(d["yv"][~fm], ss)
                if best is None or a > best[0]:
                    best = (a, l2)
            l2 = best[1] if best else 1.0
            modes[mode] = round(auroc(yt, fit_apply(d, mode, l2)), 4)

        # (2) same-protocol adaptation baselines
        avg05 = auroc(yt, 0.5 * d["sct"] + 0.5 * d["snt"])
        oracle_pure = max(a_con, a_neu)  # oracle per-dataset pick of best pure

        # auto (paper headline) — matches gated_fusion --mode auto
        mode_a, l2_a = _select_config(d["scv"], d["snv"], d["tfv"], d["yv"], 1.0)
        auto = auroc(yt, fit_apply(d, mode_a, l2_a))

        # determinacy stats
        pr = np.nanmean(d["tft"][:, PARSE_RATE_IDX])
        unk = np.nanmean(d["tft"][:, UNKNOWN_RATE_IDX])

        out["datasets"][cname] = {
            "n_test": int(len(yt)),
            "con": round(a_con, 4), "neu": round(a_neu, 4),
            "avg05": round(avg05, 4),
            "oracle_pure": round(oracle_pure, 4),
            "scores_only": modes["scores_only"],
            "symb": modes["symb"],
            "determinacy": modes["determinacy"],
            "fusion_auto": round(auto, 4),
            "auto_mode": mode_a, "auto_l2": l2_a,
            "parse_rate": round(float(pr), 3),
            "unknown_rate": round(float(unk), 3),
        }

        # (5) paired bootstrap CI: fusion_auto vs best pure
        fused = fit_apply(d, mode_a, l2_a)
        best_pure = d["sct"] if a_con >= a_neu else d["snt"]
        diffs = []
        fvals = []
        for _ in range(1000):
            idx = rng.randint(0, len(yt), len(yt))
            if len(np.unique(yt[idx])) < 2:
                continue
            af = auroc(yt[idx], fused[idx])
            ab = auroc(yt[idx], best_pure[idx])
            diffs.append(af - ab)
            fvals.append(af)
        out["datasets"][cname]["fusion_ci"] = [round(np.percentile(fvals, 2.5), 4),
                                               round(np.percentile(fvals, 97.5), 4)]
        out["datasets"][cname]["delta_best_ci"] = [round(np.percentile(diffs, 2.5), 4),
                                                    round(np.percentile(diffs, 97.5), 4)]

        # pooled curve contributions (test)
        for i in range(len(yt)):
            pooled.append((d["tft"][i, PARSE_RATE_IDX], yt[i], d["sct"][i], d["snt"][i]))

    # (3) symbolizability -> gain curve: bin by parse_rate, con-minus-neu univariate AUROC gap
    P = np.array(pooled, dtype=float)
    pr_all, y_all, sc_all, sn_all = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.99), (0.99, 1.001)]
    curve = []
    for lo, hi in bins:
        m = (pr_all >= lo) & (pr_all < hi)
        if m.sum() < 30 or len(np.unique(y_all[m])) < 2:
            curve.append({"bin": f"({lo},{hi})", "n": int(m.sum()), "gap": None})
            continue
        gap = auroc(y_all[m], sc_all[m]) - auroc(y_all[m], sn_all[m])
        curve.append({"bin": f"({lo:.1f},{hi:.2f})", "n": int(m.sum()),
                      "con_auroc": round(auroc(y_all[m], sc_all[m]), 4),
                      "neu_auroc": round(auroc(y_all[m], sn_all[m]), 4),
                      "gap": round(gap, 4)})
    out["symbolizability_curve"] = curve

    # (6) per-hop AUROC on StepGame (constraint, neural, fusion)
    id_report = json.load(open(f"{R}/eval/{CON}/evaluation_report.json"))
    # difficulty stored per-sample? check predictions
    d = load_all("id", "StepGame")
    fused = fit_apply(d, *_select_config(d["scv"], d["snv"], d["tfv"], d["yv"], 1.0))
    # per_difficulty via difficulty field from constraint report predictions
    preds = id_report["predictions"]
    khop = {p["sample_id"]: p.get("difficulty") for p in preds}
    # difficulty may be under 'k_hop' key
    if all(v is None for v in khop.values()):
        khop = {p["sample_id"]: p.get("k_hop") for p in preds}
    hop_rows = {}
    for i, sid in enumerate(d["idt"]):
        h = khop.get(int(sid))
        if h is None:
            continue
        hop_rows.setdefault(int(h), {"y": [], "con": [], "neu": [], "fus": []})
        hop_rows[int(h)]["y"].append(d["yt"][i])
        hop_rows[int(h)]["con"].append(d["sct"][i])
        hop_rows[int(h)]["neu"].append(d["snt"][i])
        hop_rows[int(h)]["fus"].append(fused[i])
    per_hop = []
    for h in sorted(hop_rows):
        r = hop_rows[h]
        y = np.array(r["y"])
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue
        per_hop.append({"hop": h, "n": len(y),
                        "con": round(auroc(y, np.array(r["con"])), 4),
                        "neu": round(auroc(y, np.array(r["neu"])), 4),
                        "fusion": round(auroc(y, np.array(r["fus"])), 4)})
    out["per_hop_stepgame"] = per_hop

    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/paper_analysis.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
