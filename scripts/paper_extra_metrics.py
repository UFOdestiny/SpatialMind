#!/usr/bin/env python3
"""Comprehensive calibration/selective-prediction metrics (P7 in review).

The reviewer asks for standard Brier, ECE, NLL, and AURC alongside AUROC and the
class-balanced Brier, all under the SAME calibrator protocol. We report, per
dataset, SpatialMind vs the strongest external baseline (the best non-fusion,
non-Random row by AUROC), on:
    AUROC (up), macro-Brier (down), standard Brier (down), ECE (down),
    NLL (down), AURC (down).

compute_all_metrics already returns ece/nll/brier/aurc/roc_auc; we add the
class-balanced macro-Brier the paper headlines. Pure post-processing.

Writes spatialmind/results/<ns>/fusion/extra_metrics.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics  # noqa: E402
import scripts.fusion as F  # noqa: E402

BACKBONE = os.environ.get("XM_BACKBONE", "mistral")
SUB = f"constraint_guided_v11_{BACKBONE}"
R = f"spatialmind/results/constraint_guided_v11_{BACKBONE}"
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]


def macro_brier(y, s):
    return F._macro_brier(y, s)


def metric_row(y, s):
    m = compute_all_metrics(np.asarray(y, int), np.asarray(s, float))
    return {"auroc": round(m["roc_auc"], 4), "macro_brier": round(macro_brier(y, s), 4),
            "brier": round(m["brier"], 4), "ece": round(m["ece"], 4),
            "nll": round(m["nll"], 4), "aurc": round(m["aurc"], 4)}


def main():
    out = {"backbone": BACKBONE, "datasets": {}}
    for tag, disp, cname in DATASETS:
        # SpatialMind
        fp = f"{R}/fusion/{tag}/evaluation_report.json"
        if not os.path.exists(fp):
            out["datasets"][disp] = {"error": "no fusion"}; continue
        pr = json.load(open(fp))["predictions"]
        fy = np.array([x["trace_label"] for x in pr], int)
        fs = np.array([x["trace_score"] for x in pr], float)
        # strongest external baseline by test AUROC (from the saved signals)
        st = F.collect_signals(R, tag, cname, "test", SUB)
        best_name, best_au, best = None, -1, None
        for k, (ids, lab, sc) in st.items():
            if k in ("random",):
                continue
            a = F.auroc(lab, sc)
            if not np.isnan(a) and a > best_au:
                best_au, best_name, best = a, k, (lab, sc)
        rec = {"spatialmind": metric_row(fy, fs)}
        if best is not None:
            rec["best_baseline"] = {"name": best_name, **metric_row(best[0], best[1])}
        out["datasets"][disp] = rec

    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/extra_metrics.json", "w"), indent=2)
    hdr = ["auroc", "macro_brier", "brier", "ece", "nll", "aurc"]
    print(f"{'dataset':10s}{'method':16s}" + "".join(f"{h:>12s}" for h in hdr))
    for disp, d in out["datasets"].items():
        if "error" in d:
            print(f"{disp:10s} {d['error']}"); continue
        for key, lab in [("spatialmind", "SpatialMind"),
                         ("best_baseline", d.get("best_baseline", {}).get("name", "--"))]:
            r = d.get(key)
            if not r:
                continue
            print(f"{disp:10s}{lab:16s}" + "".join(f"{r[h]:>12.4f}" for h in hdr))


if __name__ == "__main__":
    main()
