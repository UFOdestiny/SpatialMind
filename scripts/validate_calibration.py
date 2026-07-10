"""
Leakage-free validation of the calibration design on saved run-36261619 predictions.

Protocol (simulates the production val->test fit/apply with NO leakage):
  * Stratified split of a split's predictions into FIT half and EVAL half.
  * Calibrator is fit on the FIT half's labels only, applied to the EVAL half.
  * Metrics computed on EVAL half. Averaged over several random splits.

SpatialMind -> StructuralCalibrator (structure-aware, can change ranking).
Baselines   -> StandardCalibrator  (monotonic Platt, ranking unchanged).
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics
from models.calibration import StandardCalibrator, StructuralCalibrator

RUN = "spatialmind/results/36261619"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS = ["spatialmind", "uhead", "saplma", "factoscope", "lookback_lens", "luh_light", "mlp"]
STRUCTURAL = {"spatialmind"}


def load(split_path):
    d = json.load(open(os.path.join(BASE, split_path, "evaluation_report.json")))
    p = d["predictions"]
    cps = [np.asarray(x.get("claim_probs") or [], dtype=np.float64) for x in p]
    s = np.array([x["trace_score"] for x in p], dtype=np.float64)
    y = np.array([x["trace_label"] for x in p], dtype=np.int64)
    return cps, s, y, d["overall_metrics"]


def paths(head):
    return {"ID": f"{RUN}/eval/{head}", "babi": f"{RUN}/eval_ood/babi/{head}",
            "spartqa": f"{RUN}/eval_ood/spartqa/{head}"}


def strat_split(y, seed):
    rng = np.random.RandomState(seed)
    fit_idx, eval_idx = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        half = len(idx) // 2
        fit_idx.append(idx[:half]); eval_idx.append(idx[half:])
    return np.concatenate(fit_idx), np.concatenate(eval_idx)


def calibrate(head, cps, s, y, seeds=(2026, 7, 13, 42, 99)):
    """Return averaged EVAL-half metrics under the fit/apply protocol."""
    accs = []
    for seed in seeds:
        fit, ev = strat_split(y, seed)
        if head in STRUCTURAL:
            cal = StructuralCalibrator(C=0.01).fit(s[fit], [cps[i] for i in fit], y[fit])
            sp = cal.transform(s[ev], [cps[i] for i in ev])
        else:
            cal = StandardCalibrator().fit(s[fit], y[fit])
            sp = cal.transform(s[ev])
        accs.append(compute_all_metrics(y[ev], sp))
    return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}


if __name__ == "__main__":
    for split in ["ID", "babi", "spartqa"]:
        print(f"\n{'='*92}\n{split}: calibrated (val->test proxy, fit/apply, avg 5 splits)\n{'='*92}")
        print(f"{'head':14s} {'AUROC_raw':>9s} {'AUROC_cal':>9s} {'PRAUC_cal':>9s} {'ECE_raw':>8s} {'ECE_cal':>8s}")
        rows = []
        for h in HEADS:
            cps, s, y, orig = load(paths(h)[split])
            m = calibrate(h, cps, s, y)
            rows.append((h, orig["roc_auc"], m["roc_auc"], m["pr_auc"], orig["ece"], m["ece"]))
        for h, ar, ac, pc, e0, e1 in sorted(rows, key=lambda r: -r[2]):
            tag = " <== SM (structural)" if h == "spatialmind" else ""
            print(f"{h:14s} {ar:9.4f} {ac:9.4f} {pc:9.4f} {e0:8.4f} {e1:8.4f}{tag}")
