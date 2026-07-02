#!/usr/bin/env python3
"""
length_confound.py - Diagnose the trace-length confound in claim-level UQ.

Motivation
----------
On StepGame (and similar spatial benchmarks) harder instances tend to (a) be
answered incorrectly more often and (b) produce longer reasoning traces with more
claims. Trace length is therefore correlated with the trace label. Any scorer that
aggregates per-claim scores can exploit this length signal *without doing real
spatial reasoning* — most visibly the `random` baseline, whose per-claim scores are
noise yet whose `min`/`avg` aggregation drifts with the number of claims, giving an
AUROC well above the 0.5 it "should" have.

This module quantifies that confound and reports length-controlled metrics so the
comparison reflects genuine claim-scoring skill, not a trace-length shortcut.

What it reports
---------------
1. Confound strength: correlation between #claims (and token length) and the trace
   label; mean #claims for correct vs hallucinated traces; the same vs k_hop.
2. The `random`-baseline leak: its overall AUROC vs the 0.5 ideal, and how much of
   it vanishes once length is controlled.
3. Length-stratified AUROC/PR-AUC: metrics computed within #claim bins and pooled
   (a length-balanced estimate), for every method whose predictions are saved.
4. Per-k_hop breakdown (from each report's per_difficulty_metrics), the principled
   difficulty axis.

Usage
-----
    python analysis/length_confound.py \
        --cache_dir spatialmind/cache/cached_features/example/StepGame/Llama-3.1-8B-Instruct \
        --split test \
        --results_root spatialmind/results/<job_id> \
        --scope ID                # ID | OOD:<dataset>
        [--out analysis/out]      # optional: save CSVs + a figure

Both --cache_dir and --results_root are optional; give what you have. The cache
provides the ground-truth length/label confound; the results provide per-method
predictions for length-stratified metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Cache-side confound (ground truth)
# --------------------------------------------------------------------------- #
def load_cache_traces(cache_dir: str, split: str) -> List[dict]:
    """Load lightweight per-trace records (no features) from a Phase-1 cache split."""
    import torch

    split_dir = Path(cache_dir) / split
    chunks = sorted(split_dir.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.pt under {split_dir}")
    out = []
    for cf in chunks:
        for s in torch.load(cf, map_location="cpu", weights_only=False):
            label = int(s.get("label", -1))
            if label not in (0, 1):
                continue
            am = s.get("attention_mask")
            n_tok = int((am > 0).sum().item()) if am is not None else 0
            out.append({
                "label": label,
                "n_claims": len(s.get("claims", []) or []),
                "n_tokens": n_tok,
                "k_hop": int(s.get("k_hop", 0)),
            })
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def confound_report(traces: List[dict]) -> dict:
    """Correlation of length signals with the trace label + group means."""
    label = np.array([t["label"] for t in traces])
    n_claims = np.array([t["n_claims"] for t in traces], dtype=float)
    n_tokens = np.array([t["n_tokens"] for t in traces], dtype=float)
    k_hop = np.array([t["k_hop"] for t in traces], dtype=float)
    correct = label == 1
    return {
        "n": int(label.size),
        "positive_rate": float(label.mean()),
        "corr_nclaims_label": _safe_corr(n_claims, label),
        "corr_ntokens_label": _safe_corr(n_tokens, label),
        "corr_khop_label": _safe_corr(k_hop, label),
        "corr_nclaims_khop": _safe_corr(n_claims, k_hop),
        "mean_nclaims_correct": float(n_claims[correct].mean()) if correct.any() else float("nan"),
        "mean_nclaims_halluc": float(n_claims[~correct].mean()) if (~correct).any() else float("nan"),
        "mean_ntokens_correct": float(n_tokens[correct].mean()) if correct.any() else float("nan"),
        "mean_ntokens_halluc": float(n_tokens[~correct].mean()) if (~correct).any() else float("nan"),
    }


def length_only_auroc(traces: List[dict]) -> dict:
    """AUROC/PR-AUC of using RAW LENGTH as the (reliability) score.

    Length correlates negatively with correctness, so we score reliability as
    -n_claims (fewer claims => more reliable). This is the ceiling a pure
    length-shortcut can reach — any method near it is largely exploiting length.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    label = np.array([t["label"] for t in traces])
    if np.unique(label).size < 2:
        return {"auroc_from_length": float("nan"), "prauc_from_length": float("nan")}
    for key in ("n_claims", "n_tokens"):
        pass
    n_claims = np.array([t["n_claims"] for t in traces], dtype=float)
    score = -n_claims  # higher = more reliable
    return {
        "auroc_from_nclaims": float(roc_auc_score(label, score)),
        "prauc_from_nclaims": float(average_precision_score(label, score)),
    }


# --------------------------------------------------------------------------- #
# Results-side: length-stratified metrics per method
# --------------------------------------------------------------------------- #
def discover_reports(results_root: str, scope: str) -> Dict[str, dict]:
    """Return {method_name: report} for the requested scope.

    scope: "ID" -> eval/*/evaluation_report.json (+ eval/baselines/*/...)
           "OOD:<dataset>" -> eval_ood/<dataset>/*/...
    """
    root = Path(results_root)
    reports: Dict[str, dict] = {}
    if scope == "ID":
        globs = ["eval/*/evaluation_report.json", "eval/baselines/*/evaluation_report.json"]
    elif scope.startswith("OOD:"):
        ds = scope.split(":", 1)[1]
        globs = [f"eval_ood/{ds}/*/evaluation_report.json",
                 f"eval_ood/{ds}/baselines/*/evaluation_report.json"]
    else:
        raise ValueError(f"scope must be 'ID' or 'OOD:<dataset>', got {scope!r}")

    for g in globs:
        for p in sorted(root.glob(g)):
            if p.parent.name == "baselines":
                continue  # the baselines/ dir itself holds only combined json
            try:
                rep = json.loads(Path(p).read_text())
            except json.JSONDecodeError:
                continue
            name = rep.get("head_type") or p.parent.name
            if rep.get("predictions") or rep.get("overall_metrics"):
                reports[name] = rep
    return reports


def _bin_edges(n_claims: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges over #claims (deduplicated)."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(n_claims, qs))
    if edges.size < 2:
        edges = np.array([n_claims.min(), n_claims.max() + 1])
    return edges


def stratified_auroc(labels: np.ndarray, scores: np.ndarray, n_claims: np.ndarray,
                     n_bins: int = 4) -> dict:
    """Length-controlled AUROC: compute within #claim bins, then average (weighted).

    A method that only exploits length collapses toward 0.5 within a bin (little
    length variation left), so the gap between overall and stratified AUROC
    measures how much of its skill is a length shortcut.
    """
    from sklearn.metrics import roc_auc_score

    edges = _bin_edges(n_claims, n_bins)
    per_bin = []
    weights = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (n_claims >= lo) & (n_claims < hi) if i < len(edges) - 2 else (n_claims >= lo) & (n_claims <= hi)
        if m.sum() < 10 or np.unique(labels[m]).size < 2:
            continue
        per_bin.append(float(roc_auc_score(labels[m], scores[m])))
        weights.append(int(m.sum()))
    if not per_bin:
        return {"stratified_auroc": float("nan"), "n_bins_used": 0}
    w = np.array(weights, dtype=float)
    return {
        "stratified_auroc": float(np.average(per_bin, weights=w)),
        "n_bins_used": len(per_bin),
        "per_bin_auroc": [round(x, 4) for x in per_bin],
        "per_bin_n": weights,
    }


def method_length_analysis(report: dict, cache_by_key: Optional[dict], n_bins: int = 4) -> Optional[dict]:
    """Overall vs length-stratified AUROC for one method, using its saved predictions."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    preds = report.get("predictions")
    if not preds:
        return None
    labels = np.array([int(p["trace_label"]) for p in preds])
    scores = np.array([float(p["trace_score"]) for p in preds])
    if np.unique(labels).size < 2:
        return None
    # #claims per prediction: prefer the saved claim list length; fall back to cache.
    n_claims = np.array([len(p.get("claim_probs", []) or []) for p in preds], dtype=float)
    if (n_claims == 0).all() and cache_by_key is not None:
        return None
    overall = {
        "overall_auroc": float(roc_auc_score(labels, scores)),
        "overall_prauc": float(average_precision_score(labels, scores)),
        "corr_score_nclaims": _safe_corr(scores, n_claims),
    }
    overall.update(stratified_auroc(labels, scores, n_claims, n_bins=n_bins))
    overall["auroc_minus_stratified"] = (
        overall["overall_auroc"] - overall["stratified_auroc"]
        if not np.isnan(overall["stratified_auroc"]) else float("nan")
    )
    return overall


def per_khop_table(reports: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """{method: {k_hop: roc_auc}} from each report's per_difficulty_metrics."""
    out: Dict[str, Dict[str, float]] = {}
    for name, rep in reports.items():
        pd_metrics = rep.get("per_difficulty_metrics", {}) or {}
        out[name] = {str(k): float(v.get("roc_auc", float("nan")))
                     for k, v in pd_metrics.items()}
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def print_report(cache_dir, split, results_root, scope, n_bins, out_dir=None):
    print("=" * 78)
    print(f"  Trace-length confound diagnostic | scope={scope}")
    print("=" * 78)

    traces = None
    if cache_dir:
        traces = load_cache_traces(cache_dir, split)
        cr = confound_report(traces)
        lo = length_only_auroc(traces)
        print(f"\n[1] Confound strength (n={cr['n']}, positive/correct rate={cr['positive_rate']:.3f})")
        print(f"    corr(#claims, correct)  = {_fmt(cr['corr_nclaims_label'])}   "
              f"corr(#tokens, correct) = {_fmt(cr['corr_ntokens_label'])}")
        print(f"    corr(k_hop,   correct)  = {_fmt(cr['corr_khop_label'])}   "
              f"corr(#claims, k_hop)   = {_fmt(cr['corr_nclaims_khop'])}")
        print(f"    mean #claims: correct={_fmt(cr['mean_nclaims_correct'],2)}  "
              f"hallucination={_fmt(cr['mean_nclaims_halluc'],2)}")
        print(f"    mean #tokens: correct={_fmt(cr['mean_ntokens_correct'],1)}  "
              f"hallucination={_fmt(cr['mean_ntokens_halluc'],1)}")
        print(f"\n[2] Length-shortcut ceiling (score = -#claims):")
        print(f"    AUROC_from_#claims = {_fmt(lo['auroc_from_nclaims'])}   "
              f"PR-AUC_from_#claims = {_fmt(lo['prauc_from_nclaims'])}")
        print("    (A method near this AUROC is largely exploiting trace length, not reasoning.)")

    if results_root:
        reports = discover_reports(results_root, scope)
        if not reports:
            print(f"\n[3] No reports with predictions found under {results_root} for scope={scope}.")
        else:
            print(f"\n[3] Overall vs length-stratified AUROC ({n_bins} #claim bins):")
            print(f"    {'method':16s} {'overall':>8} {'stratif.':>9} {'Δ(short)':>9} "
                  f"{'corr(s,#cl)':>11} {'bins':>5}")
            rows = []
            for name in sorted(reports):
                a = method_length_analysis(reports[name], None, n_bins=n_bins)
                if a is None:
                    continue
                rows.append((name, a))
                print(f"    {name:16s} {_fmt(a['overall_auroc']):>8} "
                      f"{_fmt(a['stratified_auroc']):>9} {_fmt(a['auroc_minus_stratified']):>9} "
                      f"{_fmt(a['corr_score_nclaims']):>11} {a['n_bins_used']:>5}")
            print("    Δ(short) = overall − stratified AUROC. Large Δ ⇒ method leans on length.")
            print("    A well-behaved `random` baseline should have overall AUROC ≈ 0.5 and Δ ≈ 0.")

            khop = per_khop_table(reports)
            all_k = sorted({k for d in khop.values() for k in d}, key=lambda x: int(x) if x.isdigit() else 999)
            if all_k:
                print(f"\n[4] Per-k_hop ROC-AUC (principled difficulty axis):")
                header = "    {:16s}".format("method") + "".join(f"{('k'+k):>7}" for k in all_k)
                print(header)
                for name in sorted(khop):
                    line = "    {:16s}".format(name) + "".join(f"{_fmt(khop[name].get(k), 3):>7}" for k in all_k)
                    print(line)

            if out_dir:
                _save_outputs(out_dir, scope, traces, reports, rows if 'rows' in dir() else [], khop)

    print("\n" + "=" * 78)


def _save_outputs(out_dir, scope, traces, reports, rows, khop):
    import csv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = scope.replace(":", "_")
    if rows:
        with open(out / f"length_stratified_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "overall_auroc", "stratified_auroc", "delta_shortcut", "corr_score_nclaims"])
            for name, a in rows:
                w.writerow([name, a["overall_auroc"], a["stratified_auroc"],
                            a["auroc_minus_stratified"], a["corr_score_nclaims"]])
        print(f"    saved: {out / f'length_stratified_{tag}.csv'}")
    # Optional figure: overall vs stratified AUROC bars.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rows:
            names = [r[0] for r in rows]
            ov = [r[1]["overall_auroc"] for r in rows]
            st = [r[1]["stratified_auroc"] for r in rows]
            x = np.arange(len(names)); w = 0.4
            fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
            ax.bar(x - w / 2, ov, w, label="overall AUROC")
            ax.bar(x + w / 2, st, w, label="length-stratified AUROC")
            ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance (0.5)")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("AUROC"); ax.set_title(f"Length confound — {scope}")
            ax.legend(); fig.tight_layout()
            fig.savefig(out / f"length_confound_{tag}.png", dpi=200, bbox_inches="tight")
            print(f"    saved: {out / f'length_confound_{tag}.png'}")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Trace-length confound diagnostic")
    ap.add_argument("--cache_dir", default=None, help="Phase-1 cache dir (…/dataset/model)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results_root", default=None, help="results/<job_id> dir")
    ap.add_argument("--scope", default="ID", help="ID | OOD:<dataset>")
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional dir to save CSV/figure")
    args = ap.parse_args()
    if not args.cache_dir and not args.results_root:
        ap.error("provide at least one of --cache_dir / --results_root")
    print_report(args.cache_dir, args.split, args.results_root, args.scope, args.n_bins, args.out)


if __name__ == "__main__":
    main()
