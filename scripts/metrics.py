"""
metrics.py - Evaluation metrics for hallucination detection.

The reported metrics are SAMPLE-LEVEL (trace-level): one score s_i in [0, 1] per
trace (higher = more reliable) versus the trace-level correctness label y_i.

    AUROC  : ranking reliable vs unreliable traces (threshold-free)
    PR-AUC : precision-recall AUC (informative under class skew)
    Acc    : accuracy of the binary decision at a fixed threshold (0.5)
    ECE    : expected calibration error (confidence vs empirical accuracy)
    plus F1, precision, recall, Brier, NLL, AURC, risk@coverage.

`compute_all_metrics` accepts logits OR probabilities; values outside [0, 1] are
passed through a sigmoid. Claim-level helpers are provided for diagnostics only.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
        return expit(x)
    return x


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for a binary decision at threshold 0.5."""
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        c = in_bin.sum()
        if c == 0:
            continue
        ece += (c / total) * abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def compute_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(auc(recall, precision))


def _risk_coverage(labels: np.ndarray, probs: np.ndarray):
    labels_int = labels.astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)
    correct_sorted = (labels_int[order] == (probs[order] >= 0.5).astype(int)).astype(np.float64)
    n = len(labels_int)
    ks = np.arange(1, n + 1, dtype=np.int64)
    coverage = ks / float(n)
    risk = 1.0 - np.cumsum(correct_sorted) / ks
    return coverage, risk


def compute_aurc(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    coverage, risk = _risk_coverage(labels, probs)
    return float(np.trapz(risk, coverage))


def compute_risk_at_coverage(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    if labels.size == 0:
        return float("nan")
    target = float(np.clip(target, 0.0, 1.0))
    n = len(labels)
    k = int(max(1, np.ceil(target * n)))
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)[:k]
    preds = (probs[order] >= 0.5).astype(int)
    return float(1.0 - (preds == labels[order].astype(int)).mean())


def compute_all_metrics(labels: np.ndarray, logits_or_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """All sample-level metrics. `labels`: 1=correct/non-hallucination, 0=hallucination."""
    probs = np.clip(_to_probs(logits_or_probs), 1e-8, 1.0 - 1e-8)
    labels_int = np.asarray(labels).astype(int)
    preds = (probs >= threshold).astype(int)
    both = len(np.unique(labels_int)) >= 2
    try:
        nll = float(log_loss(labels_int, probs, labels=[0, 1]))
    except ValueError:
        nll = float("nan")
    try:
        brier = float(brier_score_loss(labels_int, probs))
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": float(accuracy_score(labels_int, preds)),
        "precision": float(precision_score(labels_int, preds, zero_division=0)),
        "recall": float(recall_score(labels_int, preds, zero_division=0)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_int, probs)) if both else float("nan"),
        "pr_auc": compute_pr_auc(labels_int, probs),
        "ece": compute_ece(labels_int, probs),
        "nll": nll,
        "brier": brier,
        "aurc": compute_aurc(labels_int, probs),
        "risk_at_80_cov": compute_risk_at_coverage(labels_int, probs, 0.8),
        "risk_at_90_cov": compute_risk_at_coverage(labels_int, probs, 0.9),
        "threshold": float(threshold),
    }


def compute_claim_metrics(claim_labels: np.ndarray, claim_scores: np.ndarray,
                          claim_types: np.ndarray, threshold: float = 0.5) -> dict:
    """Claim-level diagnostics (not the headline metric): overall + per-type."""
    overall = compute_all_metrics(claim_labels, claim_scores, threshold)
    type_map = {1: "reasoning", 2: "conclusion"}
    per_type = {}
    for tid, tname in type_map.items():
        mask = claim_types == tid
        if mask.sum() == 0:
            continue
        m = compute_all_metrics(claim_labels[mask], claim_scores[mask], threshold)
        m["count"] = int(mask.sum())
        per_type[tname] = m
    return {"overall": overall, "per_type": per_type}
