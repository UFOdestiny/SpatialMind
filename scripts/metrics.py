"""
metrics.py - Binary evaluation metrics for hallucination detection.

Provides:
  - Accuracy, Precision, Recall, F1
  - ECE (Expected Calibration Error)
  - PR-AUC (Precision-Recall Area Under Curve)
  - ROC-AUC
  - Brier score, NLL
  - AURC (Area Under Risk-Coverage)
  - Risk@Coverage
  - Temperature scaling / threshold search helpers
"""

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


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary classification.

    Args:
        labels:     (N,) binary labels (0 or 1)
        probs:      (N,) predicted probabilities for the positive class

    Returns:
        ECE value (lower is better)
    """
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_binary_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Binary PR-AUC."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(auc(recall, precision))


def _to_probs(logits_or_probs: np.ndarray) -> np.ndarray:
    scores = logits_or_probs.astype(np.float64)
    if scores.min() < 0 or scores.max() > 1:
        return expit(scores)
    return scores


def _risk_coverage_curve(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and selective risk curve points.

    Coverage increases from low to high by keeping highest-confidence samples first.
    """
    labels_int = labels.astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)  # highest confidence first
    correct_sorted = (labels_int[order] == (probs[order] >= 0.5).astype(int)).astype(np.float64)

    n = len(labels_int)
    ks = np.arange(1, n + 1, dtype=np.int64)
    coverage = ks / float(n)
    cumulative_acc = np.cumsum(correct_sorted) / ks
    risk = 1.0 - cumulative_acc
    return coverage, risk


def compute_aurc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Area under selective risk-coverage curve."""
    if labels.size == 0:
        return float("nan")
    coverage, risk = _risk_coverage_curve(labels, probs)
    return float(np.trapz(risk, coverage))


def compute_risk_at_coverage(labels: np.ndarray, probs: np.ndarray, target_coverage: float) -> float:
    """Selective risk after retaining top-confidence target_coverage fraction."""
    if labels.size == 0:
        return float("nan")
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    n = len(labels)
    k = int(max(1, np.ceil(target_coverage * n)))
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)
    top_idx = order[:k]
    preds = (probs[top_idx] >= 0.5).astype(int)
    acc = (preds == labels[top_idx].astype(int)).mean()
    return float(1.0 - acc)


def search_best_threshold(
    labels: np.ndarray,
    logits_or_probs: np.ndarray,
    objective: str = "f1",
) -> dict:
    """Search threshold on [0, 1] maximizing selected objective."""
    probs = _to_probs(logits_or_probs)
    labels_int = labels.astype(int)

    if labels_int.size == 0:
        return {"threshold": 0.5, "objective": objective, "score": float("nan")}

    candidates = np.linspace(0.0, 1.0, 201)
    candidates = np.unique(np.concatenate([candidates, [0.5]]))

    best_thr = 0.5
    best_score = -np.inf
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        if objective == "f1":
            score = f1_score(labels_int, preds, zero_division=0)
        elif objective == "balanced_accuracy":
            rec1 = recall_score(labels_int, preds, pos_label=1, zero_division=0)
            rec0 = recall_score(labels_int, preds, pos_label=0, zero_division=0)
            score = 0.5 * (rec0 + rec1)
        elif objective == "accuracy":
            score = accuracy_score(labels_int, preds)
        else:
            score = f1_score(labels_int, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return {
        "threshold": best_thr,
        "objective": objective,
        "score": float(best_score),
    }


def fit_temperature_scaling(
    labels: np.ndarray,
    logits_or_probs: np.ndarray,
    min_temp: float = 0.25,
    max_temp: float = 5.0,
    num_temps: int = 61,
) -> dict:
    """Fit scalar temperature on validation set by minimizing NLL."""
    labels_int = labels.astype(int)
    scores = logits_or_probs.astype(np.float64)
    if labels_int.size == 0:
        return {"temperature": 1.0, "nll_before": float("nan"), "nll_after": float("nan")}

    # If caller passed probabilities, convert to logits for scaling.
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        eps = 1e-8
        p = np.clip(scores, eps, 1 - eps)
        scores = np.log(p / (1 - p))

    probs_before = expit(scores)
    nll_before = float(log_loss(labels_int, probs_before, labels=[0, 1]))

    temps = np.logspace(np.log10(min_temp), np.log10(max_temp), num=num_temps)
    best_t = 1.0
    best_nll = nll_before
    for t in temps:
        probs_t = expit(scores / t)
        nll_t = float(log_loss(labels_int, probs_t, labels=[0, 1]))
        if nll_t < best_nll:
            best_nll = nll_t
            best_t = float(t)

    return {
        "temperature": best_t,
        "nll_before": nll_before,
        "nll_after": best_nll,
    }


def compute_all_metrics(
    labels: np.ndarray,
    logits_or_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all binary evaluation metrics.

    Args:
        labels:          (N,) binary labels (1=correct/non-hallucination, 0=hallucination)
        logits_or_probs: (N,) raw logits or probabilities.
                         If any value > 1 or < 0, treated as logits and
                         passed through sigmoid.

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, pr_auc, ece
    """
    probs = _to_probs(logits_or_probs)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

    predictions = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    has_both_classes = len(np.unique(labels_int)) >= 2

    try:
        nll = float(log_loss(labels_int, probs, labels=[0, 1]))
    except ValueError:
        nll = float("nan")
    try:
        brier = float(brier_score_loss(labels_int, probs))
    except ValueError:
        brier = float("nan")

    return {
        "accuracy": float(accuracy_score(labels_int, predictions)),
        "precision": float(precision_score(labels_int, predictions, zero_division=0)),
        "recall": float(recall_score(labels_int, predictions, zero_division=0)),
        "f1": float(f1_score(labels_int, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_int, probs)) if has_both_classes else float("nan"),
        "pr_auc": compute_binary_pr_auc(labels_int, probs),
        "ece": compute_ece(labels_int, probs),
        "nll": nll,
        "brier": brier,
        "aurc": compute_aurc(labels_int, probs),
        "risk_at_80_cov": compute_risk_at_coverage(labels_int, probs, 0.8),
        "risk_at_90_cov": compute_risk_at_coverage(labels_int, probs, 0.9),
        "threshold": float(threshold),
    }


def compute_claim_metrics(
    claim_labels: np.ndarray,
    claim_scores: np.ndarray,
    claim_types: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute overall + per-type claim metrics and simple error propagation."""
    overall = compute_all_metrics(claim_labels, claim_scores, threshold=threshold)

    type_map = {
        0: "input",
        1: "reasoning",
        2: "conclusion",
    }
    per_type = {}
    for tid, tname in type_map.items():
        mask = claim_types == tid
        if mask.sum() == 0:
            continue
        per_type[tname] = compute_all_metrics(
            claim_labels[mask],
            claim_scores[mask],
            threshold=threshold,
        )
        per_type[tname]["count"] = int(mask.sum())

    # Approximate error propagation: hallucination rate in non-input claims.
    # Label convention is 1=correct, 0=hallucinated.
    non_input = claim_types != 0
    propagation_rate = (
        float((1.0 - claim_labels[non_input]).mean())
        if non_input.any()
        else 0.0
    )

    return {
        "overall": overall,
        "per_type": per_type,
        "error_propagation_rate": propagation_rate,
    }
