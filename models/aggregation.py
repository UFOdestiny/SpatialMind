"""
aggregation.py - Claim -> trace (sample-level) score aggregation.

SpatialMind reports SAMPLE-LEVEL (trace-level) uncertainty: every method emits a
per-claim correctness probability p_k in [0, 1], and we collapse the ordered
claim probabilities of one trace into a single reliability score s(r) in [0, 1].
The FINAL evaluation metrics (AUROC / PR-AUC / Acc / ECE) are computed from these
trace scores against the trace-level ground-truth label.

Design principle — fairness: the same aggregation is applied identically to
every method (supervised heads AND unsupervised baselines). This guarantees the
comparison measures the *quality of the claim scores*, not a method-specific
readout trick. The one exception is a head that emits its own learned trace
logit (SpatialMind's multi-task head); that logit is reported separately and
also compared under the shared aggregation for an apples-to-apples row.

Summaries (following the paper):
    conc : score of the final conclusion claim, p_K              (trust the answer)
    avg  : mean of the reasoning-claim probabilities             (trust the chain)
    min  : weakest reasoning claim, min_k p_k                    (error localization)
    mix  : validation-selected convex blend of {conc, avg, min}  (default, robust)

A trace with a single conclusion-only claim degenerates gracefully: avg == min == conc.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Claim type ids (mirror data/claims.py CLAIM_TYPE2ID).
REASONING_TYPE_ID = 1
CONCLUSION_TYPE_ID = 2

SUMMARY_KEYS = ("conc", "avg", "min", "mix")


def _split_reasoning_conclusion(
    probs: Sequence[float],
    claim_type_ids: Optional[Sequence[int]],
) -> tuple[np.ndarray, float]:
    """Return (reasoning_probs, conclusion_prob) for one trace.

    The conclusion is the claim marked with CONCLUSION_TYPE_ID; if none is marked
    we fall back to the last claim (Phase 1 always places the conclusion last).
    """
    p = np.asarray(list(probs), dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.float64), float("nan")

    if claim_type_ids is not None and len(claim_type_ids) == p.size:
        types = np.asarray(list(claim_type_ids), dtype=np.int64)
        conc_positions = np.where(types == CONCLUSION_TYPE_ID)[0]
        conc_idx = int(conc_positions[-1]) if conc_positions.size else p.size - 1
        reasoning_mask = np.ones(p.size, dtype=bool)
        reasoning_mask[conc_idx] = False
        reasoning = p[reasoning_mask]
        conclusion = float(p[conc_idx])
    else:
        conclusion = float(p[-1])
        reasoning = p[:-1]

    # Degenerate conclusion-only trace: reasoning summaries fall back to conclusion.
    if reasoning.size == 0:
        reasoning = np.asarray([conclusion], dtype=np.float64)
    return reasoning, conclusion


def trace_summaries(
    probs: Sequence[float],
    claim_type_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Compute the base summaries {conc, avg, min} for one trace.

    Scores are correctness probabilities (higher = more reliable), so `min`
    surfaces the least reliable reasoning step.
    """
    reasoning, conclusion = _split_reasoning_conclusion(probs, claim_type_ids)
    if np.isnan(conclusion):
        return {"conc": float("nan"), "avg": float("nan"), "min": float("nan")}
    return {
        "conc": float(conclusion),
        "avg": float(np.mean(reasoning)),
        "min": float(np.min(reasoning)),
    }


def aggregate_trace_score(
    probs: Sequence[float],
    claim_type_ids: Optional[Sequence[int]] = None,
    method: str = "mix",
    mix_weights: Optional[dict] = None,
) -> float:
    """Collapse one trace's per-claim probabilities into a single reliability score.

    Args:
        probs:          per-claim correctness probabilities for one trace.
        claim_type_ids: per-claim type ids (used to locate the conclusion claim).
        method:         one of {conc, avg, min, mix}.
        mix_weights:    convex weights over {conc, avg, min} for method="mix".
                        Defaults to the conclusion-only readout when unset.

    Returns:
        Trace reliability score in [0, 1] (higher = more reliable / less likely
        to be a hallucination).
    """
    summaries = trace_summaries(probs, claim_type_ids)
    if any(np.isnan(v) for v in summaries.values()):
        return float("nan")

    if method in ("conc", "avg", "min"):
        return summaries[method]

    if method == "mix":
        weights = mix_weights or {"conc": 1.0, "avg": 0.0, "min": 0.0}
        total = sum(weights.get(k, 0.0) for k in ("conc", "avg", "min"))
        if total <= 0:
            return summaries["conc"]
        score = sum(weights.get(k, 0.0) * summaries[k] for k in ("conc", "avg", "min"))
        return float(score / total)

    raise ValueError(f"Unknown aggregation method: {method!r}. Use one of {SUMMARY_KEYS}.")


def aggregate_batch(
    claim_probs_per_trace: List[Sequence[float]],
    claim_type_ids_per_trace: Optional[List[Sequence[int]]] = None,
    method: str = "mix",
    mix_weights: Optional[dict] = None,
) -> np.ndarray:
    """Aggregate a list of traces' claim probabilities into per-trace scores."""
    out = np.empty(len(claim_probs_per_trace), dtype=np.float64)
    for i, probs in enumerate(claim_probs_per_trace):
        types = None
        if claim_type_ids_per_trace is not None and i < len(claim_type_ids_per_trace):
            types = claim_type_ids_per_trace[i]
        out[i] = aggregate_trace_score(probs, types, method=method, mix_weights=mix_weights)
    return out


def select_aggregation(
    claim_probs_per_trace: List[Sequence[float]],
    trace_labels: Sequence[int],
    claim_type_ids_per_trace: Optional[List[Sequence[int]]] = None,
    objective: str = "auroc",
    mix_grid_step: float = 0.25,
) -> dict:
    """Select the best claim->trace aggregation on a validation split.

    We evaluate {conc, avg, min} plus a small grid of convex `mix` blends and
    keep the setting that maximizes `objective` (AUROC by default) against the
    trace-level labels. The chosen rule is then frozen and applied to the test
    split — for every method identically.

    Returns a dict: {method, mix_weights, score, objective}.
    """
    # Local import avoids a hard sklearn dependency at module import time.
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(list(trace_labels), dtype=np.int64)

    def _score(trace_scores: np.ndarray) -> float:
        valid = ~np.isnan(trace_scores)
        y = labels[valid]
        s = trace_scores[valid]
        if y.size == 0 or np.unique(y).size < 2:
            return float("-inf")
        if objective == "prauc":
            return float(average_precision_score(y, s))
        return float(roc_auc_score(y, s))

    candidates: List[dict] = [
        {"method": "conc", "mix_weights": None},
        {"method": "avg", "mix_weights": None},
        {"method": "min", "mix_weights": None},
    ]
    # Convex grid over (conc, avg, min).
    grid = np.arange(0.0, 1.0 + 1e-9, mix_grid_step)
    for wc in grid:
        for wa in grid:
            wm = 1.0 - wc - wa
            if wm < -1e-9:
                continue
            wm = max(0.0, wm)
            if wc + wa + wm <= 0:
                continue
            candidates.append(
                {"method": "mix", "mix_weights": {"conc": float(wc), "avg": float(wa), "min": float(wm)}}
            )

    best = {"method": "conc", "mix_weights": None, "score": float("-inf"), "objective": objective}
    for cand in candidates:
        scores = aggregate_batch(
            claim_probs_per_trace,
            claim_type_ids_per_trace,
            method=cand["method"],
            mix_weights=cand["mix_weights"],
        )
        val = _score(scores)
        if val > best["score"]:
            best = {**cand, "score": val, "objective": objective}
    return best
