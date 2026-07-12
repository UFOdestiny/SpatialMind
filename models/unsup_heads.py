"""
unsup_heads.py - Unsupervised (training-free) uncertainty estimators.

These baselines derive an uncertainty score directly from cached generation
statistics (top-k token log-probs) with no learned parameters. Each estimator
produces a per-claim uncertainty score; for the SAMPLE-LEVEL evaluation we
convert uncertainty -> confidence and aggregate the per-claim confidences to a
single trace score using the SAME claim->trace protocol as the supervised heads
(models/aggregation.py). This keeps the comparison fair.

Convention: `estimate*` returns UNCERTAINTY (higher = more likely hallucination).
The evaluator turns this into a confidence/correctness score via 1 - normalized
uncertainty, so that higher = more reliable, matching the supervised heads.

Estimators:
    random        : reference lower bound
    mcp           : 1 - mean top-1 token probability
    perplexity    : mean negative log-likelihood (exp for perplexity scale)
    token_entropy : mean top-k predictive entropy
    ccp           : cumulative confidence penalty (mean -log top-1 prob)
    constraint_rule: deterministic explicit satisfiability/entailment baseline
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class UnsupervisedEstimator:
    """Base class for training-free per-claim uncertainty estimators."""

    def __init__(self, name: str):
        self.name = name
        self._rng = np.random.default_rng(2026)

    # --- subclass API --------------------------------------------------- #
    def estimate_claims(self, raw_sample: dict) -> Optional[List[float]]:
        """Return one uncertainty score per claim (order matches raw_sample['claims'])."""
        raise NotImplementedError

    # --- numeric helpers ------------------------------------------------ #
    @staticmethod
    def to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def to_probs(top_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(top_values, dtype=np.float64)
        if arr.size == 0:
            return arr
        if float(np.nanmax(arr)) <= 0.0:   # log-prob inputs
            return np.exp(np.clip(arr, -30.0, 0.0))
        return np.clip(arr, 1e-12, 1.0)

    @staticmethod
    def to_log_probs(top_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(top_values, dtype=np.float64)
        if arr.size == 0:
            return arr
        if float(np.nanmax(arr)) <= 0.0:
            return np.clip(arr, -30.0, 0.0)
        return np.log(np.clip(arr, 1e-12, 1.0))

    @staticmethod
    def effective_token_count(raw_sample: dict, max_len: int) -> int:
        am = raw_sample.get("attention_mask")
        if am is None:
            return max_len
        arr = UnsupervisedEstimator.to_numpy(am)
        if arr is None or arr.size == 0:
            return max_len
        count = int((arr > 0).sum())
        return min(max_len, count) if count > 0 else max_len

    @staticmethod
    def claim_token_ids(claim: dict, n_tokens: int) -> List[int]:
        tids = [int(t) for t in claim.get("aligned_token_ids", [])]
        return [t for t in tids if 0 <= t < n_tokens]

    def _top1_probs(self, raw_sample: dict):
        tp = self.to_numpy(raw_sample.get("token_probs"))
        if tp is None or tp.size == 0:
            return None
        top1 = tp[:, 0] if tp.ndim > 1 else tp
        probs = self.to_probs(top1)
        eff = self.effective_token_count(raw_sample, len(probs))
        return probs[:eff]


class RandomBaseline(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("random")

    def estimate_claims(self, raw_sample: dict):
        claims = raw_sample.get("claims") or []
        n = max(len(claims), 1)
        return [float(x) for x in self._rng.random(n)]


class MCPEstimator(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("mcp")

    def estimate_claims(self, raw_sample: dict):
        probs = self._top1_probs(raw_sample)
        claims = raw_sample.get("claims") or []
        if probs is None or len(claims) == 0:
            return None
        n = len(probs)
        g = float(1.0 - np.mean(probs))
        out = []
        for c in claims:
            tids = self.claim_token_ids(c, n)
            out.append(float(1.0 - np.mean(probs[tids])) if tids else g)
        return out


class CCPEstimator(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("ccp")

    def estimate_claims(self, raw_sample: dict):
        probs = self._top1_probs(raw_sample)
        claims = raw_sample.get("claims") or []
        if probs is None or len(claims) == 0:
            return None
        n = len(probs)
        nll = -np.log(np.clip(probs, 1e-12, 1.0))
        g = float(np.mean(nll))
        out = []
        for c in claims:
            tids = self.claim_token_ids(c, n)
            out.append(float(np.mean(nll[tids])) if tids else g)
        return out


class PerplexityEstimator(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("perplexity")

    def _token_nll(self, raw_sample: dict):
        ll = self.to_numpy(raw_sample.get("log_likelihoods"))
        if ll is not None and ll.size > 0:
            eff = self.effective_token_count(raw_sample, len(ll))
            return -np.asarray(ll[:eff], dtype=np.float64)
        probs = self._top1_probs(raw_sample)
        if probs is None:
            return None
        return -np.log(np.clip(probs, 1e-12, 1.0))

    def estimate_claims(self, raw_sample: dict):
        nll = self._token_nll(raw_sample)
        claims = raw_sample.get("claims") or []
        if nll is None or len(claims) == 0:
            return None
        n = len(nll)
        g = float(np.exp(np.clip(np.mean(nll), 0, 30)))
        out = []
        for c in claims:
            tids = self.claim_token_ids(c, n)
            if tids:
                out.append(float(np.exp(np.clip(np.mean(nll[tids]), 0, 30))))
            else:
                out.append(g)
        return out


class TokenEntropyEstimator(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("token_entropy")

    def estimate_claims(self, raw_sample: dict):
        tp = self.to_numpy(raw_sample.get("token_probs"))
        claims = raw_sample.get("claims") or []
        if tp is None or tp.size == 0 or len(claims) == 0:
            return None
        if tp.ndim == 1:  # degenerate 1-D cache: single "top" prob per token
            tp = tp[:, None]
        log_probs = self.to_log_probs(tp)
        eff = self.effective_token_count(raw_sample, len(log_probs))
        log_probs = log_probs[:eff]
        probs = self.to_probs(log_probs)
        entropy = -np.sum(probs * log_probs, axis=-1)
        n = len(entropy)
        g = float(np.mean(entropy))
        out = []
        for c in claims:
            tids = self.claim_token_ids(c, n)
            out.append(float(np.mean(entropy[tids])) if tids else g)
        return out


class ConstraintRuleEstimator(UnsupervisedEstimator):
    """Training-free score derived only from the explicit constraint engine."""

    def __init__(self):
        super().__init__("constraint_rule")

    def estimate_claims(self, raw_sample: dict):
        cf = self.to_numpy(raw_sample.get("claim_constraint_features"))
        claims = raw_sample.get("claims") or []
        if cf is None or cf.ndim != 2 or cf.shape[0] != len(claims):
            return None
        out = []
        for x in cf:
            parseable, feasible_with = float(x[0]), float(x[6])
            entailed, contradicted, unknown = float(x[7]), float(x[8]), float(x[9])
            repair = float(x[11])
            if parseable < 0.5:
                confidence = 0.5
            elif contradicted > 0.5 or feasible_with < 0.5:
                confidence = 0.08 + 0.12 * (1.0 - repair)
            elif entailed > 0.5:
                confidence = 0.92
            elif unknown > 0.5:
                confidence = 0.55
            else:
                confidence = 0.5
            out.append(float(1.0 - np.clip(confidence, 0.0, 1.0)))
        return out


UNSUPERVISED_HEAD_REGISTRY = {
    "random": RandomBaseline,
    "mcp": MCPEstimator,
    "perplexity": PerplexityEstimator,
    "token_entropy": TokenEntropyEstimator,
    "ccp": CCPEstimator,
    "constraint_rule": ConstraintRuleEstimator,
}

UNSUPERVISED_HEAD_ALIASES = {
    "mean_token_entropy": "token_entropy",
    "entropy": "token_entropy",
}

# Backward-compatible alias used by evaluation scripts.
BASELINE_REGISTRY = UNSUPERVISED_HEAD_REGISTRY


def resolve_baseline_name(name: str) -> str:
    return UNSUPERVISED_HEAD_ALIASES.get(name, name)


def build_estimator(name: str) -> UnsupervisedEstimator:
    canonical = resolve_baseline_name(name)
    if canonical not in UNSUPERVISED_HEAD_REGISTRY:
        raise ValueError(f"Unknown baseline {name!r}. Available: {sorted(UNSUPERVISED_HEAD_REGISTRY)}")
    return UNSUPERVISED_HEAD_REGISTRY[canonical]()
