"""Perplexity-based uncertainty: exp(mean(-log_prob)) of generated tokens.

Perplexity measures how "surprised" the model is by its own generated tokens.
Higher perplexity indicates lower confidence / higher uncertainty.

Standard definition:
    PPL = exp(-(1/N) * sum(log p(token_i)))
    
Where log p(token_i) is the log probability assigned to each generated token.

Original implementation without clipping:
    avg_neg_log_prob = -mean(log_likelihoods)
    perplexity = exp(avg_neg_log_prob)

Current implementation with safety clipping:
    avg_neg_log_prob = clip(-mean(log_likelihoods), 0, 20)
    perplexity = exp(avg_neg_log_prob)  # max ~485M

The clipping prevents numerical overflow when log probabilities are very
negative (e.g., for out-of-vocabulary or unlikely tokens).
"""

import numpy as np
from models.unsup_heads.base import UnsupervisedEstimator


class PerplexityEstimator(UnsupervisedEstimator):
    """Perplexity-based uncertainty estimator.
    
    Computes exp(mean(-log_prob)) across generated tokens.
    Higher perplexity indicates higher uncertainty.
    """
    
    def __init__(self):
        super().__init__("perplexity")

    def estimate(self, raw_sample: dict) -> float:
        log_likelihoods = raw_sample.get("log_likelihoods")
        if log_likelihoods is None:
            return 1.0

        log_likelihoods = self.to_numpy(log_likelihoods)

        if log_likelihoods.size == 0:
            return 1.0

        eff_n = self.effective_token_count(raw_sample, len(log_likelihoods))
        log_likelihoods = log_likelihoods[:eff_n]
        # Perplexity = exp(mean(-log_prob))
        avg_neg_log_prob = -np.mean(log_likelihoods)
        # Clip to avoid overflow: exp(20) ≈ 485M (reasonable upper bound)
        avg_neg_log_prob = np.clip(avg_neg_log_prob, 0, 20)
        return float(np.exp(avg_neg_log_prob))

    def estimate_claims(self, raw_sample: dict):
        log_likelihoods = raw_sample.get("log_likelihoods")
        claims = raw_sample.get("claims") or []
        if log_likelihoods is None or len(claims) == 0:
            return None
        log_likelihoods = self.to_numpy(log_likelihoods)
        if log_likelihoods.size == 0:
            return [1.0] * len(claims)

        eff_n = self.effective_token_count(raw_sample, len(log_likelihoods))
        log_likelihoods = log_likelihoods[:eff_n]
        n_tokens = len(log_likelihoods)
        scores = []
        global_avg_neg = float(np.clip(-np.mean(log_likelihoods), 0.0, 20.0))
        for claim in claims:
            tids = self.claim_token_ids(claim, n_tokens)
            if tids:
                avg_neg = float(np.clip(-np.mean(log_likelihoods[tids]), 0.0, 20.0))
            else:
                avg_neg = global_avg_neg
            scores.append(float(np.exp(avg_neg)))
        return scores
