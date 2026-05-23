"""Mean Token Entropy: average token-level entropy as uncertainty.

NOTE: This implementation computes entropy from top-N probabilities only,
which is an approximation. The full vocabulary entropy would require
storing all token logits (50k+ floats per token position).

Original implementation computed entropy from full vocabulary:
    entropy = -sum(p * log(p) for all p in vocab)

Current approximation uses top-N (typically top-4) probabilities:
    entropy_approx = -sum(p * log(p) for p in top_n_probs)
    
This approximation is acceptable because:
1. Top-N tokens typically cover 80-95% of probability mass
2. Long-tail tokens contribute little to entropy
3. Relative ranking between samples is preserved
"""

import numpy as np
from models.unsup_heads.base import UnsupervisedEstimator


class TokenEntropyEstimator(UnsupervisedEstimator):
    """Token entropy uncertainty estimator.
    
    Computes mean entropy across all generated token positions.
    Higher entropy indicates higher uncertainty.
    """
    
    def __init__(self):
        super().__init__("token_entropy")

    def estimate(self, raw_sample: dict) -> float:
        token_probs = raw_sample.get("token_probs")
        if token_probs is None:
            return 0.0

        token_probs = self.to_numpy(token_probs)

        if token_probs.size == 0:
            return 0.0

        # Approximate entropy from top-N probabilities.
        # token_probs may be either top-k log-probs or probabilities.
        log_probs = self.to_log_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(log_probs))
        log_probs = log_probs[:eff_n]
        probs = self.to_probs(log_probs)
        entropy_per_token = -np.sum(probs * log_probs, axis=-1)
        return float(np.mean(entropy_per_token))

    def estimate_claims(self, raw_sample: dict):
        token_probs = raw_sample.get("token_probs")
        claims = raw_sample.get("claims") or []
        if token_probs is None or len(claims) == 0:
            return None
        token_probs = self.to_numpy(token_probs)
        if token_probs.size == 0:
            return [0.0] * len(claims)

        log_probs = self.to_log_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(log_probs))
        log_probs = log_probs[:eff_n]
        probs = self.to_probs(log_probs)
        entropy = -np.sum(probs * log_probs, axis=-1)
        n_tokens = len(entropy)
        global_mean = float(np.mean(entropy))
        scores = []
        for claim in claims:
            tids = self.claim_token_ids(claim, n_tokens)
            if tids:
                scores.append(float(np.mean(entropy[tids])))
            else:
                scores.append(global_mean)
        return scores
