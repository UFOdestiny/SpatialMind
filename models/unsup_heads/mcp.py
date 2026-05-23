"""MCP (Maximum Class Probability): 1 - mean(max_token_prob) as uncertainty.

This estimator uses the top-1 probability from each token position to
compute uncertainty. Higher values indicate lower model confidence.

The feature extraction pipeline saves top-N probabilities per token,
where top_probs[:, 0] contains the highest probability for each position.
We compute: uncertainty = 1 - mean(top_1_prob across all tokens)

Original implementation:
    max_prob = softmax(logits).max(dim=-1)  # Over full vocab
    uncertainty = 1 - mean(max_prob)

Current implementation (using pre-extracted top-N):
    max_prob = top_probs[:, 0]  # Already sorted, first column is max
    uncertainty = 1 - mean(max_prob)
    
This is equivalent since top_probs[:, 0] == softmax(logits).max(dim=-1).
"""

import numpy as np
from models.unsup_heads.base import UnsupervisedEstimator


class MCPEstimator(UnsupervisedEstimator):
    """Maximum Class Probability uncertainty estimator.
    
    Computes 1 - mean(max_token_prob) as uncertainty score.
    Low max probability indicates model is uncertain.
    """
    
    def __init__(self):
        super().__init__("mcp")

    def estimate(self, raw_sample: dict) -> float:
        token_probs = raw_sample.get("token_probs")
        if token_probs is None:
            return 0.5

        token_probs = self.to_numpy(token_probs)

        if token_probs.size == 0:
            return 0.5

        max_probs = self._max_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(max_probs))
        max_probs = max_probs[:eff_n]
        # Uncertainty = 1 - mean confidence
        return float(1.0 - np.mean(max_probs))

    def estimate_claims(self, raw_sample: dict):
        token_probs = raw_sample.get("token_probs")
        claims = raw_sample.get("claims") or []
        if token_probs is None or len(claims) == 0:
            return None
        token_probs = self.to_numpy(token_probs)
        if token_probs.size == 0:
            return [0.5] * len(claims)

        max_probs = self._max_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(max_probs))
        max_probs = max_probs[:eff_n]
        n_tokens = len(max_probs)
        scores = []
        for claim in claims:
            tids = self.claim_token_ids(claim, n_tokens)
            if tids:
                scores.append(float(1.0 - np.mean(max_probs[tids])))
            else:
                scores.append(float(1.0 - np.mean(max_probs)))
        return scores

    @staticmethod
    def _max_probs(token_probs: np.ndarray) -> np.ndarray:
        # Top-1 value can be either log-prob (current cache) or probability.
        if token_probs.ndim > 1:
            top1 = token_probs[:, 0]
        else:
            top1 = token_probs
        return UnsupervisedEstimator.to_probs(top1)
