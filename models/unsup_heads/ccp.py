"""
ccp.py - Cumulative Confidence Penalty baseline.

CCP uses top-1 token confidence trajectory:
    ccp = mean(-log(max_token_prob))
Higher values imply lower confidence and higher uncertainty.
"""

import numpy as np

from models.unsup_heads.base import UnsupervisedEstimator


class CCPEstimator(UnsupervisedEstimator):
    def __init__(self):
        super().__init__("ccp")

    def estimate(self, raw_sample: dict) -> float:
        token_probs = raw_sample.get("token_probs")
        if token_probs is None:
            return 0.0
        token_probs = self.to_numpy(token_probs)
        if token_probs.size == 0:
            return 0.0

        max_probs = self._max_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(max_probs))
        max_probs = max_probs[:eff_n]
        return float(np.mean(-np.log(np.clip(max_probs, 1e-12, 1.0))))

    def estimate_claims(self, raw_sample: dict):
        token_probs = raw_sample.get("token_probs")
        claims = raw_sample.get("claims") or []
        if token_probs is None or len(claims) == 0:
            return None
        token_probs = self.to_numpy(token_probs)
        if token_probs.size == 0:
            return [0.0] * len(claims)

        max_probs = self._max_probs(token_probs)
        eff_n = self.effective_token_count(raw_sample, len(max_probs))
        max_probs = max_probs[:eff_n]
        n_tokens = len(max_probs)
        global_score = float(np.mean(-np.log(np.clip(max_probs, 1e-12, 1.0))))

        scores = []
        for claim in claims:
            tids = self.claim_token_ids(claim, n_tokens)
            if tids:
                c_probs = np.clip(max_probs[tids], 1e-12, 1.0)
                scores.append(float(np.mean(-np.log(c_probs))))
            else:
                scores.append(global_score)
        return scores

    @staticmethod
    def _max_probs(token_probs: np.ndarray) -> np.ndarray:
        if token_probs.ndim > 1:
            top1 = token_probs[:, 0]
        else:
            top1 = token_probs
        return UnsupervisedEstimator.to_probs(top1)
