"""Random baseline: assigns random uncertainty scores."""

import numpy as np
from config import GLOBAL_SEED
from models.unsup_heads.base import UnsupervisedEstimator


class RandomBaseline(UnsupervisedEstimator):
    def __init__(self, seed: int = GLOBAL_SEED):
        super().__init__("random")
        self._rng = np.random.RandomState(seed)

    def estimate(self, raw_sample: dict) -> float:
        return float(self._rng.random())

    def estimate_claims(self, raw_sample: dict):
        verified = raw_sample.get("verified") or []
        return [float(self._rng.random()) for _ in range(len(verified))]
