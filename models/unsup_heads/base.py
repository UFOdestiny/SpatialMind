"""Base class for unsupervised uncertainty estimators."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class UnsupervisedEstimator:
    """Base class for unsupervised uncertainty estimators.

    These methods compute uncertainty from cached Phase 1 data
    without any training.
    """

    def __init__(self, name: str):
        self.name = name

    def estimate(self, raw_sample: dict) -> float:
        """Return uncertainty score for a single sample.
        Higher = more uncertain / more likely hallucination."""
        raise NotImplementedError

    @staticmethod
    def to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def to_probs(top_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(top_values, dtype=np.float64)
        if arr.size == 0:
            return arr
        max_val = float(np.nanmax(arr))
        if max_val <= 0.0:
            # log-prob inputs
            return np.exp(np.clip(arr, -30.0, 0.0))
        # probability inputs
        return np.clip(arr, 1e-12, 1.0)

    @staticmethod
    def to_log_probs(top_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(top_values, dtype=np.float64)
        if arr.size == 0:
            return arr
        max_val = float(np.nanmax(arr))
        if max_val <= 0.0:
            return np.clip(arr, -30.0, 0.0)
        return np.log(np.clip(arr, 1e-12, 1.0))

    @staticmethod
    def effective_token_count(raw_sample: dict, max_len: int) -> int:
        attention_mask = raw_sample.get("attention_mask")
        if attention_mask is None:
            return max_len
        if hasattr(attention_mask, "detach"):
            # Convert to float32 first if bfloat16 (numpy doesn't support bfloat16)
            attention_mask = attention_mask.detach().cpu()
            if attention_mask.dtype == torch.bfloat16:
                attention_mask = attention_mask.float()
            attention_mask = attention_mask.numpy()
        arr = np.asarray(attention_mask)
        if arr.size == 0:
            return max_len
        count = int((arr > 0).sum())
        if count <= 0:
            return max_len
        return min(max_len, count)

    @staticmethod
    def claim_token_ids(claim: dict, n_tokens: int) -> List[int]:
        tids = [int(t) for t in claim.get("aligned_token_ids", [])]
        return [t for t in tids if 0 <= t < n_tokens]

    def estimate_claims(self, raw_sample: dict) -> Optional[List[float]]:
        """Optional claim-level scoring.

        Return one uncertainty score per claim (same order as ``raw_sample["verified"]``).
        Returning ``None`` falls back to broadcasting ``estimate(raw_sample)`` to all claims.
        """
        return None

    def estimate_batch(
        self,
        cache_dir: str,
        split: str,
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        claim_level: bool = True,
    ) -> tuple:
        """Return (scores, labels, k_hops) arrays for all samples in split.

        When claim_level=True, expands each sample score to one score per claim
        using cached `verified` labels.
        """
        from data.cached_features import CachedFeatureDataset

        ds = CachedFeatureDataset(
            cache_dir=cache_dir,
            split=split,
            k_hop_values=k_hop_values,
            max_samples=max_samples,
            preload_all=False,
            max_cached_chunks=2,  # Limit memory usage for large test sets
        )
        scores = []
        labels = []
        k_hops = []
        for i in range(len(ds)):
            raw = ds.load_raw(i)
            sample_score = float(self.estimate(raw))

            if claim_level and raw.get("verified"):
                verified = [int(v) for v in raw["verified"]]
                valid_pairs = [(idx, v) for idx, v in enumerate(verified) if v in (0, 1)]
                if not valid_pairs:
                    continue

                claim_scores = self.estimate_claims(raw)
                if claim_scores is None:
                    claim_scores = [sample_score] * len(verified)
                else:
                    claim_scores = [float(s) for s in claim_scores]
                    if len(claim_scores) < len(verified):
                        claim_scores.extend([sample_score] * (len(verified) - len(claim_scores)))
                    elif len(claim_scores) > len(verified):
                        claim_scores = claim_scores[: len(verified)]

                for idx, lbl in valid_pairs:
                    scores.append(claim_scores[idx])
                    labels.append(lbl)
                    k_hops.append(raw.get("k_hop", 0))
                continue

            scores.append(sample_score)
            labels.append(int(raw.get("label", 1)))
            k_hops.append(raw.get("k_hop", 0))
        return np.array(scores), np.array(labels), np.array(k_hops)
