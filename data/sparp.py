"""
sparp.py - SpaRP spatial reasoning paths dataset (UKPLab/sparp).

SpaRP (Rizvi et al., 2024) re-renders StepGame- and SpaRTUN-style spatial QA
with an explicit relation-composition path. We convert it at download time to
the SpaRTUN schema (story / question / q_type / answer[list] / candidate_answers)
plus a `num_hop` difficulty field, so this adapter reuses the SpaRTUN loader's
prompt building, multi-label ground-truth resolution, and set-membership
correctness verbatim. Two variants are used as fresh OOD spatial benchmarks:

  * SpaRP_PS1 : SpaRTUN-based, 16-way relation vocabulary, multi-hop.
  * SpaRP_PS3 : StepGame-based, 5-way relations, chained reasoning.

The HF dataset ships train/validation/test directly (no dev alias needed).
Classification task, deterministic correctness, no LLM judge.
"""

import logging
from typing import List, Optional

from datasets import load_from_disk, DatasetDict

from config import GLOBAL_SEED
from data.spartun import SpaRTUNDataset

log = logging.getLogger(__name__)


class SpaRPDataset(SpaRTUNDataset):
    """SpaRP spatial QA (SpaRTUN-schema-compatible). Difficulty = num_hop."""

    name = "sparp"
    difficulty_field = "num_hop"

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        tokenizer=None,
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        max_length: int = 1024,
        **kwargs,
    ):
        # Bypass SpaRTUN.__init__ (which aliases validation->dev); load directly.
        from data.base import BaseTaskDataset
        BaseTaskDataset.__init__(self, dataset_path, split, k_hop_values, max_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading SpaRP split='{split}' from {dataset_path}")
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            ds = ds[split]

        # Optional difficulty (num_hop) filter, reusing k_hop_values semantics.
        if k_hop_values is not None and len(k_hop_values) > 0 and "num_hop" in ds.column_names:
            ds = ds.filter(lambda r: int(r.get("num_hop", 0)) in set(k_hop_values))
            log.info(f"  Filtered to num_hop in {k_hop_values}: {len(ds)} samples")

        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.shuffle(seed=GLOBAL_SEED)
            ds = ds.select(range(max_samples))
            log.info(f"  Subsampled to {max_samples} samples (after shuffle, seed={GLOBAL_SEED})")

        self.data = ds
        log.info(f"  Final dataset size: {len(self.data)} samples")

    def get_difficulty(self, raw_item: dict) -> int:
        return int(raw_item.get("num_hop", 0) or 0)
