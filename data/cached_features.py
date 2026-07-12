"""
cached_features.py - Chunk-based dataset over Phase-1 cached trace features.

Phase 1 writes, per split:
    {cache_dir}/{split}/chunk_{i}.pt   — list of per-trace sample dicts
    {cache_dir}/{split}/manifest.json  — {total_count, chunk_size, feature_dim,
                                          chunk_sample_counts, total_pending, ...}

Each cached sample dict (see the Phase-1 data contract) contains:
    features        (L, D)      frozen per-token features
    attention_mask  (L,)        1 for generated tokens
    label           int         SAMPLE-LEVEL correctness (1 correct / 0 hallucination / -1 pending)
    verified        list[int]   per-claim correctness (1 / 0 / -1 pending)
    claims          list[dict]  {text, claim_type, claim_type_id, aligned_token_ids, ...}
    k_hop           int
    token_probs, log_likelihoods, question, generated_text, ...
    context                     trusted scene/premise text
    claim_constraint_features   (C, D_c) explicit relation-algebra features
    trace_constraint_features   (D_t,) explicit trace-consistency features

This dataset is TRACE-first: __getitem__ returns one trace with its ordered
claims, per-claim labels, and the trace-level label, ready for the claim-level
head + multi-task trace objective. Pending (-1) samples/claims are excluded from
training and evaluation.
"""

from __future__ import annotations

import gc
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

import os

import torch
from torch.utils.data import Dataset, Sampler
from spatial_constraints import CLAIM_CONSTRAINT_DIM

log = logging.getLogger(__name__)

# Number of chunks kept resident in the LRU cache. Chunks can be tens of GB each
# at full scale (feature_dim ~4.4k x 10k traces), so keep this small. Env-tunable
# for tight-RAM nodes. With num_workers>0 EACH worker holds its own cache, so the
# effective RAM is roughly (num_workers+1) x MAX_CACHED_CHUNKS x chunk_size — this
# is why training defaults to num_workers=0 (see TrainingConfig.num_workers).
MAX_CACHED_CHUNKS = int(os.environ.get("MAX_CACHED_CHUNKS", "1"))
PRELOAD_WORKERS = 2
PENDING = -1


class ChunkAwareSampler(Sampler[int]):
    """Group indices by chunk to maximize LRU cache hits; shuffle within/between chunks."""

    def __init__(self, dataset: "CachedFeatureDataset", seed: int = 42,
                 drop_last: bool = False, batch_size: int = 1):
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self._total = len(dataset)
        self._chunk_groups: dict[int, list[int]] = {}
        for global_idx, (chunk_idx, _) in enumerate(dataset._index):
            self._chunk_groups.setdefault(chunk_idx, []).append(global_idx)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        chunk_ids = list(self._chunk_groups.keys())
        order = torch.randperm(len(chunk_ids), generator=g).tolist()
        indices: list[int] = []
        for ci in order:
            group = self._chunk_groups[chunk_ids[ci]]
            perm = torch.randperm(len(group), generator=g).tolist()
            indices.extend(group[p] for p in perm)
        if self.drop_last and len(indices) % self.batch_size != 0:
            indices = indices[: len(indices) - len(indices) % self.batch_size]
        return iter(indices)

    def __len__(self) -> int:
        if self.drop_last:
            return self._total - (self._total % self.batch_size)
        return self._total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class CachedFeatureDataset(Dataset):
    """Load Phase-1 cached traces with bounded-memory chunk access."""

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        skip_pending: bool = True,
        preload_all: bool = False,
        max_cached_chunks: int = 0,
    ):
        self.split_dir = Path(cache_dir) / split
        manifest_path = self.split_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Run scripts/generate.py first."
            )
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        if int(self.manifest.get("cache_schema_version", 0)) < 2:
            raise ValueError(
                f"Legacy cache schema at {manifest_path}; regenerate with the explicit-constraint pipeline"
            )
        if skip_pending and int(self.manifest.get("total_pending", 0)) != 0:
            raise ValueError(f"{manifest_path}: trace labels remain pending; run the judge first")
        if skip_pending and int(self.manifest.get("total_claim_pending", 0)) != 0:
            raise ValueError(f"{manifest_path}: claim labels remain pending; run the judge first")

        self.total_count = self.manifest.get("total_count", 0)
        self.chunk_size = self.manifest.get("chunk_size", 0)
        self.feature_dim = self.manifest.get("feature_dim", 0)
        self._preload_all = preload_all
        self._max_cached_chunks = max_cached_chunks if max_cached_chunks > 0 else MAX_CACHED_CHUNKS
        self.chunk_files = sorted(self.split_dir.glob("chunk_*.pt"))
        self._chunk_cache: "OrderedDict[int, list]" = OrderedDict()

        self._index = self._build_index(k_hop_values, max_samples, skip_pending)
        log.info(
            "CachedFeatureDataset: split=%s, usable=%d/%d, chunks=%d, feature_dim=%d",
            split, len(self._index), self.total_count, len(self.chunk_files), self.feature_dim,
        )
        if preload_all and self.chunk_files:
            self._preload_chunks()

    # ------------------------------------------------------------------ #
    def _build_index(self, k_hop_values, max_samples, skip_pending) -> List[tuple]:
        index: List[tuple] = []
        total_pending = self.manifest.get("total_pending", 0)
        # Fast path: no filtering needed -> build straight from manifest counts.
        if (not skip_pending or total_pending == 0) and not k_hop_values:
            for chunk_idx, count in enumerate(self.manifest.get("chunk_sample_counts", [])):
                for local_idx in range(count):
                    index.append((chunk_idx, local_idx))
                    if max_samples > 0 and len(index) >= max_samples:
                        return index
            return index

        # Slow path: inspect samples to skip pending / filter difficulty.
        for chunk_idx, chunk_path in enumerate(self.chunk_files):
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            for local_idx, sample in enumerate(chunk):
                if skip_pending and int(sample.get("label", PENDING)) == PENDING:
                    continue
                if k_hop_values and sample.get("k_hop", 0) not in k_hop_values:
                    continue
                index.append((chunk_idx, local_idx))
                if max_samples > 0 and len(index) >= max_samples:
                    del chunk
                    gc.collect()
                    return index
            del chunk
            gc.collect()
        return index

    def _preload_chunks(self):
        def load_one(i):
            return i, torch.load(self.chunk_files[i], map_location="cpu", weights_only=False)
        with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as ex:
            for i, data in ex.map(load_one, range(len(self.chunk_files))):
                self._chunk_cache[i] = data

    def _load_chunk(self, chunk_idx: int) -> list:
        if chunk_idx in self._chunk_cache:
            if not self._preload_all:
                self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]
        if not self._preload_all:
            while len(self._chunk_cache) >= self._max_cached_chunks:
                self._chunk_cache.popitem(last=False)
        chunk = torch.load(self.chunk_files[chunk_idx], map_location="cpu", weights_only=False)
        self._chunk_cache[chunk_idx] = chunk
        return chunk

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self._index)

    @staticmethod
    def _usable_claims(sample: dict):
        """Return the complete claim sequence under a fail-closed label contract."""
        claims = sample.get("claims", []) or []
        verified = sample.get("verified", []) or []
        constraint_features = sample.get("claim_constraint_features")
        if constraint_features is None:
            raise ValueError(
                "Cache lacks native explicit-constraint features. Regenerate it with "
                "the constraint-aware Phase-1 pipeline (cache_schema_version >= 2)."
            )
        constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)
        if not claims:
            raise ValueError("cache sample has no extracted claims")
        if len(verified) != len(claims):
            raise ValueError("claim/verified length mismatch in cache")
        labels = [int(v) for v in verified]
        if any(v not in (0, 1) for v in labels):
            raise ValueError("pending/invalid claim label reached train/evaluation loader")
        if constraint_features.shape[0] != len(claims):
            raise ValueError("claim/constraint feature length mismatch in cache")
        return claims, labels, constraint_features

    def __getitem__(self, idx):
        chunk_idx, local_idx = self._index[idx]
        sample = self._load_chunk(chunk_idx)[local_idx]
        claims, claim_labels, claim_constraint_features = self._usable_claims(sample)
        trace_constraint_features = sample.get("trace_constraint_features")
        if trace_constraint_features is None:
            raise ValueError("Cache lacks trace_constraint_features; regenerate Phase-1 cache")
        trace_label = int(sample.get("label", PENDING))
        if trace_label not in (0, 1):
            raise ValueError("pending/invalid trace label reached train/evaluation loader")
        return {
            "features": sample["features"],
            "attention_mask": sample["attention_mask"],
            "claims": claims,
            "claim_labels": claim_labels,
            "claim_constraint_features": claim_constraint_features,
            "trace_constraint_features": torch.as_tensor(trace_constraint_features, dtype=torch.float32),
            "trace_label": trace_label,
            "k_hop": int(sample.get("k_hop", 0)),
        }

    def load_raw(self, idx) -> dict:
        """Full raw sample dict (token_probs, log_likelihoods, text, ...) for baselines."""
        chunk_idx, local_idx = self._index[idx]
        sample = self._load_chunk(chunk_idx)[local_idx]
        if "predicted_answer" not in sample and "predicted_dir" in sample:
            sample["predicted_answer"] = sample["predicted_dir"]
        if "ground_truth" not in sample and "ground_truth_dir" in sample:
            sample["ground_truth"] = sample["ground_truth_dir"]
        return sample

    def trace_label_stats(self) -> tuple:
        """(n_traces, n_correct, n_hallucinated) using top-level trace labels."""
        n = len(self._index)
        correct = 0
        chunk_to_locals: dict[int, list[int]] = {}
        for chunk_idx, local_idx in self._index:
            chunk_to_locals.setdefault(chunk_idx, []).append(local_idx)
        for chunk_idx in sorted(chunk_to_locals):
            chunk = torch.load(self.chunk_files[chunk_idx], map_location="cpu", weights_only=False)
            for local_idx in chunk_to_locals[chunk_idx]:
                lbl = int(chunk[local_idx].get("label", 1))
                correct += 1 if lbl == 1 else 0
            del chunk
            gc.collect()
        return n, correct, n - correct

    def claim_label_stats(self) -> tuple:
        """(n_claims, n_correct, n_hallucinated) over usable per-claim labels."""
        total = correct = 0
        chunk_to_locals: dict[int, list[int]] = {}
        for chunk_idx, local_idx in self._index:
            chunk_to_locals.setdefault(chunk_idx, []).append(local_idx)
        for chunk_idx in sorted(chunk_to_locals):
            chunk = torch.load(self.chunk_files[chunk_idx], map_location="cpu", weights_only=False)
            for local_idx in chunk_to_locals[chunk_idx]:
                _, labels, _ = self._usable_claims(chunk[local_idx])
                total += len(labels)
                correct += sum(labels)
            del chunk
            gc.collect()
        return total, correct, total - correct
