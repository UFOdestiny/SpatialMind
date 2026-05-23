"""
cached_features.py - Chunk-based dataset for pre-extracted LLM features.

Phase 1 saves features in chunk files:
    {cache_dir}/{split}/chunk_{i}.pt  — list of sample dicts
    {cache_dir}/{split}/manifest.json — metadata (total_count, chunk_size, feature_dim, ...)

Each sample dict contains:
    features:           (seq_len, feature_dim) tensor
    attention_mask:     (seq_len,) tensor
    claims:             list of claim dicts (with aligned_token_ids / claim_type_id)
    verified:           list of claim labels (0/1, optional -1 pending)
    k_hop:              int
    question:           str
    generated_text:     str
    dataset_name:       str
    token_probs:        (gen_len, top_n) tensor (for unsupervised baselines)
    log_likelihoods:    (gen_len,) tensor (for unsupervised baselines)
"""

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

log = logging.getLogger(__name__)

# Maximum number of chunks to keep in LRU cache.
# Keep low (2) to avoid OOM on large datasets (chunks can be 20-40 GB each).
MAX_CACHED_CHUNKS = 2
# Number of parallel threads for preloading chunks
PRELOAD_WORKERS = 2


class ChunkAwareSampler(Sampler[int]):
    """Sampler that groups indices by chunk to maximize LRU cache hits.

    Within each chunk, sample order is shuffled. Chunk order is also shuffled
    per epoch. This reduces disk I/O from O(steps) random chunk loads to
    O(num_chunks) sequential loads per epoch.
    """

    def __init__(self, dataset: "CachedFeatureDataset", seed: int = 42, drop_last: bool = False, batch_size: int = 1):
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self._total = len(dataset)

        # Group global indices by their chunk_idx
        self._chunk_groups: dict[int, list[int]] = {}
        for global_idx, (chunk_idx, _) in enumerate(dataset._index):
            self._chunk_groups.setdefault(chunk_idx, []).append(global_idx)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        chunk_ids = list(self._chunk_groups.keys())
        chunk_order = torch.randperm(len(chunk_ids), generator=g).tolist()

        indices: list[int] = []
        for ci in chunk_order:
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
    """Load pre-extracted features from chunk files on disk.

    Supports efficient random access by mapping global indices to
    (chunk_id, local_index) pairs via the manifest.
    
    Optimizations:
    - Optional preload_all mode: loads all chunks into memory upfront (faster training)
    - LRU cache eviction when not preloading (most recently used chunks kept)
    - Parallel chunk loading with ThreadPoolExecutor
    - Chunk data released immediately after indexing to reduce peak memory
    - Configurable cache size via MAX_CACHED_CHUNKS
    """

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
        """
        Args:
            cache_dir: Root cache directory.
            split: Data split name.
            k_hop_values: Filter by difficulty. None = all.
            max_samples: Limit samples. 0 = no limit.
            skip_pending: Skip samples that still contain pending claim labels (-1).
                Set to True for training, False for judge re-evaluation.
            preload_all: If True, load all chunks into memory upfront.
                Faster training at the cost of initial load time and memory.
            max_cached_chunks: Override MAX_CACHED_CHUNKS. 0 = use default.
                Set to a smaller value (e.g., 2) for large test sets to avoid OOM.
        """
        self.split_dir = Path(cache_dir) / split
        manifest_path = self.split_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run scripts/generate.py first to create cached features."
            )

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.total_count = self.manifest["total_count"]
        self.chunk_size = self.manifest["chunk_size"]
        self.feature_dim = self.manifest.get("feature_dim", 0)
        self._preload_all = preload_all
        self._max_cached_chunks = max_cached_chunks if max_cached_chunks > 0 else MAX_CACHED_CHUNKS

        # Chunk file list
        self.chunk_files = sorted(self.split_dir.glob("chunk_*.pt"))
        log.info(
            "CachedFeatureDataset: split=%s, total=%d, chunks=%d, feature_dim=%d",
            split, self.total_count, len(self.chunk_files), self.feature_dim,
        )

        # LRU cache: OrderedDict maintains insertion order for LRU eviction
        self._chunk_cache: OrderedDict[int, list] = OrderedDict()

        # Build index with memory-efficient loading
        self._index = self._build_index(k_hop_values, max_samples, skip_pending)
        
        # Preload all chunks if requested (parallel loading)
        if preload_all and len(self.chunk_files) > 0:
            self._preload_chunks()

    def _build_index(
        self,
        k_hop_values: Optional[List[int]],
        max_samples: int,
        skip_pending: bool,
    ) -> List[tuple]:
        """Build sample index while minimizing peak memory usage.
        
        Optimization: When skip_pending=True and manifest shows total_pending=0,
        and k_hop_values=None, we can build the index directly from manifest
        without loading any chunk data.
        """
        index = []
        skipped_pending = 0
        
        # Fast path: if no filtering needed, build index from manifest
        total_pending = self.manifest.get("total_pending", 0)
        total_claim_pending = self.manifest.get("total_claim_pending", 0)
        can_skip_loading = (
            (not skip_pending or (total_pending == 0 and total_claim_pending == 0))
            and k_hop_values is None
        )
        
        if can_skip_loading:
            # Build index directly from manifest without loading chunks
            chunk_counts = self.manifest.get("chunk_sample_counts", [])
            for chunk_idx, count in enumerate(chunk_counts):
                for local_idx in range(count):
                    index.append((chunk_idx, local_idx))
                    if max_samples > 0 and len(index) >= max_samples:
                        break
                if max_samples > 0 and len(index) >= max_samples:
                    break
            log.info("  After filtering: %d samples (fast path, no chunk loading)", len(index))
            return index

        # Slow path: need to load chunks to check pending/k_hop
        import gc
        for chunk_idx, chunk_path in enumerate(self.chunk_files):
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
            
            for local_idx, sample in enumerate(chunk_data):
                # Skip unjudged samples to prevent them entering training.
                pending_claim = any(int(v) == -1 for v in sample.get("verified", []))
                pending_sample = sample.get("label", -1) == -1
                if skip_pending and (pending_claim or pending_sample):
                    skipped_pending += 1
                    continue
                k_hop = sample.get("k_hop", 0)
                if k_hop_values and k_hop not in k_hop_values:
                    continue
                index.append((chunk_idx, local_idx))
                
                # Early exit if max_samples reached
                if max_samples > 0 and len(index) >= max_samples:
                    break
            
            # Release chunk memory immediately after indexing
            del chunk_data
            gc.collect()  # Force garbage collection to reclaim memory
            
            if max_samples > 0 and len(index) >= max_samples:
                break

        log.info("  After filtering: %d samples (skipped %d pending)", len(index), skipped_pending)
        return index

    def __len__(self):
        return len(self._index)

    def _preload_chunks(self):
        """Preload all chunks into memory using parallel threads."""
        num_chunks = len(self.chunk_files)
        log.info("Preloading %d chunks in parallel (workers=%d)...", num_chunks, PRELOAD_WORKERS)
        
        def load_one(chunk_idx: int) -> tuple:
            path = self.chunk_files[chunk_idx]
            data = torch.load(path, map_location="cpu", weights_only=False)
            return chunk_idx, data
        
        with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as executor:
            results = list(executor.map(load_one, range(num_chunks)))
        
        # Store all loaded chunks
        for chunk_idx, data in results:
            self._chunk_cache[chunk_idx] = data
        
        log.info("Preloaded %d chunks into memory", len(self._chunk_cache))

    def _load_chunk(self, chunk_idx: int) -> list:
        """Load chunk data with LRU caching."""
        if chunk_idx in self._chunk_cache:
            # Move to end (most recently used) - skip if preloaded (all in cache)
            if not self._preload_all:
                self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        # Evict oldest chunks if cache is full (LRU eviction) - only when not preloaded
        if not self._preload_all:
            while len(self._chunk_cache) >= self._max_cached_chunks:
                # popitem(last=False) removes the first (oldest) item
                self._chunk_cache.popitem(last=False)

        # Load and cache new chunk
        path = self.chunk_files[chunk_idx]
        chunk_data = torch.load(path, map_location="cpu", weights_only=False)
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data

    def __getitem__(self, idx):
        chunk_idx, local_idx = self._index[idx]
        chunk_data = self._load_chunk(chunk_idx)
        sample = chunk_data[local_idx]

        claims = sample.get("claims", [])
        verified = sample.get("verified", [])

        verified = sample.get("verified", [])
        sample_label = int(verified[-1]) if verified else int(sample.get("label", 1))
        return {
            "features": sample["features"],            # (seq_len, feature_dim)
            "attention_mask": sample["attention_mask"],  # (seq_len,)
            "labels": torch.tensor(sample_label, dtype=torch.long),
            "k_hop": sample.get("k_hop", 0),
            "claims": claims,
            "verified": verified,
        }

    def load_raw(self, idx) -> dict:
        """Load full raw sample dict (including token_probs, log_likelihoods)
        for unsupervised baseline evaluation.

        Handles both old field names (predicted_dir, ground_truth_dir) and
        new field names (predicted_answer, ground_truth) transparently.
        """
        chunk_idx, local_idx = self._index[idx]
        chunk_data = self._load_chunk(chunk_idx)
        sample = chunk_data[local_idx]
        # Backward compatibility: present both old and new field names
        if "predicted_answer" not in sample and "predicted_dir" in sample:
            sample["predicted_answer"] = sample["predicted_dir"]
        if "ground_truth" not in sample and "ground_truth_dir" in sample:
            sample["ground_truth"] = sample["ground_truth_dir"]
        return sample

    def get_claim_labels(self) -> list:
        """Return flattened claim-level labels for logging/evaluation stats."""
        labels = []
        for chunk_idx, local_idx in self._index:
            chunk_data = self._load_chunk(chunk_idx)
            sample = chunk_data[local_idx]
            verified = sample.get("verified")
            if verified:
                labels.extend([int(v) for v in verified if int(v) in (0, 1)])
            else:
                labels.append(int(sample.get("label", 1)))
        return labels

    def get_claim_label_stats(self) -> tuple:
        """Return (total, n_correct, n_hallucinated) without polluting the LRU cache.

        Loads one chunk at a time and releases it immediately, so peak memory
        is bounded by a single chunk instead of MAX_CACHED_CHUNKS chunks.
        Use this instead of ``get_claim_labels()`` when you only need counts.
        """
        import gc

        # Group indices by chunk for sequential, single-pass loading
        chunk_to_locals: dict[int, list[int]] = {}
        for chunk_idx, local_idx in self._index:
            chunk_to_locals.setdefault(chunk_idx, []).append(local_idx)

        total = 0
        correct = 0
        for chunk_idx in sorted(chunk_to_locals):
            # Bypass LRU cache — load directly and release after use
            chunk_data = torch.load(
                self.chunk_files[chunk_idx], map_location="cpu", weights_only=False
            )
            for local_idx in chunk_to_locals[chunk_idx]:
                sample = chunk_data[local_idx]
                verified = sample.get("verified")
                if verified:
                    for v in verified:
                        v_int = int(v)
                        if v_int in (0, 1):
                            total += 1
                            correct += v_int
                else:
                    total += 1
                    correct += int(sample.get("label", 1))
            del chunk_data
            gc.collect()

        return total, correct, total - correct
