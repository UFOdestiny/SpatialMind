import json

import pytest
import torch
from torch.utils.data import DataLoader

from data.cached_features import CachedFeatureDataset
from scripts.rebuild_constraint_cache import rebuild_split
from spatial_constraints import analyze_trace
from utils.common import collate_claim_traces


def _write_split(root, split, n=4):
    split_dir = root / split
    split_dir.mkdir(parents=True)
    samples = []
    for i in range(n):
        correct = i % 2 == 0
        claims = [
            {"text": "A is north of B", "claim_type": "reasoning", "claim_type_id": 1,
             "aligned_token_ids": [0, 1]},
            {"text": "A is north-east of C" if correct else "A is north-west of C",
             "claim_type": "conclusion", "claim_type_id": 2, "aligned_token_ids": [2, 3]},
        ]
        analysis = analyze_trace("A is north of B. B is east of C.", "Where is A?", claims)
        samples.append({
            "features": torch.randn(4, 12),
            "attention_mask": torch.ones(4),
            "label": int(correct),
            "verified": [1, int(correct)],
            "claims": claims,
            "k_hop": 2,
            "claim_constraint_features": torch.tensor([x.features for x in analysis.claims]),
            "trace_constraint_features": torch.tensor(analysis.trace_features),
            "constraint_analysis": analysis.to_dict(),
        })
    torch.save(samples, split_dir / "chunk_0.pt")
    (split_dir / "manifest.json").write_text(json.dumps({
        "cache_schema_version": 2,
        "total_count": n,
        "chunk_size": n,
        "num_chunks": 1,
        "chunk_sample_counts": [n],
        "feature_dim": 12,
        "total_pending": 0,
    }))


def test_native_constraint_cache_collates(tmp_path):
    _write_split(tmp_path, "train")
    ds = CachedFeatureDataset(str(tmp_path), "train")
    batch = next(iter(DataLoader(ds, batch_size=3, collate_fn=collate_claim_traces)))
    assert batch["features"].shape == (3, 4, 12)
    assert batch["trace_constraint_features"].shape[0] == 3
    assert [x.shape[0] for x in batch["claim_constraint_features"]] == [2, 2, 2]


def test_rebuild_constraint_cache_is_native_and_atomic(tmp_path):
    _write_split(tmp_path, "test", n=2)
    rebuild_split(tmp_path, "test")
    manifest = json.loads((tmp_path / "test" / "manifest.json").read_text())
    assert manifest["constraint_method"] == "explicit_spatial_relation_algebra_v2"
    assert manifest["claim_constraint_dim"] > 0
    sample = torch.load(tmp_path / "test" / "chunk_0.pt", weights_only=False)[0]
    assert sample["claim_constraint_features"].ndim == 2
    assert not list((tmp_path / "test").glob("*.tmp"))


def test_incomplete_claim_labels_fail_closed_without_sample_fallback(tmp_path):
    _write_split(tmp_path, "train", n=1)
    path = tmp_path / "train" / "chunk_0.pt"
    chunk = torch.load(path, weights_only=False)
    chunk[0]["verified"] = [1, -1]
    torch.save(chunk, path)
    ds = CachedFeatureDataset(str(tmp_path), "train")
    with pytest.raises(ValueError, match="pending/invalid claim label"):
        _ = ds[0]
