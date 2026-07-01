#!/usr/bin/env python3
"""Shared helpers for training / evaluation scripts (SpatialMind)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.efficiency import load_model_with_dtype

log = logging.getLogger(__name__)

# Sentinel written by Phase 1 for still-unlabeled claims/traces.
PENDING_LABEL = -1


def collate_claim_traces(batch: List[dict]) -> dict:
    """Collate cached trace samples into a padded batch for a claim-level head.

    Each input item (from CachedFeatureDataset) provides:
        features        : (L_i, D)
        attention_mask  : (L_i,)
        claims          : list of claim dicts (with aligned_token_ids, claim_type_id)
        claim_labels    : list[int]  per-claim correctness (already filtered to 0/1)
        trace_label     : int        sample-level correctness (0/1)
        k_hop           : int

    Output batch keys:
        features        : (B, L, D)          (padded)
        attention_mask  : (B, L)
        claim_masks     : list[(C_i, L)]     per-trace token-span masks
        claim_types     : list[(C_i,)]       per-trace claim-type ids
        claim_labels    : list[(C_i,)]       per-trace claim labels
        trace_labels    : (B,)               sample-level labels
        k_hops          : (B,)
    """
    seq_lens = [b["features"].shape[0] for b in batch]
    same_len = len(set(seq_lens)) == 1
    if same_len:
        features = torch.stack([b["features"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    else:
        features = pad_sequence([b["features"] for b in batch], batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)

    seq_len = features.shape[1]
    claim_masks: List[torch.Tensor] = []
    claim_types: List[torch.Tensor] = []
    claim_labels: List[torch.Tensor] = []

    for item in batch:
        claims = item.get("claims", []) or []
        labels = item.get("claim_labels", []) or []
        masks, types, labs = [], [], []
        valid_len = int(item["attention_mask"].sum().item())
        for ci, claim in enumerate(claims):
            if ci >= len(labels):
                break
            mask = torch.zeros(seq_len, dtype=torch.float32)
            for tid in claim.get("aligned_token_ids", []):
                tid = int(tid)
                if 0 <= tid < seq_len:
                    mask[tid] = 1.0
            if mask.sum() <= 0 and valid_len > 0:
                # Fallback: attach to the whole valid span so pooling is well-defined.
                mask[: min(valid_len, seq_len)] = 1.0
            masks.append(mask)
            types.append(int(claim.get("claim_type_id", 2)))
            labs.append(float(labels[ci]))
        if not masks:
            # Degenerate trace with no usable claims: single whole-trace claim.
            mask = torch.zeros(seq_len, dtype=torch.float32)
            mask[: min(valid_len, seq_len)] = 1.0
            masks.append(mask)
            types.append(2)
            labs.append(float(item.get("trace_label", 1)))
        claim_masks.append(torch.stack(masks))
        claim_types.append(torch.tensor(types, dtype=torch.long))
        claim_labels.append(torch.tensor(labs, dtype=torch.float32))

    trace_labels = torch.tensor([int(b.get("trace_label", 1)) for b in batch], dtype=torch.long)
    k_hops = torch.tensor([int(b.get("k_hop", 0)) for b in batch], dtype=torch.long)

    return {
        "features": features,
        "attention_mask": attention_mask,
        "claim_masks": claim_masks,
        "claim_types": claim_types,
        "claim_labels": claim_labels,
        "trace_labels": trace_labels,
        "k_hops": k_hops,
    }


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name, torch.float16)


def load_tokenizer_from_path(tokenizer_cls: Any, model_path: str, trust_remote_code: bool,
                             cache_dir: Optional[str], padding_side: str):
    tokenizer = tokenizer_cls.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_llm_from_path(model_cls: Any, model_path: str, torch_dtype_name: str,
                       device_map: str, trust_remote_code: bool, cache_dir: Optional[str],
                       attn_implementation: str = "eager"):
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    return load_model_with_dtype(
        model_cls.from_pretrained, model_path, torch_dtype,
        device_map=device_map, trust_remote_code=trust_remote_code,
        cache_dir=cache_dir, attn_implementation=attn_implementation,
    )
