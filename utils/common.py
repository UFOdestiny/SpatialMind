#!/usr/bin/env python3
"""Shared helpers for training/evaluation scripts."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import logging

from scripts.metrics import compute_all_metrics
from utils.efficiency import load_model_with_dtype

log = logging.getLogger(__name__)


def _flatten_to_1d_numeric(x):
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            out = []
            for v in x:
                out.extend(_flatten_to_1d_numeric(v).tolist())
            return np.asarray(out)
        return x.reshape(-1)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten_to_1d_numeric(v).tolist())
        return np.asarray(out)
    return np.asarray([x])


def collate_cached_features(batch):
    """Collate cached feature dataset items with optimized padding.
    
    Uses torch.nn.utils.rnn.pad_sequence for vectorized padding,
    which is more efficient than manual loop-based padding.
    """
    # Check if all samples have same sequence length (no padding needed)
    seq_lens = [b["features"].shape[0] for b in batch]
    all_same_len = len(set(seq_lens)) == 1
    
    if all_same_len:
        # Fast path: all same length, just stack
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
    
    # Slow path: variable lengths, use vectorized padding
    # pad_sequence pads to longest sequence in batch, batch_first=True for (B, L, D)
    padded_features = pad_sequence(
        [b["features"] for b in batch],
        batch_first=True,
        padding_value=0.0,
    )
    
    # For attention_mask, we need to create proper masks
    # pad_sequence would pad with 0, which is correct for attention masks
    padded_masks = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    
    return {
        "features": padded_features,
        "attention_mask": padded_masks,
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def collate_claim_cached_features(batch):
    """Collate claim-level samples and build per-claim token masks."""
    seq_lens = [b["features"].shape[0] for b in batch]
    all_same_len = len(set(seq_lens)) == 1
    if all_same_len:
        features = torch.stack([b["features"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    else:
        features = pad_sequence(
            [b["features"] for b in batch],
            batch_first=True,
            padding_value=0.0,
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        )
    base = {"features": features, "attention_mask": attention_mask}
    seq_len = features.shape[1]

    claim_masks = []
    claim_types = []
    claim_labels = []

    for item in batch:
        claims = item.get("claims", []) or []
        verified = item.get("verified", []) or []
        if not claims or not verified:
            raise ValueError("Claim-level training requires non-empty 'claims' and 'verified' fields.")

        masks = []
        types = []
        labels = []
        for ci, claim in enumerate(claims):
            mask = torch.zeros(seq_len, dtype=torch.float32)
            for tid in claim.get("aligned_token_ids", []):
                if 0 <= int(tid) < seq_len:
                    mask[int(tid)] = 1.0
            if mask.sum() <= 0:
                valid_len = int(item["attention_mask"].sum().item())
                if valid_len > 0:
                    mask[:valid_len] = 1.0
            masks.append(mask)
            types.append(int(claim.get("claim_type_id", 2)))
            label_val = verified[ci] if ci < len(verified) else 0
            labels.append(float(label_val))

        claim_masks.append(torch.stack(masks))
        claim_types.append(torch.tensor(types, dtype=torch.long))
        claim_labels.append(torch.tensor(labels, dtype=torch.float32))

    base["claim_masks"] = claim_masks
    base["claim_types"] = claim_types
    base["claim_labels"] = claim_labels
    return base


def make_binary_compute_metrics():
    """Create an HF Trainer-compatible binary metrics function."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = _flatten_to_1d_numeric(logits)

        # labels may be a single array or a tuple/list of multiple label groups.
        label_candidates = []
        if isinstance(labels, (tuple, list)):
            for part in labels:
                flat = _flatten_to_1d_numeric(part)
                if flat.size > 0:
                    label_candidates.append(flat)
        else:
            label_candidates.append(_flatten_to_1d_numeric(labels))

        if not label_candidates:
            # Avoid crashing trainer loop; return NaNs when labels are missing.
            return {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "ece": float("nan"),
            }

        # Pick candidate closest in length to logits.
        labels = min(label_candidates, key=lambda arr: abs(arr.size - logits.size))
        if labels.size != logits.size:
            min_len = min(labels.size, logits.size)
            log.warning(
                "Metric length mismatch (labels=%d, logits=%d); truncating to %d.",
                labels.size,
                logits.size,
                min_len,
            )
            labels = labels[:min_len]
            logits = logits[:min_len]

        return compute_all_metrics(labels, logits)

    return compute_metrics


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve configured dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_name, torch.float16)


def load_tokenizer_from_path(
    tokenizer_cls: Any,
    model_path: str,
    trust_remote_code: bool,
    cache_dir: Optional[str],
    padding_side: str,
):
    """Load tokenizer with project defaults."""
    tokenizer = tokenizer_cls.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_llm_from_path(
    model_cls: Any,
    model_path: str,
    torch_dtype_name: str,
    device_map: str,
    trust_remote_code: bool,
    cache_dir: Optional[str],
    attn_implementation: str = "eager",
):
    """Load LLM with configured dtype and HF cache options."""
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    return load_model_with_dtype(
        model_cls.from_pretrained,
        model_path,
        torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    )
