"""
claim_utils.py - Shared helpers for claim-aware heads.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def flatten_claim_targets(claim_labels: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat([x.to(device).float() for x in claim_labels], dim=0)


def claim_vectors_from_masks(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    claim_masks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Build per-sample claim vectors from token features using claim masks.
    """
    out: List[torch.Tensor] = []
    for i in range(features.shape[0]):
        token_feats = features[i]  # (L, D)
        valid = attention_mask[i].float().unsqueeze(0)  # (1, L)
        cm = claim_masks[i].to(features.device).float() * valid  # (C, L)
        denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        vecs = (cm @ token_feats) / denom
        out.append(vecs)
    return out


def flatten_claim_metadata(
    claim_vecs_per_sample: List[torch.Tensor],
    claim_types: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    vectors = []
    types = []
    for i, vecs in enumerate(claim_vecs_per_sample):
        vectors.append(vecs)
        t = claim_types[i].to(vecs.device).long()
        if t.shape[0] != vecs.shape[0]:
            if t.shape[0] > vecs.shape[0]:
                t = t[: vecs.shape[0]]
            else:
                pad = torch.full(
                    (vecs.shape[0] - t.shape[0],),
                    2,
                    dtype=torch.long,
                    device=vecs.device,
                )
                t = torch.cat([t, pad], dim=0)
        types.append(t)
    return torch.cat(vectors, dim=0), torch.cat(types, dim=0)

