"""
wrapper.py - Multi-task training/inference wrapper over a claim-level UQ head.

`ClaimUQModel` wraps any head implementing the HeadOutput contract and computes:
  - claim loss  : BCE over per-claim logits vs per-claim labels (the localization
                  signal; ablatable per claim type).
  - trace loss  : BCE over the head's learned trace logit vs the trace label, when
                  the head emits one (SpatialMind's multi-task objective). This
                  directly optimizes the reported SAMPLE-LEVEL score.
Total loss = claim_loss + trace_loss_weight * trace_loss.

It also holds `build_feature_extractor` for the plug-and-play inference adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.features.attention import AttentionExtractor
from models.features.combined import CombinedExtractor
from models.features.hidden_states import HiddenStateExtractor
from models.features.token_probs import TokenProbExtractor
from models.heads.base import HeadOutput


@dataclass
class UQOutput:
    """Batch forward result."""
    loss: Optional[torch.Tensor]
    claim_logits: torch.Tensor          # (B, max_claims, C)
    claim_mask: torch.Tensor            # (B, max_claims)
    trace_logit: Optional[torch.Tensor]  # (B, C) or None
    claim_loss: Optional[torch.Tensor] = None
    trace_loss: Optional[torch.Tensor] = None


def _binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "bce",
    pos_weight: float = 1.0,
    focal_gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary loss over 1-D logits/labels: bce | balanced_bce | focal."""
    labels = labels.float()
    if loss_type == "balanced_bce":
        pw = torch.tensor([max(pos_weight, 1e-8)], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction=reduction)
    if loss_type == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, p, 1.0 - p)
        loss = ((1.0 - pt).pow(focal_gamma) * bce)
        return loss.mean() if reduction == "mean" else loss.sum()
    return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)


class ClaimUQModel(nn.Module):
    """Trains a claim-level head with an optional multi-task trace objective."""

    def __init__(
        self,
        head: nn.Module,
        num_classes: int = 1,
        loss_type: str = "bce",
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        trace_loss_weight: float = 0.5,
        claim_pos_weight: Optional[float] = None,
        trace_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.head = head
        self.num_classes = num_classes
        self.loss_type = (loss_type or "bce").lower()
        self.focal_gamma = float(focal_gamma)
        self.trace_loss_weight = float(trace_loss_weight)
        # The claim-level and trace-level tasks have OPPOSITE class balance
        # (e.g. ~85% of claims are "correct" but only ~27% of traces are), so
        # balanced_bce must weight them separately. Fall back to a shared
        # pos_weight when per-level values are not supplied.
        self.claim_pos_weight = float(claim_pos_weight if claim_pos_weight is not None else pos_weight)
        self.trace_pos_weight = float(trace_pos_weight if trace_pos_weight is not None else pos_weight)

    def forward(
        self,
        features: torch.Tensor,                 # (B, L, D)
        attention_mask: torch.Tensor,           # (B, L)
        claim_masks: Optional[List[torch.Tensor]] = None,
        claim_types: Optional[List[torch.Tensor]] = None,
        claim_labels: Optional[List[torch.Tensor]] = None,   # list of (C_i,)
        trace_labels: Optional[torch.Tensor] = None,          # (B,)
        **kwargs,
    ) -> UQOutput:
        out: HeadOutput = self.head(features, attention_mask, claim_masks, claim_types)

        claim_loss = None
        trace_loss = None
        total = None

        if claim_labels is not None:
            # Gather valid (unpadded) per-claim logits and align with labels.
            flat_logits = []
            flat_labels = []
            for i, labels_i in enumerate(claim_labels):
                n = int(labels_i.shape[0])
                if n == 0:
                    continue
                logits_i = out.claim_logits[i, :n]                 # (n, C)
                flat_logits.append(logits_i.reshape(-1, self.num_classes))
                flat_labels.append(labels_i.to(features.device).float())
            if flat_logits:
                logits_cat = torch.cat(flat_logits, dim=0).squeeze(-1)
                labels_cat = torch.cat(flat_labels, dim=0)
                claim_loss = _binary_loss(
                    logits_cat, labels_cat, self.loss_type, self.claim_pos_weight, self.focal_gamma
                )

        if (
            trace_labels is not None
            and out.trace_logit is not None
            and self.trace_loss_weight > 0
        ):
            trace_logit = out.trace_logit.reshape(-1, self.num_classes).squeeze(-1)
            trace_loss = _binary_loss(
                trace_logit, trace_labels.to(features.device).float(),
                self.loss_type, self.trace_pos_weight, self.focal_gamma,
            )

        if claim_loss is not None:
            total = claim_loss
            if trace_loss is not None:
                total = total + self.trace_loss_weight * trace_loss
        elif trace_loss is not None:
            total = self.trace_loss_weight * trace_loss

        return UQOutput(
            loss=total,
            claim_logits=out.claim_logits,
            claim_mask=out.claim_mask,
            trace_logit=out.trace_logit,
            claim_loss=claim_loss,
            trace_loss=trace_loss,
        )

    def trainable_params(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_params())


def build_feature_extractor(
    hidden_state_layers: str = "-1",
    hidden_size: int = 4096,
    top_n_probs: int = 4,
    temperature: float = 1.0,
    attention_layers: str = "",
    attention_heads: str = "all",
    attn_history_sz: int = 3,
    pool_attention: bool = False,
    num_layers: int = 32,
    num_heads: int = 32,
) -> CombinedExtractor:
    """Build the frozen feature extractor used at inference (Phase 1 caches offline)."""
    layer_nums = None if hidden_state_layers == "all" else [int(x) for x in hidden_state_layers.split(",")]
    extractors = [
        HiddenStateExtractor(layer_nums=layer_nums, hidden_size=hidden_size),
        TokenProbExtractor(top_n=top_n_probs, temperature=temperature),
    ]
    if attention_layers:
        if str(attention_layers).strip().lower() == "all":
            attn_layers = list(range(num_layers))
        else:
            attn_layers = [int(x) for x in attention_layers.split(",")]
        extractors.append(
            AttentionExtractor(
                layer_nums=attn_layers, head_nums=attention_heads,
                attn_history_sz=attn_history_sz, pool=pool_attention,
                num_layers=num_layers, num_heads=num_heads,
            )
        )
    return CombinedExtractor(extractors)
