"""
base.py - Abstract base class and I/O contract for claim-level UQ heads.

Every head consumes the frozen per-token feature sequence of one batch of traces
plus per-claim token-span masks, and returns a `HeadOutput`:

    claim_logits : (B, max_claims, num_classes)   per-claim correctness logits,
                   right-padded with `pad_value` for traces with fewer claims.
    claim_mask   : (B, max_claims)                1 for real claims, 0 for padding.
    trace_logit  : (B, num_classes) or None       OPTIONAL learned trace-level
                   logit. Baselines leave this None (they are claim-only and are
                   aggregated to the trace level by the shared protocol). The
                   SpatialMind head returns a learned trace logit as part of its
                   multi-task contribution.

This uniform contract lets the trainer, wrapper, and evaluator treat every head
identically, and lets the fair claim->trace aggregation in models/aggregation.py
apply to all methods without special-casing.
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_LOGIT_VALUE = -100.0


@dataclass
class HeadOutput:
    """Uniform head output. See module docstring for shapes."""
    claim_logits: torch.Tensor          # (B, max_claims, C)
    claim_mask: torch.Tensor            # (B, max_claims), 1=real claim
    trace_logit: Optional[torch.Tensor] = None  # (B, C) or None


class UncertaintyHeadBase(nn.Module):
    """Base class for all claim-level UQ heads over a frozen backbone.

    Subclasses implement `forward_claims`, returning one logit tensor per claim
    for each trace. The base class packs these into the padded `HeadOutput`
    tensor and (optionally) exposes a learned trace logit via `forward_trace`.
    """

    #: Whether the head consumes claim masks (all heads here do).
    supports_claim_inputs: bool = True
    #: Whether the head emits its own learned trace-level logit.
    emits_trace_logit: bool = False
    #: Whether the head consumes explicit per-claim/per-trace constraint features.
    supports_constraint_inputs: bool = False

    def __init__(self, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    # ------------------------------------------------------------------ #
    # Subclass API
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward_claims(
        self,
        features: torch.Tensor,          # (B, L, D)
        attention_mask: torch.Tensor,    # (B, L)
        claim_masks: List[torch.Tensor], # list of (C_i, L) float masks
        claim_types: Optional[List[torch.Tensor]] = None,  # list of (C_i,) long
    ) -> List[torch.Tensor]:
        """Return a list of per-claim logit tensors, one (C_i, num_classes) per trace."""
        raise NotImplementedError

    def forward_trace(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_logits_per_trace: List[torch.Tensor],
        claim_masks: List[torch.Tensor],
        claim_types: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """Optional learned trace-level logit, shape (B, num_classes).

        Default: None (claim-only head). Override in heads that jointly model the
        trace, i.e. SpatialMind's multi-task head.
        """
        return None

    # ------------------------------------------------------------------ #
    # Unified forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        claim_masks: Optional[List[torch.Tensor]] = None,
        claim_types: Optional[List[torch.Tensor]] = None,
        claim_constraint_features: Optional[List[torch.Tensor]] = None,
        trace_constraint_features: Optional[torch.Tensor] = None,
    ) -> HeadOutput:
        bsz = features.shape[0]
        device = features.device
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[:2], dtype=torch.long, device=device)

        if claim_masks is None:
            # Degenerate path: treat the whole (valid) trace as a single claim.
            claim_masks = [attention_mask[i].unsqueeze(0).float() for i in range(bsz)]

        if self.supports_constraint_inputs:
            per_trace_logits = self.forward_claims(
                features, attention_mask, claim_masks, claim_types,
                claim_constraint_features=claim_constraint_features,
                trace_constraint_features=trace_constraint_features,
            )
        else:
            per_trace_logits = self.forward_claims(features, attention_mask, claim_masks, claim_types)
        claim_logits, claim_mask = self._pack(per_trace_logits, device)

        trace_logit = None
        if self.emits_trace_logit:
            if self.supports_constraint_inputs:
                trace_logit = self.forward_trace(
                    features, attention_mask, per_trace_logits, claim_masks, claim_types,
                    claim_constraint_features=claim_constraint_features,
                    trace_constraint_features=trace_constraint_features,
                )
            else:
                trace_logit = self.forward_trace(
                    features, attention_mask, per_trace_logits, claim_masks, claim_types
                )
        return HeadOutput(claim_logits=claim_logits, claim_mask=claim_mask, trace_logit=trace_logit)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _pack(self, per_trace_logits: List[torch.Tensor], device: torch.device):
        """Right-pad variable-length per-trace claim logits to (B, max_claims, C)."""
        if not per_trace_logits:
            return (
                torch.zeros(0, 0, self.num_classes, device=device),
                torch.zeros(0, 0, device=device),
            )
        counts = [x.shape[0] for x in per_trace_logits]
        max_claims = max(counts) if counts else 0
        max_claims = max(max_claims, 1)
        padded = []
        masks = []
        for x in per_trace_logits:
            n = x.shape[0]
            pad_n = max_claims - n
            padded.append(F.pad(x, (0, 0, 0, pad_n), value=PAD_LOGIT_VALUE))
            m = torch.zeros(max_claims, device=device)
            m[:n] = 1.0
            masks.append(m)
        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    @staticmethod
    def masked_mean(features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool (B, L, D) over valid tokens -> (B, D)."""
        mask = attention_mask.unsqueeze(-1).to(features.dtype)
        summed = (features * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    @staticmethod
    def claim_prototype(
        token_features: torch.Tensor,   # (L, D)
        claim_mask: torch.Tensor,       # (C, L)
        valid_mask: torch.Tensor,       # (L,)
    ) -> torch.Tensor:
        """Masked mean-pool token features within each claim span -> (C, D)."""
        cm = claim_mask * valid_mask.unsqueeze(0).to(claim_mask.dtype)
        denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (cm @ token_features) / denom

    def save(self, output_dir: str, config: Optional[dict] = None):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "head_weights.pth"))
        if config:
            with open(os.path.join(output_dir, "head_config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    def load(self, path: str):
        weights_path = os.path.join(path, "head_weights.pth")
        if os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))
        elif os.path.isfile(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"No head weights found at {path}")
