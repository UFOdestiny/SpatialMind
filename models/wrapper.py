"""
wrapper.py - Model wrappers for SpatialMind.

CachedFeatureModel: Phase 2 — trains head on pre-extracted features (no LLM).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from models.features.hidden_states import HiddenStateExtractor
from models.features.token_probs import TokenProbExtractor
from models.features.attention import AttentionExtractor
from models.features.combined import CombinedExtractor

@dataclass
class UQModelOutput(ModelOutput):
    """HuggingFace Trainer-compatible output for UQ heads."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CachedFeatureModel(nn.Module):
    """Phase 2 model: trains head on pre-extracted features (no LLM needed).

    Takes pre-extracted feature tensors directly, skipping the LLM forward pass.
    Uses BCEWithLogitsLoss for binary classification (num_classes=1).
    """

    def __init__(
        self,
        head: nn.Module,
        num_classes: int = 1,
        loss_type: str = "bce",
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.head = head
        self.num_classes = num_classes
        self.loss_type = (loss_type or "bce").lower()
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)

    def _binary_loss(self, logits_1d: torch.Tensor, labels_1d: torch.Tensor) -> torch.Tensor:
        labels_1d = labels_1d.float()
        if self.loss_type == "balanced_bce":
            pos_weight_t = torch.tensor(
                [max(self.pos_weight, 1e-8)],
                device=logits_1d.device,
                dtype=logits_1d.dtype,
            )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)(logits_1d, labels_1d)

        if self.loss_type == "focal":
            # Binary focal loss over logits.
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits_1d,
                labels_1d,
                reduction="none",
            )
            probs = torch.sigmoid(logits_1d)
            pt = torch.where(labels_1d > 0.5, probs, 1.0 - probs)
            focal_weight = (1.0 - pt).pow(self.focal_gamma)
            return (focal_weight * bce).mean()

        return nn.BCEWithLogitsLoss()(logits_1d, labels_1d)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        claim_masks=None,
        claim_types=None,
        claim_labels=None,
        **kwargs,
    ) -> UQModelOutput:
        if claim_masks is not None and claim_labels is not None:
            return self._forward_claim_level(features, attention_mask, claim_masks, claim_types, claim_labels)

        logits = self.head(features, attention_mask)  # (batch, num_classes)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss = self._binary_loss(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return UQModelOutput(loss=loss, logits=logits)

    def _forward_claim_level(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_masks,
        claim_types,
        claim_labels,
    ) -> UQModelOutput:
        if claim_types is None:
            claim_types = [
                torch.full((m.shape[0],), 2, dtype=torch.long, device=features.device)
                for m in claim_masks
            ]
        if getattr(self.head, "supports_claim_inputs", False):
            if hasattr(self.head, "forward_claim"):
                logits = self.head.forward_claim(
                    features=features,
                    attention_mask=attention_mask,
                    claim_masks=claim_masks,
                    claim_types=claim_types,
                )
            else:
                logits = self.head(
                    features=features,
                    attention_mask=attention_mask,
                    claim_masks=claim_masks,
                    claim_types=claim_types,
                )
            
            # Flatten labels
            labels_flat = torch.cat([x.to(features.device).float() for x in claim_labels], dim=0)
            
            # Flatten logits: extract valid claims from padded (B, max_claims, C) tensor
            # logits may be (B, max_claims, C) or (total_claims, C) depending on head impl
            if logits.dim() == 3:
                # (B, max_claims, num_classes) -> extract valid claims only
                logits_list = []
                for i in range(logits.shape[0]):
                    num_claims = len(claim_labels[i])
                    logits_list.append(logits[i, :num_claims, :])  # skip padding
                logits_flat = torch.cat(logits_list, dim=0)  # (total_claims, num_classes)
            elif logits.dim() == 2:
                # Already flat: (total_claims, num_classes)
                logits_flat = logits
            else:
                logits_flat = logits.view(-1, logits.shape[-1])
            
            if self.num_classes == 1:
                loss = self._binary_loss(logits_flat.squeeze(-1), labels_flat)
            else:
                loss = nn.CrossEntropyLoss()(logits_flat, labels_flat.long())
            return UQModelOutput(loss=loss, logits=logits_flat)

        pooled_claim_features = []
        flat_labels = []

        batch_size = features.shape[0]
        for i in range(batch_size):
            token_features = features[i]  # (L, D)
            token_mask = attention_mask[i].float()  # (L,)
            sample_claim_masks = claim_masks[i].to(features.device).float()  # (C, L)
            sample_claim_labels = claim_labels[i].to(features.device).float()  # (C,)

            # Intersect claim mask with valid tokens only.
            sample_claim_masks = sample_claim_masks * token_mask.unsqueeze(0)
            denom = sample_claim_masks.sum(dim=1, keepdim=True).clamp(min=1.0)
            claim_vecs = (sample_claim_masks @ token_features) / denom  # (C, D)

            pooled_claim_features.append(claim_vecs)
            flat_labels.append(sample_claim_labels)

        claim_features = torch.cat(pooled_claim_features, dim=0).unsqueeze(1)  # (N_claim, 1, D)
        claim_attn_mask = torch.ones(
            claim_features.shape[0], 1, device=claim_features.device, dtype=attention_mask.dtype
        )
        logits = self.head(claim_features, claim_attn_mask)  # (N_claim, num_classes)

        labels_flat = torch.cat(flat_labels, dim=0)
        if self.num_classes == 1:
            loss = self._binary_loss(logits.squeeze(-1), labels_flat)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels_flat.long())

        return UQModelOutput(loss=loss, logits=logits)

    def get_trainable_params(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        _ = exclude_embeddings
        return sum(p.numel() for p in self.head.parameters() if p.requires_grad)


def build_feature_extractor(
    hidden_state_layers: str = "-1",
    hidden_size: int = 4096,
    top_n_probs: int = 4,
    temperature: float = 1.0,
    attention_layers: str = "",
    attention_heads: str = "all",
    attn_history_sz: int = 10,
    pool_attention: bool = True,
    num_layers: int = 32,
    num_heads: int = 32,
) -> CombinedExtractor:
    """Build the combined feature extractor."""
    if hidden_state_layers == "all":
        layer_nums = None
    else:
        layer_nums = [int(x.strip()) for x in hidden_state_layers.split(",")]

    extractors = [
        HiddenStateExtractor(layer_nums=layer_nums, hidden_size=hidden_size),
        TokenProbExtractor(top_n=top_n_probs, temperature=temperature),
    ]

    if attention_layers:
        if str(attention_layers).strip().lower() == "all":
            attn_layer_nums = list(range(num_layers))
        else:
            attn_layer_nums = [int(x.strip()) for x in attention_layers.split(",")]
        extractors.append(
            AttentionExtractor(
                layer_nums=attn_layer_nums,
                head_nums=attention_heads,
                attn_history_sz=attn_history_sz,
                pool=pool_attention,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        )

    return CombinedExtractor(extractors)
