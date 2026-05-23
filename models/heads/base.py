"""
base.py - Abstract base class for uncertainty / classification heads.
"""

import os
import yaml
import torch
import torch.nn as nn
from abc import abstractmethod


class UncertaintyHeadBase(nn.Module):
    """Base class for all classification heads attached to a frozen LLM.

    Each head takes extracted features (from hidden states / token probs)
    and produces classification logits for spatial relation prediction.
    """

    def __init__(self, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute classification logits from pooled features.

        Args:
            features:       (batch, seq_len, feature_dim)
            attention_mask:  (batch, seq_len), 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, num_classes)
        """
        raise NotImplementedError

    def pool_features(self, features: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Mean-pool features over sequence length, respecting the attention mask.

        Args:
            features:       (batch, seq_len, feature_dim)
            attention_mask:  (batch, seq_len)

        Returns:
            (batch, feature_dim)
        """
        if attention_mask is None:
            return features.mean(dim=1)

        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        # Use same dtype as features for efficient computation
        if mask.dtype != features.dtype:
            mask = mask.to(features.dtype)
        summed = (features * mask).sum(dim=1)        # (batch, feature_dim)
        lengths = mask.sum(dim=1).clamp(min=1)       # (batch, 1)
        return summed / lengths

    def save(self, output_dir: str, config: dict = None):
        """Save head weights and config."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "head_weights.pth"))
        if config:
            with open(os.path.join(output_dir, "head_config.yaml"), "w") as f:
                yaml.dump(config, f)

    def load(self, path: str):
        """Load head weights."""
        weights_path = os.path.join(path, "head_weights.pth")
        if os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))
        elif os.path.isfile(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"No head weights found at {path}")
