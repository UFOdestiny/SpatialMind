"""
combined.py - Combine multiple feature extractors by concatenation.
"""

import torch
import torch.nn as nn
from typing import List


class CombinedExtractor(nn.Module):
    """Concatenate outputs of multiple feature extractors."""

    def __init__(self, extractors: List[nn.Module]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def feature_dim(self) -> int:
        return sum(e.feature_dim() for e in self.extractors)

    def output_attention(self) -> bool:
        """Whether any sub-extractor requires attention outputs from the LLM."""
        return any(
            getattr(ext, "output_attention", lambda: False)()
            for ext in self.extractors
        )

    def forward(
        self,
        hidden_states: tuple = None,
        logits: torch.Tensor = None,
        attentions: tuple = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run all extractors and concatenate.

        Args:
            hidden_states: tuple of layer hidden states
            logits: LLM logits (batch, seq_len, vocab_size)
            attentions: tuple of per-layer attention tensors
            attention_mask: (batch, seq_len)

        Returns:
            (batch, seq_len, total_feature_dim)
        """
        from models.features.hidden_states import HiddenStateExtractor
        from models.features.token_probs import TokenProbExtractor
        from models.features.attention import AttentionExtractor

        features = []
        for ext in self.extractors:
            if isinstance(ext, HiddenStateExtractor) and hidden_states is not None:
                features.append(ext(hidden_states))
            elif isinstance(ext, TokenProbExtractor) and logits is not None:
                features.append(ext(logits))
            elif isinstance(ext, AttentionExtractor) and attentions is not None:
                features.append(ext(attentions, attention_mask))

        if not features:
            raise ValueError("No features extracted. Check extractor inputs.")

        return torch.cat(features, dim=-1)
