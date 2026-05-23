"""
hidden_states.py - Extract hidden state features from LLM layers.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class HiddenStateExtractor(nn.Module):
    """Extract and concatenate hidden states from specified LLM layers.

    During forward pass, selects hidden states from the specified layers
    and concatenates them along the feature dimension.
    """

    def __init__(self, layer_nums: Optional[List[int]] = None, hidden_size: int = 4096):
        super().__init__()
        self.layer_nums = layer_nums or [-1]
        self.hidden_size = hidden_size

    def feature_dim(self) -> int:
        return self.hidden_size * len(self.layer_nums)

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        """Extract hidden states from specified layers.

        Args:
            hidden_states: tuple of (n_layers+1,) tensors, each (batch, seq_len, hidden_size).
                           Index 0 is the embedding output, indices 1..n_layers are layer outputs.

        Returns:
            Tensor of shape (batch, seq_len, feature_dim).
        """
        n_layers = len(hidden_states) - 1  # exclude embedding layer

        selected = []
        for layer_idx in self.layer_nums:
            # Resolve negative indices
            if layer_idx < 0:
                resolved = n_layers + 1 + layer_idx
            else:
                resolved = layer_idx + 1  # +1 because index 0 is embedding

            resolved = max(0, min(resolved, n_layers))
            selected.append(hidden_states[resolved])

        # Concatenate along feature dimension
        return torch.cat(selected, dim=-1)
