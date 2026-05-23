"""
attention.py - Attention weight feature extractor.

Extracts attention patterns from specified LLM layers/heads.
Following LUH's FeatureExtractorBasicAttention pattern:
  - Select attention weights from specified layers and heads
  - Use an attention history window (attn_history_sz) for lookback
  - Optionally pool (max) across layers
  - Output shape: (batch, seq_len, feature_dim)
"""

import torch
import torch.nn as nn


class AttentionExtractor(nn.Module):
    """Extract attention weight features from specified LLM layers/heads."""

    def __init__(
        self,
        layer_nums: list,
        head_nums: str = "all",
        attn_history_sz: int = 10,
        pool: bool = True,
        num_layers: int = 32,
        num_heads: int = 32,
    ):
        super().__init__()
        # Resolve negative layer indices
        self._layer_nums = [
            (l % num_layers) if l < 0 else l for l in layer_nums
        ]
        self._num_heads = num_heads
        self._attn_history_sz = attn_history_sz
        self._pool = pool

        # Resolve head selection per layer
        if head_nums == "all":
            self._head_nums = {
                l: list(range(num_heads)) for l in self._layer_nums
            }
        else:
            heads = [int(h) for h in head_nums.split(",")]
            self._head_nums = {l: heads for l in self._layer_nums}

        n_selected_heads = len(list(self._head_nums.values())[0])
        if pool:
            self._input_size = n_selected_heads
        else:
            self._input_size = sum(
                len(h) for h in self._head_nums.values()
            )

    def feature_dim(self) -> int:
        return self._input_size * self._attn_history_sz

    def output_attention(self) -> bool:
        return True

    def forward(
        self,
        attentions: tuple,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract attention features.

        Args:
            attentions: tuple of per-layer attention tensors.
                Training:    (n_layers,) each (batch, n_heads, seq_len, seq_len)
                Generation:  list of per-token tuples
            attention_mask: (batch, seq_len), 1=valid, 0=padding

        Returns:
            (batch, seq_len, feature_dim)
        """
        # attentions is a tuple of tensors: (n_layers,) x (batch, heads, seq, seq)
        # We process each token position to extract lookback attention features.

        batch_size = attentions[0].shape[0]
        seq_len = attentions[0].shape[2]

        all_features = []  # will collect (batch, attn_hist, heads, layers) per position

        for pos in range(seq_len):
            pos_features = []  # one per layer
            for layer_num in self._layer_nums:
                # (batch, n_heads, seq_len, seq_len) -> select position
                cur_attn = attentions[layer_num][:, :, pos, :pos + 1]
                # cur_attn: (batch, n_heads, 1, pos+1) ... but we need full row
                cur_attn = attentions[layer_num][:, :, pos, :]  # (batch, n_heads, seq_len)

                # Build lookback indices: most recent attn_history_sz positions
                indices = torch.arange(
                    pos, pos - self._attn_history_sz, -1,
                    device=cur_attn.device,
                )
                valid = indices >= 0
                indices = indices.clamp(min=0)

                # Gather attention values at lookback positions
                # cur_attn: (batch, n_heads, seq_len) -> index last dim
                gathered = cur_attn[:, :, indices]  # (batch, n_heads, attn_hist)
                gathered = gathered.permute(0, 2, 1)  # (batch, attn_hist, n_heads)

                # Select specific heads
                head_indices = self._head_nums[layer_num]
                gathered = gathered[:, :, head_indices]  # (batch, attn_hist, selected_heads)

                # Zero out invalid lookback positions
                gathered[:, ~valid, :] = 0.0

                pos_features.append(gathered)

            # Stack across layers: (batch, attn_hist, heads, n_layers)
            stacked = torch.stack(pos_features, dim=-1)
            all_features.append(stacked)

        # Stack across positions: (batch, seq_len, attn_hist, heads, n_layers)
        all_features = torch.stack(all_features, dim=1)

        if self._pool:
            # Max-pool across layers: (batch, seq_len, attn_hist, heads)
            all_features = torch.amax(all_features, dim=-1)

        # Reshape to (batch, seq_len, feature_dim)
        return all_features.reshape(batch_size, seq_len, -1)
