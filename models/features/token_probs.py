"""
token_probs.py - Extract top-N token probability features from LLM logits.
"""

import torch
import torch.nn as nn


class TokenProbExtractor(nn.Module):
    """Extract top-N softmax probabilities from LLM logits.

    At each token position, takes the top-N probability values
    (in log space) as features.
    """

    def __init__(self, top_n: int = 4, temperature: float = 1.0):
        super().__init__()
        self.top_n = top_n
        self.temperature = temperature

    def feature_dim(self) -> int:
        return self.top_n

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract top-N log probabilities.

        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            (batch, seq_len, top_n)
        """
        scaled = logits / max(self.temperature, 1e-8)
        log_probs = torch.log_softmax(scaled, dim=-1)
        top_values, _ = torch.topk(log_probs, self.top_n, dim=-1)
        return top_values
