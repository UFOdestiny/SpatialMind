"""Shared datatypes for generation engines."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class GenerationResult:
    """Output of a single batch generation."""
    generated_texts: List[str] = field(default_factory=list)
    generated_token_ids: Optional[torch.Tensor] = None  # (batch, gen_len)
    features: Optional[torch.Tensor] = None        # (batch, gen_len, feat_dim)
    top_probs: Optional[torch.Tensor] = None       # (batch, gen_len, top_n)
    log_likelihoods: Optional[torch.Tensor] = None  # (batch, gen_len)
