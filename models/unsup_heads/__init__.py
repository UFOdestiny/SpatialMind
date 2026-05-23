"""Unsupervised uncertainty estimators built on cached generation traces."""

from models.unsup_heads.base import UnsupervisedEstimator
from models.unsup_heads.random_baseline import RandomBaseline
from models.unsup_heads.mcp import MCPEstimator
from models.unsup_heads.perplexity import PerplexityEstimator
from models.unsup_heads.token_entropy import TokenEntropyEstimator
from models.unsup_heads.ccp import CCPEstimator

UNSUPERVISED_HEAD_REGISTRY = {
    "random": RandomBaseline,
    "mcp": MCPEstimator,
    "perplexity": PerplexityEstimator,
    "token_entropy": TokenEntropyEstimator,
    "ccp": CCPEstimator,
}

UNSUPERVISED_HEAD_ALIASES = {
    "mean_token_entropy": "token_entropy",
}

# Backward-compatible alias used by evaluation scripts.
BASELINE_REGISTRY = UNSUPERVISED_HEAD_REGISTRY
