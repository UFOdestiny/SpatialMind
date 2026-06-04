from models.heads.base import UncertaintyHeadBase
from models.heads.uq_head import UQHead

HEAD_REGISTRY = {
    "uq": UQHead,
}


def build_head(head_type: str, feature_dim: int, num_classes: int = 1, **kwargs):
    """Factory: build a head by name."""
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type '{head_type}'. Available: {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[head_type](feature_dim=feature_dim, num_classes=num_classes, **kwargs)
