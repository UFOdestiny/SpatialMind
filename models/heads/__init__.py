"""Supervised claim-level UQ head registry.

Every head implements the `UncertaintyHeadBase` contract and returns a
`HeadOutput` (per-claim logits + optional learned trace logit). Unsupervised
baselines live in `models.unsup_heads`.
"""

from models.heads.base import HeadOutput, UncertaintyHeadBase
from models.heads.spatialmind_head import (
    ConstraintOnlyHead,
    ConstraintNoConflictHead,
    ConstraintNoContextHead,
    ConstraintNoEntailmentHead,
    ConstraintNoRepairHead,
    ConstraintSpatialMindHead,
    NeuralSpatialMindHead,
    SpatialMindHead,
)
from models.heads.baselines import (
    CNNHead,
    FactoscopeHead,
    GatedMLPHead,
    LinearHead,
    LookbackLensHead,
    LUHLightHead,
    MLPHead,
    SaplmaHead,
    UHead,
)
from models.heads.ablations import (
    AblBase,
    AblCross,
    AblNoBank,
    AblNoCross,
    AblNoScope,
    AblNoType,
    AblScope,
    AblType,
)

HEAD_REGISTRY = {
    # Our method
    "spatialmind": ConstraintSpatialMindHead,
    "constraint_spatialmind": ConstraintSpatialMindHead,
    "constraint_only": ConstraintOnlyHead,
    "spatialmind_neural": NeuralSpatialMindHead,
    "constraint_no_context": ConstraintNoContextHead,
    "constraint_no_conflict": ConstraintNoConflictHead,
    "constraint_no_entailment": ConstraintNoEntailmentHead,
    "constraint_no_repair": ConstraintNoRepairHead,
    "uq": ConstraintSpatialMindHead,

    # Supervised baselines
    "saplma": SaplmaHead,
    "factoscope": FactoscopeHead,
    "lookback_lens": LookbackLensHead,
    "lookbacklens": LookbackLensHead,
    "uhead": UHead,
    "luh_head": UHead,
    "luh_light": LUHLightHead,
    "linear": LinearHead,
    "mlp": MLPHead,
    "gated_mlp": GatedMLPHead,
    "cnn": CNNHead,

    # SpatialMind ablations (cumulative)
    "abl_base": AblBase,
    "abl_cross": AblCross,
    "abl_type": AblType,
    "abl_scope": AblScope,
    # SpatialMind ablations (leave-one-out)
    "abl_no_cross": AblNoCross,
    "abl_no_type": AblNoType,
    "abl_no_scope": AblNoScope,
    "abl_no_bank": AblNoBank,
}

# Head shape kwargs that build_head forwards when the head accepts them.
_SHAPE_KEYS = ("head_dim", "n_layers", "n_heads", "dropout", "max_seq_len")


def build_head(head_type: str, feature_dim: int, num_classes: int = 1, **kwargs):
    """Factory: build a head by name.

    Unknown kwargs are tolerated (heads accept **kwargs), so callers can pass a
    superset of shape hyperparameters uniformly.
    """
    key = head_type.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head type {head_type!r}. Available: {sorted(HEAD_REGISTRY)}"
        )
    return HEAD_REGISTRY[key](feature_dim=feature_dim, num_classes=num_classes, **kwargs)


def is_registered(head_type: str) -> bool:
    return head_type.lower() in HEAD_REGISTRY
