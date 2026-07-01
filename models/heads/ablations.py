"""
ablations.py - SpatialMind ablation heads.

Each variant toggles one structural component of the full SpatialMind head so we
can measure its contribution. This matches the paper's cumulative ablation:

    spatialmind_base   : all structural modules off (local claim scorer only)
    + cross-claim       : add the BiLSTM cross-claim integration
    + typed claim       : add claim-type embeddings
    + scope-aware       : add span-position statistics
    + reliability bank   : add the reliability pattern bank  == full SpatialMind

We also expose leave-one-out variants (no_cross / no_type / no_scope / no_bank)
for a complementary ablation. All variants reuse SpatialMindHead with flags, so
the ablation is exact (identical code path minus the toggled module).
"""

from __future__ import annotations

from models.heads.spatialmind_head import SpatialMindHead


class _AblationHead(SpatialMindHead):
    """Ablation with a fixed flag profile. `emits_trace_logit` stays on so the
    multi-task trace readout is always available for sample-level scoring."""

    _flags: dict = {}

    def __init__(self, feature_dim, num_classes=1, **kwargs):
        merged = {**self._flags}
        # Explicit kwargs (from config) never override the ablation's identity flags.
        for k, v in kwargs.items():
            if k not in ("use_reliability_bank", "use_cross_claim", "use_scope", "use_type"):
                merged[k] = v
        super().__init__(feature_dim, num_classes, **merged)


# ---- Cumulative ablation (progressively add modules) ---------------------- #
class AblBase(_AblationHead):
    """No structural modules: local claim scorer + trace head only."""
    _flags = dict(use_cross_claim=False, use_type=False, use_scope=False, use_reliability_bank=False)


class AblCross(_AblationHead):
    """+ cross-claim BiLSTM integration."""
    _flags = dict(use_cross_claim=True, use_type=False, use_scope=False, use_reliability_bank=False)


class AblType(_AblationHead):
    """+ typed-claim embeddings."""
    _flags = dict(use_cross_claim=True, use_type=True, use_scope=False, use_reliability_bank=False)


class AblScope(_AblationHead):
    """+ scope-aware span statistics."""
    _flags = dict(use_cross_claim=True, use_type=True, use_scope=True, use_reliability_bank=False)


# ---- Leave-one-out ablation (remove one module from the full model) ------- #
class AblNoCross(_AblationHead):
    _flags = dict(use_cross_claim=False, use_type=True, use_scope=True, use_reliability_bank=True)


class AblNoType(_AblationHead):
    _flags = dict(use_cross_claim=True, use_type=False, use_scope=True, use_reliability_bank=True)


class AblNoScope(_AblationHead):
    _flags = dict(use_cross_claim=True, use_type=True, use_scope=False, use_reliability_bank=True)


class AblNoBank(_AblationHead):
    _flags = dict(use_cross_claim=True, use_type=True, use_scope=True, use_reliability_bank=False)
