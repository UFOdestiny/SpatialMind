"""Explicit spatial-constraint analysis used by SpatialMind.

The package converts natural-language relational claims into a small canonical
relation algebra, checks whether the induced layout is satisfiable, and exposes
claim/trace diagnostics as fixed-width numeric features for UQ heads.
"""

from spatial_constraints.analysis import (
    CLAIM_CONSTRAINT_DIM,
    CLAIM_FEATURE_NAMES,
    TRACE_CONSTRAINT_DIM,
    TRACE_FEATURE_NAMES,
    ConstraintAnalysis,
    analyze_trace,
)
from spatial_constraints.parser import CanonicalRelation, parse_relations

__all__ = [
    "CLAIM_CONSTRAINT_DIM",
    "CLAIM_FEATURE_NAMES",
    "TRACE_CONSTRAINT_DIM",
    "TRACE_FEATURE_NAMES",
    "CanonicalRelation",
    "ConstraintAnalysis",
    "analyze_trace",
    "parse_relations",
]
