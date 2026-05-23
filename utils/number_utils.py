"""Numeric conversion helpers shared across scripts."""

from __future__ import annotations

from typing import Any, Optional


def to_float(value: Any) -> Optional[float]:
    """Parse value into float, returning None when parsing fails."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(*values: Any, default: float = 0.0) -> float:
    """Return first parseable float in values, otherwise default."""
    for value in values:
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return default
