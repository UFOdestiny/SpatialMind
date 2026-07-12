"""Canonical relation ontology and relation-algebra metadata."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# (horizontal sign, vertical sign), where +x=right and +y=above.
DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = {
    "upper-right": (1, 1),
    "lower-right": (1, -1),
    "upper-left": (-1, 1),
    "lower-left": (-1, -1),
    "above": (0, 1),
    "left": (-1, 0),
    "below": (0, -1),
    "right": (1, 0),
    "overlap": (0, 0),
}

RELATION_ALIASES = {
    "north east": "upper-right", "northeast": "upper-right", "north-east": "upper-right",
    "upper right": "upper-right", "above right": "upper-right", "right above": "upper-right",
    "south east": "lower-right", "southeast": "lower-right", "south-east": "lower-right",
    "lower right": "lower-right", "below right": "lower-right", "right below": "lower-right",
    "north west": "upper-left", "northwest": "upper-left", "north-west": "upper-left",
    "upper left": "upper-left", "above left": "upper-left", "left above": "upper-left",
    "south west": "lower-left", "southwest": "lower-left", "south-west": "lower-left",
    "lower left": "lower-left", "below left": "lower-left", "left below": "lower-left",
    "north": "above", "over": "above", "on top of": "above", "above": "above",
    "south": "below", "under": "below", "below": "below",
    "west": "left", "to the left of": "left", "left of": "left", "left": "left",
    "east": "right", "to the right of": "right", "right of": "right", "right": "right",
    "same position as": "overlap", "overlaps": "overlap", "overlap": "overlap",
    "in": "inside", "inside": "inside", "within": "inside", "located in": "inside",
    "is in": "inside", "is inside": "inside", "is within": "inside",
    "contains": "contains", "contain": "contains", "has": "contains", "holds": "contains",
    "near to": "near", "close to": "near", "near": "near", "close": "near",
    "far from": "far", "far": "far",
    "touches": "touching", "touching": "touching", "in contact with": "touching",
    "disconnected from": "disconnected", "disconnected": "disconnected",
    "separate from": "disconnected",
    "at": "at", "went to": "at", "moved to": "at", "travelled to": "at",
    "traveled to": "at", "journeyed to": "at", "returned to": "at",
    "went back to": "at",
    "not in": "not-inside", "outside": "not-inside",
}

INVERSE_RELATION = {
    "upper-right": "lower-left", "lower-left": "upper-right",
    "upper-left": "lower-right", "lower-right": "upper-left",
    "above": "below", "below": "above", "left": "right", "right": "left",
    "overlap": "overlap", "inside": "contains", "contains": "inside",
    "near": "near", "far": "far", "touching": "touching",
    "disconnected": "disconnected", "at": "contains", "not-inside": "not-contains",
    "not-contains": "not-inside",
}

SYMMETRIC_RELATIONS = {"near", "far", "touching", "disconnected", "overlap"}
TRANSITIVE_RELATIONS = {"inside", "contains"}

CONTRADICTORY_RELATIONS = {
    "inside": {"not-inside", "disconnected"},
    "not-inside": {"inside"},
    "contains": {"not-contains", "disconnected"},
    "not-contains": {"contains"},
    "near": {"far"}, "far": {"near"},
    "touching": {"disconnected"}, "disconnected": {"touching", "inside", "contains"},
}

RELATION_FAMILIES = (
    "directional", "containment", "distance", "topology", "location", "path", "other"
)


def canonicalize_relation(text: str) -> Optional[str]:
    t = " ".join((text or "").lower().replace("_", " ").strip(" .,:;!?\"'").split())
    if t in DIRECTION_VECTORS or t in INVERSE_RELATION:
        return t
    return RELATION_ALIASES.get(t)


def relation_family(relation: str) -> str:
    if relation in DIRECTION_VECTORS:
        return "directional"
    if relation in {"inside", "contains", "not-inside", "not-contains"}:
        return "containment"
    if relation in {"near", "far"}:
        return "distance"
    if relation in {"touching", "disconnected"}:
        return "topology"
    if relation == "at":
        return "location"
    if relation in {"starts-at", "ends-at", "towards"}:
        return "path"
    return "other"
