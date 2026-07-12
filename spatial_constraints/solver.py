"""Fast satisfiability and entailment checks for canonical spatial relations."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple

from spatial_constraints.parser import CanonicalRelation
from spatial_constraints.relations import (
    CONTRADICTORY_RELATIONS,
    DIRECTION_VECTORS,
    INVERSE_RELATION,
    SYMMETRIC_RELATIONS,
    TRANSITIVE_RELATIONS,
)


def _axis_edges(relations: Sequence[CanonicalRelation], axis: int):
    """Return difference-constraint edges (u, v, c): value[v] <= value[u] + c."""
    edges = []
    nodes = set()
    for r in relations:
        if r.relation not in DIRECTION_VECTORS:
            continue
        nodes.update((r.subject, r.object))
        sign = DIRECTION_VECTORS[r.relation][axis]
        if sign > 0:       # subject - object >= 1 => object <= subject - 1
            edges.append((r.subject, r.object, -1.0))
        elif sign < 0:     # subject - object <= -1 => subject <= object - 1
            edges.append((r.object, r.subject, -1.0))
        else:              # equality
            edges.append((r.subject, r.object, 0.0))
            edges.append((r.object, r.subject, 0.0))
    return nodes, edges


def _difference_constraints_feasible(nodes: Set[str], edges: Sequence[Tuple[str, str, float]]) -> bool:
    if not nodes:
        return True
    dist = {n: 0.0 for n in nodes}  # equivalent to a zero-weight super-source
    for i in range(len(nodes)):
        changed = False
        for u, v, c in edges:
            if dist[v] > dist[u] + c:
                dist[v] = dist[u] + c
                changed = True
                if i == len(nodes) - 1:
                    return False
        if not changed:
            break
    return True


def _symbolic_closure(relations: Sequence[CanonicalRelation]) -> Set[Tuple[str, str, str]]:
    facts: Set[Tuple[str, str, str]] = set()
    for r in relations:
        facts.add((r.subject, r.relation, r.object))
        inv = INVERSE_RELATION.get(r.relation)
        if inv:
            facts.add((r.object, inv, r.subject))
        if r.relation in SYMMETRIC_RELATIONS:
            facts.add((r.object, r.relation, r.subject))

    changed = True
    while changed:
        changed = False
        current = list(facts)
        by_left = {}
        for s, rel, o in current:
            if rel in TRANSITIVE_RELATIONS:
                by_left.setdefault((rel, s), set()).add(o)
        for s, rel, mid in current:
            if rel not in TRANSITIVE_RELATIONS:
                continue
            for o in by_left.get((rel, mid), ()):
                fact = (s, rel, o)
                if fact not in facts:
                    facts.add(fact)
                    inv = INVERSE_RELATION.get(rel)
                    if inv:
                        facts.add((o, inv, s))
                    changed = True
        # Path endpoints inherit containing regions: starting from Utrecht also
        # entails starting from the Netherlands when Utrecht is inside it.
        current = list(facts)
        inside = {(s, o) for s, rel, o in current if rel == "inside"}
        for traveler, rel, loc in current:
            if rel not in {"starts-at", "ends-at"}:
                continue
            for child, region in inside:
                if child != loc:
                    continue
                fact = (traveler, rel, region)
                if fact not in facts:
                    facts.add(fact)
                    changed = True
    return facts


def _symbolic_feasible(relations: Sequence[CanonicalRelation]) -> bool:
    facts = _symbolic_closure(relations)
    for s, rel, o in facts:
        for bad in CONTRADICTORY_RELATIONS.get(rel, ()):
            if (s, bad, o) in facts:
                return False
        if rel in {"inside", "contains", "not-inside", "not-contains"} and s == o:
            return False
    # bAbI-style current location facts are functional after preprocessing.
    locations = {}
    for s, rel, o in facts:
        if rel == "at":
            locations.setdefault(s, set()).add(o)
    return all(len(v) <= 1 for v in locations.values())


def feasible(relations: Sequence[CanonicalRelation]) -> bool:
    if not _symbolic_feasible(relations):
        return False
    for axis in (0, 1):
        nodes, edges = _axis_edges(relations, axis)
        if not _difference_constraints_feasible(nodes, edges):
            return False
    return True


def relation_status(base: Sequence[CanonicalRelation], candidate: CanonicalRelation) -> str:
    """Return entailed, contradicted, or unknown under the current constraints."""
    if not feasible(list(base) + [candidate]):
        return "contradicted"

    if candidate.relation in DIRECTION_VECTORS:
        feasible_alternatives = 0
        for alt in DIRECTION_VECTORS:
            probe = CanonicalRelation(candidate.subject, alt, candidate.object)
            if feasible(list(base) + [probe]):
                feasible_alternatives += 1
                if feasible_alternatives > 1:
                    return "unknown"
        return "entailed"

    facts = _symbolic_closure(base)
    if (candidate.subject, candidate.relation, candidate.object) in facts:
        return "entailed"
    for bad in CONTRADICTORY_RELATIONS.get(candidate.relation, ()):
        if (candidate.subject, bad, candidate.object) in facts:
            return "contradicted"
    return "unknown"


def minimum_repair_cost(
    fixed: Sequence[CanonicalRelation],
    removable: Sequence[CanonicalRelation],
    candidate: CanonicalRelation,
    max_remove: int = 3,
) -> int:
    """Small exact repair search over prior generated claims; context stays fixed."""
    all_rel = list(fixed) + list(removable) + [candidate]
    if feasible(all_rel):
        return 0
    n = len(removable)
    for k in range(1, min(max_remove, n) + 1):
        for removed in combinations(range(n), k):
            removed = set(removed)
            keep = [r for i, r in enumerate(removable) if i not in removed]
            if feasible(list(fixed) + keep + [candidate]):
                return k
    return max_remove + 1
