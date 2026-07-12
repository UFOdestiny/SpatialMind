"""Trace-level explicit constraint analysis and fixed-width UQ features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from spatial_constraints.parser import CanonicalRelation, parse_query_entities, parse_relations
from spatial_constraints.relations import RELATION_FAMILIES, canonicalize_relation, relation_family
from spatial_constraints.solver import feasible, minimum_repair_cost, relation_status


CLAIM_FEATURE_NAMES = (
    "parseable",
    "parser_confidence",
    "known_relation",
    "context_available",
    "context_grounded_entities",
    "prefix_feasible",
    "feasible_with_claim",
    "entailed",
    "contradicted",
    "unknown",
    "first_conflict",
    "repair_cost_norm",
    "claim_position",
    "prior_claim_count_norm",
    "context_constraint_count_norm",
    "is_conclusion",
    "family_directional",
    "family_containment",
    "family_distance",
    "family_topology",
    "family_location",
    "family_path",
    "family_other",
)
TRACE_FEATURE_NAMES = (
    "context_available",
    "context_constraint_count_norm",
    "claim_count_norm",
    "parse_rate",
    "mean_parser_confidence",
    "entity_grounding_rate",
    "full_trace_feasible",
    "contradiction_rate",
    "entailment_rate",
    "unknown_rate",
    "first_conflict_position",
    "mean_repair_cost_norm",
    "max_repair_cost_norm",
    "conclusion_entailed",
    "conclusion_contradicted",
    "conclusion_unknown",
)
CLAIM_CONSTRAINT_DIM = len(CLAIM_FEATURE_NAMES)
TRACE_CONSTRAINT_DIM = len(TRACE_FEATURE_NAMES)


@dataclass
class ClaimConstraintResult:
    text: str
    parsed: List[CanonicalRelation]
    status: str
    prefix_feasible: bool
    feasible_with_claim: bool
    first_conflict: bool
    repair_cost: int
    features: List[float]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["parsed"] = [asdict(x) for x in self.parsed]
        return data


@dataclass
class ConstraintAnalysis:
    context_relations: List[CanonicalRelation]
    claims: List[ClaimConstraintResult]
    trace_features: List[float]
    full_trace_feasible: bool

    def to_dict(self) -> dict:
        return {
            "context_relations": [asdict(x) for x in self.context_relations],
            "claims": [x.to_dict() for x in self.claims],
            "trace_features": self.trace_features,
            "trace_feature_names": list(TRACE_FEATURE_NAMES),
            "claim_feature_names": list(CLAIM_FEATURE_NAMES),
            "full_trace_feasible": self.full_trace_feasible,
        }


def _latest_locations(relations: Sequence[CanonicalRelation]) -> List[CanonicalRelation]:
    """Keep only the latest bAbI state update for each subject."""
    last_at: Dict[str, CanonicalRelation] = {}
    out: List[CanonicalRelation] = []
    for r in relations:
        if r.relation == "at":
            last_at[r.subject] = r
        else:
            out.append(r)
    out.extend(last_at.values())
    return out


def _aggregate_status(statuses: Sequence[str]) -> str:
    if "contradicted" in statuses:
        return "contradicted"
    if statuses and all(x == "entailed" for x in statuses):
        return "entailed"
    return "unknown"


def analyze_trace(
    context: str,
    question: str,
    claims: Sequence[dict | str],
) -> ConstraintAnalysis:
    """Analyze ordered claims against trusted context and the prior claim prefix.

    Context relations are fixed. Earlier generated claims are treated as a
    removable prefix when computing repair cost, which separates contradiction
    with the source scene from error propagation inside the generated chain.
    """
    context_rel = _latest_locations(parse_relations(context or "", source="context"))
    context_entities = {x for r in context_rel for x in (r.subject, r.object)}
    prior_rel: List[CanonicalRelation] = []
    results: List[ClaimConstraintResult] = []
    first_conflict_seen = False
    n_claims = max(len(claims), 1)

    for i, claim in enumerate(claims):
        if isinstance(claim, dict):
            text = str(claim.get("text", ""))
            is_conclusion = int(claim.get("claim_type_id", 0)) == 2 or str(
                claim.get("claim_type", "")
            ).lower() == "conclusion"
        else:
            text = str(claim)
            is_conclusion = i == len(claims) - 1

        parsed = parse_relations(text, source="conclusion" if is_conclusion else "claim")
        predicted_nli_label = None
        if is_conclusion and not parsed:
            cleaned = text.lower().replace("the answer is", "").strip(" .,:;!?\"'")
            predicted_nli_label = next(
                (x for x in ("entailment", "contradiction", "neutral") if x in cleaned), None
            )
            relation = canonicalize_relation(cleaned)
            pair = parse_query_entities(question)
            if relation and pair:
                parsed = [CanonicalRelation(pair[0], relation, pair[1], text, 0.98, "conclusion")]
            elif predicted_nli_label:
                parsed = parse_relations(question, source="hypothesis")
        prefix = list(context_rel) + list(prior_rel)
        prefix_ok = feasible(prefix)
        with_claim_ok = feasible(prefix + parsed) if parsed else prefix_ok
        statuses = [relation_status(prefix, r) for r in parsed]
        status = _aggregate_status(statuses) if parsed else "unknown"
        if predicted_nli_label and parsed:
            expected = {
                "entailment": "entailed", "contradiction": "contradicted", "neutral": "unknown"
            }[predicted_nli_label]
            status = "entailed" if status == expected else "contradicted"
            with_claim_ok = status == "entailed"
        first_conflict = bool(parsed and prefix_ok and not with_claim_ok and not first_conflict_seen)
        if first_conflict:
            first_conflict_seen = True
        repair = 0
        if parsed and not with_claim_ok:
            repair = max(
                minimum_repair_cost(context_rel, prior_rel, r, max_remove=3) for r in parsed
            )

        grounded = 0.0
        if parsed:
            grounded = sum(
                float(r.subject in context_entities and r.object in context_entities) for r in parsed
            ) / len(parsed)
        family = relation_family(parsed[0].relation) if parsed else "other"
        confidence = sum(r.confidence for r in parsed) / len(parsed) if parsed else 0.0
        features = [
            float(bool(parsed)), confidence, float(bool(parsed)), float(bool(context_rel)), grounded,
            float(prefix_ok), float(with_claim_ok), float(status == "entailed"),
            float(status == "contradicted"), float(status == "unknown"), float(first_conflict),
            min(repair, 4) / 4.0, i / max(n_claims - 1, 1), min(len(prior_rel), 12) / 12.0,
            min(len(context_rel), 32) / 32.0, float(is_conclusion),
        ]
        features.extend(float(family == f) for f in RELATION_FAMILIES)
        assert len(features) == CLAIM_CONSTRAINT_DIM
        results.append(ClaimConstraintResult(
            text=text, parsed=parsed, status=status, prefix_feasible=prefix_ok,
            feasible_with_claim=with_claim_ok, first_conflict=first_conflict,
            repair_cost=repair, features=features,
        ))
        # An NLI verdict classifies the hypothesis; it does not assert the
        # hypothesis as another layout premise.
        if not predicted_nli_label:
            prior_rel.extend(parsed)

    parseable = [x for x in results if x.parsed]
    parse_rate = len(parseable) / max(len(results), 1)
    contradiction_rate = sum(x.status == "contradicted" for x in parseable) / max(len(parseable), 1)
    entailment_rate = sum(x.status == "entailed" for x in parseable) / max(len(parseable), 1)
    unknown_rate = sum(x.status == "unknown" for x in parseable) / max(len(parseable), 1)
    first_pos = next((i / max(len(results) - 1, 1) for i, x in enumerate(results) if x.first_conflict), 1.0)
    repairs = [min(x.repair_cost, 4) / 4.0 for x in parseable]
    mean_conf = sum((sum(r.confidence for r in x.parsed) / len(x.parsed)) for x in parseable) / max(len(parseable), 1)
    grounded_vals = [x.features[4] for x in parseable]
    conclusion_status = results[-1].status if results else "unknown"
    full_ok = feasible(list(context_rel) + list(prior_rel))
    trace_features = [
        float(bool(context_rel)), min(len(context_rel), 32) / 32.0,
        min(len(results), 16) / 16.0, parse_rate, mean_conf,
        sum(grounded_vals) / max(len(grounded_vals), 1), float(full_ok), contradiction_rate,
        entailment_rate, unknown_rate, first_pos,
        sum(repairs) / max(len(repairs), 1), max(repairs, default=0.0),
        float(conclusion_status == "entailed"), float(conclusion_status == "contradicted"),
        float(conclusion_status == "unknown"),
    ]
    assert len(trace_features) == TRACE_CONSTRAINT_DIM
    return ConstraintAnalysis(context_rel, results, trace_features, full_ok)
