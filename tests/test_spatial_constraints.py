from spatial_constraints.analysis import analyze_trace
from spatial_constraints.parser import parse_relations
from spatial_constraints.solver import feasible, relation_status


def test_directional_composition_entails_northeast():
    base = parse_relations("A is north of B. B is east of C.", source="context")
    claim = parse_relations("A is north-east of C.")[0]
    assert feasible(base)
    assert relation_status(base, claim) == "entailed"


def test_wrong_west_claim_is_contradicted():
    base = parse_relations("A is north of B. B is east of C.", source="context")
    claim = parse_relations("A is north-west of C.")[0]
    assert relation_status(base, claim) == "contradicted"
    assert not feasible(base + [claim])


def test_left_transitivity_with_qualitative_distance():
    base = parse_relations("circle is left of square. square is left of triangle.")
    claim = parse_relations("circle is left of triangle.")[0]
    assert relation_status(base, claim) == "entailed"


def test_babi_latest_location_wins():
    analysis = analyze_trace(
        "Mary moved to the kitchen. Mary went to the garden.",
        "Where is Mary?",
        [{"text": "Mary is at the garden", "claim_type_id": 2}],
    )
    assert analysis.full_trace_feasible
    assert analysis.claims[0].status == "entailed"


def test_first_invalid_claim_localization():
    analysis = analyze_trace(
        "Ravi is east of Sana. Sana is south of Theo.",
        "Where is Ravi relative to Theo?",
        [
            {"text": "Ravi is east of Sana", "claim_type_id": 1},
            {"text": "Sana is south of Theo", "claim_type_id": 1},
            {"text": "Ravi is north-east of Theo", "claim_type_id": 1},
            {"text": "north-east", "claim_type_id": 2},
        ],
    )
    assert analysis.claims[0].status == "entailed"
    assert analysis.claims[1].status == "entailed"
    assert analysis.claims[2].status == "contradicted"
    assert analysis.claims[2].first_conflict


def test_claims_after_a_conflict_are_not_evaluable():
    analysis = analyze_trace(
        "A is north of B.",
        "Where is A relative to B?",
        [
            {"text": "A is south of B", "claim_type_id": 1},
            {"text": "A is north of B", "claim_type_id": 1},
        ],
    )
    assert analysis.claims[0].status == "contradicted"
    assert analysis.claims[0].first_conflict
    assert analysis.claims[1].status == "not_evaluable"


def test_stepgame_label_conclusion_is_grounded_by_question():
    analysis = analyze_trace(
        "A is north of B. B is east of C.",
        "What is the relation of the agent A to the agent C?",
        [{"text": "The answer is upper-right", "claim_type_id": 2}],
    )
    assert analysis.claims[0].status == "entailed"
    assert analysis.claims[0].parsed[0].subject == "a"
    assert analysis.claims[0].parsed[0].object == "c"


def test_stepgame_filler_clock_and_compound_regressions():
    assert parse_relations("The objects X and Z are over there.") == []
    cases = {
        "V is over there and G is directly above it.": ("g", "above", "v"),
        "P is at a 45 degree angle to Z, in the upper lefthand corner.": ("p", "upper-left", "z"),
        "Z is above Q at 10 o'clock.": ("z", "upper-left", "q"),
        "H is on the right side and top of V.": ("h", "upper-right", "v"),
        "X is positioned in the front right corner of M.": ("x", "upper-right", "m"),
        "G is slightly off center to the top left and D is slightly off center to the bottom right.":
            ("g", "upper-left", "d"),
        "P and G are both there with the object G below the object P.": ("g", "below", "p"),
        "I and H are in a vertical line with H below I.": ("h", "below", "i"),
        "K and W are next to each other with W at the bottom K on the top.": ("w", "below", "k"),
        "C is positioned up and to the right of Y.": ("c", "upper-right", "y"),
    }
    for text, expected in cases.items():
        parsed = parse_relations(text)
        assert len(parsed) == 1, text
        assert (parsed[0].subject, parsed[0].relation, parsed[0].object) == expected
