from scripts.constraint_counterfactual import build_counterfactual_pair
from spatial_constraints import analyze_trace


def _claim(text):
    return {"text": text, "claim_type": "conclusion", "claim_type_id": 2}


def test_textual_counterfactual_is_detected_end_to_end():
    context = "A is above B. B is right of C. A is inside box one."
    relations = analyze_trace(context, "", []).to_dict()["context_relations"]
    for relation in relations:
        pair = build_counterfactual_pair(relation)
        assert pair is not None
        original_text, corrupt_text, _ = pair
        original = analyze_trace(context, "", [_claim(original_text)]).claims[0]
        corrupt = analyze_trace(context, "", [_claim(corrupt_text)]).claims[0]
        assert original.parsed and original.status == "entailed"
        assert corrupt.parsed and corrupt.status == "contradicted"
