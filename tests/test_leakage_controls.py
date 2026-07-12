import inspect

from data.claims import (
    extract_claims_from_llm_output,
    extract_claims_from_structured_generation,
    parse_structured_generation_output,
)
from scripts.evaluate import _collect_baseline_uncertainty
from utils.chain_judge import (
    build_chain_stage2_prompt,
    parse_chain_stage1_output,
    parse_chain_stage2_output,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(x["content"] for x in messages)


def test_reasoning_judge_sees_context_but_not_answer_verdict():
    prompt = build_chain_stage2_prompt(
        _Tokenizer(), question="Where is A?", context="A is north of B.",
        reasoning_claims=["A is north of B"],
    )
    assert "A is north of B." in prompt
    assert "answer_label" not in inspect.signature(build_chain_stage2_prompt).parameters
    assert "ground_truth" not in inspect.signature(build_chain_stage2_prompt).parameters


def test_reasoning_judge_parser_fails_closed():
    assert parse_chain_stage2_output(
        '{"analysis":"x","reasoning_verified":[1,0]}', expected_len=2
    )[0] == [1, 0]
    assert parse_chain_stage2_output(
        '{"analysis":"x","reasoning_verified":[1]}', expected_len=2
    )[0] is None


def test_answer_judge_parser_fails_closed():
    valid = '{"answer":"correct","conclusion_verified":1}'
    assert parse_chain_stage1_output(valid)["answer_label"] == 1
    assert parse_chain_stage1_output("Answer: correct")["answer_label"] is None
    assert parse_chain_stage1_output(f"```json\n{valid}\n```")["answer_label"] is None
    assert parse_chain_stage1_output('{"answer":"correct"}')["answer_label"] is None


def test_llm_claim_extraction_never_falls_back_or_repairs():
    response = "A is north of B. Therefore A is above B. The answer is north."
    valid = (
        '{"claims":['
        '{"type":"reasoning","text":"A is north of B"},'
        '{"type":"reasoning","text":"A is above B"},'
        '{"type":"conclusion","text":"north"}'
        ']}'
    )
    assert len(extract_claims_from_llm_output(valid, response, True)) == 3
    assert extract_claims_from_llm_output(f"```json\n{valid}\n```", response, True) == []
    assert extract_claims_from_llm_output('{"claims":[]}', response, True) == []
    repaired = (
        '{"claims":['
        '{"type":"reasoning","text":"A is north of B"},'
        '{"type":"conclusion","text":"north"},'
        '{"type":"reasoning","text":"A is above B"}'
        ']}'
    )
    assert extract_claims_from_llm_output(repaired, response, True) == []


def test_guided_generation_contract_and_alignment():
    raw = '{"reasoning":["A is north of B","B is left of C"],"conclusion":"north-west"}'
    assert parse_structured_generation_output(raw) == (
        ["A is north of B", "B is left of C"], "north-west"
    )
    claims = extract_claims_from_structured_generation(raw, answer_correct=False)
    assert [c.claim_type for c in claims] == ["reasoning", "reasoning", "conclusion"]
    assert [c.verified for c in claims] == [-1, -1, 0]
    assert all(c.aligned_token_ids for c in claims)
    out_of_order = '{"conclusion":"north-west","reasoning":["A is north of B","B is left of C"]}'
    assert len(extract_claims_from_structured_generation(out_of_order, answer_correct=True)) == 3
    assert parse_structured_generation_output(
        '{"reasoning":[],"conclusion":"north"}'
    ) is None
    assert parse_structured_generation_output(
        '{"reasoning":["x"],"conclusion":"north","extra":1}'
    ) is None
    assert parse_chain_stage2_output("not json", expected_len=2)[0] is None
    assert parse_chain_stage2_output(
        '```json\n{"reasoning_verified":[1,0]}\n```', expected_len=2
    )[0] is None


def test_baseline_estimator_never_receives_supervision_fields():
    class Dataset:
        def __len__(self): return 1
        def load_raw(self, _):
            return {
                "label": 1, "verified": [1], "ground_truth": "SECRET",
                "answer_correct": True, "claims": [{"claim_type_id": 2}], "k_hop": 1,
            }

    class Estimator:
        def estimate_claims(self, raw):
            assert not ({"label", "verified", "ground_truth", "answer_correct"} & set(raw))
            return [0.25]

    _, _, labels, _ = _collect_baseline_uncertainty(Estimator(), Dataset())
    assert labels.tolist() == [1]
