"""Unit test for the unified conclusion-claim labeling.

Spec: the conclusion claim asserts "the answer is X", so its correctness IS the
final-answer correctness. For BOTH task types the conclusion claim label is tied
to the resolved sample label, never the judge's independent semantic verdict:
  * classification: sample["label"] = deterministic parse_answer + check_correctness
  * free-form:     sample["label"] = judge answer_label (answer aligned with truth)
Reasoning claims are unaffected (still verified by the Stage-2 judge).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _apply_conclusion_label(sample, conclusion_pos):
    """Mirror of the conclusion-labeling block in judge.py:_judge_split (kept in sync)."""
    claims = sample.get("claims", []) or []
    if not claims:
        return sample
    verified = sample.get("verified", [])
    if not isinstance(verified, list) or len(verified) != len(claims):
        verified = [-1] * len(claims)
    if 0 <= conclusion_pos < len(verified):
        resolved = int(sample.get("label", -1))
        if resolved in (0, 1):
            verified[conclusion_pos] = resolved
    sample["verified"] = verified
    return sample


def test_classification_correct():
    # Deterministic label=1 (answer matched). Conclusion must be 1.
    s = {"label": 1, "claims": [{"claim_type": "reasoning"}, {"claim_type": "conclusion"}],
         "verified": [1, -1]}
    _apply_conclusion_label(s, conclusion_pos=1)
    assert s["verified"][1] == 1
    print("PASS: classification conclusion = deterministic label (1)")


def test_classification_incorrect():
    s = {"label": 0, "claims": [{"claim_type": "conclusion"}], "verified": [-1]}
    _apply_conclusion_label(s, conclusion_pos=0)
    assert s["verified"][0] == 0
    print("PASS: classification conclusion = deterministic label (0)")


def test_freeform_uses_answer_label():
    # Free-form: sample["label"] already resolved from judge answer_label. Conclusion follows it.
    s = {"label": 1, "claims": [{"claim_type": "conclusion"}], "verified": [-1]}
    _apply_conclusion_label(s, conclusion_pos=0)
    assert s["verified"][0] == 1
    print("PASS: free-form conclusion = resolved answer label (1)")


def test_unparseable_left_pending():
    # If the answer is unresolved (label=-1), the conclusion stays pending.
    s = {"label": -1, "claims": [{"claim_type": "conclusion"}], "verified": [-1]}
    _apply_conclusion_label(s, conclusion_pos=0)
    assert s["verified"][0] == -1
    print("PASS: unresolved label leaves conclusion pending (-1)")


def test_reasoning_untouched():
    s = {"label": 1, "claims": [{"claim_type": "reasoning"}, {"claim_type": "conclusion"}],
         "verified": [0, -1]}
    _apply_conclusion_label(s, conclusion_pos=1)
    assert s["verified"][0] == 0, "reasoning claim must be untouched"
    assert s["verified"][1] == 1, "conclusion tied to sample label"
    print("PASS: reasoning claim untouched, only conclusion set from label")


def test_dataset_flag():
    from data.datasets import get_dataset_cls
    assert get_dataset_cls("spartqa").needs_judge is False
    assert get_dataset_cls("StepGame").needs_judge is False
    assert get_dataset_cls("babi").needs_judge is True
    print("PASS: dataset needs_judge flags correct")


if __name__ == "__main__":
    test_classification_correct()
    test_classification_incorrect()
    test_freeform_uses_answer_label()
    test_unparseable_left_pending()
    test_reasoning_untouched()
    test_dataset_flag()
    print("\nAll conclusion-labeling tests passed.")
