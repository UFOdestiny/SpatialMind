"""
Two-stage chain judge helpers.

Stage 1:
  - strict answer/conclusion verification against ground truth.
Stage 2:
  - reasoning-claim verification by chain consistency (not strict lexical GT match).

Both stages are deliberately fail-closed. Stage 1 accepts only the exact JSON
contract. Stage 2 must contain exactly one integer 0/1 label per reasoning
claim. No embedded-JSON extraction, padding, truncation, coercion, or
answer-derived fallback is permitted; the caller drops a failed sample.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


CHAIN_STAGE1_SYSTEM_PROMPT = "Verify answer correctness. Return JSON only."

CHAIN_STAGE1_USER_TEMPLATE = (
    "Q: {question}\n"
    "Answer: {ground_truth}\n"
    "Response: {generated_text}\n"
    "Conclusion: {conclusion_claim}\n\n"
    "Return: {{\"answer\":\"correct|incorrect\",\"conclusion_verified\":0|1}}\n"
    "1=correct, 0=incorrect."
)

CHAIN_STAGE2_SYSTEM_PROMPT = (
    "You are a meticulous spatial-reasoning verifier. You check each step of a "
    "spatial deduction (directions, turns, relative positions, clock-face/compass "
    "mappings). You first think step by step, then output JSON."
)

# ANALYSIS-FIRST prompt: the judge must reason about each step BEFORE labeling it.
# This is the key to real discrimination — without the analysis field the judge
# rubber-stamps almost every step as correct (validated: only ~2pp gap between
# correct and incorrect traces). Each step is judged against the question and the
# preceding steps, NOT against the final answer (LEAKAGE CONTROL: the trace-level
# verdict is never shown).
CHAIN_STAGE2_USER_TEMPLATE = (
    "Verify each spatial reasoning step below. A step is CORRECT (1) only if it is "
    "a valid spatial inference given the trusted scene, question, and earlier steps; it is "
    "INCORRECT (0) if the direction/turn/relative position is wrong, unsupported, "
    "or contradicts the question. Judge each step on its own spatial merit — do NOT "
    "assume steps are correct, and do NOT use the final answer.\n\n"
    "Trusted scene/premises: {context}\n"
    "Question: {question}\n\n"
    "Steps:\n{reasoning_claims_numbered}\n\n"
    "First, in \"analysis\", briefly check each step's spatial logic (e.g. 'a right "
    "turn from east faces south, so step 2 is wrong'). Then output the labels.\n"
    "Output ONLY this JSON (array length EXACTLY {num_claims}, values 0 or 1):\n"
    "{{\"analysis\": \"<one short check per step>\", "
    "\"reasoning_verified\": [{example_output}]}}\n\n"
    "Example — question 'facing north, turn right, turn right, which direction?':\n"
    "Steps:\n  [0] Start facing north.\n  [1] A right turn from north faces east.\n"
    "  [2] Another right turn from east faces west.\n"
    "{{\"analysis\": \"[0] correct start. [1] right from north is east, correct. "
    "[2] right from east is south not west, incorrect.\", "
    "\"reasoning_verified\": [1, 1, 0]}}"
)


def _extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response with multiple fallback strategies."""
    text = (raw or "").strip()
    if not text:
        return None

    # Strategy 1: Direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strategy 2: Find JSON boundaries
    try:
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [s for s in (start_obj, start_arr) if s >= 0]
        if starts:
            start = min(starts)
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")
            end = max(end_obj, end_arr)
            if end >= start:
                candidate = text[start : end + 1]
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
    except Exception:
        pass

    # Strategy 3: Try to extract from markdown code blocks
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            obj = json.loads(code_block_match.group(1))
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass

    return None


def _extract_array_from_text(text: str) -> Optional[List[Any]]:
    """Extract array from text, handling various LLM output formats."""
    if not text:
        return None

    # Strategy 1: Direct array extraction
    try:
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start >= 0 and arr_end > arr_start:
            candidate = text[arr_start : arr_end + 1]
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
    except Exception:
        pass

    # Strategy 2: Comma-separated values (e.g., "1, 0, 1, 0")
    try:
        # Find sequences of 0s and 1s
        pattern = r"(?:^|[^0-9])([01](?:\s*[,\s]\s*[01])+)(?:[^0-9]|$)"
        match = re.search(pattern, text)
        if match:
            values_str = match.group(1)
            values = [int(v.strip()) for v in re.split(r"[,\s]+", values_str) if v.strip() in ("0", "1")]
            if values:
                return values
    except Exception:
        pass

    # Strategy 3: Find all standalone 0s and 1s in expected positions
    try:
        # Match patterns like "claim 1: 1", "idx 0: 0", etc.
        pattern = r"(?:claim|idx|index|#|\d)[^\d]*(\d+)[^\d]*[:=]\s*([01])"
        matches = re.findall(pattern, text.lower())
        if matches:
            result = {}
            for idx_str, val_str in matches:
                try:
                    idx = int(idx_str)
                    val = int(val_str)
                    if val in (0, 1):
                        result[idx] = val
                except Exception:
                    continue
            if result:
                max_idx = max(result.keys())
                arr = [-1] * (max_idx + 1)
                for idx, val in result.items():
                    arr[idx] = val
                return arr
    except Exception:
        pass

    return None


def _to_binary_label(v: Any) -> Optional[int]:
    """Convert various label formats to binary: 1=correct/supported, 0=incorrect/hallucinated."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv in (0, 1):
            return iv
        return None
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in {"1", "correct", "supported", "true"}:
            return 1
        if vv in {"0", "incorrect", "unsupported", "inconsistent", "false", "hallucinated"}:
            return 0
    return None


def build_chain_stage1_prompt(
    tokenizer,
    question: str,
    ground_truth: str,
    generated_text: str,
    conclusion_claim: str,
) -> str:
    user_msg = CHAIN_STAGE1_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated_text=generated_text,
        conclusion_claim=conclusion_claim,
    )
    messages = [
        {"role": "system", "content": CHAIN_STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAIN_STAGE1_SYSTEM_PROMPT}\n\n{user_msg}\nJSON:"


def parse_chain_stage1_output(response: str) -> Dict[str, Optional[int]]:
    """Parse only the exact Stage-1 JSON contract."""
    try:
        obj = json.loads((response or "").strip())
    except Exception:
        obj = None
    answer_label = None
    conclusion_verified = None
    if isinstance(obj, dict) and set(obj) == {"answer", "conclusion_verified"}:
        ans = obj.get("answer")
        conclusion = obj.get("conclusion_verified")
        if ans in ("correct", "incorrect") and type(conclusion) is int and conclusion in (0, 1):
            answer_label = 1 if ans == "correct" else 0
            conclusion_verified = conclusion
    return {
        "answer_label": answer_label,
        "conclusion_verified": conclusion_verified,
    }


def build_chain_stage1_schema() -> dict:
    """Exact JSON schema for answer/conclusion guided decoding."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["correct", "incorrect"]},
            "conclusion_verified": {"type": "integer", "enum": [0, 1]},
        },
        "required": ["answer", "conclusion_verified"],
        "additionalProperties": False,
    }


def build_chain_stage2_schema(expected_len: int) -> dict:
    """JSON schema for guided decoding of the Stage-2 reasoning judge.

    Constrains output to {"analysis": str, "reasoning_verified": [0/1 * expected_len]},
    so the array is ALWAYS the right length with only 0/1 values — no parse
    failures, no padding/truncation. `analysis` first so the CoT precedes labels.
    """
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "maxLength": 800},
            "reasoning_verified": {
                "type": "array",
                "items": {"type": "integer", "enum": [0, 1]},
                "minItems": int(expected_len),
                "maxItems": int(expected_len),
            },
        },
        "required": ["analysis", "reasoning_verified"],
        "additionalProperties": False,
    }


def build_chain_stage2_prompt(
    tokenizer,
    question: str,
    context: str,
    reasoning_claims: List[str],
) -> str:
    num_claims = len(reasoning_claims)

    # Build numbered list of claims (easier for LLM to match output length)
    reasoning_claims_numbered = "\n".join(
        f"  [{i}] {str(t).strip()}" for i, t in enumerate(reasoning_claims)
    )

    # Build example output that exactly matches expected length
    # e.g., "1, 1, 0, 1" for 4 claims
    example_output = ", ".join(["1" if i % 2 == 0 else "0" for i in range(num_claims)])
    if num_claims == 0:
        example_output = ""

    user_msg = CHAIN_STAGE2_USER_TEMPLATE.format(
        context=context,
        question=question,
        reasoning_claims_numbered=reasoning_claims_numbered,
        num_claims=num_claims,
        example_output=example_output,
    )
    messages = [
        {"role": "system", "content": CHAIN_STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAIN_STAGE2_SYSTEM_PROMPT}\n\n{user_msg}\n\nJSON:"


def parse_chain_stage2_output(
    response: str,
    expected_len: int,
    log_failures: bool = False,
) -> Tuple[Optional[List[int]], str]:
    """Strictly parse the guided JSON; any defect drops the entire sample."""
    if expected_len <= 0:
        return [], "exact"
    try:
        obj = json.loads((response or "").strip())
    except Exception:
        if log_failures:
            log.debug("Stage-2 non-JSON output dropped: %s", (response or "")[:200])
        return None, "failed"
    vals = obj.get("reasoning_verified") if isinstance(obj, dict) else None
    if not isinstance(vals, list) or len(vals) != expected_len:
        return None, "failed"
    out = []
    for value in vals:
        if type(value) is not int or value not in (0, 1):
            return None, "failed"
        out.append(value)
    return out, "exact"


def _extract_indexed_values(text: str, expected_len: int) -> Optional[List[int]]:
    """Extract values by index patterns like [0]: 1, [1]: 0, etc."""
    if not text or expected_len <= 0:
        return None

    # Initialize with -1 (not found)
    result = [-1] * expected_len

    # Pattern variations for indexed values
    patterns = [
        r'\[(\d+)\]\s*[:=]\s*([01])\b',         # [0]: 1 or [0]=1
        r'\[(\d+)\]\s*[:=]?\s*(supported|unsupported|correct|incorrect|true|false)',  # [0]: supported
        r'(?:claim|idx|index)\s*(\d+)\s*[:=]\s*([01])\b',  # claim 0: 1
        r'(?:claim|idx|index)\s*(\d+)\s*[:=]?\s*(supported|unsupported|correct|incorrect|true|false)',
        r'(\d+)\s*[:=]\s*([01])\b',              # 0: 1 or 0=1
    ]

    found_count = 0
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                idx = int(match.group(1))
                val_str = match.group(2).strip().lower()
                if idx < 0 or idx >= expected_len:
                    continue
                if val_str in ("1", "supported", "correct", "true"):
                    result[idx] = 1
                    found_count += 1
                elif val_str in ("0", "unsupported", "incorrect", "false"):
                    result[idx] = 0
                    found_count += 1
            except (ValueError, IndexError):
                continue

    # Only return if we found at least some indexed values
    if found_count > 0:
        return result
    return None


# Backward compatibility wrapper
def parse_chain_stage2_output_compat(response: str, expected_len: int) -> Optional[List[int]]:
    """Backward-compatible wrapper that returns None only on complete failure."""
    result, status = parse_chain_stage2_output(response, expected_len)
    return result
