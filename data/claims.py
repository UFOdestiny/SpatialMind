"""
claims.py - Claim-level extraction helpers for cached feature generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CLAIM_TYPE2ID: Dict[str, int] = {
    "reasoning": 1,
    "conclusion": 2,
}

# Sentinel for a claim whose correctness has not yet been assigned. Reasoning
# claims start pending and are labeled by the Stage-2 reasoning judge.
PENDING_LABEL: int = -1


@dataclass
class SpatialClaim:
    text: str
    claim_type: str
    aligned_token_ids: List[int]
    verified: int


_REASONING_LINE_RE = re.compile(r"^\s*reasoning\s*:\s*(.+?)\s*$", flags=re.IGNORECASE)
_CONCLUSION_RE = re.compile(
    r"^\s*conclusion\s*:\s*(?:the answer is\s*)?(.+?)\s*\.?\s*$",
    flags=re.IGNORECASE,
)

_REASONING_SECTION_RE = re.compile(
    r"reasoning\s*:\s*(.*?)(?=conclusion\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CONCLUSION_SECTION_RE = re.compile(
    r"conclusion\s*:\s*(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_CONCLUSION_MARKER_RE = re.compile(r"\bconclusion\s*:", flags=re.IGNORECASE)

_GENERIC_CLAIM_PATTERNS = (
    re.compile(r"^\s*reasoning[:.]?\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*conclusion[:.]?\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*merge\s+intermediate\s+reasoning[:.]?\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*merge\s+reasoning[:.]?\s*$", flags=re.IGNORECASE),
)

_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"reasoning|"
    r"conclusion|"
    r"merge\s+intermediate\s+reasoning|"
    r"merge\s+reasoning"
    r")\s*[:.\-]*\s*",
    flags=re.IGNORECASE,
)


def _normalize_claim_type(raw_type: str) -> str:
    rt = (raw_type or "").strip().lower()
    if not rt:
        return ""
    if rt in CLAIM_TYPE2ID:
        return rt
    if rt in {"premise", "observation", "restate", "input_restate", "fact", "given"}:
        return "reasoning"
    if rt in {"infer", "inference", "deduction", "combine", "composed", "intermediate"}:
        return "reasoning"
    if rt in {"final", "final_answer", "answer", "result"}:
        return "conclusion"
    return ""


def _clean_claim_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()
    t = t.strip("\"'`")
    t = re.sub(r"^\s*[-*]\s*", "", t)
    t = re.sub(r"^\s*\d+\s*[\).:\-]\s*", "", t)
    t = re.sub(r"^\s*(step|claim)\s*\d+\s*:\s*", "", t, flags=re.IGNORECASE)
    t = _SECTION_PREFIX_RE.sub("", t, count=1)
    t = re.sub(r"^\s*the\s+answer\s+is\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*answer\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip("\"'`")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_generic_claim_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    for p in _GENERIC_CLAIM_PATTERNS:
        if p.match(t):
            return True
    return False


def _extract_section_text(regex: re.Pattern, text: str) -> str:
    m = regex.search(text or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _split_section_claims(section_text: str) -> List[str]:
    """Split a reasoning/conclusion-adjacent section into atomic claims."""
    s = (section_text or "").strip()
    if not s:
        return []

    # Normalize inline content into a single line for robust splitting.
    s = re.sub(r"\s+", " ", s)

    # Primary split by bullets/numbered items.
    parts = re.split(r"(?:^|\s)(?:[-*]|\d+[.)])\s+", s)
    if len(parts) <= 1:
        # Fallback to sentence-level split when explicit bullets are absent.
        parts = [p for p in re.split(r"(?<=[\.;])\s+", s) if p.strip()]
        if not parts:
            parts = [s]
    else:
        parts = [p for p in parts if p.strip()]

    claims: List[str] = []
    seen = set()
    for part in parts:
        # Secondary split for inline bullet chains.
        subparts = re.split(r"\s+-\s+", part)
        for sp in subparts:
            c = _clean_claim_text(sp)
            if not c:
                continue
            if _is_generic_claim_text(c):
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(c)
    return claims


def _extract_freeform_reasoning_candidates(text: str) -> List[str]:
    """Extract reasoning claims from free-form text before conclusion marker."""
    if not text:
        return []
    m = _CONCLUSION_MARKER_RE.search(text)
    prefix = text[: m.start()] if m else text
    return _split_section_claims(prefix)


def _extract_structured_candidates(text: str) -> List[Tuple[str, str]]:
    """Extract (claim_type, text) candidates from reasoning/conclusion structure.

    Design choice: keep only reasoning claims and one final conclusion.
    """
    candidates: List[Tuple[str, str]] = []
    seen = set()

    # Preferred section for reasoning claims.
    reasoning_text = _extract_section_text(_REASONING_SECTION_RE, text)
    for c in _split_section_claims(reasoning_text):
        key = ("reasoning", c.lower())
        if key not in seen:
            seen.add(key)
            candidates.append(("reasoning", c))

    # If no explicit reasoning marker is present,
    # fallback to free-form reasoning extraction from pre-conclusion text.
    has_structured_reasoning = bool(_REASONING_SECTION_RE.search(text or ""))
    if not has_structured_reasoning:
        for c in _extract_freeform_reasoning_candidates(text):
            key = ("reasoning", c.lower())
            if key not in seen:
                seen.add(key)
                candidates.append(("reasoning", c))

    conclusion_text = _extract_section_text(_CONCLUSION_SECTION_RE, text)
    conc = _clean_claim_text(conclusion_text)
    if conc and not _is_generic_claim_text(conc):
        key = ("conclusion", conc.lower())
        if key not in seen:
            seen.add(key)
            candidates.append(("conclusion", conc))

    # Ensure exactly one conclusion at the end.
    conclusion_candidates = [x for x in candidates if x[0] == "conclusion"]
    reasoning_candidates = [x for x in candidates if x[0] == "reasoning"]
    if conclusion_candidates:
        candidates = reasoning_candidates + [conclusion_candidates[-1]]
    else:
        candidates = reasoning_candidates

    return candidates


def _keep_reasoning_and_single_conclusion_claims(
    claims: List[SpatialClaim],
    max_reasoning_claims: int = 12,
) -> List[SpatialClaim]:
    """Keep reasoning claims and exactly one final conclusion claim."""
    reasoning_claims = [c for c in claims if c.claim_type == "reasoning"]
    conclusion_claims = [c for c in claims if c.claim_type == "conclusion"]

    if len(reasoning_claims) > max_reasoning_claims:
        reasoning_claims = reasoning_claims[: int(max_reasoning_claims)]

    if conclusion_claims:
        return reasoning_claims + [conclusion_claims[-1]]
    return reasoning_claims


def _synthesize_conclusion_text(generated_text: str) -> str:
    text = (generated_text or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    for ln in reversed(lines):
        lower = ln.lower()
        if "answer is" in lower or lower.startswith("conclusion:"):
            return ln
    return lines[-1]


def _token_spans_with_tokenizer(
    text: str,
    tokenizer=None,
) -> List[Tuple[int, int]]:
    """
    LUH-style alignment granularity:
    align claims on generated-token coordinates (not prompt coordinates).
    """
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping", [])
            if offsets:
                return [(int(s), int(e)) for s, e in offsets if int(e) > int(s)]
        except Exception:
            pass
    # Fallback when tokenizer offsets are unavailable.
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end()))
    return spans


def _span_to_token_ids(token_spans: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    ids: List[int] = []
    for i, (s, e) in enumerate(token_spans):
        if e <= start or s >= end:
            continue
        ids.append(i)
    return ids


def extract_claims_from_generation(
    generated_text: str,
    answer_correct: bool,
    tokenizer=None,
    generated_token_ids: Optional[List[int]] = None,
) -> List[SpatialClaim]:
    """
    Extract claims from CoT-style generation text and align to token positions
    in generated text space (whitespace-token approximation).
    """
    text = (generated_text or "").strip()
    if not text:
        return []

    token_spans = _token_spans_with_tokenizer(text, tokenizer=tokenizer)
    claims: List[SpatialClaim] = []

    structured_candidates = _extract_structured_candidates(text)
    if structured_candidates:
        lowered_full = text.lower()
        search_pos = 0
        for claim_type, claim_text in structured_candidates:
            claim_text_clean = _clean_claim_text(claim_text)
            if not claim_text_clean or _is_generic_claim_text(claim_text_clean):
                continue

            start = lowered_full.find(claim_text_clean.lower(), search_pos)
            if start < 0:
                start = lowered_full.find(claim_text_clean.lower())
            if start < 0:
                aligned = list(range(len(token_spans)))
            else:
                end = start + len(claim_text_clean)
                aligned = _span_to_token_ids(token_spans, start, end)
                search_pos = end
                if not aligned:
                    aligned = list(range(len(token_spans)))

            if claim_type == "conclusion":
                verified = 1 if answer_correct else 0
            else:
                # Reasoning claims are UNLABELED by extraction. They must be
                # verified by the Stage-2 reasoning judge; leaving them pending
                # (-1) ensures the judge labels them instead of silently
                # defaulting to "correct" (which makes claim-level supervision
                # vacuous). See scripts/judge.py Stage-2.
                verified = PENDING_LABEL

            claims.append(
                SpatialClaim(
                    text=claim_text_clean,
                    claim_type=claim_type,
                    aligned_token_ids=aligned,
                    verified=verified,
                )
            )

        if claims:
            if generated_token_ids is not None:
                max_idx = max(0, len(generated_token_ids) - 1)
                for claim in claims:
                    claim.aligned_token_ids = [tid for tid in claim.aligned_token_ids if 0 <= tid <= max_idx]
                    if not claim.aligned_token_ids:
                        claim.aligned_token_ids = [max_idx]

            if not any(c.claim_type == "conclusion" for c in claims):
                claims.append(
                    SpatialClaim(
                        text=_synthesize_conclusion_text(text),
                        claim_type="conclusion",
                        aligned_token_ids=list(range(len(token_spans))),
                        verified=1 if answer_correct else 0,
                    )
                )
            claims = _keep_reasoning_and_single_conclusion_claims(claims, max_reasoning_claims=12)
            return claims

    for line_match in re.finditer(r".+", text):
        line = line_match.group(0).strip()
        if not line:
            continue

        claim_type = None
        claim_text = None
        m_reasoning = _REASONING_LINE_RE.match(line)
        if m_reasoning:
            claim_type = "reasoning"
            claim_text = m_reasoning.group(1).strip()
        else:
            m_conc = _CONCLUSION_RE.match(line)
            if m_conc:
                claim_type = "conclusion"
                claim_text = m_conc.group(1).strip()

        if claim_type is None or not claim_text:
            continue
        claim_text = _clean_claim_text(claim_text)
        if not claim_text or _is_generic_claim_text(claim_text):
            continue

        start, end = line_match.span()
        aligned = _span_to_token_ids(token_spans, start, end)
        if not aligned:
            continue

        if claim_type == "conclusion":
            verified = 1 if answer_correct else 0
        else:
            # Reasoning claims are unlabeled by extraction; the Stage-2 judge
            # assigns their correctness. Pending (-1) so they are not silently
            # treated as correct.
            verified = PENDING_LABEL

        claims.append(
            SpatialClaim(
                text=claim_text,
                claim_type=claim_type,
                aligned_token_ids=aligned,
                verified=verified,
            )
        )

    # Fallback: at least keep one conclusion claim covering all generated tokens.
    if not claims:
        claims.append(
            SpatialClaim(
                text=text,
                claim_type="conclusion",
                aligned_token_ids=list(range(len(token_spans))),
                verified=1 if answer_correct else 0,
            )
        )

    claims = _keep_reasoning_and_single_conclusion_claims(claims, max_reasoning_claims=12)

    # Clamp aligned ids to available generated-token count when provided by engine.
    if generated_token_ids is not None:
        max_idx = max(0, len(generated_token_ids) - 1)
        for claim in claims:
            claim.aligned_token_ids = [tid for tid in claim.aligned_token_ids if 0 <= tid <= max_idx]
            if not claim.aligned_token_ids:
                claim.aligned_token_ids = [max_idx]

    return claims


def extract_claims_from_llm_output(
    llm_output: str,
    generated_text: str,
    answer_correct: bool,
    tokenizer=None,
    generated_token_ids: Optional[List[int]] = None,
) -> List[SpatialClaim]:
    """
    Parse LLM-produced JSON claims and align them to generated text tokens.

    Expected JSON format:
      {"claims": [{"claim_type": "reasoning|conclusion", "text": "...", "verified": 0|1}]}
    or
      [{"claim_type": "...", "text": "...", "verified": 0|1}, ...]
    """
    text = (generated_text or "").strip()
    if not text:
        return []

    parsed_claims = []
    raw = (llm_output or "").strip()
    if raw:
        try:
            json_candidate = raw
            start_obj = raw.find("{")
            start_arr = raw.find("[")
            starts = [s for s in [start_obj, start_arr] if s >= 0]
            if starts:
                start = min(starts)
                end_obj = raw.rfind("}")
                end_arr = raw.rfind("]")
                end = max(end_obj, end_arr)
                if end >= start:
                    json_candidate = raw[start : end + 1]
            obj = json.loads(json_candidate)
            if isinstance(obj, dict):
                parsed_claims = obj.get("claims", [])
            elif isinstance(obj, list):
                parsed_claims = obj
        except Exception:
            parsed_claims = []

    if not parsed_claims:
        return extract_claims_from_generation(
            generated_text=text,
            answer_correct=answer_correct,
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
        )

    token_spans = _token_spans_with_tokenizer(text, tokenizer=tokenizer)
    lowered = text.lower()
    search_pos = 0
    claims: List[SpatialClaim] = []

    for item in parsed_claims:
        if not isinstance(item, dict):
            continue
        ctype = _normalize_claim_type(str(item.get("claim_type", item.get("type", ""))))
        ctext = _clean_claim_text(str(item.get("text", "")))
        if ctype not in CLAIM_TYPE2ID or not ctext or _is_generic_claim_text(ctext):
            continue

        ctext_lower = ctext.lower()
        start = lowered.find(ctext_lower, search_pos)
        if start < 0:
            start = lowered.find(ctext_lower)
        if start < 0:
            # Fallback: keep claim but align to all generated tokens.
            aligned = list(range(len(token_spans)))
        else:
            end = start + len(ctext_lower)
            aligned = _span_to_token_ids(token_spans, start, end)
            search_pos = end
            if not aligned:
                aligned = list(range(len(token_spans)))

        verified = None
        raw_verified = item.get("verified", item.get("label", None))
        if isinstance(raw_verified, bool):
            verified = int(raw_verified)
        elif isinstance(raw_verified, (int, float)) and int(raw_verified) in (0, 1):
            verified = int(raw_verified)
        elif isinstance(raw_verified, str):
            rv = raw_verified.strip().lower()
            # New convention: 1=correct/supported, 0=hallucinated/unsupported
            if rv in {"1", "correct", "true", "supported"}:
                verified = 1
            elif rv in {"0", "incorrect", "false", "hallucinated", "unsupported"}:
                verified = 0

        if verified is None:
            # Conclusion mirrors answer correctness; reasoning stays pending (-1)
            # for the Stage-2 judge rather than defaulting to "correct".
            if ctype == "conclusion":
                verified = 1 if answer_correct else 0
            else:
                verified = PENDING_LABEL
        claims.append(
            SpatialClaim(
                text=ctext,
                claim_type=ctype,
                aligned_token_ids=aligned,
                verified=verified,
            )
        )

    if not claims:
        return extract_claims_from_generation(
            generated_text=text,
            answer_correct=answer_correct,
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
        )

    if not any(c.claim_type == "conclusion" for c in claims):
        claims.append(
            SpatialClaim(
                text=_synthesize_conclusion_text(text),
                claim_type="conclusion",
                aligned_token_ids=list(range(len(token_spans))),
                verified=1 if answer_correct else 0,
            )
        )

    claims = _keep_reasoning_and_single_conclusion_claims(claims, max_reasoning_claims=12)

    if generated_token_ids is not None:
        max_idx = max(0, len(generated_token_ids) - 1)
        for claim in claims:
            claim.aligned_token_ids = [tid for tid in claim.aligned_token_ids if 0 <= tid <= max_idx]
            if not claim.aligned_token_ids:
                claim.aligned_token_ids = [max_idx]

    return claims
