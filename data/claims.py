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
    """Validate the claim contract without trimming or repairing it."""
    reasoning_claims = [c for c in claims if c.claim_type == "reasoning"]
    conclusion_claims = [c for c in claims if c.claim_type == "conclusion"]
    if not reasoning_claims or len(reasoning_claims) > int(max_reasoning_claims):
        return []
    if len(conclusion_claims) != 1:
        return []
    if claims[-1].claim_type != "conclusion":
        return []
    if len(reasoning_claims) + 1 != len(claims):
        return []
    return claims


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


def parse_structured_generation_output(raw: str) -> Optional[Tuple[List[str], str]]:
    """Parse the exact guided-generation JSON contract."""
    try:
        obj = json.loads((raw or "").strip())
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != {"reasoning", "conclusion"}:
        return None
    reasoning = obj.get("reasoning")
    conclusion = obj.get("conclusion")
    if not isinstance(reasoning, list) or not 1 <= len(reasoning) <= 6:
        return None
    if any(not isinstance(x, str) or not x.strip() for x in reasoning):
        return None
    if not isinstance(conclusion, str) or not conclusion.strip():
        return None
    return [x.strip() for x in reasoning], conclusion.strip()


def extract_claims_from_structured_generation(
    generated_text: str,
    answer_correct: bool,
    tokenizer=None,
    generated_token_ids: Optional[List[int]] = None,
) -> List[SpatialClaim]:
    """Build aligned claims directly from guided backbone JSON output.

    Claim strings are located in the original JSON text, so their token masks
    remain aligned with the frozen features extracted from those exact tokens.
    """
    parsed = parse_structured_generation_output(generated_text)
    if parsed is None:
        return []
    reasoning, conclusion = parsed
    text = (generated_text or "").strip()
    token_spans = _token_spans_with_tokenizer(text, tokenizer=tokenizer)
    if not token_spans:
        return []

    claims: List[SpatialClaim] = []
    search_pos = 0
    used_spans = set()
    for ctype, claim_text in [*(('reasoning', x) for x in reasoning), ('conclusion', conclusion)]:
        # JSON encoding may escape a claim. Spatial outputs normally match the
        # plain string; accept the JSON-encoded body as the exact serialized form.
        candidates = [claim_text, json.dumps(claim_text, ensure_ascii=False)[1:-1]]
        found = None
        for candidate in candidates:
            for start_at in (search_pos, 0):
                start = text.find(candidate, start_at)
                while start >= 0:
                    span = (start, start + len(candidate))
                    if span not in used_spans:
                        found = span
                        break
                    start = text.find(candidate, start + 1)
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return []
        aligned = _span_to_token_ids(token_spans, found[0], found[1])
        if not aligned:
            return []
        used_spans.add(found)
        search_pos = max(search_pos, found[1])
        claims.append(
            SpatialClaim(
                text=claim_text,
                claim_type=ctype,
                aligned_token_ids=aligned,
                verified=(1 if answer_correct else 0) if ctype == "conclusion" else PENDING_LABEL,
            )
        )

    if generated_token_ids is not None:
        if len(generated_token_ids) == 0:
            return []
        max_idx = len(generated_token_ids) - 1
        for claim in claims:
            claim.aligned_token_ids = [i for i in claim.aligned_token_ids if 0 <= i <= max_idx]
            if not claim.aligned_token_ids:
                return []
    return claims


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
                return []
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

    # Fail closed: format/extraction failures are dropped by the caller.
    if not claims:
        return []
    if not any(c.claim_type == "conclusion" for c in claims):
        return []

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
      {"claims": [{"type": "reasoning|conclusion", "text": "..."}, ...]}

    This parser is deliberately fail-closed. It does not extract an embedded
    JSON substring, accept alternate keys, fall back to regex claims, discard
    malformed items, or repair the claim sequence.
    """
    text = (generated_text or "").strip()
    if not text:
        return []

    raw = (llm_output or "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, dict) or set(obj) != {"claims"}:
        return []
    parsed_claims = obj.get("claims")
    if not isinstance(parsed_claims, list) or not 3 <= len(parsed_claims) <= 6:
        return []
    if any(not isinstance(item, dict) or set(item) != {"type", "text"} for item in parsed_claims):
        return []
    raw_types = [str(item["type"]).strip().lower() for item in parsed_claims]
    if any(t not in CLAIM_TYPE2ID for t in raw_types):
        return []
    if raw_types[-1] != "conclusion" or raw_types.count("conclusion") != 1:
        return []

    token_spans = _token_spans_with_tokenizer(text, tokenizer=tokenizer)
    lowered = text.lower()
    search_pos = 0
    claims: List[SpatialClaim] = []

    for item in parsed_claims:
        ctype = str(item["type"]).strip().lower()
        ctext = _clean_claim_text(str(item["text"]))
        if ctype not in CLAIM_TYPE2ID or not ctext or _is_generic_claim_text(ctext):
            return []

        ctext_lower = ctext.lower()
        start = lowered.find(ctext_lower, search_pos)
        if start < 0:
            start = lowered.find(ctext_lower)
        if start < 0:
            return []
        end = start + len(ctext_lower)
        aligned = _span_to_token_ids(token_spans, start, end)
        search_pos = end
        if not aligned:
            return []

        # Extraction and supervision are separate stages. Ignore any unsolicited
        # `verified`/`label` field emitted by the extractor model: reasoning
        # labels must come from the context-grounded Stage-2 judge, while the
        # conclusion label is exactly the independently computed answer label.
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

    claims = _keep_reasoning_and_single_conclusion_claims(claims, max_reasoning_claims=12)
    if not claims:
        return []

    if generated_token_ids is not None:
        if len(generated_token_ids) == 0:
            return []
        max_idx = max(0, len(generated_token_ids) - 1)
        for claim in claims:
            claim.aligned_token_ids = [tid for tid in claim.aligned_token_ids if 0 <= tid <= max_idx]
            if not claim.aligned_token_ids:
                return []

    return claims
