"""Rule-based normalization of spatial statements into canonical triples.

The parser is intentionally conservative: an unparsed claim is represented as
unknown rather than being silently treated as spatially consistent. This makes
parser coverage a measurable part of the method instead of hidden label noise.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, List

from spatial_constraints.relations import RELATION_ALIASES, canonicalize_relation


@dataclass(frozen=True)
class CanonicalRelation:
    subject: str
    relation: str
    object: str
    source_text: str = ""
    confidence: float = 1.0
    source: str = "claim"

    def to_dict(self) -> dict:
        return asdict(self)


_ENTITY_CLEAN = re.compile(r"^(?:the|a|an)\s+", re.I)
_STEP_PREFIX = re.compile(r"^(?:reasoning\s*:)?\s*(?:[-*]|step\s*\d+\s*[:.)-])\s*", re.I)


def normalize_entity(text: str) -> str:
    t = (text or "").strip(" \t\r\n.,;:!?\"'`()[]{}")
    t = _ENTITY_CLEAN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    # Remove discourse-only prefixes without destroying entity names.
    t = re.sub(r"^(?:therefore|thus|hence|so|combining these|we know that)\s+", "", t)
    return t


def split_statements(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\b(?:Reasoning|Conclusion)\s*:\s*", "", str(text), flags=re.I)
    chunks = re.split(r"(?:\n+|(?<=[.!?;])\s+|\s+-\s+)", text)
    out = []
    for c in chunks:
        c = _STEP_PREFIX.sub("", c.strip())
        if c:
            out.append(c)
    return out


def _relation_phrases() -> List[str]:
    phrases = set(RELATION_ALIASES)
    phrases.update({
        "upper-right", "lower-right", "upper-left", "lower-left", "above", "below",
        "left", "right", "overlap", "inside", "contains", "near", "far", "touching",
        "disconnected", "at", "not-inside",
    })
    return sorted(phrases, key=len, reverse=True)


_REL_ALT = "|".join(re.escape(x) for x in _relation_phrases())

# Subject-copula-relation-object: "A is left of B", "A is in box one".
_COPULA = re.compile(
    rf"^(?P<s>.+?)\s+(?:is|was|lies|sits|stands|exists)\s+"
    rf"(?P<r>{_REL_ALT})(?:\s+of|\s+to|\s+with)?\s+(?P<o>.+?)$", re.I
)
# Active containment / topology: "box A contains a circle".
_ACTIVE = re.compile(
    rf"^(?P<s>.+?)\s+(?P<r>contains|contain|has|holds|touches|overlaps)\s+(?P<o>.+?)$", re.I
)
# bAbI state changes: "Mary moved to the bathroom".
_MOTION = re.compile(
    r"^(?P<s>.+?)\s+(?P<r>moved to|went to|travelled to|traveled to|journeyed to|returned to|went back to)\s+(?P<o>.+?)$",
    re.I,
)
# Inverted directional form: "To the south of X is Y" => Y below X.
_INVERTED = re.compile(
    rf"^(?:to\s+the\s+)?(?P<r>{_REL_ALT})(?:\s+of|\s+to)?\s+(?P<o>.+?)\s+(?:is|lies|sits|stands)\s+(?P<s>.+?)$",
    re.I,
)
_PATH = re.compile(
    r"^(?P<s>.+?)\s+(?:is\s+|was\s+)?(?:driving|walking|running|travelling|traveling|moving)\s+"
    r"from\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)$", re.I,
)


def _direction_from_phrase(text: str) -> str | None:
    t = " ".join((text or "").lower().replace("–", "-").split())
    # StepGame uses "over there" as discourse filler, not a north relation.
    t = re.sub(r"\bover\s+there\b", "there", t)
    # Clock-face formulations used heavily by StepGame.
    clock = re.search(
        r"(?:between\s+)?(10|11|12|[1-9])(?::00)?(?:\s+and\s+(\d+))?(?!\d)", t
    )
    if clock and ("clock" in t or "position" in t or "between" in t):
        hour = int(clock.group(1)) if clock else -1
        if hour in (1, 2): return "upper-right"
        if hour in (4, 5): return "lower-right"
        if hour in (7, 8): return "lower-left"
        if hour in (10, 11): return "upper-left"
        if hour == 12: return "above"
        if hour == 3: return "right"
        if hour == 6: return "below"
        if hour == 9: return "left"

    up = bool(re.search(r"\b(?:upper|above|top|north|over|front|up)\b", t))
    down = bool(re.search(r"\b(?:lower|below|bottom|south|under|down)\b", t))
    left = bool(re.search(r"\b(?:left|lefthand|west)\b", t))
    right = bool(re.search(r"\b(?:right|righthand|east)\b", t))
    if up and right and not down and not left: return "upper-right"
    if up and left and not down and not right: return "upper-left"
    if down and right and not up and not left: return "lower-right"
    if down and left and not up and not right: return "lower-left"
    if up and not down: return "above"
    if down and not up: return "below"
    if left and not right: return "left"
    if right and not left: return "right"
    return None


def _symbol_entities(text: str) -> List[str]:
    out = []
    for x in re.findall(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])", text):
        if x not in out:
            out.append(x)
    return out


def _parse_stepgame_style(s: str, source: str) -> List[CanonicalRelation]:
    """Parse StepGame's large paraphrase inventory without using gold labels."""
    entities = _symbol_entities(s)
    if len(entities) != 2:
        return []
    first, second = entities

    # Pure co-occurrence statements assert no spatial edge.
    if re.fullmatch(
        r"(?:the\s+objects?\s+)?[A-Z](?:\s+and\s+[A-Z])?\s+"
        r"(?:are|is)\s+(?:both\s+)?(?:over\s+)?there", s, re.I,
    ):
        return []

    # "V is over there and G is directly above it": G is the located object
    # and V is its reference.  Handle before generic pair parsing.
    there = re.match(
        r"^([A-Z])\s+is\s+(?:over\s+)?there\s+(?:and|with)\s+([A-Z])\s+"
        r"(?:is\s+)?(.+)$", s, re.I,
    )
    if there:
        rel = _direction_from_phrase(there.group(3))
        if rel:
            return [CanonicalRelation(
                there.group(2).lower(), rel, there.group(1).lower(), s, 0.98, source
            )]

    # "If H is the center of a clock face, T is located between 2 and 3."
    m = re.search(
        r"if\s+([A-Z])\s+is\s+the\s+center.*?,\s*([A-Z])\s+is\s+located\s+(.+)$", s, re.I
    )
    if m:
        rel = _direction_from_phrase(m.group(3))
        return [CanonicalRelation(m.group(2).lower(), rel, m.group(1).lower(), s, 0.98, source)] if rel else []

    # Prefer an explicit relational clause embedded in a two-object filler,
    # e.g. "P and G are both there with the object G below the object P".
    # Using the first-mentioned entity for such templates reverses the edge.
    direct = re.search(
        r"(?:the\s+object\s+)?((?-i:[A-Z]))\s+(?:is\s+)?"
        r"((?:directly\s+)?(?:above|below|under|over|to\s+the\s+right\s+of|"
        r"to\s+the\s+left\s+of|on\s+the\s+right\s+of|on\s+the\s+left\s+of))\s+"
        r"(?:the\s+object\s+|object\s+)?((?-i:[A-Z]))\b", s, re.I,
    )
    if direct and re.search(r"\b(?:and|with)\b", s, re.I):
        rel = _direction_from_phrase(direct.group(2))
        if rel:
            return [CanonicalRelation(
                direct.group(1).lower(), rel, direct.group(3).lower(), s, 0.98, source
            )]

    positioned_pair = re.search(
        r"\b((?-i:[A-Z]))\s+(?:is\s+)?(?:at|on)\s+(?:the\s+)?"
        r"(top|bottom|left|right)\s+(?:and\s+)?((?-i:[A-Z]))\b", s, re.I,
    )
    if positioned_pair:
        rel = _direction_from_phrase(positioned_pair.group(2))
        if rel:
            return [CanonicalRelation(
                positioned_pair.group(1).lower(), rel, positioned_pair.group(3).lower(),
                s, 0.97, source,
            )]

    # Pair descriptions state both endpoints. Prefer the descriptor attached to
    # the first-mentioned entity; otherwise invert the second's descriptor.
    if re.search(r"\b(?:and|with)\b", s, re.I):
        positions = {}
        for entity in (first, second):
            matches = list(re.finditer(
                rf"\b{entity}\b\s+(?:is\s+)?(?:slightly\s+off\s+center\s+to\s+)?"
                r"(?:on|at|to)?\s*(?:the\s+)?"
                r"(upper right|upper left|lower right|lower left|top left|top right|"
                r"bottom left|bottom right|top|bottom|left|right)",
                s, re.I,
            ))
            if matches:
                positions[entity] = _direction_from_phrase(matches[-1].group(1))
        if first in positions and second in positions:
            return [CanonicalRelation(first.lower(), positions[first], second.lower(), s, 0.95, source)]

    # Normal subject-first descriptions. Use the text after the subject and
    # reject pure co-occurrence statements such as "A and B are there".
    if re.search(r"\b(?:are|is)\s+(?:both\s+)?(?:over\s+)?there\b", s, re.I) and not _direction_from_phrase(s):
        return []
    subject = first
    obj = second
    direction_text = s
    # In "X is there and Y is at 2", Y is the directional subject.
    m = re.search(r"\b([A-Z])\b\s+is\s+there\s+and\s+([A-Z])\b\s+is\s+(.+)$", s, re.I)
    if m:
        subject, obj, direction_text = m.group(2), m.group(1), m.group(3)
    else:
        start = re.match(r"^(?:the\s+object(?:\s+labeled)?\s+)?([A-Z])\b(.+)$", s, re.I)
        if start:
            subject = start.group(1)
            obj = second if subject.upper() == first else first
            direction_text = start.group(2)
    rel = _direction_from_phrase(direction_text)
    if not rel:
        return []
    return [CanonicalRelation(subject.lower(), rel, obj.lower(), s, 0.94, source)]


def _make(match: re.Match, statement: str, source: str, confidence: float) -> CanonicalRelation | None:
    rel = canonicalize_relation(match.group("r"))
    subject = normalize_entity(match.group("s"))
    obj = normalize_entity(match.group("o"))
    if not rel or not subject or not obj or subject == obj and rel != "overlap":
        return None
    return CanonicalRelation(subject, rel, obj, statement, confidence, source)


def parse_statement(statement: str, source: str = "claim") -> List[CanonicalRelation]:
    s = (statement or "").strip(" \t\r\n.,;:!?\"'")
    s = _STEP_PREFIX.sub("", s)
    s = re.sub(r"^(?:therefore|thus|hence|so|combining these,?)\s+", "", s, flags=re.I)
    if not s:
        return []

    stepgame = _parse_stepgame_style(s, source)
    if stepgame:
        return stepgame

    path = _PATH.match(s)
    if path:
        subject = normalize_entity(path.group("s"))
        return [
            CanonicalRelation(subject, "starts-at", normalize_entity(path.group("origin")), s, 0.98, source),
            CanonicalRelation(subject, "ends-at", normalize_entity(path.group("destination")), s, 0.98, source),
        ]

    # Handle conjunctions such as "A is left of and above B" by expanding the
    # shared subject/object structure before trying the simple patterns.
    conj = re.match(
        rf"^(?P<s>.+?)\s+(?:is|was)\s+(?P<r1>{_REL_ALT})\s+and\s+(?P<r2>{_REL_ALT})"
        rf"(?:\s+of|\s+to)?\s+(?P<o>.+?)$", s, flags=re.I,
    )
    if conj:
        out = []
        for key in ("r1", "r2"):
            rel = canonicalize_relation(conj.group(key))
            if rel:
                out.append(CanonicalRelation(
                    normalize_entity(conj.group("s")), rel, normalize_entity(conj.group("o")),
                    s, 0.95, source,
                ))
        return out

    for pattern, confidence in ((_MOTION, 1.0), (_COPULA, 0.95), (_ACTIVE, 0.95), (_INVERTED, 0.9)):
        m = pattern.match(s)
        if m:
            parsed = _make(m, s, source, confidence)
            return [parsed] if parsed else []
    return []


def parse_relations(text: str | Iterable[str], source: str = "claim") -> List[CanonicalRelation]:
    statements = list(text) if not isinstance(text, str) else split_statements(text)
    out: List[CanonicalRelation] = []
    seen = set()
    for statement in statements:
        for relation in parse_statement(str(statement), source=source):
            key = (relation.subject, relation.relation, relation.object)
            if key not in seen:
                seen.add(key)
                out.append(relation)
    return out


def parse_query_entities(question: str) -> tuple[str, str] | None:
    """Extract ordered (subject, reference) entities from common QA templates."""
    q = (question or "").strip()
    patterns = (
        r"relation\s+of\s+(?:the\s+)?(?:agent|object)?\s*(.+?)\s+to\s+(?:the\s+)?(?:agent|object)?\s*(.+?)[?]?$",
        r"where\s+is\s+(.+?)\s+relative\s+to\s+(.+?)[?]?$",
        r"what\s+is\s+the\s+position\s+of\s+(.+?)\s+(?:relative\s+)?to\s+(.+?)[?]?$",
    )
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.I)
        if m:
            return normalize_entity(m.group(1)), normalize_entity(m.group(2))
    return None
