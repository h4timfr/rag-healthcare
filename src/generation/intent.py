"""Deterministic (non-LLM) question-intent classification for the production
answer path.

The intent decides how the generation prompt is shaped (which ANSWER STYLE
block is attached, whether a multi-part answer plan is built) and whether
multi-document retrieval balances evidence across the selected documents.
It is pure string/regex work: no model call, no extra latency, fully
testable, and the same question always yields the same intent.

On UNSUPPORTED: it is part of the vocabulary but is NOT lexically
detectable. Checked against this project's own QA dataset, the unsupported
questions ("What was the patient's blood type?", "What chemotherapy regimen
did the patient receive?") have exactly the same surface form as answerable
factual/numeric questions - only the absence of supporting evidence makes
them unsupported. So classify_intent() returns UNSUPPORTED on the one
deterministic signal that actually exists at classification time - no
evidence was retrieved - and otherwise returns the question's shape. An
answerable-looking question whose evidence turns out not to contain the
answer is handled by the evidence-first generation rules (prompt_builder)
and detected afterwards by validation.validate_answer().
"""

import re

FACTUAL = "factual"
NUMERIC = "numeric"
SUMMARY = "summary"
EACH_DOCUMENT = "each_document"
COMPARISON = "comparison"
MULTI_PART = "multi_part"
FOLLOW_UP = "follow_up"
UNSUPPORTED = "unsupported"

ALL_INTENTS = (
    FACTUAL,
    NUMERIC,
    SUMMARY,
    EACH_DOCUMENT,
    COMPARISON,
    MULTI_PART,
    FOLLOW_UP,
    UNSUPPORTED,
)

# Intents whose answer must cover several documents - these are the ones
# that get document-balanced retrieval allocation and a per-document
# answer structure.
MULTI_DOCUMENT_INTENTS = (EACH_DOCUMENT, COMPARISON)

# A follow-up only exists relative to an earlier turn, so these are only
# consulted when conversation history is actually present.
_FOLLOW_UP_PATTERNS = (
    re.compile(r"\byou (just )?(say|said|mention|mentioned|note|noted|wrote|write|told|tell)\b"),
    re.compile(r"\bwhat about\b"),
    re.compile(r"\bhow about\b"),
    # "the second document" / "that one" - but deliberately NOT bare
    # ordinals like "her second hospital admission", which are ordinary
    # content words, not references to a previous answer.
    re.compile(r"\b(first|second|third|fourth|last|other|next)\s+(document|one|report|file|pdf)\b"),
    re.compile(r"^\s*(and|also|what else|anything else)\b"),
    re.compile(r"\b(that|this)\s+(one|document|report)\b"),
)

_EACH_DOCUMENT_PATTERNS = (
    re.compile(r"\beach\b"),
    re.compile(r"\bevery\b"),
    re.compile(r"\ball (of )?(the )?(selected )?(documents|reports|files|pdfs)\b"),
    re.compile(r"\bseparately\b"),
    re.compile(r"\bper document\b"),
    re.compile(r"\bdocument by document\b"),
)

_COMPARISON_PATTERNS = (
    re.compile(r"\bcompar"),  # compare, comparison, comparing
    re.compile(r"\bdiffer"),  # differ, differs, difference, different
    re.compile(r"\bin common\b"),
    re.compile(r"\bsimilarit"),
    re.compile(r"\bsimilar to\b"),
    re.compile(r"\bversus\b"),
    re.compile(r"\bvs\.?\b"),
    re.compile(r"\bboth (documents|reports)\b"),
    re.compile(r"\bcontrast\b"),
)

_SUMMARY_PATTERNS = (
    re.compile(r"\bsummar(ise|ize|y|ies)\b"),
    re.compile(r"\boverview\b"),
    re.compile(r"\bwhat is (this|the) (document|report|file|pdf) about\b"),
    re.compile(r"\bwhat does (this|the) (document|report) (say|cover|contain)\b"),
    re.compile(r"\bgist\b"),
    re.compile(r"\bmain (points|purpose|topic)\b"),
)

# Deliberately keyword-driven rather than "contains a digit": questions like
# "the 2-year well check" or "the January 12 visit" carry digits but are
# ordinary factual questions, and answering them as if a bare number were
# wanted degrades the answer.
_NUMERIC_PATTERNS = (
    re.compile(r"\bhow (many|much|long|old|often|far|big|large)\b"),
    re.compile(r"\bon what date\b"),
    re.compile(r"\bwhat (date|size|percentage|percent|value|number|dose|dosage|rate|score)\b"),
    re.compile(r"\b(percentage|percent|dosage)\b"),
    re.compile(r"\bsize in\b"),
    re.compile(r"\bnumber of\b"),
    re.compile(r"\bhow many\b"),
    re.compile(r"\bin (centimeters|centimetres|cm|mm|millimeters|mg|ml|kg|pounds|lbs)\b"),
    re.compile(r"\bwhat was the (size|length|width|diameter|weight|height|count|total)\b"),
)

# Second half of a multi-part question: ", and what ...", "and how many ...",
# "; and list ...". Requires an interrogative or an imperative verb after the
# connector so ordinary conjunctions ("the mother and father", "aneurysm and
# rupture") are not mistaken for a second requested component.
_PART_SPLIT = re.compile(
    r"[,;]?\s+(?:and|as well as)\s+"
    r"(?=(?:what|which|who|whose|when|where|why|how|did|does|do|is|are|was|were|"
    r"list|describe|explain|summari[sz]e|tell|give|provide|state|identify|name)\b)",
    flags=re.IGNORECASE,
)


def split_question_parts(query: str) -> list[str]:
    """Split a question into its separately-requested components.

    Returns a single-element list for an ordinary question, so callers can
    treat `len(...) > 1` as "this is multi-part" without a second predicate.
    """
    parts = [part.strip(" ,;") for part in _PART_SPLIT.split(query or "")]
    return [part for part in parts if part] or [query]


def is_multi_part(query: str) -> bool:
    return len(split_question_parts(query)) > 1


def _matches(patterns, text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_intent(
    query: str,
    history: list[dict] | None = None,
    has_evidence: bool = True,
) -> str:
    """Classify `query` into one of ALL_INTENTS.

    Priority order is deliberate and is what makes the classifier
    predictable on real questions:

    unsupported  no evidence was retrieved at all - nothing can be answered
    follow_up    only when history exists AND the question points back at it
    each_document  "each"/"every"/"separately" beats "summarize", so
                 "Summarize each selected document separately" is a
                 per-document answer, not one blended summary
    comparison   "compare"/"in common"/"differ"
    multi_part   two or more separately-requested components
    summary      "summarize"/"overview"/"main purpose"
    numeric      an explicit quantity/date/measurement is being asked for
    factual      everything else (the default)
    """
    q = (query or "").strip().lower()

    if not has_evidence:
        return UNSUPPORTED
    if not q:
        return FACTUAL

    if history and _matches(_FOLLOW_UP_PATTERNS, q):
        return FOLLOW_UP
    if _matches(_EACH_DOCUMENT_PATTERNS, q):
        return EACH_DOCUMENT
    if _matches(_COMPARISON_PATTERNS, q):
        return COMPARISON
    if is_multi_part(q):
        return MULTI_PART
    if _matches(_SUMMARY_PATTERNS, q):
        return SUMMARY
    if _matches(_NUMERIC_PATTERNS, q):
        return NUMERIC
    return FACTUAL


def needs_document_balance(query: str, history: list[dict] | None = None) -> bool:
    """Whether retrieval should deliberately spread its evidence budget
    across every selected document for this question.

    True for each-document and comparison questions, and for a follow-up
    that is itself one of those ("and compare the other two?") - which is
    why the question is re-classified without the history signal rather
    than stopping at FOLLOW_UP.
    """
    if classify_intent(query, history=history) in MULTI_DOCUMENT_INTENTS:
        return True
    return classify_intent(query, history=None) in MULTI_DOCUMENT_INTENTS
