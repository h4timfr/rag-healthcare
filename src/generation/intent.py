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
    # Document-scoped only. Bare "\beach\b"/"\bevery\b" also matched ordinary
    # content-level questions - "What were the findings for each diagnostic
    # category?" - where "each" refers to something INSIDE the document, not
    # to separate documents. That misclassification activated the
    # per-document ANSWER STYLE (build_answer_style() reads EACH_DOCUMENT
    # straight off classify_intent(), unguarded by document_inventory) on a
    # single-document question with no document list to draw section names
    # from, and the model fabricated headings like "Pediatric Report.pdf:"
    # to comply with it. Requiring a document noun immediately after
    # "each"/"every" keeps "each selected document"/"every document?"
    # matching while "each diagnostic category"/"each treatment category"
    # no longer do.
    re.compile(r"\b(each|every)\s+(selected\s+)?(document|documents|report|reports|file|files|pdf|pdfs)\b"),
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

# A question that is ONLY a document reference - "what about the third
# document?", "tell me about the Pediatric Report" - asks for that
# document's contents and nothing narrower.
#
# It needs its own rule because once the reference itself is resolved (by
# src/retrieval/document_reference.py) there is no question left in the
# words. Given the factual answer style, llama3.2:3b answered "what about
# the third document?" by copying the first evidence block back out
# verbatim, labels and all - it had a document and no question to ask of it.
# Treating the reference as a request for that document's contents is what
# the user meant, and the summary style already forbids the regurgitation.
_REFERENCE_FILLER = (
    r"(?:the|that|this|those|other|next|last|same|a|an|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|\d+(?:th)?)"
)

_BARE_DOCUMENT_REFERENCE_PATTERNS = (
    # "what about the third document?" / "how about that report?"
    re.compile(
        rf"^\s*(?:what|how)\s+about\s+(?:{_REFERENCE_FILLER}\s+)*"
        r"(?:document|report|file|pdf)s?\s*\??\s*$"
    ),
    # "tell me about the Pediatric Report." - any reference, but it must be
    # a document being asked about, not a topic within one.
    re.compile(r"^\s*tell\s+me\s+about\b.*\b(?:document|report|file|pdf)s?\s*[.?]?\s*$"),
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


# ---------------------------------------------------------------------------
# Multi-component factual detection.
#
# is_multi_part() above finds a question phrased as two or more separate
# QUESTIONS joined by "and"/"as well as" - "what medications ..., and what
# condition ...". That is a different shape from a single factual question
# that still asks for several distinct pieces of information in one answer:
# "what were the findings for each diagnostic category?" or "what were the
# findings regarding X, Y, and Z?" are each one grammatical question - there
# is no interrogative-led second half for split_question_parts() to find -
# yet the answer genuinely has several required parts: one per category
# value, or one per named topic. Reusing is_multi_part()/MULTI_PART here was
# considered and rejected: MULTI_PART's answer style is a numbered
# "Part 1: ... Part 2: ..." plan built by decomposing the question into
# STANDALONE questions, and neither shape above decomposes into standalone
# questions - there is nothing to number, only components not to omit.
# requests_multiple_components() is a narrower, purely lexical signal for
# exactly that, so a FACTUAL question's answer style can ask for completeness
# across components instead of brevity - see prompt_builder.build_answer_style.
# ---------------------------------------------------------------------------

# A bare "each"/"every" that reaches this point already failed
# _EACH_DOCUMENT_PATTERNS (checked earlier, with higher priority - see
# classify_intent), so it is not asking for one section per selected
# document. It is asking about multiple values of something INSIDE the
# evidence (each diagnostic category, each treatment type, every visit),
# which is exactly the signal that the answer must enumerate more than one
# thing.
_EACH_OR_EVERY = re.compile(r"\b(each|every)\b")

# A comma-separated list of three or more named items ending in "..., and X"
# / "..., or X" - e.g. "examinations, laboratory tests, and treatments".
# Requires at least two commas (three items), not a bare "X and Y", so
# ordinary two-item conjunctions ("mother and father", "nausea and
# vomiting") are not mistaken for an enumerated list of requested topics; a
# genuine three-item list is a much stronger, still fully generic signal
# that several distinct things were named.
_COMPONENT_LIST = re.compile(r"[A-Za-z][\w\s]*?,\s*(?:[A-Za-z][\w\s]*?,\s*)+(?:and|or)\s+[A-Za-z]")


def requests_multiple_components(query: str) -> bool:
    """Whether a FACTUAL question's answer must cover more than one distinct
    component - either an implied enumeration ("each"/"every" something,
    document-scoped "each document" having already been ruled out by
    classify_intent's own priority order) or an explicit three-or-more-item
    list of named topics.

    Deliberately generic: no document- or domain-specific vocabulary, so it
    applies the same way to any document's content, not just one project's
    test corpus.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    return bool(_EACH_OR_EVERY.search(q) or _COMPONENT_LIST.search(q))


# "most/least/top frequent|common|prevalent|popular <0-2 filler words>
# <plural noun>" - e.g. "most frequent treatment categories", "least common
# causes", "top prevalent complaint types". The final word must itself look
# plural (ends in "s"), which is what keeps a singular-answer question like
# "what was the most frequent complaint?" (one item wanted) from matching -
# deliberately generic wording, no domain vocabulary, no item names.
_ENUMERATION_RANKING = re.compile(
    r"\b(?:most|least|top)\s+(?:frequent|common|prevalent|popular)\s+"
    r"(?:\w+\s+){0,2}?(\w+s)\b"
)


def is_enumeration_question(query: str) -> bool:
    """Whether a FACTUAL question's answer must be a genuine enumerated list
    - and, where frequency/ranking language is used, correctly ORDERED -
    rather than a single fact.

    requests_multiple_components() above covers two shapes whose answer
    must name several distinct things: an explicit "X, Y, and Z" list, and
    a bare "each/every <noun>" enumeration. This adds a third: a
    most/least-frequent RANKING question with no explicit list and no
    "each"/"every" in sight ("what were the most frequent treatment
    categories?") whose answer still has to enumerate - and order - several
    items, not one. All three shapes are ORed together here so every
    FACTUAL question whose answer must be a list, named/implied/ranked, is
    covered by one signal.
    """
    if requests_multiple_components(query):
        return True
    q = (query or "").strip().lower()
    if not q:
        return False
    return bool(_ENUMERATION_RANKING.search(q))


# An explicit request for a number of items: "give 3-5 specific reasons",
# "list two causes", "name 3 risk factors". A count word, an optional range,
# up to two filler words, then a plural noun - the plural is what keeps
# "give 1 reason" out. Generic wording only: no domain vocabulary.
_COUNT_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
_ITEM_COUNT_REQUEST = re.compile(
    r"\b(?:give|list|name|provide|identify|state|cite|mention|offer)\b[^.?!]{0,20}?\b"
    + _COUNT_WORD
    + r"(?:\s*(?:-|–|to|or)\s*"
    + _COUNT_WORD
    + r")?\s+(?:\w+\s+){0,2}?(\w+s)\b"
)


# A comparison between two PARTS of a document - "the initial problem list with
# the revised problem list", "the two sections", "the first and second
# versions" - as opposed to a comparison between two concepts ("how does
# keyword retrieval differ from embedding retrieval?"). Only the former should
# get the section-identity answer style: on a concept comparison that style
# returned an empty answer in 2 of 3 runs, while the original style answered
# 3/3.
_DOCUMENT_PART_NOUN = re.compile(r"\b(?:lists?|sections?|versions?|tables?|drafts?|revisions?)\b")


def compares_document_sections(query: str) -> bool:
    """Whether a comparison question is about parts of the document itself."""
    return bool(_DOCUMENT_PART_NOUN.search((query or "").lower()))


_MEDICATION_REQUEST = re.compile(r"\b(?:medications?|medicines?|drugs?|prescriptions?|prescribed)\b")


def asks_for_medications(query: str) -> bool:
    """Whether the question asks for medications - a category whose answer
    must not drift into tests, procedures or lifestyle advice."""
    return bool(_MEDICATION_REQUEST.search((query or "").lower()))


def requests_item_count(query: str) -> bool:
    """Whether the question explicitly asks for a number of items.

    Kept separate from is_enumeration_question() on purpose: that function
    also drives the structured-question research diagnostic, and widening it
    would change that study's flags. Only the production answer style and
    output budget read this. Measured need: "... Give 3-5 specific reasons"
    was classified FACTUAL, so its prompt said "in one or two sentences" and
    2-3 of 5 answers came back as a single sentence."""
    q = (query or "").strip().lower()
    return bool(q) and bool(_ITEM_COUNT_REQUEST.search(q))


# First-person requests for advice about the user's own (or a dependent's)
# health: "should I ...", "for me/my ...", "my child ...", "I take ...".
# Document questions are phrased about the patients in the documents, so
# this is deliberately narrow; checked against all 68 product-QA and
# benchmark questions with no matches.
_PERSONAL_ADVICE = re.compile(
    r"\b(should|can|could|may|must|do|shall) (i|we)\b"
    r"|\b(my|our) (child|son|daughter|baby|kid|kids|children|mother|mom|father|dad|"
    r"wife|husband|partner|parent|parents|family|own)\b"
    r"|\bfor (me|my|us|our)\b"
    r"|\bi (have|take|am taking|was prescribed|need|feel)\b"
    r"|\bi'm (taking|pregnant|on)\b",
    flags=re.IGNORECASE,
)


def is_personal_advice_request(query: str) -> bool:
    """Whether the question asks about the user's own health, rather than
    about what the documents report."""
    return bool(_PERSONAL_ADVICE.search(query or ""))


# ---------------------------------------------------------------------------
# Decomposition for follow-up-shaped multi-part questions.
#
# split_question_parts() above only finds parts joined by "and"/"as well as"
# inside one sentence. It never split "What happened during the second
# admission? What about the first one?" - two separate sentences, the second
# a bare tag question - so that question retrieved as a single blended
# embedding of both admissions, and the model answered mostly about
# whichever one dominated it. decompose_question() below finds that shape
# too, and resolves the elliptical second part ("the first one") into a
# standalone question by reusing the first part's own phrasing - the same
# information a person tracking the conversation would use, applied
# deterministically instead of via another model call.
# ---------------------------------------------------------------------------

# A second sentence that is itself a tag question: "...? What about the
# first one?", "...? How about the third?". Splits only right before it, so
# an ordinary multi-sentence answer-shaped question is not touched.
_TAG_QUESTION_SPLIT = re.compile(
    r"(?<=[?.])\s+(?=(?:what|how)\s+about\b)",
    flags=re.IGNORECASE,
)

# A part (or a whole standalone question) that names nothing but a position:
# "what about the first?", "how about the second one?". Capture group 1 is
# the ordinal it asks for.
_ELLIPTICAL_REFERENCE = re.compile(
    r"^\s*(?:what|how)\s+about\s+the\s+"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)"
    r"(?:\s+one)?\s*\??\s*$",
    flags=re.IGNORECASE,
)

# The "<ordinal> <noun>" phrase an elliptical reference borrows its noun
# from, e.g. "second admission" in "What happened during the second
# admission?" - deliberately the same ordinal vocabulary as
# _ELLIPTICAL_REFERENCE, so a resolved reference only ever substitutes one
# ordinal for another rather than inventing wording.
_ORDINAL_ANCHOR = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\b\s+(\w+)",
    flags=re.IGNORECASE,
)


def _resolve_elliptical_reference(part: str, anchor_text: str | None) -> str:
    """Resolve a bare ordinal-reference fragment into a standalone question
    by substituting its ordinal into `anchor_text`'s own "<ordinal> <noun>"
    phrase:

        anchor: "What happened during the second admission?"
        part:   "What about the first one?"
        ->      "What happened during the first admission?"

    Returns `part` unchanged if it is not an elliptical reference, or if no
    anchor (or no substitutable ordinal+noun within it) is available - this
    never guesses at content it has no basis for.
    """
    match = _ELLIPTICAL_REFERENCE.match(part.strip())
    if not match or not anchor_text:
        return part

    new_ordinal = match.group(1)
    anchor_match = _ORDINAL_ANCHOR.search(anchor_text)
    if anchor_match is None:
        return part

    old_ordinal = anchor_match.group(1)
    return re.sub(
        rf"\b{re.escape(old_ordinal)}\b",
        new_ordinal,
        anchor_text,
        count=1,
        flags=re.IGNORECASE,
    )


def decompose_question(query: str, history: list[dict] | None = None) -> list[str]:
    """Split `query` into standalone sub-questions, resolving an elliptical
    ordinal reference ("what about the first?") against the nearest
    available anchor before it is used for retrieval or shown to the model.

    Two splitting passes, applied in order:
      1. split_question_parts() - "X, and Y" - unchanged from today.
      2. _TAG_QUESTION_SPLIT on each resulting piece - "X? What about Y?" -
         so a trailing tag question is never left stuck to the part before
         it, whichever pass produced that part.

    Every part after the first that matches _ELLIPTICAL_REFERENCE is
    resolved against the PRECEDING part (already-resolved, so a chain of
    several - "the second? the first? the third?" - each resolve against
    the nearest one with real content).

    If splitting finds only one part and IT is itself an elliptical
    reference ("What about the first?" as the entire message), it is
    resolved against the most recent user turn in `history` instead - the
    only case where history reaches this function, and only to supply the
    anchor phrase, never to be embedded or searched itself (see
    retrieve_for_question(), which is the only caller that matters for
    retrieval).

    Returns a single-element list for an ordinary question, exactly like
    split_question_parts(), so `len(...) > 1` means "multi-part" either way.
    """
    query = query or ""
    parts = split_question_parts(query)

    expanded: list[str] = []
    for part in parts:
        tagged = [p.strip() for p in _TAG_QUESTION_SPLIT.split(part) if p and p.strip()]
        expanded.extend(tagged if len(tagged) > 1 else [part])
    parts = expanded

    if len(parts) > 1:
        resolved = [parts[0]]
        for part in parts[1:]:
            resolved.append(_resolve_elliptical_reference(part, resolved[-1]))
        return resolved

    if history:
        last_user = next(
            (str(turn.get("content", "")) for turn in reversed(history) if turn.get("role") == "user"),
            None,
        )
        if last_user:
            return [_resolve_elliptical_reference(query, last_user)]

    return parts


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

    # Checked before follow_up: "what about the second document?" IS a
    # follow-up, but its answer shape is a summary of that document, and
    # resolving what it points at is now the reference resolver's job rather
    # than something the answer style has to talk the model through.
    if _matches(_BARE_DOCUMENT_REFERENCE_PATTERNS, q):
        return SUMMARY

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
