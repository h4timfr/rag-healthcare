"""Lightweight deterministic post-generation validation.

This does NOT rewrite, retry, or "correct" an answer - it only reports what
it can prove about the text that was generated, so the QA harness can
separate a genuine model failure from a grading artifact, and so a failure
mode can be counted instead of argued about.

The checks exist because of three failures observed in the product QA
baseline, all of which the model produced while holding the correct
evidence in its prompt:

  refusal_despite_evidence  "The text does not provide information on the
                            medications the patient was discharged on" -
                            said with all four pages of that document
                            retrieved, two of which list them.
  meta_commentary           "the context provided does not contain a
                            summary... only contains snippets from pages 3,
                            5, 6 and 8" - describing the retrieval instead
                            of answering from it.
  hedging                   "Without seeing the actual content of the
                            reports, it is difficult to say" - written
                            underneath the reports' actual content.

Every signal is a literal phrase or a counted mention. Nothing here infers
meaning, so a clean validation result is not a claim that the answer is
correct - only that it does not exhibit these specific failures.
"""

from __future__ import annotations

from ..text_match import contains_exact, contains_term, mentions_document
from . import intent as intent_module

# "I could not answer" phrasing. Broad and lowercase-matched: refusal
# wording varies a lot for a small local model.
REFUSAL_PHRASES = (
    "insufficient",
    "not supported",
    "no relevant evidence",
    "not mentioned",
    "does not mention",
    "not provided",
    "cannot find",
    "can't find",
    "no information",
    "not available",
    "not stated",
    "not state",
    "not list",
    "not report",
    "no record",
    "no results",
    "not specified",
    "does not provide",
    "doesn't provide",
    "not contain",
    "no mention",
    "unable to find",
    "context does not",
    "not indicate",
    "no data",
    "not addressed",
    "not discussed",
    "no answer",
    "unable to answer",
    "cannot answer",
    "can't answer",
    "cannot determine",
    "does not include",
    "not include information",
    # Terse forms. The model answered an unsupported question with exactly
    # "Not given." - the correct behaviour - and was scored as having failed
    # to flag it, because every phrase above is a sentence-length refusal.
    "not given",
    "not reported",
    "no such",
    "not found in",
)

# Talking about the retrieval machinery instead of answering the question.
# The user never sees "the context" and did not ask about it.
META_COMMENTARY_PHRASES = (
    "the context provided",
    "the provided context",
    "the context only",
    "the context does not",
    "based on the context",
    "in the context above",
    "retrieved chunk",
    "the chunks",
    "the snippets",
    "the provided information",
    "the provided text",
    "the text provided",
    "the supplied evidence",
    "the evidence provided",
    "the given context",
    "according to the context",
    "the passages provided",
    "the excerpts",
)

# Refusing to commit while the evidence sits in the prompt.
HEDGE_PHRASES = (
    "it is difficult to say",
    "difficult to determine",
    "without seeing",
    "without access to",
    "i cannot see",
    "i don't have access",
    "i do not have access",
    "it is unclear from",
    # "appear to be" is deliberately absent: it catches the baseline's
    # hedged comparison ("both appear to be medical reports") but also
    # ordinary clinical language ("the findings appear to be consistent
    # with"). That answer is already caught by "without seeing" and "it is
    # difficult to say", so the generic phrasing buys nothing and would
    # cost false positives.
    "may contain information",
    "it is likely that",
    "likely related to",
)


def _found(text: str, phrases) -> list[str]:
    lowered = (text or "").lower()
    return [phrase for phrase in phrases if phrase in lowered]


def _content_words(text: str) -> list[str]:
    """Words from a question part that carry enough meaning to look for in
    an answer - used only to check that a multi-part answer addressed each
    part, never to judge whether the content is correct."""
    stopwords = {
        "what", "which", "who", "whose", "when", "where", "why", "how",
        "was", "were", "is", "are", "did", "does", "do", "the", "a", "an",
        "of", "on", "in", "at", "to", "for", "and", "or", "with", "that",
        "this", "it", "its", "her", "his", "their", "she", "he", "they",
        "patient", "patients", "during", "found", "about", "from", "by",
        "besides", "also", "any", "there", "been", "be", "has", "have",
        "had", "will", "would", "can", "could", "should", "me", "you",
    }
    words = [w.strip(".,;:?!()[]\"'").lower() for w in (text or "").split()]
    return [w for w in words if len(w) > 3 and w not in stopwords]


def part_is_addressed(answer: str, part: str) -> bool:
    """Whether `answer` appears to address the question part `part`.

    Deliberately weak: a part counts as addressed if any one of its content
    words (morphology-aware) shows up in the answer. It is a missing-section
    detector, not a correctness check - a part the answer never touches at
    all is the thing worth flagging.
    """
    words = _content_words(part)
    if not words:
        return True
    return any(contains_term(answer, word) for word in words)


def validate_answer(
    answer: str | None,
    intent: str,
    *,
    evidence_sources: list[str] | None = None,
    required_documents: list[str] | None = None,
    question: str | None = None,
) -> dict:
    """Check one generated answer for the failure modes above.

    `evidence_sources` are the document names that actually reached the
    prompt for this turn - the difference between an honest "I don't have
    that" and a refusal issued on top of the answer. `required_documents`
    are the documents an each-document/comparison answer is supposed to be
    organised around. `question` enables the multi-part coverage check.

    Returns a flat, JSON-serialisable dict. `issues` is the short list a
    human should read; `ok` is simply whether it is empty.
    """
    text = (answer or "").strip()
    evidence_sources = evidence_sources or []
    required_documents = required_documents or []

    issues: list[str] = []

    is_empty = not text
    if is_empty:
        issues.append("empty_answer")

    refusal_phrases_found = _found(text, REFUSAL_PHRASES)
    meta_phrases_found = _found(text, META_COMMENTARY_PHRASES)
    hedge_phrases_found = _found(text, HEDGE_PHRASES)

    has_evidence = bool(evidence_sources)

    # A refusal is only a defect when evidence was actually supplied. With
    # no evidence retrieved, refusing is the correct behaviour.
    refusal_despite_evidence = bool(refusal_phrases_found) and has_evidence
    if refusal_despite_evidence and intent != intent_module.UNSUPPORTED:
        issues.append("refusal_despite_evidence")

    if meta_phrases_found:
        issues.append("meta_commentary")

    if hedge_phrases_found and has_evidence:
        issues.append("hedging_despite_evidence")

    # Document coverage for answers that must be organised per document.
    documents_mentioned = [d for d in required_documents if mentions_document(text, d)]
    documents_missing = [d for d in required_documents if d not in documents_mentioned]
    if intent in intent_module.MULTI_DOCUMENT_INTENTS and required_documents and documents_missing:
        issues.append("missing_document_coverage")

    # Multi-part coverage.
    parts: list[str] = []
    missing_parts: list[str] = []
    if question and intent == intent_module.MULTI_PART:
        parts = intent_module.split_question_parts(question)
        if len(parts) > 1:
            missing_parts = [p for p in parts if not part_is_addressed(text, p)]
            if missing_parts:
                issues.append("missing_answer_part")

    # An unsupported question SHOULD end in an explicit insufficiency
    # statement; silently answering it anyway is the failure.
    unsupported_not_flagged = (
        intent == intent_module.UNSUPPORTED and not refusal_phrases_found and not is_empty
    )
    if unsupported_not_flagged:
        issues.append("unsupported_not_flagged")

    return {
        "ok": not issues,
        "issues": issues,
        "empty_answer": is_empty,
        "refusal_detected": bool(refusal_phrases_found),
        "refusal_phrases": refusal_phrases_found,
        "refusal_despite_evidence": refusal_despite_evidence,
        "meta_commentary": bool(meta_phrases_found),
        "meta_commentary_phrases": meta_phrases_found,
        "hedging": bool(hedge_phrases_found),
        "hedge_phrases": hedge_phrases_found,
        "evidence_document_count": len(set(evidence_sources)),
        "documents_required": list(required_documents),
        "documents_mentioned": documents_mentioned,
        "documents_missing": documents_missing,
        "parts_expected": len(parts),
        "parts_missing": missing_parts,
        "answer_chars": len(text),
    }


def is_refusal(answer: str | None) -> bool:
    """Whether an answer reads as "I could not answer that" - the single
    shared definition, used by the QA grader's unsupported-category check
    as well as by validate_answer()."""
    return any(contains_exact(answer or "", phrase) for phrase in REFUSAL_PHRASES)
