"""Deterministic resolution of a document reference in a question to the
selected documents it names.

Retrieval matches a query against chunk TEXT. Filenames live in metadata and
appear in no chunk, so a question that names a document cannot steer
retrieval toward it: "What medications are listed in the OMC report?" asked
over three selected documents retrieved five Pediatric chunks and answered
with a Pediatric medication - faithful to its evidence, and about the wrong
document. The same gap makes "what about the third document?" unanswerable
except by luck, because an ordinal carries no content words at all.

This module closes that gap before retrieval runs, using only the current
inventory and string matching. No model call: a document reference is a
naming problem, not a comprehension problem, and a deterministic answer is
one that can be tested and explained.

Resolution is tried in order of how explicit the reference is:

    filename   the filename or its basename appears in the question
    ordinal    "the second document", "the first and third documents"
    keyword    a filename's identifying word plus a document noun, where
               that word identifies exactly one selected document
    history    a bare demonstrative ("that report") whose antecedent in the
               recent conversation is exactly one document

Two rules keep it honest. Every candidate comes from the inventory passed
in, so a reference can only ever resolve to a currently-selected document -
never to a deleted or unselected one. And a reference that fits more than
one document resolves to NONE of them: it reports itself ambiguous and the
caller keeps its existing behaviour, rather than picking one and being
quietly wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text_match import document_keywords

# Words that mean "the document itself" rather than its contents. A
# reference needs one of these before an ordinal or an identifying word is
# read as naming a document - otherwise "the second hospital admission"
# becomes "the second document", and a question about paediatric care
# becomes a question about Pediatric Report.pdf.
_DOCUMENT_NOUNS = (
    "document",
    "documents",
    "report",
    "reports",
    "file",
    "files",
    "pdf",
    "pdfs",
    "paper",
    "papers",
)

_ORDINAL_WORDS = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
    "sixth": 5,
    "6th": 5,
    "seventh": 6,
    "7th": 6,
    "eighth": 7,
    "8th": 7,
    "ninth": 8,
    "9th": 8,
    "tenth": 9,
    "10th": 9,
}

# "that report", "this document", "it" - a reference that only means
# something relative to what was just discussed.
_DEMONSTRATIVE = re.compile(
    r"\b(that|this|the same|it|its)\b", re.IGNORECASE
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class DocumentReference:
    """What a question's document reference resolved to.

    `documents` are inventory entries in inventory order, empty when nothing
    resolved. `kind` names the rule that matched, for explainability.
    `ambiguous` is True when a reference WAS detected but fits more than one
    selected document; `detail` then says which, so a caller can surface it.
    """

    documents: tuple[dict, ...] = ()
    kind: str = "none"
    ambiguous: bool = False
    detail: str = ""

    def __bool__(self) -> bool:
        return bool(self.documents)

    @property
    def document_ids(self) -> list[str]:
        return [d["document_id"] for d in self.documents]

    @property
    def sources(self) -> list[str]:
        return [d["source"] for d in self.documents]


def _normalise(text: str) -> str:
    """Lowercase with every run of non-alphanumerics collapsed to one space,
    so "RAG_Healthcare_Review1.pdf" and "RAG Healthcare Review1" compare
    equal - filenames get typed back with different punctuation."""
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def _has_document_noun(query: str) -> bool:
    lowered = f" {_normalise(query)} "
    return any(f" {noun} " in lowered for noun in _DOCUMENT_NOUNS)


# A basename shorter than this is not a usable reference on its own. The
# filename "a.pdf" reduces to the token "a", which is a word people write
# constantly - matched loosely it claimed "Summarize each selected document
# separately" for itself.
_MIN_BASENAME_CHARS = 3


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    """Whether `needle` appears in `haystack` as consecutive whole tokens.

    Whole tokens, not a raw substring: matching "a.pdf" by substring finds
    the letter "a" inside "separately", and every question then belongs to
    that document.
    """
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def _by_filename(query: str, inventory: list[dict]) -> list[dict]:
    """Documents whose filename or basename appears in the question, as
    whole words and in order, so punctuation and the extension are optional
    ("RAG_Healthcare_Review1.pdf" is named by "the RAG Healthcare Review1
    document")."""
    query_tokens = _tokens(query)
    matches = []
    for doc in inventory:
        source = doc["source"]
        basename = re.sub(r"\.[a-z0-9]+$", "", source, flags=re.IGNORECASE)
        candidates = [source]
        if len(_normalise(basename)) >= _MIN_BASENAME_CHARS:
            candidates.append(basename)
        if any(_contains_token_sequence(query_tokens, _tokens(c)) for c in candidates):
            matches.append(doc)
    return matches


def _by_ordinal(query: str, inventory: list[dict]) -> list[dict]:
    """Documents named by position: "the second document", "the first and
    third documents".

    Every ordinal in the question is collected rather than only one adjacent
    to the noun, because "the first and third documents" attaches the noun
    to the last ordinal only. Out-of-range positions are dropped, not
    clamped - "the fifth document" of three names nothing, and guessing
    would be worse than saying so.
    """
    if not _has_document_noun(query):
        return []

    positions = []
    for word in _normalise(query).split():
        position = _ORDINAL_WORDS.get(word)
        if position is not None and position not in positions:
            positions.append(position)

    if re.search(r"\blast\b", _normalise(query)) and inventory:
        last = len(inventory) - 1
        if last not in positions:
            positions.append(last)

    return [inventory[p] for p in sorted(positions) if p < len(inventory)]


def _by_keyword(query: str, inventory: list[dict]) -> tuple[list[dict], list[dict]]:
    """Documents named by an identifying word from their filename.

    Returns (resolved, contested). A word only resolves if it identifies
    exactly ONE document in the current inventory - "omc" does, "healthcare"
    would not if two selected files were both named "...Healthcare...". A
    contested word means the user did name something, ambiguously, which the
    caller must not guess at.

    Unlike text_match.mentions_document, which requires every identifying
    word and is right for grading a generated answer, one word is enough
    here: people write "the OMC report", not "the OMC Cardio report".
    """
    if not _has_document_noun(query):
        return [], []

    words = set(_normalise(query).split())

    owners: dict[str, list[dict]] = {}
    for doc in inventory:
        for keyword in document_keywords(doc["source"]):
            owners.setdefault(keyword, []).append(doc)

    resolved: list[dict] = []
    contested: list[dict] = []
    for keyword, docs in owners.items():
        if keyword not in words:
            continue
        if len(docs) == 1:
            if docs[0] not in resolved:
                resolved.append(docs[0])
        else:
            for doc in docs:
                if doc not in contested:
                    contested.append(doc)
    return resolved, contested


def _by_history(query: str, inventory: list[dict], history: list[dict] | None) -> list[dict]:
    """The antecedent of a bare demonstrative ("what about that report?").

    Only the most recent exchange is consulted, and only documents named in
    it count. If that exchange named several documents - as the answer to
    "summarize each selected document" does - there is no unambiguous
    antecedent, every candidate is returned, and the caller treats that as
    ambiguous rather than picking the first.
    """
    if not history or not _has_document_noun(query):
        return []
    if not _DEMONSTRATIVE.search(query):
        return []

    recent = " ".join(str(turn.get("content", "")) for turn in history[-2:])
    recent_tokens = _tokens(recent)
    recent_words = set(recent_tokens)

    named = []
    for doc in inventory:
        basename = re.sub(r"\.[a-z0-9]+$", "", doc["source"], flags=re.IGNORECASE)
        keywords = document_keywords(doc["source"])
        by_name = len(_normalise(basename)) >= _MIN_BASENAME_CHARS and _contains_token_sequence(
            recent_tokens, _tokens(basename)
        )
        if by_name or (keywords and all(k in recent_words for k in keywords)):
            named.append(doc)
    return named


def resolve_document_reference(
    query: str,
    inventory: list[dict],
    history: list[dict] | None = None,
) -> DocumentReference:
    """Resolve any document reference in `query` against `inventory`.

    Returns an empty DocumentReference when the question names no document,
    which is the common case and means "leave retrieval alone".
    """
    if not query or not inventory:
        return DocumentReference()

    # The naming tiers are combined, not tried in turn. One question can
    # name one document by filename and another by keyword, and taking only
    # the first tier that matched silently drops the rest:
    # "What do the Pediatric Report and the OMC Cardiology report have in
    # common?" matched "Pediatric Report" as a filename and stopped, so a
    # two-document comparison was scoped to one document and answered with
    # no evidence from the other at all.
    matched: list[dict] = []
    kinds: list[str] = []
    for finder, kind in ((_by_filename, "filename"), (_by_ordinal, "ordinal")):
        found = finder(query, inventory)
        if found:
            kinds.append(kind)
        for doc in found:
            if doc not in matched:
                matched.append(doc)

    resolved, contested = _by_keyword(query, inventory)
    if resolved:
        kinds.append("keyword")
    for doc in resolved:
        if doc not in matched:
            matched.append(doc)

    if matched:
        # Inventory order, so "the third and first documents" and "the first
        # and third documents" resolve identically.
        ordered = tuple(doc for doc in inventory if doc in matched)
        return DocumentReference(ordered, kind="+".join(kinds))

    # Only a question that named nothing definite can be ambiguous.
    if contested:
        return DocumentReference(
            kind="ambiguous",
            ambiguous=True,
            detail="matches " + ", ".join(d["source"] for d in contested),
        )

    from_history = _by_history(query, inventory, history)
    if len(from_history) == 1:
        return DocumentReference(tuple(from_history), kind="history")
    if len(from_history) > 1:
        return DocumentReference(
            kind="ambiguous",
            ambiguous=True,
            detail="recent conversation names " + ", ".join(d["source"] for d in from_history),
        )

    return DocumentReference()
