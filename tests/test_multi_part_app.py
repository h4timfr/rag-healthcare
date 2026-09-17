"""End-to-end regression coverage, at the app.retrieve_and_build_prompt()
level, for the exact production failure live testing exposed:

    "What happened during the second admission? What about the first one?"

retrieved evidence that did not clearly cover both requested parts, and the
answer was mostly about one admission - because retrieve_and_build_prompt()
embedded and searched the whole compound question as a single query. These
tests exercise the real wiring (app.retrieve_and_build_prompt ->
retrieve_for_question -> app's own, patchable retrieve_hybrid -> build_prompt)
with a fake retrieve_hybrid, so the fix is proven at the same call site the
Streamlit UI actually uses, not only in the two modules it's built from.
"""

import pytest

import app
from src.exceptions import InsufficientRetrievalError


def _chunk(chunk_id, document_id, source, page, text, rank=1):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": 1.0 / rank,
        "rank": rank,
    }


def test_regression_second_admission_then_what_about_the_first_one(monkeypatch):
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        if "second" in query.lower():
            return [_chunk(1, "d1", "a.pdf", 3, "Second admission: chest pain, discharged stable.")]
        if "first" in query.lower():
            return [_chunk(2, "d1", "a.pdf", 1, "First admission: fever, treated with antibiotics.")]
        return []

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What happened during the second admission? What about the first one?",
        document_id="d1",
        history=[{"role": "user", "content": "Tell me about this patient's hospital stays."}],
    )

    # Retrieval ran once per part, not once for the whole compound question.
    assert calls == [
        "What happened during the second admission?",
        "What happened during the first admission?",
    ]

    # Both parts' evidence made it into the merged result.
    texts = " ".join(c["text"] for c in chunks)
    assert "Second admission" in texts
    assert "First admission" in texts
    assert {c["part"] for c in chunks} == {1, 2}

    # The prompt structurally corresponds to both requested parts: a
    # numbered plan naming each admission, and evidence grouped under a
    # matching PART heading - not the single-reference FOLLOW_UP style a
    # bare "what about" trigger with history present would otherwise get.
    assert "This question has 2 parts" in prompt
    assert "Part 1: What happened during the second admission?" in prompt
    assert "Part 2: What happened during the first admission?" in prompt
    assert "PART 1 (What happened during the second admission?):" in prompt
    assert "PART 2 (What happened during the first admission?):" in prompt
    assert "Second admission: chest pain" in prompt
    assert "First admission: fever" in prompt
    assert "continues the conversation above" not in prompt


def test_regression_question_works_with_no_history_too(monkeypatch):
    """The same question asked as the very first message of a session (no
    history to resolve a bare reference against) must still decompose -
    both parts are self-contained, so history was never required here."""
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        return [_chunk(len(calls), "d1", "a.pdf", 1, query)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What happened during the second admission? What about the first one?",
        document_id="d1",
    )

    assert len(calls) == 2
    assert len(chunks) == 2


def test_one_part_with_evidence_one_without_reaches_the_prompt_as_such(monkeypatch):
    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        if "second" in query.lower():
            return [_chunk(1, "d1", "a.pdf", 3, "Second admission: chest pain.")]
        return []  # nothing found for "first admission"

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What happened during the second admission? What about the first one?",
        document_id="d1",
    )

    assert len(chunks) == 1
    assert "Second admission: chest pain." in prompt
    assert "PART 2 (What happened during the first admission?):\n  EVIDENCE: none" in prompt
    # The instruction to still answer the covered part, honestly, for the
    # uncovered one, must be present.
    assert "say so for that part only and still answer the others" in prompt


def test_both_parts_empty_raises_insufficient_retrieval(monkeypatch):
    monkeypatch.setattr(app, "retrieve_hybrid", lambda query, top_k=5, document_id=None, timing=None: [])

    with pytest.raises(InsufficientRetrievalError):
        app.retrieve_and_build_prompt(
            "What happened during the second admission? What about the first one?",
            document_id="d1",
        )


def test_ordinary_single_part_question_is_unaffected(monkeypatch):
    """A plain single-document, single-part question must take exactly the
    old one-call path - proves the fix doesn't touch the common case."""
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        calls.append((query, top_k, document_id))
        return [_chunk(1, "d1", "a.pdf", 1, "the patient was discharged on metformin")]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What medication was the patient prescribed?", document_id="d1"
    )

    assert calls == [("What medication was the patient prescribed?", 5, "d1")]
    assert "part" not in chunks[0]
    assert "This question has" not in prompt
