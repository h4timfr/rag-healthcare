"""Tests for src.retrieval.hybrid_retriever.retrieve_for_question().

Live testing exposed a real production failure: "What happened during the
second admission? What about the first one?" retrieved evidence that did not
clearly cover both requested parts, and the answer was mostly about one
admission - because the whole compound question was embedded and searched as
ONE query. retrieve_for_question() decomposes such a question (see
src/generation/intent.decompose_question) and retrieves each part
separately, then merges the results deterministically.

These tests use a fake `retrieve_fn` rather than a real FAISS/SQLite index,
so the retrieval-merging logic (decomposition -> per-part calls -> dedup ->
re-rank -> part tagging) is exercised in isolation from real embeddings -
document-scoped/real-index retrieval itself is already covered by
test_hybrid_retriever.py and test_document_isolation.py.
"""

import pytest

from src.retrieval.hybrid_retriever import retrieve_for_question


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


# --- single-part passthrough ---------------------------------------------------


def test_single_part_question_makes_exactly_one_retrieve_call():
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append((query, top_k, document_id))
        return [_chunk(1, "d1", "a.pdf", 2, "some text")]

    result = retrieve_for_question(
        "What was the patient's chief complaint?",
        top_k=5,
        document_id="d1",
        retrieve_fn=fake_retrieve,
    )

    assert calls == [("What was the patient's chief complaint?", 5, "d1")]
    assert len(result) == 1
    assert "part" not in result[0]
    assert "part_query" not in result[0]


def test_single_part_question_result_is_unmodified_passthrough():
    """No re-ranking, no merging - the exact object retrieve_hybrid returned."""
    original = [_chunk(1, "d1", "a.pdf", 2, "text", rank=1)]

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return original

    result = retrieve_for_question("What medication was prescribed?", retrieve_fn=fake_retrieve)

    assert result == original


def test_narrow_retrieve_fn_signature_still_works():
    """A mock with no document_id/timing parameter at all (e.g. chat.py's
    existing tests) must not break for an ordinary single-part question -
    the conditional-kwarg pattern must omit both when they're unset."""
    calls = []

    def fake_retrieve(query, top_k=5):
        calls.append(query)
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    result = retrieve_for_question("What was the diagnosis?", retrieve_fn=fake_retrieve)

    assert calls == ["What was the diagnosis?"]
    assert len(result) == 1


def test_ordinal_document_reference_is_a_single_part_question():
    """"the second document" is a document reference, not a multi-part
    question - must take the single-call path, not per-part retrieval."""
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    retrieve_for_question("What about the second document?", retrieve_fn=fake_retrieve)

    assert calls == ["What about the second document?"]


# --- two-part direct questions --------------------------------------------------


def test_two_part_question_retrieves_each_part_separately():
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        if "medications" in query.lower():
            return [_chunk(1, "d1", "a.pdf", 2, "discharged on metformin")]
        return [_chunk(2, "d1", "a.pdf", 5, "condition found: pneumonia")]

    result = retrieve_for_question(
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?",
        retrieve_fn=fake_retrieve,
    )

    assert len(calls) == 2
    assert "medications" in calls[0].lower()
    assert "condition" in calls[1].lower()
    assert {c["chunk_id"] for c in result} == {1, 2}


def test_merged_chunks_carry_part_and_part_query():
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return [_chunk(hash(query) % 1000, "d1", "a.pdf", 1, query)]

    result = retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )

    assert [c["part"] for c in result] == [1, 2]
    assert result[0]["part_query"] == "What was ordered"
    assert "medications" in result[1]["part_query"]


def test_merged_ranks_are_one_consistent_sequence():
    """Citation numbers must stay a single sequence across parts, exactly as
    for any other question - not restart at 1 for the second part."""
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return [
            _chunk(1, "d1", "a.pdf", 1, "a", rank=1),
            _chunk(2, "d1", "a.pdf", 2, "b", rank=2),
        ]

    result = retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )

    assert [c["rank"] for c in result] == list(range(1, len(result) + 1))


# --- follow-up-shaped multi-part questions (the regression) --------------------


def test_regression_second_admission_then_what_about_the_first_one():
    """Exact regression case from live testing."""
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        if "second" in query.lower():
            return [_chunk(1, "d1", "a.pdf", 3, "second admission: chest pain, discharged stable")]
        if "first" in query.lower():
            return [_chunk(2, "d1", "a.pdf", 1, "first admission: fever, treated with antibiotics")]
        return []

    result = retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        document_id="d1",
        retrieve_fn=fake_retrieve,
    )

    assert calls == [
        "What happened during the second admission?",
        "What happened during the first admission?",
    ]
    texts = [c["text"] for c in result]
    assert any("second admission" in t for t in texts)
    assert any("first admission" in t for t in texts)
    assert {c["part"] for c in result} == {1, 2}


def test_what_about_the_first_alone_uses_history_as_a_clean_retrieval_query():
    """A standalone elliptical follow-up must reach retrieval as the
    resolved, clean question - never the raw "what about the first?" text,
    and never the conversation transcript itself."""
    calls = []
    history = [
        {"role": "user", "content": "What happened during the second admission?"},
        {"role": "assistant", "content": "During the second admission, she presented with chest pain."},
    ]

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        return [_chunk(1, "d1", "a.pdf", 1, "first admission text")]

    retrieve_for_question("What about the first?", history=history, retrieve_fn=fake_retrieve)

    assert calls == ["What happened during the first admission?"]


def test_follow_up_tag_question_with_history_still_decomposes_within_message():
    """When the current message already has two parts, history is not
    needed to resolve the second one - the first part is its own anchor."""
    calls = []
    history = [{"role": "user", "content": "unrelated earlier question"}]

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(query)
        return [_chunk(len(calls), "d1", "a.pdf", 1, query)]

    retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        history=history,
        retrieve_fn=fake_retrieve,
    )

    assert calls == [
        "What happened during the second admission?",
        "What happened during the first admission?",
    ]


# --- one part with evidence, one without ----------------------------------------


def test_one_part_with_evidence_other_without_still_returns_the_one_that_has_it():
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        if "second" in query.lower():
            return [_chunk(1, "d1", "a.pdf", 3, "second admission text")]
        return []  # nothing found for "first admission"

    result = retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        retrieve_fn=fake_retrieve,
    )

    assert len(result) == 1
    assert result[0]["part"] == 1
    assert "second admission" in result[0]["text"]


def test_both_parts_empty_returns_empty_list_not_an_error():
    """retrieve_for_question itself never raises for no evidence - that is
    retrieve_and_build_prompt()'s InsufficientRetrievalError, unchanged."""
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return []

    result = retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        retrieve_fn=fake_retrieve,
    )

    assert result == []


# --- deduplication ---------------------------------------------------------------


def test_chunk_retrieved_by_both_parts_is_not_duplicated():
    shared = _chunk(1, "d1", "a.pdf", 2, "mentions both admissions")

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return [shared]

    result = retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        retrieve_fn=fake_retrieve,
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == 1
    assert result[0]["part"] == 1  # kept under the first part to surface it


# --- document_id scoping and budget ----------------------------------------------


def test_document_id_is_forwarded_to_every_part():
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(document_id)
        return [_chunk(len(calls), "d1", "a.pdf", 1, query)]

    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        document_id="d1",
        retrieve_fn=fake_retrieve,
    )

    assert calls == ["d1", "d1"]


def test_per_part_budget_is_capped_by_max_context_chunks():
    from src import config

    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        calls.append(top_k)
        return []

    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        top_k=config.TOP_K,
        retrieve_fn=fake_retrieve,
    )

    assert all(k * 2 <= config.MAX_CONTEXT_CHUNKS or k == 1 for k in calls)
    assert all(k >= 1 for k in calls)


# --- timing accumulation ----------------------------------------------------------


def test_timing_sums_across_parts_when_requested():
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        if timing is not None:
            timing["embedding_ms"] = 2.0
            timing["dense_ms"] = 3.0
            timing["keyword_ms"] = 1.0
            timing["fusion_ms"] = 0.5
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    timing: dict = {}
    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        timing=timing,
        retrieve_fn=fake_retrieve,
    )

    assert timing["embedding_ms"] == pytest.approx(4.0)
    assert timing["dense_ms"] == pytest.approx(6.0)
    assert timing["keyword_ms"] == pytest.approx(2.0)
    assert timing["fusion_ms"] == pytest.approx(1.0)


def test_timing_not_populated_when_not_requested():
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        assert timing is None
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    result = retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )
    assert result  # no exception raised above is the real assertion
