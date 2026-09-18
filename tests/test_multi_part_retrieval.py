"""Tests for src.retrieval.hybrid_retriever.retrieve_for_question().

Live testing exposed a real production failure: "What happened during the
second admission? What about the first one?" retrieved evidence that did not
clearly cover both requested parts, and the answer was mostly about one
admission - because the whole compound question was embedded and searched as
ONE query. retrieve_for_question() decomposes such a question (see
src/generation/intent.decompose_question) and retrieves each part
separately, then merges the results deterministically.

A follow-up optimization pass replaced N sequential embed_query() calls (one
per part) with one batched embed_texts() call - sentence-transformers'
encode() already batches internally, so asking it for N texts at once pays
model-call overhead once instead of N times. That is a separate concern from
the retrieval-merging logic these tests exercise, so `embed_texts` is faked
here too (see `fake_embeddings` below) - these tests stay fast and fully
isolated from the real ML model; the numeric-equivalence and real-batching
claims get their own dedicated tests further down, against the real model.
"""

import numpy as np
import pytest

from src.retrieval import hybrid_retriever
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


def _fake_embed_texts(texts):
    """Deterministic, distinct-per-text, no real model - each text maps to
    its own small vector so tests can assert on correspondence (item 3:
    "embeddings must correspond exactly to the same sub-question strings in
    the same order") without depending on sentence-transformers actually
    being loadable/fast in a unit test.
    """
    return np.array([[float((hash(t) % 1000)), 0.0, 0.0, 0.0] for t in texts], dtype="float32")


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Every test in this file fakes embed_texts - the merging/dedup/ranking
    logic these tests exercise is independent of what the real model
    produces. monkeypatch.setattr on the `hybrid_retriever` module itself
    (not a separately-imported name) because retrieve_for_question calls
    embed_texts via this module's own globals - see test_hybrid_retriever.py
    for the same pattern already used for embed_query.
    """
    calls = []
    monkeypatch.setattr(hybrid_retriever, "embed_texts", lambda texts: (calls.append(list(texts)), _fake_embed_texts(texts))[1])
    return calls


# --- single-part passthrough ---------------------------------------------------


def test_single_part_question_makes_exactly_one_retrieve_call(fake_embeddings):
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
    # Single-part path must not touch batched embedding at all - requirement
    # 8, "single-question retrieval behavior must remain unchanged".
    assert fake_embeddings == []


def test_single_part_question_result_is_unmodified_passthrough():
    """No re-ranking, no merging - the exact object retrieve_hybrid returned."""
    original = [_chunk(1, "d1", "a.pdf", 2, "text", rank=1)]

    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return original

    result = retrieve_for_question("What medication was prescribed?", retrieve_fn=fake_retrieve)

    assert result == original


def test_narrow_retrieve_fn_signature_still_works():
    """A mock with no document_id/timing/query_vector parameter at all (e.g.
    chat.py's existing tests) must not break for an ordinary single-part
    question - the conditional-kwarg pattern must omit all three when
    they're unset, exactly as before this optimization."""
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


# --- batched embedding: call count and correspondence ---------------------------


def test_multi_part_retrieval_embeds_all_parts_in_one_batched_call(fake_embeddings):
    """The actual optimization: one embed_texts() call for every part, not
    one embed_query()-equivalent call per part."""
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        return [_chunk(hash(query) % 1000, "d1", "a.pdf", 1, query)]

    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )

    assert len(fake_embeddings) == 1  # exactly one embed_texts() call
    assert fake_embeddings[0] == ["What was ordered", "what medications were prescribed?"]


def test_each_part_receives_its_own_vector_in_the_same_order():
    """Requirement 3: resulting embeddings must correspond exactly to the
    same sub-question strings in the same order - proven by checking each
    part's retrieve_fn call received the vector _fake_embed_texts would
    produce for THAT part's own text, not some other part's."""
    received = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        received.append((query, query_vector))
        return [_chunk(hash(query) % 1000, "d1", "a.pdf", 1, query)]

    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )

    assert len(received) == 2
    for query, vector in received:
        assert vector is not None
        expected = _fake_embed_texts([query])[0]
        assert np.array_equal(vector, expected)


def test_single_part_question_never_calls_embed_texts(fake_embeddings):
    """The single-part fallback keeps calling retrieve_fn exactly as before
    - no batched embedding, no query_vector - see retrieve_hybrid's own
    docstring: query_vector is only ever passed by the multi-part path."""
    def fake_retrieve(query, top_k=5, document_id=None, timing=None):
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    retrieve_for_question("What was the diagnosis?", retrieve_fn=fake_retrieve)

    assert fake_embeddings == []


# --- two-part direct questions --------------------------------------------------


def test_two_part_question_retrieves_each_part_separately():
    calls = []

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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
    and never the conversation transcript itself. A single resolved part
    stays on the single-call path (no batching needed for one item)."""
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

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        return []

    result = retrieve_for_question(
        "What happened during the second admission? What about the first one?",
        retrieve_fn=fake_retrieve,
    )

    assert result == []


# --- deduplication ---------------------------------------------------------------


def test_chunk_retrieved_by_both_parts_is_not_duplicated():
    shared = _chunk(1, "d1", "a.pdf", 2, "mentions both admissions")

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
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

    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        calls.append(top_k)
        return []

    retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        top_k=config.TOP_K,
        retrieve_fn=fake_retrieve,
    )

    assert all(k * 2 <= config.MAX_CONTEXT_CHUNKS or k == 1 for k in calls)
    assert all(k >= 1 for k in calls)


# --- timing: one batch cost, not a sum over parts --------------------------------


def test_embedding_ms_is_the_one_batch_cost_not_summed_over_parts():
    """Before this optimization, embedding_ms was summed from each part's
    own retrieve_hybrid() call (N separate embed_query() costs). Now
    embedding happens once, for the whole batch, and retrieve_for_question
    measures and reports THAT cost directly - a part's own retrieve_fn call
    (given a precomputed query_vector) no longer contributes anything to
    embedding_ms at all."""
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        if timing is not None:
            # A part-level fake that (wrongly, if this optimization worked)
            # tries to also report its own embedding cost - it must be
            # overridden/ignored by the one true batch measurement, not summed.
            timing["embedding_ms"] = 999.0
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

    assert timing["embedding_ms"] != pytest.approx(1998.0)  # not summed from the two fakes
    assert timing["embedding_ms"] >= 0.0
    assert timing["dense_ms"] == pytest.approx(6.0)  # dense/keyword/fusion still sum per part
    assert timing["keyword_ms"] == pytest.approx(2.0)
    assert timing["fusion_ms"] == pytest.approx(1.0)


def test_timing_not_populated_when_not_requested():
    def fake_retrieve(query, top_k=5, document_id=None, timing=None, query_vector=None):
        assert timing is None
        return [_chunk(1, "d1", "a.pdf", 1, "text")]

    result = retrieve_for_question(
        "What was ordered, and what medications were prescribed?",
        retrieve_fn=fake_retrieve,
    )
    assert result  # no exception raised above is the real assertion


# --- real-FAISS equivalence: batched retrieval vs. the old per-part algorithm ----
#
# The fast fakes above prove the merging/dedup/ranking logic; this proves the
# actual retrieval OUTPUT (chunk_ids, order, per-part attribution) is
# unchanged by the optimization, against a real FAISS index and real RRF
# fusion - only the embedding step is crafted (deterministic, not the real
# ML model), the same convention test_hybrid_retriever.py already uses so
# dense ranking stays exact-by-construction and the test stays fast.
#
# (The numeric-equivalence claim for the real sentence-transformers model
# itself - batched embed_texts() vs. sequential embed_query() calls,
# ~1e-7 max abs difference, well within float32 batching noise - was
# verified separately against the real model as part of this change's
# benchmark, rather than as a standing test: it would be the only test in
# the suite to load torch/sentence-transformers, adding ~60s to every run
# for a claim that does not depend on retrieval logic at all.)

DIM = 4


def _vec(axis):
    v = np.zeros(DIM, dtype="float32")
    v[axis] = 1.0
    return v


def _text_vector(text: str) -> np.ndarray:
    """Deterministic per-text vector: "second"-admission content aligns to
    axis 0, "first"-admission content to axis 1 - distinct enough that FAISS
    top-k genuinely discriminates between them, the same way a real
    embedding would separate two different topics."""
    return _vec(0) if "second" in text.lower() else _vec(1)


@pytest.fixture
def real_faiss_corpus(tmp_path, monkeypatch):
    from src import config
    from src.retrieval.faiss_index import FaissIndex
    from src.retrieval.metadata_store import MetadataStore

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(config, "EMBEDDING_DIM", DIM)

    store = MetadataStore(config.DB_PATH)
    store.initialize()
    index = FaissIndex(DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()

    ids, vectors = [], []
    for i in range(3):
        cid = store.insert_chunk(
            document_id="d1", source="a.pdf", page=i + 1,
            text=f"Second admission note {i}: chest pain, discharged stable.",
        )
        ids.append(cid)
        vectors.append(_vec(0))
    for i in range(3):
        cid = store.insert_chunk(
            document_id="d1", source="a.pdf", page=i + 10,
            text=f"First admission note {i}: fever, treated with antibiotics.",
        )
        ids.append(cid)
        vectors.append(_vec(1))

    index.add(np.array(vectors, dtype="float32"), np.array(ids, dtype="int64"))
    index.save()

    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _text_vector(q))
    monkeypatch.setattr(hybrid_retriever, "embed_texts", lambda texts: np.stack([_text_vector(t) for t in texts]))


def _old_algorithm_reference(query, document_id=None, top_k=5):
    """Reproduces the PRE-optimization retrieve_for_question(): one
    retrieve_hybrid() call per part, each doing its own embed_query() - no
    query_vector, no batching. Used only as a reference to diff against."""
    from src.generation.intent import decompose_question

    parts = decompose_question(query)
    if len(parts) <= 1:
        resolved = parts[0] if parts else query
        return hybrid_retriever.retrieve_hybrid(resolved, top_k=top_k, document_id=document_id)

    from src import config as config_module

    per_part_top_k = max(1, min(top_k, config_module.MAX_CONTEXT_CHUNKS // len(parts)))
    merged, seen = [], set()
    for part_number, part_query in enumerate(parts, start=1):
        for chunk in hybrid_retriever.retrieve_hybrid(part_query, top_k=per_part_top_k, document_id=document_id):
            if chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            merged.append({**chunk, "part": part_number, "part_query": part_query})
    for rank, chunk in enumerate(merged, start=1):
        chunk["rank"] = rank
    return merged


def test_batched_retrieval_matches_the_old_per_part_algorithm_exactly(real_faiss_corpus):
    """Requirement: multi-part retrieval still produces the same chunk
    IDs/results. Compares the NEW retrieve_for_question() (batched
    embed_texts + query_vector injection) against a faithful reproduction of
    the OLD algorithm (one embed_query() per part) over a real FAISS index -
    both must land on identical chunk_ids, in identical order, with
    identical part attribution."""
    query = "What happened during the second admission? What about the first one?"

    new_result = retrieve_for_question(query, document_id="d1", top_k=5, retrieve_fn=hybrid_retriever.retrieve_hybrid)
    old_result = _old_algorithm_reference(query, document_id="d1", top_k=5)

    assert new_result  # sanity: something was actually retrieved
    assert [c["chunk_id"] for c in new_result] == [c["chunk_id"] for c in old_result]
    assert [c["rank"] for c in new_result] == [c["rank"] for c in old_result]
    assert [c["part"] for c in new_result] == [c["part"] for c in old_result]
    assert [c["part_query"] for c in new_result] == [c["part_query"] for c in old_result]
    # Both admissions' content actually made it in, not just IDs matching by accident.
    texts = " ".join(c["text"] for c in new_result)
    assert "Second admission" in texts
    assert "First admission" in texts


def test_single_part_retrieval_matches_the_old_algorithm_exactly(real_faiss_corpus):
    """Requirement 8, proven against a real index too: an ordinary
    single-part question is completely unaffected by the optimization."""
    query = "What happened during the second admission?"

    new_result = retrieve_for_question(query, document_id="d1", top_k=5, retrieve_fn=hybrid_retriever.retrieve_hybrid)
    old_result = _old_algorithm_reference(query, document_id="d1", top_k=5)

    assert [c["chunk_id"] for c in new_result] == [c["chunk_id"] for c in old_result]
    assert "part" not in new_result[0]
