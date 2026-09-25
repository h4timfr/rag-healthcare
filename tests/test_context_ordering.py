"""Tests for src.generation.prompt_builder.order_chunks_for_presentation().

Retrieval audit finding (read-only diagnosis, then this experiment): for two
real single-document questions, the chunk containing the actual answer was
retrieved and RRF-ranked among the selected chunks, but sat at or near the
END of the evidence list, behind several topically-adjacent chunks that
discuss the same subject without the specific figures asked for. Given that
evidence, generation described the existence of the right table/section
instead of reading its values. This is a presentation-only reorder of
chunks retrieval has ALREADY selected - it never changes which chunks were
retrieved, never touches RRF's own scores, and is bounded so it can only
pull a small, capped number of chunks forward, never reshuffle everything.
"""

import pytest

from src.generation.prompt_builder import (
    MAX_PROMOTED,
    MIN_TERM_MATCHES,
    PROTECT_FRONT,
    order_chunks_for_presentation,
)


def _chunk(chunk_id, text, rank, document_id="d1", page=1, source="a.pdf"):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": 1.0 / rank,
        "rank": rank,
    }


# --- strongest RRF results stay near the front ---------------------------------


def test_top_rrf_result_is_never_displaced():
    """The single strongest RRF result must stay first even when a later
    chunk has ideal promotion evidence - PROTECT_FRONT applies structurally,
    not by outscoring anything."""
    chunks = [
        _chunk(1, "the strongest RRF result, on topic but no numbers", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(3, "diagnostic category findings: skin 10.2%, trauma 9.6%", 3),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert result[0]["chunk_id"] == 1
    assert result[1]["chunk_id"] == 2


def test_protected_front_chunks_are_unaffected_by_promotion_eligibility():
    """A protected-front chunk that ALSO qualifies for promotion is simply
    left where it is - protection is not "promote it too"."""
    chunks = [
        _chunk(1, "diagnostic category findings: skin 10.2%, trauma 9.6%", 1),
        _chunk(2, "second strongest, unrelated", 2),
        _chunk(3, "third, unrelated", 3),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in result] == [1, 2, 3]


# --- a lower-ranked chunk with strong evidence can be promoted -----------------


def test_lower_ranked_chunk_with_numeric_evidence_is_promoted():
    chunks = [
        _chunk(1, "strongest RRF result, generic prose about the topic", 1),
        _chunk(2, "second strongest RRF result, also generic", 2),
        _chunk(3, "an unrelated chunk about something else entirely", 3),
        _chunk(4, "another unrelated chunk", 4),
        _chunk(5, "the diagnostic category findings were: skin 10.2%, trauma 9.6%", 5),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    ids = [c["chunk_id"] for c in result]
    # Promoted into the window right after the protected front, not merely
    # "somewhere higher" - this is what actually fixes a tail-position miss.
    assert ids.index(5) == PROTECT_FRONT
    assert ids[:PROTECT_FRONT] == [1, 2]


def test_the_exact_regression_shape_chunk_at_the_tail_is_promoted():
    """Mirrors the real Failure A/B shape: the answer-bearing chunk is LAST
    among 8 retrieved chunks, behind several topically-adjacent ones."""
    chunks = [_chunk(i, f"topically adjacent filler chunk {i}", i) for i in range(1, 8)]
    chunks.append(
        _chunk(8, "laboratory findings: urinalysis 30.5%, blood determination 29.5%, X-ray 21%", 8)
    )
    result = order_chunks_for_presentation(
        "What were the main findings regarding examinations, laboratory tests, and treatments?", chunks
    )
    ids = [c["chunk_id"] for c in result]
    assert ids.index(8) <= PROTECT_FRONT + MAX_PROMOTED - 1
    assert ids.index(8) < 7  # strictly earlier than its original last position


# --- unrelated lexical overlap does not dominate --------------------------------


def test_topical_overlap_without_numbers_is_not_promoted():
    """A chunk that discusses the same subject at length, but carries no
    numbers, must not be promoted ahead of the numeric-evidence chunk - and
    must not be promoted at all if it never crosses MIN_TERM_MATCHES with
    the numeric gate applied."""
    chunks = [
        _chunk(1, "strongest RRF result", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(
            3,
            "diagnostic category findings are discussed extensively here, in general terms, "
            "with no specific figures reported anywhere in this passage",
            3,
        ),
        _chunk(4, "the findings by diagnostic category were: skin 10.2%, trauma 9.6%", 4),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    ids = [c["chunk_id"] for c in result]
    # The numeric chunk (4) is promoted; the numberless topical chunk (3) is not.
    assert ids.index(4) < ids.index(3)
    assert ids.index(3) == 3  # chunk 3 kept its original relative position


def test_single_incidental_term_match_does_not_promote():
    """One matching word (below MIN_TERM_MATCHES) must not promote a chunk
    even if it also happens to contain a number - avoids promoting on
    coincidental overlap."""
    chunks = [
        _chunk(1, "strongest RRF result", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(3, "category 5 is unrelated to this question's actual subject", 3),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in result] == [1, 2, 3]


def test_numberless_chunk_never_promotes_regardless_of_term_count():
    """Even with every significant query term present, zero numbers means
    zero score - the gate in _numeric_evidence_score, not just a low count."""
    chunks = [
        _chunk(1, "strongest RRF result", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(3, "the findings for each diagnostic category are discussed at length below", 3),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in result] == [1, 2, 3]


# --- bounded: promotion cannot reshuffle the whole context ----------------------


def test_promotion_is_capped_at_max_promoted():
    chunks = [_chunk(1, "strongest, generic", 1), _chunk(2, "second, generic", 2)]
    for i in range(3, 9):
        chunks.append(_chunk(i, f"diagnostic category findings entry {i}: value {i}.{i}%", i))
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    promoted_window = [c["chunk_id"] for c in result[PROTECT_FRONT : PROTECT_FRONT + MAX_PROMOTED]]
    assert len(promoted_window) == MAX_PROMOTED
    # Everything past the capped window keeps arriving in original RRF order.
    remainder_ids = [c["chunk_id"] for c in result[PROTECT_FRONT + MAX_PROMOTED :]]
    assert remainder_ids == sorted(remainder_ids)


# --- deterministic ordering ------------------------------------------------------


def test_ordering_is_deterministic_across_repeated_calls():
    chunks = [
        _chunk(1, "strongest RRF result", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(3, "third, no numbers", 3),
        _chunk(4, "findings for each diagnostic category: 10.2%", 4),
        _chunk(5, "findings for each diagnostic category: 9.6%", 5),
    ]
    query = "What were the findings for each diagnostic category?"
    first = [c["chunk_id"] for c in order_chunks_for_presentation(query, chunks)]
    for _ in range(5):
        again = [c["chunk_id"] for c in order_chunks_for_presentation(query, chunks)]
        assert again == first


def test_equal_scoring_promoted_chunks_keep_their_relative_rrf_order():
    """Tie-breaking must be deterministic, not incidental - two chunks with
    the same promotion score keep the order RRF originally gave them."""
    chunks = [
        _chunk(1, "strongest RRF result", 1),
        _chunk(2, "second strongest RRF result", 2),
        _chunk(3, "diagnostic category finding: 10.2%", 3),
        _chunk(4, "diagnostic category finding: 9.6%", 4),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    ids = [c["chunk_id"] for c in result]
    assert ids.index(3) < ids.index(4)


def test_input_list_is_not_mutated():
    chunks = [
        _chunk(1, "strongest", 1),
        _chunk(2, "second", 2),
        _chunk(3, "no numbers here", 3),
        _chunk(4, "diagnostic category finding: 10.2%", 4),
    ]
    original_order = [c["chunk_id"] for c in chunks]
    original_ranks = [c["rank"] for c in chunks]
    order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in chunks] == original_order
    assert [c["rank"] for c in chunks] == original_ranks


def test_rank_field_is_reassigned_to_new_presentation_position():
    chunks = [
        _chunk(1, "strongest", 1),
        _chunk(2, "second", 2),
        _chunk(3, "no numbers", 3),
        _chunk(4, "diagnostic category finding: 10.2%", 4),
    ]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["rank"] for c in result] == list(range(1, len(result) + 1))
    # The RRF score itself is untouched - it still reflects retrieval, not presentation.
    by_id = {c["chunk_id"]: c["score"] for c in result}
    assert by_id[4] == pytest.approx(1.0 / 4)


# --- edge cases -------------------------------------------------------------------


def test_fewer_chunks_than_protect_front_is_a_noop():
    chunks = [_chunk(1, "only chunk", 1)]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in result] == [1]


def test_query_with_no_significant_terms_is_a_noop():
    chunks = [_chunk(i, f"chunk {i} with 10.{i}% figures", i) for i in range(1, 6)]
    result = order_chunks_for_presentation("what is it", chunks)
    assert [c["chunk_id"] for c in result] == [1, 2, 3, 4, 5]


def test_no_chunk_qualifies_for_promotion_leaves_order_unchanged():
    chunks = [_chunk(i, "generic prose with no numbers or matching terms", i) for i in range(1, 6)]
    result = order_chunks_for_presentation("What were the findings for each diagnostic category?", chunks)
    assert [c["chunk_id"] for c in result] == [1, 2, 3, 4, 5]


# --- existing single-document retrieval behavior remains intact ----------------


def test_app_retrieve_and_build_prompt_single_document_still_reorders(monkeypatch):
    """End-to-end through the real production call site: a single-document
    question with a tail-positioned, numeric-evidence chunk gets it
    promoted in the final prompt/chunks app.py hands to the UI and Ollama."""
    import app

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        chunks = [_chunk(1, "strongest, generic", 1), _chunk(2, "second, generic", 2)]
        for i in range(3, top_k):
            chunks.append(_chunk(i, f"filler chunk {i}", i))
        chunks.append(_chunk(top_k, "diagnostic category findings: skin 10.2%, trauma 9.6%", top_k))
        return chunks

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What were the findings for each diagnostic category?", top_k=8, document_id="d1"
    )

    ids = [c["chunk_id"] for c in chunks]
    assert ids.index(8) <= PROTECT_FRONT + MAX_PROMOTED - 1  # promoted, not last
    # The prompt's own evidence numbering must match the same reordered list -
    # citations in the answer must point at the same chunk the UI shows.
    promoted_position = ids.index(8) + 1
    assert f"[{promoted_position}] Source: a.pdf, page 1" in prompt
    assert "skin 10.2%, trauma 9.6%" in prompt


def test_app_retrieve_and_build_prompt_ordinary_question_unaffected(monkeypatch):
    """A question with no promotable chunk produces the exact same chunk
    order and prompt as before this feature existed."""
    import app

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        return [_chunk(i, f"ordinary prose chunk {i}, no special evidence", i) for i in range(1, top_k + 1)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, prompt = app.retrieve_and_build_prompt(
        "What medication was the patient prescribed?", top_k=5, document_id="d1"
    )

    assert [c["chunk_id"] for c in chunks] == [1, 2, 3, 4, 5]


# --- existing each-document/comparison balancing behavior remains intact -------


def test_multi_document_scope_is_not_reordered(monkeypatch):
    """document_inventory (each-document/comparison scope) must bypass this
    feature entirely - its own balanced allocation owns ordering there."""
    import app

    inventory = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 3},
        {"document_id": "d2", "source": "b.pdf", "page_count": 4},
    ]

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None, balance_by_document=False):
        return [
            _chunk(1, "doc a strongest", 1, document_id="d1", source="a.pdf"),
            _chunk(2, "doc b strongest", 2, document_id="d2", source="b.pdf"),
            _chunk(6, "doc a: diagnostic category findings 10.2%", 3, document_id="d1", source="a.pdf"),
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, _prompt = app.retrieve_and_build_prompt(
        "What were the findings for each diagnostic category?",
        top_k=5,
        document_inventory=inventory,
    )

    # Untouched: exactly retrieve_hybrid's own order, no promotion applied.
    assert [c["chunk_id"] for c in chunks] == [1, 2, 6]


def test_multi_part_tagged_chunks_are_not_reordered(monkeypatch):
    """A multi-part retrieval result (chunks tagged with `part`) has its own
    PART-grouped structure in build_part_context() - this feature must not
    touch it."""
    import numpy as np

    import app
    from src.retrieval import hybrid_retriever

    # decompose_question() splits this into 2 parts, which routes through
    # retrieve_for_question()'s batched embed_texts() call - faked here (not
    # the real ML model) for the same reason test_multi_part_app.py already
    # does this: these tests exercise chunk-list plumbing, not embeddings.
    monkeypatch.setattr(
        hybrid_retriever,
        "embed_texts",
        lambda texts: np.zeros((len(texts), 4), dtype="float32"),
    )

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None, query_vector=None):
        return [_chunk(hash(query) % 1000, query, 1)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, _prompt = app.retrieve_and_build_prompt(
        "What was ordered, and what medications were prescribed?", top_k=5, document_id="d1"
    )

    assert all("part" in c for c in chunks)
