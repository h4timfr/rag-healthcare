"""Unit tests for src/evaluation/metrics.py - the contamination-free scoring
functions that replace scripts/evaluate.py's scoring for any trusted run.
"""

import pytest

from src.evaluation.metrics import (
    DEV_CATEGORY_RANGES,
    aggregate,
    category_for_question,
    chunk_is_relevant,
    document_aware_page_recall,
    evaluate_question,
    matched_terms,
    ndcg_at_k,
    reciprocal_rank_at_k,
    term_coverage,
)


# --- category_for_question ---------------------------------------------

def test_category_for_question_matches_dev_ranges():
    assert category_for_question("Q01") == "Problem / Objectives"
    assert category_for_question("Q16") == "Problem / Objectives"
    assert category_for_question("Q17") == "RAG Concepts"
    assert category_for_question("Q30") == "Literature Survey"
    assert category_for_question("Q50") == "Conclusion"


def test_category_for_question_out_of_range_is_uncategorized():
    assert category_for_question("Q99") == "Uncategorized"


def test_category_for_question_custom_ranges():
    ranges = {"A": range(1, 3), "B": range(3, 5)}
    assert category_for_question("Q01", ranges) == "A"
    assert category_for_question("Q03", ranges) == "B"
    assert category_for_question("Q05", ranges) == "Uncategorized"


def test_dev_category_ranges_partition_fifty_questions_without_overlap():
    covered = set()
    for r in DEV_CATEGORY_RANGES.values():
        for n in r:
            assert n not in covered, f"question number {n} covered by two categories"
            covered.add(n)
    assert covered == set(range(1, 51))


# --- document_aware_page_recall -----------------------------------------

def test_page_recall_full_match():
    chunks = [{"page": 2, "source": "a.pdf"}, {"page": 5, "source": "a.pdf"}]
    assert document_aware_page_recall(chunks, [2], "a.pdf") == 1.0


def test_page_recall_partial_match():
    chunks = [{"page": 2, "source": "a.pdf"}]
    assert document_aware_page_recall(chunks, [2, 3], "a.pdf") == 0.5


def test_page_recall_no_expected_pages_is_trivially_one():
    assert document_aware_page_recall([], [], "a.pdf") == 1.0


def test_page_recall_ignores_matching_page_from_wrong_document():
    """The core audit fix: a same-numbered page from a DIFFERENT document
    must not count as a hit."""
    chunks = [{"page": 2, "source": "unrelated_other_document.pdf"}]
    assert document_aware_page_recall(chunks, [2], "a.pdf") == 0.0


def test_page_recall_without_expected_source_falls_back_to_page_only():
    chunks = [{"page": 2, "source": "anything.pdf"}]
    assert document_aware_page_recall(chunks, [2], expected_source=None) == 1.0


# --- chunk_is_relevant -----------------------------------------------------

def test_chunk_is_relevant_true_for_matching_source_and_page():
    chunk = {"page": 2, "source": "a.pdf"}
    assert chunk_is_relevant(chunk, [2, 5], "a.pdf") is True


def test_chunk_is_relevant_false_for_wrong_page():
    chunk = {"page": 3, "source": "a.pdf"}
    assert chunk_is_relevant(chunk, [2, 5], "a.pdf") is False


def test_chunk_is_relevant_false_for_wrong_document_even_if_page_matches():
    """The same audit-driven rule as document_aware_page_recall: page number
    alone is never sufficient."""
    chunk = {"page": 2, "source": "unrelated_other_document.pdf"}
    assert chunk_is_relevant(chunk, [2], "a.pdf") is False


def test_chunk_is_relevant_ignores_source_when_expected_source_is_none():
    chunk = {"page": 2, "source": "anything.pdf"}
    assert chunk_is_relevant(chunk, [2], expected_source=None) is True


# --- reciprocal_rank_at_k (MRR@5) ------------------------------------------

def _chunk(page, source="a.pdf"):
    return {"page": page, "source": source}


def test_mrr_relevant_at_rank_1():
    chunks = [_chunk(2), _chunk(9), _chunk(9)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == pytest.approx(1.0)


def test_mrr_relevant_at_rank_2():
    chunks = [_chunk(9), _chunk(2), _chunk(9)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == pytest.approx(1 / 2)


def test_mrr_relevant_at_rank_3():
    chunks = [_chunk(9), _chunk(9), _chunk(2)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == pytest.approx(1 / 3)


def test_mrr_relevant_at_rank_5():
    chunks = [_chunk(9), _chunk(9), _chunk(9), _chunk(9), _chunk(2)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == pytest.approx(1 / 5)


def test_mrr_relevant_beyond_k_does_not_count():
    chunks = [_chunk(9), _chunk(9), _chunk(9), _chunk(9), _chunk(9), _chunk(2)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf", k=5) == 0.0


def test_mrr_no_relevant_result_is_zero():
    chunks = [_chunk(9), _chunk(10), _chunk(11)]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == 0.0


def test_mrr_uses_first_relevant_hit_with_multiple_relevant_chunks():
    chunks = [_chunk(9), _chunk(5), _chunk(2)]  # both 5 and 2 are relevant
    assert reciprocal_rank_at_k(chunks, [2, 5], "a.pdf") == pytest.approx(1 / 2)


def test_mrr_cross_document_page_collision_is_not_relevant():
    """A chunk whose page matches but whose source doesn't must be skipped -
    the reciprocal rank should come from the next TRUE match, not this one."""
    chunks = [_chunk(2, source="unrelated.pdf"), _chunk(2, source="a.pdf")]
    assert reciprocal_rank_at_k(chunks, [2], "a.pdf") == pytest.approx(1 / 2)


def test_mrr_empty_expected_pages_is_trivially_one():
    assert reciprocal_rank_at_k([_chunk(9)], [], "a.pdf") == 1.0


# --- ndcg_at_k (nDCG@5) -----------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    """All expected pages retrieved at the very top ranks == ideal ranking."""
    chunks = [_chunk(2), _chunk(5)]
    assert ndcg_at_k(chunks, [2, 5], "a.pdf") == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_1_is_one():
    chunks = [_chunk(2), _chunk(9), _chunk(9)]
    assert ndcg_at_k(chunks, [2], "a.pdf") == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_3_is_discounted():
    import math
    chunks = [_chunk(9), _chunk(9), _chunk(2)]
    expected = (1 / math.log2(4)) / (1 / math.log2(2))
    assert ndcg_at_k(chunks, [2], "a.pdf") == pytest.approx(expected)
    assert 0 < ndcg_at_k(chunks, [2], "a.pdf") < 1.0


def test_ndcg_no_relevant_result_is_zero():
    chunks = [_chunk(9), _chunk(10)]
    assert ndcg_at_k(chunks, [2], "a.pdf") == 0.0


def test_ndcg_multiple_relevant_chunks_suboptimal_order_is_between_zero_and_one():
    """Both expected pages found, but not in the ideal rank-1/rank-2 order."""
    chunks = [_chunk(9), _chunk(2), _chunk(9), _chunk(5)]
    score = ndcg_at_k(chunks, [2, 5], "a.pdf")
    assert 0.0 < score < 1.0


def test_ndcg_cross_document_page_collision_not_credited():
    chunks = [_chunk(2, source="unrelated.pdf")]
    assert ndcg_at_k(chunks, [2], "a.pdf") == 0.0


def test_ndcg_duplicate_chunks_from_same_relevant_page_do_not_exceed_one():
    """Two distinct chunks both from the one relevant page must not double
    the DCG beyond IDCG (which assumes one relevant hit per expected page)."""
    chunks = [_chunk(2), _chunk(2)]  # same page, appears twice
    score = ndcg_at_k(chunks, [2], "a.pdf")
    assert score == pytest.approx(1.0)
    assert score <= 1.0 + 1e-9


def test_ndcg_never_exceeds_one_even_with_many_duplicate_relevant_pages():
    chunks = [_chunk(2)] * 5
    score = ndcg_at_k(chunks, [2], "a.pdf")
    assert score <= 1.0 + 1e-9


def test_ndcg_empty_expected_pages_is_trivially_one():
    assert ndcg_at_k([_chunk(9)], [], "a.pdf") == 1.0


# --- matched_terms / term_coverage (no aliases) --------------------------

def test_matched_terms_exact_substring():
    matched, missing = matched_terms(
        "The system uses FAISS for retrieval.", ["FAISS", "keyword search"]
    )
    assert matched == ["FAISS"]
    assert missing == ["keyword search"]


def test_matched_terms_normalizes_case_and_punctuation():
    matched, _ = matched_terms("It stores Source / Citation info.", ["source / citation"])
    assert matched == ["source / citation"]


def test_matched_terms_no_alias_leniency_for_paraphrase():
    """Confirms the hard-coded alias from the legacy harness is gone: a
    paraphrase of an expected term must NOT be credited."""
    matched, missing = matched_terms(
        "The database stores embeddings so it can run similarity search.",
        ["Stores embeddings for similarity search"],
    )
    # "stores embeddings for similarity search" is not a literal substring
    # of the answer (word order differs) - no alias should rescue this.
    assert matched == []
    assert missing == ["Stores embeddings for similarity search"]


def test_term_coverage_fraction():
    answer = "Uses FAISS and SQLite."
    assert term_coverage(answer, ["FAISS", "SQLite", "Ollama"]) == pytest.approx(2 / 3)


def test_term_coverage_empty_expected_terms_is_one():
    assert term_coverage("anything", []) == 1.0


# --- evaluate_question: error accounting ---------------------------------

def _item(**overrides):
    base = {
        "id": "Q01",
        "question": "What is X?",
        "expected_pages": [2],
        "expected_terms": ["healthcare documents"],
        "source": "doc.pdf",
    }
    base.update(overrides)
    return base


def test_evaluate_question_pass_case():
    def fake_retrieve(q):
        return [{"page": 2, "source": "doc.pdf", "text": "about healthcare documents"}]

    def fake_prompt(q, chunks):
        return "prompt"

    def fake_generate(prompt):
        return "This concerns healthcare documents."

    result = evaluate_question(_item(), fake_retrieve, fake_prompt, fake_generate)
    assert result["failure_type"] == "PASS"
    assert result["retrieval_recall"] == 1.0
    assert result["term_coverage"] == 1.0
    assert result["mrr_at_5"] == 1.0
    assert result["ndcg_at_5"] == 1.0
    assert result["error_type"] is None
    assert result["retrieval_ms"] is not None and result["generation_ms"] is not None


def test_evaluate_question_retrieval_content_miss():
    def fake_retrieve(q):
        return [{"page": 9, "source": "doc.pdf", "text": "unrelated"}]

    result = evaluate_question(
        _item(), fake_retrieve, lambda q, c: "p", lambda p: "healthcare documents"
    )
    assert result["failure_type"] == "RETRIEVAL_CONTENT_MISS"
    assert result["missing_pages"] == [2]


def test_evaluate_question_generation_content_miss():
    def fake_retrieve(q):
        return [{"page": 2, "source": "doc.pdf", "text": "about healthcare documents"}]

    result = evaluate_question(
        _item(), fake_retrieve, lambda q, c: "p", lambda p: "something else entirely"
    )
    assert result["failure_type"] == "GENERATION_CONTENT_MISS"
    assert result["missing_terms"] == ["healthcare documents"]


def test_evaluate_question_retrieval_error_is_recorded_not_dropped():
    def raising_retrieve(q):
        raise RuntimeError("index not found")

    result = evaluate_question(
        _item(), raising_retrieve, lambda q, c: "p", lambda p: "answer"
    )
    assert result["failure_type"] == "RETRIEVAL_ERROR"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "index not found"
    assert result["retrieval_recall"] == 0.0
    assert result["term_coverage"] == 0.0
    assert result["mrr_at_5"] == 0.0
    assert result["ndcg_at_5"] == 0.0
    assert result["retrieval_ms"] is None


def test_evaluate_question_generation_error_is_recorded_not_dropped():
    def fake_retrieve(q):
        return [{"page": 2, "source": "doc.pdf", "text": "about healthcare documents"}]

    def raising_generate(prompt):
        raise ConnectionError("ollama unavailable")

    result = evaluate_question(
        _item(), fake_retrieve, lambda q, c: "p", raising_generate
    )
    assert result["failure_type"] == "GENERATION_ERROR"
    assert result["error_type"] == "ConnectionError"
    assert result["retrieval_recall"] == 1.0  # retrieval itself did succeed
    assert result["term_coverage"] == 0.0
    assert result["mrr_at_5"] == 1.0  # retrieval succeeded, so ranking metrics are real too
    assert result["ndcg_at_5"] == 1.0
    assert result["generation_ms"] is None


def test_evaluate_question_prompt_fn_never_receives_ground_truth():
    """The prompt_fn must only be called with (question, chunks) - proving
    the harness cannot leak expected_terms/expected_pages into generation."""
    captured = {}

    def fake_retrieve(q):
        return [{"page": 2, "source": "doc.pdf", "text": "t"}]

    def capturing_prompt(query, chunks):
        captured["args"] = (query, chunks)
        return "prompt"

    evaluate_question(_item(), fake_retrieve, capturing_prompt, lambda p: "answer")

    query, chunks = captured["args"]
    assert query == "What is X?"
    assert chunks == [{"page": 2, "source": "doc.pdf", "text": "t"}]


# --- aggregate -------------------------------------------------------------

def test_aggregate_counts_and_never_drops_errors_from_mean():
    results = [
        {"failure_type": "PASS", "retrieval_recall": 1.0, "term_coverage": 1.0,
         "retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
        {"failure_type": "RETRIEVAL_ERROR", "retrieval_recall": 0.0, "term_coverage": 0.0,
         "retrieval_ms": None, "generation_ms": None, "total_ms": None},
    ]
    summary = aggregate(results)

    assert summary["attempted"] == 2
    assert summary["scored"] == 2
    assert summary["passes"] == 1
    assert summary["retrieval_errors"] == 1
    assert summary["retrieval_failures"] == 1
    # mean recall must include the error's 0.0, not just the passing item's 1.0
    assert summary["mean_retrieval_recall"] == pytest.approx(0.5)
    assert summary["mean_term_coverage"] == pytest.approx(0.5)
    # latency mean is computed only over the one item that has timings
    assert summary["timed_count"] == 1
    assert summary["mean_retrieval_ms"] == pytest.approx(10.0)


def test_aggregate_empty_results():
    summary = aggregate([])
    assert summary["attempted"] == 0
    assert summary["mean_retrieval_recall"] is None


def test_aggregate_computes_mean_mrr_and_ndcg():
    results = [
        {"failure_type": "PASS", "retrieval_recall": 1.0, "term_coverage": 1.0,
         "mrr_at_5": 1.0, "ndcg_at_5": 1.0,
         "retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
        {"failure_type": "PASS", "retrieval_recall": 1.0, "term_coverage": 1.0,
         "mrr_at_5": 0.5, "ndcg_at_5": 0.6,
         "retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
    ]
    summary = aggregate(results)
    assert summary["mean_mrr_at_5"] == pytest.approx(0.75)
    assert summary["mean_ndcg_at_5"] == pytest.approx(0.8)
    assert summary["ranking_metrics_count"] == 2


def test_aggregate_backward_compatible_with_results_missing_ranking_metrics():
    """Older persisted results predate MRR@5/nDCG@5 and lack these keys -
    aggregate() must not crash, and must report them as unavailable (None)
    rather than silently treating a missing key as 0."""
    results = [
        {"failure_type": "PASS", "retrieval_recall": 1.0, "term_coverage": 1.0,
         "retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
    ]
    summary = aggregate(results)
    assert summary["mean_mrr_at_5"] is None
    assert summary["mean_ndcg_at_5"] is None
    assert summary["ranking_metrics_count"] == 0
    # unrelated metrics (already present in the old data) must be unaffected
    assert summary["mean_retrieval_recall"] == pytest.approx(1.0)
    assert summary["timed_count"] == 1
