"""Contamination-free evaluation scoring functions.

This module replaces the scoring logic in scripts/evaluate.py (kept, unmodified,
as a historical/audited artifact - see its module docstring). Differences from
that legacy version, per the evaluation audit:

- document_aware_page_recall() requires BOTH source and page to match, not
  page number alone (legacy page_recall() was document-blind).
- matched_terms()/term_coverage() have no hard-coded per-question alias list.
- evaluate_question() never silently drops a failed question: retrieval and
  generation errors are caught separately, recorded with their type/message,
  and scored as zero rather than excluded from the dataset.
- category_for_question() is a pure, reusable function (ranges passed in),
  not implicitly reliant on question ordering global state.

Ground truth (expected_pages/expected_terms/expected_source) is read only by
the harness that calls these functions - never passed into prompt
construction or the model. See src/generation/eval_prompt_builder.py.
"""

import math
import re
import time
from statistics import mean

# Category ranges for the existing 50-question DEVELOPMENT/REGRESSION benchmark
# (data/eval_questions.json) only. Identical to the legacy scripts/evaluate.py
# ranges, preserved so dev-benchmark category breakdowns remain comparable.
# The publication healthcare benchmark carries its own "category" field per
# question instead of relying on ID position.
DEV_CATEGORY_RANGES = {
    "Problem / Objectives": range(1, 17),
    "RAG Concepts": range(17, 25),
    "Literature Survey": range(25, 31),
    "Methodology": range(31, 37),
    "Architecture": range(37, 43),
    "Workflow": range(43, 48),
    "Conclusion": range(48, 51),
}

# Fixed k for the ranking metrics (MRR@5 / nDCG@5) - independent of whatever
# top_k the retriever happens to be configured with.
RANKING_K = 5


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def category_for_question(question_id: str, ranges: dict[str, range] = DEV_CATEGORY_RANGES) -> str:
    """Map a Q<NN>-style id to a category via numeric ranges. Pure function."""
    number = int(question_id[1:])
    for category, question_range in ranges.items():
        if number in question_range:
            return category
    return "Uncategorized"


def document_aware_page_recall(
    chunks: list[dict],
    expected_pages: list[int],
    expected_source: str | None = None,
) -> float:
    """Fraction of expected_pages actually retrieved, scoped to expected_source.

    A retrieved page only counts if BOTH its source and page number match the
    ground truth. If expected_source is None, falls back to page-only
    matching (legacy behavior) - callers running against a known single
    document should always pass expected_source.
    """
    if expected_source is not None:
        retrieved_pages = {
            int(c["page"]) for c in chunks if c.get("source") == expected_source
        }
    else:
        retrieved_pages = {int(c["page"]) for c in chunks}

    expected = set(expected_pages)
    if not expected:
        return 1.0

    return len(retrieved_pages & expected) / len(expected)


def chunk_is_relevant(
    chunk: dict,
    expected_pages: list[int],
    expected_source: str | None = None,
) -> bool:
    """A retrieved chunk counts as relevant only if BOTH its source and page
    match the ground truth - the same document-aware rule used by
    document_aware_page_recall(), applied per-chunk instead of as a set. A
    same-numbered page from an unrelated document is never relevant.

    Ground truth is the existing expected_pages/source fields (no explicit
    relevant-chunk-id field exists in the current dataset schema; adding one
    was judged unnecessary since document+page already uniquely identifies
    the ground-truth evidence for every question in this dataset).
    """
    if expected_source is not None and chunk.get("source") != expected_source:
        return False
    try:
        page = int(chunk["page"])
    except (KeyError, TypeError, ValueError):
        return False
    return page in set(expected_pages)


def reciprocal_rank_at_k(
    chunks: list[dict],
    expected_pages: list[int],
    expected_source: str | None = None,
    k: int = 5,
) -> float:
    """MRR (single-query reciprocal rank) at k: 1/rank of the first relevant
    chunk within the top k, or 0.0 if none of the top k are relevant. 1.0 if
    expected_pages is empty (no ground truth to satisfy - same convention as
    document_aware_page_recall).
    """
    if not expected_pages:
        return 1.0
    for rank, chunk in enumerate(chunks[:k], start=1):
        if chunk_is_relevant(chunk, expected_pages, expected_source):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    chunks: list[dict],
    expected_pages: list[int],
    expected_source: str | None = None,
    k: int = 5,
) -> float:
    """nDCG@k with binary relevance, document-aware, standard log2 discount:
    DCG@k = sum(rel_i / log2(rank_i + 1)) for the top k; nDCG@k = DCG / IDCG.

    Relevance is credited per distinct EXPECTED PAGE, not per chunk: once a
    page has been credited at its first (highest-ranked) occurrence, a later
    chunk from that same page contributes 0 further gain. Without this,
    two different chunks that happen to share one relevant page could each
    score independently and push DCG above IDCG (computed from the count of
    distinct expected pages), which would make nDCG exceed 1.0. This keeps
    the metric numerically valid while staying consistent with how
    document_aware_page_recall() already treats expected_pages as a set.
    """
    if not expected_pages:
        return 1.0

    expected = set(expected_pages)
    seen_pages = set()
    dcg = 0.0
    for rank, chunk in enumerate(chunks[:k], start=1):
        if not chunk_is_relevant(chunk, expected_pages, expected_source):
            continue
        page = int(chunk["page"])
        if page in seen_pages:
            continue
        seen_pages.add(page)
        dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(expected))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def matched_terms(answer: str, expected_terms: list[str]) -> tuple[list[str], list[str]]:
    """Which expected_terms appear (as a normalized substring) in the answer.

    No per-question aliases or special cases - uniform substring matching for
    every term, on every question.
    """
    answer_norm = normalize(answer)
    matched, missing = [], []

    for term in expected_terms:
        if normalize(term) in answer_norm:
            matched.append(term)
        else:
            missing.append(term)

    return matched, missing


def term_coverage(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    matched, _ = matched_terms(answer, expected_terms)
    return len(matched) / len(expected_terms)


def evaluate_question(
    item: dict,
    retrieval_fn,
    prompt_fn,
    generate_fn,
    default_source: str | None = None,
) -> dict:
    """Evaluate one question. Never raises - retrieval and generation errors
    are caught separately and recorded as scored (zero-credit) failures with
    their error type/message, so no question is ever silently dropped from
    an aggregate.

    prompt_fn must be a contamination-free prompt builder (e.g.
    src.generation.eval_prompt_builder.build_eval_prompt) - ground truth
    (expected_pages/expected_terms) is never passed to it.
    """
    question = item["question"]
    expected_pages = item.get("expected_pages", [])
    expected_terms = item.get("expected_terms", [])
    expected_source = item.get("source") or default_source
    category = item.get("category") or category_for_question(item["id"])

    base = {
        "id": item["id"],
        "category": category,
        "question": question,
        "expected_pages": expected_pages,
        "expected_source": expected_source,
        "expected_terms": expected_terms,
    }

    try:
        retrieval_start = time.perf_counter()
        chunks = retrieval_fn(question)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    except Exception as exc:
        return {
            **base,
            "retrieved_pages": [],
            "missing_pages": list(expected_pages),
            "matched_terms": [],
            "missing_terms": list(expected_terms),
            "retrieval_recall": 0.0,
            "term_coverage": 0.0,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "failure_type": "RETRIEVAL_ERROR",
            "retrieval_ms": None,
            "generation_ms": None,
            "total_ms": None,
            "answer": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    retrieved_pages = sorted(
        {
            int(c["page"])
            for c in chunks
            if expected_source is None or c.get("source") == expected_source
        }
    )
    missing_pages = [p for p in expected_pages if p not in retrieved_pages]
    recall = document_aware_page_recall(chunks, expected_pages, expected_source)
    mrr = reciprocal_rank_at_k(chunks, expected_pages, expected_source, k=RANKING_K)
    ndcg = ndcg_at_k(chunks, expected_pages, expected_source, k=RANKING_K)

    try:
        prompt = prompt_fn(question, chunks)
        generation_start = time.perf_counter()
        answer = generate_fn(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000
    except Exception as exc:
        return {
            **base,
            "retrieved_pages": retrieved_pages,
            "missing_pages": missing_pages,
            "matched_terms": [],
            "missing_terms": list(expected_terms),
            "retrieval_recall": recall,
            "term_coverage": 0.0,
            "mrr_at_5": mrr,
            "ndcg_at_5": ndcg,
            "failure_type": "GENERATION_ERROR",
            "retrieval_ms": retrieval_ms,
            "generation_ms": None,
            "total_ms": None,
            "answer": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    matched, missing_terms = matched_terms(answer, expected_terms)
    coverage = (len(matched) / len(expected_terms)) if expected_terms else 1.0

    if missing_pages:
        failure_type = "RETRIEVAL_CONTENT_MISS"
    elif missing_terms:
        failure_type = "GENERATION_CONTENT_MISS"
    else:
        failure_type = "PASS"

    return {
        **base,
        "retrieved_pages": retrieved_pages,
        "missing_pages": missing_pages,
        "matched_terms": matched,
        "missing_terms": missing_terms,
        "retrieval_recall": recall,
        "term_coverage": coverage,
        "mrr_at_5": mrr,
        "ndcg_at_5": ndcg,
        "failure_type": failure_type,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": retrieval_ms + generation_ms,
        "answer": answer,
        "error_type": None,
        "error_message": None,
    }


def aggregate(results: list[dict]) -> dict:
    """Summarize a list of evaluate_question() results.

    attempted == scored always: every attempted question contributes to the
    mean recall/coverage (errors score 0.0), so failures can never be
    silently excluded from the reported averages. Latency means are computed
    only over questions that produced a timing value, with that count
    reported explicitly (timed_count) rather than hidden.
    """
    attempted = len(results)

    counts = {
        "passes": 0,
        "retrieval_content_misses": 0,
        "generation_content_misses": 0,
        "retrieval_errors": 0,
        "generation_errors": 0,
    }
    label_to_key = {
        "PASS": "passes",
        "RETRIEVAL_CONTENT_MISS": "retrieval_content_misses",
        "GENERATION_CONTENT_MISS": "generation_content_misses",
        "RETRIEVAL_ERROR": "retrieval_errors",
        "GENERATION_ERROR": "generation_errors",
    }
    for r in results:
        counts[label_to_key[r["failure_type"]]] += 1

    recall_values = [r["retrieval_recall"] for r in results]
    coverage_values = [r["term_coverage"] for r in results]
    # .get(): older persisted results predate MRR@5/nDCG@5 and won't have
    # these keys - they're excluded from the mean rather than raising, with
    # the count reported explicitly (mirrors how timed_count already works).
    mrr_values = [r["mrr_at_5"] for r in results if r.get("mrr_at_5") is not None]
    ndcg_values = [r["ndcg_at_5"] for r in results if r.get("ndcg_at_5") is not None]
    timed = [r for r in results if r["retrieval_ms"] is not None and r["generation_ms"] is not None]

    return {
        "attempted": attempted,
        "scored": attempted,
        "passes": counts["passes"],
        "retrieval_failures": counts["retrieval_content_misses"] + counts["retrieval_errors"],
        "generation_failures": counts["generation_content_misses"] + counts["generation_errors"],
        "retrieval_content_misses": counts["retrieval_content_misses"],
        "generation_content_misses": counts["generation_content_misses"],
        "retrieval_errors": counts["retrieval_errors"],
        "generation_errors": counts["generation_errors"],
        "mean_retrieval_recall": mean(recall_values) if recall_values else None,
        "mean_term_coverage": mean(coverage_values) if coverage_values else None,
        "mean_mrr_at_5": mean(mrr_values) if mrr_values else None,
        "mean_ndcg_at_5": mean(ndcg_values) if ndcg_values else None,
        "ranking_metrics_count": len(mrr_values),
        "timed_count": len(timed),
        "mean_retrieval_ms": mean(r["retrieval_ms"] for r in timed) if timed else None,
        "mean_generation_ms": mean(r["generation_ms"] for r in timed) if timed else None,
        "mean_total_ms": mean(r["total_ms"] for r in timed) if timed else None,
    }
