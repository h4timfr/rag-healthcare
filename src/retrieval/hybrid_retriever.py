"""Hybrid dense + keyword retrieval using Reciprocal Rank Fusion."""

import time

from .. import config
from ..embedding.embedder import embed_query
from .faiss_index import FaissIndex
from .keyword_index import KeywordIndex
from .metadata_store import MetadataStore


def retrieve_hybrid(
    query: str,
    top_k: int = config.TOP_K,
    candidate_k: int = 10,
    rrf_k: int = 60,
    document_id: str | list[str] | None = None,
    timing: dict | None = None,
) -> list[dict]:
    """Retrieve using FAISS + FTS5 and fuse rankings with weighted RRF.

    When document_id is given, both the dense and keyword candidate lists are
    restricted to that document (or, if document_id is a list, to that SET of
    documents - multi-document analysis mode) via SQLite metadata - FAISS/FTS5
    have no notion of document_id themselves - before RRF fusion, so the
    fused result only ever contains chunks from those document(s). When
    document_id is None, retrieval is global and behaves exactly as before.
    An explicitly empty list means "no documents to search": retrieval
    returns no results rather than silently falling back to a global search,
    so an empty multi-document selection can never leak unselected documents.
    RRF weights/rrf_k/candidate_k are unchanged regardless of how many
    documents are scoped.

    If `timing` is given, it is populated in place (milliseconds) with
    `embedding_ms`, `dense_ms`, `keyword_ms`, and `fusion_ms` - pure
    instrumentation, added around the existing steps without changing their
    order or logic, for the app's per-turn performance panel and the
    scripts/profile_query.py CLI. Omit it (the default) for zero overhead
    beyond a few negligible perf_counter() calls.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty")

    document_ids: list[str] | None = None
    if document_id is not None:
        document_ids = [document_id] if isinstance(document_id, str) else list(document_id)
        if not document_ids:
            return []

    faiss_index = FaissIndex(
        config.EMBEDDING_DIM,
        config.FAISS_INDEX_PATH,
    )
    faiss_index.load()

    keyword_index = KeywordIndex(config.DB_PATH)
    store = MetadataStore(config.DB_PATH)

    # When scoping to one or more documents, over-retrieve candidates across
    # the whole index so filtering down to those documents still leaves
    # enough of their chunks to fuse from, even if they don't dominate the
    # global top-k.
    effective_candidate_k = candidate_k
    if document_ids is not None:
        effective_candidate_k = max(candidate_k, store.count())

    # Dense retrieval
    t0 = time.perf_counter()
    query_vector = embed_query(query)
    t1 = time.perf_counter()

    dense_scores, dense_ids = faiss_index.search(
        query_vector,
        effective_candidate_k,
    )
    t2 = time.perf_counter()

    dense_ids = [
        int(chunk_id)
        for chunk_id in dense_ids
        if chunk_id != -1
    ]

    # Keyword retrieval
    keyword_results = keyword_index.search(
        query,
        effective_candidate_k,
    )
    t3 = time.perf_counter()

    keyword_ids = [
        int(result["chunk_id"])
        for result in keyword_results
    ]

    if document_ids is not None:
        allowed = store.filter_chunk_ids_by_documents(
            list(set(dense_ids) | set(keyword_ids)),
            document_ids,
        )
        dense_ids = [chunk_id for chunk_id in dense_ids if chunk_id in allowed]
        keyword_ids = [chunk_id for chunk_id in keyword_ids if chunk_id in allowed]

    # Weighted Reciprocal Rank Fusion
    fused_scores: dict[int, float] = {}

    dense_weight = 2.0
    keyword_weight = 1.0

    for rank, chunk_id in enumerate(dense_ids, start=1):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
            dense_weight / (rrf_k + rank)
        )

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
            keyword_weight / (rrf_k + rank)
        )

    ranked_ids = [
        chunk_id
        for chunk_id, _ in sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]

    metadata = store.get_chunks(ranked_ids)

    results = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        meta = metadata.get(chunk_id)

        if meta is None:
            continue

        results.append(
            {
                **meta,
                "score": fused_scores[chunk_id],
                "rank": rank,
            }
        )

    t4 = time.perf_counter()

    if timing is not None:
        timing["embedding_ms"] = (t1 - t0) * 1000
        timing["dense_ms"] = (t2 - t1) * 1000
        timing["keyword_ms"] = (t3 - t2) * 1000
        timing["fusion_ms"] = (t4 - t3) * 1000

    return results