"""Hybrid dense + keyword retrieval using Reciprocal Rank Fusion."""

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
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve using FAISS + FTS5 and fuse rankings with weighted RRF.

    When document_id is given, both the dense and keyword candidate lists are
    restricted to that document (via SQLite metadata - FAISS/FTS5 have no
    notion of document_id themselves) before RRF fusion, so the fused result
    only ever contains chunks from that one document. When document_id is
    None, retrieval is global and behaves exactly as before.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty")

    faiss_index = FaissIndex(
        config.EMBEDDING_DIM,
        config.FAISS_INDEX_PATH,
    )
    faiss_index.load()

    keyword_index = KeywordIndex(config.DB_PATH)
    store = MetadataStore(config.DB_PATH)

    # When scoping to one document, over-retrieve candidates across the whole
    # index so filtering down to that document still leaves enough of its
    # chunks to fuse from, even if they don't dominate the global top-k.
    effective_candidate_k = candidate_k
    if document_id is not None:
        effective_candidate_k = max(candidate_k, store.count())

    # Dense retrieval
    query_vector = embed_query(query)

    dense_scores, dense_ids = faiss_index.search(
        query_vector,
        effective_candidate_k,
    )

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

    keyword_ids = [
        int(result["chunk_id"])
        for result in keyword_results
    ]

    if document_id is not None:
        allowed = store.filter_chunk_ids_by_document(
            list(set(dense_ids) | set(keyword_ids)),
            document_id,
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

    return results