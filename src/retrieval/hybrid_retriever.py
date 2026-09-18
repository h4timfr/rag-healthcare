"""Hybrid dense + keyword retrieval using Reciprocal Rank Fusion."""

import time

from .. import config
from ..embedding.embedder import embed_query, embed_texts
from ..generation.intent import decompose_question
from .faiss_index import FaissIndex
from .keyword_index import KeywordIndex
from .metadata_store import MetadataStore


def _allocate_by_document(
    ranked_by_score: list[int],
    document_of_chunk: dict[int, str],
    document_count: int,
    top_k: int,
) -> list[int]:
    """Pick `top_k` chunks from an already-fused ranking, giving each
    document a fair share of the budget first.

    "Summarize each selected document separately" over three documents used
    to be answered from whichever documents happened to rank highest overall:
    the budget was TOP_K * 3, but nothing stopped one document taking most of
    it, and a document with no chunks in the final list can only be written
    about by guessing.

    The fix is purely in selection, at the very end of retrieval. Every
    upstream stage is untouched: the same FAISS search, the same FTS5 search,
    the same RRF weights and rrf_k, the same candidate_k, the same
    document-scope filtering. All that changes is which of the ALREADY fused
    and ALREADY scoped candidates survive truncation.

    Pass 1 gives each document up to `top_k // document_count` of its own
    best-scoring chunks. Pass 2 fills whatever is left by global score.

    Crucially this cannot pull an irrelevant document into the answer: a
    document only has chunks here if its chunks were actually retrieved by
    the dense or keyword search for this query. A document that matched
    nothing contributes nothing, and its budget goes back to pass 2.
    """
    quota = max(1, top_k // document_count)

    taken: list[int] = []
    per_document: dict[str, int] = {}

    for chunk_id in ranked_by_score:
        document = document_of_chunk.get(chunk_id)
        if document is None:
            continue
        if per_document.get(document, 0) < quota:
            per_document[document] = per_document.get(document, 0) + 1
            taken.append(chunk_id)

    taken = taken[:top_k]

    if len(taken) < top_k:
        already = set(taken)
        taken.extend(
            chunk_id
            for chunk_id in ranked_by_score
            # Same document_of_chunk guard as pass 1: a chunk deleted
            # between fusion and this point has no document, and would be
            # dropped again by get_chunks() below - spending a slot on it
            # would silently shrink the evidence the answer is built from.
            if chunk_id not in already and chunk_id in document_of_chunk
        )
        taken = taken[:top_k]

    return taken


def retrieve_hybrid(
    query: str,
    top_k: int = config.TOP_K,
    candidate_k: int = 10,
    rrf_k: int = 60,
    document_id: str | list[str] | None = None,
    timing: dict | None = None,
    balance_by_document: bool = False,
    query_vector=None,
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

    `balance_by_document` (default False - single-document retrieval and
    ordinary multi-document questions are completely unaffected) changes only
    which chunks are kept from the END of the fused ranking, for
    each-document and comparison questions. See _allocate_by_document().

    `query_vector` (optional): a precomputed, already-normalized embedding
    for `query` - e.g. from one batched embed_texts() call across several
    sub-questions of a single multi-part retrieval (see
    retrieve_for_question(), the only caller that passes this). When given,
    embed_query() is never called here, and `timing["embedding_ms"]` is left
    untouched by this function - the caller already paid that cost once, for
    the whole batch, and is responsible for reporting it itself. Omitted
    (the default - every caller except retrieve_for_question's multi-part
    path), behavior is exactly as before: embed_query(query) runs here and
    populates timing["embedding_ms"] as it always has. The resulting search
    is identical either way - FAISS never knows or cares where the vector
    came from.
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
    embedded_here = query_vector is None
    if embedded_here:
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

    ranked_by_score = [
        chunk_id
        for chunk_id, _ in sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    if balance_by_document and document_ids and len(document_ids) > 1:
        ranked_ids = _allocate_by_document(
            ranked_by_score,
            store.document_ids_for_chunks(ranked_by_score),
            len(document_ids),
            top_k,
        )
        ranked_ids.sort(key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
    else:
        ranked_ids = ranked_by_score[:top_k]

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
        if embedded_here:
            timing["embedding_ms"] = (t1 - t0) * 1000
        timing["dense_ms"] = (t2 - t1) * 1000
        timing["keyword_ms"] = (t3 - t2) * 1000
        timing["fusion_ms"] = (t4 - t3) * 1000

    return results


def _call_retrieve(retrieve_fn, query, top_k, document_id, timing, query_vector=None):
    """retrieve_fn(query, top_k=..., [document_id=...], [timing=...],
    [query_vector=...]), omitting document_id/timing/query_vector whenever
    they are None - the same conditional-kwarg pattern
    retrieve_and_build_prompt() already uses, kept here so a narrower test
    double of retrieve_hybrid (no document_id/timing/query_vector parameter
    at all, e.g. chat.py's existing tests) keeps working unchanged for the
    single-part path, which is what almost every question still takes and
    which never passes query_vector at all (see retrieve_for_question).
    """
    kwargs = {"top_k": top_k}
    if document_id is not None:
        kwargs["document_id"] = document_id
    if timing is not None:
        kwargs["timing"] = timing
    if query_vector is not None:
        kwargs["query_vector"] = query_vector
    return retrieve_fn(query, **kwargs)


def retrieve_for_question(
    query: str,
    history: list[dict] | None = None,
    top_k: int = config.TOP_K,
    document_id: str | list[str] | None = None,
    timing: dict | None = None,
    retrieve_fn=None,
) -> list[dict]:
    """retrieve_hybrid(), decomposition-aware: a question with more than one
    requested part - "X, and what about Y?", "X? What about Y?", or a bare
    "what about the first?" resolved against `history` - is retrieved ONE
    PART AT A TIME instead of as a single blended embedding of the whole
    question. See intent.decompose_question() for how parts are found and
    elliptical references resolved.

    Why this exists: a single embedding of a compound question is a blend of
    both parts' meaning, and nothing guarantees either part's own evidence
    survives into the fused top-k. "What happened during the second
    admission? What about the first one?" retrieved chunks skewed toward
    whichever admission dominated the blended embedding, and the model
    answered mostly about that one - faithful to the (lopsided) evidence it
    was given, not a generation defect. Retrieving each part on its own
    guarantees each one gets its own shot at the index - the same guarantee
    _allocate_by_document() already gives each selected document for
    each-document questions, applied along the other axis.

    A single-part question - the overwhelming majority - takes exactly the
    old path: one retrieve_hybrid() call with the caller's own top_k,
    document_id and timing, returning results unchanged. `history` never
    reaches embedding or FTS5 either way - it only ever resolves a part's
    text before retrieval runs (see decompose_question), never blurs the
    query itself with conversation transcript.

    For 2+ parts, the per-part budget is `top_k` chunks split evenly across
    the parts, capped by config.MAX_CONTEXT_CHUNKS - the same ceiling
    multi-document balanced allocation already answers to, reused here
    rather than a second magic number, so a question cannot grow its own
    prompt without bound just by having more parts. Each returned chunk
    carries `part` (1-indexed) and `part_query` (that part's own resolved
    question text) in addition to the usual fields, so prompt_builder can
    group evidence by requested part; chunks are deduplicated by chunk_id
    (the first part to surface a chunk keeps it) and re-ranked 1..N over the
    merged, deduplicated list, so citation numbers stay one consistent
    sequence exactly as they are for any other question.

    `retrieve_fn` defaults to this module's own retrieve_hybrid. A caller
    that monkeypatches a different reference to it (app.py's and chat.py's
    tests both patch their own module-level `retrieve_hybrid` name) must
    pass that same reference explicitly - the patch has no effect on this
    module's name otherwise.
    """
    if retrieve_fn is None:
        retrieve_fn = retrieve_hybrid

    parts = decompose_question(query, history=history)

    if len(parts) <= 1:
        resolved_query = parts[0] if parts else query
        return _call_retrieve(retrieve_fn, resolved_query, top_k, document_id, timing)

    per_part_top_k = max(1, min(top_k, config.MAX_CONTEXT_CHUNKS // len(parts)))

    # Embed every part in ONE batched model call instead of N separate ones.
    # sentence-transformers' encode() already batches internally - asking it
    # for N texts at once pays the per-call overhead (tokenization, model
    # dispatch) once, not N times, which is what made this path measurably
    # slower than a single-part retrieval (~9.3s for 2 parts on the
    # reference machine, embedding alone). The embeddings this returns are
    # numerically equivalent, element for element and in the SAME ORDER as
    # `parts`, to calling embed_query() on each part one at a time - see
    # test_multi_part_retrieval.py's equivalence tests, which assert the
    # actual FAISS search results this produces are identical to the old
    # per-part embed_query() behavior, not just that the code runs.
    #
    # FTS5 keyword search is deliberately NOT batched here: it already runs
    # in ~10-20ms per part, and there is no way to combine several distinct
    # MATCH queries into one FTS5 call without changing what "top-k keyword
    # matches for THIS part" means - that would touch ranking, which this
    # change must not do.
    embed_start = time.perf_counter()
    part_vectors = embed_texts(parts)
    batch_embedding_ms = (time.perf_counter() - embed_start) * 1000

    merged: list[dict] = []
    seen_chunk_ids: set = set()
    part_timings: list[dict] = []

    for part_number, (part_query, part_vector) in enumerate(zip(parts, part_vectors), start=1):
        part_timing = {} if timing is not None else None
        part_chunks = _call_retrieve(
            retrieve_fn, part_query, per_part_top_k, document_id, part_timing, query_vector=part_vector
        )
        if part_timing is not None:
            part_timings.append(part_timing)
        for chunk in part_chunks:
            if chunk["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk["chunk_id"])
            merged.append({**chunk, "part": part_number, "part_query": part_query})

    for rank, chunk in enumerate(merged, start=1):
        chunk["rank"] = rank

    if timing is not None:
        # embedding_ms is the ONE batch cost above, not a sum over parts -
        # summing would double- (or N-) count a cost that was only ever
        # paid once. dense_ms/keyword_ms/fusion_ms genuinely do happen once
        # per part, so those are still summed exactly as before.
        timing["embedding_ms"] = batch_embedding_ms
        for key in ("dense_ms", "keyword_ms", "fusion_ms"):
            timing[key] = sum(t.get(key, 0.0) for t in part_timings)

    return merged