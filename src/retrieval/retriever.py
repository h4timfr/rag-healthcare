"""Query-time retrieval: embed the query, search FAISS, resolve metadata from SQLite."""

from .. import config
from ..embedding.embedder import embed_query
from ..exceptions import EmptyQueryError, InsufficientRetrievalError
from .faiss_index import FaissIndex
from .metadata_store import MetadataStore


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return the top-k retrieved chunks for a query, each with source/page/text/score.

    Raises EmptyQueryError, IndexNotFoundError, DatabaseNotFoundError, or
    InsufficientRetrievalError as appropriate.
    """
    query = query.strip()
    if not query:
        raise EmptyQueryError("Query must not be empty")

    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load()

    store = MetadataStore(config.DB_PATH)

    query_vector = embed_query(query)
    scores, ids = index.search(query_vector, top_k)

    candidates = [(int(i), float(s)) for i, s in zip(ids, scores) if i != -1]
    if not candidates:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    metadata = store.get_chunks([chunk_id for chunk_id, _ in candidates])

    results = []
    for chunk_id, score in candidates:
        meta = metadata.get(chunk_id)
        if meta is None:
            # FAISS returned an id with no matching SQLite row (index/db drift) - skip it.
            continue
        results.append({**meta, "score": score})

    if not results:
        raise InsufficientRetrievalError(
            "Retrieved vector ids had no matching metadata rows (FAISS/SQLite out of sync)"
        )

    return results
