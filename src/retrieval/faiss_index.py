"""FAISS index wrapper: inner-product search over normalized vectors (= cosine
similarity), with explicit chunk_id vector ids via IndexIDMap so the index
stays reliably mapped to SQLite rows.

`faiss` is imported lazily inside each method rather than at module level:
importing it eagerly (e.g. via app.py's module-level import chain) added
noticeable cold-start latency to every `streamlit run`, even before any
document was uploaded or queried.
"""

import os
import threading
from pathlib import Path

import numpy as np

from ..exceptions import IndexNotFoundError

# Every write to an index file is load -> mutate -> save of the WHOLE file, so
# two writers overlapping in time lose an update: both load the same old
# state, and whichever saves last silently drops the other's vectors. Streamlit
# runs each browser session as a thread of one process, so two concurrent
# uploads were measured doing exactly that - both reported success, one
# document's chunks ended up in SQLite with no FAISS vectors. Writers hold this
# lock across their whole load..save section. It is process-wide only: a
# separate process (the ingest CLI) writing the same file concurrently is not
# covered.
INDEX_WRITE_LOCK = threading.RLock()


class FaissIndex:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = None

    def load_or_create(self):
        """Load the index from disk if present, otherwise create an empty one."""
        import faiss

        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        return self.index

    def load(self):
        """Load the index from disk; raises if it does not exist."""
        import faiss

        if not Path(self.index_path).exists():
            raise IndexNotFoundError(
                f"FAISS index not found at '{self.index_path}'. Run ingestion first."
            )
        self.index = faiss.read_index(self.index_path)
        return self.index

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if self.index is None:
            raise RuntimeError("Index not loaded/created. Call load_or_create() first.")
        self.index.add_with_ids(vectors.astype("float32"), ids.astype("int64"))

    def save(self) -> None:
        import faiss

        if self.index is None:
            raise RuntimeError("No index to save. Call load_or_create() first.")
        target = Path(self.index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target, then atomically swap it in: a crash mid-write
        # or a query reading the file concurrently can never see a half-written
        # index, which would make every later load fail.
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            faiss.write_index(self.index, str(tmp))
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, chunk_ids) arrays of length <= top_k. chunk_id == -1 means no match."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(query_vector, top_k)
        return scores[0], ids[0]

    def remove_ids(self, ids) -> int:
        """Remove vectors by id in place (caller must still call save()).

        IndexFlatIP (wrapped by IndexIDMap) stores vectors uncompressed, so
        exact removal-by-id is supported directly - no rebuild-from-scratch
        or re-embedding is needed. Returns the number of vectors removed.
        """
        import faiss

        if self.index is None:
            raise RuntimeError("Index not loaded/created. Call load() first.")
        ids_arr = np.asarray(list(ids), dtype="int64")
        if ids_arr.size == 0:
            return 0
        selector = faiss.IDSelectorBatch(ids_arr)
        return int(self.index.remove_ids(selector))
