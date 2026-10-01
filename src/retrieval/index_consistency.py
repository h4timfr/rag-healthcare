"""Startup consistency check/repair between the SQLite metadata store and the
FAISS index (see faiss_index.py's module docstring for the write path that
motivates this: a lost update or a mid-crash leaves one store ahead of the
other).

check_consistency() is read-only and cheap - it compares chunk_id SETS
(never chunk text or vectors) plus the index's own dimension, so it is safe
to run on every app startup. repair() only re-embeds or removes the specific
chunk_ids the check found wrong; it never touches a chunk that is already
consistent, and it never deletes a SQLite row (the one place chunk TEXT
lives) to fix a FAISS-side problem.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from ..embedding.embedder import embed_texts
from .faiss_index import FaissIndex, cross_process_index_lock
from .metadata_store import MetadataStore

# A vector search only means "cosine similarity" (see faiss_index.py's module
# docstring) if every stored vector has unit norm - IndexFlatIP computes a
# plain dot product, not a normalized one. A vector outside this tolerance
# means something bypassed embed_texts()'s normalize_embeddings=True, e.g. a
# vector inserted by a future code path that forgot to normalize.
_NORM_TOLERANCE = 1e-3


@dataclass
class ConsistencyReport:
    total_sqlite_chunks: int
    total_faiss_vectors: int
    dense_invisible_chunk_ids: list[int] = field(default_factory=list)
    """In SQLite, missing from FAISS - the known lost-update failure. Text is
    still available, so these are repairable by re-embedding."""
    orphan_faiss_ids: list[int] = field(default_factory=list)
    """In FAISS, missing from SQLite - e.g. a delete interrupted between its
    SQLite commit and its FAISS save (see app.delete_document's docstring).
    Already harmless (hybrid_retriever drops any chunk_id with no SQLite
    row), but repairable by removing the orphan vector."""
    duplicate_faiss_ids: list[int] = field(default_factory=list)
    """A chunk_id appearing more than once in the FAISS id map - should be
    structurally impossible via add()/remove_ids(), reported as a hard
    inconsistency if ever seen rather than silently tolerated."""
    dimension_mismatch: bool = False
    expected_dim: int = 0
    actual_dim: int | None = None
    unnormalized_chunk_ids: list[int] = field(default_factory=list)
    faiss_corrupted: bool = False
    corruption_detail: str | None = None

    @property
    def is_consistent(self) -> bool:
        """Whether retrieval is currently safe to use as-is. Unnormalized
        vectors are reported but do NOT affect this: they still return
        *a* score, just not a directly comparable cosine similarity, so
        retrieval degrades rather than breaks - see repair()'s docstring for
        why they are never auto-corrected."""
        return not (
            self.dense_invisible_chunk_ids
            or self.orphan_faiss_ids
            or self.duplicate_faiss_ids
            or self.dimension_mismatch
            or self.faiss_corrupted
        )


def _faiss_ids_and_dim(faiss_index_path: str, dim: int):
    """Returns (id_list, actual_dim, corrupted_detail). id_list has one entry
    per FAISS vector, in index order, duplicates included (so callers can
    detect them) - or None if the index file doesn't exist yet."""
    if not Path(faiss_index_path).exists():
        return [], dim, None

    import faiss

    try:
        index = faiss.read_index(faiss_index_path)
    except Exception as exc:  # noqa: BLE001 - faiss raises its own RuntimeError subtype
        return None, None, str(exc)

    ids = [int(x) for x in faiss.vector_to_array(index.id_map)]
    return ids, index.d, None


def _vector_norms(faiss_index_path: str, chunk_ids: list[int]) -> dict[int, float]:
    """Norm of each requested chunk_id's own stored vector. Reads the raw
    float array directly rather than via search(), since search() only
    returns vectors near a query - this needs every vector, unconditionally."""
    if not chunk_ids:
        return {}
    import faiss

    index = faiss.read_index(faiss_index_path)
    flat = faiss.downcast_index(index.index)
    xb = faiss.rev_swig_ptr(flat.get_xb(), flat.ntotal * flat.d).reshape(flat.ntotal, flat.d)
    ids = [int(x) for x in faiss.vector_to_array(index.id_map)]
    position = {cid: i for i, cid in enumerate(ids)}  # last occurrence wins for a duplicate
    return {cid: float(np.linalg.norm(xb[position[cid]])) for cid in chunk_ids if cid in position}


def check_consistency(
    db_path: str | None = None,
    faiss_index_path: str | None = None,
    dim: int | None = None,
    verify_normalization: bool = True,
) -> ConsistencyReport:
    """Read-only comparison of the two stores. Never raises for a missing
    database or index file - either is simply "0 chunks"/"0 vectors", which
    is the correct, consistent state before the first document is ingested.
    """
    db_path = db_path or config.DB_PATH
    faiss_index_path = faiss_index_path or config.FAISS_INDEX_PATH
    dim = dim if dim is not None else config.EMBEDDING_DIM

    sql_ids: list[int] = []
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            )
            if cur.fetchone():
                sql_ids = [row[0] for row in conn.execute("SELECT chunk_id FROM chunks")]
        finally:
            conn.close()
    sql_id_set = set(sql_ids)

    faiss_ids, actual_dim, corruption = _faiss_ids_and_dim(faiss_index_path, dim)

    if faiss_ids is None:
        return ConsistencyReport(
            total_sqlite_chunks=len(sql_id_set),
            total_faiss_vectors=0,
            faiss_corrupted=True,
            corruption_detail=corruption,
            expected_dim=dim,
        )

    faiss_id_set = set(faiss_ids)
    seen: set[int] = set()
    duplicates = sorted({i for i in faiss_ids if i in seen or seen.add(i)})

    report = ConsistencyReport(
        total_sqlite_chunks=len(sql_id_set),
        total_faiss_vectors=len(faiss_ids),
        dense_invisible_chunk_ids=sorted(sql_id_set - faiss_id_set),
        orphan_faiss_ids=sorted(faiss_id_set - sql_id_set),
        duplicate_faiss_ids=duplicates,
        expected_dim=dim,
        actual_dim=actual_dim,
        dimension_mismatch=bool(faiss_ids) and actual_dim != dim,
    )

    # A dimension mismatch means the vectors are not this embedding model's
    # output at all - reading them as if they were (e.g. to check norms)
    # would only produce meaningless numbers on top of an already-fatal
    # problem, so that check is skipped here.
    if verify_normalization and not report.dimension_mismatch:
        present_sql_ids = sorted(sql_id_set & faiss_id_set)
        norms = _vector_norms(faiss_index_path, present_sql_ids)
        report.unnormalized_chunk_ids = sorted(
            cid for cid, norm in norms.items() if abs(norm - 1.0) > _NORM_TOLERANCE
        )

    return report


def repair(
    db_path: str | None = None,
    faiss_index_path: str | None = None,
    dim: int | None = None,
) -> ConsistencyReport:
    """Fix what check_consistency() found, using the least invasive action
    for each problem, then return the post-repair report.

    - dense_invisible_chunk_ids: re-embed that chunk's own TEXT (still in
      SQLite - nothing was lost) and add the vector back. Never touches any
      chunk that was already fine.
    - orphan_faiss_ids: removed from FAISS only. Their SQLite row (if any -
      by definition there is none) is never touched - this can only ever
      shrink FAISS's extra vectors, never SQLite's data.
    - duplicate_faiss_ids: FAISS's remove_ids() removes EVERY vector with a
      matching id, not "all but one" - so a duplicate id is removed
      entirely and then re-embedded exactly once, the same as a
      dense-invisible chunk, rather than trying to guess which of the
      duplicate copies (which may not even be identical) was the right one.
    - faiss_corrupted: the one case with no incremental fix (a corrupt file
      has no readable id map to diff against) - rebuilds the WHOLE index by
      re-embedding every chunk currently in SQLite. This is the sole
      "rebuild everything" path, taken only when nothing narrower is
      possible, and is logged as such by the caller via the returned report
      (faiss_corrupted was set on the report the caller already has).
    - dimension_mismatch / unnormalized_chunk_ids: never auto-repaired. Both
      indicate the vectors are not what this embedding model would have
      produced, and silently re-embedding over them is a data-changing
      decision this function does not make on its own - reported for a
      human (or a future, explicit --rebuild flag) to act on.
    """
    db_path = db_path or config.DB_PATH
    faiss_index_path = faiss_index_path or config.FAISS_INDEX_PATH
    dim = dim if dim is not None else config.EMBEDDING_DIM

    # The snapshot is taken under the writer lock: an ingest between its
    # SQLite commit and its FAISS save looks exactly like a dense-invisible
    # chunk, and acting on a snapshot taken before that ingest finished
    # would add its vectors a second time.
    with cross_process_index_lock(faiss_index_path):
        before = check_consistency(db_path, faiss_index_path, dim, verify_normalization=False)

        if before.faiss_corrupted:
            _rebuild_from_sqlite(db_path, faiss_index_path, dim)
        elif before.dense_invisible_chunk_ids or before.orphan_faiss_ids or before.duplicate_faiss_ids:
            index = FaissIndex(dim, faiss_index_path)
            index.load_or_create()

            to_remove = sorted(set(before.orphan_faiss_ids) | set(before.duplicate_faiss_ids))
            if to_remove:
                index.remove_ids(to_remove)

            # A duplicate id had ALL its copies removed just above (FAISS
            # has no "remove all but one" op), so it needs the same
            # re-embed-from-SQLite-text treatment as a dense-invisible
            # chunk to end up with exactly one vector again.
            needs_embedding = sorted(set(before.dense_invisible_chunk_ids) | set(before.duplicate_faiss_ids))
            if needs_embedding:
                store = MetadataStore(db_path)
                chunks = store.get_chunks(needs_embedding)
                # Only chunk_ids get_chunks() actually found are repairable;
                # one vanishing between the check and here (a concurrent
                # delete) is not an error, just nothing left to re-embed.
                ids = [cid for cid in needs_embedding if cid in chunks]
                if ids:
                    vectors = embed_texts([chunks[cid]["text"] for cid in ids])
                    index.add(vectors, np.array(ids, dtype="int64"))

            if to_remove or needs_embedding:
                index.save()

        return check_consistency(db_path, faiss_index_path, dim)


def _rebuild_from_sqlite(db_path: str, faiss_index_path: str, dim: int) -> None:
    """Re-embed every chunk currently in SQLite into a brand new FAISS
    index, replacing a corrupt one. SQLite is untouched and is the sole
    source of truth for this - a chunk's text and metadata are never
    regenerated, only its vector."""
    rows = []
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        try:
            if conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            ).fetchone():
                rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
        finally:
            conn.close()

    with cross_process_index_lock(faiss_index_path):
        # The corrupt file is already confirmed unreadable (that's why we're
        # here) and unsalvageable - nothing in repair() ever reads chunk
        # vectors back out of FAISS, only chunk_ids and TEXT out of SQLite,
        # so there is nothing left to lose by removing it before creating a
        # fresh index in its place. With no SQLite rows every vector it
        # could hold is an orphan, so it is removed then too: left in place,
        # every later ingest's load_or_create() - which calls
        # faiss.read_index() on any EXISTING path - re-raises the read error.
        Path(faiss_index_path).unlink(missing_ok=True)
        if not rows:
            return
        index = FaissIndex(dim, faiss_index_path)
        index.load_or_create()
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        vectors = embed_texts(texts)
        index.add(vectors, np.array(ids, dtype="int64"))
        index.save()
