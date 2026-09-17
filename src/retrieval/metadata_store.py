"""SQLite storage for chunk metadata (document_id, source, page, text).

chunk_id is the SQLite autoincrement primary key, and the same integer is used
as the FAISS vector id (via IndexIDMap) so the two stores stay in lockstep.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..exceptions import DatabaseNotFoundError

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(
    chunk_id UNINDEXED,
    text
);
"""


class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def initialize(self) -> None:
        """Create the database schema and synchronize the FTS index."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connection(create=True) as conn:
            conn.executescript(SCHEMA)

            conn.execute("DELETE FROM chunks_fts")

            conn.execute(
                """
                INSERT INTO chunks_fts (chunk_id, text)
                SELECT chunk_id, text
                FROM chunks
                """
            )

    @contextmanager
    def _connection(self, create: bool = False):
        """One connection = one transaction: every statement executed by the
        caller through `conn` inside this block commits together at the end,
        or - if anything raises - is rolled back together and none of it is
        persisted. This is what makes insert_chunks() atomic: it just runs
        its whole loop inside one `_connection()` block instead of one per
        chunk (see insert_chunk()/insert_chunks() below).
        """
        if not create and not Path(self.db_path).exists():
            raise DatabaseNotFoundError(
                f"SQLite metadata database not found at '{self.db_path}'. Run ingestion first."
            )

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_chunk(
        self,
        document_id: str,
        source: str,
        page: int,
        text: str,
    ) -> int:
        """Insert a single chunk into SQLite and the keyword index, committed
        immediately. Kept for callers (and existing tests) that genuinely
        want to insert one chunk at a time; ingestion of a whole document
        should use insert_chunks() instead so all of its chunks share one
        transaction - see that method's docstring for why.
        """
        return self.insert_chunks(document_id, source, [(page, text)])[0]

    def insert_chunks(
        self,
        document_id: str,
        source: str,
        chunks: list[tuple[int, str]],
    ) -> list[int]:
        """Insert every chunk of one document - `chunks` as a list of
        (page, text) pairs, in order - inside a single transaction.

        Either all chunks are committed, or (if any insert fails partway
        through) none are: this is what makes a document ingestion atomic
        from SQLite's perspective. Before this existed, ingestion inserted
        one chunk at a time via insert_chunk(), each committing on its own;
        a failure partway through a document left it partially indexed, and
        a later re-ingestion attempt would see document_chunk_count() > 0
        and skip re-ingesting the (incomplete) document entirely.

        Returns the resulting chunk_ids, in the same order as `chunks`.
        """
        chunk_ids = []
        with self._connection(create=True) as conn:
            created_at = datetime.now(timezone.utc).isoformat()
            for page, text in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks (document_id, source, page, text, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (document_id, source, page, text, created_at),
                )
                chunk_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                    (chunk_id, text),
                )
                chunk_ids.append(chunk_id)
        return chunk_ids

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, dict]:
        """Resolve a list of chunk_ids to their stored metadata/text."""
        if not chunk_ids:
            return {}

        with self._connection() as conn:
            placeholders = ",".join("?" * len(chunk_ids))

            rows = conn.execute(
                f"SELECT chunk_id, document_id, source, page, text "
                f"FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()

        return {row["chunk_id"]: dict(row) for row in rows}

    def count(self) -> int:
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

    def document_chunk_count(self, document_id: str) -> int:
        """Return how many chunks are already stored for this document_id (0 if none).

        Used to detect that a document was already ingested (same content hash)
        before inserting duplicate rows/vectors for it.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return row[0]

    def list_documents(self) -> list[dict]:
        """Return one summary row per distinct document currently indexed."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT document_id, source, COUNT(*) AS chunk_count,
                       COUNT(DISTINCT page) AS page_count
                FROM chunks
                GROUP BY document_id, source
                ORDER BY MIN(created_at)
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> list[int]:
        """Permanently remove every chunk belonging to document_id from both
        the chunks table and the FTS index, in one transaction.

        Returns the chunk_ids that were removed (empty list if the document
        had no chunks). The caller is responsible for also removing these
        same ids from the FAISS index - this method only owns the SQLite side.
        """
        with self._connection(create=True) as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            chunk_ids = [row["chunk_id"] for row in rows]

            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
                conn.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )

            return chunk_ids

    def filter_chunk_ids_by_document(self, chunk_ids: list[int], document_id: str) -> set[int]:
        """Return the subset of chunk_ids that belong to document_id.

        Used to scope hybrid retrieval candidates to one document: FAISS/FTS5
        don't know about document_id, so this post-filters their candidate
        chunk_ids against the SQLite metadata relation that does.
        """
        if not chunk_ids:
            return set()

        with self._connection() as conn:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT chunk_id FROM chunks "
                f"WHERE document_id = ? AND chunk_id IN ({placeholders})",
                [document_id, *chunk_ids],
            ).fetchall()

        return {row["chunk_id"] for row in rows}

    def document_ids_for_chunks(self, chunk_ids: list[int]) -> dict[int, str]:
        """Map each chunk_id to the document it belongs to.

        Used by retrieve_hybrid's document-balanced allocation, which has to
        know which document each already-fused candidate came from before it
        decides which ones to keep. SQLite metadata stays the sole authority
        for that relation, exactly as the filter_* methods above.
        """
        if not chunk_ids:
            return {}

        with self._connection() as conn:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT chunk_id, document_id FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()

        return {row["chunk_id"]: row["document_id"] for row in rows}

    def filter_chunk_ids_by_documents(self, chunk_ids: list[int], document_ids: list[str]) -> set[int]:
        """Same as filter_chunk_ids_by_document, but scopes to a SET of
        documents (multi-document analysis mode) instead of exactly one -
        used by retrieve_hybrid() whenever document_id is given as a list.
        SQLite metadata remains the sole authority for which chunk_ids
        belong to which documents; FAISS/FTS5 candidates are always
        post-filtered against it before RRF fusion, exactly as the
        single-document path already does.
        """
        if not chunk_ids or not document_ids:
            return set()

        with self._connection() as conn:
            chunk_placeholders = ",".join("?" * len(chunk_ids))
            doc_placeholders = ",".join("?" * len(document_ids))
            rows = conn.execute(
                f"SELECT chunk_id FROM chunks "
                f"WHERE document_id IN ({doc_placeholders}) AND chunk_id IN ({chunk_placeholders})",
                [*document_ids, *chunk_ids],
            ).fetchall()

        return {row["chunk_id"] for row in rows}