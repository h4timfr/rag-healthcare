"""Keyword retrieval over chunk text using SQLite FTS5."""

import re
import sqlite3

from ..exceptions import DatabaseNotFoundError


class KeywordIndex:
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(self, db_path: str):
        self.db_path = db_path

    @classmethod
    def _sanitize_query(cls, query: str) -> str:
        """Convert natural-language text into a focused FTS5 OR query."""
        terms = re.findall(r"[A-Za-z0-9_]+", query.lower())

        terms = [
            term
            for term in terms
            if term not in cls.STOPWORDS
        ]

        if not terms:
            return ""

        return " OR ".join(
            f'"{term}"'
            for term in terms
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []

        fts_query = self._sanitize_query(query)

        if not fts_query:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise DatabaseNotFoundError(
                f"Could not open SQLite database at "
                f"'{self.db_path}': {exc}"
            ) from exc

        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                """
                SELECT
                    chunk_id,
                    bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()

            return [
                {
                    "chunk_id": int(row["chunk_id"]),
                    "score": float(row["score"]),
                }
                for row in rows
            ]
        finally:
            conn.close()