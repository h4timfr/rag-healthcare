"""Keyword retrieval over chunk text using SQLite FTS5."""

import re
import sqlite3

from ..exceptions import DatabaseNotFoundError
from ..text_match import stem

# Shortest stem used as a prefix by expand_stems (see _sanitize_query):
# below this a prefix starts matching unrelated words.
MIN_STEM_PREFIX_CHARS = 5


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
    def _sanitize_query(cls, query: str, matches_nothing=None) -> str:
        """Convert natural-language text into a focused FTS5 OR query.

        `matches_nothing` (default None - the research benchmark and every
        other caller keep the original query exactly) is a predicate on a
        quoted term. For a term it reports as matching no indexed text, the
        term's stem is ORed in as a prefix query. The FTS5 porter tokenizer
        splits some word families: "allergic" stems to "allerg" but the
        document's "Allergy" to "allergi", so "What is the patient allergic
        to?" matched no allergy text at all, and its only matching term,
        "patient", then gave every chunk EXCEPT the allergy passage a
        keyword-rank bonus in fusion, pushing it (dense rank 1) out of the
        top five. "allerg*" matches the whole family.

        Only a term with no exact match is expanded. Expanding every term
        was measured to reorder production rankings on the research
        benchmark (3 questions worse, 1 better): for a term porter already
        handles, the prefix matches the same chunks, and the duplicate OR
        term double-counts that term in BM25. A query whose terms all match
        is therefore left exactly as it was.
        """
        terms = re.findall(r"[A-Za-z0-9_]+", query.lower())

        terms = [
            term
            for term in terms
            if term not in cls.STOPWORDS
        ]

        if not terms:
            return ""

        parts = [f'"{term}"' for term in terms]
        if matches_nothing is not None:
            for term in terms:
                prefix = stem(term)
                if (
                    prefix != term
                    and len(prefix) >= MIN_STEM_PREFIX_CHARS
                    and prefix.isalpha()
                    and f"{prefix}*" not in parts
                    and matches_nothing(f'"{term}"')
                ):
                    parts.append(f"{prefix}*")

        return " OR ".join(parts)

    def search(self, query: str, top_k: int = 5, expand_stems: bool = False) -> list[dict]:
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
            if expand_stems:

                def matches_nothing(fts_term: str) -> bool:
                    row = conn.execute(
                        "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                        (fts_term,),
                    ).fetchone()
                    return row is None

                fts_query = self._sanitize_query(query, matches_nothing=matches_nothing)

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