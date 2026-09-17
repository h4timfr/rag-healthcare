"""Ensures the project root is on sys.path so tests can `import src...`,
and guards the production index against the test suite."""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402  (after the sys.path insert above)

_PRODUCTION_DB = Path(config.DB_PATH).resolve()
_PRODUCTION_INDEX = Path(config.FAISS_INDEX_PATH).resolve()


def _document_fingerprint() -> tuple | None:
    """(document_id, chunk_count) for every document in the production
    SQLite index, or None if there is no index (clean checkout, or CI).

    Per-document counts rather than a single total: a test that deleted one
    document and added another, or that added chunks to a document already
    there, leaves the total unchanged and would otherwise pass unnoticed.
    Read-only URI, so the check itself can never be the thing that writes.
    """
    if not _PRODUCTION_DB.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{_PRODUCTION_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return tuple(
            conn.execute(
                "SELECT document_id, COUNT(*) FROM chunks "
                "GROUP BY document_id ORDER BY document_id"
            )
        )
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _index_fingerprint() -> tuple | None:
    """(size, mtime_ns) of the production FAISS index, or None if absent.

    A stat rather than reading the index: it needs no faiss import, couples
    to nothing in the FAISS API, costs ~0.03ms, and detects any write.
    Hashing the bytes would also catch a same-size same-mtime rewrite, but
    its cost grows with the index, which the suite pays on every test.

    This is what catches the asymmetric-patch mistake: a fixture that
    redirects config.DB_PATH to tmp_path but forgets
    config.FAISS_INDEX_PATH writes vectors straight into the real index
    while its rows go somewhere harmless. SQLite alone sees nothing wrong.
    """
    if not _PRODUCTION_INDEX.exists():
        return None
    try:
        stat = _PRODUCTION_INDEX.stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def describe_drift(before: tuple, after: tuple) -> str | None:
    """Plain description of how the production index changed, or None if it
    did not. Pure so it can be tested without touching anything real.

    `before`/`after` are (documents, index) pairs as returned by the two
    fingerprint helpers above.
    """
    if before == after:
        return None

    before_docs, before_index = before
    after_docs, after_index = after
    problems = []

    if before_docs != after_docs:
        old = dict(before_docs or ())
        new = dict(after_docs or ())
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(
            doc for doc in set(old) & set(new) if old[doc] != new[doc]
        )
        if added:
            problems.append(f"added {added}")
        if removed:
            problems.append(f"removed {removed}")
        for doc in changed:
            problems.append(f"{doc} chunk count {old[doc]} -> {new[doc]}")

    if before_index != after_index:
        problems.append("the FAISS index file was written")

    return "; ".join(problems) if problems else "the index changed"


@pytest.fixture(autouse=True)
def production_index_is_off_limits():
    """Fail any test that mutates the real data/metadata.db or data/faiss.index.

    Tests redirect config.DB_PATH and config.FAISS_INDEX_PATH to a tmp_path,
    but those redirects are easy to lose by accident - a monkeypatch.undo()
    intended for one patch reverts the fixture's redirects too, and the next
    ingest lands in the user's real library. That happened: three stray
    "doc.pdf" documents appeared in the production index and were only
    noticed by looking at the running app.

    Snapshotting both stores turns an invisible side effect into a failing
    test that names the culprit and says what it did.
    """
    before = (_document_fingerprint(), _index_fingerprint())
    yield
    after = (_document_fingerprint(), _index_fingerprint())

    drift = describe_drift(before, after)
    assert drift is None, (
        f"this test mutated the PRODUCTION index: {drift}.\n"
        f"  SQLite: {_PRODUCTION_DB}\n"
        f"  FAISS:  {_PRODUCTION_INDEX}\n"
        "Redirect BOTH config.DB_PATH and config.FAISS_INDEX_PATH to "
        "tmp_path - redirecting only one still writes to the other. Note "
        "that monkeypatch.undo() reverts those redirects too.\n"
        "If the Streamlit app was running and you uploaded or deleted a "
        "document while the suite ran, that also trips this check, and the "
        "test named here is innocent."
    )
