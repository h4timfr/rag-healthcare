"""Tests for the production-index guard in conftest.py.

The guard exists because of a real incident: three stray "doc.pdf"
documents appeared in the user's live library after a test lost its
config.DB_PATH redirect to monkeypatch.undo(). It was found by looking at
the running app, not by the suite.

The comparison logic is a pure function so it can be exercised here without
mutating anything real - these tests never touch the production index, which
is the whole point of the thing they are testing.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    _document_fingerprint,
    _index_fingerprint,
    describe_drift,
)


def snapshot(documents, index=(100, 1)):
    """Build a (documents, index) snapshot in the shape the guard uses."""
    return (tuple(documents), index)


# --- what the guard must catch ------------------------------------------------


def test_no_change_is_no_drift():
    before = snapshot([("a", 9), ("b", 11)])
    assert describe_drift(before, before) is None


def test_added_document_is_reported():
    """The original incident: a test ingesting into the real library."""
    before = snapshot([("a", 9)])
    after = snapshot([("a", 9), ("leaked", 1)])

    drift = describe_drift(before, after)

    assert drift is not None
    assert "added ['leaked']" in drift


def test_removed_document_is_reported():
    before = snapshot([("a", 9), ("b", 11)])
    after = snapshot([("a", 9)])

    assert "removed ['b']" in describe_drift(before, after)


def test_swap_of_equal_size_is_reported():
    """The gap that motivated moving past a bare count: deleting one
    document and adding another leaves the total unchanged."""
    before = snapshot([("a", 9), ("b", 11)])
    after = snapshot([("a", 9), ("c", 11)])

    drift = describe_drift(before, after)

    assert "added ['c']" in drift
    assert "removed ['b']" in drift


def test_chunk_count_change_within_a_document_is_reported():
    """Also invisible to a document count: re-ingesting the same document
    with different chunking, or appending chunks to it."""
    before = snapshot([("a", 9)])
    after = snapshot([("a", 14)])

    assert "a chunk count 9 -> 14" in describe_drift(before, after)


def test_faiss_write_is_reported_even_when_sqlite_is_untouched():
    """The asymmetric-patch mistake: a fixture redirects config.DB_PATH but
    not config.FAISS_INDEX_PATH, so the rows land in tmp_path and the
    vectors land in the user's real index. SQLite alone sees nothing."""
    documents = [("a", 9)]
    before = snapshot(documents, index=(47954, 111))
    after = snapshot(documents, index=(52000, 222))

    assert "the FAISS index file was written" == describe_drift(before, after)


def test_both_stores_changing_reports_both():
    before = snapshot([("a", 9)], index=(100, 1))
    after = snapshot([("a", 9), ("b", 2)], index=(200, 2))

    drift = describe_drift(before, after)

    assert "added ['b']" in drift
    assert "the FAISS index file was written" in drift


# --- what the guard must NOT do -----------------------------------------------


def test_absent_index_is_not_drift():
    """A clean checkout or CI has no production index at all. That must not
    fail every test in the suite."""
    assert describe_drift((None, None), (None, None)) is None


def test_index_appearing_during_a_test_is_drift():
    """Creating the production index from nothing is exactly the leak this
    guards against."""
    assert describe_drift((None, None), snapshot([("a", 1)])) is not None


def test_fingerprints_do_not_write_to_the_production_index():
    """The check must not be the thing that mutates. Both helpers are
    read-only, so calling them repeatedly changes nothing."""
    first = (_document_fingerprint(), _index_fingerprint())
    for _ in range(3):
        assert (_document_fingerprint(), _index_fingerprint()) == first


def test_document_fingerprint_uses_a_read_only_connection(monkeypatch):
    """A writable connection here could itself corrupt the thing it is
    protecting - and would quietly create the file on a clean checkout."""
    opened = {}

    real_connect = sqlite3.connect

    def spy(dsn, *args, **kwargs):
        opened["dsn"] = dsn
        opened["uri"] = kwargs.get("uri")
        return real_connect(dsn, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    _document_fingerprint()

    if opened:  # skipped silently when there is no production index
        assert opened["uri"] is True
        assert "mode=ro" in opened["dsn"]


def test_fingerprint_shape_matches_what_describe_drift_expects():
    """The helpers and the comparison must not drift apart: a fingerprint
    has to be usable directly as half of a snapshot."""
    documents = _document_fingerprint()
    index = _index_fingerprint()

    assert describe_drift((documents, index), (documents, index)) is None

    if documents:
        assert all(len(row) == 2 for row in documents)
        assert all(isinstance(count, int) for _, count in documents)
    if index:
        assert len(index) == 2


# --- the guard is actually installed ------------------------------------------


def test_guard_fixture_is_autouse_for_every_test(request):
    """If it stops being autouse it protects nothing, silently."""
    assert "production_index_is_off_limits" in request.fixturenames


@pytest.mark.parametrize("documents", [[("a", 1)], [("a", 1), ("b", 2)], []])
def test_drift_description_is_never_empty_when_something_changed(documents):
    before = snapshot(documents)
    after = snapshot(list(documents) + [("zz", 3)])

    drift = describe_drift(before, after)

    assert drift and drift.strip()


# --- proof that the guard actually fires --------------------------------------


GUARD_PROOF_CONFTEST = '''
import importlib.util
import sqlite3
from pathlib import Path

import pytest

# Load the project's real conftest BY PATH under a distinct module name:
# this directory has its own conftest.py, so a plain "import conftest"
# resolves to that one instead.
_spec = importlib.util.spec_from_file_location(
    "project_conftest_under_test", "{project}/conftest.py"
)
_project_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_project_conftest)
describe_drift = _project_conftest.describe_drift

FAKE_DB = Path(r"{db}")
FAKE_INDEX = Path(r"{index}")


def _documents():
    if not FAKE_DB.exists():
        return None
    conn = sqlite3.connect(f"file:{{FAKE_DB}}?mode=ro", uri=True)
    try:
        return tuple(conn.execute(
            "SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id ORDER BY document_id"))
    finally:
        conn.close()


def _index():
    if not FAKE_INDEX.exists():
        return None
    s = FAKE_INDEX.stat()
    return (s.st_size, s.st_mtime_ns)


@pytest.fixture(autouse=True)
def guard():
    before = (_documents(), _index())
    yield
    drift = describe_drift(before, (_documents(), _index()))
    assert drift is None, f"this test mutated the PRODUCTION index: {{drift}}."
'''

GUARD_PROOF_TESTS = '''
import sqlite3
from pathlib import Path

DB = Path(r"{db}")
INDEX = Path(r"{index}")


def test_innocent_test_passes():
    assert 1 + 1 == 2


def test_leaking_test_is_caught():
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO chunks VALUES (99, 'leaked', 'leak.pdf', 1, 'text')")
    conn.commit()
    conn.close()


def test_faiss_only_leak_is_caught():
    INDEX.write_bytes(b"x" * 4096)
'''


def test_guard_fails_the_test_that_leaks(tmp_path):
    """End to end, in a subprocess against a throwaway stand-in for the
    production index - the real one is never touched.

    An assertion that the guard passes proves nothing; this proves it fails
    the right test, and only that test. Both leak shapes are covered: a row
    written to SQLite, and a write to the index file with SQLite untouched.
    """
    project = Path(__file__).resolve().parents[1].as_posix()
    db = tmp_path / "metadata.db"
    index = tmp_path / "faiss.index"

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, document_id TEXT, "
        "source TEXT, page INTEGER, text TEXT)"
    )
    conn.execute("INSERT INTO chunks VALUES (1, 'kept', 'a.pdf', 1, 'text')")
    conn.commit()
    conn.close()
    index.write_bytes(b"x" * 2048)

    (tmp_path / "conftest.py").write_text(
        GUARD_PROOF_CONFTEST.format(project=project, db=db, index=index), encoding="utf-8"
    )
    (tmp_path / "test_leak.py").write_text(
        GUARD_PROOF_TESTS.format(db=db, index=index), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    # The guard asserts in fixture TEARDOWN, which pytest reports as an
    # error rather than a failure: each test body itself completes, and
    # then the guard objects to what it left behind.
    assert "3 passed, 2 errors" in result.stdout
    assert "test_leaking_test_is_caught" in result.stdout
    assert "test_faiss_only_leak_is_caught" in result.stdout
    assert "mutated the PRODUCTION index" in result.stdout
    assert "added ['leaked']" in result.stdout
    assert "FAISS index file was written" in result.stdout
