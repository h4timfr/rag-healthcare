"""Concurrent writes to the shared FAISS index file.

Every index write is load -> mutate -> save of the whole file. Streamlit runs
each browser session as a thread of one process, so two overlapping writers
used to lose an update: both loaded the same old index and the last save
silently dropped the other's vectors, leaving that document's chunks in SQLite
with no dense vectors (and un-repairable, since dedup skips any re-upload).

These tests force the overlap with a barrier inside the embedding step, which
is exactly where a real upload spends most of its time.
"""

import sqlite3
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import pytest

import app
import scripts.ingest as ingest_script
from src import config
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text)
    doc.save(str(path))
    doc.close()


def _unit_vectors(n):
    rng = np.random.default_rng(n)
    v = rng.standard_normal((n, config.EMBEDDING_DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sqlite_ids():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        return {row[0] for row in conn.execute("SELECT chunk_id FROM chunks")}
    finally:
        conn.close()


def _faiss_ids():
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    return [int(x) for x in faiss.vector_to_array(index.id_map)]


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    return tmp_path


def _embed_behind_barrier(monkeypatch, parties):
    # A timeout, not a bare wait: if embedding were ever serialized behind the
    # write lock, the barrier could never fill and this raises instead of
    # hanging the suite.
    barrier = threading.Barrier(parties, timeout=15)

    def fake_embed(texts):
        barrier.wait()
        return _unit_vectors(len(texts))

    monkeypatch.setattr(ingest_script, "embed_texts", fake_embed)


def _run_concurrently(targets):
    errors = []
    results = [None] * len(targets)

    def wrap(i, fn):
        try:
            results[i] = fn()
        except Exception as e:  # noqa: BLE001 - surfaced via the assertion below
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(i, fn)) for i, fn in enumerate(targets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return results


def test_concurrent_ingestion_of_different_documents_keeps_every_vector(isolated_index, monkeypatch):
    pdf_a, pdf_b = isolated_index / "a.pdf", isolated_index / "b.pdf"
    _make_pdf(pdf_a, "alpha " * 400)
    _make_pdf(pdf_b, "bravo " * 400)
    _embed_behind_barrier(monkeypatch, parties=2)

    results = _run_concurrently([
        lambda: ingest_script.ingest_pdf(str(pdf_a)),
        lambda: ingest_script.ingest_pdf(str(pdf_b)),
    ])

    assert all(r["skipped"] is False for r in results)
    faiss_ids = _faiss_ids()
    assert len(faiss_ids) == len(set(faiss_ids))
    assert set(faiss_ids) == _sqlite_ids()


def test_concurrent_ingestion_of_the_same_file_indexes_it_once(isolated_index, monkeypatch):
    pdf = isolated_index / "doc.pdf"
    _make_pdf(pdf, "charlie " * 400)
    _embed_behind_barrier(monkeypatch, parties=2)

    results = _run_concurrently([
        lambda: ingest_script.ingest_pdf(str(pdf)),
        lambda: ingest_script.ingest_pdf(str(pdf)),
    ])

    assert sorted(r["skipped"] for r in results) == [False, True]
    faiss_ids = _faiss_ids()
    assert len(faiss_ids) == len(set(faiss_ids)) == len(_sqlite_ids())


def test_delete_during_ingestion_leaves_indexes_consistent(isolated_index, monkeypatch):
    pdf_keep, pdf_new = isolated_index / "keep.pdf", isolated_index / "new.pdf"
    _make_pdf(pdf_keep, "delta " * 400)
    _make_pdf(pdf_new, "echo " * 400)
    monkeypatch.setattr(ingest_script, "embed_texts", lambda texts: _unit_vectors(len(texts)))
    doomed = ingest_script.ingest_pdf(str(pdf_keep))["document_id"]

    # Hold the ingestion INSIDE its load..save window (insert_chunks runs
    # after the index is loaded and before it is saved) long enough for the
    # delete to run. Unsynchronized, the delete completes in that window and
    # the ingestion then saves the index it loaded before the delete -
    # resurrecting the deleted document's vectors as orphans.
    in_window = threading.Event()
    real_insert = MetadataStore.insert_chunks

    def insert_then_pause(self, *args, **kwargs):
        ids = real_insert(self, *args, **kwargs)
        in_window.set()
        time.sleep(0.5)
        return ids

    monkeypatch.setattr(MetadataStore, "insert_chunks", insert_then_pause)

    def delete_once_ingest_is_mid_write():
        assert in_window.wait(timeout=15)
        return app.delete_document(doomed)

    _run_concurrently([
        lambda: ingest_script.ingest_pdf(str(pdf_new)),
        delete_once_ingest_is_mid_write,
    ])

    assert set(_faiss_ids()) == _sqlite_ids()
    assert _sqlite_ids()  # the new document survived


def test_failed_save_leaves_the_existing_index_intact(isolated_index, monkeypatch):
    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()
    index.add(_unit_vectors(3), np.array([1, 2, 3], dtype="int64"))
    index.save()

    def exploding_write(idx, path):
        with open(path, "wb") as fh:
            fh.write(b"half-written")
        raise OSError("disk full")

    index.add(_unit_vectors(1), np.array([4], dtype="int64"))
    real_write = faiss.write_index
    # Not monkeypatch.undo(): that would also revert the fixture's redirection
    # of config.FAISS_INDEX_PATH, and the assertion below would then read the
    # real production index.
    monkeypatch.setattr(faiss, "write_index", exploding_write)
    with pytest.raises(OSError):
        index.save()
    monkeypatch.setattr(faiss, "write_index", real_write)

    assert sorted(_faiss_ids()) == [1, 2, 3]
    leftovers = [p.name for p in isolated_index.iterdir() if ".tmp-" in p.name]
    assert leftovers == []


# --- cross-process (real subprocesses, not threads) ----------------------------
#
# INDEX_WRITE_LOCK is a threading.RLock: it serializes threads of ONE process
# (what the tests above exercise) but gives zero protection between two
# separate processes - e.g. the ingest CLI run while the Streamlit app is
# also up. Verified by removing cross_process_index_lock()'s OS file lock and
# re-running this test: it reproduces the identical lost-update, confirming
# the OS lock (not just INDEX_WRITE_LOCK) is what these two processes need.

_CROSS_PROCESS_WORKER = """
import sys, time
sys.path.insert(0, {project_root!r})
from src import config
config.DB_PATH = {db_path!r}
config.FAISS_INDEX_PATH = {faiss_path!r}
import numpy as np
import scripts.ingest as ingest_module

def fake_embed(texts):
    time.sleep(0.6)  # force temporal overlap between the two processes
    rng = np.random.default_rng(len(texts) + int(sys.argv[2]))
    v = rng.standard_normal((len(texts), config.EMBEDDING_DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)

ingest_module.embed_texts = fake_embed
ingest_module.ingest_pdf(sys.argv[1])
"""


def test_cross_process_ingestion_is_serialized(isolated_index):
    """Two real OS processes ingesting different documents at the same time
    must not lose either one's vectors - reproduced without
    cross_process_index_lock's file lock, fixed with it."""
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[1])
    worker = isolated_index / "_worker.py"
    worker.write_text(
        _CROSS_PROCESS_WORKER.format(
            project_root=root, db_path=str(config.DB_PATH), faiss_path=str(config.FAISS_INDEX_PATH)
        ),
        encoding="utf-8",
    )
    pdf_a, pdf_b = isolated_index / "a.pdf", isolated_index / "b.pdf"
    _make_pdf(pdf_a, "alpha " * 400)
    _make_pdf(pdf_b, "bravo " * 400)

    procs = [
        subprocess.Popen([sys.executable, str(worker), str(pdf_a), "1"]),
        subprocess.Popen([sys.executable, str(worker), str(pdf_b), "2"]),
    ]
    for p in procs:
        assert p.wait(timeout=30) == 0

    faiss_ids = _faiss_ids()
    assert len(faiss_ids) == len(set(faiss_ids))
    assert set(faiss_ids) == _sqlite_ids()
    assert len(_sqlite_ids()) == 6  # both documents' 3 chunks each, nothing lost


_LOCK_HOLDER = """
import sys, time
sys.path.insert(0, {project_root!r})
from src.retrieval.faiss_index import cross_process_index_lock
with cross_process_index_lock({faiss_path!r}):
    print("held", flush=True)
    time.sleep({hold_seconds})
"""


def test_cross_process_lock_waits_out_a_long_holder_instead_of_raising(isolated_index):
    """Windows' msvcrt LK_LOCK gives up after ~10 one-second retries and
    raises OSError(EDEADLOCK) - measured at 9.1s - so a writer queued behind
    a long critical section (a startup rebuild re-embeds every chunk under
    the lock) failed instead of blocking as documented."""
    import subprocess
    import sys

    from src.retrieval.faiss_index import cross_process_index_lock

    hold_seconds = 12
    root = str(Path(__file__).resolve().parents[1])
    holder_script = isolated_index / "_holder.py"
    holder_script.write_text(
        _LOCK_HOLDER.format(
            project_root=root, faiss_path=str(config.FAISS_INDEX_PATH), hold_seconds=hold_seconds
        ),
        encoding="utf-8",
    )
    holder = subprocess.Popen(
        [sys.executable, str(holder_script)], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        start = time.perf_counter()
        with cross_process_index_lock(config.FAISS_INDEX_PATH):
            waited = time.perf_counter() - start
        assert waited > hold_seconds - 3  # it genuinely waited for the holder
    finally:
        assert holder.wait(timeout=30) == 0
