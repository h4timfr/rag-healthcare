"""src/retrieval/index_consistency.py: detecting and repairing every way the
SQLite metadata store and the FAISS index can drift apart. Each inconsistent
state below is constructed directly (not just "hope a race reproduces it")
so the check/repair logic itself is what's under test.
"""

import sqlite3

import faiss
import numpy as np
import pytest

from src import config
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.index_consistency import check_consistency, repair
from src.retrieval.metadata_store import MetadataStore


@pytest.fixture
def store_paths(tmp_path, monkeypatch):
    db_path = str(tmp_path / "metadata.db")
    faiss_path = str(tmp_path / "faiss.index")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", faiss_path)
    return db_path, faiss_path


def _unit_vectors(n, seed=0, dim=None):
    dim = dim or config.EMBEDDING_DIM
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _seed_sqlite(db_path, n, document_id="doc1"):
    store = MetadataStore(db_path)
    store.initialize()
    return store.insert_chunks(document_id, "a.pdf", [(1, f"chunk {i} text") for i in range(n)])


def _seed_faiss(faiss_path, ids, seed=0, dim=None):
    index = FaissIndex(dim or config.EMBEDDING_DIM, faiss_path)
    index.load_or_create()
    index.add(_unit_vectors(len(ids), seed=seed, dim=dim), np.array(ids, dtype="int64"))
    index.save()
    return index


def _faiss_id_set(faiss_path):
    # `index` MUST be a named local, not chained inline as
    # faiss.read_index(path).id_map - the SWIG-wrapped id_map array is a
    # view into memory owned by the index object, and an unbound temporary
    # can be garbage-collected before vector_to_array reads it, silently
    # producing an empty/garbage array. (Found by this exact bug: this
    # function briefly did the unsafe chained version.)
    index = faiss.read_index(faiss_path)
    return set(int(x) for x in faiss.vector_to_array(index.id_map))


def _sqlite_id_set(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}
    finally:
        conn.close()


# --- empty states ----------------------------------------------------------------


def test_no_database_and_no_index_is_reported_consistent(store_paths):
    report = check_consistency()
    assert report.is_consistent
    assert report.total_sqlite_chunks == 0
    assert report.total_faiss_vectors == 0


def test_database_with_no_chunks_table_yet_is_consistent(store_paths):
    db_path, _ = store_paths
    # initialize() creates the schema; simulate an even earlier state - a
    # database FILE that exists but predates the schema (e.g. created by an
    # unrelated process, or a future migration in progress).
    sqlite3.connect(db_path).close()
    report = check_consistency()
    assert report.is_consistent


def test_matching_database_and_index_is_consistent(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids)

    report = check_consistency()

    assert report.is_consistent
    assert report.total_sqlite_chunks == 3
    assert report.total_faiss_vectors == 3


# --- dense-invisible chunks (the known lost-update failure) ---------------------


def test_detects_chunks_present_in_sqlite_but_missing_from_faiss(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 5)
    _seed_faiss(faiss_path, ids[2:])  # first two never made it into FAISS

    report = check_consistency()

    assert not report.is_consistent
    assert report.dense_invisible_chunk_ids == ids[:2]
    assert report.orphan_faiss_ids == []


def test_repair_re_embeds_dense_invisible_chunks_without_touching_the_rest(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 5)
    _seed_faiss(faiss_path, ids[2:])

    repaired = repair()

    assert repaired.is_consistent
    assert _faiss_id_set(faiss_path) == set(ids)
    # The chunks that were already fine were never re-embedded/re-added -
    # their vectors are untouched (same as originally seeded).
    idx = faiss.read_index(faiss_path)
    id_list = [int(x) for x in faiss.vector_to_array(idx.id_map)]
    flat = faiss.downcast_index(idx.index)
    xb = faiss.rev_swig_ptr(flat.get_xb(), flat.ntotal * flat.d).reshape(flat.ntotal, flat.d)
    original = {cid: v for cid, v in zip(ids[2:], _unit_vectors(3, seed=0))}
    for cid, vec in zip(id_list, xb):
        if cid in original:
            assert np.allclose(vec, original[cid])


def test_repaired_chunk_is_actually_retrievable_by_dense_search(store_paths):
    """End-to-end: after repair, the previously dense-invisible chunk must
    be findable by its own (re-embedded) vector, not just present in the
    id map."""
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids[1:])  # chunk_id ids[0] is dense-invisible

    repair()

    index = FaissIndex(config.EMBEDDING_DIM, faiss_path)
    index.load()
    # Re-derive the exact vector repair() would have embedded and search
    # with it - it must come back as an exact top-1 match (score ~1.0).
    from src.embedding.embedder import embed_texts

    query_vector = embed_texts([f"chunk 0 text"])[0]
    scores, result_ids = index.search(query_vector, top_k=1)
    assert int(result_ids[0]) == ids[0]
    assert scores[0] == pytest.approx(1.0, abs=1e-4)


def test_dense_invisible_chunk_deleted_between_check_and_repair_is_not_an_error(store_paths):
    """A concurrent delete removing the very chunk repair() was about to fix
    must not raise - there is simply nothing left to repair for it."""
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids[1:])
    MetadataStore(db_path).delete_document("doc1")  # removes all 3 SQLite rows

    repaired = repair()

    assert repaired.dense_invisible_chunk_ids == []
    assert repaired.total_sqlite_chunks == 0


# --- orphan / duplicate FAISS ids -------------------------------------------------


def test_detects_vectors_present_in_faiss_but_missing_from_sqlite(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids + [999])  # 999 has no SQLite row

    report = check_consistency()

    assert not report.is_consistent
    assert report.orphan_faiss_ids == [999]
    assert report.dense_invisible_chunk_ids == []


def test_repair_removes_orphan_vectors_without_touching_sqlite(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids + [999])

    repaired = repair()

    assert repaired.is_consistent
    assert _faiss_id_set(faiss_path) == set(ids)
    assert _sqlite_id_set(db_path) == set(ids)  # untouched


def test_detects_and_repairs_duplicate_faiss_ids(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 2)
    index = FaissIndex(config.EMBEDDING_DIM, faiss_path)
    index.load_or_create()
    index.add(_unit_vectors(3), np.array([ids[0], ids[0], ids[1]], dtype="int64"))
    index.save()

    report = check_consistency()
    assert report.duplicate_faiss_ids == [ids[0]]

    repaired = repair()
    assert repaired.is_consistent
    assert _faiss_id_set(faiss_path) == set(ids)


# --- dimension mismatch and unnormalized vectors (report only, never auto-fixed) --


def test_detects_dimension_mismatch_and_does_not_attempt_repair(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 2)
    _seed_faiss(faiss_path, ids, dim=16)  # wrong dimension for config.EMBEDDING_DIM

    report = check_consistency()
    assert report.dimension_mismatch
    assert report.actual_dim == 16
    assert report.expected_dim == config.EMBEDDING_DIM
    assert not report.is_consistent

    repaired = repair()
    assert repaired.dimension_mismatch  # unchanged - not auto-repaired
    assert repaired.actual_dim == 16


def test_dimension_mismatch_skips_the_normalization_check_rather_than_reading_garbage(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 2)
    _seed_faiss(faiss_path, ids, dim=16)

    report = check_consistency()
    assert report.unnormalized_chunk_ids == []


def test_detects_unnormalized_vectors_and_does_not_attempt_repair(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 2)
    index = FaissIndex(config.EMBEDDING_DIM, faiss_path)
    index.load_or_create()
    vectors = _unit_vectors(2)
    vectors[0] *= 5.0  # break normalization for just the first one
    index.add(vectors, np.array(ids, dtype="int64"))
    index.save()

    report = check_consistency()
    assert report.unnormalized_chunk_ids == [ids[0]]
    # Unnormalized alone does not make retrieval unsafe to use - it degrades
    # (a non-cosine score for that one chunk), it does not break.
    assert report.is_consistent

    repaired = repair()
    assert repaired.unnormalized_chunk_ids == [ids[0]]  # unchanged - not auto-repaired


def test_normalization_check_can_be_skipped(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 2)
    vectors = _unit_vectors(2)
    vectors[0] *= 5.0
    index = FaissIndex(config.EMBEDDING_DIM, faiss_path)
    index.load_or_create()
    index.add(vectors, np.array(ids, dtype="int64"))
    index.save()

    report = check_consistency(verify_normalization=False)
    assert report.unnormalized_chunk_ids == []


# --- corrupted / partially-written index file ------------------------------------


def test_detects_a_corrupted_index_file_without_crashing(store_paths):
    db_path, faiss_path = store_paths
    _seed_sqlite(db_path, 2)
    with open(faiss_path, "wb") as f:
        f.write(b"not a real faiss index, or a truncated write")

    report = check_consistency()

    assert report.faiss_corrupted
    assert report.corruption_detail
    assert not report.is_consistent


def test_repair_rebuilds_a_corrupted_index_from_sqlite_text(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 4)
    with open(faiss_path, "wb") as f:
        f.write(b"garbage")

    repaired = repair()

    assert repaired.is_consistent
    assert not repaired.faiss_corrupted
    assert _faiss_id_set(faiss_path) == set(ids)


def test_rebuild_from_sqlite_produces_vectors_that_actually_retrieve_correctly(store_paths):
    db_path, faiss_path = store_paths
    _seed_sqlite(db_path, 3, document_id="doc1")
    with open(faiss_path, "wb") as f:
        f.write(b"garbage")

    repair()

    from app import answer_question

    # Real production entry point, real (unmocked) embedding + retrieval.
    result = answer_question("What does chunk 1 text say?", document_id="doc1")
    assert result["chunks"]


def test_repair_of_corrupted_index_with_no_sqlite_chunks_produces_an_empty_valid_index(store_paths):
    """With no SQLite rows, every vector the unreadable file could hold is
    an orphan by definition, so removing it loses nothing. Leaving it in
    place left the app permanently unable to ingest: every upload's
    load_or_create() re-raised the same read error."""
    db_path, faiss_path = store_paths
    MetadataStore(db_path).initialize()
    with open(faiss_path, "wb") as f:
        f.write(b"garbage")

    repaired = repair()

    assert repaired.is_consistent
    assert not repaired.faiss_corrupted
    assert repaired.total_sqlite_chunks == 0
    FaissIndex(config.EMBEDDING_DIM, faiss_path).load_or_create()  # the next ingest can proceed


def test_repair_of_corrupted_index_with_no_database_at_all_is_also_cleared(store_paths):
    _db_path, faiss_path = store_paths
    with open(faiss_path, "wb") as f:
        f.write(b"garbage")

    assert repair().is_consistent


# --- concurrent writer ------------------------------------------------------------


def test_repair_does_not_duplicate_a_chunk_a_concurrent_ingest_is_still_writing(store_paths, monkeypatch):
    """Ingest commits SQLite rows, then adds+saves their vectors, all under
    the index lock. A repair() whose check ran outside that lock saw the
    in-flight chunk as dense-invisible, waited for the lock, and then added
    it a second time after the ingest had already saved it - measured: one
    chunk_id ending up with two FAISS vectors."""
    import threading
    import time

    from src.retrieval import index_consistency
    from src.retrieval.faiss_index import cross_process_index_lock

    monkeypatch.setattr(index_consistency, "embed_texts", lambda texts: _unit_vectors(len(texts), seed=7))
    db_path, faiss_path = store_paths
    _seed_faiss(faiss_path, _seed_sqlite(db_path, 2))

    in_window, proceed = threading.Event(), threading.Event()

    def ingest():
        with cross_process_index_lock(faiss_path):
            new_ids = MetadataStore(db_path).insert_chunks("doc2", "b.pdf", [(1, "late chunk")])
            in_window.set()
            proceed.wait(10)
            index = FaissIndex(config.EMBEDDING_DIM, faiss_path)
            index.load_or_create()
            index.add(_unit_vectors(1, seed=3), np.array(new_ids, dtype="int64"))
            index.save()

    writer = threading.Thread(target=ingest)
    writer.start()
    assert in_window.wait(10)
    repairer = threading.Thread(target=repair)
    repairer.start()
    time.sleep(0.5)  # let repair() reach the lock while the ingest is mid-write
    proceed.set()
    writer.join(10)
    repairer.join(10)

    report = check_consistency()
    assert report.duplicate_faiss_ids == []
    assert report.is_consistent
    assert report.total_faiss_vectors == report.total_sqlite_chunks == 3


# --- startup integration -----------------------------------------------------------


def test_app_startup_check_repairs_before_the_first_query(store_paths, monkeypatch):
    """The exact scenario Phase 1 exists for: a process was killed between
    the SQLite commit and the FAISS save, leaving a real document
    dense-invisible. Verifies app.py's own startup wiring, not just the
    module directly."""
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3, document_id="realdoc")
    _seed_faiss(faiss_path, ids[1:])  # simulates the interrupted process

    import app

    app.ensure_index_consistency()

    assert _faiss_id_set(faiss_path) == set(ids)
    result = app.answer_question("What does chunk 0 text say?", document_id="realdoc")
    assert result["chunks"]


def test_app_startup_check_is_a_no_op_when_already_consistent(store_paths):
    db_path, faiss_path = store_paths
    ids = _seed_sqlite(db_path, 3)
    _seed_faiss(faiss_path, ids)
    mtime_before = __import__("os").path.getmtime(faiss_path)

    import app

    app.ensure_index_consistency()

    assert __import__("os").path.getmtime(faiss_path) == mtime_before  # never re-saved
