import numpy as np
import pytest

from src.exceptions import IndexNotFoundError
from src.retrieval.faiss_index import FaissIndex


def _unit_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_load_missing_index_raises(tmp_path):
    index = FaissIndex(dim=8, index_path=str(tmp_path / "missing.index"))
    with pytest.raises(IndexNotFoundError):
        index.load()


def test_add_search_save_load_roundtrip(tmp_path):
    dim = 8
    index_path = str(tmp_path / "test.index")

    index = FaissIndex(dim=dim, index_path=index_path)
    index.load_or_create()

    vectors = _unit_vectors(5, dim)
    ids = np.array([10, 11, 12, 13, 14], dtype="int64")
    index.add(vectors, ids)
    index.save()

    reloaded = FaissIndex(dim=dim, index_path=index_path)
    reloaded.load()

    scores, result_ids = reloaded.search(vectors[0], top_k=1)
    assert result_ids[0] == 10
    assert scores[0] == pytest.approx(1.0, abs=1e-4)


def test_search_returns_neg_one_for_missing_matches(tmp_path):
    dim = 8
    index = FaissIndex(dim=dim, index_path=str(tmp_path / "test.index"))
    index.load_or_create()
    vectors = _unit_vectors(2, dim)
    index.add(vectors, np.array([1, 2], dtype="int64"))

    scores, ids = index.search(vectors[0], top_k=5)
    assert list(ids[:2]) == [1, 2]
    assert all(i in (-1, 1, 2) for i in ids)


def test_remove_ids_deletes_vectors_and_persists_after_save(tmp_path):
    dim = 8
    index_path = str(tmp_path / "test.index")
    index = FaissIndex(dim=dim, index_path=index_path)
    index.load_or_create()

    vectors = _unit_vectors(4, dim)
    ids = np.array([1, 2, 3, 4], dtype="int64")
    index.add(vectors, ids)
    assert index.index.ntotal == 4

    removed = index.remove_ids([2, 3])
    assert removed == 2
    assert index.index.ntotal == 2
    index.save()

    reloaded = FaissIndex(dim=dim, index_path=index_path)
    reloaded.load()
    assert reloaded.index.ntotal == 2

    scores, result_ids = reloaded.search(vectors[0], top_k=4)
    remaining_ids = {int(i) for i in result_ids if i != -1}
    assert remaining_ids == {1, 4}


def test_remove_ids_with_empty_list_is_a_noop(tmp_path):
    dim = 8
    index = FaissIndex(dim=dim, index_path=str(tmp_path / "test.index"))
    index.load_or_create()
    vectors = _unit_vectors(2, dim)
    index.add(vectors, np.array([1, 2], dtype="int64"))

    removed = index.remove_ids([])

    assert removed == 0
    assert index.index.ntotal == 2
