import pytest

from src.exceptions import DatabaseNotFoundError
from src.retrieval.metadata_store import MetadataStore


def test_query_before_initialize_raises(tmp_path):
    store = MetadataStore(str(tmp_path / "missing.db"))
    with pytest.raises(DatabaseNotFoundError):
        store.get_chunks([1])


def test_insert_and_retrieve_roundtrip(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()

    chunk_id = store.insert_chunk(
        document_id="doc123", source="file.pdf", page=3, text="hello world"
    )
    assert isinstance(chunk_id, int)

    rows = store.get_chunks([chunk_id])
    assert rows[chunk_id]["document_id"] == "doc123"
    assert rows[chunk_id]["source"] == "file.pdf"
    assert rows[chunk_id]["page"] == 3
    assert rows[chunk_id]["text"] == "hello world"


def test_get_chunks_empty_list_returns_empty_dict(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    assert store.get_chunks([]) == {}


def test_count(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="d", source="s", page=1, text="t")
    store.insert_chunk(document_id="d", source="s", page=2, text="t2")
    assert store.count() == 2


def test_document_chunk_count_zero_for_unknown_document(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    assert store.document_chunk_count("nonexistent") == 0


def test_document_chunk_count_reflects_inserted_rows(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="t1")
    store.insert_chunk(document_id="docA", source="a.pdf", page=2, text="t2")
    store.insert_chunk(document_id="docB", source="b.pdf", page=1, text="t3")

    assert store.document_chunk_count("docA") == 2
    assert store.document_chunk_count("docB") == 1


def test_list_documents_groups_by_document_id(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="t1")
    store.insert_chunk(document_id="docA", source="a.pdf", page=2, text="t2")
    store.insert_chunk(document_id="docB", source="b.pdf", page=1, text="t3")

    docs = {d["document_id"]: d for d in store.list_documents()}
    assert docs["docA"]["source"] == "a.pdf"
    assert docs["docA"]["chunk_count"] == 2
    assert docs["docB"]["chunk_count"] == 1


def test_list_documents_empty_when_no_chunks(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    assert store.list_documents() == []


def test_delete_document_removes_only_its_own_chunks(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")
    store.insert_chunk(document_id="docA", source="a.pdf", page=2, text="beta")
    store.insert_chunk(document_id="docB", source="b.pdf", page=1, text="gamma")

    removed = store.delete_document("docA")

    assert len(removed) == 2
    assert store.document_chunk_count("docA") == 0
    assert store.document_chunk_count("docB") == 1
    assert store.count() == 1


def test_delete_document_removes_fts_rows(tmp_path):
    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="unique_keyword_term")

    store.delete_document("docA")

    keyword_index = KeywordIndex(db_path)
    assert keyword_index.search("unique_keyword_term") == []


def test_delete_document_unknown_id_is_a_noop(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")

    removed = store.delete_document("does-not-exist")

    assert removed == []
    assert store.count() == 1
