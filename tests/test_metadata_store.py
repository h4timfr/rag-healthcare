import sqlite3

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


# --- FTS5 Porter stemming ----------------------------------------------------
#
# Retrieval audit finding: a query for "diagnostic categories" (plural) never
# matched a chunk whose text only said "diagnostic category" (singular,
# taken verbatim from a real table header) - FTS5's default tokenizer does
# exact-token matching only. porter unicode61 (see SCHEMA) fixes this by
# stemming both the indexed text and the query to a common root.


@pytest.mark.parametrize(
    "indexed_text,query_word",
    [
        ("Diagnostic category", "categories"),
        ("laboratory test", "tests"),
        ("vaginal examination", "examinations"),
    ],
)
def test_plural_query_matches_singular_indexed_text(tmp_path, indexed_text, query_word):
    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    chunk_id = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text=indexed_text)

    keyword_index = KeywordIndex(db_path)
    results = keyword_index.search(query_word)

    assert {r["chunk_id"] for r in results} == {chunk_id}


@pytest.mark.parametrize(
    "indexed_text,query_word",
    [
        ("diagnostic categories", "category"),
        ("laboratory tests", "test"),
        ("vaginal examinations", "examination"),
    ],
)
def test_singular_query_matches_plural_indexed_text(tmp_path, indexed_text, query_word):
    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    chunk_id = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text=indexed_text)

    keyword_index = KeywordIndex(db_path)
    results = keyword_index.search(query_word)

    assert {r["chunk_id"] for r in results} == {chunk_id}


def test_stemming_does_not_cause_false_positive_matches(tmp_path):
    """Stemming must not turn the index into a fuzzy/substring matcher -
    only words that actually share a stem should match."""
    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="cardiovascular disease")

    keyword_index = KeywordIndex(db_path)
    assert keyword_index.search("categories") == []


# --- rebuild_fts_index --------------------------------------------------------


def test_rebuild_fts_index_picks_up_a_changed_tokenizer_definition(tmp_path):
    """The scenario rebuild_fts_index() exists for: a database whose
    chunks_fts was created under an OLD definition (here, simulated by
    building it without the porter tokenizer directly, bypassing SCHEMA)
    does not gain stemming just from calling initialize() again - only
    rebuild_fts_index() (unconditional DROP, then recreate from the CURRENT
    SCHEMA) picks up the change."""
    import sqlite3

    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);
        """
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, document_id, source, page, text, created_at) "
        "VALUES (1, 'docA', 'a.pdf', 1, 'diagnostic category', 'now')"
    )
    conn.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (1, 'diagnostic category')")
    conn.commit()
    conn.close()

    store = MetadataStore(db_path)
    keyword_index = KeywordIndex(db_path)

    # Old (non-stemming) table: plural query does not match the singular text.
    assert keyword_index.search("categories") == []

    # initialize() is IF-NOT-EXISTS - it must NOT silently fix this.
    store.initialize()
    assert keyword_index.search("categories") == []

    # rebuild_fts_index() unconditionally drops and recreates from SCHEMA.
    store.rebuild_fts_index()
    assert {r["chunk_id"] for r in keyword_index.search("categories")} == {1}


def test_rebuild_fts_index_preserves_the_chunks_table(tmp_path):
    """rebuild_fts_index() only ever touches chunks_fts - the chunks table
    (and, by extension, everything FAISS/document metadata depend on) is
    read from, never written to."""
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")
    store.insert_chunk(document_id="docA", source="a.pdf", page=2, text="beta")

    before = store.list_documents()
    store.rebuild_fts_index()
    after = store.list_documents()

    assert before == after
    assert store.count() == 2


def test_rebuild_fts_index_reindexes_every_existing_chunk(tmp_path):
    """Every chunk already in the chunks table must be searchable again
    after a rebuild - not just chunks inserted after it."""
    from src.retrieval.keyword_index import KeywordIndex

    db_path = str(tmp_path / "metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    ids = [
        store.insert_chunk(document_id="docA", source="a.pdf", page=i, text=f"unique_term_{i}")
        for i in range(5)
    ]

    store.rebuild_fts_index()

    keyword_index = KeywordIndex(db_path)
    for i, chunk_id in enumerate(ids):
        assert {r["chunk_id"] for r in keyword_index.search(f"unique_term_{i}")} == {chunk_id}


def test_delete_document_unknown_id_is_a_noop(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")

    removed = store.delete_document("does-not-exist")

    assert removed == []
    assert store.count() == 1


# --- insert_chunks: transactionality ----------------------------------------


def test_insert_chunks_successful_multi_chunk_commits_all(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()

    chunk_ids = store.insert_chunks(
        document_id="docA",
        source="a.pdf",
        chunks=[(1, "first"), (2, "second"), (3, "third")],
    )

    assert len(chunk_ids) == 3
    assert len(set(chunk_ids)) == 3  # distinct ids
    assert store.document_chunk_count("docA") == 3
    assert store.count() == 3

    rows = store.get_chunks(chunk_ids)
    assert [rows[cid]["text"] for cid in chunk_ids] == ["first", "second", "third"]

    # FTS index must reflect every inserted chunk too, not just the chunks table.
    from src.retrieval.keyword_index import KeywordIndex

    keyword_index = KeywordIndex(str(tmp_path / "metadata.db"))
    assert {r["chunk_id"] for r in keyword_index.search("first OR second OR third", top_k=10)} == set(chunk_ids)


def test_insert_chunks_failure_partway_rolls_back_the_whole_document(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()

    # The `page` column is NOT NULL, so a None page fails the 2nd insert -
    # the 1st chunk must not survive the rollback either.
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_chunks(
            document_id="docA",
            source="a.pdf",
            chunks=[(1, "first"), (None, "second"), (3, "third")],
        )

    assert store.document_chunk_count("docA") == 0
    assert store.count() == 0


def test_insert_chunks_rollback_does_not_affect_other_documents(tmp_path):
    """A failed insert_chunks() call for one document must not roll back or
    otherwise disturb chunks already committed for a different document -
    each insert_chunks() call is its own independent transaction."""
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    store.insert_chunks(document_id="docA", source="a.pdf", chunks=[(1, "existing")])

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_chunks(document_id="docB", source="b.pdf", chunks=[(None, "bad")])

    assert store.document_chunk_count("docA") == 1
    assert store.document_chunk_count("docB") == 0
    assert store.count() == 1


def test_insert_chunk_singular_still_commits_immediately(tmp_path):
    """insert_chunk() (the existing single-chunk public API) must keep
    working exactly as before - one call, one committed row."""
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()

    chunk_id = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="hello")

    assert isinstance(chunk_id, int)
    assert store.document_chunk_count("docA") == 1
    assert store.get_chunks([chunk_id])[chunk_id]["text"] == "hello"
