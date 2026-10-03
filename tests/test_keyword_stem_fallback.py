"""Production keyword retrieval falls back to a stem prefix for a query word
that matches nothing - research retrieval is left exactly as it was.

Production failure this guards against: "What is the patient allergic to?"
did not retrieve the passage "Allergy: Penicillin; experienced rash and
hives". Dense retrieval ranked that passage FIRST, but the FTS5 porter
tokenizer stems "allergic" to "allerg" and "Allergy" to "allergi", so the
keyword query matched no allergy text. Its only matching term, "patient",
then gave every chunk except the allergy passage (which never says
"patient") a keyword-rank bonus in fusion, pushing it out of the top five.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

import app
from src import config
from src.retrieval import hybrid_retriever
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.keyword_index import MIN_STEM_PREFIX_CHARS, KeywordIndex
from src.retrieval.metadata_store import MetadataStore

DIM = 4
QUESTION = "What is the patient allergic to?"
ALLERGY_TEXT = "Allergy: Penicillin; experienced rash and hives in 1985."


def _vec(cos_to_query):
    v = np.zeros(DIM, dtype="float32")
    v[0] = cos_to_query
    v[1] = np.sqrt(1.0 - cos_to_query**2)
    return v.astype("float32")


@pytest.fixture
def allergy_corpus(tmp_path, monkeypatch):
    """The real failure in miniature: the allergy chunk is the best dense
    match but never says "patient"; six other chunks all say "patient"."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(config, "EMBEDDING_DIM", DIM)

    store = MetadataStore(config.DB_PATH)
    store.initialize()
    index = FaissIndex(DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()

    ids, vectors = [], []
    allergy_id = store.insert_chunk(document_id="d1", source="w.pdf", page=2, text=ALLERGY_TEXT)
    ids.append(allergy_id)
    vectors.append(_vec(0.99))
    for i in range(6):
        cid = store.insert_chunk(
            document_id="d1", source="w.pdf", page=3 + i,
            text=f"The patient was examined; section {i} of the physical examination.",
        )
        ids.append(cid)
        vectors.append(_vec(0.95 - 0.05 * i))

    index.add(np.array(vectors, dtype="float32"), np.array(ids, dtype="int64"))
    index.save()
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _vec(1.0))
    return {"allergy_id": allergy_id, "db": config.DB_PATH}


# --- the failure and the fix, end to end -------------------------------------


def test_default_retrieval_still_misses_the_allergy_passage(allergy_corpus):
    """Documents the original mechanism - and that the default is unchanged."""
    ids = [c["chunk_id"] for c in hybrid_retriever.retrieve_hybrid(QUESTION, top_k=3, document_id="d1")]
    assert allergy_corpus["allergy_id"] not in ids


def test_stem_fallback_retrieves_the_allergy_passage(allergy_corpus):
    ids = [
        c["chunk_id"]
        for c in hybrid_retriever.retrieve_hybrid(QUESTION, top_k=3, document_id="d1", expand_keyword_stems=True)
    ]
    assert ids[0] == allergy_corpus["allergy_id"]


def test_app_retrieval_uses_the_fallback(allergy_corpus):
    chunks, prompt = app.retrieve_and_build_prompt(QUESTION, top_k=config.TOP_K, document_id="d1")
    assert chunks[0]["chunk_id"] == allergy_corpus["allergy_id"]
    assert "Penicillin" in prompt


# --- the query rewrite itself ------------------------------------------------


def test_porter_splits_allergic_from_allergy(allergy_corpus):
    index = KeywordIndex(allergy_corpus["db"])
    assert index.search("allergic", 10) == []
    assert [r["chunk_id"] for r in index.search("allergic", 10, expand_stems=True)] == [
        allergy_corpus["allergy_id"]
    ]


def test_default_query_is_unchanged():
    assert KeywordIndex._sanitize_query(QUESTION) == '"patient" OR "allergic"'


def test_only_a_term_that_matches_nothing_is_expanded():
    """A term porter already handles must not be expanded: its prefix matches
    the same chunks, and the duplicate OR term double-counts it in BM25
    (measured to reorder production rankings on the research benchmark)."""
    unmatched = {'"allergic"'}
    query = KeywordIndex._sanitize_query(
        "patients allergic diagnosed", matches_nothing=lambda t: t in unmatched
    )
    assert query == '"patients" OR "allergic" OR "diagnosed" OR allerg*'


def test_short_or_unchanged_stems_are_never_expanded():
    query = KeywordIndex._sanitize_query("dose renal pain 1985", matches_nothing=lambda t: True)
    assert "*" not in query  # stems unchanged, under MIN_STEM_PREFIX_CHARS, or not alphabetic
    assert MIN_STEM_PREFIX_CHARS >= 5


def test_fallback_expands_a_shared_stem_once():
    query = KeywordIndex._sanitize_query("allergic allergies", matches_nothing=lambda t: True)
    assert query.count("allerg*") == 1


# --- research isolation ------------------------------------------------------


def test_shared_retrieve_hybrid_defaults_to_no_expansion():
    params = inspect.signature(hybrid_retriever.retrieve_hybrid).parameters
    assert params["expand_keyword_stems"].default is False


def test_app_retrieve_hybrid_opts_in(monkeypatch):
    captured = {}

    def fake_shared(query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(app, "_shared_retrieve_hybrid", fake_shared)
    app.retrieve_hybrid("q", top_k=5, document_id="d1")
    assert captured == {"expand_keyword_stems": True, "top_k": 5, "document_id": "d1"}


def test_research_benchmark_uses_the_shared_function_not_the_app_one():
    """The frozen benchmark's research configurations must keep calling the
    shared retrieve_hybrid (expansion off); only its production
    configuration goes through app."""
    source = (Path(__file__).resolve().parent.parent / "research" / "counterbalanced_retrieval_benchmark.py").read_text(
        encoding="utf-8"
    )
    assert "from src.retrieval.hybrid_retriever import retrieve_hybrid" in source
    assert "from app import retrieve_hybrid" not in source
