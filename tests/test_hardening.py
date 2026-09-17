"""Edge-case regression tests for the paths this change touched (Phase 10).

Scope is deliberately narrow: the failure modes that the new intent
classification, balanced allocation, evidence context and validation layer
could plausibly break, plus the isolation guarantees they sit next to.
"""

import numpy as np
import pymupdf
import pytest

import app
from src import config
from src.evaluation import product_qa
from src.exceptions import InsufficientRetrievalError
from src.generation import intent as intent_module
from src.generation.prompt_builder import (
    build_multi_document_context,
    build_prompt,
    format_conversation_history,
)
from src.generation.validation import validate_answer
from src.retrieval import hybrid_retriever
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.metadata_store import MetadataStore


def _chunk(document_id="d1", source="a.pdf", page=1, text="t", rank=1):
    return {
        "chunk_id": rank, "document_id": document_id, "source": source,
        "page": page, "text": text, "score": 1.0 / rank, "rank": rank,
    }


# --- API consistency ----------------------------------------------------------


def test_qa_categories_and_intent_vocabulary_stay_in_sync():
    """The QA dataset's category strings are used directly as intents by the
    validator. If the two lists drift, validation silently stops applying
    the right rules to a whole category."""
    assert set(product_qa.REQUIRED_CATEGORIES) == set(intent_module.ALL_INTENTS)


def test_refusal_phrases_have_a_single_definition():
    """The grader used to keep its own copy of this list."""
    from src.generation import validation

    assert not hasattr(product_qa, "REFUSAL_PHRASES")
    assert validation.REFUSAL_PHRASES


# --- malformed / degenerate inputs -------------------------------------------


def test_empty_chunk_list_still_produces_a_well_formed_prompt():
    prompt = build_prompt("a question", [], disable_benchmark_scope=True)
    assert prompt.rstrip().endswith("ANSWER:")
    assert "EVIDENCE:" in prompt


def test_multi_document_context_with_no_chunks_flags_every_document():
    inventory = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 1},
        {"document_id": "d2", "source": "b.pdf", "page_count": 1},
    ]

    context = build_multi_document_context([], inventory)

    assert context.count("No relevant evidence was retrieved") == 2


def test_multi_document_context_with_empty_inventory_is_empty():
    assert build_multi_document_context([_chunk()], []) == ""


def test_chunk_from_an_unselected_document_is_dropped_not_reattributed():
    """Defence in depth for the prompt layer: even if a stray chunk reached
    it, it must not be attached to an inventory document."""
    inventory = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]
    chunks = [_chunk("d1", "a.pdf", 1, "kept text", 1), _chunk("d2", "b.pdf", 1, "stray text", 2)]

    context = build_multi_document_context(chunks, inventory)

    assert "kept text" in context
    assert "stray text" not in context
    assert "b.pdf" not in context


def test_validator_tolerates_missing_optional_inputs():
    report = validate_answer("some answer", intent_module.FACTUAL)
    assert report["evidence_document_count"] == 0
    assert report["parts_expected"] == 0
    assert report["documents_missing"] == []


def test_intent_classification_tolerates_none_query():
    assert intent_module.classify_intent(None) == intent_module.FACTUAL
    assert intent_module.split_question_parts(None) == [None]


def test_history_entries_missing_content_are_not_silently_dropped():
    with pytest.raises(KeyError):
        format_conversation_history([{"role": "user"}])


# --- isolation and deletion under the new retrieval path ----------------------


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors, dtype="float32")


@pytest.fixture
def two_documents(tmp_path, monkeypatch):
    import scripts.ingest as ingest_script

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _fake_embed_texts([q])[0])

    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    _make_pdf(a, "insulin glucose diabetes management guidance")
    _make_pdf(b, "coronary artery cardiac catheterization report")

    return {
        "a": app.ingest_uploaded_pdf("a.pdf", a.read_bytes())["document_id"],
        "b": app.ingest_uploaded_pdf("b.pdf", b.read_bytes())["document_id"],
    }


def test_deleted_document_cannot_be_reached_by_balanced_retrieval(two_documents):
    """Deletion must hold on the balanced path too, not just the default one."""
    app.delete_document(two_documents["a"])

    results = retrieve_hybrid(
        "summarize each selected document separately",
        top_k=10,
        document_id=sorted(two_documents.values()),
        balance_by_document=True,
    )

    assert all(c["document_id"] == two_documents["b"] for c in results)


def test_balanced_retrieval_over_a_missing_index_fails_loudly(tmp_path, monkeypatch):
    """Not a silent empty result: a missing FAISS index is a setup error and
    must keep saying so on the balanced path too."""
    from src.exceptions import IndexNotFoundError

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "empty.index"))
    MetadataStore(config.DB_PATH).initialize()
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _fake_embed_texts([q])[0])

    with pytest.raises(IndexNotFoundError):
        retrieve_hybrid("anything", document_id=["gone"], balance_by_document=True)


def test_empty_selection_short_circuits_before_touching_the_index(tmp_path, monkeypatch):
    """An empty selection returns nothing without loading anything, so it
    can never fall back to a global search."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "empty.index"))

    assert retrieve_hybrid("anything", document_id=[], balance_by_document=True) == []


def test_empty_retrieval_still_raises_insufficient_retrieval(monkeypatch):
    monkeypatch.setattr(app, "retrieve_hybrid", lambda query, **kwargs: [])

    with pytest.raises(InsufficientRetrievalError):
        app.retrieve_and_build_prompt("Summarize each selected document separately.")


def test_balanced_allocation_skips_chunks_whose_metadata_vanished():
    """A chunk deleted between fusion and allocation has no document, and
    must be skipped rather than crashing or being attributed to one."""
    from src.retrieval.hybrid_retriever import _allocate_by_document

    taken = _allocate_by_document([1, 2, 3], {1: "A", 3: "B"}, document_count=2, top_k=3)

    assert 2 not in taken
    assert set(taken) == {1, 3}


def test_single_document_scope_is_unaffected_by_the_new_wiring(two_documents, monkeypatch):
    monkeypatch.setattr(app, "generate", lambda prompt: "mock answer")

    result = app.answer_question("insulin guidance", document_id=two_documents["a"])

    assert result["chunks"]
    assert all(c["document_id"] == two_documents["a"] for c in result["chunks"])
