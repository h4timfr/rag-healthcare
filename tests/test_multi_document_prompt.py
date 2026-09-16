"""Tests for multi-document synthesis: prompt_builder's document-grouped
context/rules (used only via build_prompt(document_inventory=...)), and
app.py's build_document_inventory() - the single point that decides which
documents' names/evidence ever reach the generation context.

Single-document/active-document behavior is asserted unchanged throughout:
build_prompt() with no document_inventory (or document_inventory=None) must
produce byte-identical output to before this feature existed.
"""

import numpy as np
import pymupdf
import pytest

import app
from src import config
from src.generation.prompt_builder import (
    build_multi_document_context,
    build_prompt,
    detect_multi_document_intent,
)
from src.retrieval import hybrid_retriever


def _chunk(document_id, source, page, text, rank):
    return {
        "chunk_id": rank,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": 1.0 / rank,
        "rank": rank,
    }


INVENTORY_3 = [
    {"document_id": "d1", "source": "Pediatric Report.pdf", "page_count": 3},
    {"document_id": "d2", "source": "OMC Report Sample - Cardio.pdf", "page_count": 4},
    {"document_id": "d3", "source": "RAG_Healthcare_Review1.pdf", "page_count": 10},
]


# --- all 3 selected documents have evidence ---------------------------------


def test_all_three_documents_named_in_context_when_all_have_evidence():
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "pediatric evidence text", 1),
        _chunk("d2", "OMC Report Sample - Cardio.pdf", 4, "cardio evidence text", 2),
        _chunk("d3", "RAG_Healthcare_Review1.pdf", 1, "rag pipeline evidence text", 3),
    ]

    prompt = build_prompt("tell me about all the documents each of them", chunks, document_inventory=INVENTORY_3)

    for doc in INVENTORY_3:
        assert doc["source"] in prompt
    assert "pediatric evidence text" in prompt
    assert "cardio evidence text" in prompt
    assert "rag pipeline evidence text" in prompt
    assert "No relevant evidence was retrieved" not in prompt


# --- only 2 of 3 selected documents have evidence ---------------------------


def test_document_without_evidence_is_explicitly_flagged_not_invented():
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "pediatric evidence text", 1),
        _chunk("d2", "OMC Report Sample - Cardio.pdf", 4, "cardio evidence text", 2),
        # no chunks for d3 (RAG_Healthcare_Review1.pdf)
    ]

    context = build_multi_document_context(chunks, INVENTORY_3)
    prompt = build_prompt("summarize each document", chunks, document_inventory=INVENTORY_3)

    sections = context.split("Document:")
    d3_section = next(s for s in sections if "RAG_Healthcare_Review1.pdf" in s)
    assert "No relevant evidence was retrieved for this document" in d3_section

    # The other two documents' sections must NOT carry that placeholder.
    d1_section = next(s for s in sections if "Pediatric Report.pdf" in s)
    d2_section = next(s for s in sections if "Cardio.pdf" in s)
    assert "No relevant evidence was retrieved" not in d1_section
    assert "No relevant evidence was retrieved" not in d2_section

    # All three document names still appear (so the model knows what was
    # selected), and the rules instruct it not to invent content for d3.
    assert "RAG_Healthcare_Review1.pdf" in prompt
    assert "No relevant evidence was retrieved for this document" in prompt
    assert "does NOT mean you have evidence" in prompt


def test_document_without_evidence_never_gets_fabricated_chunk_text():
    """Selected != evidence: a document absent from `chunks` must never
    contribute any chunk text to its own section - only the explicit
    no-evidence placeholder."""
    chunks = [_chunk("d1", "Pediatric Report.pdf", 1, "pediatric evidence text", 1)]

    context = build_multi_document_context(chunks, INVENTORY_3)
    sections = {
        doc["source"]: part
        for doc, part in zip(
            INVENTORY_3, context.split("Document: ")[1:]
        )
    }
    assert "pediatric evidence text" in sections["Pediatric Report.pdf"]
    assert sections["OMC Report Sample - Cardio.pdf"].strip().startswith(
        "OMC Report Sample - Cardio.pdf\n\n(No relevant evidence"
    )
    assert sections["RAG_Healthcare_Review1.pdf"].strip().startswith(
        "RAG_Healthcare_Review1.pdf\n\n(No relevant evidence"
    )


# --- "each document" / comparison intent detection --------------------------


@pytest.mark.parametrize(
    "query",
    [
        "tell me about all the documents each of them",
        "Summarize each selected document separately.",
        "Compare the selected documents.",
        "What do they have in common and what differs between them?",
        "give me every document's summary",
    ],
)
def test_multi_document_intent_detected(query):
    assert detect_multi_document_intent(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What medication was the patient prescribed?",
        "What is the aneurysm size?",
        "What does the report say about follow-up care?",
    ],
)
def test_multi_document_intent_not_detected_for_narrow_questions(query):
    assert detect_multi_document_intent(query) is False


def test_each_document_query_adds_structure_instruction():
    chunks = [_chunk("d1", "Pediatric Report.pdf", 1, "text", 1)]
    prompt = build_prompt("Summarize each selected document separately.", chunks, document_inventory=INVENTORY_3)
    assert "one paragraph per selected document" in prompt


def test_comparison_query_adds_structure_instruction():
    chunks = [_chunk("d1", "Pediatric Report.pdf", 1, "text", 1)]
    prompt = build_prompt("Compare the selected documents.", chunks, document_inventory=INVENTORY_3)
    assert "one paragraph per selected document" in prompt


def test_narrow_question_in_multi_document_scope_skips_structure_instruction():
    """Multi-document grounding rules still apply (selected != evidence etc.),
    but the model isn't forced into a per-document-section structure unless
    the question actually asks for all/each/every/comparison."""
    chunks = [_chunk("d1", "Pediatric Report.pdf", 1, "text", 1)]
    prompt = build_prompt("What medication was prescribed?", chunks, document_inventory=INVENTORY_3)
    assert "MULTI-DOCUMENT RULES" in prompt
    assert "one paragraph per selected document" not in prompt


# --- single-document mode is unaffected --------------------------------------


def test_build_prompt_without_inventory_is_unchanged():
    chunks = [_chunk("d1", "a.pdf", 1, "hello world", 1)]

    prompt_default = build_prompt("what is this about?", chunks)
    prompt_explicit_none = build_prompt("what is this about?", chunks, document_inventory=None)

    assert prompt_default == prompt_explicit_none
    assert "MULTI-DOCUMENT RULES" not in prompt_default
    assert "=== DOCUMENT:" not in prompt_default
    assert "hello world" in prompt_default


def test_active_scope_never_builds_a_document_inventory(monkeypatch):
    """Mirrors app.py's render_chat wiring: analysis_scope == "active" must
    always pass document_inventory=None to retrieve_and_build_prompt, so
    build_prompt() takes its original single-document path."""

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):
        return [{"chunk_id": 1, "document_id": document_id, "source": "a.pdf", "page": 1, "text": "t", "score": 0.1, "rank": 1}]

    captured = {}
    real_build_prompt = app.build_prompt

    def spy_build_prompt(query, chunks, document_inventory=None, **kwargs):
        captured["document_inventory"] = document_inventory
        return real_build_prompt(query, chunks, document_inventory=document_inventory, **kwargs)

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(app, "build_prompt", spy_build_prompt)

    app.retrieve_and_build_prompt("a question", document_id="docA", document_inventory=None)

    assert captured["document_inventory"] is None


# --- unselected documents never leak into the prompt -------------------------


def test_build_document_inventory_excludes_unselected_documents():
    documents = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 1},
        {"document_id": "d2", "source": "b.pdf", "page_count": 2},
        {"document_id": "d3", "source": "c.pdf", "page_count": 3},
    ]

    inventory = app.build_document_inventory(documents, {"d1", "d3"})

    sources = {d["source"] for d in inventory}
    assert sources == {"a.pdf", "c.pdf"}
    assert "b.pdf" not in sources


def test_build_document_inventory_empty_selection_returns_empty_list():
    documents = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]
    assert app.build_document_inventory(documents, set()) == []


def test_unselected_document_name_never_appears_in_multi_document_prompt():
    inventory = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]
    # Even if a stray chunk from a non-inventory document ever reached this
    # function (shouldn't happen given retrieval scoping - defense in depth
    # for the prompt-building layer itself), it must not be attributed to,
    # or otherwise surface, a document outside the inventory.
    chunks = [
        _chunk("d1", "a.pdf", 1, "alpha text", 1),
        _chunk("d2", "b.pdf", 1, "unselected document text", 2),
    ]

    prompt = build_prompt("tell me about all documents", chunks, document_inventory=inventory)

    assert "b.pdf" not in prompt
    assert "unselected document text" not in prompt
    assert "alpha text" in prompt


# --- end-to-end (real ingestion pipeline) -------------------------------------


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
def three_real_documents(tmp_path, monkeypatch):
    import scripts.ingest as ingest_script

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_a, pdf_b, pdf_c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    _make_pdf(pdf_a, "Diabetes management guideline about insulin therapy and glucose monitoring.")
    _make_pdf(pdf_b, "Cardiology report about coronary arteries and cardiac catheterization.")
    _make_pdf(pdf_c, "Pediatric growth chart guidance about height and weight percentiles.")

    a = app.ingest_uploaded_pdf("diabetes.pdf", pdf_a.read_bytes())
    b = app.ingest_uploaded_pdf("cardio.pdf", pdf_b.read_bytes())
    c = app.ingest_uploaded_pdf("pediatric.pdf", pdf_c.read_bytes())

    return {"a": a["document_id"], "b": b["document_id"], "c": c["document_id"]}


def _first_chunk_vector(document_id: str) -> np.ndarray:
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    text = conn.execute(
        "SELECT text FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
    ).fetchone()[0]
    conn.close()
    return _fake_embed_texts([text])[0]


def test_end_to_end_multi_document_prompt_covers_all_selected_with_evidence(
    three_real_documents, monkeypatch
):
    doc_a, doc_b, doc_c = three_real_documents["a"], three_real_documents["b"], three_real_documents["c"]
    documents = app.list_indexed_documents()
    selected = {doc_a, doc_b, doc_c}
    inventory = app.build_document_inventory(documents, selected)

    # Align the query vector with doc_a so retrieval realistically surfaces
    # at least that document; top_k=5 default with 3 short docs typically
    # surfaces evidence for all three from this small a real hybrid search.
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _first_chunk_vector(doc_a))

    chunks, prompt = app.retrieve_and_build_prompt(
        "tell me about all the documents each of them",
        document_id=sorted(selected),
        document_inventory=inventory,
    )

    assert "diabetes.pdf" in prompt
    assert "cardio.pdf" in prompt
    assert "pediatric.pdf" in prompt
    assert "one paragraph per selected document" in prompt
