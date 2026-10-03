"""One document ticked in "Selected documents" is a single-document turn.

Production failure this guards against: with exactly one document selected,
"What treatment was she taking for it?" named a drug that is not in the
evidence ("beta blockers") in 4 of 10 runs. The turn kept the
multi-document prompt ("MULTI-DOCUMENT RULES (1 documents are selected)")
because the fall-back to the single-document prompt only ran when the
question NAMED a document. On identical chunks the single-document prompt
produced 0 of 10. Product QA never saw this: its harness has always run a
one-document selection as Active-document scope.
"""

import app
from src import config

DOC = {"document_id": "d1", "source": "UMNwriteup.pdf", "page_count": 6}
OTHER = {"document_id": "d2", "source": "Other.pdf", "page_count": 3}
QUESTION = "What treatment was she taking for it?"


def _capture(monkeypatch):
    captured = {"calls": []}

    def fake_retrieve_hybrid(query, **kwargs):
        captured["calls"].append(kwargs)
        return [
            {
                "chunk_id": 1, "document_id": "d1", "source": "UMNwriteup.pdf", "page": 2,
                "text": "Diagnosed with hypertension.", "score": 0.5, "rank": 1,
            }
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    return captured


def test_is_multi_document_inventory():
    assert not app.is_multi_document_inventory(None)
    assert not app.is_multi_document_inventory([])
    assert not app.is_multi_document_inventory([DOC])
    assert app.is_multi_document_inventory([DOC, OTHER])


def test_one_selected_document_uses_the_single_document_prompt(monkeypatch):
    _capture(monkeypatch)
    _, prompt = app.retrieve_and_build_prompt(
        QUESTION, top_k=config.TOP_K, document_id=["d1"], document_inventory=[DOC]
    )
    assert "MULTI-DOCUMENT RULES" not in prompt
    assert "documents are selected" not in prompt


def test_one_selected_document_matches_active_scope(monkeypatch):
    """Same prompt and same retrieval request as Active-document scope."""
    captured = _capture(monkeypatch)
    _, selected_prompt = app.retrieve_and_build_prompt(
        QUESTION, top_k=config.TOP_K, document_id=["d1"], document_inventory=[DOC]
    )
    _, active_prompt = app.retrieve_and_build_prompt(QUESTION, top_k=config.TOP_K, document_id="d1")

    selected_call, active_call = captured["calls"]
    assert selected_prompt == active_prompt
    assert selected_call["top_k"] == active_call["top_k"] == config.TOP_K
    assert "balance_by_document" not in selected_call


def test_one_selected_document_keeps_its_retrieval_scope(monkeypatch):
    captured = _capture(monkeypatch)
    app.retrieve_and_build_prompt(QUESTION, top_k=config.TOP_K, document_id=["d1"], document_inventory=[DOC])
    assert captured["calls"][0]["document_id"] == ["d1"]


def test_two_selected_documents_keep_the_multi_document_prompt(monkeypatch):
    _capture(monkeypatch)
    _, prompt = app.retrieve_and_build_prompt(
        QUESTION, top_k=config.TOP_K, document_id=["d1", "d2"], document_inventory=[DOC, OTHER]
    )
    assert "MULTI-DOCUMENT RULES" in prompt


def test_one_selected_document_gets_the_single_document_budget():
    assert app.resolve_num_predict(QUESTION, None, [DOC]) == app.resolve_num_predict(QUESTION, None, None)
    assert app.resolve_num_predict(QUESTION, None, [DOC, OTHER]) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
