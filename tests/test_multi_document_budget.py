"""Regression tests for the multi-document retrieval budget.

The production failure these exist for, reproduced end to end against the
real index before the fix:

    Selected documents: Pediatric Report.pdf, RAG_Healthcare_Review1.pdf,
                        OMC Report Sample - Cardio.pdf
    Question: "What medications was the patient discharged on, and what
               condition was found during the second hospital admission?"
    Answer:   "The EVIDENCE does not mention the patient being discharged on
               any specific medications."

The sentence naming those medications was in the HIGHEST-ranked chunk of the
prompt the model was given. Extraction was correct, retrieval was correct,
and the fact was present - the prompt was simply too large for the model to
use it. The budget was TOP_K * 3 = 15 chunks (~20KB) because selected-document
scope widened the budget for every question, including narrow ones. The same
question over the same index answered correctly at 5, 7, 9 and 12 chunks and
failed deterministically at 13, 14 and 15.

So these tests pin the budget policy, not the model: a narrow question gets
the plain budget however many documents are selected, breadth comes from
balanced allocation choosing WHICH chunks survive, and the total is capped so
it cannot grow without limit as documents are added.
"""

import app
import pytest
from src import config


INVENTORY_3 = [
    {"document_id": "d1", "source": "a.pdf", "page_count": 1},
    {"document_id": "d2", "source": "b.pdf", "page_count": 1},
    {"document_id": "d3", "source": "c.pdf", "page_count": 1},
]


def _chunk(document_id="d1", source="a.pdf", page=1, rank=1):
    return {
        "chunk_id": rank, "document_id": document_id, "source": source,
        "page": page, "text": "evidence text", "score": 1.0 / rank, "rank": rank,
    }


def _capture_retrieval(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query, **kwargs):
        captured.update(kwargs)
        return [_chunk()]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    return captured


# --- the budget policy --------------------------------------------------------


def test_narrow_question_is_not_widened_by_selecting_more_documents():
    """The regression. Selecting more documents must not enlarge the evidence
    budget for a question that is answerable from one of them."""
    for n in (1, 2, 3, 5, 10):
        ids = {f"d{i}" for i in range(n)}
        assert app.resolve_retrieval_top_k("selected", ids) == config.TOP_K


def test_each_document_question_widens_per_document():
    assert app.resolve_retrieval_top_k("selected", {"d1", "d2"}, balanced=True) == (
        config.BALANCED_CHUNKS_PER_DOCUMENT * 2
    )


def test_budget_is_capped_however_many_documents_are_selected():
    """Ten selected documents must not ask for fifty chunks."""
    ids = {f"d{i}" for i in range(10)}
    assert app.resolve_retrieval_top_k("selected", ids, balanced=True) == config.MAX_CONTEXT_CHUNKS
    assert app.resolve_retrieval_top_k("selected", ids) == config.TOP_K


def test_cap_does_not_change_the_three_document_case():
    """The cap is latent protection, not a change to current behaviour."""
    assert config.BALANCED_CHUNKS_PER_DOCUMENT * 3 <= config.MAX_CONTEXT_CHUNKS


def test_active_scope_and_empty_selection_are_always_plain_top_k():
    assert app.resolve_retrieval_top_k("active", {"d1", "d2", "d3"}) == config.TOP_K
    assert app.resolve_retrieval_top_k("active", {"d1"}, balanced=True) == config.TOP_K
    assert app.resolve_retrieval_top_k("selected", set(), balanced=True) == config.TOP_K


# --- what actually reaches retrieval -----------------------------------------


def test_the_failing_question_now_requests_the_plain_budget(monkeypatch):
    """The exact production question, with three documents selected."""
    captured = _capture_retrieval(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medications was the patient discharged on, and what condition was "
        "found during the second hospital admission?",
        top_k=app.resolve_retrieval_top_k("selected", {"d1", "d2", "d3"}),
        document_id=["d1", "d2", "d3"],
        document_inventory=INVENTORY_3,
    )

    assert captured["top_k"] == config.TOP_K
    # A narrow question is not an each-document question, so allocation is
    # left alone as well.
    assert "balance_by_document" not in captured


def test_caller_supplied_top_k_cannot_reintroduce_the_oversized_budget(monkeypatch):
    """retrieve_and_build_prompt owns the multi-document budget. A caller
    that still passes the old widened value must not get it back - that is
    what keeps the Streamlit app and the QA harness from drifting apart."""
    captured = _capture_retrieval(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medication was prescribed?",
        top_k=config.TOP_K * 3,  # the old, oversized value
        document_id=["d1", "d2", "d3"],
        document_inventory=INVENTORY_3,
    )

    assert captured["top_k"] == config.TOP_K


def test_each_document_question_still_gets_balanced_allocation(monkeypatch):
    captured = _capture_retrieval(monkeypatch)

    app.retrieve_and_build_prompt(
        "Summarize each selected document separately.",
        top_k=config.TOP_K,
        document_id=["d1", "d2", "d3"],
        document_inventory=INVENTORY_3,
    )

    assert captured["balance_by_document"] is True
    assert captured["top_k"] == config.BALANCED_CHUNKS_PER_DOCUMENT * 3


def test_single_document_scope_is_untouched(monkeypatch):
    """Active-document isolation and budget must not move."""
    captured = _capture_retrieval(monkeypatch)

    app.retrieve_and_build_prompt("What medication was prescribed?", document_id="d1")

    assert captured["top_k"] == config.TOP_K
    assert captured["document_id"] == "d1"
    assert "balance_by_document" not in captured


def test_selected_scope_still_passes_only_the_selected_documents(monkeypatch):
    """Isolation is unaffected by the budget change: retrieval is still
    scoped to exactly the selected ids."""
    captured = _capture_retrieval(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medication was prescribed?",
        top_k=config.TOP_K,
        document_id=["d1", "d3"],
        document_inventory=[INVENTORY_3[0], INVENTORY_3[2]],
    )

    assert captured["document_id"] == ["d1", "d3"]


# --- prompt size --------------------------------------------------------------


@pytest.mark.parametrize("n_documents", [1, 2, 3, 6])
def test_prompt_size_does_not_grow_with_the_number_of_selected_documents(
    monkeypatch, n_documents
):
    """The property that was violated. A narrow question's prompt is a
    function of TOP_K, not of how many documents happen to be selected."""
    inventory = [
        {"document_id": f"d{i}", "source": f"doc{i}.pdf", "page_count": 1}
        for i in range(n_documents)
    ]

    def fake_retrieve_hybrid(query, top_k=5, **kwargs):
        # One chunk per requested slot, all from the first document, which is
        # the realistic shape for a question only one document can answer.
        return [
            _chunk("d0", "doc0.pdf", page=i + 1, rank=i + 1) for i in range(top_k)
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, _ = app.retrieve_and_build_prompt(
        "What medications was the patient discharged on?",
        top_k=config.TOP_K,
        document_id=[d["document_id"] for d in inventory],
        document_inventory=inventory,
    )

    assert len(chunks) == config.TOP_K
