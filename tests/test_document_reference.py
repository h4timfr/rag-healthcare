"""Tests for document-reference resolution
(src/retrieval/document_reference.py) and its wiring into the production
retrieval path.

The gap these close: retrieval matches chunk TEXT, and a filename appears in
no chunk. Asked "What medications are listed in the OMC report?" over three
selected documents, the app retrieved five Pediatric chunks and answered
"albuterol" - a real medication, from the real evidence, out of the wrong
document. An ordinal is worse still: "what about the third document?" has no
content words at all, so retrieval had nothing to match and only ever got
the right document by luck.

The inventory used here mirrors the real production one, including its
order, because ordinal references are defined by that order:

    1. Pediatric Report.pdf
    2. RAG_Healthcare_Review1.pdf
    3. OMC Report Sample - Cardio.pdf
"""

import app
import pytest
from src import config
from src.retrieval.document_reference import DocumentReference, resolve_document_reference


INVENTORY = [
    {"document_id": "ped", "source": "Pediatric Report.pdf", "page_count": 3},
    {"document_id": "rag", "source": "RAG_Healthcare_Review1.pdf", "page_count": 10},
    {"document_id": "omc", "source": "OMC Report Sample - Cardio.pdf", "page_count": 4},
]

HISTORY_EACH_DOCUMENT = [
    {"role": "user", "content": "Summarize each selected document separately."},
    {
        "role": "assistant",
        "content": "Pediatric Report.pdf: a paediatric case review. "
        "RAG_Healthcare_Review1.pdf: a proposed RAG pipeline. "
        "OMC Report Sample - Cardio.pdf: a cardiology board review.",
    },
]

HISTORY_ONE_DOCUMENT = [
    {"role": "user", "content": "Tell me about the Pediatric Report."},
    {"role": "assistant", "content": "Pediatric Report.pdf describes a paediatric case review."},
]


def _ids(reference) -> list[str]:
    return reference.document_ids


# --- filename references ------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What medications are listed in the OMC report?", ["omc"]),
        ("Tell me about the Pediatric Report.", ["ped"]),
        ("Summarize RAG_Healthcare_Review1.pdf", ["rag"]),
        ("what does pediatric report.pdf say about the cough?", ["ped"]),
        # Punctuation in the filename need not be reproduced exactly.
        ("Summarize the RAG Healthcare Review1 document", ["rag"]),
    ],
)
def test_filename_and_near_filename_references_resolve(query, expected):
    assert _ids(resolve_document_reference(query, INVENTORY)) == expected


def test_distinctive_keyword_resolves_without_the_full_filename():
    """"the OMC report" names one document by one identifying word; nobody
    types "the OMC Cardio report"."""
    reference = resolve_document_reference("What is in the OMC report?", INVENTORY)
    assert _ids(reference) == ["omc"]
    assert reference.kind == "keyword"


# --- ordinal references -------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What about the first document?", ["ped"]),
        ("What about the second document?", ["rag"]),
        ("What about the third document?", ["omc"]),
        ("What about the last document?", ["omc"]),
        ("Tell me about the 2nd report", ["rag"]),
    ],
)
def test_ordinal_references_follow_inventory_order(query, expected):
    assert _ids(resolve_document_reference(query, INVENTORY)) == expected


def test_multiple_ordinals_resolve_to_exactly_those_documents():
    reference = resolve_document_reference("Compare the first and third documents.", INVENTORY)
    assert _ids(reference) == ["ped", "omc"]
    assert reference.kind == "ordinal"


def test_documents_named_in_different_styles_all_resolve():
    """Regression. One question can name one document by filename and
    another by an identifying word. Trying the naming rules in turn and
    keeping only the first that matched dropped the rest: this question
    matched "Pediatric Report" as a filename, stopped there, and a
    two-document comparison was scoped to one document with no evidence
    from the other at all."""
    query = "What do the Pediatric Report and the OMC Cardiology report have in common?"

    reference = resolve_document_reference(query, INVENTORY)

    assert _ids(reference) == ["ped", "omc"]
    assert reference.kind == "filename+keyword"


def test_an_ordinal_and_a_name_in_one_question_both_resolve():
    reference = resolve_document_reference("Compare the first document and the OMC report.", INVENTORY)
    assert _ids(reference) == ["ped", "omc"]


def test_resolution_is_returned_in_inventory_order():
    """"the third and first documents" and "the first and third documents"
    must resolve identically."""
    a = resolve_document_reference("Compare the first and third documents.", INVENTORY)
    b = resolve_document_reference("Compare the third and first documents.", INVENTORY)
    assert _ids(a) == _ids(b) == ["ped", "omc"]


def test_out_of_range_ordinal_resolves_to_nothing():
    """Three documents have no fifth. Guessing would be worse than saying
    nothing, so the selection is left alone."""
    assert resolve_document_reference("What about the fifth document?", INVENTORY).documents == ()


def test_ordinal_needs_a_document_noun():
    """The regression this guard exists for: a clinical ordinal is not a
    document reference. "the second hospital admission" must not resolve."""
    query = (
        "What medications was the patient discharged on, and what condition was "
        "found during the second hospital admission?"
    )
    assert resolve_document_reference(query, INVENTORY).documents == ()


# --- conversation-context references -----------------------------------------


def test_unambiguous_antecedent_resolves_from_recent_history():
    reference = resolve_document_reference(
        "What about that report?", INVENTORY, history=HISTORY_ONE_DOCUMENT
    )
    assert _ids(reference) == ["ped"]
    assert reference.kind == "history"


def test_ambiguous_antecedent_does_not_guess():
    """After an answer naming all three documents, "that report" has no
    single antecedent. It must resolve to NONE of them and say so."""
    reference = resolve_document_reference(
        "What about that report?", INVENTORY, history=HISTORY_EACH_DOCUMENT
    )
    assert reference.documents == ()
    assert reference.ambiguous is True
    assert "Pediatric Report.pdf" in reference.detail


def test_demonstrative_without_history_resolves_to_nothing():
    assert resolve_document_reference("What about that report?", INVENTORY).documents == ()


def test_contested_keyword_is_ambiguous_not_a_guess():
    """Two selected documents sharing an identifying word cannot be told
    apart by it."""
    inventory = [
        {"document_id": "c1", "source": "Cardio Report One.pdf", "page_count": 1},
        {"document_id": "c2", "source": "Cardio Report Two.pdf", "page_count": 1},
    ]

    reference = resolve_document_reference("What is in the cardio report?", inventory)

    assert reference.documents == ()
    assert reference.ambiguous is True


# --- questions that reference nothing -----------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Compare the selected documents.",
        "Summarize each selected document separately.",
        "What medication was the patient prescribed?",
        "What does the report say about follow-up care?",
        "What is the aneurysm size?",
    ],
)
def test_ordinary_questions_reference_no_document(query):
    """Requirement 8: no reference means current behaviour, untouched. Note
    "What does the report say..." has a document noun but names nothing."""
    assert resolve_document_reference(query, INVENTORY).documents == ()


def test_empty_inputs_are_safe():
    assert resolve_document_reference("", INVENTORY).documents == ()
    assert resolve_document_reference("the OMC report", []).documents == ()
    assert resolve_document_reference(None, INVENTORY).documents == ()


def test_reference_is_falsy_when_nothing_resolved():
    assert not DocumentReference()
    assert bool(resolve_document_reference("What about the first document?", INVENTORY))


# --- isolation ----------------------------------------------------------------


def test_resolution_can_only_return_selected_documents():
    """A reference to a document that is NOT selected resolves to nothing -
    it cannot widen the selection."""
    selected_two = INVENTORY[:2]  # Pediatric, RAG review - no OMC

    reference = resolve_document_reference("What is in the OMC report?", selected_two)

    assert reference.documents == ()


def test_stale_reference_after_deletion_cannot_produce_an_invalid_id():
    """The inventory is rebuilt from what is indexed now, so a document that
    has been deleted is simply absent and unreferenceable."""
    remaining = [d for d in INVENTORY if d["document_id"] != "omc"]

    reference = resolve_document_reference("Tell me about the OMC report", remaining)

    assert reference.documents == ()
    # And an ordinal now means the new third position, which does not exist.
    assert resolve_document_reference("What about the third document?", remaining).documents == ()


def test_ordinals_track_the_current_selection_not_the_whole_library():
    """With two documents selected, "the second document" is the second
    SELECTED one."""
    selected = [INVENTORY[0], INVENTORY[2]]  # Pediatric, OMC
    assert _ids(resolve_document_reference("What about the second document?", selected)) == ["omc"]


# --- wiring into retrieval ----------------------------------------------------


def _capture(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query, **kwargs):
        captured.update(kwargs)
        return [
            {
                "chunk_id": 1, "document_id": "omc", "source": "OMC Report Sample - Cardio.pdf",
                "page": 1, "text": "evidence", "score": 0.5, "rank": 1,
            }
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    return captured


def test_named_document_scopes_retrieval_to_that_document(monkeypatch):
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medications are listed in the OMC report?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert captured["document_id"] == ["omc"]
    assert captured["top_k"] == config.TOP_K


def test_ordinal_scopes_retrieval_to_that_document(monkeypatch):
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What about the third document?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert captured["document_id"] == ["omc"]


def test_multi_document_reference_scopes_to_exactly_those_documents(monkeypatch):
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "Compare the first and third documents.",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert captured["document_id"] == ["ped", "omc"]
    assert "rag" not in captured["document_id"]


def test_single_document_reference_uses_the_single_document_prompt(monkeypatch):
    """Narrowed to one document, the turn is no longer multi-document. The
    multi-document rules ("1 documents are selected", plus a rule about
    documents whose evidence was not retrieved) made the model answer that
    the referenced document was a different file with an empty evidence
    block."""
    _capture(monkeypatch)

    _, prompt = app.retrieve_and_build_prompt(
        "What about the third document?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert "MULTI-DOCUMENT RULES" not in prompt


def test_multi_document_reference_keeps_the_multi_document_prompt(monkeypatch):
    """Two referenced documents is still a multi-document turn, and the
    unreferenced one is gone from the prompt entirely."""
    _capture(monkeypatch)

    _, prompt = app.retrieve_and_build_prompt(
        "Compare the first and third documents.",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert "MULTI-DOCUMENT RULES" in prompt
    assert "Pediatric Report.pdf" in prompt
    assert "OMC Report Sample - Cardio.pdf" in prompt
    assert "RAG_Healthcare_Review1.pdf" not in prompt


def test_unreferenced_question_keeps_the_full_selection(monkeypatch):
    """Requirement 8, at the wiring level."""
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medication was the patient prescribed?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert captured["document_id"] == ["ped", "rag", "omc"]


def test_ambiguous_reference_keeps_the_full_selection(monkeypatch):
    """Requirement 10: no silent guess."""
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What about that report?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
        history=HISTORY_EACH_DOCUMENT,
    )

    assert captured["document_id"] == ["ped", "rag", "omc"]


def test_each_document_question_still_balances_over_the_full_selection(monkeypatch):
    """Requirement 12: the context-budget fix is preserved."""
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "Summarize each selected document separately.",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
    )

    assert captured["document_id"] == ["ped", "rag", "omc"]
    assert captured["balance_by_document"] is True
    assert captured["top_k"] == config.BALANCED_CHUNKS_PER_DOCUMENT * 3


def test_active_document_mode_never_resolves_references(monkeypatch):
    """Requirement 9. Active scope passes no inventory, so the resolver is
    never consulted and isolation is exactly as before."""
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medications are listed in the OMC report?", document_id="ped"
    )

    assert captured["document_id"] == "ped"
    assert captured["top_k"] == config.TOP_K
    assert "balance_by_document" not in captured


def test_history_still_never_reaches_retrieval(monkeypatch):
    """Requirement 11: resolution reads history, retrieval still does not."""
    captured = _capture(monkeypatch)

    app.retrieve_and_build_prompt(
        "What about that report?",
        top_k=config.TOP_K,
        document_id=["ped", "rag", "omc"],
        document_inventory=INVENTORY,
        history=HISTORY_ONE_DOCUMENT,
    )

    assert captured["query"] if "query" in captured else True
    assert "history" not in captured
    assert captured["document_id"] == ["ped"]
