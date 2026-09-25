"""Tests for the deterministic question-intent classifier
(src/generation/intent.py) and multi-part decomposition.

The classifier decides prompt shape and whether multi-document retrieval
balances its budget, so a misclassification silently changes an answer's
structure.

The questions written out below are synthetic, but each one mirrors the
SHAPE of a real case in data/product_qa.json - that file is git-ignored
because its questions embed real clinical facts from private sample
reports, and reproducing them here would defeat that. The real dataset is
still asserted against, by loading it at run time when it is present (see
test_every_real_dataset_question_classifies_as_its_category).
"""

import json

import pytest

from src import config
from src.generation import intent as intent_module
from src.generation.intent import (
    ALL_INTENTS,
    COMPARISON,
    EACH_DOCUMENT,
    FACTUAL,
    FOLLOW_UP,
    MULTI_PART,
    NUMERIC,
    SUMMARY,
    UNSUPPORTED,
    classify_intent,
    is_multi_part,
    needs_document_balance,
    split_question_parts,
)

HISTORY = [
    {"role": "user", "content": "Summarize each selected document separately."},
    {"role": "assistant", "content": "Pediatric Report.pdf: ..."},
]


# --- question shapes ----------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        # factual
        ("What was the chief complaint at the follow-up visit?", FACTUAL),
        ("What was the reason the patient was referred to the clinic?", FACTUAL),
        ("What embedding model does the proposed system use?", FACTUAL),
        # numeric
        ("On what date did the clinic call the family about the imaging results?", NUMERIC),
        ("What was the size in centimeters of the lesion found during the procedure?", NUMERIC),
        ("How many studies are cited in the literature survey?", NUMERIC),
        # summary
        ("Summarize this document.", SUMMARY),
        # each document
        ("Summarize each selected document separately.", EACH_DOCUMENT),
        ("What is the main purpose of each document?", EACH_DOCUMENT),
        # comparison
        ("What do the clinic note and the specialist report have in common?", COMPARISON),
        ("Compare the selected documents.", COMPARISON),
        # multi-part
        ("Besides the imaging, what test did the clinician order at the visit, and what medications were prescribed at the follow-up?", MULTI_PART),
        ("What medications was the patient discharged on, and what condition was found during the second admission?", MULTI_PART),
        ("What are the three planned generation-evaluation metrics, and what technology is used for the vector database?", MULTI_PART),
    ],
)
def test_question_shapes_classify_as_expected(query, expected):
    assert classify_intent(query) == expected


def test_every_real_dataset_question_classifies_as_its_category():
    """The synthetic questions above are only useful if they really do
    mirror the dataset. This asserts the classifier against the actual
    private dataset when it is present, without committing its text.

    Two categories are expected NOT to match their dataset label, and that
    is the documented design rather than a miss:
      unsupported  is not lexically detectable (see the intent module).
      follow_up    depends on conversation history, which a single question
                   carries no trace of; only later turns of a case are
                   follow-ups, and only in context.
    """
    dataset_path = config.PROJECT_ROOT / "data" / "product_qa.json"
    if not dataset_path.exists():
        pytest.skip("private product QA dataset not present (it is git-ignored)")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    mismatches = []
    for case in dataset["cases"]:
        category = case["category"]
        if category in (UNSUPPORTED, FOLLOW_UP):
            continue
        first_turn = case["turns"][0]
        if classify_intent(first_turn["question"]) != category:
            mismatches.append(case["id"])

    assert mismatches == []


def test_follow_up_needs_history_to_be_a_follow_up():
    """The same text is a follow-up mid-conversation and an ordinary
    question as an opening turn - there is nothing to refer back to."""
    query = "What about the medications in that report?"
    assert classify_intent(query, history=HISTORY) == FOLLOW_UP
    assert classify_intent(query, history=None) != FOLLOW_UP


@pytest.mark.parametrize(
    "query",
    [
        "What did you just say caused that?",
        "What about the medications in the second document?",
        "And what else did they find?",
    ],
)
def test_follow_up_phrasings_detected_with_history(query):
    assert classify_intent(query, history=HISTORY) == FOLLOW_UP


@pytest.mark.parametrize(
    "query",
    [
        "What about the second document?",
        "What about the third document?",
        "How about the other report?",
        "What about that report?",
        "Tell me about the Pediatric Report.",
    ],
)
def test_a_bare_document_reference_asks_for_that_document(query):
    """A question that is only a document reference has no question left in
    it once the reference is resolved. Classified factual, the model
    answered "what about the third document?" by copying the first evidence
    block back out verbatim; what the user wants is that document's
    contents. Resolving WHICH document is the reference resolver's job (see
    tests/test_document_reference.py), not the answer style's."""
    assert classify_intent(query, history=HISTORY) == SUMMARY
    assert classify_intent(query, history=None) == SUMMARY


def test_a_reference_with_real_content_is_not_a_bare_reference():
    """"the medications in the second document" asks something specific, so
    it keeps its follow-up shape rather than becoming a summary."""
    query = "What about the medications in the second document?"
    assert classify_intent(query, history=HISTORY) != SUMMARY


def test_tell_me_about_a_topic_is_not_a_document_reference():
    assert classify_intent("Tell me about the patient's medications") == FACTUAL


def test_ordinal_in_clinical_content_is_not_a_follow_up_reference():
    """"the second admission" is content, not a reference to the
    second document - the ordinal pattern is scoped to document words for
    exactly this reason."""
    query = "What condition was found during the second admission?"
    assert classify_intent(query, history=HISTORY) != FOLLOW_UP


# --- unsupported --------------------------------------------------------------


def test_unsupported_questions_look_factual_by_shape():
    """Documented, deliberate behaviour: an unsupported question is not
    lexically distinguishable from an answerable one. These three are the
    dataset's unsupported cases and they classify by their surface form."""
    assert classify_intent("What was the patient's blood type?") == FACTUAL
    assert classify_intent("What chemotherapy regimen did the patient receive?") == FACTUAL
    assert classify_intent("What accuracy percentage did the proposed RAG system achieve?") == NUMERIC


def test_no_evidence_is_the_one_deterministic_unsupported_signal():
    assert classify_intent("What was the patient's blood type?", has_evidence=False) == UNSUPPORTED


def test_unsupported_is_part_of_the_vocabulary():
    assert UNSUPPORTED in ALL_INTENTS
    assert len(ALL_INTENTS) == 8


# --- priority ordering --------------------------------------------------------


def test_each_document_beats_summary():
    """"Summarize each selected document separately" must produce per-document
    sections, not one blended summary."""
    assert classify_intent("Summarize each selected document separately.") == EACH_DOCUMENT


# --- each/every: document-scoped, not bare -------------------------------------
#
# Regression for a real bug: bare "\beach\b"/"\bevery\b" matched ordinary
# content-level questions too ("each diagnostic category", "each treatment
# category"), where "each"/"every" refers to something INSIDE the document,
# not to separate documents. That misclassified them as EACH_DOCUMENT, which
# activates the per-document ANSWER STYLE unconditionally (build_answer_style()
# reads the label straight off classify_intent(), not gated on
# document_inventory) - on a single-document question with no document list
# to draw section names from, the model fabricated headings like
# "Pediatric Report.pdf:". The fix requires a document noun immediately
# after "each"/"every".


@pytest.mark.parametrize(
    "query",
    [
        "What were the findings for each diagnostic category?",
        "What were the percentages for each treatment category?",
        "What was recorded for every diagnosis category?",
    ],
)
def test_content_level_each_every_is_not_each_document(query):
    assert classify_intent(query) != EACH_DOCUMENT


@pytest.mark.parametrize(
    "query",
    [
        "Summarize each selected document separately.",
        "What is the main purpose of each document?",
        "What is the main purpose of every document?",
        "Summarize each document.",
    ],
)
def test_document_level_each_every_is_still_each_document(query):
    assert classify_intent(query) == EACH_DOCUMENT


# --- requests_multiple_components (Task 6) -------------------------------
#
# A narrower, purely lexical signal from a single FACTUAL question's answer
# needing to cover several distinct components - see the docstring in
# src/generation/intent.py for why this is deliberately not is_multi_part().


@pytest.mark.parametrize(
    "query",
    [
        "What was the patient's chief complaint?",
        "What medication was prescribed?",
        "Who is the patient's mother and father?",
        "",
    ],
)
def test_requests_multiple_components_is_false_for_simple_questions(query):
    assert intent_module.requests_multiple_components(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "What were the findings for each diagnostic category?",
        "What were the percentages for each treatment category?",
        "What did the report say about diet, exercise, and medication?",
        "What were the main findings regarding examinations, laboratory tests, and treatments?",
    ],
)
def test_requests_multiple_components_is_true_for_multi_component_questions(query):
    assert intent_module.requests_multiple_components(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Should I take sedatives for my anxiety, and what dose?",
        "Is it safe for me to take vitamins every day?",
        "Should I stop taking warfarin?",
        "What treatment would you recommend for my high blood pressure?",
        "My child has a cough like this patient. What medicine should I give her?",
        "I take metformin - can I also take ibuprofen?",
    ],
)
def test_personal_advice_requests_are_detected(query):
    assert intent_module.is_personal_advice_request(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What medications was the patient discharged on?",
        "Should the patient have been given antibiotics?",
        "What treatment did the doctor recommend?",
        "Compare the selected documents.",
        "What were the most frequent treatment categories?",
    ],
)
def test_questions_about_the_documents_are_not_personal_advice(query):
    assert intent_module.is_personal_advice_request(query) is False


def test_multi_part_beats_numeric():
    query = "How many studies are cited, and what database was used?"
    assert classify_intent(query) == MULTI_PART


def test_comparison_beats_multi_part():
    query = "What do they have in common and what differs between them?"
    assert classify_intent(query) == COMPARISON


def test_empty_query_defaults_to_factual():
    assert classify_intent("") == FACTUAL
    assert classify_intent("   ") == FACTUAL


# --- multi-part decomposition -------------------------------------------------


def test_split_two_part_question():
    parts = split_question_parts(
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?"
    )
    assert len(parts) == 2
    assert parts[0] == "What medications was the patient discharged on"
    assert parts[1].startswith("what condition was found")


def test_split_preserves_leading_qualifier_with_its_part():
    parts = split_question_parts(
        "Besides the imaging, what test did the clinician order at the visit, and "
        "what medications were prescribed at the follow-up?"
    )
    assert len(parts) == 2
    assert "Besides the imaging" in parts[0]
    assert "medications" in parts[1]


def test_ordinary_question_is_one_part():
    query = "What was the patient's chief complaint?"
    assert split_question_parts(query) == [query]
    assert is_multi_part(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "What did the mother and father report?",
        "What was the aneurysm and its rupture site?",
        "Describe the diagnosis and treatment.",
    ],
)
def test_plain_conjunctions_are_not_multi_part(query):
    """A bare "and" joining two noun phrases is one question. Only a
    connector followed by a new interrogative or imperative splits."""
    assert is_multi_part(query) is False


def test_imperative_second_part_splits():
    assert is_multi_part("Summarize the document and list the medications.") is True


def test_split_never_returns_empty():
    assert split_question_parts("") == [""]


# --- decompose_question: follow-up-shaped multi-part questions ----------------
#
# split_question_parts() only ever found parts joined by "and"/"as well as"
# inside one sentence. decompose_question() additionally finds a second
# sentence that is itself a tag question ("X? What about Y?"), and resolves
# an elliptical ordinal reference ("the first one") against the nearest
# available anchor - a preceding part of the same question, or the most
# recent user turn in history for a standalone reference.

from src.generation.intent import decompose_question  # noqa: E402


def test_regression_second_admission_then_what_about_the_first_one():
    """Exact regression case: live testing showed this answered mostly
    about the first admission, with retrieved evidence that did not clearly
    cover both requested parts - a single blended embedding of the whole
    compound question, not two separately-retrieved parts."""
    parts = decompose_question("What happened during the second admission? What about the first one?")
    assert parts == [
        "What happened during the second admission?",
        "What happened during the first admission?",
    ]


def test_tag_question_split_is_multi_part():
    query = "What medication was given during the second visit? What about the first?"
    parts = decompose_question(query)
    assert len(parts) == 2
    assert parts[1] == "What medication was given during the first visit?"


def test_what_about_the_first_alone_resolves_against_history():
    history = [
        {"role": "user", "content": "What happened during the second admission?"},
        {"role": "assistant", "content": "During the second admission, ..."},
    ]
    parts = decompose_question("What about the first?", history=history)
    assert parts == ["What happened during the first admission?"]


def test_what_about_the_first_alone_without_history_is_left_unresolved():
    """No anchor to resolve against - returned as-is rather than guessed at."""
    assert decompose_question("What about the first?") == ["What about the first?"]


def test_ordinal_document_reference_is_not_treated_as_a_question_part():
    """"the second document" is a document reference (see
    src/retrieval/document_reference.py), not an elliptical content
    reference - decompose_question must not try to resolve it as one."""
    assert decompose_question("What about the second document?") == ["What about the second document?"]


def test_one_part_with_evidence_one_without_still_decomposes_both():
    """Decomposition itself doesn't know about evidence - it must produce
    both parts regardless of whether retrieval later finds anything for
    either one (that is retrieve_for_question()'s / build_part_context()'s
    job, covered in their own tests)."""
    parts = decompose_question("What happened during the second admission? What about the first one?")
    assert len(parts) == 2


def test_single_part_question_is_unchanged_by_decomposition():
    query = "What was the patient's chief complaint?"
    assert decompose_question(query) == [query]
    assert decompose_question(query, history=[{"role": "user", "content": "anything"}]) == [query]


def test_plain_and_conjunction_multi_part_still_works_via_decompose():
    """decompose_question must not regress the existing "X, and Y" split."""
    query = (
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?"
    )
    assert decompose_question(query) == split_question_parts(query)


def test_three_way_chain_resolves_each_against_the_nearest_anchor():
    query = "What happened during the second admission? What about the third? What about the first?"
    parts = decompose_question(query)
    assert parts == [
        "What happened during the second admission?",
        "What happened during the third admission?",
        "What happened during the first admission?",
    ]


# --- document balance ---------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Summarize each selected document separately.",
        "Compare the selected documents.",
        "What do these reports have in common?",
        "What is the main purpose of every document?",
    ],
)
def test_balance_requested_for_each_document_and_comparison(query):
    assert needs_document_balance(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What medication was the patient prescribed?",
        "What is the aneurysm size?",
        "Summarize this document.",
        "List all medications the patient received.",
    ],
)
def test_balance_not_requested_for_narrow_questions(query):
    """Including "list all medications": a bare "all" is not a request to
    cover every document, and forcing balanced allocation for it would
    spend the evidence budget on documents the question never asked about."""
    assert needs_document_balance(query) is False


def test_follow_up_that_is_itself_a_comparison_still_balances():
    query = "And compare the other two documents?"
    assert classify_intent(query, history=HISTORY) == FOLLOW_UP
    assert needs_document_balance(query, history=HISTORY) is True


def test_every_intent_constant_is_in_all_intents():
    for name in (FACTUAL, NUMERIC, SUMMARY, EACH_DOCUMENT, COMPARISON, MULTI_PART, FOLLOW_UP, UNSUPPORTED):
        assert name in ALL_INTENTS
    assert set(intent_module.MULTI_DOCUMENT_INTENTS) == {EACH_DOCUMENT, COMPARISON}
