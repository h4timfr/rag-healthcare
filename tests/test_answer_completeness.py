"""Regression tests for three narrowly gated answer-completeness changes,
each found by manual QA on a two-column clinical write-up and confirmed by a
controlled ablation on the same retrieved evidence:

1. A single-document comparison of two PARTS of the document (an initial vs a
   revised problem list) gets a section-identity answer style and the larger
   output budget. A concept comparison keeps the original style and budget -
   the section style broke concept answers in 2 of 3 runs.
2. Multi-part answers are told to include every item each part's evidence
   gives, not the first few.
3. A question asking for a number of items ("give 3-5 reasons") gets the
   existing list-style instructions and budget instead of "one or two
   sentences".

None of these touch retrieval, and the research evaluation prompt
(eval_prompt_builder) is asserted unchanged.
"""

import pytest

import app
from src import config
from src.generation import intent as intent_module
from src.generation.eval_prompt_builder import build_eval_prompt
from src.generation.prompt_builder import (
    ANSWER_STYLES,
    FACTUAL_MULTI_COMPONENT_STYLE,
    MEDICATION_ONLY_RULE,
    MULTI_PART_EXHAUSTIVE_RULE,
    SINGLE_DOCUMENT_COMPARISON_STYLE,
    build_prompt,
)

CHUNKS = [
    {"chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 4, "text": "Initial Problem List 1. Chest pain", "score": 1.0, "rank": 1},
    {"chunk_id": 2, "document_id": "d1", "source": "a.pdf", "page": 4, "text": "Revised Problem List 1. Chest pain", "score": 0.5, "rank": 2},
]
INVENTORY_2 = [
    {"document_id": "d1", "source": "Pediatric Report.pdf", "page_count": 3},
    {"document_id": "d2", "source": "OMC Report Sample - Cardio.pdf", "page_count": 4},
]

SECTION_COMPARISON = (
    "Compare the patient's initial problem list with the revised problem list. What changed between "
    "the two lists? Give the items that were added, removed, and renamed/reworded."
)
CONCEPT_COMPARISON = "What is the key difference between traditional LLM use and RAG?"
COUNT_REQUEST = (
    "Why does the document conclude that unstable angina is the most likely diagnosis rather than GERD "
    "or a pulmonary infection? Give 3-5 specific reasons from the document."
)
MULTI_PART = (
    "What medications had the patient previously taken, what happened when she took them, and what "
    "medications were recommended in the current treatment plan?"
)


def _production(query, chunks=CHUNKS):
    return build_prompt(query, chunks, disable_benchmark_scope=True)


# --- 1. single-document section comparison ---------------------------------------


@pytest.mark.parametrize(
    "query",
    [SECTION_COMPARISON, "How does the first section differ from the second section?", "Compare the two versions of the plan."],
)
def test_comparisons_of_document_parts_are_detected(query):
    assert intent_module.compares_document_sections(query)


@pytest.mark.parametrize(
    "query",
    [CONCEPT_COMPARISON, "How does keyword retrieval differ from embedding retrieval?",
     "How much more likely are African Americans to develop end-stage renal disease compared to whites?"],
)
def test_concept_comparisons_are_not_section_comparisons(query):
    assert not intent_module.compares_document_sections(query)


def test_section_comparison_gets_the_section_identity_style():
    prompt = _production(SECTION_COMPARISON)
    assert SINGLE_DOCUMENT_COMPARISON_STYLE in prompt
    assert ANSWER_STYLES[intent_module.COMPARISON] not in prompt
    assert prompt.index(SINGLE_DOCUMENT_COMPARISON_STYLE) < prompt.index("USER QUESTION:")


def test_concept_comparison_keeps_the_original_style():
    prompt = _production(CONCEPT_COMPARISON)
    assert ANSWER_STYLES[intent_module.COMPARISON] in prompt
    assert SINGLE_DOCUMENT_COMPARISON_STYLE not in prompt


def test_multi_document_comparison_prompt_is_unchanged():
    prompt = build_prompt(SECTION_COMPARISON, CHUNKS, document_inventory=INVENTORY_2)
    assert SINGLE_DOCUMENT_COMPARISON_STYLE not in prompt
    assert ANSWER_STYLES[intent_module.COMPARISON] in prompt


def test_section_comparison_gets_the_larger_budget_but_a_concept_comparison_does_not():
    """The section comparison was cut off at the default 256 tokens."""
    assert app.resolve_num_predict(SECTION_COMPARISON, None, None) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
    assert app.resolve_num_predict(CONCEPT_COMPARISON, None, None) is None


# --- 2. multi-part exhaustiveness --------------------------------------------------


def test_multi_part_plan_requires_every_item_per_part():
    prompt = _production(MULTI_PART)
    assert "This question has 2 parts" in prompt
    assert MULTI_PART_EXHAUSTIVE_RULE in prompt


def test_single_part_prompt_does_not_get_the_multi_part_rule():
    assert MULTI_PART_EXHAUSTIVE_RULE not in _production("What is the patient's age?")


def test_multi_part_medication_question_is_limited_to_drugs():
    """With "include every item", one medication answer in eight listed lab
    tests and a ventriculogram as medications."""
    prompt = _production(MULTI_PART)
    assert MEDICATION_ONLY_RULE in prompt
    assert prompt.index(MULTI_PART_EXHAUSTIVE_RULE) < prompt.index(MEDICATION_ONLY_RULE) < prompt.index("USER QUESTION:")


def test_multi_part_question_about_something_else_gets_no_medication_rule():
    prompt = _production("What happened during the first admission, and what was found on the scan?")
    assert MULTI_PART_EXHAUSTIVE_RULE in prompt
    assert MEDICATION_ONLY_RULE not in prompt


@pytest.mark.parametrize(
    "query,expected",
    [("What medications was she discharged on?", True), ("Which drugs were prescribed?", True),
     ("What tests were ordered?", False), ("What was the diet plan?", False)],
)
def test_asks_for_medications(query, expected):
    assert intent_module.asks_for_medications(query) is expected


def test_section_comparison_treats_rewording_as_the_same_item():
    """A shared item ("FH of early ASCVD") was reported as added; renamed
    items were reported as remove-plus-add."""
    prompt = _production(SECTION_COMPARISON)
    assert "renamed or rephrased, treat it as the same item" in prompt
    assert "renamed or rephrased" not in _production(CONCEPT_COMPARISON)


# --- 3. explicit item counts ------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [COUNT_REQUEST, "List two causes of the readmission.", "Name 3 risk factors mentioned in the report.",
     "Give five findings from the examination.",
     # Also a request for several items (one date per admission), so the
     # list style is the right answer shape here too.
     "Give the date of the 2 admissions."],
)
def test_explicit_item_count_requests_are_detected(query):
    assert intent_module.requests_item_count(query)


@pytest.mark.parametrize(
    "query",
    ["What is the patient's age?", "Give 1 reason for the admission.", "What was the patient's blood pressure?",
     "What happened in 2004?"],
)
def test_ordinary_questions_are_not_count_requests(query):
    assert not intent_module.requests_item_count(query)


def test_count_request_gets_the_list_style_not_one_or_two_sentences():
    prompt = _production(COUNT_REQUEST)
    assert FACTUAL_MULTI_COMPONENT_STYLE in prompt
    assert ANSWER_STYLES[intent_module.FACTUAL] not in prompt


def test_count_request_gets_the_larger_budget():
    assert app.resolve_num_predict(COUNT_REQUEST, None, None) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_plain_factual_question_is_unchanged():
    prompt = _production("What is the patient's age?")
    assert ANSWER_STYLES[intent_module.FACTUAL] in prompt
    assert app.resolve_num_predict("What is the patient's age?", None, None) is None


# --- the research evaluation prompt is untouched ----------------------------------


@pytest.mark.parametrize("query", [SECTION_COMPARISON, COUNT_REQUEST, MULTI_PART])
def test_research_evaluation_prompt_carries_none_of_these_changes(query):
    prompt = build_eval_prompt(query, CHUNKS)
    for added in (SINGLE_DOCUMENT_COMPARISON_STYLE, MULTI_PART_EXHAUSTIVE_RULE, FACTUAL_MULTI_COMPONENT_STYLE):
        assert added not in prompt
