"""Tests for Task 15's factual-enumeration fix:

- intent.is_enumeration_question() - a broader detector than
  requests_multiple_components(), adding a third shape (a most/least-
  frequent RANKING question with no explicit list and no "each"/"every").
- prompt_builder.build_answer_style()'s gate on it (FACTUAL_MULTI_COMPONENT_STYLE
  now also covers the ranking shape, with an added ordering instruction).
- app.resolve_num_predict()'s matching generation-budget widening, so an
  enumeration question that now has the right evidence (Task 13) isn't cut
  off before it finishes using it.

Every other intent's own answer style and budget behavior must be
unaffected - see the "other intents unchanged" sections below.
"""
import pytest

import app
from src import config
from src.generation import intent as intent_module
from src.generation.intent import COMPARISON, EACH_DOCUMENT, FACTUAL, FOLLOW_UP, SUMMARY
from src.generation.prompt_builder import FACTUAL_MULTI_COMPONENT_STYLE, build_answer_style

# --- is_enumeration_question(): the three recognized shapes, plus negatives ---


@pytest.mark.parametrize(
    "query",
    [
        "What was the patient's chief complaint?",
        "What medication was prescribed?",
        "What was the most frequent complaint?",  # singular - one item wanted, not a list
        "",
    ],
)
def test_is_enumeration_question_false_for_simple_questions(query):
    assert intent_module.is_enumeration_question(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "What were the findings for each diagnostic category?",  # each/every
        "What did the report say about diet, exercise, and medication?",  # explicit 3+ list
        "What were the most frequent treatment categories?",  # ranking
        "What were the least common causes of hospital readmission?",  # ranking, unrelated domain
    ],
)
def test_is_enumeration_question_true_for_all_three_shapes(query):
    assert intent_module.is_enumeration_question(query) is True


# --- answer style ---------------------------------------------------------------


def test_simple_factual_question_keeps_the_concise_style():
    style = build_answer_style("What was the patient's chief complaint?", FACTUAL)
    assert "Give the answer directly, in one or two sentences" in style
    assert style != FACTUAL_MULTI_COMPONENT_STYLE


def test_each_every_factual_question_gets_the_enumeration_style():
    style = build_answer_style("What were the findings for each diagnostic category?", FACTUAL)
    assert style == FACTUAL_MULTI_COMPONENT_STYLE


def test_explicit_three_component_factual_question_gets_the_enumeration_style():
    style = build_answer_style(
        "What were the main findings regarding examinations, laboratory tests, and treatments?",
        FACTUAL,
    )
    assert style == FACTUAL_MULTI_COMPONENT_STYLE


def test_most_frequent_ranking_question_gets_the_enumeration_style():
    style = build_answer_style("What were the most frequent treatment categories?", FACTUAL)
    assert style == FACTUAL_MULTI_COMPONENT_STYLE


def test_enumeration_style_asks_for_evidence_supported_ordering():
    assert "order the EVIDENCE's own numbers support" in FACTUAL_MULTI_COMPONENT_STYLE


def test_enumeration_style_forbids_retrieval_process_discussion():
    assert "Never describe the retrieval process" in FACTUAL_MULTI_COMPONENT_STYLE


def test_other_intents_ignore_the_enumeration_signal_and_keep_their_own_style():
    assert "Summarise the content" in build_answer_style(
        "Summarize the most frequent findings for each category.", SUMMARY
    )
    assert "one short section per selected document" in build_answer_style(
        "Summarize each selected document separately.", EACH_DOCUMENT
    )
    assert "concrete similarities first" in build_answer_style(
        "Compare the most frequent findings for skin, trauma, and digestive categories.", COMPARISON
    )
    assert "continues the conversation above" in build_answer_style(
        "What about the most frequent examinations, labs, and treatments?", FOLLOW_UP
    )


# --- generation budget -----------------------------------------------------------


def test_simple_factual_question_gets_no_budget_override():
    assert app.resolve_num_predict("What was the patient's chief complaint?", None, None) is None


def test_budget_is_not_widened_for_every_factual_question():
    """Gated on is_enumeration_question(), not on FACTUAL alone."""
    assert app.resolve_num_predict("What was the size in centimeters of the aneurysm?", None, None) is None


def test_each_every_factual_question_gets_the_larger_budget():
    result = app.resolve_num_predict("What were the findings for each diagnostic category?", None, None)
    assert result == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_explicit_three_component_factual_question_gets_the_larger_budget():
    result = app.resolve_num_predict(
        "What were the main findings regarding examinations, laboratory tests, and treatments?",
        None,
        None,
    )
    assert result == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_most_frequent_ranking_question_gets_the_larger_budget():
    result = app.resolve_num_predict("What were the most frequent treatment categories?", None, None)
    assert result == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_summary_question_budget_is_unaffected_by_the_enumeration_rule():
    """Even worded to contain "most frequent" language, a SUMMARY-classified
    question is not gated into the enumeration budget - is_enumeration_question()
    is only consulted when classify_intent() itself resolves to FACTUAL."""
    assert app.resolve_num_predict("Summarize the most frequent findings for each category.", None, None) is None


def test_each_document_and_comparison_budget_are_unaffected_by_the_enumeration_rule():
    """Neither classifies as FACTUAL, so the new clause never applies to
    them directly - their own established budget behavior (via
    document_inventory, tested elsewhere) is untouched by this change."""
    assert app.resolve_num_predict("Summarize each selected document separately.", None, None) is None
    assert (
        app.resolve_num_predict(
            "Compare the most frequent findings for skin, trauma, and digestive categories.", None, None
        )
        is None
    )


def test_follow_up_budget_is_unaffected_by_the_enumeration_rule():
    history = [{"role": "user", "content": "Summarize each selected document."}]
    assert (
        app.resolve_num_predict("What about the most frequent examinations, labs, and treatments?", history, None)
        is None
    )


def test_pre_existing_multi_part_budget_behavior_is_unaffected():
    """Regression guard: the existing multi_part -> larger-budget path
    (tests/test_production_fixes.py) must still work exactly as before,
    independent of the new enumeration clause."""
    query = "What happened during the second admission? What about the first one?"
    assert app.resolve_num_predict(query, None, None) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


# --- no domain-specific vocabulary in production code ----------------------------


def test_enumeration_style_and_detector_hardcode_no_cdc_facts_or_forced_headings():
    from src.generation import prompt_builder

    style = prompt_builder.FACTUAL_MULTI_COMPONENT_STYLE.lower()
    pattern_source = intent_module._ENUMERATION_RANKING.pattern.lower()
    forbidden = [
        "examinations:",
        "laboratory tests:",
        "treatments:",
        "10.2",
        "9.6",
        "7.5",
        "6.5",
        "6.4",
        "30.5",
        "29.5",
        "12.9",
        "12.4",
        "cdc",
        "diagnostic category",
        "treatment categor",
        "sedative",
        "minor surgery",
    ]
    for term in forbidden:
        assert term not in style
        assert term not in pattern_source
