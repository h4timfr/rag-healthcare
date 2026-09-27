"""Tests for src/text_match.py - the deterministic term matching the QA
grader and the answer validator share.

The asymmetry is the point and is asserted in both directions: required
terms match morphological variants (a lenient miss is a false failure),
forbidden terms and numbers match exactly (a lenient hit is an invented
failure).
"""

import pytest

from src.text_match import (
    contains_exact,
    contains_term,
    contains_unnegated,
    document_keywords,
    mentions_document,
    stem,
)


# --- the case this exists for -------------------------------------------------


def test_syncopal_satisfies_required_syncope():
    """The product QA baseline scored this correct answer as a miss."""
    answer = "The patient's chief complaint when she presented to the emergency department was a syncopal episode."
    assert contains_term(answer, "syncope") is True
    # ...and the old strict behaviour is still available and still says no.
    assert contains_exact(answer, "syncope") is False


@pytest.mark.parametrize(
    "word,expected",
    [
        ("syncope", "syncop"),
        ("syncopal", "syncop"),
        ("arteries", "arter"),
        ("artery", "arter"),
        ("studies", "stud"),
        ("study", "stud"),
        ("cardiological", "cardiolog"),
        ("doses", "dose"),
        ("retrieval", "retriev"),
        ("retrieving", "retriev"),
    ],
)
def test_stem_normalises_common_variants(word, expected):
    assert stem(word) == expected


def test_stem_never_shortens_below_the_minimum():
    """Short words are left alone rather than being reduced to a stub that
    collides with unrelated words."""
    assert stem("care") == "care"
    assert stem("dose") == "dose"
    assert stem("is") == "is"


def test_stem_leaves_numbers_untouched():
    assert stem("6.2") == "6.2"
    assert stem("3.1") == "3.1"
    assert stem("2006") == "2006"


# --- leniency must not become wrong -------------------------------------------


def test_numbers_require_exact_match():
    """A measured value is matched as a literal string and never
    morphologically - the decimal point and the digits are the fact."""
    assert contains_term("the lesion measured 6.2 cm", "6.2") is True
    # Substring matching is literal, so a needle inside a longer number does
    # match. Required facts are authored as the value that should appear.
    assert contains_term("the reading was 16.2", "6.2") is True
    # A different value never matches, and stemming never softens a number.
    assert contains_term("the lesion measured 6.2 cm", "7.5") is False
    assert contains_term("the value was 62", "6.2") is False
    assert contains_term("the value was 6.25", "6.2") is True


def test_unrelated_short_words_do_not_match():
    assert contains_term("he drove the car", "care") is False
    assert contains_term("the surgeon operated", "surgery") is False


def test_multi_word_terms_must_stay_contiguous():
    assert contains_term("this is the standard of care here", "standard of care") is True
    assert contains_term("the standard was met and the care was good", "standard of care") is False


def test_multi_word_term_tolerates_plural_inside_phrase():
    assert contains_term("these are the standards of care", "standard of care") is True


def test_empty_inputs():
    assert contains_term("some answer", "") is False
    assert contains_term("", "syncope") is False
    assert contains_exact("", "x") is False


# --- document references ------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("OMC Report Sample - Cardio.pdf", ["omc", "cardio"]),
        ("Pediatric Report.pdf", ["pediatric"]),
        ("RAG_Healthcare_Review1.pdf", ["rag", "healthcare"]),
    ],
)
def test_document_keywords_drop_generic_words(source, expected):
    assert document_keywords(source) == expected


def test_document_keywords_falls_back_when_all_words_are_generic():
    """A filename made only of generic words must still be referable."""
    assert document_keywords("report.pdf") == ["report"]


def test_natural_document_reference_counts_as_a_mention():
    """The comparison answer that was graded as unstructured referred to both
    documents correctly - just not by filename."""
    answer = (
        "The Pediatric Report and the OMC Cardiology report both document a "
        "deviation from the standard of care."
    )
    assert mentions_document(answer, "Pediatric Report.pdf") is True
    assert mentions_document(answer, "OMC Report Sample - Cardio.pdf") is True


def test_exact_filename_still_counts_as_a_mention():
    answer = "Pediatric Report.pdf: the child was seen for a routine checkup."
    assert mentions_document(answer, "Pediatric Report.pdf") is True


def test_unmentioned_document_is_not_a_mention():
    answer = "The Pediatric Report describes a routine checkup."
    assert mentions_document(answer, "OMC Report Sample - Cardio.pdf") is False
    assert mentions_document(answer, "RAG_Healthcare_Review1.pdf") is False


def test_one_generic_keyword_alone_does_not_credit_a_document():
    """In a corpus of healthcare documents, "healthcare" appears in most
    answers. Crediting a document that was never discussed would make the
    each-document coverage check meaningless."""
    answer = "This healthcare document describes a routine checkup."

    assert mentions_document(answer, "RAG_Healthcare_Review1.pdf") is False


def test_all_identifying_words_are_required():
    assert mentions_document("the RAG healthcare review covers retrieval", "RAG_Healthcare_Review1.pdf") is True
    assert mentions_document("the RAG review covers retrieval", "RAG_Healthcare_Review1.pdf") is False


def test_partial_cardio_reference_alone_is_not_enough():
    answer = "The cardiology findings included an aneurysm."
    assert mentions_document(answer, "OMC Report Sample - Cardio.pdf") is False


def test_mentions_document_handles_missing_source():
    assert mentions_document("anything", "") is False


# --- contains_unnegated: forbidden-phrase grading that a denial does not trip ---
#
# Phase 8 regression: the product QA grader's forbidden_terms check used
# contains_exact(), which cannot tell "X" from "X is not true" - Q19's real
# forbidden term "blood type is" flagged the CORRECT refusal "The patient's
# blood type is not mentioned in the provided evidence." as if it were the
# hallucinated "The patient's blood type is A+." it exists to catch.


def test_positive_forbidden_phrase_is_flagged():
    assert contains_unnegated("The patient's blood type is A+.", "blood type is") is True


def test_negated_forbidden_phrase_is_not_flagged():
    assert contains_unnegated("The patient's blood type is not mentioned.", "blood type is") is False


def test_negation_further_away_does_not_suppress_a_real_hit():
    """The negation cue must be genuinely NEAR the phrase - a "not" from an
    unrelated earlier clause must not exempt a real assertion later on."""
    answer = "The chart does not discuss diet at all. Separately, the patient's blood type is A+."
    assert contains_unnegated(answer, "blood type is") is True


def test_partial_phrase_does_not_match():
    """A forbidden multi-word phrase must appear whole - a subset of its
    words appearing elsewhere is not the phrase."""
    assert contains_unnegated("The patient's blood type was recorded.", "blood type is") is False


def test_quoted_phrase_is_still_flagged():
    """Quoting the phrase back does not make a hallucinated fact safe -
    only an actual negation nearby does."""
    assert contains_unnegated('The record states "blood type is O negative".', "blood type is") is True


def test_unrelated_occurrence_of_a_numeric_forbidden_term_still_matches():
    """A known, unchanged limitation: like contains_exact(), this is still a
    substring match - a forbidden number appearing in an unrelated context
    is not distinguished from a real hit. Documented here as expected
    current behavior, not something this function claims to solve."""
    assert contains_unnegated("Rated 5.0 out of 5 stars.", "5.0") is True


def test_every_occurrence_negated_is_not_flagged():
    """More than one occurrence of the phrase, ALL of them denied."""
    answer = "Blood type is not recorded. The chart does not say the blood type is known either."
    assert contains_unnegated(answer, "blood type is") is False


def test_one_unnegated_occurrence_among_several_is_flagged():
    """If even one occurrence is a real, unnegated claim, that is a genuine
    violation regardless of other, negated occurrences nearby."""
    answer = "The old chart said the blood type is unknown, but the new record says the blood type is A+."
    assert contains_unnegated(answer, "blood type is") is True


def test_empty_needle_or_haystack():
    assert contains_unnegated("", "blood type is") is False
    assert contains_unnegated("The blood type is A+.", "") is False
