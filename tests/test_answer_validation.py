"""Tests for src/generation/validation.py - the deterministic
post-generation checker.

The three answers asserted verbatim below are the real failing outputs from
the product QA baseline. They are the reason this module exists, so they are
the regression fixtures: if the validator ever stops flagging them, it has
stopped measuring the thing it was built for.

The validator never rewrites an answer. These tests only assert what it
reports.
"""

import pytest

from src.generation import intent as intent_module
from src.generation.validation import is_refusal, part_is_addressed, validate_answer

SOURCES = ["OMC Report Sample - Cardio.pdf"]


# --- the three baseline failures ---------------------------------------------


def test_flags_refusal_issued_on_top_of_retrieved_evidence():
    """Q15: said with all four pages of that document in the prompt, two of
    which list the discharge medications."""
    answer = (
        "The text does not provide information on the medications the patient was "
        "discharged on or the condition found during the second admission."
    )

    report = validate_answer(answer, intent_module.MULTI_PART, evidence_sources=SOURCES)

    assert report["refusal_detected"] is True
    assert report["refusal_despite_evidence"] is True
    assert "refusal_despite_evidence" in report["issues"]
    assert report["ok"] is False


def test_flags_meta_commentary_about_the_retrieval():
    """Q09: described which pages were available instead of summarising."""
    answer = (
        "Unfortunately, the context provided does not contain a summary of the "
        "document. The context only contains snippets from the document, "
        "specifically pages 3, 5, 6, and 8."
    )

    report = validate_answer(answer, intent_module.SUMMARY, evidence_sources=["RAG_Healthcare_Review1.pdf"])

    assert report["meta_commentary"] is True
    assert "meta_commentary" in report["issues"]
    assert "the context provided" in report["meta_commentary_phrases"]


def test_flags_hedging_written_on_top_of_the_evidence():
    """Q12: the documents' content was in the prompt."""
    answer = (
        "Without seeing the actual content of the reports, it is difficult to say "
        "exactly what they have in common."
    )

    report = validate_answer(
        answer,
        intent_module.COMPARISON,
        evidence_sources=["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
    )

    assert report["hedging"] is True
    assert "hedging_despite_evidence" in report["issues"]


# --- a refusal is not always a defect ----------------------------------------


def test_refusal_with_no_evidence_is_not_flagged():
    """Refusing when nothing was retrieved is the correct behaviour, so it
    must not be counted as a failure."""
    answer = "The documents do not contain that information."

    report = validate_answer(answer, intent_module.FACTUAL, evidence_sources=[])

    assert report["refusal_detected"] is True
    assert report["refusal_despite_evidence"] is False
    assert "refusal_despite_evidence" not in report["issues"]


def test_refusal_on_an_unsupported_question_is_expected_not_flagged():
    answer = "The selected documents do not state the patient's blood type."

    report = validate_answer(answer, intent_module.UNSUPPORTED, evidence_sources=SOURCES)

    assert "refusal_despite_evidence" not in report["issues"]
    assert "unsupported_not_flagged" not in report["issues"]


def test_unsupported_question_answered_anyway_is_flagged():
    """Confidently answering a question the documents cannot support is the
    failure mode for this category."""
    answer = "The patient's blood type was O negative."

    report = validate_answer(answer, intent_module.UNSUPPORTED, evidence_sources=SOURCES)

    assert "unsupported_not_flagged" in report["issues"]


# --- structural coverage ------------------------------------------------------


def test_missing_document_coverage_flagged_for_each_document():
    answer = "Pediatric Report.pdf: the child was seen for a routine checkup."

    report = validate_answer(
        answer,
        intent_module.EACH_DOCUMENT,
        evidence_sources=["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
        required_documents=["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
    )

    assert report["documents_missing"] == ["OMC Report Sample - Cardio.pdf"]
    assert "missing_document_coverage" in report["issues"]


def test_natural_document_references_satisfy_coverage():
    """The validator uses the same semantic document matching as the grader,
    so a correctly structured answer is not flagged for avoiding filenames."""
    answer = (
        "The Pediatric Report covers a routine checkup. The OMC Cardiology "
        "report covers an emergency admission."
    )

    report = validate_answer(
        answer,
        intent_module.EACH_DOCUMENT,
        evidence_sources=["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
        required_documents=["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
    )

    assert report["documents_missing"] == []
    assert "missing_document_coverage" not in report["issues"]


def test_missing_answer_part_flagged_for_multi_part():
    question = (
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?"
    )
    answer = "She was discharged on metformin and empagliflozin."

    report = validate_answer(
        answer, intent_module.MULTI_PART, evidence_sources=SOURCES, question=question
    )

    assert report["parts_expected"] == 2
    assert len(report["parts_missing"]) == 1
    assert "missing_answer_part" in report["issues"]


def test_both_parts_answered_is_not_flagged():
    question = (
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?"
    )
    answer = (
        "1. She was discharged on metformin and empagliflozin. "
        "2. During the second admission a abdominal aortic "
        "aneurysm was found."
    )

    report = validate_answer(
        answer, intent_module.MULTI_PART, evidence_sources=SOURCES, question=question
    )

    assert report["parts_missing"] == []
    assert "missing_answer_part" not in report["issues"]


# --- basics -------------------------------------------------------------------


def test_empty_answer_flagged():
    report = validate_answer("", intent_module.FACTUAL, evidence_sources=SOURCES)
    assert report["empty_answer"] is True
    assert "empty_answer" in report["issues"]
    assert report["ok"] is False


def test_none_answer_is_treated_as_empty():
    report = validate_answer(None, intent_module.FACTUAL, evidence_sources=SOURCES)
    assert report["empty_answer"] is True


def test_clean_answer_reports_ok():
    answer = "The patient was discharged on metformin and empagliflozin [2]."

    report = validate_answer(answer, intent_module.FACTUAL, evidence_sources=SOURCES)

    assert report["issues"] == []
    assert report["ok"] is True
    assert report["answer_chars"] > 0


def test_report_is_json_serialisable_and_flat():
    import json

    report = validate_answer("an answer", intent_module.FACTUAL, evidence_sources=SOURCES)
    json.dumps(report)  # must not raise
    assert isinstance(report["issues"], list)


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The document does not mention that.", True),
        ("There is no information about it.", True),
        # Terse refusals count: this is a real answer the model gave to an
        # unsupported question, and it is exactly the right one.
        ("Not given.", True),
        ("That figure is not reported in the documents.", True),
        ("The aneurysm measured 6.2 cm.", False),
        ("She was discharged on metformin and empagliflozin.", False),
    ],
)
def test_is_refusal(answer, expected):
    assert is_refusal(answer) is expected


def test_terse_refusal_satisfies_the_unsupported_category():
    report = validate_answer("Not given.", intent_module.UNSUPPORTED, evidence_sources=SOURCES)
    assert "unsupported_not_flagged" not in report["issues"]


def test_part_is_addressed_is_a_coverage_check_not_a_correctness_check():
    """A wrong answer that engages with the part still counts as addressed -
    this detects missing sections, nothing more."""
    assert part_is_addressed("She was discharged on aspirin.", "what medications was she discharged on") is True
    assert part_is_addressed("The weather was fine.", "what medications was she discharged on") is False
