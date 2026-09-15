"""Validates data/healthcare_eval_questions.json: structure, balance across
documents/categories, ground truth traceable to a real corpus document, and
that no question is actually about the RAG software itself.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "healthcare_eval_questions.json"
CORPUS_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_corpus.json"

REQUIRED_FIELDS = {"id", "question", "source", "document_id", "expected_pages", "expected_terms", "category"}
VALID_CATEGORIES = {
    "factual_extraction", "numeric_detail", "section_retrieval",
    "terminology_semantic", "multi_part",
}
FORBIDDEN_TERMS = [
    "faiss", "rag ", "rag-", "rag.", "chunking", "chunk size", "embedding model",
    "ollama", "streamlit", "retrieval-augmented", "vector database",
    "hybrid retrieval", "dense retrieval", "sqlite",
]


@pytest.fixture(scope="module")
def questions():
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus_manifest():
    with CORPUS_MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifest_by_source(corpus_manifest):
    lookup = {}
    for doc in corpus_manifest:
        if doc["filename"]:
            lookup[doc["filename"]] = doc
        if doc["title"]:
            lookup[doc["title"]] = doc
    return lookup


def test_question_count_is_approximately_45(questions):
    assert 40 <= len(questions) <= 50


def test_no_duplicate_ids(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_no_duplicate_question_text(questions):
    texts = [q["question"].strip().lower() for q in questions]
    assert len(texts) == len(set(texts))


def test_all_required_fields_present(questions):
    for q in questions:
        missing = REQUIRED_FIELDS - q.keys()
        assert not missing, f"{q.get('id')} missing fields: {missing}"


def test_expected_pages_and_terms_are_non_empty(questions):
    for q in questions:
        assert q["expected_pages"], f"{q['id']} has no expected_pages"
        assert q["expected_terms"], f"{q['id']} has no expected_terms"


def test_all_categories_are_valid(questions):
    for q in questions:
        assert q["category"] in VALID_CATEGORIES, f"{q['id']} has unknown category {q['category']!r}"


def test_category_distribution_is_balanced(questions):
    counts = Counter(q["category"] for q in questions)
    assert set(counts) == VALID_CATEGORIES, "every category must be represented"
    total = len(questions)
    for category, n in counts.items():
        assert n / total <= 0.35, f"category {category!r} is {n}/{total} of all questions - too dominant"


def test_per_document_distribution_is_reasonable(questions):
    counts = Counter(q["source"] for q in questions)
    assert len(counts) == 5, "expected exactly 5 distinct source documents"
    for source, n in counts.items():
        assert 5 <= n <= 12, f"{source} has {n} questions - outside the ~8-10/document target range"


def test_source_matches_a_known_corpus_document(questions, manifest_by_source):
    for q in questions:
        assert q["source"] in manifest_by_source, (
            f"{q['id']} source {q['source']!r} is not in data/evaluation_corpus.json"
        )


def test_document_id_matches_manifest(questions, manifest_by_source):
    for q in questions:
        doc = manifest_by_source[q["source"]]
        assert q["document_id"] == doc["document_id"], (
            f"{q['id']} document_id {q['document_id']!r} != manifest's {doc['document_id']!r} for {q['source']!r}"
        )


def test_expected_pages_within_document_page_count(questions, manifest_by_source):
    for q in questions:
        max_page = manifest_by_source[q["source"]]["page_count"]
        for page in q["expected_pages"]:
            assert 1 <= page <= max_page, (
                f"{q['id']} expected page {page} out of range for {q['source']} (has {max_page} pages)"
            )


def test_no_questions_about_the_rag_software_itself(questions):
    for q in questions:
        text = q["question"].lower()
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"{q['id']} mentions forbidden term {term!r}: {q['question']}"
