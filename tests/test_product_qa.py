"""Tests for the production-only answer-quality QA harness
(src/evaluation/product_qa.py, scripts/run_product_qa.py, data/product_qa.json).

Never calls a real Ollama server: every test that exercises run_turn()/
run_case() monkeypatches product_qa.generate with a deterministic fake, and
uses an isolated tmp_path SQLite/FAISS index (never the real production
data/metadata.db) so this suite stays hermetic and fast, consistent with
every other test file in this project.
"""

import numpy as np
import pymupdf
import pytest

from src import config
from src.evaluation import product_qa
from src.exceptions import RagError


# --- dataset structural validity ---------------------------------------------


def test_real_dataset_loads_and_has_expected_shape():
    dataset = product_qa.load_dataset()
    assert "documents" in dataset
    assert "cases" in dataset
    assert len(dataset["cases"]) > 0


def test_real_dataset_covers_all_eight_categories():
    dataset = product_qa.load_dataset()
    categories = {case["category"] for case in dataset["cases"]}
    assert categories == set(product_qa.REQUIRED_CATEGORIES)


def test_real_dataset_case_ids_are_unique():
    dataset = product_qa.load_dataset()
    ids = [case["id"] for case in dataset["cases"]]
    assert len(ids) == len(set(ids))


def test_real_dataset_every_case_has_required_fields():
    dataset = product_qa.load_dataset()
    for case in dataset["cases"]:
        assert case["id"]
        assert case["category"] in product_qa.REQUIRED_CATEGORIES
        assert isinstance(case["selected_document_sources"], list) and case["selected_document_sources"]
        assert isinstance(case["turns"], list) and case["turns"]
        for turn in case["turns"]:
            assert turn["question"]
            assert "expected_characteristics" in turn
            assert "required_sources" in turn
            assert "required_facts" in turn
            assert "unacceptable_behavior" in turn


def test_real_dataset_follow_up_cases_have_multiple_turns():
    dataset = product_qa.load_dataset()
    follow_ups = [c for c in dataset["cases"] if c["category"] == "follow_up"]
    assert follow_ups
    for case in follow_ups:
        assert len(case["turns"]) >= 2


def test_real_dataset_each_document_and_comparison_cases_select_multiple_documents():
    dataset = product_qa.load_dataset()
    for case in dataset["cases"]:
        if case["category"] in ("each_document", "comparison"):
            assert len(case["selected_document_sources"]) >= 2


def test_real_dataset_unsupported_cases_have_no_required_facts():
    """Unsupported questions have no correct factual answer by design -
    grading them relies on refusal detection, not fact coverage."""
    dataset = product_qa.load_dataset()
    for case in dataset["cases"]:
        if case["category"] == "unsupported":
            for turn in case["turns"]:
                assert turn["required_facts"] == []


def test_real_dataset_referenced_sources_match_documents_list():
    dataset = product_qa.load_dataset()
    known_sources = {d["source"] for d in dataset["documents"]}
    for case in dataset["cases"]:
        for source in case["selected_document_sources"]:
            assert source in known_sources


# --- evaluate_turn() pure grading logic ---------------------------------------


def test_evaluate_turn_passes_with_full_fact_and_document_coverage():
    spec = {"required_facts": ["cough", "runs"], "required_sources": ["a.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "The mother complained that the child coughs when he runs.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }
    metrics = product_qa.evaluate_turn(spec, result, "factual")
    assert metrics["fact_coverage"] == 1.0
    assert metrics["document_coverage"] == 1.0
    assert metrics["passed"] is True


def test_evaluate_turn_fails_on_missing_required_facts():
    spec = {"required_facts": ["cough", "runs"], "required_sources": [], "forbidden_terms": []}
    result = {"generated_ok": True, "answer": "The mother complained about a fever.", "sources": []}
    metrics = product_qa.evaluate_turn(spec, result, "factual")
    assert metrics["fact_coverage"] == 0.0
    assert metrics["passed"] is False


def test_evaluate_turn_fails_on_forbidden_term_present():
    spec = {"required_facts": ["6.2"], "required_sources": [], "forbidden_terms": ["3.1"]}
    result = {"generated_ok": True, "answer": "The aneurysm measured 3.1 cm.", "sources": []}
    metrics = product_qa.evaluate_turn(spec, result, "numeric")
    assert metrics["forbidden_term_hit"] is True
    assert metrics["passed"] is False


def test_evaluate_turn_does_not_flag_a_negated_forbidden_phrase():
    """Regression: Q19's real forbidden_terms include "blood type is" (to
    catch a hallucinated "The patient's blood type is A+"), but the CORRECT
    refusal "The patient's blood type is not mentioned in the provided
    evidence." contains that exact substring too - it must not be flagged."""
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": ["blood type is"]}
    result = {
        "generated_ok": True,
        "answer": "The patient's blood type is not mentioned in the provided evidence.",
        "sources": [],
    }
    metrics = product_qa.evaluate_turn(spec, result, "unsupported")
    assert metrics["forbidden_term_hit"] is False
    assert metrics["forbidden_terms_found"] == []


def test_evaluate_turn_still_flags_the_same_phrase_when_actually_asserted():
    """The negation fix must not become a blanket exemption - the identical
    phrase, genuinely asserted (no negation nearby), still fails."""
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": ["blood type is"]}
    result = {"generated_ok": True, "answer": "The patient's blood type is A+.", "sources": []}
    metrics = product_qa.evaluate_turn(spec, result, "unsupported")
    assert metrics["forbidden_term_hit"] is True
    assert metrics["forbidden_terms_found"] == ["blood type is"]


def test_evaluate_turn_fails_when_generation_failed():
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": []}
    result = {"generated_ok": False, "answer": None, "sources": []}
    metrics = product_qa.evaluate_turn(spec, result, "factual")
    assert metrics["passed"] is False


def test_evaluate_turn_grading_is_case_insensitive_and_substring_based():
    """Do not require exact wording - a fact/source hit anywhere in the
    answer text, any case, must count."""
    spec = {"required_facts": ["SYNCOPE"], "required_sources": ["a.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "She presented with syncope after a fall.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }
    metrics = product_qa.evaluate_turn(spec, result, "factual")
    assert metrics["fact_coverage"] == 1.0


def test_evaluate_turn_each_document_structure_ok_when_all_named():
    spec = {
        "required_facts": [],
        "required_sources": ["a.pdf", "b.pdf"],
        "forbidden_terms": [],
    }
    result = {
        "generated_ok": True,
        "answer": "a.pdf: summary one.\n\nb.pdf: summary two.",
        "sources": [{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 1}],
    }
    metrics = product_qa.evaluate_turn(spec, result, "each_document")
    assert metrics["structure_ok"] is True
    assert metrics["passed"] is True


def test_evaluate_turn_each_document_structure_not_ok_when_one_document_skipped():
    spec = {"required_facts": [], "required_sources": ["a.pdf", "b.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "a.pdf: summary one.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }
    metrics = product_qa.evaluate_turn(spec, result, "each_document")
    assert metrics["structure_ok"] is False
    assert metrics["passed"] is False


def test_evaluate_turn_unsupported_refusal_correct_when_model_refuses():
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": ["type A", "type B"]}
    result = {
        "generated_ok": True,
        "answer": "The context does not mention the patient's blood type.",
        "sources": [],
    }
    metrics = product_qa.evaluate_turn(spec, result, "unsupported")
    assert metrics["refusal_detected"] is True
    assert metrics["refusal_correct"] is True
    assert metrics["passed"] is True


def test_evaluate_turn_unsupported_fails_when_model_invents_an_answer():
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": ["type a"]}
    result = {"generated_ok": True, "answer": "The patient's blood type was Type A.", "sources": []}
    metrics = product_qa.evaluate_turn(spec, result, "unsupported")
    assert metrics["forbidden_term_hit"] is True
    assert metrics["refusal_correct"] is False
    assert metrics["passed"] is False


# --- evaluator fixes: morphology, document references, validation -------------
#
# These are the grading artifacts the first baseline produced - correct
# answers scored as failures. Each one is asserted against the real answer
# text the model generated, so the grader cannot regress into measuring its
# own suffix handling again.


def test_morphological_variant_satisfies_a_required_fact():
    """Q02's real answer. "a syncopal episode" is a correct report of a
    chief complaint of syncope; the first grader scored it 0%."""
    spec = {"required_facts": ["syncope"], "required_sources": ["a.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "The patient's chief complaint when she presented to the emergency "
                  "department was a syncopal episode.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }

    metrics = product_qa.evaluate_turn(spec, result, "factual")

    assert metrics["fact_coverage"] == 1.0
    assert metrics["passed"] is True
    # ...and the leniency is recorded, not hidden.
    assert metrics["fact_hits_morphological_only"] == ["syncope"]


def test_exact_fact_hits_are_not_reported_as_morphological():
    spec = {"required_facts": ["syncope"], "required_sources": [], "forbidden_terms": []}
    result = {"generated_ok": True, "answer": "She presented with syncope.", "sources": []}

    metrics = product_qa.evaluate_turn(spec, result, "factual")

    assert metrics["fact_hits"] == ["syncope"]
    assert metrics["fact_hits_morphological_only"] == []


def test_forbidden_terms_stay_strict_and_numeric():
    """Leniency in this direction would invent a failure. A forbidden "3.1"
    must not fire on a correct "6.2", and must not be softened by stemming."""
    spec = {"required_facts": ["6.2"], "required_sources": [], "forbidden_terms": ["3.1", "7.5"]}
    result = {
        "generated_ok": True,
        "answer": "A abdominal aortic aneurysm measuring 6.2 cm was found.",
        "sources": [{"source": "a.pdf", "page": 2}],
    }

    metrics = product_qa.evaluate_turn(spec, result, "numeric")

    assert metrics["forbidden_term_hit"] is False
    assert metrics["fact_coverage"] == 1.0
    assert metrics["passed"] is True


def test_natural_document_reference_counts_as_structure():
    """Q12's real answer referred to both documents correctly, just not by
    filename, and was graded as having no document sections at all."""
    spec = {
        "required_facts": [],
        "required_sources": ["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
        "forbidden_terms": [],
    }
    result = {
        "generated_ok": True,
        "answer": "The Pediatric Report and the OMC Cardiology report both document a "
                  "deviation from the standard of care.",
        "sources": [
            {"source": "Pediatric Report.pdf", "page": 1},
            {"source": "OMC Report Sample - Cardio.pdf", "page": 2},
        ],
    }

    metrics = product_qa.evaluate_turn(spec, result, "comparison")

    assert metrics["structure_ok"] is True
    assert len(metrics["documents_named_in_answer"]) == 2
    # The stricter signal is still reported alongside it.
    assert metrics["documents_named_exact_filename"] == []


def test_a_genuinely_skipped_document_is_still_a_structure_failure():
    """The relaxation must not make the check vacuous."""
    spec = {
        "required_facts": [],
        "required_sources": ["Pediatric Report.pdf", "OMC Report Sample - Cardio.pdf"],
        "forbidden_terms": [],
    }
    result = {
        "generated_ok": True,
        "answer": "The Pediatric Report describes a routine checkup.",
        "sources": [
            {"source": "Pediatric Report.pdf", "page": 1},
            {"source": "OMC Report Sample - Cardio.pdf", "page": 2},
        ],
    }

    metrics = product_qa.evaluate_turn(spec, result, "comparison")

    assert metrics["structure_ok"] is False
    assert metrics["passed"] is False


def test_evaluate_turn_exposes_validation_results():
    """Phase 7: the validator's findings travel with the metrics so a
    failure can be attributed instead of argued about."""
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": [],
            "question": "What medications was she discharged on?"}
    result = {
        "generated_ok": True,
        "answer": "The text does not provide information on the medications.",
        "sources": [{"source": "a.pdf", "page": 2}],
    }

    metrics = product_qa.evaluate_turn(spec, result, "factual")

    assert metrics["validation_ok"] is False
    assert "refusal_despite_evidence" in metrics["validation_issues"]
    assert metrics["validation"]["evidence_document_count"] == 1


def test_validation_flags_meta_commentary_on_a_summary():
    spec = {"required_facts": [], "required_sources": [], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "The context provided does not contain a summary of the document.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }

    metrics = product_qa.evaluate_turn(spec, result, "summary")

    assert "meta_commentary" in metrics["validation_issues"]


def test_clean_answer_reports_no_validation_issues():
    spec = {"required_facts": ["metformin"], "required_sources": ["a.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "She was discharged on metformin and empagliflozin [2].",
        "sources": [{"source": "a.pdf", "page": 2}],
    }

    metrics = product_qa.evaluate_turn(spec, result, "factual")

    assert metrics["validation_ok"] is True
    assert metrics["validation_issues"] == []
    assert metrics["passed"] is True


def test_grading_is_deterministic():
    spec = {"required_facts": ["syncope"], "required_sources": ["a.pdf"], "forbidden_terms": ["3.1"]}
    result = {
        "generated_ok": True,
        "answer": "She had a syncopal episode.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }

    first = product_qa.evaluate_turn(spec, result, "factual")
    second = product_qa.evaluate_turn(spec, result, "factual")

    assert first == second


def test_metrics_stay_json_serialisable():
    """run_product_qa.py writes these straight to disk for manual
    inspection - a metric that cannot be serialised loses the raw record."""
    import json

    spec = {"required_facts": ["syncope"], "required_sources": ["a.pdf"], "forbidden_terms": []}
    result = {
        "generated_ok": True,
        "answer": "She had a syncopal episode.",
        "sources": [{"source": "a.pdf", "page": 1}],
    }

    json.dumps(product_qa.evaluate_turn(spec, result, "factual"))


# --- resolve_document_ids() ---------------------------------------------------


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    return tmp_path


def _fake_embed_texts(texts):
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors, dtype="float32")


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def two_ingested_docs(isolated_index, monkeypatch):
    import scripts.ingest as ingest_script

    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_a, pdf_b = isolated_index / "a.pdf", isolated_index / "b.pdf"
    _make_pdf(pdf_a, "Diabetes management guideline about insulin therapy.")
    _make_pdf(pdf_b, "Cardiology report about coronary arteries.")

    import app

    result_a = app.ingest_uploaded_pdf("a.pdf", pdf_a.read_bytes())
    result_b = app.ingest_uploaded_pdf("b.pdf", pdf_b.read_bytes())
    return {"a.pdf": result_a["document_id"], "b.pdf": result_b["document_id"]}


def test_resolve_document_ids_resolves_known_sources(two_ingested_docs):
    resolved = product_qa.resolve_document_ids(["a.pdf", "b.pdf"])
    assert resolved == two_ingested_docs


def test_resolve_document_ids_raises_with_missing_source_named(two_ingested_docs):
    with pytest.raises(RagError, match="c.pdf"):
        product_qa.resolve_document_ids(["a.pdf", "c.pdf"])


# --- run_turn() / run_case() using the real production pipeline functions ----


def test_run_turn_single_document_uses_active_scope_path(two_ingested_docs, monkeypatch):
    from src.retrieval import hybrid_retriever

    vec = _fake_embed_texts(["insulin therapy"])[0]
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vec)
    monkeypatch.setattr(product_qa, "generate", lambda prompt, num_predict=None: "a real answer")

    result = product_qa.run_turn("What is this about?", ["a.pdf"], None, two_ingested_docs)

    assert result["generated_ok"] is True
    assert result["answer"] == "a real answer"
    assert all(s["source"] == "a.pdf" for s in result["sources"])
    assert result["latency_ms"] >= 0.0


def test_run_turn_multi_document_uses_selected_scope_and_widened_top_k(two_ingested_docs, monkeypatch):
    from src.retrieval import hybrid_retriever

    vec = _fake_embed_texts(["overview"])[0]
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vec)

    captured = {}

    def fake_generate(prompt, num_predict=None):
        captured["num_predict"] = num_predict
        captured["prompt"] = prompt
        return "a multi-document answer"

    monkeypatch.setattr(product_qa, "generate", fake_generate)

    result = product_qa.run_turn(
        "Summarize each document.", ["a.pdf", "b.pdf"], None, two_ingested_docs
    )

    assert result["generated_ok"] is True
    # Multi-document generation must use the larger configured budget.
    assert captured["num_predict"] == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
    # And the prompt must be the multi-document, document-grouped one.
    assert "MULTI-DOCUMENT RULES" in captured["prompt"]


def test_run_turn_reports_insufficient_retrieval_as_a_graceful_error(two_ingested_docs, monkeypatch):
    # app.py does `from ... import retrieve_hybrid`, binding its own name -
    # patch that name (as production_app.retrieve_hybrid), not the
    # hybrid_retriever module's, or app.py's call site won't see it.
    monkeypatch.setattr(product_qa.production_app, "retrieve_hybrid", lambda *a, **k: [])
    monkeypatch.setattr(product_qa, "generate", lambda prompt, num_predict=None: "unused")

    result = product_qa.run_turn("anything", ["a.pdf"], None, two_ingested_docs)

    assert result["generated_ok"] is False
    assert result["error_type"] == "InsufficientRetrievalError"


def test_run_case_threads_history_between_turns(two_ingested_docs, monkeypatch):
    from src.retrieval import hybrid_retriever

    vec = _fake_embed_texts(["insulin therapy"])[0]
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vec)

    captured_histories = []

    def fake_generate(prompt, num_predict=None):
        return "answer text"

    # Capture the REAL function before patching - re-importing it from
    # inside the wrapper would just look up the already-patched (fake)
    # attribute again and recurse forever.
    real_retrieve_and_build_prompt = product_qa.production_app.retrieve_and_build_prompt

    def fake_retrieve_and_build_prompt(query, **kwargs):
        captured_histories.append(kwargs.get("history"))
        return real_retrieve_and_build_prompt(query, **kwargs)

    monkeypatch.setattr(product_qa, "generate", fake_generate)
    monkeypatch.setattr(product_qa.production_app, "retrieve_and_build_prompt", fake_retrieve_and_build_prompt)

    case = {
        "id": "T01",
        "category": "follow_up",
        "selected_document_sources": ["a.pdf"],
        "turns": [
            {"question": "first question", "required_facts": [], "required_sources": [], "forbidden_terms": []},
            {"question": "second question", "required_facts": [], "required_sources": [], "forbidden_terms": []},
        ],
    }
    product_qa.run_case(case, two_ingested_docs)

    assert captured_histories[0] is None
    assert captured_histories[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer text"},
    ]


def test_run_case_retrieval_query_is_never_history_embedded(two_ingested_docs, monkeypatch):
    from src.retrieval import hybrid_retriever

    vec = _fake_embed_texts(["insulin therapy"])[0]
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vec)
    monkeypatch.setattr(product_qa, "generate", lambda prompt, num_predict=None: "answer text")

    captured_queries = []
    # Spy on app.py's own bound name (see the note in the insufficient-
    # retrieval test above), not the hybrid_retriever module's.
    real_retrieve_hybrid = product_qa.production_app.retrieve_hybrid

    def spy_retrieve_hybrid(query, **kwargs):
        captured_queries.append(query)
        return real_retrieve_hybrid(query, **kwargs)

    monkeypatch.setattr(product_qa.production_app, "retrieve_hybrid", spy_retrieve_hybrid)

    case = {
        "id": "T02",
        "category": "follow_up",
        "selected_document_sources": ["a.pdf"],
        "turns": [
            {"question": "unrelated earlier topic", "required_facts": [], "required_sources": [], "forbidden_terms": []},
            {"question": "the clean follow-up question", "required_facts": [], "required_sources": [], "forbidden_terms": []},
        ],
    }
    product_qa.run_case(case, two_ingested_docs)

    assert captured_queries[1] == "the clean follow-up question"
    assert "unrelated earlier topic" not in captured_queries[1]


# --- aggregate() / _percentile() / worst_failures() ---------------------------


def test_percentile_matches_known_values():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert product_qa._percentile(values, 0.5) == 30.0
    assert product_qa._percentile([], 0.5) == 0.0
    assert product_qa._percentile([42.0], 0.9) == 42.0


def _fake_turn(passed, latency_ms, fact_coverage=None, document_coverage=None, source_attr=True):
    return {
        "spec": {"question": "q"},
        "result": {"latency_ms": latency_ms, "answer": "a"},
        "metrics": {
            "passed": passed,
            "generated_ok": True,
            "fact_coverage": fact_coverage,
            "document_coverage": document_coverage,
            "source_attribution_present": source_attr,
        },
    }


def test_aggregate_computes_overall_and_per_category_metrics():
    case_results = [
        {"id": "Q1", "category": "factual", "turns": [_fake_turn(True, 100.0, 1.0, 1.0)]},
        {"id": "Q2", "category": "factual", "turns": [_fake_turn(False, 300.0, 0.0, 1.0)]},
        {"id": "Q3", "category": "numeric", "turns": [_fake_turn(True, 200.0, 1.0, None)]},
    ]
    summary = product_qa.aggregate(case_results)

    assert summary["n_cases"] == 3
    assert summary["n_turns"] == 3
    assert summary["category_counts"] == {"factual": 2, "numeric": 1}
    assert summary["overall"]["passed"] == 2
    assert summary["overall"]["pass_rate"] == pytest.approx(2 / 3)
    assert summary["by_category"]["factual"]["pass_rate"] == pytest.approx(0.5)
    assert summary["by_category"]["numeric"]["pass_rate"] == 1.0
    assert summary["overall"]["p50_latency_ms"] == 200.0


def test_worst_failures_sorted_worst_first_and_skips_passing_turns():
    case_results = [
        {
            "id": "Q1", "category": "factual",
            "turns": [_fake_turn(True, 50.0, 1.0, 1.0)],
        },
        {
            "id": "Q2", "category": "factual",
            "turns": [_fake_turn(False, 50.0, 0.2, 1.0)],
        },
        {
            "id": "Q3", "category": "factual",
            "turns": [_fake_turn(False, 50.0, 0.0, 0.0)],
        },
    ]
    for c in case_results:
        for t in c["turns"]:
            t["result"]["error"] = None

    worst = product_qa.worst_failures(case_results, limit=10)

    assert [f["id"] for f in worst] == ["Q3", "Q2"]
