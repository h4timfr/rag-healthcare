"""Production-only answer-quality QA harness for the Streamlit app's real
RAG pipeline (retrieval + prompt construction + Ollama generation).

Deliberately separate from the research/publication evaluation harness
(scripts/run_evaluation.py + src/generation/eval_prompt_builder.py +
this package's own metrics.py/aggregation.py, all untouched by this file)
and from the dev-regression benchmark (data/eval_questions.json) - this
module answers a different, narrower question: "is the app's actual
generated text good, for realistic single/multi-document/follow-up product
questions?", not "what is retrieval precision/recall across a corpus?".

Uses the exact same production code path a real user's browser session
would go through: app.py's pure helper functions (retrieve_and_build_prompt,
build_document_inventory, resolve_retrieval_top_k - none of them touch
Streamlit, see app.py's own module docstring) plus
src.generation.ollama_client.generate(). Nothing about retrieval, prompt
construction, or generation is reimplemented here - this file only adds
dataset loading, running cases through that existing pipeline, and grading
the results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import app as production_app
from src import config
from src.exceptions import (
    EmptyQueryError,
    InsufficientRetrievalError,
    ModelUnavailableError,
    OllamaUnavailableError,
    RagError,
)
from src.generation.ollama_client import generate
from src.generation.validation import is_refusal, validate_answer
from src.retrieval.metadata_store import MetadataStore
from src.text_match import contains_exact, contains_term, mentions_document

DEFAULT_DATASET_PATH = config.PROJECT_ROOT / "data" / "product_qa.json"

REQUIRED_CATEGORIES = (
    "factual",
    "numeric",
    "summary",
    "each_document",
    "comparison",
    "multi_part",
    "follow_up",
    "unsupported",
)


def load_dataset(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_DATASET_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_document_ids(sources: list[str]) -> dict[str, str]:
    """source filename -> current document_id, resolved live against
    whatever is actually indexed right now (never hardcoded at call time),
    so the harness stays correct even if the same PDFs were re-ingested
    (same content hash) or the dataset is pointed at a different index via
    config.DB_PATH. Raises RagError naming exactly which dataset sources
    aren't currently indexed, rather than failing case-by-case later.
    """
    store = MetadataStore(config.DB_PATH)
    documents = store.list_documents()
    by_source = {d["source"]: d["document_id"] for d in documents}

    missing = [s for s in sources if s not in by_source]
    if missing:
        raise RagError(
            f"product_qa dataset references document(s) not currently indexed: {missing}. "
            f"Currently indexed: {sorted(by_source)}"
        )
    return {s: by_source[s] for s in sources}


def run_turn(
    question: str,
    selected_sources: list[str],
    history: list[dict] | None,
    id_by_source: dict[str, str],
) -> dict:
    """Run one QA turn through the exact production retrieval/prompt/
    generation pipeline and return a result dict. Never raises for an
    expected RagError/Ollama failure - returns an error-flagged result
    instead, so a batch run can continue past one bad case (mirrors
    app.ingest_uploaded_pdfs' own one-bad-item-must-not-abort-the-batch
    pattern).
    """
    document_ids = [id_by_source[s] for s in selected_sources]

    if len(document_ids) == 1:
        # Mirrors Active-document scope exactly: a single document_id
        # string, no document_inventory, plain config.TOP_K.
        document_id: str | list[str] = document_ids[0]
        document_inventory = None
        top_k = config.TOP_K
    else:
        # Mirrors Selected-documents scope exactly - same functions
        # app.py's render_chat calls, called the same way.
        document_id = document_ids
        selected_set = set(document_ids)
        store = MetadataStore(config.DB_PATH)
        documents = store.list_documents()
        document_inventory = production_app.build_document_inventory(documents, selected_set)
        top_k = production_app.resolve_retrieval_top_k("selected", selected_set)

    start = time.perf_counter()
    try:
        chunks, prompt = production_app.retrieve_and_build_prompt(
            question,
            top_k=top_k,
            document_id=document_id,
            document_inventory=document_inventory,
            history=history,
        )
    except (EmptyQueryError, InsufficientRetrievalError) as e:
        return {
            "question": question,
            "error": str(e),
            "error_type": type(e).__name__,
            "answer": None,
            "chunks": [],
            "sources": [],
            "prompt": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "generated_ok": False,
        }

    # Same multi-document generation budget the real app uses (see
    # app.py's render_chat) - only when multiple documents are selected.
    num_predict = config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT if document_inventory else None

    try:
        answer = generate(prompt, num_predict=num_predict)
    except (OllamaUnavailableError, ModelUnavailableError) as e:
        return {
            "question": question,
            "error": str(e),
            "error_type": type(e).__name__,
            "answer": None,
            "chunks": chunks,
            "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
            "prompt": prompt,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "generated_ok": False,
        }

    latency_ms = (time.perf_counter() - start) * 1000
    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]
    return {
        "question": question,
        "error": None,
        "error_type": None,
        "answer": answer,
        "chunks": chunks,
        "sources": sources,
        "prompt": prompt,
        "latency_ms": latency_ms,
        "generated_ok": True,
    }


def evaluate_turn(turn_spec: dict, result: dict, category: str) -> dict:
    """Score one turn's result against its QA spec.

    Grading is deliberately asymmetric, because the two directions have
    opposite failure costs:

    required_facts  matched with src.text_match.contains_term, which allows
                    morphological variants ("a syncopal episode" satisfies a
                    required "syncope"). Missing this scored a correct answer
                    as a failure in the first baseline and told us nothing
                    about the model. Leniency here can only convert a false
                    failure into a pass.

    forbidden_terms matched with exact case-insensitive substring only.
                    Leniency here would invent a failure that did not happen,
                    and a measured value must be compared literally - a
                    number is never stemmed or normalised.

    required document mentions use src.text_match.mentions_document, which
    accepts the filename or an equivalent natural reference. Requiring the
    literal "OMC Report Sample - Cardio.pdf" inside prose marked a correctly
    structured comparison as unstructured because the model wrote "the OMC
    Cardiology report" - again measuring the grader, not the answer.

    Everything stays deterministic and inspectable: the same answer always
    scores the same, and every metric names the terms it matched, so any
    score can be traced back to a specific string decision.
    """
    answer = result.get("answer") or ""
    generated_ok = bool(result.get("generated_ok"))

    required_facts = turn_spec.get("required_facts", [])
    fact_hits = [f for f in required_facts if contains_term(answer, f)]
    fact_hits_exact = [f for f in required_facts if contains_exact(answer, f)]
    fact_coverage = (len(fact_hits) / len(required_facts)) if required_facts else None

    required_sources = turn_spec.get("required_sources", [])
    retrieved_sources = {s["source"] for s in result.get("sources", [])}
    doc_hits_retrieval = [s for s in required_sources if s in retrieved_sources]
    document_coverage = (len(doc_hits_retrieval) / len(required_sources)) if required_sources else None

    forbidden_terms = turn_spec.get("forbidden_terms", [])
    forbidden_hits = [t for t in forbidden_terms if contains_exact(answer, t)]

    refusal_detected = is_refusal(answer)

    validation = validate_answer(
        answer,
        category,
        evidence_sources=[s["source"] for s in result.get("sources", [])],
        required_documents=required_sources if category in ("each_document", "comparison") else None,
        question=turn_spec.get("question"),
    )

    metrics: dict = {
        "generated_ok": generated_ok,
        "fact_coverage": fact_coverage,
        "fact_hits": fact_hits,
        # Facts that matched only after morphology normalisation - the
        # answers the previous grader would have scored as misses.
        "fact_hits_morphological_only": [f for f in fact_hits if f not in fact_hits_exact],
        "document_coverage": document_coverage,
        "documents_retrieved": sorted(retrieved_sources),
        "forbidden_term_hit": len(forbidden_hits) > 0,
        "forbidden_terms_found": forbidden_hits,
        "source_attribution_present": bool(result.get("sources")),
        "refusal_detected": refusal_detected,
        "validation": validation,
        "validation_ok": validation["ok"],
        "validation_issues": validation["issues"],
    }

    if category in ("each_document", "comparison"):
        # Structure check: every required document must be referred to in the
        # answer text itself (not just retrieved) - an each-document or
        # comparison answer must actually be organized around the documents,
        # not merely have their evidence mixed in. A natural reference counts;
        # the exact filename is recorded separately for transparency.
        named = [s for s in required_sources if mentions_document(answer, s)]
        named_exact = [s for s in required_sources if contains_exact(answer, s)]
        metrics["documents_named_in_answer"] = named
        metrics["documents_named_exact_filename"] = named_exact
        metrics["structure_ok"] = (len(named) == len(required_sources)) if required_sources else None

    if category == "unsupported":
        metrics["refusal_correct"] = generated_ok and refusal_detected and not forbidden_hits

    checks = [generated_ok]
    if fact_coverage is not None:
        checks.append(fact_coverage >= 0.5)
    if document_coverage is not None:
        checks.append(document_coverage >= 0.5)
    checks.append(not forbidden_hits)
    if category in ("each_document", "comparison") and metrics.get("structure_ok") is not None:
        checks.append(metrics["structure_ok"])
    if category == "unsupported":
        checks.append(metrics.get("refusal_correct", False))
    metrics["passed"] = all(checks)

    return metrics


def run_case(case: dict, id_by_source: dict[str, str]) -> dict:
    """Run every turn of one QA case in order, threading conversation
    history exactly like app.py's render_chat does: each turn gets a
    snapshot of prior turns taken BEFORE that turn's own question/answer
    are appended, and the running question is always the clean current
    question - never a history-embedded string - passed to retrieval.
    """
    selected_sources = case["selected_document_sources"]
    history: list[dict] = []
    turn_results = []

    for turn_spec in case["turns"]:
        prior_history = list(history) if history else None
        result = run_turn(turn_spec["question"], selected_sources, prior_history, id_by_source)
        metrics = evaluate_turn(turn_spec, result, case["category"])
        turn_results.append({"spec": turn_spec, "result": result, "metrics": metrics})

        history.append({"role": "user", "content": turn_spec["question"]})
        history.append(
            {"role": "assistant", "content": result.get("answer") or f"Error: {result.get('error')}"}
        )

    return {
        "id": case["id"],
        "category": case["category"],
        "selected_document_sources": selected_sources,
        "selected_document_ids": [id_by_source[s] for s in selected_sources],
        "turns": turn_results,
    }


def run_dataset(dataset: dict, id_by_source: dict[str, str] | None = None) -> list[dict]:
    """Run every case in a loaded dataset. If `id_by_source` isn't given,
    it's resolved live via resolve_document_ids()."""
    if id_by_source is None:
        sources = sorted({s for case in dataset["cases"] for s in case["selected_document_sources"]})
        id_by_source = resolve_document_ids(sources)
    return [run_case(case, id_by_source) for case in dataset["cases"]]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _turn_summary(turns: list[dict]) -> dict:
    n = len(turns)
    if n == 0:
        return {
            "n_turns": 0, "passed": 0, "pass_rate": 0.0, "generated_ok_rate": 0.0,
            "avg_fact_coverage": None, "avg_document_coverage": None,
            "source_attribution_rate": 0.0, "validation_ok_rate": 0.0,
            "avg_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
        }

    passed = sum(1 for t in turns if t["metrics"]["passed"])
    generated_ok = sum(1 for t in turns if t["metrics"]["generated_ok"])
    validation_ok = sum(1 for t in turns if t["metrics"].get("validation_ok"))
    fact_covs = [t["metrics"]["fact_coverage"] for t in turns if t["metrics"]["fact_coverage"] is not None]
    doc_covs = [t["metrics"]["document_coverage"] for t in turns if t["metrics"]["document_coverage"] is not None]
    source_attr = sum(1 for t in turns if t["metrics"]["source_attribution_present"])
    latencies = [t["result"]["latency_ms"] for t in turns]

    return {
        "n_turns": n,
        "passed": passed,
        "pass_rate": passed / n,
        "generated_ok_rate": generated_ok / n,
        "avg_fact_coverage": (sum(fact_covs) / len(fact_covs)) if fact_covs else None,
        "avg_document_coverage": (sum(doc_covs) / len(doc_covs)) if doc_covs else None,
        "source_attribution_rate": source_attr / n,
        "validation_ok_rate": validation_ok / n,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def aggregate(case_results: list[dict]) -> dict:
    """Overall + per-category metric summary across every turn of every
    case, including latency mean/p50/p95 (per the task's explicit ask)."""
    all_turns = [t for case in case_results for t in case["turns"]]
    categories = sorted({case["category"] for case in case_results})

    issue_counts: dict[str, int] = {}
    for turn in all_turns:
        for issue in turn["metrics"].get("validation_issues") or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    return {
        "n_cases": len(case_results),
        "n_turns": len(all_turns),
        "category_counts": {
            cat: sum(1 for c in case_results if c["category"] == cat) for cat in categories
        },
        "validation_issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: -kv[1])),
        "overall": _turn_summary(all_turns),
        "by_category": {
            cat: _turn_summary([t for c in case_results if c["category"] == cat for t in c["turns"]])
            for cat in categories
        },
    }


def worst_failures(case_results: list[dict], limit: int = 10) -> list[dict]:
    """Failing turns, worst-first (lowest combined coverage), for the
    report's "worst failing examples" section and manual inspection."""
    failures = []
    for case in case_results:
        for i, turn in enumerate(case["turns"]):
            if turn["metrics"]["passed"]:
                continue
            coverages = [
                c for c in (turn["metrics"].get("fact_coverage"), turn["metrics"].get("document_coverage"))
                if c is not None
            ]
            score = (sum(coverages) / len(coverages)) if coverages else 0.0
            failures.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "turn_index": i,
                    "question": turn["spec"]["question"],
                    "answer": turn["result"].get("answer"),
                    "error": turn["result"].get("error"),
                    "metrics": turn["metrics"],
                    "score": score,
                }
            )
    failures.sort(key=lambda f: f["score"])
    return failures[:limit]
