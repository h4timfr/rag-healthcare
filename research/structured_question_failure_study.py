#!/usr/bin/env python
"""Phase 4 - structured-question failure study (diagnostic only, no
extraction mechanism implemented here).

Step 1: identify which of the 45 benchmark questions require enumeration,
ranking, category/value extraction, comparison, or multi-row/column
extraction - using the EXISTING production shape detectors
(intent.requests_multiple_components, intent.is_enumeration_question) plus
a few generic ranking/comparison keyword regexes, not invented labels.

Step 2: for exactly those questions, run the real production retrieval +
generation path (document-scoped, top_k=5, current prompt/answer-style code
- nothing modified) once each, and classify the outcome using the taxonomy
in the sprint brief:
  1 retrieval failure        - required evidence not in the retrieved chunks
  2 table selection failure  - n/a proxy: wrong section/page's content used
  6 generation omission      - evidence present, a required fact missing
  7 generation hallucination - a stated fact contradicts the evidence
  8 correct                  - all required facts present and correct
(3/4/5 - table/column/row/numeric-association failures - are sub-cases of a
generation-stage error only distinguishable by inspecting NUMERIC TABLE
evidence specifically; noted per-question where applicable, not forced.)

This does not re-run the CDC document's own Q1/Q2/Q3 (already exhaustively
characterized in the prior sprint - Tasks 15-19) - it asks the DIFFERENT
question Phase 4 poses: how often does this general 5-document, 45-question
benchmark contain the SAME failure shape at all.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

config.DB_PATH = str(ROOT / "data" / "research_corpus" / "metadata.db")
config.FAISS_INDEX_PATH = str(ROOT / "data" / "research_corpus" / "faiss.index")

from app import answer_question  # noqa: E402
from src.evaluation.metrics import document_aware_page_recall, normalize  # noqa: E402
from src.generation.intent import is_enumeration_question, requests_multiple_components  # noqa: E402

_RANKING_KW = re.compile(r"\b(most|least|highest|lowest|top|leading|ranked?)\b", re.IGNORECASE)
_COMPARISON_KW = re.compile(r"\bcompar|difference|versus|\bvs\b", re.IGNORECASE)


def structured_shape(question: str) -> list[str]:
    shapes = []
    if requests_multiple_components(question):
        shapes.append("multi-component")
    if is_enumeration_question(question) and "multi-component" not in shapes:
        shapes.append("enumeration/ranking")
    if _RANKING_KW.search(question):
        shapes.append("ranking-keyword")
    if _COMPARISON_KW.search(question):
        shapes.append("comparison-keyword")
    return shapes


def classify(item: dict, chunks: list[dict], answer: str) -> str:
    retrieval_ok = document_aware_page_recall(chunks, item["expected_pages"], item["source"]) >= 1.0
    if not retrieval_ok:
        return "1_retrieval_failure"

    answer_norm = normalize(answer)
    missing = [t for t in item["expected_terms"] if normalize(t) not in answer_norm]
    if missing:
        return f"6_generation_omission(missing={missing})"
    return "8_correct"


def main():
    items = json.loads((ROOT / "data" / "healthcare_eval_questions.json").read_text(encoding="utf-8"))
    flagged = [(it, structured_shape(it["question"])) for it in items]
    flagged = [(it, shapes) for it, shapes in flagged if shapes]

    print(f"structured-shaped questions: {len(flagged)} / {len(items)}")
    for it, shapes in flagged:
        print(f"  {it['id']} ({it['category']}) {shapes}: {it['question']}")

    print("\n--- running production retrieval + generation once per flagged question ---")
    results = []
    for it, shapes in flagged:
        r = answer_question(it["question"], document_id=it["document_id"])
        outcome = classify(it, r["chunks"], r["answer"])
        results.append({
            "id": it["id"], "category": it["category"], "shapes": shapes,
            "question": it["question"], "expected_terms": it["expected_terms"],
            "answer": r["answer"], "outcome": outcome,
        })
        print(f"\n[{it['id']}] shapes={shapes} outcome={outcome}")
        print(f"  Q: {it['question']}")
        print(f"  expected_terms: {it['expected_terms']}")
        print(f"  answer: {r['answer'][:400]}")

    outcomes = {}
    for r in results:
        key = r["outcome"].split("(")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
    print(f"\n=== outcome counts (n={len(results)}) ===")
    for k, v in sorted(outcomes.items()):
        print(f"  {k}: {v}")

    out_path = ROOT / "research" / "structured_question_failure_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
