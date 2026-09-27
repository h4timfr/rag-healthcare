#!/usr/bin/env python
"""Phase 2 - chunk-level evidence resolution audit for the 45-question
research benchmark.

The benchmark's only ground truth is PAGE-level (expected_pages) plus a set
of expected_terms the answer must contain. This does not invent new ground
truth: for each question it asks a narrower, answerable question - "given
the existing page-level label, which chunk(s) on those pages actually
contain each expected term?" - and reports the answer, unresolved cases
included, rather than guessing.

Classification per question:
  EXACT        every expected_term is found in exactly one chunk on an
               expected page (possibly a different chunk per term)
  AMBIGUOUS    at least one expected_term is found on an expected page in
               MORE than one chunk - the existing label cannot pick out a
               single chunk without a human choosing between them
  UNRESOLVED   at least one expected_term is not found in ANY chunk on an
               expected page at all - the current label may be wrong, the
               term wording may not match the chunk text under this
               normalization, or the term legitimately spans a chunk
               boundary

Output: research/chunk_evidence_report.json (full detail) and a printed
summary. Never writes to data/healthcare_eval_questions.json - Phase 2's
brief is explicit that new ground truth must not be invented, so no
automatic chunk_id label is written back without this being reviewed first.
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import normalize  # noqa: E402

RESEARCH_DB = str(ROOT / "data" / "research_corpus" / "metadata.db")


def chunks_for_document(conn, document_id):
    rows = conn.execute(
        "SELECT chunk_id, page, text FROM chunks WHERE document_id=? ORDER BY chunk_id", (document_id,)
    ).fetchall()
    return rows


def audit_question(conn, item):
    rows = chunks_for_document(conn, item["document_id"])
    expected_pages = set(item["expected_pages"])
    on_page = [(cid, page, text) for cid, page, text in rows if page in expected_pages]

    term_chunk_map = {}
    for term in item["expected_terms"]:
        norm_term = normalize(term)
        matches = [cid for cid, page, text in on_page if norm_term in normalize(text)]
        term_chunk_map[term] = matches

    unresolved_terms = [t for t, matches in term_chunk_map.items() if len(matches) == 0]
    ambiguous_terms = [t for t, matches in term_chunk_map.items() if len(matches) > 1]

    # 200-word chunks with a 40-word overlap mean a short/common term near a
    # page boundary trivially appears in several CONSECUTIVE chunk_ids that
    # all cover the same underlying passage - not a genuine choice between
    # different locations. Distinguishing this from real ambiguity is not
    # inventing ground truth: overlapping chunks are, by construction,
    # duplicate text: the check_consistency ranges below are read directly
    # off chunk_id contiguity, not guessed.
    def is_contiguous_run(chunk_ids: list[int]) -> bool:
        s = sorted(chunk_ids)
        return all(b - a == 1 for a, b in zip(s, s[1:]))

    overlap_artifact_terms = [t for t in ambiguous_terms if is_contiguous_run(term_chunk_map[t])]
    genuinely_ambiguous_terms = [t for t in ambiguous_terms if t not in overlap_artifact_terms]

    if unresolved_terms:
        status = "UNRESOLVED"
    elif genuinely_ambiguous_terms:
        status = "AMBIGUOUS"
    elif overlap_artifact_terms:
        status = "EXACT_VIA_CHUNK_RANGE"  # resolvable to a contiguous run, not one exact chunk_id
    else:
        status = "EXACT"

    resolved_chunk_ids = sorted({matches[0] for matches in term_chunk_map.values() if len(matches) == 1})
    multi_chunk = status == "EXACT" and len(resolved_chunk_ids) > 1

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_pages": sorted(expected_pages),
        "expected_terms": item["expected_terms"],
        "term_chunk_map": term_chunk_map,
        "status": status,
        "multi_chunk": multi_chunk,
        "resolved_chunk_ids": resolved_chunk_ids if status in ("EXACT",) else [],
        "unresolved_terms": unresolved_terms,
        "ambiguous_terms": genuinely_ambiguous_terms,
        "overlap_artifact_terms": overlap_artifact_terms,
    }


def main():
    items = json.loads((ROOT / "data" / "healthcare_eval_questions.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(RESEARCH_DB)
    try:
        results = [audit_question(conn, item) for item in items]
    finally:
        conn.close()

    by_status = defaultdict(list)
    for r in results:
        by_status[r["status"]].append(r["id"])

    print(f"total questions: {len(results)}")
    for status in ("EXACT", "EXACT_VIA_CHUNK_RANGE", "AMBIGUOUS", "UNRESOLVED"):
        ids = by_status.get(status, [])
        print(f"  {status:<22} {len(ids):>3}  {ids}")
    multi = [r["id"] for r in results if r["multi_chunk"]]
    print(f"  (of EXACT) multi-chunk evidence: {len(multi)}  {multi}")

    print("\nby category:")
    cat_status = defaultdict(lambda: defaultdict(int))
    for r in results:
        cat_status[r["category"]][r["status"]] += 1
    for cat, counts in sorted(cat_status.items()):
        print(f"  {cat:<22} {dict(counts)}")

    print("\n--- examples ---")
    for status in ("EXACT_VIA_CHUNK_RANGE", "AMBIGUOUS", "UNRESOLVED"):
        for r in results:
            if r["status"] == status:
                print(f"\n[{status}] {r['id']} ({r['category']}): {r['question']}")
                print(f"  expected_pages={r['expected_pages']}  expected_terms={r['expected_terms']}")
                print(f"  term_chunk_map={r['term_chunk_map']}")
                break  # one example per status is enough for the console; full detail is in the JSON

    out_path = ROOT / "research" / "chunk_evidence_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull detail written to {out_path}")


if __name__ == "__main__":
    main()
