#!/usr/bin/env python
"""Retrieval-ambiguity study (Task 19): can generic retrieval statistics,
computed before generation, flag queries whose top-K evidence misses the
answer?

Read-only with respect to every index. Reproduces retrieve_hybrid()'s own
dense/keyword/fusion steps from the production building blocks (no
reimplemented ranking), so every signal describes exactly what production
retrieval saw.

Labels come from each question's own ground truth and are used ONLY to score
the signals afterwards - no signal reads them:
  - research benchmark: document-aware page recall@K < 1  -> retrieval miss
  - production-index questions: listed target chunk ids / required facts

Outputs (gitignored - question text includes private clinical content):
  output/ambiguity_study/signals.json
  output/ambiguity_study/summary.txt

Run:  python research/ambiguity_study.py
"""

import json
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config, text_match  # noqa: E402
from src.embedding.embedder import embed_query  # noqa: E402
from src.evaluation.metrics import document_aware_page_recall, normalize, reciprocal_rank_at_k  # noqa: E402
from src.generation.prompt_builder import _significant_query_terms  # noqa: E402
from src.retrieval.faiss_index import FaissIndex  # noqa: E402
from src.retrieval.keyword_index import KeywordIndex  # noqa: E402
from src.retrieval.metadata_store import MetadataStore  # noqa: E402

OUT_DIR = ROOT / "output" / "ambiguity_study"
RESEARCH_DB = str(ROOT / "data" / "research_corpus" / "metadata.db")
RESEARCH_FAISS = str(ROOT / "data" / "research_corpus" / "faiss.index")
PROD_DB = str(ROOT / "data" / "metadata.db")
PROD_FAISS = str(ROOT / "data" / "faiss.index")

RRF_K = 60
DENSE_W, KEYWORD_W = 2.0, 1.0
CANDIDATE_K = 10
TABLE_HEADING = re.compile(r"\bTable\s+(\d+)[a-z]?\.\s+[A-Z]")


# ---------------------------------------------------------------------------
# Retrieval internals, exactly as retrieve_hybrid() computes them
# ---------------------------------------------------------------------------

class Corpus:
    def __init__(self, db_path, faiss_path):
        self.store = MetadataStore(db_path)
        self.keyword = KeywordIndex(db_path)
        self.faiss = FaissIndex(config.EMBEDDING_DIM, faiss_path)
        self.faiss.load()
        self.count = self.store.count()
        flat = __import__("faiss").downcast_index(self.faiss.index.index)
        vecs = __import__("faiss").rev_swig_ptr(flat.get_xb(), flat.ntotal * flat.d)
        ids = __import__("faiss").vector_to_array(self.faiss.index.id_map)
        self.vectors = {int(i): np.array(vecs[k * flat.d:(k + 1) * flat.d]) for k, i in enumerate(ids)}

    def rankings(self, query, document_id=None):
        """dense [(id, cos)], keyword [(id, bm25)], fused [(id, rrf)] - same
        candidate pool, filtering and fusion as retrieve_hybrid()."""
        cand_k = max(CANDIDATE_K, self.count) if document_id else CANDIDATE_K
        qv = embed_query(query)
        d_scores, d_ids = self.faiss.search(qv, cand_k)
        dense = [(int(i), float(s)) for i, s in zip(d_ids, d_scores) if i != -1]
        kw = [(r["chunk_id"], r["score"]) for r in self.keyword.search(query, cand_k)]
        if document_id:
            allowed = self.store.filter_chunk_ids_by_documents(
                list({i for i, _ in dense} | {i for i, _ in kw}), [document_id]
            )
            dense = [p for p in dense if p[0] in allowed]
            kw = [p for p in kw if p[0] in allowed]
        fused = {}
        for r, (cid, _) in enumerate(dense, 1):
            fused[cid] = fused.get(cid, 0.0) + DENSE_W / (RRF_K + r)
        for r, (cid, _) in enumerate(kw, 1):
            fused[cid] = fused.get(cid, 0.0) + KEYWORD_W / (RRF_K + r)
        fused = sorted(fused.items(), key=lambda p: p[1], reverse=True)
        return dense, kw, fused, qv

    def doc_chunk_count(self, document_id):
        return self.store.document_chunk_count(document_id) if document_id else self.count


# ---------------------------------------------------------------------------
# Signals - none of these read ground truth
# ---------------------------------------------------------------------------

def kendall_tau(rank_a, rank_b):
    """Kendall tau-b over the union of two top lists; an item absent from a
    list is tied at rank len(list)+1 in it."""
    items = list(dict.fromkeys(rank_a + rank_b))
    if len(items) < 2:
        return None
    pa = {c: i for i, c in enumerate(rank_a)}
    pb = {c: i for i, c in enumerate(rank_b)}
    xa = [pa.get(c, len(rank_a)) for c in items]
    xb = [pb.get(c, len(rank_b)) for c in items]
    conc = disc = ta = tb = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            da, db = xa[i] - xa[j], xb[i] - xb[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / denom if denom else None


def signals(corpus, query, document_id, k):
    dense, kw, fused, qv = corpus.rankings(query, document_id)
    top_ids = [c for c, _ in fused[:k]]
    meta = corpus.store.get_chunks(top_ids)
    top = [meta[c] for c in top_ids if c in meta]
    d_ids = [c for c, _ in dense]
    k_ids = [c for c, _ in kw]
    cos = [s for _, s in dense]
    bm = [s for _, s in kw]
    rrf = [s for _, s in fused[:k]]

    d5, k5 = set(d_ids[:k]), set(k_ids[:k])
    union = d5 | k5
    pages = [c["page"] for c in top]

    p = np.array(rrf) / sum(rrf) if rrf else np.array([])
    rrf_entropy = float(-(p * np.log(p)).sum() / math.log(len(p))) if len(p) > 1 else None

    vecs = [corpus.vectors[c] for c in top_ids if c in corpus.vectors]
    pair = [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]

    terms = _significant_query_terms(query)
    top1_text = top[0]["text"] if top else ""
    per_chunk_hits = [sum(text_match.contains_term(c["text"], t) for t in terms) for c in top]

    tables = set()
    for c in top:
        tables.update(m.group(1) for m in TABLE_HEADING.finditer(c["text"]))

    doc_chunks = corpus.doc_chunk_count(document_id)
    return {
        "top_ids": top_ids,
        "top_pages": pages,
        "S_dense_fts_jaccard": len(d5 & k5) / len(union) if union else None,
        "S_dense_fts_kendall10": kendall_tau(d_ids[:10], k_ids[:10]),
        "S_top1_agree": float(bool(d_ids and k_ids and d_ids[0] == k_ids[0])),
        "S_dense_top1_cos": cos[0] if cos else None,
        "S_dense_margin12": (cos[0] - cos[1]) if len(cos) > 1 else None,
        "S_dense_spread10": (cos[0] - mean(cos[1:10])) if len(cos) > 2 else None,
        "S_fts_match_fraction": (len(kw) / doc_chunks) if doc_chunks else None,
        "S_fts_ratio21": (bm[1] / bm[0]) if len(bm) > 1 and bm[0] else None,
        "S_rrf_margin12": ((rrf[0] - rrf[1]) / rrf[0]) if len(rrf) > 1 else None,
        "S_rrf_entropy": rrf_entropy,
        "S_page_span": (max(pages) - min(pages)) if pages else None,
        "S_distinct_pages": len(set(pages)),
        "S_redundancy": mean(pair) if pair else None,
        "S_top1_term_coverage": (sum(text_match.contains_term(top1_text, t) for t in terms) / len(terms)) if terms else None,
        "S_chunks_with_half_terms": sum(h >= max(1, len(terms) / 2) for h in per_chunk_hits) if terms else None,
        "S_distinct_tables": len(tables),
    }


SIGNAL_KEYS = [
    "S_dense_fts_jaccard", "S_dense_fts_kendall10", "S_top1_agree", "S_dense_top1_cos",
    "S_dense_margin12", "S_dense_spread10", "S_fts_match_fraction", "S_fts_ratio21",
    "S_rrf_margin12", "S_rrf_entropy", "S_page_span", "S_distinct_pages", "S_redundancy",
    "S_top1_term_coverage", "S_chunks_with_half_terms", "S_distinct_tables",
]


# ---------------------------------------------------------------------------
# Separability statistics
# ---------------------------------------------------------------------------

def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 * P(tie). Threshold-free."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_ci(pos, neg, n=2000, seed=0):
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        bp = [rng.choice(pos) for _ in pos]
        bn = [rng.choice(neg) for _ in neg]
        stats.append(auroc(bp, bn))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def max_auroc(rows, labels, k):
    """Best separability over ALL signals for one labelling - the statistic a
    researcher implicitly reports when they pick the best-looking signal."""
    best = 0.0
    for key in SIGNAL_KEYS:
        pos = [r[f"signals@{k}"][key] for r, lab in zip(rows, labels) if lab and r[f"signals@{k}"][key] is not None]
        neg = [r[f"signals@{k}"][key] for r, lab in zip(rows, labels) if not lab and r[f"signals@{k}"][key] is not None]
        a = auroc(pos, neg)
        if a is not None:
            best = max(best, a, 1 - a)
    return best


def familywise_p(rows, labels, k, n=2000, seed=0):
    """Permutation test of the max-over-signals AUROC. With few positives, the
    best of many signals looks strong by chance; this asks whether the
    observed best beats the best you get from shuffled labels."""
    observed = max_auroc(rows, labels, k)
    rng = random.Random(seed)
    null = []
    for _ in range(n):
        perm = labels[:]
        rng.shuffle(perm)
        null.append(max_auroc(rows, perm, k))
    null.sort()
    return observed, null[int(0.95 * n)], sum(v >= observed for v in null) / n


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

def research_items():
    with open(ROOT / "data" / "healthcare_eval_questions.json", encoding="utf-8") as f:
        return json.load(f)


PROD_EXTERNAL = [
    # (label, question, target chunk ids that must appear in top-K)
    ("CDC-Q1", "What were the findings for each diagnostic category?", [81]),
    ("CDC-Q2", "What were the main findings regarding examinations, laboratory tests, and treatments?", [87]),
    ("CDC-Q3", "What were the most frequent treatment categories?", [90]),
]
CDC_DOC = "6a08a44c84a10b22"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    research = Corpus(RESEARCH_DB, RESEARCH_FAISS)
    prod = Corpus(PROD_DB, PROD_FAISS)

    rows = []
    for item in research_items():
        for mode in ("scoped", "global"):
            doc = item["document_id"] if mode == "scoped" else None
            row = {"set": "research", "id": item["id"], "category": item["category"],
                   "document_id": item["document_id"], "mode": mode}
            for k in (5, 10):
                s = signals(research, item["question"], doc, k)
                chunks = [dict(page=p, source=item["source"]) for p in s["top_pages"]]
                # page source must match: re-fetch real sources for global mode
                real = research.store.get_chunks(s["top_ids"])
                chunks = [{"page": real[c]["page"], "source": real[c]["source"]} for c in s["top_ids"] if c in real]
                row[f"recall@{k}"] = document_aware_page_recall(chunks, item["expected_pages"], item["source"])
                # Content-based alternative label: page-level relevance both
                # over-credits (a non-answer chunk from the right page) and
                # under-credits (the answer retrieved from another page).
                text = normalize(" ".join(real[c]["text"] for c in s["top_ids"] if c in real))
                row[f"term_recall@{k}"] = (
                    sum(normalize(t) in text for t in item["expected_terms"]) / len(item["expected_terms"])
                )
                row[f"mrr@{k}"] = reciprocal_rank_at_k(chunks, item["expected_pages"], item["source"], k=k)
                row[f"signals@{k}"] = {key: s[key] for key in SIGNAL_KEYS}
            rows.append(row)

    for label, q, targets in PROD_EXTERNAL:
        row = {"set": "production", "id": label, "category": "external", "document_id": CDC_DOC, "mode": "scoped"}
        for k in (5, 10):
            s = signals(prod, q, CDC_DOC, k)
            row[f"recall@{k}"] = sum(t in s["top_ids"] for t in targets) / len(targets)
            row[f"signals@{k}"] = {key: s[key] for key in SIGNAL_KEYS}
        rows.append(row)

    # determinism check: recompute one question's signals and compare
    probe = research_items()[0]
    again = signals(research, probe["question"], probe["document_id"], 5)
    first = next(r for r in rows if r["id"] == probe["id"] and r["mode"] == "scoped")["signals@5"]
    deterministic = all(
        (again[k] == first[k]) or (again[k] is not None and first[k] is not None and abs(again[k] - first[k]) < 1e-9)
        for k in SIGNAL_KEYS
    )

    (OUT_DIR / "signals.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [f"deterministic re-run of one question: {deterministic}", ""]
    for mode in ("scoped", "global"):
        for k in (5,):
            rs = [r for r in rows if r["set"] == "research" and r["mode"] == mode]
            fail = [r for r in rs if r[f"recall@{k}"] < 1]
            ok = [r for r in rs if r[f"recall@{k}"] >= 1]
            rec10 = [r for r in fail if r["recall@10"] > r[f"recall@{k}"]]
            lines.append(f"=== research benchmark | mode={mode} | K={k} ===")
            lines.append(f"retrieval misses: {len(fail)}/{len(rs)}  ids={[r['id'] for r in fail]}")
            lines.append(f"  of which K=10 recovers more evidence: {len(rec10)} ids={[r['id'] for r in rec10]}")
            lines.append(f"  miss categories: {sorted(r['category'] for r in fail)}")
            if len(fail) < 2:
                lines.append("  too few misses for separability statistics")
                lines.append("")
                continue
            lines.append(f"  {'signal':<26}{'AUROC':>7}  {'95% CI':>15}  dir   {'miss median':>12}{'ok median':>11}")
            for key in SIGNAL_KEYS:
                pos = [r[f"signals@{k}"][key] for r in fail if r[f"signals@{k}"][key] is not None]
                neg = [r[f"signals@{k}"][key] for r in ok if r[f"signals@{k}"][key] is not None]
                a = auroc(pos, neg)
                if a is None:
                    continue
                direction = "high" if a >= 0.5 else "low"
                sep = max(a, 1 - a)
                lo, hi = bootstrap_ci(pos, neg)
                lo, hi = (lo, hi) if a >= 0.5 else (1 - hi, 1 - lo)
                lines.append(
                    f"  {key:<26}{sep:>7.3f}  [{lo:.3f}, {hi:.3f}]  {direction:<5}"
                    f"{median(pos):>12.3f}{median(neg):>11.3f}"
                )
            for label_name, key in (("page-recall miss", "recall@5"), ("term-recall miss", "term_recall@5")):
                labels = [r[key] < 1 for r in rs]
                obs, null95, p = familywise_p(rs, labels, k)
                lines.append(
                    f"  family-wise test [{label_name}, {sum(labels)} positives]: max AUROC over "
                    f"{len(SIGNAL_KEYS)} signals = {obs:.3f}, null 95th pct = {null95:.3f}, p = {p:.3f}"
                )
            lines.append("")

    lines.append("=== production-index external questions (scoped) ===")
    for r in rows:
        if r["set"] == "production":
            s = r["signals@5"]
            lines.append(
                f"  {r['id']}: recall@5={r['recall@5']} recall@10={r['recall@10']} "
                + " ".join(f"{k.removeprefix('S_')}={s[k]:.3f}" for k in
                           ("S_top1_term_coverage", "S_dense_spread10", "S_rrf_margin12", "S_page_span"))
            )

    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
