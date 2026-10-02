# System Architecture

Component-level description of the system as implemented at commit `fd1cdd5`,
written to be precise enough to draw a paper figure from. Parameter values are
the ones in `src/config.py` and the named functions; nothing here describes
planned work. Results are in [FINAL_REPORT.md](FINAL_REPORT.md).

## 1. Overview

```
                         INGESTION (per document, before questions)
 PDF ──► pdf_extractor ──► cleaner ──► chunker ──► embedder ──┬──► FAISS index (vectors, id = chunk_id)
        (PyMuPDF + word-                (200 w,   (MiniLM-L6,  │
         geometry repair)                40 overlap) 384-d)     └──► SQLite: chunks table + chunks_fts (FTS5)

                         QUESTION ANSWERING (per turn)
 question ──► intent / decomposition ──► hybrid retrieval ──► presentation order ──► prompt builder
              (regex, no model call)      dense ║ FTS5 → RRF    (lexical promotion)     (evidence + style)
                                                                                           │
 UI ◄── answer (streamed) + Sources + evidence passages + timing ◄── Ollama llama3.2:3b ◄──┘
```

Every component runs on the local machine. At run time the app talks to
Ollama at `127.0.0.1:11434` and serves Streamlit on `localhost`
(`.streamlit/config.toml`, usage statistics off). Two qualifications: which
interfaces Ollama itself listens on is Ollama's own setting, not this
repository's; and the embedding model is loaded with `local_files_only=True`
first but falls back to downloading it from the Hugging Face Hub if it is not
cached (`embedder.get_model`). The Ollama model must also be pulled once.

## 2. Component responsibilities

| Component | Module | Responsibility |
|---|---|---|
| Extraction | `src/ingestion/pdf_extractor.py` | Page text via PyMuPDF; quality checks; word-geometry repair; image-only page detection |
| Cleaning | `src/ingestion/cleaner.py` | Unicode/whitespace normalisation of page text |
| Chunking | `src/ingestion/chunker.py` | Fixed-size word windows within a page |
| Embedding | `src/embedding/embedder.py` | Sentence embeddings, cache-first model loading (download fallback), warm-up |
| Dense index | `src/retrieval/faiss_index.py` | FAISS storage, atomic save, cross-process writer lock |
| Metadata + keyword index | `src/retrieval/metadata_store.py`, `keyword_index.py` | Chunk rows, FTS5 search, document listing and deletion |
| Hybrid retrieval | `src/retrieval/hybrid_retriever.py` | Dense + keyword candidates, weighted RRF, document scope, per-part retrieval, per-document balance |
| Consistency | `src/retrieval/index_consistency.py` | Startup check and repair of FAISS vs SQLite |
| Document references | `src/retrieval/document_reference.py` | Resolve "the second document", file names, keywords to document ids |
| Intent | `src/generation/intent.py` | Deterministic question classification and decomposition |
| Prompting | `src/generation/prompt_builder.py` | Production prompts (single- and multi-document) |
| Research prompting | `src/generation/eval_prompt_builder.py` | Neutral prompt for the research benchmark only |
| Generation | `src/generation/ollama_client.py` | Ollama HTTP calls, streaming, timing, truncation flag |
| Answer validation | `src/generation/validation.py` | Deterministic checks, used by the product QA harness |
| Orchestration + UI | `app.py`, `src/ui/icons.py` | Streamlit app, upload, library, scope, chat turn lifecycle, evidence display |
| Ingestion entry point | `scripts/ingest.py` | Full ingest of one PDF (used by the CLI and the app) |

## 3. Ingestion

`scripts/ingest.py::ingest_pdf` (the app's upload handler writes the upload to
a temporary file and calls the same function):

1. **Document identity.** `document_id = sha256(file bytes)[:16]`. A document
   already present is skipped (checked again under the writer lock).
2. **Extraction.** `extract_pages` opens the PDF with PyMuPDF and reads each
   page with `page.get_text("text")`. If the text shows quality problems
   (abnormal mean word length, whitespace ratio or control-character ratio),
   the page is re-read from PyMuPDF's word segmentation
   (`get_text("words")`, regrouped by block and line), and the repair is kept
   only if it reduces the problems. Pages with images but no text are reported
   as scanned and excluded (no OCR). Output: `[{page, text}]`, 1-based pages.
   Text order is PyMuPDF's block order; multi-column layouts are not
   reconstructed (a known limitation).
3. **Cleaning.** `clean_text`: NFKC normalisation, control-character removal,
   whitespace and punctuation normalisation.
4. **Chunking.** `chunk_pages` -> `chunk_page_text`: whitespace-split words,
   windows of 200 words with 40 words overlap (stride 160), never crossing a
   page boundary. Each chunk keeps its page number.
5. **Embedding** (outside the lock): `all-MiniLM-L6-v2`, 384 dimensions,
   L2-normalised, loaded with `local_files_only=True` first, so no network
   call is made when the model is cached; if it is not cached, the load falls
   back to a normal (downloading) load.
6. **Write** (inside `cross_process_index_lock`, an OS file lock):
   - insert all chunk rows in one SQLite transaction (`chunks` table and
     `chunks_fts`); the `AUTOINCREMENT` `chunk_id`s are returned;
   - add the vectors to FAISS with those ids;
   - save FAISS atomically (temporary file + rename);
   - if the FAISS step fails, the document's rows are deleted again.

## 4. Storage

| Store | Content | Key |
|---|---|---|
| `data/metadata.db` (SQLite) table `chunks` | `chunk_id`, `document_id`, `source` (file name), `page`, `text`, `created_at` | `chunk_id INTEGER PRIMARY KEY AUTOINCREMENT` |
| same file, virtual table `chunks_fts` | FTS5 over `text`, `tokenize='porter unicode61'` | `chunk_id` (unindexed column) |
| `data/faiss.index` | `IndexIDMap(IndexFlatIP)`, 384-d, normalised vectors (inner product = cosine) | vector id = `chunk_id` |

The shared `chunk_id` is the join key between dense hits and chunk text.
At application start `check_consistency` compares the stores and `repair`
re-embeds chunks without a vector, removes orphan or duplicate vectors and
rebuilds a corrupt index from SQLite.

## 5. Retrieval

### 5.1 Single query: `retrieve_hybrid(query, top_k, document_id(s))`

1. Embed the query (same model).
2. **Dense candidates:** FAISS search, `candidate_k = 10`.
3. **Keyword candidates:** stopword-filtered query terms joined with `OR`,
   FTS5 `MATCH`, ordered by `bm25`, `candidate_k = 10`.
4. When scoped to one or more documents, `candidate_k` is raised to the total
   chunk count and the candidates are then filtered to the scope, so no
   in-scope chunk is lost to out-of-scope competitors (the keyword list can
   still be empty if no in-scope chunk matches any query term).
5. **Fusion:** weighted Reciprocal Rank Fusion,
   `score(c) = 2.0 / (60 + rank_dense(c)) + 1.0 / (60 + rank_keyword(c))`
   (a missing rank contributes 0). Ties keep insertion order (dense
   candidates first, then keyword-only ones; Python's stable sort), so the
   result is deterministic.
6. Return the `top_k` fused chunks with page, source, score and per-stage
   timings (`embedding_ms`, `dense_ms`, `keyword_ms`, `fusion_ms`).

### 5.2 Per-turn wrapper: `app.retrieve_and_build_prompt`

- **Scope:** the active document, a selection of documents, or the whole
  library, chosen in the UI; document references in the question ("the
  second document", file names) are resolved by `document_reference.py`.
- **top_k:** 5 (`config.TOP_K`); 8 for a factual question naming an explicit
  "X, Y, and Z" component list.
- **Multi-part questions** (`intent.decompose_question`): each part is
  retrieved separately (`retrieve_for_question`, parts embedded in one batch)
  with `max(1, min(top_k, MAX_CONTEXT_CHUNKS // parts))` chunks per part
  (`MAX_CONTEXT_CHUNKS = 10`); results are merged and de-duplicated, and each
  chunk is tagged with its part.
- **Multi-document balance:** for each-document or comparison questions over
  several selected documents (`intent.needs_document_balance`), the budget is
  `min(BALANCED_CHUNKS_PER_DOCUMENT x documents, MAX_CONTEXT_CHUNKS)` =
  `min(3 x documents, 10)`, and `_allocate_by_document` gives each document a
  share of it before the remainder is filled by fused score.
- **Presentation order** (single-document, single-part turns only):
  `order_chunks_for_presentation` keeps the first `PROTECT_FRONT = 2` fused
  chunks in place and may promote up to `MAX_PROMOTED = 3` later chunks whose
  text contains a number **and** at least `MIN_TERM_MATCHES = 2` distinct
  significant query terms (`_numeric_evidence_score`; a chunk without a number
  scores 0). Promoted chunks are ordered by that score, ties by original rank;
  the rest keep their RRF order. It reorders, never adds or removes.

## 6. Prompt construction

`prompt_builder.build_production_prompt` (one document) or
`build_multi_document_prompt` (several; it adds multi-document rules after the
instructions), in this order:

1. Instructions (`PRODUCTION_SYSTEM_INSTRUCTION`): the evidence is
   authoritative; check all of it before saying something is unavailable;
   never fill gaps from model knowledge; keep the source's clinical wording;
   no diagnosis or treatment directive; cite claims as `[n]`.
2. **Evidence:** numbered passages `[n]` with source file name and page;
   grouped under "PART N" headings when a multi-part question was retrieved
   per part; for multi-document prompts, an inventory of the documents in
   scope and per-document grouping.
3. Optional recent conversation (follow-up turns).
4. **One ANSWER STYLE block**, placed after the evidence so it is the last
   instruction before the question:
   - a multi-part question gets the *answer plan*: one line per part, the
     "include every item the EVIDENCE gives" rule and, if the question asks
     about medications, the "drugs or therapeutic agents only" rule;
   - otherwise the style for the intent from `intent.classify_intent`
     (factual, numeric, summary, each-document, comparison, follow-up,
     unsupported); list-type and explicit item-count factual questions get
     the list style; a single-document comparison of two parts of the
     document gets the section-identity comparison style.
5. Personal-advice reminder, only when `is_personal_advice_request` matches.
6. The user question, then the `ANSWER:` cue.

The end-to-end research harness (`scripts/run_evaluation.py`) uses
`eval_prompt_builder.build_eval_prompt` instead - a fixed neutral prompt - so
production prompt changes do not reach its prompts; the retrieval-only
benchmark involves no prompt at all.

## 7. Generation

`ollama_client.generate_stream` posts to `http://127.0.0.1:11434/api/generate`
with model `llama3.2:3b`, temperature 0.1, top_p 0.9, `keep_alive` 10 min,
and `num_predict` from `app.resolve_num_predict`: 256 by default; 768 for
multi-document scope, multi-part questions, list-type or explicit-count
factual questions, and single-document section comparisons. The stream yields
tokens to the UI; the final message supplies Ollama's timings and
`done_reason`, from which a truncated answer (`length`) is flagged. The
Ollama model and the embedder are warmed up in background threads once per
browser session, as soon as a document is active.

## 8. Evidence and source display

For each assistant turn the UI shows:

- the streamed answer, with a "cut off" label if generation hit the token
  limit;
- **Sources:** the documents and pages of the evidence passages that were in
  the prompt (grouped by document);
- **Evidence:** the verbatim text of each passage, in prompt order (grouped
  by document when several documents contributed);
- **Timing:** retrieval stages, prompt construction, time to first token,
  generation, end to end.

Sources are derived from the retrieved chunks, not parsed from the answer, so
they are always present even when the model writes no inline `[n]` citation.

## 9. Chat turn lifecycle (application reliability)

A question is committed to the conversation before answering
(`queue_question`), the input is disabled while a turn is pending, the turn is
finished with either an answer (`finish_pending_turn`) or a structured
`UserFacingError` turn (`error_turn`) that can be retried (`retry_last_turn`).

## 10. Evaluation components (not on the request path)

| Script | Purpose |
|---|---|
| `scripts/build_research_corpus.py` | Builds the separate research store under `data/research_corpus/` |
| `research/counterbalanced_retrieval_benchmark.py` | Retrieval-only benchmark (5 configurations; the three methods timed in a Latin-square order; Wilcoxon + Holm) |
| `scripts/run_evaluation.py`, `scripts/aggregate_runs.py` | End-to-end research runs and their aggregation |
| `scripts/run_product_qa.py`, `src/evaluation/product_qa.py` | Product QA through the application path |
| `research/chunk_evidence_audit.py`, `structured_question_failure_study.py`, `ambiguity_study.py` | Audits and diagnostic studies |
