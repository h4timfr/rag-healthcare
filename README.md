# RAG for Healthcare Documents

A local, retrieval-augmented question-answering system for healthcare PDFs.
Upload a document, ask questions about it, and get grounded answers with page
citations - generated entirely on your own machine, with no cloud API calls.

> This tool provides information grounded in uploaded documents and is not a
> substitute for professional medical advice.

## What it does

- **Ingests PDFs**: extracts text page-by-page, cleans/normalizes it, splits
  it into overlapping word-window chunks, embeds each chunk, and indexes it.
- **Answers questions grounded in your documents**: hybrid dense + keyword
  retrieval finds the most relevant chunks, which are passed to a local LLM
  (Ollama) to generate a cited answer - the model is instructed to answer
  only from the retrieved context, not its own general knowledge.
- **Keeps documents isolated by default**: a question is answered using only
  the currently active document, so uploading a second document never lets
  answers leak in evidence from a different one (opt-in "ask across all
  documents" mode is available for when that isolation isn't wanted).
- **Runs as a Streamlit app** with a document library (upload, select,
  delete), chat history, and an evidence panel showing exactly which chunks
  the answer was grounded in.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Ingestion                  │
  PDF upload  ───▶  │  extract (PyMuPDF) → clean → chunk       │
                    │  → embed (MiniLM-L6-v2) → index          │
                    └─────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
            FAISS (dense vectors)              SQLite + FTS5 (keyword)
                    │                                  │
                    └────────────────┬─────────────────┘
                                      ▼
                    ┌─────────────────────────────────────────┐
  question   ───▶   │     Hybrid retrieval (weighted RRF)      │
                    │  optionally scoped to one active         │
                    │  document_id (isolation)                 │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │   Prompt builder → Ollama (llama3.2:3b)  │
                    │   → grounded, cited answer               │
                    └─────────────────────────────────────────┘
```

Chunk metadata (`document_id`, `source`, `page`, `text`) lives in SQLite;
the same integer id is used as the FAISS vector id (`IndexIDMap`), so the two
stores stay in lockstep. Deleting a document removes its rows/FTS entries
from SQLite and its vectors from FAISS (exact removal, not a rebuild - see
"Document deletion" below).

## Technology stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Dense retrieval | FAISS (`IndexFlatIP` over normalized vectors = cosine similarity) |
| Keyword retrieval | SQLite FTS5 (BM25) |
| Fusion | Weighted Reciprocal Rank Fusion (RRF) |
| Generation | Ollama, `llama3.2:3b`, local only |
| Metadata storage | SQLite |
| Tests | pytest |

## Prerequisites

- **Python 3.11+** (developed and tested on Python 3.13)
- **[Ollama](https://ollama.com)** installed and running locally, with the
  required model pulled:

  ```bash
  ollama pull llama3.2:3b
  ```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

## Running the app

```bash
python -m streamlit run app.py
```

This starts the dashboard at `http://localhost:8501`. The embedding model
and FAISS are imported lazily, so the page itself loads quickly even before
Ollama or any document is ready; the embedding model loads on first
upload/query.

## Using the app

**Uploading documents.** Drop one or more PDFs into the sidebar uploader.
Each file is processed independently through the ingestion pipeline
(extract → clean → chunk → embed → index) with its own status
(indexed / skipped / failed) - one bad file never aborts the rest of the
batch. Re-uploading a file whose exact byte content is already indexed
(matched by a SHA-256 content hash, not filename) is detected and skipped
rather than duplicated.

**Active-document isolation.** Exactly one document is "active" at a time
(shown in the sidebar and above the chat). By default, **questions are
answered using only the active document** - retrieval is scoped to its
`document_id`, so evidence from other indexed documents can never leak into
an answer. Selecting a different document in the library switches the active
document immediately; it does not retroactively change earlier answers in
the chat history, since each answer's sources are stored with that turn.

**Multiple documents.** All indexed documents appear in the sidebar library
with filename, page count, and chunk count. With 2+ documents indexed, an
opt-in "Ask across all documents" checkbox is available for questions that
should search the whole corpus instead of one document - isolation is still
the default every time the app starts.

**Document deletion.** Removing a document is a real, permanent delete, not
a soft/fake one: its rows and FTS entries are removed from SQLite and its
vectors are removed from FAISS (`IndexIDMap.remove_ids` - exact removal on
the underlying flat index, not an approximation or a full rebuild). SQLite is
committed first; FAISS is only mutated and saved after that succeeds, so a
failure mid-operation can never leave metadata pointing at vectors that
don't exist. If the active document is deleted, the active selection falls
back to the most recently indexed remaining document. A confirmation step is
required before deletion.

## Retrieval: dense vs. hybrid

- **Dense** retrieval embeds the query with the same MiniLM model used for
  documents and does a cosine-similarity search over FAISS.
- **Keyword** retrieval runs the query (stopword-filtered, OR'd terms)
  against a SQLite FTS5 index with BM25 ranking.
- **Hybrid** fuses both ranked lists with weighted Reciprocal Rank Fusion
  (dense weight 2.0, keyword weight 1.0, `rrf_k=60`), so a chunk that ranks
  well on either signal contributes to the final top-k. This is what the app
  uses by default; `scripts/query.py` and `chat.py` also use it.
- **Document-balanced allocation** applies only when several documents are
  selected *and* the question asks about each of them or compares them
  ("summarize each selected document separately", "compare the selected
  documents"). It changes nothing upstream - same FAISS search, same FTS5
  search, same RRF weights and `candidate_k` - and only changes which of the
  already-fused candidates survive truncation, giving each document a share
  of the budget before the remainder is filled by score. A document that
  matched nothing still contributes nothing, so balancing cannot pull
  irrelevant material into an answer. See
  [`_allocate_by_document`](src/retrieval/hybrid_retriever.py).

## Answer generation

The production answer path is shaped by the question, using deterministic
rules only - no second model call, and no extra latency.

- **Intent classification** ([`src/generation/intent.py`](src/generation/intent.py))
  sorts a question into `factual`, `numeric`, `summary`, `each_document`,
  `comparison`, `multi_part`, `follow_up` or `unsupported` with a small set
  of regexes. `unsupported` is deliberately never inferred from wording -
  "What was the patient's blood type?" is indistinguishable from an
  answerable factual question, and only missing evidence makes it
  unsupported.
- **Evidence-first prompting**
  ([`src/generation/prompt_builder.py`](src/generation/prompt_builder.py))
  tells the model the retrieved evidence is authoritative, that it must
  check it before declaring anything unavailable, and that describing "the
  context" is not answering the question. Each intent adds exactly one short
  ANSWER STYLE block; multi-part questions additionally get a numbered plan
  naming each requested component.
- **Structured evidence context** for multi-document questions groups
  retrieved text into self-describing `DOCUMENT:` / `PAGE:` / `EVIDENCE [n]:`
  blocks, so no block can be read without knowing which document it came
  from. A selected document with no retrieved evidence is stated explicitly
  rather than being omitted.
- **Post-generation validation**
  ([`src/generation/validation.py`](src/generation/validation.py)) is a
  deterministic checker for empty answers, refusals issued on top of
  retrieved evidence, meta-commentary, hedging, missing document sections
  and missing answer parts. It reports; it never rewrites an answer. Results
  are surfaced by the product QA harness.

The legacy `scripts/evaluate.py` benchmark path is unchanged and still uses
the original prompt and instruction set.

## Where local data lives

All runtime data is local-only and lives under `data/`, which is
git-ignored (see "Private data" below):

| Path | Contents |
|---|---|
| `data/metadata.db` | Production SQLite metadata (chunk text, source, page) |
| `data/faiss.index` | Production FAISS vector index |
| `data/raw_pdfs/` | Empty by design - uploaded PDF bytes are written to a temp directory during ingestion and deleted immediately after, never persisted here |
| `data/research_corpus/` | A separate index used only for the research benchmark (see below), fully isolated from the production index above |

Configuration (paths, chunk size, top-k, model names) is centralized in
[`src/config.py`](src/config.py).

## Private data: what's intentionally excluded from Git

This repository ships **code**, not runtime data. The following are
git-ignored and exist only on the machine that ran the app - see
[`.gitignore`](.gitignore) for the exact patterns:

- `data/metadata.db`, `data/faiss.index`, `data/research_corpus/*.db`,
  `data/research_corpus/*.index` - regenerable indexes, and the production
  ones contain full text of whatever PDFs a user has uploaded locally.
- `data/raw_pdfs/*` - any raw uploaded PDF bytes (in normal operation this
  directory is always empty; see above).
- `data/research_corpus/raw/*.pdf` - the four public-domain CDC/NHLBI
  documents used by the research benchmark are not vendored into the repo;
  `scripts/build_research_corpus.py` expects them downloaded locally first
  (source URLs are printed by the script and documented below), so the repo
  carries the reproduction code, not a copy of government PDFs.
- `data/evaluation_corpus.json`, `data/healthcare_eval_questions.json`,
  `data/evaluation_results*.json`, `data/evaluation_summary*.txt`,
  `data/evaluation_aggregate*` - the publication healthcare benchmark
  evaluates one private, user-provided sample clinical report in addition to
  the four public documents. Its original PDF was never retained on disk
  (see `app.py`'s upload handling), but these generated files contain text
  and model-generated answers derived from its already-extracted chunk text,
  so none of them are committed, regardless of which documents they cover.
- The aggregate research **report and figures** (`research/REPORT.md`,
  `research/results/*.csv`, `research/figures/*.png`) **are** committed -
  they contain only summary statistics (recall, MRR, nDCG) and document
  titles, never retrieved chunk text or generated answers.

If you fork or clone this repo, `data/` will be empty except for the
`raw_pdfs/.gitkeep` and `research_corpus/raw/.gitkeep` placeholders -that's
expected. Upload your own documents to populate it locally.

## Building the research corpus without committing private data

The research benchmark indexes five documents: four public-domain
CDC/NHLBI documents plus one private sample clinical report. Only the public
portion is reproducible from this repo:

1. Download the four public-domain documents into `data/research_corpus/raw/`:

   | File | Source |
   |---|---|
   | `cdc_diabetes_statistics_2020.pdf` | https://stacks.cdc.gov/view/cdc/85309/cdc_85309_DS1.pdf |
   | `nhlbi_dash_blood_pressure_guide.pdf` | https://www.nhlbi.nih.gov/files/docs/public/heart/new_dash.pdf |
   | `cdc_dental_infection_control_2003.pdf` | https://stacks.cdc.gov/view/cdc/134781/cdc_134781_DS1.pdf |
   | `cdc_ckd_fact_sheet_2014.pdf` | https://stacks.cdc.gov/view/cdc/21850/cdc_21850_DS1.pdf |

2. `scripts/build_research_corpus.py` also migrates a fifth, private
   document from the production index (`data/metadata.db`) if present. This
   step is specific to the original author's local environment and cannot be
   reproduced from a fresh clone - the script will raise if that document
   isn't already in your own production index, which is expected if you
   haven't uploaded it yourself.

   ```bash
   python scripts/build_research_corpus.py
   ```

   This writes `data/research_corpus/{metadata.db,faiss.index}` and
   regenerates `data/evaluation_corpus.json` (the provenance manifest).

## Testing

```bash
pytest -q --basetemp .pytest_tmp
```

(`--basetemp .pytest_tmp` avoids a Windows temp-directory cleanup
`PermissionError` that plain `pytest -q` can hit in this environment.)

## Evaluation and research

There are two evaluation harnesses; **only `scripts/run_evaluation.py`
produces trustworthy numbers** (contamination-free, document-aware page
recall, records every attempted question including errors,
`scripts/evaluate.py` is kept unmodified as a historical reference for what
the earlier version got wrong - see its module docstring).

- **Dev-regression benchmark** (`data/eval_questions.json`, 50 questions
  derived entirely from this project's own specification document,
  `data/RAG_Healthcare_Review1.pdf`): useful for catching regressions in
  this repo's own pipeline, not evidence of general healthcare-document QA
  quality.
- **Publication healthcare benchmark** (45 questions across 5 real
  healthcare documents, balanced across five question-type categories):
  built into the separate research index described above. Its raw
  question/answer data is intentionally not committed (see "Private data"),
  but the finished report and aggregate metrics are: see
  [`research/REPORT.md`](research/REPORT.md) and
  [`research/results/`](research/results/) for methodology and findings, and
  `scripts/aggregate_runs.py` for how repeated runs are combined into a
  paired Dense-vs-Hybrid statistical comparison (Wilcoxon signed-rank test).
  Generation is not deterministic (temperature 0.1, no fixed seed), so any
  single run is one sample, not a stable estimate - this is why the
  benchmark is designed to be run multiple times and aggregated rather than
  trusted from a single pass.

## CLI tools

Besides the Streamlit app:

```bash
python scripts/ingest.py path/to/document.pdf   # ingest one PDF into the production index
python scripts/query.py "your question"         # one-shot hybrid-retrieval Q&A
python chat.py                                  # interactive terminal chat with conversation history
```

## Project structure

```
rag-healthcare/
├── app.py                  # Streamlit application
├── chat.py                 # Interactive terminal chat client
├── conftest.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── config.py           # All tunable parameters
│   ├── ingestion/          # PDF extraction, cleaning, chunking
│   ├── embedding/          # Embedding model wrapper
│   ├── retrieval/          # FAISS, FTS5, hybrid retriever, metadata store
│   ├── generation/         # Prompt building, Ollama client
│   └── evaluation/         # Metrics and aggregation for the eval harness
├── scripts/                # ingest / query / evaluate / build-corpus CLIs
├── tests/                  # pytest suite
├── research/               # Evaluation report, results, figures, report generator
├── data/
│   ├── raw_pdfs/.gitkeep
│   └── research_corpus/raw/.gitkeep
└── .claude/launch.json     # Dev-server config for the Claude Code browser preview
```
