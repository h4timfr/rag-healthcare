# RAG for Healthcare Documents

A local, retrieval-augmented question-answering system for healthcare PDFs.
Upload a document, ask questions about it, and get grounded answers with the
source pages listed under each one - generated entirely on your own machine,
with no cloud API calls (after a one-time download of the two models).

> This tool provides information grounded in uploaded documents and is not a
> substitute for professional medical advice.

## What it does

- **Ingests PDFs**: extracts text page-by-page, cleans/normalizes it, splits
  it into overlapping word-window chunks, embeds each chunk, and indexes it.
- **Answers questions grounded in your documents**: hybrid dense + keyword
  retrieval finds the most relevant chunks, which are passed to a local LLM
  (Ollama) to generate an answer - the model is instructed to answer only
  from the retrieved context, not its own general knowledge. Every answer
  lists the pages it was given and shows the exact evidence chunks; the
  model is also asked to cite evidence numbers inline, but `llama3.2:3b`
  rarely does (2 of 46 measured answers), so the Sources list is the
  provenance to rely on.
- **Keeps documents isolated by default**: a question is answered using only
  the document currently in scope, so uploading a second document never lets
  evidence from a different one leak into an answer. Analysing several
  documents together is an explicit, opt-in choice.
- **Understands which document you mean**: "what medications are in the OMC
  report?", "what about the third document?" and "compare the first and
  third documents" are resolved to specific documents deterministically,
  before retrieval runs, and the answer is scoped to exactly those.
- **Runs as a Streamlit app** with a document library (upload, scope,
  delete), chat history, page-level source citations, and an evidence panel
  showing exactly which chunks the answer was grounded in.

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
                    │   → grounded answer + its sources        │
                    └─────────────────────────────────────────┘
```

Chunk metadata (`document_id`, `source`, `page`, `text`) lives in SQLite;
the same integer id is used as the FAISS vector id (`IndexIDMap`), so the two
stores stay in lockstep. Deleting a document removes its rows/FTS entries
from SQLite and its vectors from FAISS (exact removal, not a rebuild - see
"Document deletion" below). A component-level description, with every
parameter, is in [`research/ARCHITECTURE.md`](research/ARCHITECTURE.md).

## Technology stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Dense retrieval | FAISS (`IndexFlatIP` over normalized vectors = cosine similarity) |
| Keyword retrieval | SQLite FTS5 (BM25, `porter unicode61` tokenizer) |
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

This starts the dashboard at `http://localhost:8501`, bound to this machine
only. The embedding model and FAISS are imported lazily, so the page itself
loads quickly even before Ollama or any document is ready. Once a document
exists, the embedding model (a ~16 s torch import when cold) and the Ollama
model are warmed up in the background so the first question does not pay
for them. The embedding model is loaded from the local Hugging Face cache
without contacting the Hub; only the very first run downloads it.

On every start the app compares the SQLite chunk ids with the FAISS vector
ids and repairs drift - a chunk with no vector (e.g. a process killed between
the two writes) is re-embedded from its stored text, an orphan vector is
removed, and an unreadable index is rebuilt from SQLite. Index writes from
the app and from `scripts/ingest.py` are serialised by an OS file lock, so
the CLI can run alongside the app.

## Using the app

**Uploading documents.** Drop one or more PDFs into the sidebar uploader.
Each file is processed independently through the ingestion pipeline
(extract → clean → chunk → embed → index) with its own status
(indexed / skipped / failed) - one bad file never aborts the rest of the
batch. Re-uploading a file whose exact byte content is already indexed
(matched by a SHA-256 content hash, not filename) is detected and skipped
rather than duplicated.

**Choosing what gets analysed.** The sidebar has one document library and
one **Analysis scope** switch that decides what each entry in it does:

| Scope | What the library shows | What a question can see |
|---|---|---|
| *One document at a time* (default) | a button per document | only that document |
| *Several documents together* | a tick box per document | only the ticked documents |

Whichever is chosen, the documents in scope are listed above the chat before
you ask anything, and marked "in scope" in the library - the app never
leaves you guessing which documents an answer could have come from.
Retrieval is scoped by `document_id`, so evidence from a document that is
not in scope cannot reach an answer. Changing the scope does not rewrite
earlier answers: each turn stores the sources it actually used.

**Referring to documents by name or position.** In multi-document scope you
can name a document instead of ticking it - "what medications are in the OMC
report?", "summarise the second document", "compare the first and third
documents". References are resolved deterministically against the documents
currently in scope (filename, distinctive filename words, ordinal position,
or an unambiguous antecedent from the previous turn). A reference that fits
more than one document resolves to none of them and the scope is left alone,
rather than the app quietly guessing.

**Document deletion.** Removing a document is a real, permanent delete, not
a soft/fake one: its rows and FTS entries are removed from SQLite and its
vectors are removed from FAISS (`IndexIDMap.remove_ids` - exact removal on
the underlying flat index, not an approximation or a full rebuild). SQLite is
committed first; FAISS is only mutated and saved after that succeeds, so a
failure mid-operation can never leave metadata pointing at vectors that
don't exist. If the active document is deleted, the active selection falls
back to the most recently indexed remaining document. A confirmation step is
required before deletion.

## Extraction quality and repair

PyMuPDF's normal text extraction runs first and its output is kept whenever
it looks like prose. Each page is measured (mean word length, whitespace
ratio, control-character ratio), and a page that fails is re-extracted from
PyMuPDF's geometry-aware word segmentation, which finds word boundaries from
glyph positions instead of trusting the font's character mapping. The repair
is kept only when it measurably improves the page, so a page with genuinely
no separable words stays flagged rather than being quietly replaced. It is
deterministic and local - no OCR, no model - and it can only move separators,
never invent words.

This exists because of a real document in this project. One report is a
subsetted Calibri whose `/Encoding` maps character code 2 to glyph `/g3` and
code 3 to `/g882` - names that carry no Unicode meaning - while its
`/ToUnicode` CMap starts at code 0x25 and omits both. Extraction therefore
emitted U+0002 between every word and U+0003 for every hyphen. Those are not
whitespace to `str.split()`, so each line collapsed into one token: FTS5
could not match a keyword inside a 900-character token, the word-based
chunker saw a whole page as ~40 "words" and never split it, and the model was
left reading a wall of concatenated text. Repaired, that document's mean word
length goes from 74-86 characters to 5.2-5.3 and its page count of chunks
from 4 to 11.

Scanned or image-only PDFs are still unsupported and still fail with an
explicit error; OCR is deliberately not part of this pipeline. A damaged or
truncated PDF that yields no text is reported as damaged, not as a scan.

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
  (for example "give a short overview of every selected document" or "how do
  the selected documents differ?"). It changes nothing upstream - same FAISS search, same FTS5
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
  "What was the patient's eye colour?" is indistinguishable from an
  answerable factual question, and only missing evidence makes it
  unsupported.
- **Evidence-first prompting**
  ([`src/generation/prompt_builder.py`](src/generation/prompt_builder.py))
  tells the model the retrieved evidence is authoritative, that it must
  check it before declaring anything unavailable, and that describing "the
  context" is not answering the question. Each intent adds exactly one short
  ANSWER STYLE block; multi-part questions additionally get a numbered plan
  naming each requested component.
- **Completeness rules**, each narrowly gated and checked before it was
  kept (see the exploratory, single-document intervention study in
  [`research/FINAL_REPORT.md`](research/FINAL_REPORT.md)): multi-part plans
  ask for every item the evidence gives per part, and for drugs only when the
  question is about medications; a question asking for a number of items
  ("give 3-5 reasons") gets the list style; a comparison of two parts of one
  document (an initial vs a revised list) gets section-identity instructions.
- **Output budget.** 256 tokens by default; 768 for multi-document scope,
  multi-part questions, list-type and item-count questions, and
  single-document section comparisons. An answer that still hits the limit
  is labelled as cut off in the UI.
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

## Limitations

This is a **prototype**, not a medical device and not a diagnostic system.
It retrieves passages from documents you supply and asks a small local model
to summarise them. It makes no claim of clinical accuracy, and every answer
should be checked against the cited pages.

Known limitations, measured rather than assumed:

- **The answer is only as good as the retrieved evidence.** If the relevant
  passage is not retrieved, the model cannot recover it. Source attribution
  shows which passages were used; it is not evidence that the answer is
  correct.
- **Small-model arithmetic and counting are unreliable.** Asked how many
  studies a literature survey cites, both `llama3.2:3b` and a 7B model
  answered incorrectly (3 and 4, against a true 5). Counting across
  passages failed repeatedly in this project's tests.
- **Multi-column layouts are not reconstructed.** Text is read in PyMuPDF's
  block order, so margin commentary in a two-column document can be
  interleaved into body text (and from there into an answer).
- **Some omissions persist.** With the evidence retrieved and the
  instructions explicit, `llama3.2:3b` still omitted some items in the
  project's tests (an unnamed historical medication, a conditional one, one
  of several lines of reasoning) and sometimes mixed up which items two lists
  share. No prompt variant tested removed these; whether a different model
  would is untested.
- **No OCR.** Scanned or image-only pages carry no text layer. They are
  detected and reported at upload, and are simply absent from the index.
- **Broken PDF text layers can only partly be repaired.** A font whose
  encoding loses word boundaries is repaired from glyph geometry where the
  layout allows it; where glyphs genuinely abut, the page is flagged as
  unusable rather than silently indexed as garbage. Hyphens are not
  recovered by the repair.
- **Document references are lexical.** A reference resolves by filename,
  filename words, ordinal position, or an unambiguous recent antecedent. A
  paraphrase with none of those ("the cardiology one") will not resolve, and
  falls back to the current scope rather than guessing.
- **Conversation history informs generation, never retrieval.** A follow-up
  is retrieved on its own words, so a question that depends entirely on an
  earlier turn for its search terms may retrieve poorly.
- **Answers vary between runs.** Generation is sampled at a low but non-zero
  temperature, so the same question can produce different phrasing and
  occasionally a different (still grounded) choice of supporting detail.
- **Inline citations are unreliable.** The model is asked to cite evidence
  numbers but seldom does; the per-answer Sources list and evidence panel
  are always present and are the reliable record of what an answer used.
- **Ordinal references follow library order.** "The second document" means
  the second in-scope document as listed in the library (upload order), not
  the order a previous answer happened to discuss them in.
- **One question at a time.** While an answer is being written the input
  and the library controls are disabled. A question is recorded the moment
  it is sent; if answering it fails (for example, Ollama stops), the
  conversation keeps the question and offers Retry. Refreshing the page
  starts a new session and clears the conversation.
- **Single-machine, single-user.** There is no authentication, no
  multi-user isolation, and no audit log. Do not expose it on a network you
  do not control.

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
  but the finished reports and aggregate metrics are.
  [`research/FINAL_REPORT.md`](research/FINAL_REPORT.md) is the current
  report: dense vs. hybrid vs. production-path retrieval with
  Holm-corrected paired tests, the chunk-level ground-truth audit,
  descriptive end-to-end results (research harness, not the app's path), a
  failure taxonomy, an exploratory answer-completeness intervention study,
  safety probes, a claim audit and reproduction notes. Companion documents:
  [`CONTRIBUTION.md`](research/CONTRIBUTION.md) (research questions, claims
  and explicit non-claims), [`ARCHITECTURE.md`](research/ARCHITECTURE.md)
  (component-level design), [`RESULTS_TABLE.md`](research/RESULTS_TABLE.md)
  (paper-ready tables) and [`DEMO_SCRIPT.md`](research/DEMO_SCRIPT.md) (a
  review walkthrough).
  [`research/REPORT.md`](research/REPORT.md) is the earlier generated report
  for publication runs 1-4, which predate the FTS5 Porter-stemming change
  (its hybrid MRR/nDCG are lower than the current index's); it is kept as a
  historical record. Re-run the retrieval benchmark with
  `python research/counterbalanced_retrieval_benchmark.py` (writes a new
  timestamped file under `research/results/`).
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
│   ├── generation/         # Intent rules, prompt building, answer validation, Ollama client
│   ├── evaluation/         # Metrics, aggregation and product QA grading
│   └── ui/                 # Icon set for the Streamlit app
├── scripts/                # ingest / query / evaluate / build-corpus / product QA CLIs
├── tests/                  # pytest suite
├── research/
│   ├── FINAL_REPORT.md     # Current research report
│   ├── CONTRIBUTION.md     # Research questions, contributions, non-claims
│   ├── ARCHITECTURE.md     # Component-level system description
│   ├── RESULTS_TABLE.md    # Paper-ready result tables
│   ├── DEMO_SCRIPT.md      # Review demo walkthrough
│   ├── REPORT.md           # Historical generated report (runs 1-4)
│   ├── results/            # Frozen retrieval benchmark results
│   ├── figures/            # Report figures
│   └── *.py                # Benchmark, audits, diagnostic studies, report generator
├── data/
│   ├── raw_pdfs/.gitkeep
│   └── research_corpus/raw/.gitkeep
└── .claude/launch.json     # Dev-server config for the Claude Code browser preview
```
