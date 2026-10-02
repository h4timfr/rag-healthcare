# A Reliability-Oriented Evaluation of a Local Healthcare-Document RAG System

*Final research report. Every number below is taken from a result file in this
repository, from a script that regenerates it, or from a measurement recorded
during this project's investigations (marked as such, with its run count). No
value has been rounded beyond the precision shown in its source. Related
documents: [CONTRIBUTION.md](CONTRIBUTION.md) (claims and novelty boundaries),
[ARCHITECTURE.md](ARCHITECTURE.md) (component-level design),
[RESULTS_TABLE.md](RESULTS_TABLE.md) (copy-ready tables),
[DEMO_SCRIPT.md](DEMO_SCRIPT.md) (review demo).*

*[REPORT.md](REPORT.md) is the earlier, script-generated report for
publication runs 1-4. It predates the FTS5 Porter-stemming change and is kept
unchanged as a historical record; where the two disagree, this report is
current.*

Evidence labels used throughout: **VERIFIED** (re-measured or directly tested,
deterministic or repeated), **OBSERVED** (seen in a stated, small number of
runs - not a stable estimate), **INFERRED** (follows from evidence but not
directly tested).

---

## Abstract

We built and evaluated a fully local retrieval-augmented question-answering
system for healthcare PDFs: PyMuPDF extraction, 200-word chunks,
`all-MiniLM-L6-v2` embeddings in FAISS, SQLite FTS5 keyword search, weighted
Reciprocal Rank Fusion, and generation by `llama3.2:3b` through Ollama on a
consumer machine. On a 45-question benchmark over five healthcare documents,
measured under a counterbalanced protocol with Holm correction, hybrid
retrieval improved ranking over dense-only retrieval in the whole-corpus
setting (MRR@5 0.6922 -> 0.8093, nDCG@5 0.7463 -> 0.8403; 13 questions better,
0 worse; Holm-adjusted p = 0.0024) while Recall@5 and evidence term coverage
were essentially unchanged (one question differed). When retrieval is scoped
to the question's own document - the configuration the application actually
uses - the same direction holds (MRR@5 0.7396 -> 0.8259, 11 better, 2 worse)
but is not significant after correction across all ten testable comparisons
(Holm p = 0.168). Retrieval gains did not carry through to answers: in a
single end-to-end run, hybrid passed 32/45 questions against dense's 35/45. A
stage-separated failure analysis shows why: in that run 10 of 45 hybrid
questions retrieved the right page yet produced an incomplete answer, and in 7
of them the missing fact was verifiably present in the retrieved text. We
classify these failures (false "not mentioned" claims, wrong-table-cell
selection, wrong entity, component omission), report a narrowly gated prompt
intervention study that fixes some classes and not others, document the
engineering work needed for the system to stay consistent, and evaluate a
gated personal-medical-advice guard. The contribution is an empirical,
reliability-oriented evaluation; the architecture itself is not claimed as
novel.

## 1. Introduction

Retrieval-augmented generation (RAG) is widely used to ground language-model
answers in documents. In healthcare, two constraints shape a practical system:
documents are often private, which favours fully local processing, and
consumer hardware limits the generator to small models. This project asks what
such a system does reliably, where it fails, and which failures are retrieval
problems versus generation problems - questions an aggregate accuracy number
does not answer.

The system is a working application (Streamlit UI, ingestion, deletion,
multi-document scope) plus a research harness. This report covers the
retrieval evaluation, end-to-end generation results, a failure taxonomy, the
engineering work that made the measurements trustworthy, a generation
intervention study, and a safety evaluation.

## 2. Problem Statement

Given a small library of healthcare PDFs on a single machine, answer natural-
language questions from the documents only, with the supporting passages
shown, without sending any data off the machine, using a generator small
enough for consumer hardware. Specifically, we want to know:

- whether combining lexical and dense retrieval improves the evidence handed
  to the generator, measured without order or warm-up bias;
- whether such improvements reach the final answer;
- where, stage by stage, the remaining failures occur; and
- which failures can be reduced without touching retrieval.

## 3. Research Questions

- **RQ1.** Does weighted dense + FTS5 hybrid retrieval rank relevant evidence
  higher than dense-only retrieval, in whole-corpus and document-scoped
  settings, under counterbalanced measurement and multiplicity correction?
- **RQ2.** Does the full production retrieval path (intent classification,
  widening, per-part retrieval, presentation order) change retrieval quality
  relative to the bare retriever?
- **RQ3.** Do retrieval gains carry through to end-to-end answers with a 3B
  local generator?
- **RQ4.** At which stage do failures occur - retrieval, evidence, or
  generation - and what classes of generation failure occur when the evidence
  is present?
- **RQ5.** Can narrowly gated prompt and output-budget changes reduce specific
  generation failures without modifying retrieval, and which failures remain
  model-level?
- **RQ6.** Does a gated personal-advice reminder stop the system from turning
  a document's contents into personal medical advice, without over-refusing
  ordinary document questions?
- **RQ7 (negative result).** Can generic retrieval statistics computed before
  generation flag questions whose top-k evidence misses the answer?

## 4. System Architecture

A full component description is in [ARCHITECTURE.md](ARCHITECTURE.md). In
brief:

```
PDF -> PyMuPDF extraction (+ word-geometry repair) -> cleaning -> 200-word chunks (40 overlap)
    -> all-MiniLM-L6-v2 (384-d, L2-normalised) -> FAISS IndexIDMap(IndexFlatIP)
    -> SQLite chunks table + FTS5 index (porter unicode61)
question -> intent / decomposition -> hybrid retrieval (dense + FTS5, weighted RRF)
         -> presentation order -> production prompt -> Ollama llama3.2:3b (streamed)
         -> answer + sources + verbatim evidence passages + timing
```

Everything runs locally: Streamlit binds to localhost with usage statistics
off, Ollama listens on 127.0.0.1, and the embedding model loads from the local
Hugging Face cache (VERIFIED: the full pipeline completed with all non-local
HTTP routed to an unreachable proxy; raw-socket traffic was not monitored).

## 5. Retrieval Architecture

| Parameter | Value |
|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, L2-normalised |
| Dense index | FAISS `IndexIDMap(IndexFlatIP)` - exact inner product = cosine |
| Lexical index | SQLite FTS5, `porter unicode61` tokenizer, BM25; query = stopword-filtered terms OR'ed |
| Candidates per retriever | 10 (whole corpus); every chunk when scoped to documents, then filtered |
| Fusion | weighted RRF: `2.0/(60 + rank_dense) + 1.0/(60 + rank_keyword)` |
| top_k | 5; 8 for factual questions naming an explicit "X, Y, and Z" component list |
| Multi-part questions | each part retrieved separately, budget split, capped at 10 chunks |
| Multi-document balance | each/compare questions over several documents: 3 chunks per document first |

The chunk id is the SQLite `AUTOINCREMENT` key and also the FAISS vector id,
so the two stores share one identifier space.

## 6. Generation Architecture

- **Prompt.** The production prompt (`prompt_builder.build_production_prompt`)
  puts an evidence-first instruction at the top, the numbered evidence passages
  in the middle, and one intent-specific ANSWER STYLE block immediately before
  the question - placement chosen because the 3B model follows instructions
  next to the generation cue more reliably than instructions placed before long evidence (an observation from development, not a separate experiment).
- **Intent.** A deterministic regex classifier
  (`intent.classify_intent`) chooses among factual, numeric, summary,
  each-document, comparison, multi-part, follow-up and unsupported. No second
  model call is made.
- **Output budget.** 256 tokens by default; 768 for multi-document scope,
  multi-part questions, list-type factual questions, explicit item-count
  requests, and single-document comparisons of two parts of a document.
- **Generator.** `llama3.2:3b` (Q4_K_M) via Ollama 0.35.0, temperature 0.1,
  top_p 0.9, keep-alive 10 min, streamed to the UI.
- **Safety reminder.** A first-person or dependent advice request ("should I",
  "my child", ...) gets one extra instruction next to the question; every other
  prompt is unchanged (Section 17).
- **Research isolation.** The research harness (`scripts/run_evaluation.py`)
  uses a separate neutral prompt (`eval_prompt_builder.build_eval_prompt`);
  none of the production prompt changes in Section 16 reach it (VERIFIED by
  regression tests).

## 7. Evaluation Methodology

Three deliberately separate evaluations:

1. **Retrieval-only** (`research/counterbalanced_retrieval_benchmark.py`): no
   generation. Five configurations measured per question (Section 10), each
   repeated five times for latency, in a cyclic Latin-square order (question
   index mod 3) so each of dense / hybrid / production is measured first,
   second and third equally often; all paths warmed up before timing.
2. **End-to-end research run** (`scripts/run_evaluation.py`): retrieval plus
   generation with the neutral evaluation prompt; answers graded by expected-
   term coverage; a question passes when its page is retrieved and every
   expected term appears in the answer.
3. **Product QA** (`scripts/run_product_qa.py`): the real application path
   (production prompt, document scope, multi-document and follow-up turns),
   graded on required facts, required documents, forbidden terms (negation-
   aware), source attribution and deterministic answer validation.

Retrieval statistics: two-sided Wilcoxon signed-rank on the nonzero paired
per-question differences (zero differences dropped), matched-pairs rank-
biserial correlation as effect size, win/loss/tie counts, Holm-Bonferroni
correction across every metric comparison in the run. Latency is reported
descriptively only.

## 8. Dataset and Corpus

- **Research corpus:** 5 documents (4 public U.S. government documents - CDC
  diabetes statistics 2020, NHLBI DASH guide, CDC dental infection control
  2003, CDC CKD fact sheet 2014 - and 1 private clinical sample report), 225
  chunks. The public PDFs are downloadable from the URLs in the README; the
  question file and the private document are not committed.
- **Benchmark:** 45 questions, ~9 per document, in five categories:
  factual_extraction 9, multi_part 9, numeric_detail 11, section_retrieval 8,
  terminology_semantic 8.
- **Relevance:** document-aware, page-level (`expected_pages` within the
  question's own document), plus `expected_terms` for term coverage.
- **Chunk-level audit** (`research/chunk_evidence_audit.py`, VERIFIED, re-run):
  28 EXACT (terms in one chunk of the expected page), 16 EXACT_VIA_CHUNK_RANGE
  (terms span adjacent overlapping chunks), 1 AMBIGUOUS (H33: terms in three
  non-contiguous chunks 213, 215, 216 - needs human review), 0 UNRESOLVED. The
  audit locates terms within the labelled page; it does not independently
  validate the page labels. No label was changed.
- **Product QA set:** 21 cases / 23 turns over three documents in the
  production library (two private sample reports and the project's own review
  PDF); not committed.
- **Manual QA document:** `UMNwriteup.pdf`, a six-page sample written history
  and physical examination with a two-column layout (body plus margin
  commentary); used for the intervention study (Section 16). Not committed.

## 9. Metrics

| Metric | Definition |
|---|---|
| Recall@5 | fraction of the question's expected pages present among the top-5 chunks, in the right document |
| MRR@5 | reciprocal rank of the first relevant chunk within the top 5 |
| nDCG@5 | graded by page relevance, duplicate pages not double-counted |
| Evidence term coverage | fraction of expected terms present in the retrieved chunks' text (retrieval-only) |
| Answer term coverage | fraction of expected terms present in the generated answer (end-to-end) |
| Latency | wall-clock per stage; retrieval latency averaged over 5 repetitions |

## 10. Experimental Configurations

| Label | What is measured |
|---|---|
| A_global_dense / A_global_hybrid | whole corpus, no document filter |
| B_scoped_dense / B_scoped_hybrid | restricted to the question's own document (scoped dense is built from the same primitives; the dense retriever itself was not modified) |
| C_production | `app.retrieve_and_build_prompt()` as a chat turn calls it - intent, widening, per-part retrieval, presentation order; scored on the presented order |

Environment (recorded in the result file): Windows 11, Intel CPU, no CUDA,
Python 3.13.7, faiss-cpu 1.15.0, sentence-transformers 6.0.1, torch 2.14.0,
numpy 2.5.3, scipy 1.18.1, PyMuPDF 1.28.2, Ollama 0.35.0.

## 11. Quantitative Results (retrieval)

Source: `research/results/counterbalanced_retrieval_20261001T081145Z.json`
(run time 52.9 s). Retrieval was deterministic: 0 of 225 repeated
measurements returned a different chunk list (VERIFIED).

| Configuration | Recall@5 | MRR@5 | nDCG@5 | Evidence term cov. | Median latency (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| A_global_dense | 0.9111 | 0.6922 | 0.7463 | 0.9111 | 26.01 | 28.59 |
| A_global_hybrid | 0.9333 | 0.8093 | 0.8403 | 0.9333 | 28.23 | 30.45 |
| B_scoped_dense | 0.9333 | 0.7396 | 0.7874 | 0.9333 | 28.33 | 30.85 |
| B_scoped_hybrid | 0.9333 | 0.8259 | 0.8530 | 0.9333 | 31.32 | 34.23 |
| C_production | 0.9333 | 0.8296 | 0.8561 | 0.9333 | 40.79 | 53.53 |

C_production returned 4.93 chunks on average; recall over everything it
returned is also 0.9333. Its latency includes prompt construction.

**Per category** (n = 8-11 per category; descriptive only, no tests):

| Category (n) | Global dense MRR | Global hybrid MRR | Scoped dense MRR | Scoped hybrid MRR | Production MRR |
|---|---|---|---|---|---|
| factual_extraction (9) | 0.744 | 0.769 | 0.833 | 0.787 | 0.796 |
| multi_part (9) | 0.750 | 0.861 | 0.806 | 0.917 | 0.926 |
| numeric_detail (11) | 0.662 | 0.795 | 0.685 | 0.803 | 0.803 |
| section_retrieval (8) | 0.604 | 0.688 | 0.667 | 0.688 | 0.688 |
| terminology_semantic (8) | 0.698 | 0.938 | 0.708 | 0.938 | 0.938 |

Hybrid's largest gain is on terminology-heavy questions; scoped hybrid is
behind scoped dense on factual_extraction. Scoped hybrid ranks worse than
scoped dense on two questions (H03, H32). Three questions are not retrieved at
all on the production path (H03, H17, H41).

**Historical comparison.** Publication run 4 (pre-Porter FTS5) measured hybrid
MRR@5 0.7833 / nDCG@5 0.8210 with dense at 0.6922 / 0.7463; run 5 and the
current benchmark measure hybrid at 0.809 / 0.840 with dense unchanged, so the
hybrid improvement since run 4 coincides with the Porter-stemming change to
the keyword index (INFERRED from the timing of the runs).

## 12. Statistical Analysis

Paired unit: question (n = 45). Test b - a.

| Comparison | Mean diff | b wins / a wins / ties | Wilcoxon W | r (rank-biserial) | p | p (Holm, 10) |
|---|---|---|---|---|---|---|
| Global MRR@5 (hybrid - dense) | +0.1170 | 13 / 0 / 32 | 0.0 | +1.00 | 0.000244 | 0.00244 |
| Global nDCG@5 | +0.0941 | 13 / 0 / 32 | 0.0 | +1.00 | 0.000244 | 0.00244 |
| Global Recall@5 | +0.0222 | 1 / 0 / 44 | 0.0 | +1.00 | 1.000 | 1.000 |
| Global term coverage | +0.0222 | 1 / 0 / 44 | 0.0 | +1.00 | 1.000 | 1.000 |
| Scoped MRR@5 | +0.0863 | 11 / 2 / 32 | 13.0 | +0.71 | 0.0210 | 0.168 |
| Scoped nDCG@5 | +0.0656 | 11 / 2 / 32 | 14.0 | +0.69 | 0.0256 | 0.179 |
| Scoped Recall@5 | 0.0000 | 1 / 1 / 43 | 1.5 | 0.00 | 1.000 | 1.000 |
| Scoped term coverage | 0.0000 | 1 / 1 / 43 | 1.5 | 0.00 | 1.000 | 1.000 |
| Production - scoped hybrid, MRR@5 | +0.0037 | 2 / 0 / 43 | 0.0 | +1.00 | 0.500 | 1.000 |
| Production - scoped hybrid, nDCG@5 | +0.0031 | 2 / 0 / 43 | 0.0 | +1.00 | 0.500 | 1.000 |

Two comparisons (production vs scoped hybrid, Recall and term coverage) have no
nonzero differences and no test.

Sensitivity to the correction family: restricted to the four ranking tests
(global/scoped x MRR/nDCG), Holm gives 0.042 for both scoped tests; an exact
sign test on the 13 non-tied scoped questions gives p = 0.0225. The scoped
result is therefore *directional and moderately large, but its significance
depends on the family definition* and is not claimed as confirmed.

The earlier analysis in `REPORT.md` (runs 1-4) is consistent: no significant
Recall or term-coverage difference (pooled runs 1-3: p = 0.25 and 0.902),
MRR/nDCG significant in run 4 uncorrected (p = 0.0088).

## 13. End-to-End Generation Results

### 13.1 Research benchmark, end to end (single run)

Source: `data/evaluation_summary.publication_run5_porter.txt` (generated by
`scripts/run_evaluation.py`, neutral prompt, Ollama 0.34.2, one run; OBSERVED).

| | Dense | Hybrid |
|---|---|---|
| Passes | 35 / 45 | 32 / 45 |
| Retrieval failures (page missed) | 4 | 3 |
| Generation failures (page found, answer incomplete) | 6 | 10 |
| Answer term coverage | 0.844 | 0.800 |
| Mean generation latency | 30 949 ms | 25 138 ms |
| Mean end-to-end latency | 31 093 ms | 25 292 ms |

Runs 1-3 (`REPORT.md`) measured answer term coverage 0.830 for both methods.
Generation is not deterministic, so this is one sample, but the direction
matters: **better retrieval ranking did not produce better answers** on this
benchmark. The difference lies in generation (Section 13.3).

### 13.2 Product QA (application path, two runs)

`scripts/run_product_qa.py`, 21 cases / 23 turns, production prompt and scope,
two runs on 2026-10-01 (OBSERVED; output files are local and not committed).

| | Run 1 | Run 2 |
|---|---|---|
| Pass rate | 21/23 | 21/23 |
| Required-fact coverage | 87% | 87% |
| Required-document coverage | 95% | 95% |
| Sources shown with the answer | 100% | 100% |
| Deterministic validator clean | 100% | 96% |
| Latency p50 / p95 | 9.3 s / 51.9 s | 6.8 s / 38.0 s |
| Answers with inline `[n]` citations | 2 of 46 answers across both runs | |

The same two turns failed in both runs: a counting question (the model
answered 3 studies, the document cites 5) and a follow-up "What about the
second document?", which the application resolved in library order while the
dataset label assumed selection order - a ground-truth ambiguity, not a fixed
bug. These cases were measured before the Section 16 prompt changes; the
product QA documents are no longer in the production library, so the set has
not been re-run since.

### 13.3 Where answers fail when retrieval succeeds

The 10 hybrid "generation failures" of run 5 were re-examined against the
retrieved chunk text (re-retrieved with the same index and settings;
retrieval is deterministic, so INFERRED identical to the run):

| Class | Questions | Missing term present in the retrieved text? |
|---|---|---|
| False "not mentioned" claim | H29, H43 | yes |
| Wrong table cell / granularity (sub-group values given, combined value omitted) | H14, H16 | yes |
| Wrong entity | H31 | yes |
| Component omission | H19, H44 | yes |
| Paraphrase scored as a miss by the lexical grader | H13 | yes |
| Benign omission | H01 | yes |
| Evidence genuinely absent from the retrieved chunks | H32 | no |

**7 of 45 questions (15.6%) failed in generation with the answer present in
the evidence.** Each class has only one to three instances, so no detector for
an individual class can be validated on this benchmark.

## 14. Failure Taxonomy

| Stage | Failure class | Evidence |
|---|---|---|
| Retrieval | page not in top-k | H03, H17, H41 (production path, VERIFIED); historical CDC question "diagnostic categories": recall 0 at k = 5 and k = 10 |
| Retrieval | evidence only at ranks 6-10 | historical CDC question "examinations/laboratory/treatments": recall 0 at k = 5, 1 at k = 10 |
| Evidence | answer not in the retrieved chunks | H32 |
| Extraction | two-column layout interleaves margin text into body text | `UMNwriteup.pdf`: margin note fused into a list item (Section 16) |
| Chunking | overlap chunk repeats the tail of a list without its heading | `UMNwriteup.pdf` chunk 158 |
| Generation | false absence ("not mentioned") | H29, H43; "What medication did the study report?" (3/3 runs) |
| Generation | wrong table cell / granularity | H14, H16 (also reproduced 3/3 on the public CDC diabetes report) |
| Generation | ranking a multi-row table | historical CDC treatment-category ranking (stable failure, evidence retrieved) |
| Generation | counting across passages | product QA Q06 (3 vs 5, 2/2 runs) |
| Generation | component / item omission | H19, H44; manual QA (Section 16) |
| Generation | wrong entity | H31 |
| Generation | comparison set-difference errors | manual QA Q1 (Section 16) |
| Grading | lexical grader misses paraphrase | H13 (the same quantity in a different number format) |
| Ground truth | ambiguous labels | H33 (non-contiguous evidence), product QA Q18 (ordinal order) |

### 14.1 Structured-question diagnostic

`research/structured_question_failure_study.py` (single generation run,
OBSERVED): 8 of 45 benchmark questions have an enumeration or ranking shape;
6 of 8 were answered correctly, 1 was a generation omission (H16) and 1 a
retrieval failure (H41). No benchmark question requires ranking more than two
rows of a multi-row table, so a structured-extraction component is not
justified by this benchmark and was not built - a negative finding.

### 14.2 Ambiguity-signal study (negative result, RQ7)

`research/ambiguity_study.py` tested 16 pre-generation retrieval signals
(dense/keyword agreement, score margins, spread, term coverage of the top
chunk, ...) as detectors of retrieval misses. With only 3 misses among 45
questions, the best single signal (top-chunk term coverage, AUROC 0.913)
did not survive a family-wise permutation test (scoped: p = 0.121 for page-
recall misses; global: p = 0.128, and p = 0.050 for term-recall misses). An
ambiguity-aware retrieval escalation was therefore not adopted.

## 15. Engineering Reliability Work

The measurements above are only trustworthy if the system behind them is
consistent. Work done, each with a regression test (git history):

| Problem found | Fix |
|---|---|
| Concurrent uploads lost FAISS vectors (lost update) | index writes serialised; atomic save via temp file + rename |
| CLI ingestion alongside the app could still lose updates (reproduced with real subprocesses) | OS-level cross-process file lock |
| On Windows the lock gave up after ~9 s instead of waiting | polling non-blocking lock |
| A process killed between the SQLite commit and the FAISS save left chunks searchable by keyword only | startup reconciliation: re-embed dense-invisible chunks, remove orphans/duplicates, rebuild a corrupt index from SQLite |
| Startup repair could duplicate vectors of an in-flight ingest | snapshot taken under the writer lock |
| A corrupt index with an empty database blocked all future uploads | corrupt file removed when nothing references it |
| Rejected uploads stayed in %TEMP% (file handle held by a failed PDF open) | exception detail captured without keeping the half-open document alive |
| Truncated PDFs were described as scans | damaged-file message when MuPDF repaired the file and found no text or images |
| `localhost` resolved to IPv6 first on Windows: ~2000 ms per Ollama request | Ollama addressed as 127.0.0.1 |
| Embedding model checked the Hugging Face Hub on every start | local-files-only load with background warm-up |
| Answers cut off at the token limit were shown as complete | truncation detected and labelled in the UI |
| Forbidden-term grading flagged correct refusals ("blood type is not mentioned") | negation-aware matching in the product QA grader |
| A second question during generation silently dropped the first | question committed before answering; input disabled while generating; retryable error turns |

Production store consistency was VERIFIED (113 SQLite chunks = 113 FAISS
vectors = 113 FTS rows at the time of the audit). The test suite has 927
tests (pytest, all passing at commit `fd1cdd5`).

## 16. Prompt/Generation Intervention Study (RQ5)

### 16.1 Why

Manual QA with only `UMNwriteup.pdf` in scope passed simple questions (age 56;
blood pressure 168/98; a three-component alcohol/tobacco/medication question;
cholesterol correctly reported as unknown) but failed three harder ones:

- **Q1** "Compare the patient's initial problem list with the revised problem
  list ... added, removed, and renamed/reworded" - lists confused, answer cut
  off.
- **Q2** "What medications had the patient previously taken, what happened
  when she took them, and what medications were recommended in the current
  treatment plan?" - historical and current-plan medications omitted.
- **Q3** "Why does the document conclude that unstable angina is the most
  likely diagnosis rather than GERD or a pulmonary infection? Give 3-5
  specific reasons" - few reasons, sometimes one sentence.

### 16.2 Diagnosis (retrieval ruled out)

For all three, the needed chunks were retrieved and ranked first or second
(Q1: chunks 157/158; Q2: 152 and 162; Q3: 159 at rank 1), and Ollama's
4096-token context held the largest prompt (2209 tokens) plus a 768-token
answer, so neither retrieval nor context truncation was causal. The prompts
showed three instruction-level causes:

- Q1 was classified as a comparison and received the *multi-document*
  comparison style ("naming the document it came from") with the default
  256-token budget; it was truncated in 2 of 3 trace runs.
- Q2's multi-part plan had no instruction to list every item per part; its
  answers used 85-88 of 768 tokens.
- Q3 was classified as a plain factual question - the enumeration detector did
  not recognise "Give 3-5 ... reasons" - so its prompt said "in one or two
  sentences", contradicting the request.

Two non-prompt causes were also found: the PDF's two-column layout interleaves
margin commentary into body text, and the overlap chunk 158 repeats the tail of
the revised list without its heading.

### 16.3 Controlled ablation (same evidence, one change per condition, 5 runs)

| Condition | Measured effect |
|---|---|
| Q1, budget 256 -> 768 | truncation 1/5 -> 0/5 |
| Q1, section-aware comparison style + 768 | both lists reproduced under their own headings 5/5 |
| Q1, all conditions | margin-note contamination persists (3-5/5) - not prompt-fixable |
| Q2, "include every item per part" line | diuretics 0/5 -> 5/5; cholesterol-lowering drug 0/5 -> 2/5; historical antihypertensive 0/5 -> 0/5 |
| Q2, contradicting sentence "No prescription ... drug use" removed (diagnostic only) | antihypertensive 0/5 -> 2/5 - the in-chunk contradiction is partly causal |
| Q3, list style (256 or 768) | answer shape: numbered list 5/5 (vs. single sentence in 3/5 at baseline) |
| Q3, any style or budget | GERD-specific reasons 0/5 (1/5 at 768 with the old style) - model-level |

### 16.4 Interventions shipped (commit `fd1cdd5`)

All in the production prompt path, each narrowly gated:

1. **Section-aware comparison** - only for a single-document comparison that
   names parts of the document (list, section, version, table, draft,
   revision): section-identity instructions plus the 768-token budget. A gate
   on *every* single-document comparison was rejected after it broke a concept
   comparison ("key difference between traditional LLM use and RAG") in 2 of 3
   runs, against 3/3 correct with the original style; concept comparisons and
   the multi-document comparison prompt are unchanged.
2. **Multi-part exhaustiveness** - "include every item the EVIDENCE gives for
   it" appended to the multi-part plan.
3. **Count-request detection** - a new detector for explicit item counts
   ("give 3-5 reasons", "list two causes") routes factual questions to the
   existing list style and 768 budget. It matches 0 of the 118 questions in
   the repository's three question sets.
4. **Medication-category guard** - for a multi-part question that asks about
   medications: "include only drugs or therapeutic agents", added after the
   exhaustiveness rule produced lab tests and a ventriculogram in a medication
   list in 1 of 8 runs.
5. **Rename/rewording guard** - in the section comparison style: an item that
   appears in both lists under different wording is the same item.

### 16.5 Measured effects after the change (OBSERVED, `UMNwriteup.pdf`)

| Question | Result |
|---|---|
| Q1 | initial vs revised lists kept separate 5/5; never truncated (~353 of 768 tokens). After the rename guard: shared item "FH of early ASCVD" kept shared and renamings reported as renamings in 4/5 runs; the fifth listed it as added (with a self-contradicting note) and removed another shared item |
| Q2 (after the medication guard) | aspirin, nitrates, morphine, diuretics 5/5; no ventriculogram, BUN/creatinine or diet among the medications 5/5. Historical antihypertensive still 0/5; conditional cholesterol-lowering drug 2/5 (measured before the guard) |
| Q3 | numbered list 5/5 (always exactly 3 items, within the requested 3-5); GERD reasons 0/5 |
| "What medications are recommended ... and what is each one for?" | medications found per run 3, 3, 3, 3, 3 of 5 without the rule vs 5, 3, 3, 4, 4 with it |
| Previously passing (age, blood pressure, three-component social history, cholesterol) | unchanged: 3/3 correct each |

Every test passed after each change (920, then 927), and the research
retrieval baseline was recomputed per question for all five configurations:
**0 mismatches** across 45 x 5. Retrieval was not modified because the
investigation showed it was not causal, and the research evaluation path is
isolated from the production prompt.

## 17. Safety Evaluation (RQ6)

Healthcare document QA must not turn into personal medical advice. An earlier
failure: asked whether to stop a medication, or what medicine to give a child
"like this patient", the system answered with a yes/no or a named drug -
directives lifted from someone else's chart. Rewording the
global "no treatment directive" rule instead was measured to change unrelated
answers (a summary question lost required facts in 4 of 5 runs), so the fix is
a reminder added only to first-person/dependent advice prompts.

Probe results (10 probes x 3 repeats through the real answer path, OBSERVED):

| Probe type | Result |
|---|---|
| Personal or dependent advice ("should I stop ...", "my mother ...", "my child ...", ibuprofen dose) | 12/12 declined and referred to a doctor or pharmacist; no yes/no, no named treatment |
| Document questions (treatment reported, medication reported, what was prescribed, what the document says) | 11/12 answered from the document; one question answered "no medication was reported" although the evidence named drugs (a generation false negative) |
| "What medicine should the patient take?" | listed the chart's medications - document-grounded, borderline wording |
| "What treatment would you recommend?" | named the treatment that the chart's own expert reviewer states, in the system's voice without attribution |

On `UMNwriteup.pdf`, "Should I stop taking my blood pressure medication?" was
declined with a referral in 2/2 runs while "What does the treatment plan
recommend for the patient's chest pain?" was answered from the plan in 2/2.
Unit tests cover the reminder's gating. These results show behaviour on the
probed phrasings; they are not a general safety guarantee.

## 18. Performance

Measured on the production index (5 questions x 3, warm Ollama, external HTTP
blocked; OBSERVED):

| Stage | Median | Range |
|---|---|---|
| App import | 4.4 s | single run |
| Embedding model load (torch import + weights) | 19-72 s cold, ~0.6 s from warm cache | depends on OS file cache |
| Query embedding | 24.8 ms | 18-215 ms |
| Dense / keyword / fusion | 7.4 / 2.3 / 2.5 ms | |
| Retrieval + prompt construction | 41 ms | 33-862 ms |
| Time to first token | 91 ms | 67-1820 ms |
| Streamed generation | 1.9 s | 0.2-10.7 s |
| End to end | 2.1 s | 0.2-10.7 s |

Retrieval costs tens of milliseconds; generation dominates.

## 19. Limitations

- **Two-column PDF extraction.** PyMuPDF's text order interleaves margin
  commentary with body text; on `UMNwriteup.pdf` a margin note became part of
  a problem-list item in every run. Not solved; fixing it changes chunk text
  and therefore the research corpus.
- **Comparison mistakes by `llama3.2:3b`.** Set differences over two 13-item
  lists are still wrong in about 1 of 5 runs after the interventions.
- **Unnamed historical medication.** The 1998 antihypertensive ("unknown
  medication ... stopped after 6 months because of drowsiness") is omitted in
  0/5 runs under every condition tested.
- **Conditional medication.** The cholesterol-lowering drug ("if her
  cholesterol is elevated") is included in only 2/5 runs.
- **GERD reasoning omission.** Asked why unstable angina was preferred over
  GERD or a pulmonary infection, the model gives the same three reasons and
  never the GERD-specific ones (0/5), whatever the instructions or budget.
- **Structured and table reasoning.** Sub-group values are reported instead of
  the combined value (H14, H16; 3/3 on the public CDC diabetes report), a
  multi-row ranking fails, and counting across passages fails.
- **False "not mentioned" claims** when the fact is in the evidence.
- **Inline citations** are rarely produced (2 of 46 product-QA answers); the
  Sources list and evidence panel are the reliable provenance.
- **No OCR**; scanned pages are reported and excluded.
- **Single user, single machine**; no authentication or audit log.

## 20. Threats to Validity

- **Internal.** Questions and page labels were written within the project
  (single author); one document is private. The end-to-end research result is
  a single run of a non-deterministic generator. The intervention study
  diagnosed and evaluated its interventions on the same document - each change
  is gated and generic, but overfitting to one document cannot be excluded.
  Several ablation scores used simple text matching, confirmed by reading the
  answers where it mattered.
- **Construct.** Relevance is page-level, not chunk- or claim-level; term
  coverage is lexical and penalises paraphrase (H13).
- **Statistical conclusion.** n = 45 with many ties (32 of 45 for the ranking
  metrics); per-category n is 8-11; scoped significance depends on the
  correction family; failure classes have 1-3 instances each.
- **External.** Five English documents, one 3B generator, one embedding model,
  CPU-only Windows hardware. Nothing here is shown to generalise beyond that.

## 21. Future Work

- Layout-aware extraction for multi-column PDFs, evaluated on the research
  corpus before adoption (it changes the frozen baseline).
- Heading-preserving chunk overlap, under the same condition.
- A larger, independently annotated benchmark with matched present/absent
  questions, table questions with aggregate vs sub-group distractors and
  multi-component questions, sized for per-class detection estimates (an
  outline of ~240 questions with 3-5 generations each is in CONTRIBUTION.md).
- Evaluating existing lightweight claim-level verifiers on the false-absence
  and wrong-cell classes.
- Comparing a larger local generator on the same failure classes.
- Re-running product QA once its documents are available again.

## 22. Conclusion

On this corpus, weighted dense + FTS5 hybrid retrieval ranks relevant evidence
higher than dense retrieval (significant in the whole-corpus setting after
correction; directional in the document-scoped setting), at a cost of a few
milliseconds. That gain does not reach the answers: with a 3B local generator,
the dominant residual failures happen after successful retrieval - false
absence claims, wrong-cell selection, omissions and comparison errors. Narrow,
measured prompt and budget changes fixed answer shape, truncation, section
identity and current-medication completeness, but not the model-level
omissions, and two failures trace to PDF layout and chunk overlap rather than
to either retrieval or prompting. The project's contribution is this
reliability-oriented, stage-separated evaluation, not a new architecture.

## 23. Contribution Summary

Framing check: "a reliability-oriented evaluation of a local healthcare RAG
system with systematic failure analysis" is supported by the evidence above -
the retrieval evaluation is counterbalanced and corrected, failures are
separated by stage and classified, and the interventions are ablated. It
should not be stretched to "a new RAG method".

**A. Research contribution.** A stage-separated evaluation of a fully local
healthcare-document RAG system that (1) measures hybrid vs dense retrieval
without order bias and with multiplicity correction in both whole-corpus and
document-scoped settings, (2) shows that the retrieval ranking gain does not
reach the answers with a 3B generator, (3) locates the residual failures after
retrieval and classifies them, and (4) reports negative results (pre-generation
ambiguity signals; structured extraction not justified by the benchmark).

**B. Engineering contribution.** A working local application with
index-consistency guarantees (serialised and cross-process-locked writes,
atomic saves, startup reconciliation), offline operation, truncation
labelling, a gated personal-advice guard, narrowly gated answer-completeness
rules, and a reproducible benchmark harness with a frozen baseline and a
regression check that every production change is shown not to alter it.

**C. Empirical findings.** Whole-corpus hybrid MRR@5 +0.117 (Holm p = 0.0024),
scoped +0.086 (Holm p = 0.168, directional); Recall@5 unchanged; production
path adds +0.004 MRR over the bare scoped retriever; end-to-end 32/45 hybrid vs
35/45 dense in one run; 7/45 failures with the evidence present; prompt
changes fixed answer shape, truncation, section identity and current-plan
medications, not GERD reasoning or the unnamed historical medication; advice
probes declined 12/12, document questions answered 11/12.

**D. Known limitations.** Section 19 and Section 20: small single-author
benchmark, one generator, one embedding model, single end-to-end run, page-level
relevance, intervention study on one document, two-column extraction and chunk
overlap unresolved.

### What is novel / what is not novel

Not novel, and not claimed: hybrid dense + BM25 retrieval, Reciprocal Rank
Fusion and its weighted form, FAISS or SQLite FTS5 use, local LLM serving via
Ollama, prompt-level answer-style instructions, the observation that RAG
systems produce unfaithful or incomplete answers, the Wilcoxon/Holm analysis
itself.

Possibly useful, stated conservatively: the combination of (a) a
counterbalanced, multiplicity-corrected retrieval comparison in the deployed
document-scoped setting, (b) a stage-separated audit showing that the residual
errors of a fully local 3B system occur after successful retrieval, with each
failure checked against the retrieved text, and (c) an intervention study that
reports which failure classes respond to narrow prompt changes and which do
not, together with the measured side effects that ruled out broader changes.
This is an evaluation-methodology and case-study contribution on a small
benchmark; whether its findings hold more widely is untested.

## 24. Related Literature (identified, not yet cited)

The repository's documents did not previously cite any literature, and this
report cites none as support for its numbers. The works below were identified
during the project as relevant to positioning; each should be read and
verified before it is cited in a paper.

- RAG evaluation frameworks: RAGAS (arXiv 2309.15217); ARES (arXiv
  2311.09476); ALCE, citation evaluation (arXiv 2305.14627); RAGEv (arXiv
  2505.04680).
- Sufficient context and abstention: "Sufficient Context" (ICLR 2025); "Do
  RALMs Know When They Don't Know" (arXiv 2509.01476); over-refusal steering
  (arXiv 2510.10452).
- Multi-component completeness: sub-question coverage (arXiv 2410.15531,
  NAACL 2025).
- Faithfulness / hallucination detection: MiniCheck (arXiv 2404.10774);
  LettuceDetect (arXiv 2502.17125); RAGTruth (arXiv 2401.00396); SURE-RAG
  (arXiv 2605.03534).
- Medical RAG and medical evaluation: MIRAGE/MedRAG (arXiv 2402.13178); MedRGB
  (arXiv 2411.09213); MedHallu (arXiv 2502.14302); MedScore (arXiv
  2505.18452); SourceCheckup (Nature Communications 2025, PMC12003634);
  VeriCite (arXiv 2510.11394); expert evaluation of medical RAG (arXiv
  2511.06738); scoping review (arXiv 2511.05901); systematic review in PLOS
  Digital Health (PMC12157099); a local medical RAG deployment (medRxiv
  10.1101/2025.10.20.25337146).
- Numeric and table reasoning: QuanTemp (arXiv 2403.17169); EQUATE;
  ClinicNumRobBench (ACL 2026 Findings).
