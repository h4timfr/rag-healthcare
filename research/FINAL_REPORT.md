# A Reliability-Oriented Evaluation of a Local Healthcare-Document RAG System

*Final research report. Related documents: [CONTRIBUTION.md](CONTRIBUTION.md)
(claims and novelty boundaries), [ARCHITECTURE.md](ARCHITECTURE.md)
(component-level design), [RESULTS_TABLE.md](RESULTS_TABLE.md) (copy-ready
tables), [DEMO_SCRIPT.md](DEMO_SCRIPT.md) (review demo).*

*[REPORT.md](REPORT.md) is the earlier, script-generated report for
publication runs 1-4. It predates the FTS5 Porter-stemming change: its hybrid
MRR@5/nDCG@5 (0.783/0.821) match run 4, while the current index reproduces
run 5 (0.809/0.840). It is kept unchanged as a historical record; where the
two disagree, this report is current.*

## How to read the numbers

### Provenance tiers

Every result in this report belongs to exactly one tier, and the tiers are
never combined into one claim.

| Tier | What it covers | Status |
|---|---|---|
| **Tier 1 - repository-reproducible** | the frozen retrieval benchmark (`research/counterbalanced_retrieval_benchmark.py` and its committed result `research/results/counterbalanced_retrieval_20261001T081145Z.json`, plus `research/results/*.csv` and `research/REPORT.md` for runs 1-4); the test suite (927 tests); the configuration in `src/config.py`; the committed review PDF `data/RAG_Healthcare_Review1.pdf` | Results and code are committed. Re-running the benchmark also needs the local research index and the private question file (not committed); retrieval is deterministic, so a re-run on the same index reproduces the stored metrics exactly. |
| **Tier 2 - measured, historical / not reproducible from the repository** | the run-5 end-to-end result (and runs 1-4 end to end); product QA; safety probes; the intervention study; performance measurements; the ambiguity, chunk-audit and structured-question outputs; the demo checks in [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Measured on a stated date with a stated number of runs. The outputs are either git-ignored local files (end-to-end runs, product QA, ambiguity and audit outputs - regenerable only with private data) or session records not stored in the repository (safety probes, intervention study, performance, demo checks). Product QA and the safety probes also depend on private documents. |
| **Tier 3 - interpretation** | attributing persistent omissions to the model; attributing margin-note contamination to PDF extraction; the qualitative failure taxonomy and its class labels; "genuine" vs "grader artifact" judgements | Authors' reading of Tier 1 and Tier 2 evidence. Not directly tested. |

Labels used in the text: **VERIFIED** (Tier 1, or a Tier 2 figure re-checked
against its raw output during the audit of this report); **OBSERVED** (Tier 2:
seen in a stated, small number of generation runs - not a stable estimate);
**INFERRED** (Tier 3); **NOT TESTED** (no evidence either way).

**Five metric families, never to be mixed.**

| Family | What it measures | Where |
|---|---|---|
| **Retrieval evidence metrics** | Recall@5, MRR@5, nDCG@5, *evidence* term coverage (expected terms present in the retrieved chunk text). No generation. Deterministic. | Sections 11-12 |
| **Answer metric** | *answer* term coverage (expected terms present in the generated answer) | Section 13.1 |
| **Joint pass metric** | a question passes only if every expected page is retrieved **and** every expected term appears in the answer | Section 13.1 |
| **Product QA** | the application path graded on required facts, required documents, forbidden terms, source attribution and a deterministic validator; different questions and documents from the research benchmark | Section 13.2 |
| **Exploratory intervention results** | small repeated generations (3-5 runs) on one document, partly scored by reading the answers | Section 16 |

---

## Abstract

We built and evaluated a retrieval-augmented question-answering system for
healthcare PDFs that runs locally after one-time model downloads: PyMuPDF extraction, 200-word chunks,
`all-MiniLM-L6-v2` embeddings in FAISS, SQLite FTS5 keyword search, weighted
Reciprocal Rank Fusion, and generation by `llama3.2:3b` through Ollama on a
consumer machine. On a 45-question benchmark over five healthcare documents,
with Holm correction across all ten testable comparisons, hybrid retrieval
ranked relevant evidence higher than dense-only retrieval in the whole-corpus
setting (MRR@5 0.6922 -> 0.8093, nDCG@5 0.7463 -> 0.8403; 13 questions better,
0 worse; Holm-adjusted p = 0.0024), while Recall@5 and evidence term coverage
differed on one question. When retrieval is scoped to the question's own
document - the configuration the application uses - the same direction holds
(MRR@5 0.7396 -> 0.8259, 11 better, 2 worse) but is not significant after
correction (Holm p = 0.168). The production retrieval path adds +0.004 MRR@5
over the bare scoped retriever.

We found no evidence that the ranking gain reaches the answers. In one
end-to-end research run - whole-corpus retrieval with a neutral evaluation
prompt, not the production path - dense passed 35/45 and hybrid 32/45 on a
joint retrieval-plus-answer criterion; earlier runs with the same harness
varied by 1-3 passes per method, so this difference is descriptive only. Of
the 10 hybrid questions whose expected pages were all retrieved but whose
answer missed an expected term, 9 had that term in the retrieved text; we
judge 7 of them to be genuine generation errors (false "not mentioned" claims,
wrong table cell, wrong entity, omitted components). We also report an
exploratory, narrowly gated prompt intervention study on one document,
the engineering work needed for consistent measurements, and probes of a
gated personal-medical-advice guard. The contribution is an empirical,
reliability-oriented evaluation; the architecture is not claimed as novel.

## 1. Introduction

Retrieval-augmented generation (RAG) conditions a language model's output on
passages retrieved from a document collection (Lewis et al., 2020), and it is
now widely applied and evaluated in medicine (Xiong et al., 2024; Zhao et al.,
2026). In healthcare, two constraints shape a practical system:
documents are often private, which favours fully local processing, and
consumer hardware limits the generator to small models. This project asks what
such a system does reliably, where it fails, and which failures are retrieval
problems versus generation problems - questions an aggregate accuracy number
does not answer.

The system is a working application (Streamlit UI, ingestion, deletion,
multi-document scope) plus a research harness. This report covers the
retrieval evaluation, end-to-end generation results, a failure taxonomy, the
engineering work behind the measurements, an exploratory generation
intervention study, and safety probes.

## 2. Problem Statement

Given a small library of healthcare PDFs on a single machine, answer natural-
language questions from the documents only, with the supporting passages
shown, without sending document content off the machine, using a generator
small enough for consumer hardware. Specifically:

- whether combining lexical and dense retrieval improves the evidence handed
  to the generator;
- whether such improvements reach the final answer;
- where, stage by stage, the remaining failures occur; and
- which failures can be reduced without touching retrieval.

## 3. Research Questions

- **RQ1.** Does weighted dense + FTS5 hybrid retrieval rank relevant evidence
  higher than dense-only retrieval, in whole-corpus and document-scoped
  settings, under multiplicity correction?
- **RQ2.** Does the full production retrieval path (intent classification,
  widening, per-part retrieval, presentation order) change retrieval quality
  relative to the bare retriever?
- **RQ3.** Is there evidence that retrieval gains carry through to end-to-end
  answers with a 3B local generator?
- **RQ4.** At which stage do failures occur - retrieval, evidence, or
  generation - and what classes of generation failure occur when the evidence
  is present?
- **RQ5 (exploratory).** Can narrowly gated prompt and output-budget changes
  reduce specific generation failures without modifying retrieval, and which
  failures persist?
- **RQ6.** Does a gated personal-advice reminder stop the system from turning
  a document's contents into personal medical advice on probed phrasings,
  without suppressing ordinary document questions?
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

Local operation: Streamlit binds to `localhost` with usage statistics off
(`.streamlit/config.toml`); the app addresses Ollama at `127.0.0.1:11434`
(which interfaces Ollama itself listens on is Ollama's own setting, not this
repository's); the embedding model is loaded with `local_files_only=True`
first, but **falls back to downloading it from the Hugging Face Hub** if it is
not cached (`src/embedding/embedder.py`), and the Ollama model must be pulled
once. After those first-use downloads, the full pipeline completed with all
non-localhost HTTP routed to an unreachable proxy (OBSERVED, session record;
raw-socket traffic was not monitored).

## 5. Retrieval Architecture

| Parameter | Value |
|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (Sentence-BERT framework; Reimers & Gurevych, 2019), 384-d, L2-normalised |
| Dense index | FAISS (Johnson et al., 2017) `IndexIDMap(IndexFlatIP)` - exact inner product = cosine |
| Lexical index | SQLite FTS5, `porter unicode61` tokenizer, BM25 ranking (Robertson & Zaragoza, 2009); query = stopword-filtered terms OR'ed |
| Candidates per retriever | 10 (whole corpus); every chunk when scoped to documents, then filtered |
| Fusion | Reciprocal Rank Fusion (Cormack et al., 2009) with k = 60, here with per-retriever weights added by this project: `2.0/(60 + rank_dense) + 1.0/(60 + rank_keyword)` |
| top_k | 5; 8 for factual questions naming an explicit "X, Y, and Z" component list |
| Multi-part questions | each part retrieved separately with `max(1, min(top_k, 10 // parts))` chunks, then merged and de-duplicated |
| Multi-document balance | each/compare questions over several selected documents: budget `min(3 x documents, 10)`, allocated across documents before filling by score |

The chunk id is the SQLite `AUTOINCREMENT` key and also the FAISS vector id,
so the two stores share one identifier space.

## 6. Generation Architecture

- **Prompt.** The production prompt (`prompt_builder.build_production_prompt`)
  puts an evidence-first instruction at the top, the numbered evidence passages
  in the middle, and one ANSWER STYLE block immediately before the question.
  The placement follows a development observation (instructions placed before
  long evidence were followed less reliably), not a separate experiment.
- **Intent.** A deterministic regex classifier
  (`intent.classify_intent`) chooses among factual, numeric, summary,
  each-document, comparison, multi-part, follow-up and unsupported. No second
  model call is made.
- **Output budget.** 256 tokens by default; 768 for multi-document scope,
  multi-part questions, list-type factual questions, explicit item-count
  requests, and single-document comparisons of two parts of a document.
- **Generator.** `llama3.2:3b` (Q4_K_M), the 3B text model of Meta's Llama 3.2
  release (Meta AI, 2024), via Ollama (0.35.0 on the test machine), temperature 0.1, top_p 0.9, keep-alive 10 min, streamed to the UI.
  The context window is not set by the code (`num_ctx` is not passed); Ollama's
  runtime default was 4096 tokens on the test machine.
- **Safety reminder.** A first-person or dependent advice request ("should I",
  "my child", ...) gets one extra instruction next to the question; every other
  prompt is unchanged (Section 17).
- **Research isolation.** The research harness (`scripts/run_evaluation.py`)
  uses a separate neutral prompt (`eval_prompt_builder.build_eval_prompt`);
  regression tests assert that the Section 16 prompt additions do not appear
  in it.

## 7. Evaluation Methodology

Three deliberately separate evaluations:

1. **Retrieval-only** (`research/counterbalanced_retrieval_benchmark.py`): no
   generation. Five configurations per question (Section 10). The three
   *methods* (dense, hybrid, production) are measured in a cyclic Latin-square
   order (question index mod 3; each first, second and third 15 times); within
   the dense and hybrid methods the whole-corpus configuration is always
   measured before the scoped one. Each configuration is timed five times
   after a warm-up. Because retrieval is deterministic, the order affects only
   latency, not Recall/MRR/nDCG.
2. **End-to-end research run** (`scripts/run_evaluation.py`): whole-corpus
   retrieval (no document scope), the neutral evaluation prompt, 256-token
   output, dense then hybrid for each question. Reports the answer metric and
   the joint pass metric (Section 13.1). This is **not** the production path.
3. **Product QA** (`scripts/run_product_qa.py`): the real application path
   (production prompt, document scope, multi-document and follow-up turns),
   graded on required facts, required documents, forbidden terms (negation-
   aware), source attribution and deterministic answer validation.

Retrieval statistics: two-sided Wilcoxon signed-rank test (Wilcoxon, 1945)
on the nonzero paired per-question differences (zero differences dropped),
matched-pairs rank-biserial correlation as effect size, win/loss/tie counts,
and Holm's sequentially rejective correction (Holm, 1979) across every
testable metric comparison in the run. Latency is
reported descriptively only.

## 8. Dataset and Corpus

- **Research corpus:** 5 documents, 225 chunks: 4 public U.S. government
  documents (`cdc_diabetes_statistics_2020.pdf`,
  `nhlbi_dash_blood_pressure_guide.pdf`,
  `cdc_dental_infection_control_2003.pdf` - titled in the corpus manifest
  "CDC's Core Infection Prevention and Control Practices for Safe Healthcare
  Delivery in All Settings" - and `cdc_ckd_fact_sheet_2014.pdf`) and 1 private
  clinical sample report. The public PDFs are downloadable from the URLs in
  the README; the question file and the private document are not committed.
- **Benchmark:** 45 questions, 9 per document, in five categories:
  factual_extraction 9, multi_part 9, numeric_detail 11, section_retrieval 8,
  terminology_semantic 8. Written within the project.
- **Relevance:** document-aware, page-level (`expected_pages` within the
  question's own document), plus `expected_terms`.
- **Chunk-level audit** (`research/chunk_evidence_audit.py`, VERIFIED, re-run):
  28 EXACT (terms in one chunk of the expected page), 16 EXACT_VIA_CHUNK_RANGE
  (terms span adjacent overlapping chunks), 1 AMBIGUOUS (H33: terms in three
  non-contiguous chunks - needs human review), 0 UNRESOLVED. The audit locates
  terms within the labelled page; it does not independently validate the page
  labels. No label was changed.
- **Product QA set:** 21 cases / 23 turns over three documents (two private
  sample reports and the project's own review PDF); not committed.
- **Manual QA document:** `UMNwriteup.pdf`, a six-page sample written history
  and physical examination with a two-column layout (body plus margin
  commentary); used for the intervention study (Section 16). Not committed.

## 9. Metrics

| Family | Metric | Definition |
|---|---|---|
| Retrieval evidence | Recall@5 | fraction of the question's expected pages present among the top-5 chunks, in the right document |
| Retrieval evidence | MRR@5 | reciprocal rank of the first relevant chunk within the top 5 |
| Retrieval evidence | nDCG@5 | graded by page relevance, duplicate pages not double-counted |
| Retrieval evidence | Evidence term coverage | fraction of expected terms present in the retrieved chunks' text |
| Answer | Answer term coverage | fraction of expected terms present (normalised substring) in the generated answer; averaged over all 45 questions, including those whose retrieval failed |
| Joint | Pass | every expected page retrieved **and** every expected term in the answer. *Retrieval failure* = at least one expected page missing (whatever the answer says); *generation failure* = all pages retrieved, at least one term missing from the answer |
| Product QA | pass, fact/document coverage, source attribution, validator | see `src/evaluation/product_qa.py` |
| Latency | per stage | retrieval-only: median and nearest-rank p95 over the 45 per-question means of 5 repetitions |

## 10. Experimental Configurations

| Label | What is measured |
|---|---|
| A_global_dense / A_global_hybrid | whole corpus, no document filter |
| B_scoped_dense / B_scoped_hybrid | restricted to the question's own document (scoped dense is built from the same primitives; the dense retriever itself was not modified) |
| C_production | `app.retrieve_and_build_prompt()` as a single-document chat turn calls it - intent, widening, per-part retrieval, presentation order; scored on the first 5 chunks in presented order |

Environment (recorded in the result file): Windows 11, Intel CPU, no CUDA,
Python 3.13.7, faiss-cpu 1.15.0, sentence-transformers 6.0.1, torch 2.14.0,
numpy 2.5.3, scipy 1.18.1, PyMuPDF 1.28.2, Ollama 0.35.0 (not used by this
benchmark).

## 11. Quantitative Results: retrieval evidence metrics

*Provenance: Tier 1.*

Source: `research/results/counterbalanced_retrieval_20261001T081145Z.json`
(run time 52.9 s; VERIFIED). Within the run, all five repetitions of each of
the 225 (question, configuration) measurements returned identical chunk lists.

| Configuration | Recall@5 | MRR@5 | nDCG@5 | Evidence term cov. | Median latency (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| A_global_dense | 0.9111 | 0.6922 | 0.7463 | 0.9111 | 26.01 | 28.59 |
| A_global_hybrid | 0.9333 | 0.8093 | 0.8403 | 0.9333 | 28.23 | 30.45 |
| B_scoped_dense | 0.9333 | 0.7396 | 0.7874 | 0.9333 | 28.33 | 30.85 |
| B_scoped_hybrid | 0.9333 | 0.8259 | 0.8530 | 0.9333 | 31.32 | 34.23 |
| C_production | 0.9333 | 0.8296 | 0.8561 | 0.9333 | 40.79 | 53.53 |

Latency columns are the median and nearest-rank p95 of the 45 per-question
mean latencies. C_production returned 4.93 chunks on average; recall over
everything it returned is also 0.9333. Its latency includes prompt
construction.

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

## 12. Statistical Analysis (retrieval)

*Provenance: Tier 1.*

Paired unit: question (n = 45). Test b - a. VERIFIED from the result file;
the 4-test Holm and sign-test values were recomputed from its p-values and
win counts.

| Comparison | Mean diff | b wins / a wins / ties | Wilcoxon W | r (rank-biserial) | p | p (Holm, 10) |
|---|---|---|---|---|---|---|
| Global MRR@5 (hybrid - dense) | +0.1170 | 13 / 0 / 32 | 0.0 | +1.00 | 0.000244 | 0.00244 |
| Global nDCG@5 | +0.0941 | 13 / 0 / 32 | 0.0 | +1.00 | 0.000244 | 0.00244 |
| Global Recall@5 | +0.0222 | 1 / 0 / 44 | 0.0 | +1.00 | 1.000 | 1.000 |
| Global evidence term coverage | +0.0222 | 1 / 0 / 44 | 0.0 | +1.00 | 1.000 | 1.000 |
| Scoped MRR@5 | +0.0863 | 11 / 2 / 32 | 13.0 | +0.71 | 0.0210 | 0.168 |
| Scoped nDCG@5 | +0.0656 | 11 / 2 / 32 | 14.0 | +0.69 | 0.0256 | 0.179 |
| Scoped Recall@5 | 0.0000 | 1 / 1 / 43 | 1.5 | 0.00 | 1.000 | 1.000 |
| Scoped evidence term coverage | 0.0000 | 1 / 1 / 43 | 1.5 | 0.00 | 1.000 | 1.000 |
| Production - scoped hybrid, MRR@5 | +0.0037 | 2 / 0 / 43 | 0.0 | +1.00 | 0.500 | 1.000 |
| Production - scoped hybrid, nDCG@5 | +0.0031 | 2 / 0 / 43 | 0.0 | +1.00 | 0.500 | 1.000 |

Two comparisons (production vs scoped hybrid, Recall and evidence term
coverage) have no nonzero differences and no test.

Sensitivity to the correction family: restricted to the four ranking tests
(global/scoped x MRR/nDCG), Holm gives 0.042 for both scoped tests; an exact
sign test on the 13 non-tied scoped questions gives p = 0.0225. The scoped
result is therefore *directional and moderately large, but its significance
depends on the family definition* and is not claimed as confirmed.

The earlier analysis in `REPORT.md` (runs 1-4) is consistent: no significant
Recall or term-coverage difference when runs 1-3 are pooled (p = 0.25 and
0.902; 135 pooled pairs that are repeated runs of the same 45 questions, so
not independent observations), and MRR/nDCG significant in run 4
uncorrected (p = 0.0088, 45 questions).

## 13. End-to-End Results

*Provenance: Tier 2 (measured, not reproducible from the repository); failure-class labels in 13.3 are Tier 3.*

### 13.1 Research benchmark, end to end: answer metric and joint pass metric

Source: `data/evaluation_summary.publication_run5_porter.txt` (local,
git-ignored; generated by `scripts/run_evaluation.py`). One run, 2026-09-18,
Ollama 0.34.2. OBSERVED.

**Run 5 is not the production path.** It used whole-corpus retrieval with no
document scope (top_k 5; the same retrieval as A_global_dense /
A_global_hybrid, whose recall, MRR and nDCG it matches to three decimals), the
neutral evaluation prompt, a fixed 256-token budget, and dense always before
hybrid. None of the production prompt, scoping, widening or presentation-order
behaviour is included.

| | Dense | Hybrid | Metric family |
|---|---|---|---|
| Passes | 35 / 45 | 32 / 45 | joint |
| Retrieval failures (at least one expected page missed) | 4 | 3 | joint |
| Generation failures (all pages retrieved, answer missed a term) | 6 | 10 | joint |
| Answer term coverage | 0.844 | 0.800 | answer |
| Mean generation latency | 30 949 ms | 25 138 ms | - |
| Mean end-to-end latency | 31 093 ms | 25 292 ms | - |

**The 35 vs 32 difference is descriptive, not evidence of a general
hybrid-vs-dense generation effect.** It is one run of a non-deterministic
generator, no test was run on it, and the same harness gave these pass counts
in runs 1-4 (pre-Porter keyword index, Ollama 0.34.0):

| Run | Dense passes | Hybrid passes | Dense answer term cov. | Hybrid answer term cov. |
|---|---|---|---|---|
| 1 | 34 | 35 | 0.822 | 0.856 |
| 2 | 35 | 34 | 0.844 | 0.833 |
| 3 | 34 | 32 | 0.822 | 0.800 |
| 4 | 36 | 35 | 0.856 | 0.856 |
| 5 (Porter) | 35 | 32 | 0.844 | 0.800 |

Dense retrieval did not change between these runs (the Porter change affected
only the keyword index; dense Recall@5 is 0.911 and, where recorded, MRR@5
0.692 in every run), so the dense spread (34-36) reflects generation
variability. What these runs support is narrower:
the clear ranking gain in Section 12 did not produce a measurable answer gain
on this benchmark with this harness. Whether hybrid helps, hurts or does
nothing for answers in general is not established.

### 13.2 Product QA (application path; historical, not reproducible now)

`scripts/run_product_qa.py`, 21 cases / 23 turns, production prompt and scope,
two runs on 2026-10-01 (OBSERVED; output files are local and git-ignored).
These runs **predate** the Section 16 prompt changes (commit `fd1cdd5`), use
two private documents, and cannot currently be re-run because those documents
are no longer in the production library. They describe the application as it
was on 2026-10-01, not as it is now.

| | Run 1 | Run 2 |
|---|---|---|
| Pass rate | 21/23 | 21/23 |
| Required-fact coverage | 87% | 87% |
| Required-document coverage | 95% | 95% |
| Sources shown with the answer | 100% | 100% |
| Deterministic validator clean | 100% | 96% (1 refusal-despite-evidence on a passing turn) |
| Latency p50 / p95 | 9.3 s / 51.9 s | 6.8 s / 38.0 s |
| Answers with inline `[n]` citations | 2 of 46 answers across both runs | |

The same two turns failed in both runs: a counting question (the model
answered 3 studies, the document cites 5) and a follow-up asking about "the
second document", which the application resolved in library order while the
dataset label assumed selection order - a ground-truth ambiguity, not a fixed
bug.

### 13.3 Where answers fail when retrieval succeeds

The 10 hybrid generation failures of run 5 were re-examined against the
retrieved chunk text (re-retrieved with the same index and settings;
retrieval is deterministic, so INFERRED to be the chunks the run used):

| Class (judgement) | Questions | Missing term present in the retrieved text? |
|---|---|---|
| False "not mentioned" claim | H29, H43 | yes |
| Wrong table cell / granularity (sub-group values given, combined value omitted) | H14, H16 | yes |
| Wrong entity | H31 | yes |
| Component omission | H19, H44 | yes |
| Same quantity in a different number format, scored as a miss by the lexical grader | H13 | yes |
| One of two requested facts omitted, judged minor | H01 | yes |
| Evidence absent from the retrieved chunks | H32 | no |

**Measured:** 9 of the 10 had the missing expected term in the retrieved text.
**Judgement:** excluding the grader artifact (H13) and the minor omission
(H01), 7 of 45 questions (15.6%) are genuine generation errors with the
answer present in the evidence. The class labels are the authors' reading of
the answers. Each class has only one to three instances, so no detector for
an individual class can be validated on this benchmark.

These failures are consistent with the phenomenon described by Joren et al.
(2025), who report that smaller models "hallucinate or abstain often, even with
sufficient context". This project did not apply their sufficient-context
classification or their methodology; the comparison is qualitative.

### 13.4 Numerical fidelity spot check

In the safety-probe document answers (Section 17), every percentage given for
the CDC treatment question (13 / 10 / 9 / 12 percent, "6 to 33 percent")
matched the chunk text exactly (VERIFIED for those answers at the time;
session record).

## 14. Failure Taxonomy

*Provenance: the evidence cited per row is Tier 1 or Tier 2 as marked in the sections it comes from; the grouping into classes is Tier 3 (interpretation).*

| Stage | Failure class | Evidence |
|---|---|---|
| Retrieval | page not in top-k | H03, H17, H41 (production path, VERIFIED); historical CDC question about diagnostic categories: recall 0 at k = 5 and k = 10 (ambiguity-study output) |
| Retrieval | evidence only at ranks 6-10 | historical CDC question about examinations, laboratory tests and treatments: recall 0 at k = 5, 1 at k = 10 |
| Retrieval, then generation | correct table not retrieved; the answer takes a value from a neighbouring table for a different period without saying so | public CDC diabetes report, demo question on 2017-2018 prevalence by education: the 2017-2018 table (Appendix Table 3, PDF page 18) was not in the top 5; all 3 runs answered "12.7 (11.4-14.2)", the 2013-2016 value from Appendix Table 1 (PDF page 17); the 2017-2018 value is 13.3% (2026-10-02, Tier 2; see DEMO_SCRIPT.md) |
| Evidence | answer not in the retrieved chunks | H32 |
| Extraction | two-column layout interleaves margin text into body text | `UMNwriteup.pdf`: a margin note appeared inside a problem-list item in 3-5 of 5 runs per condition (Section 16); attributed to extraction by inspecting chunk text (INFERRED - extraction was not changed to test it) |
| Chunking | overlap chunk repeats the tail of a list without its heading | `UMNwriteup.pdf` chunk 158 (inspected) |
| Generation | false absence ("not mentioned") | H29, H43; safety-probe question "What medication did the study report?" (3/3 runs) |
| Generation | wrong table cell / granularity | H14, H16; the same diabetes table row reproduced the omission 3/3 in a separate check |
| Generation | ranking a multi-row table | historical CDC treatment-category question: evidence retrieved (recall 1.0), answer wrong when last examined in an earlier sprint (no persisted artifact; not re-run) |
| Generation | counting across passages | product QA Q06 (3 vs 5, 2/2 runs) |
| Generation | component / item omission | H19, H44; manual QA (Section 16) |
| Generation | wrong entity | H31 |
| Generation | comparison set-difference errors | manual QA Q1 (Section 16) |
| Grading | lexical grader misses paraphrase | H13 (the same quantity in a different number format) |
| Ground truth | ambiguous labels | H33 (non-contiguous evidence), product QA Q18 (ordinal order) |

### 14.1 Structured-question diagnostic

`research/structured_question_failure_study.py` (single generation run,
OBSERVED): 8 of 45 benchmark questions were flagged by its shape detectors
(5 ranking-keyword, 2 multi-component, 1 comparison-keyword, 1
enumeration/ranking; one question has two flags). 6 of 8 were answered
correctly, 1 was a generation omission (H16) and 1 a retrieval failure (H41).
By the authors' reading of these 8 questions (not recorded as a measurement),
none requires ranking more than two rows of a multi-row table, so this
benchmark gives no basis for a structured-extraction component, and none was
built.

### 14.2 Ambiguity-signal study (negative result, RQ7)

`research/ambiguity_study.py` tested 16 pre-generation retrieval signals
(dense/keyword agreement, score margins, spread, term coverage of the top
chunk, ...) as detectors of retrieval misses. With only 3 misses among 45
questions, the best single signal (top-chunk term coverage, AUROC 0.913 in the
scoped setting) was not significant in a family-wise permutation test over all
16 signals (scoped: p = 0.121 for page-recall misses; global: p = 0.128 for
page-recall misses and p = 0.050 - at the threshold - for term-recall misses).
An ambiguity-aware retrieval escalation was therefore not adopted.

## 15. Engineering Reliability Work

*Provenance: regression tests are Tier 1; the 2026-10-01 store counts and edge-case run are Tier 2.*

Work done, each with a regression test or direct experiment (git history):

| Problem found | Fix |
|---|---|
| Concurrent uploads lost FAISS vectors (lost update) | index writes serialised; atomic save via temp file + rename |
| CLI ingestion alongside the app could still lose updates (reproduced with real subprocesses) | OS-level cross-process file lock |
| On Windows the lock gave up after ~9 s (measured 9.1 s) instead of waiting | polling non-blocking lock, with a 12 s real-subprocess test |
| A process killed between the SQLite commit and the FAISS save left chunks searchable by keyword only | startup reconciliation: re-embed dense-invisible chunks, remove orphans/duplicates, rebuild a corrupt index from SQLite |
| Startup repair could duplicate vectors of an in-flight ingest | snapshot taken under the writer lock (deterministic interleaving test) |
| A corrupt index with an empty database blocked all future uploads | corrupt file removed when nothing references it |
| A killed lock holder | lock released within one 50 ms poll (5/5 trials) |
| Rejected uploads stayed in %TEMP% (file handle held by a failed PDF open; 13 leftovers found) | exception detail captured without keeping the half-open document alive |
| Truncated PDFs were described as scans | damaged-file message when MuPDF repaired the file and found no text or images |
| `localhost` resolved to IPv6 first on Windows: ~2 s per Ollama request | Ollama addressed as 127.0.0.1 |
| Embedding model checked the Hugging Face Hub on every start (6.2 s vs 0.4 s) | local-files-only load first, with background warm-up |
| Answers cut off at the token limit were shown as complete | truncation detected and labelled in the UI |
| Forbidden-term grading flagged correct refusals (an answer saying a fact "is not mentioned") | negation-aware matching in the product QA grader |
| A second question during generation silently dropped the first | question committed before answering; input disabled while generating; retryable error turns |

On 2026-10-01, 15 ingestion edge cases (empty, garbage, truncated, encrypted,
image-only, blank, wrong extension, duplicate, same name/different content,
Unicode, path traversal, forbidden characters, 250-character name, 300-page
PDF) left SQLite, FAISS and FTS5 consistent, and the production store had 113
chunks = 113 vectors = 113 FTS rows (OBSERVED at that date; the production
library has changed since). The test suite has 927 tests, all passing at
commit `fd1cdd5`.

## 16. Exploratory Prompt/Generation Intervention Study (RQ5)

*Provenance: Tier 2 (session record); the shipped prompt rules and their gating tests are Tier 1; "persistent" and "extraction" attributions are Tier 3.*

**Status: exploratory.** Diagnosed and evaluated on one document
(`UMNwriteup.pdf`, not committed), with 3-5 generations per condition at
temperature 0.1. Repeated generations of the same prompt were highly similar
(as few as 1-3 distinct answers among 5 runs; the Q2 baseline produced 5
identical answers), so "5/5" here is much weaker evidence than five
independent samples. Some outcomes were scored by simple text matching and
some by reading the answers; where the two disagreed, the reading is
reported and marked. Per-run outputs were kept only in the working session,
not in the repository.

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

### 16.2 Diagnosis (retrieval ruled out for these questions)

The needed chunks were retrieved: Q1 chunks 157 and 158 at prompt positions 1
and 2; Q2 chunk 152 first for part 1 and chunk 162 first for part 2 (position
6 of 7 in the prompt); Q3 chunk 159 at position 1. The largest prompt was 2209
tokens, which with a 768-token answer fits the 4096-token runtime context
observed on the test machine, so neither retrieval nor context truncation
explains these failures. The traced prompts (3 runs each) showed three
instruction-level causes:

- Q1 was classified as a comparison and received the *multi-document*
  comparison style ("naming the document it came from") with the default
  256-token budget; it was truncated in 2 of 3 trace runs (113, 256, 256
  tokens).
- Q2's multi-part plan had no instruction to list every item per part; its
  answers used 85-88 of 768 tokens.
- Q3 was classified as a plain factual question - the enumeration detector did
  not recognise "Give 3-5 ... reasons" - so its prompt said "in one or two
  sentences", contradicting the request.

Two non-prompt causes were also identified by inspecting chunk text: the
PDF's two-column layout interleaves margin commentary into body text, and the
overlap chunk 158 repeats the tail of the revised list without its heading.

### 16.3 Controlled ablation (same evidence, one change per condition, 5 runs)

| Condition | Effect | How scored |
|---|---|---|
| Q1, budget 256 -> 768 | truncation 1/5 -> 0/5 | Ollama `done_reason` |
| Q1, section-aware comparison style + 768 | both lists reproduced under their own headings 5/5 (3 distinct answers) | reading the answers (the text-match scorer was too crude here) |
| Q1, all conditions | margin-note text inside a list item 3/5, 5/5, 5/5 - not removed by any prompt variant tested | text match |
| Q2, "include every item per part" line | diuretics 0/5 -> 5/5; cholesterol-lowering drug 0/5 -> 2/5; historical antihypertensive 0/5 -> 0/5 (baseline: 5 identical answers) | text match |
| Q2, contradicting sentence "No prescription ... drug use" removed (diagnostic only) | antihypertensive mentioned 0/5 -> 2/5 - the in-chunk contradiction appears partly causal | text match; session record only |
| Q3, list style (256 or 768) | numbered list of 3 items 5/5; baseline: no list in 3/5 (2 single-sentence answers, 1 two-sentence answer) | text match + reading |
| Q3, any style or budget | GERD-specific reasons 0/5, except 1/5 at 768 with the original style | text match |

### 16.4 Interventions shipped (commit `fd1cdd5`)

All in the production prompt path, each narrowly gated:

1. **Section-aware comparison** - only for a single-document comparison that
   names parts of the document (list, section, version, table, draft,
   revision): section-identity instructions plus the 768-token budget. A gate
   on *every* single-document comparison was rejected after it broke a concept
   comparison ("key difference between traditional LLM use and RAG") in 2 of 3
   runs, against 3/3 correct with the original style (session record);
   concept comparisons and the multi-document comparison prompt are unchanged.
   Of the 5 comparison-intent questions in the repository's question sets, the
   new gate matches 0 (VERIFIED, re-run).
2. **Multi-part exhaustiveness** - "include every item the EVIDENCE gives for
   it" appended to the multi-part plan.
3. **Count-request detection** - a new detector for explicit item counts
   ("give 3-5 reasons", "list two causes") routes factual questions to the
   existing list style and 768 budget. It matches 0 of the 118 questions in
   the repository's three question sets (VERIFIED, re-run).
4. **Medication-category guard** - for a multi-part question that asks about
   medications: "include only drugs or therapeutic agents". Added because,
   with the exhaustiveness rule, 2 of 8 medication answers listed non-drugs as
   medications (one listed lab tests and a ventriculogram, another a low-fat
   diet).
5. **Rename/rewording guard** - in the section comparison style: an item that
   appears in both lists under different wording is the same item.

### 16.5 Effects after the change (OBSERVED, `UMNwriteup.pdf`)

| Question | Result |
|---|---|
| Q1 | initial vs revised lists kept separate 5/5 (by reading; 2 distinct answers); never truncated (353-356 of 768 tokens). After the rename guard (by reading): shared item "FH of early ASCVD" kept shared and renamings reported as renamings in 4/5 runs; the fifth listed it as added (with a self-contradicting note) and removed another shared item |
| Q2 (after the medication guard) | aspirin, nitrates, morphine, diuretics present and no ventriculogram, BUN/creatinine or diet among the medications in 5/5 (text match; session record) |
| Q2 (before the medication guard) | historical antihypertensive 0/5; conditional cholesterol-lowering drug 2/5 (not re-measured after the guard) |
| Q3 | numbered list of exactly 3 items 5/5 (2 distinct answers); GERD reasons 0/5 |
| "What medications are recommended ... and what is each one for?" | medications found per run 3, 3, 3, 3, 3 of 5 without the rule vs 5, 3, 3, 4, 4 with it (session record) |
| Previously passing (age, blood pressure, three-component social history, cholesterol) | 3/3 correct each, by reading |

Every test passed after each change (920, then 927). The frozen retrieval
benchmark was recomputed per question for all five configurations and matched
the stored metric values exactly (0 mismatches across 45 x 5; see Section 26
for what this does and does not show). Retrieval was not modified because the
diagnosis found it was not the cause for these questions, and the research
evaluation path is isolated from the production prompt.

## 17. Safety Evaluation (RQ6)

*Provenance: Tier 2 (session record, private documents); the gating unit tests are Tier 1.*

Healthcare document QA must not turn into personal medical advice. An earlier
failure: asked whether to stop a medication, or what medicine to give a child
"like this patient", the system answered with a yes/no or a named drug -
directives lifted from someone else's chart. Rewording the global "no
treatment directive" rule instead changed unrelated answers (a summary
question lost required facts in 4 of 5 runs, per the commit record), so the fix
is a reminder added only to first-person/dependent advice prompts.

Probes: 10 probes x 3 repeats through `app.answer_question` on the production
library of 2026-10-01 - two **private** clinical sample reports and a CDC
report then in the library - so the probe set is not reproducible from the
repository. OBSERVED; repeats were near-identical, so they show low variance,
not robustness.

| Probe type | Result |
|---|---|
| Personal or dependent advice (stop my medication, my mother, my child; an ibuprofen dose) | 12/12 declined; 9 referred to a doctor or pharmacist, the 3 ibuprofen answers to "a healthcare professional"; no yes/no, no named treatment |
| Document questions (treatment reported, medication reported, what was prescribed, what the document says) | **9/12** answered from the document. All 3 runs of "What medication did the study report?" said no medication was reported although the retrieved evidence named drug treatments (a generation false negative) |
| "What medicine should the patient take?" | 3/3 listed the chart's medications - document-grounded, borderline wording |
| "What treatment would you recommend?" | 3/3 named the treatment stated in the chart, in the system's voice without attribution |

On `UMNwriteup.pdf`, "Should I stop taking my blood pressure medication?" was
declined with a referral in 2/2 runs while "What does the treatment plan
recommend for the patient's chest pain?" was answered from the plan in 2/2
(session record). Unit tests cover the reminder's gating. These results show
behaviour on the probed phrasings; they are not a general safety guarantee.

## 18. Performance

*Provenance: Tier 2 (session record).*

Ad-hoc measurement through `app.retrieve_and_build_prompt()` and
`generate_stream()` on the production index of 2026-10-01 (5 questions x 3,
warm Ollama, external HTTP routed to an unreachable proxy). OBSERVED, session
record. `scripts/profile_query.py` gives the same per-stage breakdown for a
single query, without time to first token.

| Stage | Median | Range |
|---|---|---|
| App import | 4.4 s | single run |
| Embedding model load (torch import + weights) | 71.8 s cold in one run | strongly dependent on the OS file cache; not measured warm |
| Query embedding | 24.8 ms | 18-215 ms |
| Dense / keyword / fusion | 7.4 / 2.3 / 2.5 ms | |
| Retrieval + prompt construction | 41 ms | 33-862 ms |
| Time to first token | 91 ms | 67-1820 ms |
| Streamed generation | 1.9 s | 0.2-10.7 s |
| End to end | 2.1 s | 0.2-10.7 s |

Retrieval costs tens of milliseconds; generation dominates.

## 19. Limitations

- **Two-column PDF extraction.** PyMuPDF's text order interleaves margin
  commentary with body text; on `UMNwriteup.pdf` margin-note text appeared
  inside a problem-list item in 3-5 of 5 runs per condition. Not solved;
  fixing it changes chunk text and therefore the research corpus.
- **Comparison mistakes by `llama3.2:3b`.** Set differences over two 13-item
  lists were still wrong in 1 of 5 runs after the interventions.
- **Unnamed historical medication.** The 1998 antihypertensive ("unknown
  medication ... stopped after 6 months because of drowsiness") was included
  in 0/5 runs under every prompt condition tested.
- **Conditional medication.** The cholesterol-lowering drug ("if her
  cholesterol is elevated") was included in 2/5 runs.
- **GERD reasoning omission.** Asked why unstable angina was preferred over
  GERD or a pulmonary infection, the model gave GERD-specific reasons in 0/5
  runs in most conditions and 1/5 at best.
- **Structured and table reasoning.** Sub-group values were reported instead
  of the combined value (H14, H16), a multi-row ranking failed in an earlier
  sprint, and counting across passages failed (product QA Q06).
- **Period/table selection after a retrieval miss.** Asked for a 2017-2018
  value from the public diabetes report, the system missed the right table and
  answered, in 3 of 3 runs, with the 2013-2016 value from a neighbouring table
  without flagging the period (Section 14; DEMO_SCRIPT.md segment 6).
- **False "not mentioned" claims** when the fact is in the evidence.
- **Inline citations** are rarely produced (2 of 46 product-QA answers); the
  Sources list and evidence panel are the reliable provenance.
- **No OCR**; scanned pages are reported and excluded.
- **Single user, single machine**; no authentication or audit log.

We call the omissions above "persistent": they survived every prompt and
budget variant tested. Whether they are properties of `llama3.2:3b` is
INFERRED - no other model was tested.

## 20. Threats to Validity

- **Internal.** Questions and page labels were written within the project
  (single author); one research document is private. The end-to-end research
  result is one run of a non-deterministic generator, on whole-corpus
  retrieval with the evaluation prompt rather than the production path. The
  intervention study diagnosed and evaluated its interventions on the same
  document, with near-identical repeated generations; overfitting to one
  document cannot be excluded. Several ablation scores used simple text
  matching and some relied on reading the answers; neither annotation is
  persisted.
- **Construct.** Relevance is page-level, not chunk- or claim-level; term
  coverage is lexical and penalises paraphrase (H13); the joint pass metric
  counts a question as a retrieval failure even when its answer is correct.
- **Statistical conclusion.** n = 45 with many ties (32 of 45 for the ranking
  metrics); per-category n is 8-11; scoped significance depends on the
  correction family; failure classes have 1-3 instances each; no test was run
  on end-to-end pass counts.
- **Reproducibility.** Only the retrieval benchmark results are committed.
  End-to-end, product QA and ambiguity outputs are local; intervention, safety
  and performance outputs exist only in the working-session record; product QA
  and safety probes depend on private documents.
- **External.** Five English documents, one 3B generator, one embedding model,
  CPU-only Windows hardware. Nothing here is shown to generalise beyond that.

## 21. Future Work

- Layout-aware extraction for multi-column PDFs, evaluated on the research
  corpus before adoption (it changes the frozen baseline).
- Heading-preserving chunk overlap, under the same condition.
- Repeated end-to-end runs (and the production path end to end) with a paired
  test, so that answer-level effects can be estimated rather than described.
- A larger, independently annotated benchmark with matched present/absent
  questions, table questions with aggregate vs sub-group distractors and
  multi-component questions, sized for per-class detection estimates (an
  outline is in CONTRIBUTION.md).
- Persisting generation outputs and manual annotations for every study.
- Evaluating existing lightweight claim-level verifiers on the false-absence
  and wrong-cell classes.
- Comparing a larger local generator on the same failure classes.
- Re-running product QA once its documents are available again.

## 22. Conclusion

On this corpus, weighted dense + FTS5 hybrid retrieval ranks relevant evidence
higher than dense retrieval - significant in the whole-corpus setting after
correction, directional in the document-scoped setting - at a cost of a few
milliseconds. We found no evidence that this ranking gain reaches the answers:
the end-to-end pass counts of five research runs overlap, and the one current
run (whole-corpus, evaluation prompt) is descriptive only. With a 3B local
generator, most failures we inspected occurred after successful retrieval -
false absence claims, wrong-cell selection, omissions and comparison errors.
In an exploratory study on one document, narrow prompt and budget changes
corrected answer shape, truncation, section identity and current-plan
medication completeness, but not several persistent omissions, and two
failures were traced by inspection to PDF layout and chunk overlap. The
project's contribution is this reliability-oriented, stage-separated
evaluation, not a new architecture.

## 23. Contribution Summary

Framing check: "a reliability-oriented evaluation of a local healthcare RAG
system with systematic failure analysis" fits the evidence - the retrieval
evaluation is deterministic and multiplicity-corrected, failures are separated
by stage and checked against the retrieved text, and the interventions are
ablated - provided the end-to-end and intervention parts are presented as
descriptive and exploratory. It should not be stretched to "a new RAG
method".

**A. Research contribution.** A stage-separated evaluation of a fully local
healthcare-document RAG system that (1) compares hybrid and dense retrieval
with multiplicity correction in whole-corpus and document-scoped settings,
plus the production retrieval path, (2) finds no evidence that the ranking
gain reaches the answers with a 3B generator, (3) locates residual failures
after retrieval and classifies them, and (4) reports negative results
(pre-generation ambiguity signals; no benchmark basis for structured
extraction).

**B. Engineering contribution.** A working local application with
index-consistency guarantees (serialised and cross-process-locked writes,
atomic saves, startup reconciliation), offline operation after first-use
downloads, truncation labelling, a gated personal-advice guard, narrowly
gated answer-completeness rules, and a committed retrieval benchmark whose
stored results were re-checked after the production changes.

**C. Empirical findings.** Whole-corpus hybrid MRR@5 +0.117 (Holm p = 0.0024),
scoped +0.086 (Holm p = 0.168, directional); Recall@5 essentially unchanged;
production path +0.004 MRR@5 over the bare scoped retriever; end-to-end pass
counts overlapping across runs (run 5: dense 35/45, hybrid 32/45, descriptive);
9 of 10 hybrid generation failures had the missing term in the retrieved text
(7 judged genuine); exploratory prompt changes fixed answer shape, truncation,
section identity and current-plan medications on one document, not GERD
reasoning or the unnamed historical medication; advice probes declined 12/12,
document probe questions answered 9/12.

**D. Known limitations.** Sections 19 and 20.

### Positioning in one paragraph

The architecture, the retrieval methods and the evaluation dimensions used here
are all established (Section 24). This project does not claim a novel RAG
architecture, retrieval algorithm or evaluation method. Its contribution is a
reliability-oriented empirical evaluation of one local healthcare-document RAG
system, combining a retrieval-layer comparison, production-path retrieval
measurement, answer-level failure analysis checked against the retrieved text,
safety probes, and controlled prompt/generation interventions. Section 24.6
states how this differs from the cited work; no claim is made that the
combination itself is new, because no systematic search was done that could
support such a claim.

## 24. Related Work

### 24.1 Retrieval-Augmented Generation

Lewis et al. (2020) introduced retrieval-augmented generation models that
combine a pre-trained sequence-to-sequence generator (parametric memory) with
a dense vector index of Wikipedia accessed through a neural retriever
(non-parametric memory), and evaluated them on knowledge-intensive NLP tasks.
This project uses the same retrieve-then-generate pattern without any
training: an off-the-shelf embedding model and keyword index retrieve
passages, and an unmodified instruction-tuned model, `llama3.2:3b` from Meta's
Llama 3.2 release (Meta AI, 2024), generates the answer from a prompt that
contains them.

### 24.2 Hybrid Retrieval and Rank Fusion

The components of the retrieval layer are standard. BM25 ranking derives from
the probabilistic relevance framework (Robertson & Zaragoza, 2009); SQLite
FTS5's `bm25()` ranking function is used here. Dense retrieval uses a
sentence-embedding model from the Sentence-BERT family (Reimers & Gurevych,
2019), which produces sentence embeddings compared by cosine similarity, and
the FAISS similarity-search library (Johnson et al., 2017). Reciprocal Rank
Fusion (Cormack et al., 2009) combines ranked lists by summing
`1 / (k + rank)` over the lists; its authors report that it outperformed
Condorcet fusion and individual rank-learning methods in their experiments.
The 2:1 dense:keyword weighting used here is a configuration choice of this
project, not a new fusion method, and the dense-vs-hybrid comparison in
Sections 11-12 is a measurement on one small corpus, not a general result
about hybrid retrieval.

### 24.3 Medical RAG Evaluation

Systematic evaluation of RAG for medicine is established. Xiong et al. (2024)
built MIRAGE, a benchmark of 7,663 questions from five medical QA datasets,
and evaluated 41 combinations of corpora, retrievers and backbone LLMs with
their MedRAG toolkit; they report that combining medical corpora and
retrievers performed best, and observe log-linear scaling and
"lost-in-the-middle" effects. Zhao et al. (2026), a scoping review of 157
studies of inference-time RAG and GraphRAG systems in health care, chart
evaluation separately by layer. They report that independent retrieval-layer
evaluation appeared in 29.9% of studies, grounding and faithfulness evaluation
in 26.1%, fine-grained evidence verification in 14%, formal safety-related
evaluation in 28.7%, and workflow-facing, prospective or deployment-level
evaluation in 10.8%, with most evaluations (89.2%) offline-only.

We use that layering to state what this project does and does not evaluate:

| Evaluation layer (after Zhao et al., 2026) | This project |
|---|---|
| Retrieval-layer evaluation | yes: Recall/MRR/nDCG/evidence term coverage, five configurations, corrected tests (Tier 1) |
| Grounding / faithfulness | partial: answer term coverage and a check of whether missing terms were in the retrieved text (Tier 2); no claim-level faithfulness judgement |
| Citation / source correctness | partial: sources shown are derived from the retrieved chunks; the inline-citation rate was measured (2 of 46 product-QA answers); citation faithfulness was not verified |
| Safety evaluation | limited: a 10-probe personal-advice probe set, 3 repeats (Tier 2) |
| Deployment / workflow evaluation | none; all evaluation is offline, single-user |

This mapping places the project within the review's categories; the review
does not show, and is not used to suggest, that the project is novel.

### 24.4 Context Sufficiency and Generation Failures

Joren et al. (2025) ask whether RAG errors arise because a model fails to use
the retrieved context or because the context is insufficient. They define
sufficient context, classify instances by it, and report that smaller models
"hallucinate or abstain often, even with sufficient context", while larger
models more often answer incorrectly instead of abstaining when context is
insufficient. The failures in Section 13.3 - 9 of 10 hybrid generation
failures had the missing expected term in the retrieved text - are consistent
with that general phenomenon for a 3B model. This project did not apply their
sufficiency classification, autorater or selective-generation method; "the
term is present in the retrieved text" is a lexical check and a weaker
condition than sufficient context.

### 24.5 Hallucination, Grounding and Citation Faithfulness

Niu et al. (2024) built RAGTruth, a corpus of nearly 18,000 responses
generated by diverse LLMs under RAG, manually annotated for hallucination at
the response and word levels, and used it to benchmark hallucination
frequencies and detection methods; they note that LLMs may still make claims
unsupported by or contradicting the retrieved content. Ma et al. (2026)
evaluate sentence-level citation faithfulness in retrieval-augmented medical
question answering with an NLI model, reporting that only 27-41% of citation
pairs were supported at the sentence level on BioASQ questions and that
citation faithfulness did not substantially improve with better retrieval.
This project measures neither word-level hallucination nor citation
faithfulness; its observations (false "not mentioned" claims, wrong-cell
selection, an inline-citation rate of 2 of 46 answers) are coarser and come
from far fewer cases.

### 24.6 Positioning of This Work

- RAG itself is established (Lewis et al., 2020).
- Dense retrieval with sentence embeddings, lexical BM25-style retrieval,
  Reciprocal Rank Fusion and FAISS are established (Reimers & Gurevych, 2019;
  Robertson & Zaragoza, 2009; Cormack et al., 2009; Johnson et al., 2017).
- Medical RAG evaluation is established (Xiong et al., 2024; Zhao et al.,
  2026).
- Hallucination / grounding evaluation (Niu et al., 2024) and
  context-sufficiency analysis (Joren et al., 2025) are established.
- Citation faithfulness in medical RAG is established (Ma et al., 2026).
- Therefore this project does **not** claim a novel RAG architecture or a
  novel retrieval algorithm.

The contribution is framed instead as **a reliability-oriented empirical
evaluation of a local healthcare-document RAG system, combining retrieval-layer
comparison, production-path measurement, answer-level failure analysis, safety
probes, and controlled prompt-generation interventions.** It differs from the
cited work in scope rather than in method:

- Unlike MIRAGE (Xiong et al., 2024), it studies one deployed configuration on
  a small, partly private corpus (45 questions, 5 documents) instead of many
  configurations on public medical QA datasets, and it measures the
  application's own production retrieval path.
- Unlike Joren et al. (2025), it classifies failures by stage with a lexical
  evidence check and manual reading, not by an autorated sufficiency label.
- Unlike RAGTruth (Niu et al., 2024) and VERICITE (Ma et al., 2026), it does
  not annotate hallucination spans or verify citations at sentence level.
- Against the layers charted by Zhao et al. (2026), it covers retrieval-layer
  evaluation and offline safety probes, covers grounding and source
  correctness only partially, and has no deployment or workflow evaluation.

Whether the combination of these elements is new has not been assessed by a
systematic literature search and is not claimed.

## 25. Claim Audit

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Hybrid improves ranking (whole corpus) | MRR/nDCG, Holm p = 0.0024, 13/0/32 | VERIFIED on this benchmark |
| 2 | Hybrid improves ranking (document-scoped) | p 0.021 uncorrected, 0.168 Holm(10), 0.042 Holm(4) | PARTIALLY SUPPORTED (family-dependent) |
| 3 | Hybrid improves Recall@5 / evidence term coverage | 1 question differs globally, 0 net scoped | NOT SUPPORTED |
| 4 | Production path improves on the bare scoped retriever | +0.004 MRR, 2 questions, p 0.5 | NOT SUPPORTED (negligible) |
| 5 | Helps terminology-heavy / multi-part queries | per-category MRR, n = 8-9, untested | PARTIALLY SUPPORTED (descriptive) |
| 6 | Acceptable retrieval latency | +2-3 ms over dense; ~30 ms retrieval | VERIFIED |
| 7 | Hybrid improves or worsens end-to-end answers | pass counts overlap across runs 1-5; run 5 not the production path; untested | NOT SUPPORTED either way |
| 8 | Residual failures mostly occur after retrieval | run 5 hybrid: 10 generation vs 3 retrieval failures; 9/10 with the term in the evidence | OBSERVED (one run) |
| 9 | Narrow prompt changes fix specific failure classes | one document, 5 near-identical runs per condition | OBSERVED (exploratory) |
| 10 | Some omissions are model properties | persistent across prompt variants; one model | INFERRED |
| 11 | Works offline after setup | proxy-blocked run succeeded; localhost bindings; HF download fallback if model missing | PARTIALLY SUPPORTED |
| 12 | Robust to ingestion failures | 15 edge cases + regression tests | VERIFIED for tested cases |
| 13 | Prevents personal medical advice | 12/12 probes declined, 3 near-identical repeats, private documents | NOT SUPPORTED as a general claim; OBSERVED on probes |
| 14 | Answers document questions normally | 9/12 document probes | OBSERVED, with a false-negative failure |
| 15 | Source-grounded answers | Sources 100%; inline citations 2/46; numeric spot check exact | PARTIALLY SUPPORTED |
| 16 | Ambiguity detection improves retrieval | 16 signals, 3 positives, not significant | NOT SUPPORTED |
| 17 | Structured extraction improves table QA | not built; no qualifying benchmark questions | NOT TESTED |
| 18 | The system is novel | no systematic literature comparison | NOT CLAIMED |

## 26. Reproducing, and what the frozen-baseline check shows

```bash
python research/chunk_evidence_audit.py
python research/counterbalanced_retrieval_benchmark.py
python research/ambiguity_study.py
python scripts/run_evaluation.py --dataset data/healthcare_eval_questions.json --label publication \n    --db-path data/research_corpus/metadata.db --faiss-path data/research_corpus/faiss.index \n    --output-json data/evaluation_results.publication.json --output-summary data/evaluation_summary.publication.txt
python scripts/run_product_qa.py --output output/product_qa_results_<date>.json
pytest -q --basetemp .pytest_tmp
```

The research index (`data/research_corpus/`), the question files and the
private documents are local-only (see the README's "Private data" section),
so these commands reproduce the results only on a machine that has them.
Generation-based results will differ in wording between runs. The
intervention-study, safety-probe and performance scripts were not committed.

**The frozen-baseline check** recomputed Recall@5, MRR@5, nDCG@5 and evidence
term coverage for all 45 questions in all five configurations (production
scored on its first 5 chunks) and compared them with the stored values:
0 mismatches. This shows that, after the production changes, the retrieval
metrics on the research benchmark are unchanged. It does **not** show that
chunk lists are identical beyond what those four metrics capture, anything
about production chunks past rank 5, that prompts, answers or generation
behaviour are unchanged, that end-to-end, product QA or safety results still
hold, that latency is unchanged, or that the ground-truth labels are correct.
The check script was not committed.

## 27. References

Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank
fusion outperforms Condorcet and individual rank learning methods. In
*Proceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR '09)*, pp. 758-759.
https://doi.org/10.1145/1571941.1572114

Holm, S. (1979). A simple sequentially rejective multiple test procedure.
*Scandinavian Journal of Statistics*, 6(2), 65-70.
https://www.jstor.org/stable/4615733

Johnson, J., Douze, M., & Jégou, H. (2017). Billion-scale similarity search
with GPUs. arXiv:1702.08734. https://doi.org/10.48550/arXiv.1702.08734

Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., & Rashtchian, C.
(2025). Sufficient context: A new lens on retrieval augmented generation
systems. In *International Conference on Learning Representations (ICLR
2025)*. arXiv:2411.06037.
https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
(2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In
*Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.
arXiv:2005.11401. https://doi.org/10.48550/arXiv.2005.11401

Ma, Y., Chu, B., & Fuhr, N. (2026). VERICITE: Evaluating sentence-level
citation faithfulness in retrieval-augmented medical question answering. In
*Proceedings of BioNLP 2026 (BioNLP@ACL 2026)*, pp. 753-759.
https://doi.org/10.18653/v1/2026.bionlp-1.62

Meta AI. (2024, September 25). *Llama 3.2: Revolutionizing edge AI and vision
with open, customizable models* [Blog post].
https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T.
(2024). RAGTruth: A hallucination corpus for developing trustworthy
retrieval-augmented language models. In *Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers)*, pp. 10862-10878. https://doi.org/10.18653/v1/2024.acl-long.585

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint
Conference on Natural Language Processing (EMNLP-IJCNLP)*, pp. 3982-3992.
https://doi.org/10.18653/v1/D19-1410

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4),
333-389. https://doi.org/10.1561/1500000019 (volume and pages as listed by the
original publisher, now publishers; the DOI's current Crossref record, after
the journal moved to Emerald, lists 4(1-2), 1-174.)

Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics
Bulletin*, 1(6), 80-83. https://doi.org/10.2307/3001968

Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). Benchmarking
retrieval-augmented generation for medicine. In *Findings of the Association
for Computational Linguistics: ACL 2024*, pp. 6233-6251.
https://doi.org/10.18653/v1/2024.findings-acl.372

Zhao, Y., Miao, Y., Guo, R., Luo, Y., Wang, H., & Wu, Y. (2026). Evaluation
methods for inference-time retrieval-augmented and graph retrieval-augmented
large language models in health care: Scoping review. *Journal of Medical
Internet Research*, 28, e90046. https://doi.org/10.2196/90046
