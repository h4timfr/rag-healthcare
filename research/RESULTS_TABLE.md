# Results Tables (paper-ready)

Copy-ready summary of the results in [FINAL_REPORT.md](FINAL_REPORT.md), which
gives the definitions, provenance and caveats. The tables are grouped by
metric family; numbers from different families must not be compared with
each other.

| Family | Tables | Status |
|---|---|---|
| Retrieval evidence metrics (no generation) | 1, 2 | VERIFIED; committed result file; deterministic |
| Answer metric and joint pass metric (research harness, whole-corpus retrieval, evaluation prompt) | 3 | OBSERVED; one run per row; local result files |
| Product QA (application path) | 4 | OBSERVED; historical (before commit `fd1cdd5`), private documents, not re-runnable now |
| Exploratory intervention results | 5 | OBSERVED; one document, 5 near-identical runs per condition, not persisted in the repository |
| Safety probes | 6 | OBSERVED; private documents, 3 near-identical repeats |

Retrieval: `research/results/counterbalanced_retrieval_20261001T081145Z.json`.
n = 45 questions, 5 documents, 225 chunks, k = 5.

**Table 1. Retrieval evidence metrics and latency.** Term coverage here is
*evidence* term coverage (expected terms present in the retrieved chunk
text). Latency in ms: median and nearest-rank p95 over the 45 per-question
means of 5 timed repetitions.

| Configuration | Recall@5 | MRR@5 | nDCG@5 | Evidence term cov. | Median | p95 |
|---|---|---|---|---|---|---|
| Global dense | 0.911 | 0.692 | 0.746 | 0.911 | 26.0 | 28.6 |
| Global hybrid | 0.933 | 0.809 | 0.840 | 0.933 | 28.2 | 30.5 |
| Scoped dense | 0.933 | 0.740 | 0.787 | 0.933 | 28.3 | 30.9 |
| Scoped hybrid | 0.933 | 0.826 | 0.853 | 0.933 | 31.3 | 34.2 |
| Production path | 0.933 | 0.830 | 0.856 | 0.933 | 40.8* | 53.5* |

\* includes prompt construction.

**Table 2. Paired tests on retrieval metrics** (Wilcoxon signed-rank on
nonzero differences; W / L / T = b wins / a wins / ties; r = matched-pairs
rank-biserial; Holm over the 10 testable comparisons).

| Comparison | Δ mean | W / L / T | r | p | p Holm |
|---|---|---|---|---|---|
| Global hybrid - dense, MRR@5 | +0.117 | 13 / 0 / 32 | 1.00 | 0.00024 | 0.0024 |
| Global hybrid - dense, nDCG@5 | +0.094 | 13 / 0 / 32 | 1.00 | 0.00024 | 0.0024 |
| Global hybrid - dense, Recall@5 | +0.022 | 1 / 0 / 44 | 1.00 | 1.00 | 1.00 |
| Scoped hybrid - dense, MRR@5 | +0.086 | 11 / 2 / 32 | 0.71 | 0.021 | 0.168 |
| Scoped hybrid - dense, nDCG@5 | +0.066 | 11 / 2 / 32 | 0.69 | 0.026 | 0.179 |
| Scoped hybrid - dense, Recall@5 | 0.000 | 1 / 1 / 43 | 0.00 | 1.00 | 1.00 |
| Production - scoped hybrid, MRR@5 | +0.004 | 2 / 0 / 43 | 1.00 | 0.50 | 1.00 |

Evidence term coverage gives the same results as Recall@5 in both settings
(Global +0.022, 1/0/44; Scoped 0.000, 1/1/43). Sensitivity: Holm over the 4
ranking tests gives 0.042 for both scoped comparisons; sign test on the 13
non-tied scoped questions p = 0.0225. The scoped result is directional, not
confirmed.

**Table 3. End-to-end research run: answer metric and joint pass metric.**
`scripts/run_evaluation.py`, **not the production path**: whole-corpus
retrieval (no document scope), neutral evaluation prompt, 256-token output,
`llama3.2:3b`. Pass = every expected page retrieved **and** every expected
term in the answer. Answer term coverage = expected terms present in the
answer, averaged over all 45 questions.

| Run | Dense pass | Hybrid pass | Dense answer term cov. | Hybrid answer term cov. |
|---|---|---|---|---|
| 1 | 34/45 | 35/45 | 0.822 | 0.856 |
| 2 | 35/45 | 34/45 | 0.844 | 0.833 |
| 3 | 34/45 | 32/45 | 0.822 | 0.800 |
| 4 | 36/45 | 35/45 | 0.856 | 0.856 |
| 5 (Porter keyword index, current) | 35/45 | 32/45 | 0.844 | 0.800 |

Runs 1-4 used the pre-Porter keyword index and Ollama 0.34.0; run 5 used
Ollama 0.34.2. Each row is one run of a non-deterministic generator and no
test was run: the run-5 difference (35 vs 32) is **descriptive** and is not
evidence of a general hybrid-vs-dense effect on answers. Run 5 detail: retrieval
failures (at least one expected page missed) 4 dense / 3 hybrid; generation
failures (all pages retrieved, answer missed a term) 6 dense / 10 hybrid; mean
end-to-end latency 31.1 s / 25.3 s.

Of the 10 run-5 hybrid generation failures, 9 had the missing expected term in
the retrieved text (measured). Of those 9, the authors judge 7 to be genuine
generation errors (false absence 2, wrong cell/granularity 2, wrong entity 1,
component omission 2); the other 2 are a grader number-format miss and a minor
omission. The tenth had the evidence absent.

**Table 4. Product QA through the application path** (21 cases / 23 turns,
2 runs on 2026-10-01). Historical: measured before the prompt changes of
commit `fd1cdd5`, on two private documents plus the project's review PDF; not
re-runnable now.

| Run | Pass | Fact coverage | Document coverage | Sources shown | Validator clean | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| 1 | 21/23 | 87% | 95% | 100% | 100% | 9.3 / 51.9 s |
| 2 | 21/23 | 87% | 95% | 100% | 96% | 6.8 / 38.0 s |

Inline `[n]` citations: 2 of 46 answers.

**Table 5. Exploratory generation intervention study** (`UMNwriteup.pdf`, one
document; 5 runs per condition at temperature 0.1, often near-identical;
evidence identical across conditions). "Class" is the authors' attribution;
"persistent" means not changed by any prompt or budget variant tested.

| Failure | Before | After | How scored | Class |
|---|---|---|---|---|
| Section comparison answer cut off | 1/5 (2/3 in trace) | 0/5 | Ollama `done_reason` | budget |
| Initial vs revised lists kept separate | not scored at baseline | 5/5 | reading | instruction |
| Shared/renamed items handled correctly | not scored at baseline | 4/5 | reading | instruction (residual error 1/5) |
| Current-plan diuretics listed | 0/5 | 5/5 | text match | instruction |
| Non-drugs listed as medications | 2/8 (with the exhaustive rule, before the guard) | 0/5 | text match + reading | instruction |
| Conditional cholesterol drug listed | 0/5 | 2/5 | text match | partly persistent |
| Unnamed historical antihypertensive | 0/5 | 0/5 | text match | persistent (in-chunk contradiction partly causal) |
| "Give 3-5 reasons" answered without a list | 3/5 (2 single-sentence) | 0/5 (numbered list, 3 items) | text match + reading | instruction |
| GERD-specific reasons given | 0/5 | 0/5 | text match | persistent |
| Margin note inside list item | 3/5 | 5/5 | text match | extraction (inferred) |

**Table 6. Safety probes** (10 probes x 3 repeats on the production library of
2026-10-01, including two private documents).

| Probe type | Result |
|---|---|
| Personal or dependent advice | 12/12 declined (9 referred to a doctor or pharmacist, 3 to a healthcare professional) |
| Document questions about treatment/medication | 9/12 answered from the document; one question gave a false "no medication reported" in all 3 runs |
