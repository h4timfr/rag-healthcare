# Results Tables (paper-ready)

Copy-ready summary of the measured results. Sources: retrieval -
`research/results/counterbalanced_retrieval_20261001T081145Z.json`;
end-to-end - `data/evaluation_summary.publication_run5_porter.txt`; other rows -
FINAL_REPORT.md sections named in each caption. n = 45 questions, 5 documents,
225 chunks, k = 5.

**Table 1. Retrieval quality and latency** (5 timing repetitions per
question, counterbalanced order; latency in ms).

| Configuration | Recall@5 | MRR@5 | nDCG@5 | Term cov. | Median | p95 |
|---|---|---|---|---|---|---|
| Global dense | 0.911 | 0.692 | 0.746 | 0.911 | 26.0 | 28.6 |
| Global hybrid | 0.933 | 0.809 | 0.840 | 0.933 | 28.2 | 30.5 |
| Scoped dense | 0.933 | 0.740 | 0.787 | 0.933 | 28.3 | 30.9 |
| Scoped hybrid | 0.933 | 0.826 | 0.853 | 0.933 | 31.3 | 34.2 |
| Production path | 0.933 | 0.830 | 0.856 | 0.933 | 40.8* | 53.5* |

\* includes prompt construction.

**Table 2. Paired tests** (Wilcoxon signed-rank on nonzero differences;
r = matched-pairs rank-biserial; Holm over the 10 testable comparisons).

| Comparison | Δ mean | W / L / T | r | p | p Holm |
|---|---|---|---|---|---|
| Global hybrid - dense, MRR@5 | +0.117 | 13 / 0 / 32 | 1.00 | 0.00024 | 0.0024 |
| Global hybrid - dense, nDCG@5 | +0.094 | 13 / 0 / 32 | 1.00 | 0.00024 | 0.0024 |
| Global hybrid - dense, Recall@5 | +0.022 | 1 / 0 / 44 | 1.00 | 1.00 | 1.00 |
| Scoped hybrid - dense, MRR@5 | +0.086 | 11 / 2 / 32 | 0.71 | 0.021 | 0.168 |
| Scoped hybrid - dense, nDCG@5 | +0.066 | 11 / 2 / 32 | 0.69 | 0.026 | 0.179 |
| Scoped hybrid - dense, Recall@5 | 0.000 | 1 / 1 / 43 | 0.00 | 1.00 | 1.00 |
| Production - scoped hybrid, MRR@5 | +0.004 | 2 / 0 / 43 | 1.00 | 0.50 | 1.00 |

Sensitivity: Holm over the 4 ranking tests gives 0.042 for both scoped
comparisons; sign test on the 13 non-tied scoped questions p = 0.0225.

**Table 3. End-to-end (single run, neutral evaluation prompt, llama3.2:3b).**

| | Dense | Hybrid |
|---|---|---|
| Pass (page retrieved and all expected terms in answer) | 35/45 | 32/45 |
| Retrieval failure | 4 | 3 |
| Generation failure (page retrieved) | 6 | 10 |
| Answer term coverage | 0.844 | 0.800 |
| Mean end-to-end latency (s) | 31.1 | 25.3 |

Of the 10 hybrid generation failures, 7 had the missing fact in the retrieved
text (false absence 2, wrong cell/granularity 2, wrong entity 1, component
omission 2); 1 was a grader paraphrase miss, 1 a benign omission, 1 had the
evidence absent.

**Table 4. Product QA through the application path** (21 cases / 23 turns,
2 runs).

| Pass | Fact coverage | Document coverage | Sources shown | Latency p50 / p95 |
|---|---|---|---|---|
| 21/23, 21/23 | 87% | 95% | 100% | 9.3 / 51.9 s; 6.8 / 38.0 s |

**Table 5. Generation intervention study** (`UMNwriteup.pdf`, 5 runs per
condition; evidence identical across conditions).

| Failure | Before | After | Class |
|---|---|---|---|
| Section comparison answer cut off | 1/5 (2/3 in trace) | 0/5 | budget |
| Initial vs revised lists kept separate | wrong style (multi-document) | 5/5 | instruction |
| Shared/renamed items handled correctly | - | 4/5 | instruction (residual model error) |
| Current-plan diuretics listed | 0/5 | 5/5 | instruction |
| Non-drugs listed as medications | 1/8 (with exhaustive rule) | 0/5 | instruction |
| Conditional cholesterol drug listed | 0/5 | 2/5 | partly model-level |
| Unnamed historical antihypertensive | 0/5 | 0/5 | model-level / in-chunk contradiction |
| "Give 3-5 reasons" answered in a single sentence | 3/5 | 0/5 (numbered list, 3 items) | instruction |
| GERD-specific reasons given | 0/5 | 0/5 | model-level |
| Margin note inside list item | 3-5/5 | 3-5/5 | extraction |

**Table 6. Safety probes** (10 probes x 3 repeats).

| Probe type | Result |
|---|---|
| Personal or dependent advice | 12/12 declined with referral |
| Document questions about treatment/medication | 11/12 answered from the document |
