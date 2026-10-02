# Research Contribution

What this project contributes, what it found, and what it does not claim.
All numbers come from [FINAL_REPORT.md](FINAL_REPORT.md), which names each
source file.

## Research problem

A healthcare-document question-answering system that must stay on one machine
(private documents) is limited to small local generators. Aggregate accuracy
does not tell the builder of such a system whether to invest in retrieval,
prompting, extraction or the model. The problem studied is: **where, stage by
stage, does a fully local healthcare RAG system fail, and which of those
failures respond to which kind of intervention?**

## Research questions

1. Does weighted dense + FTS5 hybrid retrieval rank relevant evidence higher
   than dense-only retrieval, measured without order bias and with
   multiplicity correction, in whole-corpus and document-scoped settings?
2. Does the production retrieval path (intent, widening, per-part retrieval,
   presentation order) change retrieval quality relative to the bare
   retriever?
3. Do retrieval gains carry through to end-to-end answers with a 3B local
   generator?
4. Where do failures occur - retrieval, evidence, extraction, generation,
   grading - and what classes occur when the evidence is present?
5. Which generation failures can narrowly gated prompt and output-budget
   changes reduce without modifying retrieval, and which remain?
6. Does a gated personal-advice reminder prevent document content being turned
   into personal advice without suppressing document questions?
7. Can retrieval statistics computed before generation flag retrieval misses?

## Hypotheses (stated where the evidence allows a test)

| Hypothesis | Outcome |
|---|---|
| H1. Hybrid improves ranking (MRR/nDCG) over dense in the whole corpus | supported: +0.117 MRR@5, 13/0 wins/losses, Holm p = 0.0024 |
| H2. The same holds when retrieval is scoped to the question's document | directional only: +0.086 MRR@5, 11/2, Holm p = 0.168 over 10 tests (0.042 over the 4 ranking tests) |
| H3. Hybrid improves Recall@5 | not supported: 1 question differs (global), 1/1 (scoped) |
| H4. A ranking gain improves end-to-end answer pass rate | not supported in the one end-to-end run: 32/45 hybrid vs 35/45 dense |
| H5. Pre-generation retrieval signals detect retrieval misses | not supported: best AUROC 0.913 on 3 positives, family-wise p = 0.121 |

Research questions 4-6 are descriptive (failure classification, intervention
and probe results over small numbers of runs) and are not framed as
hypothesis tests.

## Concrete contributions

1. **A counterbalanced, corrected retrieval comparison in the deployed
   setting.** Five configurations - global and document-scoped dense and
   hybrid, plus the full production path - measured per question in a Latin-
   square order with warm-up, repeated five times, Wilcoxon signed-rank with
   rank-biserial effect sizes and Holm correction, with determinism checked
   (0 of 225 repeats differed).
2. **A stage-separated failure analysis.** Every end-to-end failure checked
   against the retrieved chunk text, separating retrieval misses from
   evidence-present generation failures (7 of 45), plus extraction, chunking,
   grading and ground-truth failure classes.
3. **An intervention study with ablations and side-effect measurements.**
   Each generation change was diagnosed on traced prompts, ablated one factor
   at a time over 5 runs, gated narrowly, and checked for regressions on
   previously passing questions, on concept comparisons and on the frozen
   retrieval baseline. It distinguishes fixable failure classes (answer
   shape, truncation, section identity, current-plan medication completeness)
   from model-level ones (GERD reasoning, an unnamed historical medication).
4. **Negative results reported**: no end-to-end gain from better ranking; no
   usable pre-generation miss detector on this benchmark; no benchmark
   justification for a structured-extraction component; a broader prompt
   change and a global safety-rule rewrite rejected for measured side effects.
5. **A reproducible local system and harness** (engineering, not research
   novelty): index-consistency guarantees, offline operation, a frozen
   baseline with a per-question regression check.

## Empirical findings

- Hybrid retrieval improves ranking, not coverage, at about 2-3 ms extra
  retrieval latency (median 26.0 -> 28.2 ms global; 28.3 -> 31.3 ms scoped).
- The full production path adds +0.004 MRR@5 over the bare scoped hybrid
  retriever (2 questions differ) at about 9 ms more (prompt building included).
- With a 3B generator, the ranking gain did not reach the answers; most
  residual errors happen after successful retrieval.
- Generation failure classes with evidence present: false "not mentioned",
  wrong table cell / granularity, wrong entity, component omission, comparison
  set-difference errors, counting.
- Instruction-level failures (wrong answer shape, truncation, wrong comparison
  style, incomplete per-part lists) responded to narrow prompt and budget
  changes; model-level omissions did not, under any condition tested.
- Two failures are upstream of both retrieval and prompting: two-column PDF
  extraction and heading-less overlap chunks.
- The gated advice reminder declined 12/12 personal-advice probes and answered
  11/12 document questions; rewording the global rule instead degraded an
  unrelated summary in 4 of 5 runs.

## Novelty boundaries

The architecture is standard. Hybrid lexical + dense retrieval, (weighted)
Reciprocal Rank Fusion, FAISS, SQLite FTS5, sentence-transformer embeddings,
local inference with Ollama, intent-specific prompt instructions and
refusal/abstention prompting are all established techniques. Observations that
RAG answers can be unfaithful or incomplete, that small models omit
components of multi-part questions, and that hallucinated absence occurs are
also reported in prior work (see FINAL_REPORT.md Section 24 for works
identified for positioning, to be verified before citation).

What may be useful to others is methodological and empirical: the
combination of counterbalanced, corrected measurement in the deployed scoped
setting, failure attribution checked against the actual retrieved text, and
an intervention study that reports side effects and non-effects - on a small,
fully local system.

## Explicit non-claims

This project does **not** claim:

- a new retrieval, fusion, chunking or generation method;
- that hybrid retrieval improves answer accuracy (it did not in our run);
- that the document-scoped ranking improvement is statistically confirmed;
- clinical validity, safety for clinical use, or suitability for patient care;
- that the safety guard handles phrasings beyond those probed;
- generalisation beyond five English documents, one 3B generator, one
  embedding model and CPU-only hardware;
- that the intervention results generalise beyond the one document they were
  measured on;
- that the failure-class frequencies are stable estimates (each class has
  1-3 instances);
- any comparison with published systems or benchmarks (none was run).

## What a stronger study would need

A larger, independently annotated benchmark (for example ~240 questions in
matched present/absent pairs, table questions with aggregate vs sub-group
distractors, and multi-component questions, each generated 3-5 times) would
allow per-class detection estimates, a second local generator for comparison,
claim-level rather than term-level grading, and layout-aware extraction
evaluated on the frozen research corpus.
