# Research Contribution

What this project contributes, what it found, and what it does not claim.
All numbers come from [FINAL_REPORT.md](FINAL_REPORT.md), which names each
source, its metric family and its evidence status.

Metric families used below, never to be mixed: **retrieval evidence metrics**
(Recall/MRR/nDCG/evidence term coverage; deterministic, committed results),
the **answer metric** (answer term coverage) and the **joint pass metric**
(all expected pages retrieved and all expected terms in the answer) from the
research harness, **product QA** (application path, historical), and
**exploratory intervention results** (one document, few near-identical
generations).

Provenance tiers (defined in FINAL_REPORT.md): **Tier 1** - repository-
reproducible (frozen retrieval benchmark, test suite, configuration);
**Tier 2** - measured but historical or not reproducible from the repository
(end-to-end runs, product QA, safety probes, intervention study, performance,
ambiguity and audit outputs, demo checks); **Tier 3** - interpretation (model
attribution, extraction attribution, failure-class labels).

## Research problem

A healthcare-document question-answering system that must stay on one machine
(private documents) is limited to small local generators. Aggregate accuracy
does not tell the builder of such a system whether to invest in retrieval,
prompting, extraction or the model. The problem studied is: **where, stage by
stage, does a fully local healthcare RAG system fail, and which of those
failures respond to which kind of intervention?**

## Research questions

1. Does weighted dense + FTS5 hybrid retrieval rank relevant evidence higher
   than dense-only retrieval, with multiplicity correction, in whole-corpus
   and document-scoped settings?
2. Does the production retrieval path (intent, widening, per-part retrieval,
   presentation order) change retrieval quality relative to the bare
   retriever?
3. Is there evidence that retrieval gains carry through to end-to-end answers
   with a 3B local generator?
4. Where do failures occur - retrieval, evidence, extraction, generation,
   grading - and what classes occur when the evidence is present?
5. (Exploratory) Which generation failures can narrowly gated prompt and
   output-budget changes reduce without modifying retrieval, and which
   persist?
6. Does a gated personal-advice reminder prevent document content being turned
   into personal advice on probed phrasings, without suppressing document
   questions?
7. Can retrieval statistics computed before generation flag retrieval misses?

## Hypotheses (stated where the evidence allows a test)

| Hypothesis | Outcome | Tier |
|---|---|---|
| H1. Hybrid improves ranking (MRR/nDCG) over dense in the whole corpus | supported: +0.117 MRR@5, 13/0 wins/losses, Holm p = 0.0024 | 1 |
| H2. The same holds when retrieval is scoped to the question's document | directional only: +0.086 MRR@5, 11/2, Holm p = 0.168 over 10 tests (0.042 over the 4 ranking tests) | 1 |
| H3. Hybrid improves Recall@5 | not supported: 1 question differs (global), 1/1 (scoped) | 1 |
| H4. A ranking gain improves the end-to-end joint pass rate | not tested statistically; no supporting evidence. Five single runs of the research harness (whole-corpus retrieval, evaluation prompt - not the production path) gave overlapping pass counts (dense 34-36, hybrid 32-35); run 5 had dense 35/45, hybrid 32/45, which is descriptive only | 2 |
| H5. Pre-generation retrieval signals detect retrieval misses | not supported: best AUROC 0.913 on 3 positives, family-wise p = 0.121 | 2 |

Research questions 4-6 are descriptive (failure classification, exploratory
intervention and probe results over small numbers of runs) and are not framed
as hypothesis tests.

## Concrete contributions

1. **A corrected retrieval comparison in the deployed setting.** Five
   configurations - global and document-scoped dense and hybrid, plus the full
   production retrieval path - measured per question with Wilcoxon
   signed-rank tests, rank-biserial effect sizes and Holm correction.
   Retrieval is deterministic (0 of 225 question-configuration measurements
   changed across 5 repetitions), so these ranking results do not depend on
   measurement order; latency was measured with the three methods in a
   counterbalanced order after a warm-up.
2. **A stage-separated failure analysis.** Every generation failure of the
   end-to-end run checked against the retrieved chunk text: 9 of 10 hybrid
   generation failures had the missing term in the evidence (7 judged genuine
   generation errors), plus extraction, chunking, grading and ground-truth
   failure classes.
3. **An exploratory intervention study with ablations and side-effect
   checks.** Each generation change was diagnosed on traced prompts, ablated
   one factor at a time over 5 runs on one document, gated narrowly, and
   checked against previously passing questions, concept comparisons and the
   stored retrieval benchmark metrics. On that document it separates failures
   that responded to instruction changes (answer shape, truncation, section
   identity, current-plan medication completeness) from failures that
   persisted under every variant tested (GERD reasoning, an unnamed historical
   medication). Repeated generations were near-identical, so these are
   observations, not estimates.
4. **Negative results reported**: no evidence that the ranking gain improves
   answers; no usable pre-generation miss detector on this benchmark; no
   benchmark basis for a structured-extraction component; a broader prompt
   change and a global safety-rule rewrite rejected for measured side effects.
5. **A local system and harness** (engineering, not research novelty):
   index-consistency guarantees, offline operation after first-use model
   downloads, and a committed retrieval benchmark whose stored per-question
   metrics were re-checked after the production changes (the re-check script
   itself is not committed).

## Empirical findings

- Hybrid retrieval improves ranking, not coverage, at about 2-3 ms extra
  retrieval latency (median 26.0 -> 28.2 ms global; 28.3 -> 31.3 ms scoped).
- The full production path adds +0.004 MRR@5 over the bare scoped hybrid
  retriever (2 questions differ) at about 9 ms more (prompt building included).
- With a 3B generator and this harness, there is no measurable answer-level
  gain from the ranking improvement; in the inspected run most residual errors
  occurred after successful retrieval.
- Generation failure classes with evidence present: false "not mentioned",
  wrong table cell / granularity, wrong entity, component omission, comparison
  set-difference errors, counting.
- In the exploratory study, instruction-level failures (wrong answer shape,
  truncation, wrong comparison style, incomplete per-part lists) responded to
  narrow prompt and budget changes; some omissions persisted under every
  condition tested. Attributing those to the model is an inference - only one
  model was tested.
- Two failures were traced, by inspecting chunk text, to processing upstream
  of retrieval and prompting: two-column PDF extraction and heading-less
  overlap chunks (not tested by changing extraction or chunking).
- The gated advice reminder declined 12/12 personal-advice probes; 9/12
  document-question probes were answered from the document (one question
  gave a false "nothing reported" in all 3 runs). Rewording the global rule
  instead degraded an unrelated summary in 4 of 5 runs.

## Novelty boundaries and positioning

Full related work and references: FINAL_REPORT.md Sections 24 and 27.

- RAG itself is established (Lewis et al., 2020).
- Dense retrieval with sentence embeddings (Reimers & Gurevych, 2019), BM25
  lexical ranking (Robertson & Zaragoza, 2009), Reciprocal Rank Fusion
  (Cormack et al., 2009) and FAISS (Johnson et al., 2017) are established;
  the 2:1 fusion weighting used here is a configuration choice.
- Medical RAG evaluation is established (Xiong et al., 2024), and recent
  healthcare RAG evaluation has been charted by layer - retrieval, grounding
  and faithfulness, evidence verification, safety, deployment (Zhao et al.,
  2026).
- Hallucination evaluation under RAG (Niu et al., 2024) and context-sufficiency
  analysis, including the finding that smaller models often hallucinate or
  abstain even with sufficient context (Joren et al., 2025), are established.
- Citation faithfulness in medical RAG has been evaluated at sentence level
  (Ma et al., 2026).
- Therefore this project does not claim a novel RAG architecture, retrieval
  algorithm or evaluation method. Intent-specific prompt instructions,
  the personal-advice reminder and local serving with Ollama are engineering
  choices, not claimed contributions.

The contribution is framed as **a reliability-oriented empirical evaluation
of a local healthcare-document RAG system, combining retrieval-layer
comparison, production-path measurement, answer-level failure analysis,
safety probes, and controlled prompt-generation interventions.** It differs
from the cited work in scope, not method: one deployed configuration on a
small, partly private corpus rather than a large public benchmark (Xiong et
al., 2024); a lexical evidence check and manual reading rather than an
autorated sufficiency label (Joren et al., 2025); no hallucination-span or
sentence-level citation annotation (Niu et al., 2024; Ma et al., 2026); and,
in the layers charted by Zhao et al. (2026), retrieval-layer evaluation and
offline safety probes, partial grounding and source-correctness checks, and
no deployment or workflow evaluation. Whether this combination is new has not
been assessed by a systematic literature search and is not claimed.

## Explicit non-claims

This project does **not** claim:

- a new retrieval, fusion, chunking or generation method;
- that hybrid retrieval improves, or worsens, answer accuracy;
- that the document-scoped ranking improvement is statistically confirmed;
- that the end-to-end research runs measure the production path;
- clinical validity, safety for clinical use, or suitability for patient care;
- that the safety guard handles phrasings beyond those probed;
- generalisation beyond five English documents, one 3B generator, one
  embedding model and CPU-only hardware;
- that the intervention results generalise beyond the one document they were
  measured on, or that their "5/5" counts are independent samples;
- that the failure-class frequencies are stable estimates (each class has
  1-3 instances);
- that the product QA results describe the current application (they predate
  the latest prompt changes);
- any comparison with published systems or benchmarks (none was run).

## What a stronger study would need

A larger, independently annotated benchmark (for example ~240 questions in
matched present/absent pairs, table questions with aggregate vs sub-group
distractors, and multi-component questions, each generated 3-5 times),
repeated end-to-end runs of the production path with a paired test, persisted
generation outputs and annotations, a second local generator for comparison,
claim-level rather than term-level grading, and layout-aware extraction
evaluated on the frozen research corpus.
