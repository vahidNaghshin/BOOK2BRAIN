# Phase 2 evaluation report: knowledge extraction quality

Measured on 2026-11-25 against manually reviewed reference results. Every number below comes from
`python -m tools.evaluate_extraction ...`; the full machine-readable reports (matched pairs, misses, extras,
equations, relationships, run metadata) are in [`evaluation/results/`](../../evaluation/results/).
How to run it: [README, Evaluating extraction](../../README.md#evaluating-extraction-phase-2-day-5).

## 1. What was evaluated, and what was not

| | Result |
|---|---|
| Real model output evaluated | **Yes, one model**: `stormy-claude-haiku-5.5` (Claude Haiku 5.5, temperature 0, 4096 max output tokens, structured output mode `none`), the four extraction runs already stored in the development database for *Introduction to Latent Variable Energy-Based Models* (prompt versions `1.1.0` and `1.2.0`). |
| Passages covered | Reading-order passages 10-15 (pages 7-11). Passages 10-14 were extracted directly; 15 only appears through units read from passage 14's neighbouring context. Categories extracted: concept, principle, equation, relationships. |
| Categories never extracted on the real book | method, framework, evidence, example. They are covered by the second dataset, which has not been run against a live model (see [model-comparison.md](model-comparison.md)). |
| Harness validated | Yes, with scripted models through the real pipeline (`tests/test_evaluation_runner.py`). That proves the metrics measure what they claim and that runs are repeatable. **It says nothing about any model's quality.** |
| Live evaluation of `study_guide_v1` | **Not run**: no model was usable (Stormy needs `STORMY_API_TOKEN`, which is not configured; Ollama is not running; no OpenAI/Anthropic key). Outstanding. |
| Reference review | Both datasets are `draft`: written by the AI assistant from the passage text before any extraction was inspected, **not yet signed off by a human**. |

## 2. Method

* **Reference results** (`evaluation/datasets/`) are separate from extraction results and are never produced by a model. For each passage they list the units a careful reader expects (acceptable names, key facts, forbidden phrases, source quote, page range), required versus merely acceptable units, relationships, and for equations the symbols, acceptable capture statuses and whether the source is damaged enough that a careful system must flag it.
* **Counting rules** are stated in each report ("Metric definitions") so any figure can be re-derived. The important ones: recall counts required units whose own category was extracted on their passage; precision counts unmatched units only in passages whose reference is exhaustive; equation figures include equations the model read while extracting a neighbouring passage; a metric with no denominator is `n/a`.
* **Matching is lexical** (names, key facts, symbols). It shows presence, placement and traceability. It cannot show that a paraphrase preserves the author's meaning, so "key-fact coverage" is a *proxy* (and was 100% here partly because matching already requires some facts; do not read it as semantic accuracy). The reports list every matched pair for a reviewer.
* **Repeatability**: `--repeat N` runs the same passages N times, each on a fresh evaluation book (the pipeline never stores the same unit twice for a passage, so repeating on one book would find the earlier units). Each report records model configuration, generation parameters, prompt versions and run ids.

Two corrections were made to the method *after* looking at the first real results; both are disclosed because they could have flattered the model:

1. Matching was too lenient (a name that was a word inside a longer title, or a single shared word, recognised a unit). It was tightened (a name must be most of the title; a name alone needs some of the key facts). This lowered, not raised, the figures.
2. A reference unit is now only scored when *its own category* was extracted on its passage. Before, a required principle was counted as missed on passages where only concepts had been requested.

One reference unit (passage 12, "probabilistic models become intractable in high dimensions") that the source states in its first sentence and the first draft had missed was added as **optional**, so it cannot raise recall (noted in the dataset).

## 3. Results on the real book (Claude Haiku 5.5, 17 extractions, 25 model calls)

Dataset `latent_ebm_v1` v1.0.1, 6 passages, 45 reference units (27 required).

| Dimension | Metric | Result |
|---|---|---|
| **A. Completeness** | Required units found (recall) | **88.2% (15/17)**: concepts 9/9, principles 5/7 |
| | In passages with several required units | 85.7% (12/14) |
| | Precision (matched, or acceptable, units) | 74.1% (20/27); 1 further unmatched unit in a non-exhaustive passage needs a reviewer |
| | Passage that should stay empty (a roadmap) left without unexpected units | **0/1**: it produced a content-free "EBMs" unit and repeated two concepts |
| | Required units apparently merged into another | 1 candidate (training of an EBM, folded into the "energy function used only for inference" unit) |
| **B. Semantic accuracy** | Key facts present in matched units (lexical proxy) | 100% (60/60) |
| | Matched units containing a phrase the reviewer ruled out | 0 |
| | Units the pipeline flagged as unsupported | 3 (all equations) |
| | Required relationships found | 33% (1/3); type accepted for the one found; 1 of 2 judgeable stored relationships is expected by the reference |
| **C. Source accuracy** | Matched units citing the right passage | **100% (20/20)** |
| | Matched units with pages inside the passage's pages | 100% (20/20) |
| | Units whose sources verified / quotes found in their passage | 100% (35/35) / 100% (37/37) |
| | Units whose primary source is the passage that was extracted | **71.4% (25/35)** (finding 1) |
| **D. Structure** | Answers that matched their Pydantic schema, counting retries | **71.4% (15/21)** |
| | Extractions that succeeded on the first call | 64.7% (11/17) |
| | Matched units in the expected category | 85.0% (17/20) |
| **E. Duplication** | Redundant units among stored units | 17.1% (6/35); 10 more duplicates were refused by the pipeline |
| **F. Equations** | Found / original expression kept | 100% (5/5) / 100% (5/5) |
| | Phase 1 source text, image reference kept | 100% (2/2), 100% (2/2) |
| | Page of the source retained | **50% (2/4)** (finding 2) |
| | Capture status as expected / captured or appropriately flagged | **60% (3/5)** |
| **Operations** | Failed extractions / needing review | 0% (0/17) / 64.7% (11/17) |
| | Retries per extraction | 0.47 (8/17) |
| | Time per passage / total model time | 39.3 s / 196 s |
| | Tokens in / out | 86,735 / 25,439; cost `n/a` (no price configured) |

What the schema failures were. The 6 invalid knowledge answers (3 concept, 3 principle; the 29% above) carried 13 `invalid_value` issues (a field of the wrong type, 11 of them on principle answers) and 1 invalid JSON. Separately, 2 of the 4 relationship calls were invalid: 5 proposed relationships used a type combination the relationship matrix rejects (for example an *example* "example_of" a concept) and one had no endpoint in the passage. Every invalid answer was corrected on retry; none ended in a failed extraction.

## 4. Findings

Ordered by importance. "Fixed" means fixed on Day 5 with a regression test.

1. **Units are sometimes read from a neighbouring passage and cited as primary (open, now visible).** 10 of 35 stored units (29%) have a primary source that is not the passage being extracted, although the prompt says to extract only from the target. The citation is true (the cited passage holds the text, and 100% of sources verified), but the same knowledge is then extracted again when that passage is processed, and run accounting attributes it to the wrong passage. In one case ("Energy-guided model") a unit cites passage 13 as primary with a quote from it, while most of its content comes from passage 12, cited as supporting: verification checks citations and quotes, not that every claim is in the cited text. The validator only warns (`target_passage_not_cited`, 9 occurrences). *Day 5 change:* such a unit now raises a review flag `source_outside_target` (new migration `6a1c3d9e7f20`), and the evaluation reports the rate. Making it a correctable validation error changes the validation contract and was deferred. The 10 stored units pre-date the flag and are not flagged.
2. **Equation captures were bound to the wrong Phase 1 equation (fixed).** Three equations read from passage 15 and one from passage 13 were attached to whichever equation Phase 1 had found in the passage being *processed*: the capture of "Latent variable energy function" carried the text and page of the Gibbs equation on page 10. Two causes: the capture looked in the processed passage rather than the cited one, and the "only equation of the passage is the likely one" shortcut applied to any reading. Both are fixed (`tests/test_day5_equation_provenance.py`). **The affected rows already stored are not repaired** (they were created before the fix); re-extracting the equations would regenerate them.
3. **Legacy equation units have no capture record (open).** The Day 4 migration back-filled snapshots and attempt links but not `equation_sources`, so equation units extracted before Day 4 (1 here) have no capture status. Nothing is lost, but they show no status on the Knowledge screen.
4. **Over-extraction on structural text.** The chapter roadmap passage should hold no knowledge. Claude Haiku 5.5 produced three concept units for it (one with the content "EBMs"). The prompt tells the model to return nothing when there is nothing, but a thin unit is still accepted.
5. **Same idea in two categories.** "Label bias problem" is a concept *and* a principle; "negative log-likelihood training" appears under two titles. Duplicate detection is per passage and category by design, so these survive. They are counted as redundant here.
6. **Relationships are sparse and partly mistyped.** 1 of 3 required relationships found. Most stored relationships are `related_to` between near-synonymous units, and several attempts proposed type combinations that the relationship matrix rejects. All stored relationships carry the origin `extracted` with a quote, so none is flagged; they are better than no links but are not a knowledge graph.
7. **Equations are preserved, not understood, as designed.** Damaged equations (the fraction bar and integral lost in the PDF text) were kept with the original text and flagged: both Phase 1 equations were captured and flagged as incomplete / needs review, as the reference requires. The model's own reading was kept apart from the source text.
8. **Extractions are largely extractive.** Reading the units of passages 10-15 against the source, the definitions are mostly near-verbatim sentences of the text; none of the pairs read contradicted the source. That is faithful, but it is not synthesis.

## 5. What the numbers do not show

* They describe **one model on five passages**; a ratio such as 15/17 moves by six points with a single unit. No claim is made that Claude Haiku 5.5 is better or worse than any other model.
* A passing schema check was never treated as correctness. Semantic accuracy is **unavailable as a number**; the qualitative reading of the unit text against the passages (findings 4-8) is the only semantic assessment, was done by the same assistant that prepared the reference, and should be repeated by a human reviewer from the "Matched pairs for review" table.
* Method, framework, evidence and example extraction, and the synthetic dataset's equation cases (image-only, incomplete), have not been measured on a live model. The scripted-model tests show that the pipeline keeps an image-only equation as a source-derived `image_only` unit and flags a damaged one, but a real model's behaviour is unmeasured.

## 6. Harness validation

`tests/test_evaluation_metrics.py` pins every metric on small hand-built cases: perfect extraction, missing and merged units, categories never requested, name-only and short-name false matches, duplicates, wrong passage or pages, forbidden phrases, equation preservation (original, source, image, page, status, flagging, image-only), relationships (both directions, wrong type, unjudgeable), reliability counts, and malformed datasets.

`tests/test_evaluation_runner.py` runs the real pipeline over `study_guide_v1` with scripted models and checks that: a reference-following model scores 100% on recall, source, schema validity and equation handling; a degraded model scores lower exactly where it was degraded and the pipeline's own duplicate protection is counted; a model citing a neighbouring passage is caught by the source metrics; a provider outage is a 100% failure rate and `n/a` schema validity, not silently missing knowledge; two runs give identical figures on separate books with distinct run ids; the evaluation book is removed or kept; unknown models leave nothing behind; stored datasets are located by anchor and are read-only.
