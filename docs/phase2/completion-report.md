# Phase 2 completion report: knowledge extraction

Date: 2026-11-25 · Test suite: 1435 passed, 7 skipped (unrelated), 0 failed · Details:
[evaluation](evaluation-report.md), [model comparison](model-comparison.md), [tests and integrity](test-and-integrity-results.md), [performance](performance.md).

## Verdict

**Phase 2 can be considered complete as an engineering deliverable, with three open conditions that cannot be closed from this environment.** The pipeline works end to end on a book processed by Phase 1,
every recovery scenario that was required is tested, provenance is recorded and verified, migrations are safe on the
development database, and extraction results are inspectable. It should **not** be described as quality-proven: quality has been measured for one model on five
passages, against a reference that no human has signed off yet.

| Open condition | Why it is open | How to close it |
|---|---|---|
| 1. Human review of the two reference datasets (`evaluation/datasets/*.json`, status `draft`) | They were prepared by the AI assistant; a person must read them against the passages | Read them, correct them, set `review.status` to `reviewed` |
| 2. Live model comparison | No model was usable on the day (no `STORMY_API_TOKEN`, no Ollama, no other key) | `python -m tools.evaluate_extraction --dataset study_guide_v1 --model A --model B --types all --relationships --repeat 3 --yes` ([model-comparison.md](model-comparison.md)) |
| 3. Decide on the primary-source defect (finding 1 below) | A design decision about the validation contract | Choose strict validation (recommended) or accept the flag-only behaviour |

## What was implemented and verified

### Phase 2, Days 1-4 (verified, not re-implemented)

Knowledge data model and schemas for seven categories; configurable providers (Ollama, OpenAI, Anthropic, Stormy, any OpenAI-compatible server) with capability-aware generation parameters;
versioned prompts; parsing and layered validation (structural, source, semantic, quality); per-passage extraction with context preparation and context-window fitting; duplicate detection; multiple
source references with verification; equation source preservation and capture status; attempts with model/configuration/prompt snapshots; review flags; relationship extraction; resumable runs; the Extraction and Knowledge screens.

### Day 5 additions

| Deliverable | What it is |
|---|---|
| Evaluation method and tooling | `app/evaluation/` (reference-dataset schema, pure metrics, runner, reports) and `python -m tools.evaluate_extraction` (live runs, model comparison, repeats, stored-extraction evaluation, Markdown + JSON reports with the model, parameters and prompt versions of every run) |
| Two manually reviewed reference datasets | `study_guide_v1` (12 invented passages, all 7 categories, relationships, complete / incomplete / image-only equations, a rejected claim, an empty passage) and `latent_ebm_v1` (6 passages of the real book) |
| Quality evaluation of the real book | [Evaluation report](evaluation-report.md) |
| Inspection | Knowledge screen *Overview*: units by category, equations by capture status, relationships by type, open flags, call outcomes, failed extractions |
| Tests | +71 tests (1364 → 1435); baseline repaired (collection error, 8 tests that read the development database, a debug probe); a guard against any test touching the application database |
| Defects fixed | Equation captures were attached to the wrong Phase 1 equation (two causes); units that cite a neighbouring passage as primary are now flagged (`source_outside_target`, migration `6a1c3d9e7f20`) |
| Data safety | `python -m tools.check_migrations_on_copy` (verified on a copy of the development database); applied to the development database: 35 units, 16 flags unchanged |
| Performance | `python -m tools.benchmark_extraction`: ~65 ms per passage of pipeline overhead against ~39 s of model time; linear to 300 passages |
| Documentation | README (provider, model, run, inspect, test, evaluate), this folder |

## Quality findings (summary)

Claude Haiku 5.5 on passages 10-15 of the real book ([details](evaluation-report.md)):

* **Strong:** every matched unit cites the right passage and pages (20/20), all sources and quotes verify, required concepts 9/9, no failed extractions, both damaged equations preserved with their original text and flagged, no phrase the reviewer ruled out.
* **Weaker:** required principles 5/7; 29% of knowledge answers (6 of 21 calls) failed their schema (all corrected on retry; 0.47 retries per extraction); 17% of stored units are redundant (mostly the same idea as a concept and a principle); a passage that should yield nothing yielded three units; 1 of 3 required relationships found.
* **Defects:** 29% of units cite a neighbouring passage as primary (open, flagged from now on); equation captures were mis-bound (fixed for new runs, 4 stored rows not repaired); pre-Day 4 equation units have no capture record.

The comparison across models is **outstanding**.

## Known limitations and recommendations

| # | Limitation | Recommendation for the next phase |
|---|---|---|
| 1 | A model can cite a neighbouring passage as primary and mix its content into a unit; only a warning and the new flag exist. Quote verification checks the quote, not that every claim is in the cited text. | Make "primary source must be the target passage" a correctable validation error for prompt versions that state the rule; consider a claim-level check (a model-free overlap test between the unit's text and the cited passages) |
| 2 | Four already-stored equation captures are mis-bound, ten units are unflagged, and pre-Day 4 equation units have no capture row | A small repair command: recompute equation captures and `source_outside_target` flags for existing units |
| 3 | Evaluation matching is lexical; semantic accuracy is a human judgement; the reference sets are small drafts | Human sign-off; add passages for method, framework, evidence and example from a real book; use a second human reviewer for disagreements |
| 4 | Only one model has output; no cost figures (no prices configured in `config/llm_models.json`) | Run the comparison; add prices; compare at several temperatures if models allow it |
| 5 | Duplicate detection is per passage and category: the same idea as concept and principle, or in overlapping passages, is stored twice | A later, optional merge or "same as" link step across categories and passages (a reference-guided design, not automatic merging) |
| 6 | Relationships are few and often `related_to`; 2 of the 4 relationship calls failed validation on type combinations | Give the relationship prompt the allowed type matrix; extract relationships over chapters rather than within one passage |
| 7 | Over-extraction on structural text (roadmaps, tables of contents) | Skip passages classified as front matter / roadmap in Phase 1, or require a minimum content in a unit |
| 8 | No semantic validator beyond quotes and equation text; no confidence calibration | Keep confidence as an indicator only; evaluate calibration once several models are compared |
| 9 | Runs execute on one background thread; no rate-limit-aware scheduling for hosted providers | Not needed at today's scale (see [performance](performance.md)); revisit with books of thousands of passages or parallel models |
| 10 | The Streamlit screens are for inspection, not review: no assignment, history of decisions or bulk actions | A review workflow only if the knowledge is used as ground truth for downstream AI |

## Deferred, by design (not Phase 2)

Embeddings and semantic search; graph database; chat and retrieval; equation reconstruction and equation-to-Python conversion (the
`equation_reconstructions` table is an unused placeholder); human review workflow; LangChain or agent orchestration; distributed processing; automatic merging of duplicate knowledge.

## Reproducing everything

```bash
docker compose up -d && alembic upgrade head
pytest                                                           # 1435 passed
python -m tools.evaluate_extraction --dataset latent_ebm_v1 --stored   # the real-book evaluation (read only)
python -m tools.check_migrations_on_copy                         # migrations on a copy of your database
python -m tools.benchmark_extraction --passages 300 --chapters 20 --drop
```
