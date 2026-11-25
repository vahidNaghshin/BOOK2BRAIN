# Phase 2 test, integration and data-integrity results

Run on 2026-11-25: `pytest` → **1435 passed, 7 skipped, 0 failed** (1 m 49 s; 1442 collected). The 7 skips are the
`sample_book` structure tests, which need a book longer than the one in `storage/original` (unchanged, unrelated to Phase 2).
`alembic check` reports no difference between the models and the migrations. About 730 of the tests cover Phase 2
(knowledge models and schemas, extraction layer, pipeline, provenance, equations, relationships, recovery, screens, evaluation).

**No test calls a model or reads the application database.** Models are replaced by a scripted provider
(`tests/extraction_db_helpers.FunctionProvider`); a new autouse fixture makes any use of the application database fail the test.

## Baseline problems found and fixed on Day 5

| Problem | Fix |
|---|---|
| `tests/test_knowledge_extraction_service.py` imported a class (`KnowledgeExtractionRunner`) that no longer exists, so the whole suite could not be collected. It was an earlier draft of `test_knowledge_extraction_pipeline.py`, which covers every one of its 15 tests. | Removed (a copy is kept in the session folder, not in the repository). |
| `tests/test_zz_probe.py`, an untracked debugging probe that re-imported and re-ran the whole Knowledge UI test module. | Removed. |
| 8 UI tests (`test_structure_ui.py`, `test_library.py`) failed because they patched the database only on their own screen; the shared book selector still read the **development database** (they passed only while it was empty). | They now use the shared `patch_ui` helper; a fixture refuses any use of the application database. |

## Coverage of the required test areas

| Required area | Where it is tested |
|---|---|
| KnowledgeUnit types, KnowledgeRelationship, SourceReference, ExtractionRun, ExtractionAttempt, equation representation, model configuration metadata | `test_knowledge_schemas` (132), `test_knowledge_models` (59), `test_extraction_models`, `test_day4_models`, `test_extraction_config`, `test_extraction_attempts` |
| Provider-independent interface, model selection, prompt selection and versioning, structured parsing, Pydantic validation, provider errors and timeouts, unsupported capabilities | `test_extraction_service`, `test_extraction_providers`, `test_extraction_prompts_parsing` (50), `test_extraction_config`, `test_extraction_check` |
| Passage retrieval, context preparation, several units per passage, persistence, duplicate detection, source association, processing status | `test_passage_retrieval`, `test_context_preparation`, `test_knowledge_extraction_pipeline` (49), `test_knowledge_fingerprint` |
| Source validation, multiple sources, relationship creation and validation, invalid references, provenance retrieval | `test_provenance` (23), `test_relationships` (33), `test_relationship_extraction` (35), `test_knowledge_service` |
| Equations: complete, incomplete, ambiguous, image-only, context and pages | `test_equation_capture` (35), `test_day5_equation_provenance`, `test_extraction_service` |
| Retry limits, failed attempts, interruption, resumability, idempotency, duplicate prevention | `test_extraction_attempts`, `test_resumable_processing` (15), `test_knowledge_extraction_pipeline`, `test_day5_integration` |
| Complete book or chapter extraction, model metadata, retrieval and inspection, database integrity | `test_day5_integration`, `test_day5_integrity`, `test_knowledge_ui`, `test_extraction_ui` |
| Evaluation (new) | `test_evaluation_metrics` (32), `test_evaluation_runner` (17) |

## End-to-end integration (`tests/test_day5_integration.py`)

A DOCX book (three chapters, two real OMML equations) is uploaded and processed by the **real Phase 1 pipeline**:
parsing, equation extraction, structure detection and passage generation, producing three passages and two equations. Only the model is scripted. The test
then walks the required workflow: select the processed book; select a chapter; select an active model (the plan shows
1 passage, 2 extraction calls and 1 relationship call); start the run (status `pending`, then progress events for each category and the relationship step);
the model is told the book and chapter; the whole book is extracted; the result is read back through `KnowledgeService`: counts by category
(3 concepts, 2 equations), the unit's source passage text, chapter, and the unknown page status of a DOCX (no pages are invented); the
model-read equation is captured **complete** with its Phase 1 text, and the equation the model skipped is kept as a source-derived unit;
relationships exist; every execution and attempt carries the model configuration (no secret value), prompt name and version; passage records are completed or needs-review;
a second plan finds nothing to do and a second run is refused; **the Phase 1 passages, equations and structure are byte-for-byte unchanged**.

## Failure and recovery

| Scenario | Test | Verified |
|---|---|---|
| Provider unavailable or timing out | `test_a_failed_passage_is_visible_then_recovered...`, `test_a_provider_that_is_down...` (evaluation), `test_extraction_service` | Retried up to the limit (3 calls recorded for a limit of 2), each attempt kept with its error, the passage fails alone, the run is `needs_review`, the failure shows in the Knowledge overview and as an `unresolved_extraction_error` flag |
| Malformed structured response / Pydantic failure / invalid source reference | `test_an_interrupted_and_partly_failed_run...`, `test_extraction_attempts`, `test_extraction_service` | Correction prompt, `schema_invalid` / `source_invalid` outcomes recorded, validation failures counted, raw response kept |
| Retry limit exceeded | `test_extraction_attempts`, `test_day5_integration` | Failed execution with every attempt and the last raw response |
| Application interrupted mid-run | `test_an_interrupted_and_partly_failed_run...` (`KeyboardInterrupt` during the second passage) | Run `cancelled` with `interrupted`, the finished passage keeps its unit, resume repeats only the interrupted and unreached passages |
| Resume | same, `test_resumable_processing` | No duplicate units (titles are exactly Concept 0..2), counters correct (validation failures and retries counted), a finished run cannot be resumed again |
| Reprocessing on request | `test_reprocessing_a_finished_passage...` | Refused without `force`; with `force` the model is asked again and the repeated answers are recognised as the stored units (3 duplicates, still 3 units) |
| Relationships and sources after resume | `test_resumable_processing`, `test_relationship_extraction` | No second relationship, sources unchanged |
| Unresolved failures visible | `test_knowledge_ui` (overview) | Failed extractions (page, category, reason, model, prompt) and open flags are shown, expanded when something failed |

## Data integrity

* **Foreign keys.** `tests/test_day5_integrity.py` pins the ON DELETE rule of all 36 foreign keys in the 11 Phase 2 tables against the real migrated schema, and checks that none can block a delete. Rows meaningful only with their parent cascade (units' sources, attempts of an execution, flags of a unit); references to things that can be regenerated (passages, structure, runs, executions) are cleared and never delete knowledge.
* **Delete behaviour.** Deleting a document deletes its units, sources, relationships, equation sources and flags; a book-wide run and its executions stay as the book's history, with passage references cleared, until the book is deleted, after which no Phase 2 row remains. Deleting a run keeps the knowledge and clears its run reference.
* **Constraints** (unit, source, relationship, run, attempt, flag, equation source values; uniqueness of primary source, relationship edge, passage per run, open flag per cause, attempt number): `test_knowledge_models`, `test_day4_models`, `test_extraction_models`.
* **Transactions.** A storage failure inside one extraction is recorded as a failed execution and the run continues; nothing from the failed attempt remains (`test_knowledge_extraction_pipeline`, `test_extraction_attempts`).
* **Snapshots and versions.** Model configuration snapshots do not change when the configuration changes, no credentials are stored, and every attempt records the prompt name and version actually used (`test_extraction_attempts`, `test_knowledge_extraction_pipeline`, `test_day5_integration`).

## Migrations

| Check | Result |
|---|---|
| Clean database: upgrade to head, models equal migrations, downgrade removes tables, upgrade again, Phase 2 migrations keep and roll back existing records | `test_migrations` (6), `test_day4_migration` (3): pass |
| New migration `6a1c3d9e7f20` (adds the `source_outside_target` review-flag code) | Applied by the clean-database tests, by the check below, and to the development database |
| **Existing development database** (`python -m tools.check_migrations_on_copy`; a copy is made and checked, the original is only read) | Copy of the 839-row database at `1eb6b9dd22a0`: `upgrade head` changes no data. Rolled back to `b9221d0781db` (the pre-Day 4 schema) and upgraded again: **Phase 1 tables byte-for-byte identical, no row or id lost by the upgrade**, `alembic check` clean. Rolling back is destructive by design: it removed 4 attempts, 2 executions and 1 relationship that exist only in the newer schema. The upgrade rebuilt 7 passage records and, as before, back-fills no equation sources or flags. Also tested from `1eb6b9dd22a0`: pass. |
| Development database after `alembic upgrade head` | Revision `6a1c3d9e7f20`; 35 units and 16 flags unchanged |

## Screens (basic inspection)

All required inspection points exist and are tested (`test_extraction_ui`, `test_knowledge_ui`, `test_day4_ui`); no business logic is in the screens (they call `KnowledgeExtractionService`, `KnowledgeService`, `ProvenanceService`, `RelationshipService`, `ReviewFlagService`).

| Required | Screen |
|---|---|
| Select a book or chapter (or section) | Extraction |
| Select a model configuration | Extraction (only usable models; unusable ones listed with the reason) |
| Start a run, see progress and status; resume, cancel | Extraction |
| Units by category | **Knowledge → Overview** (new) and the category filter |
| Inspect a unit; sources and pages; relationships; equation status; model and prompt metadata; failed attempts; review flags | Knowledge (unit detail; *How it was extracted*; flags with Resolve) |
| Failed extractions and open review flags at a glance | **Knowledge → Overview** (new) |
