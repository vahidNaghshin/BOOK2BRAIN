# Phase 2 performance and reliability checks

Measured with `python -m tools.benchmark_extraction` on 2026-11-25 (macOS, local PostgreSQL in Docker, Python 3.12).
A scripted provider answers instantly, so these figures are **the pipeline's own cost** (context preparation,
validation, persistence, provenance, equation capture, review flags, run bookkeeping). A real model's latency
adds to it. The benchmark uses its own `_benchmark` database and removes the book it builds.

Setup: concept + equation extraction plus a relationship call per passage (3 model calls per passage), one Phase 1
equation in every fifth passage.

| Size | Total | Per passage | Per model call | Queries | Queries per passage | Peak Python memory | Process max RSS |
|---|---|---|---|---|---|---|---|
| 60 passages (6 chapters) | 3.5 s | 58 ms | 19 ms | 2,581 | 43 | 3.0 MB | 103 MB |
| 300 passages (20 chapters) | 19.6 s | 66 ms | 22 ms | 12,819 | 43 | 5.8 MB | 107 MB |

* **Scales linearly.** 5x the passages gave 1.1x the time per passage and the same queries per passage. Python memory stays in single-digit megabytes because each model call is committed on its own.
* **The model dominates.** The stored Claude Haiku 5.5 runs took **39 s per passage** (about 8 s per call); the pipeline adds about **65 ms per passage**, under 0.2% of that. There is no bottleneck worth optimising now, and no reason for distributed processing.
* **Reading the results back is fast** at 300 passages: summary 4-6 ms, a page of 15 units 2-3 ms, the new overview 6-12 ms, a search across all units 5 ms, a unit's full detail about 4 ms (10 queries), resume check 1 ms, an "anything left to do?" plan 2-7 ms.
* **Retries and tokens** are recorded per run, execution and attempt (see the [evaluation report](evaluation-report.md): 25 calls for 17 extractions, 86,735 tokens in and 25,439 out for the stored Haiku runs).

## Observations (not acted on)

| Observation | Cost today | Possible later improvement |
|---|---|---|
| The run's cancel check re-reads `extraction_runs.status` before every extraction and passage: 4 queries per passage (1,200 at 300 passages) | ~0.5 ms each | Check every few seconds instead of every call |
| Passages are read repeatedly (most likely neighbouring passages for context, once per category): about 4.4 `passages` SELECTs per passage, each loading the full text | ~0.4 ms each | Cache a sliding window of passages for the run |
| A `knowledge_units` SELECT that runs 3.2 times per passage (most likely the duplicate check) doubled in cost, 0.7 ms to 1.4 ms, between 60 and 300 passages | grows with the number of units in the book | Confirm with a query plan; restrict it to the passage's own units (the rule only compares units of the same primary passage) |
| Each model call writes an execution and an attempt row plus the unit rows in separate commits | 0.9 ms per insert | None needed; this is what makes runs resumable |

## Limits of this check

Memory is the interpreter's allocations (`tracemalloc`) plus the process peak; it excludes PostgreSQL. The benchmark ran against a local database with no network latency, and with an instant model, so it
says nothing about hosted providers' rate limits. It was not run against books larger than 300 passages.
