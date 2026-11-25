# Phase 2 model comparison

**Status: not performed. Comparative evaluation is outstanding.** This document records why, what is ready, and
exactly how to run it. No comparative claim is made anywhere in the Phase 2 reports.

## Why it could not be done

`python -m app.extraction.check` on 2026-11-25 reported:

| Model configuration | Provider | State |
|---|---|---|
| `ollama-llama3.1-8b` | ollama (local) | "ready" by configuration, but **no Ollama server is running or installed** on this machine, so it cannot be called |
| `openai-gpt-4o-mini` | openai | no `OPENAI_API_KEY` |
| `stormy-claude-sonnet-5.5`, `-haiku-5.5`, `-opus-5.5`, `stormy-llama-v3.3-70b`, `stormy-gpt-6-luna`, `stormy-pixtral-large`, `stormy-palmyra-x5-v1` | stormy (hosted) | no `STORMY_API_TOKEN` in the environment or `.env` |
| `anthropic-claude`, `stormy-stable-diffusion-v3.5L` | | inactive |

So *zero* configurations were usable on the day. Output from exactly one model, `stormy-claude-haiku-5.5`, already
exists in the database from earlier days, and it is evaluated in the [evaluation report](evaluation-report.md). The
four stored runs used two prompt versions (`1.1.0`, `1.2.0`) on different passages, so even those runs are not a
controlled comparison of anything.

Sending the evaluation passages to a hosted provider without a configured credential was not attempted, and none was
invented. The synthetic dataset contains no book text, so running it on a hosted model discloses no third-party content.

## What is ready

* **Same inputs for every model.** `study_guide_v1` carries its passages; each model is run on an identical fresh copy
  (same passages, same scope, same categories). Prompt versions default to the latest and can be pinned for all models with
  `EXTRACTION_PROMPT_VERSIONS`. The output schemas are the Pydantic content models shared by all providers. The evaluation criteria are
  the same reference and the same counting rules.
* **Actual configuration is recorded** per run: model configuration id and snapshot (no secrets), provider, the model id the
  provider reported, temperature / max output tokens / structured-output mode actually sent, prompt names and versions, context
  handling, run id, start and end times.
* **Compared in one table**: schema validation success, recall, precision and key-fact coverage, source accuracy, unsupported
  content (forbidden phrases, flags), duplicates, equation preservation, processing time per passage, token usage and cost
  (when the provider reports tokens and a price is configured; otherwise `n/a`), retries and failures. The report states that no column is
  ranked: a larger or more expensive model is not assumed better.
* **Verified without a model.** `tests/test_evaluation_runner.py::test_a_weaker_model_scores_lower_where_it_is_weaker_and_the_comparison_shows_both`
  runs two scripted models through the real pipeline and checks that the comparison table separates them where they differ. That checks the
  report, not any real model.

## How to run it

```bash
# 1. make two or more models usable (see README: Configuring the LLM provider)
echo 'STORMY_API_TOKEN=...' >> .env              # git-ignored; or run a local model with Ollama
python -m app.extraction.check --all             # which models are ready

# 2. same passages, every model, repeated to see variation between runs
python -m tools.evaluate_extraction --dataset study_guide_v1 \
    --model stormy-claude-haiku-5.5 --model stormy-claude-sonnet-5.5 --model stormy-llama-v3.3-70b \
    --types all --relationships --repeat 3 --yes
```

The command prints the plan (calls, models) first, asks before sending to a hosted model unless `--yes`, and writes
`evaluation/results/study_guide_v1-<time>.md` and `.json` with the comparison table at the top.

## What to look at when it has been run

* Read the **matched pairs** and **unmatched extractions** of each model before the percentages: matching is lexical.
* The synthetic set is small (23 reference units); a difference of one unit is a few points. Use `--repeat` and prefer differences that
  appear in every repeat and in several categories.
* Compare **equation handling** on the incomplete and image-only passages (does the model invent an expression, or leave the unit to be kept from the source?)
  and the **rejected claim** passage (does it extract the myth as fact?): those are the cases where schema validity says least.
* Compare **cost and time together with quality**; the report gives tokens and, if prices are in `config/llm_models.json`, an estimate.
