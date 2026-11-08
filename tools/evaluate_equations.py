"""Measure equation detection and recognition quality against known ground truth.

    python -m tools.evaluate_equations                 # all scenarios
    python -m tools.evaluate_equations --scenario scanned-enrichment

Equations are rendered into PDF pages from known LaTeX (tests/factories.MATH_CORPUS), then
extracted with the real pipeline (real Docling, real models). Nothing is mocked, so the first
run downloads Docling's models and takes a few minutes.

Metrics (per scenario):
  regions     equation regions kept (ground truth has 20)
  exact       prediction equals the ground truth after canonicalisation (spaces and
              \\left/\\right removed). Strict: equivalent but differently written LaTeX counts as a miss.
  similar     token-level similarity >= 0.95 (difflib ratio over LaTeX tokens)
  mean sim    average token similarity
"""

import argparse
import tempfile
import time
import uuid
from pathlib import Path

from app.config import Settings
from app.parsing.equation.equation_service import EquationService
from app.parsing.parser_factory import ParserFactory
from tests.factories import MATH_CORPUS, make_math_pdf
from tools.metrics import canonical, similarity
SCENARIOS = {
    "digital-enrichment": dict(scanned=False, enrichment=True),
    "scanned-enrichment": dict(scanned=True, enrichment=True),
    "digital-mathocr": dict(scanned=False, enrichment=False),
    "scanned-mathocr": dict(scanned=True, enrichment=False),
}


def run(name: str, scanned: bool, enrichment: bool) -> dict:
    truths = [latex for _, latex in MATH_CORPUS]
    settings = Settings(
        database_url="postgresql+psycopg://u:p@h/d",
        storage_root=Path(tempfile.mkdtemp()),
        formula_enrichment_enabled=enrichment,
        math_ocr_enabled=True,
        _env_file=None,
    )
    path = Path(tempfile.mkdtemp()) / "math.pdf"
    path.write_bytes(make_math_pdf(truths, scanned=scanned))
    document_id = uuid.uuid4()

    started = time.monotonic()
    parsing = ParserFactory(settings).get_parser("pdf").parse(path, document_id=document_id, original_filename="math.pdf")
    outcome = EquationService(settings).extract(path=path, file_format="pdf", document_id=document_id, parsing=parsing)
    seconds = time.monotonic() - started
    found = outcome.result.equations

    # Each truth is paired with its most similar prediction (detection can miss or add regions,
    # so pairing by position would misalign everything after the first miss).
    scores, matches = [], []
    for truth in truths:
        best = max(found, key=lambda e: similarity(e.latex, truth), default=None)
        score = similarity(best.latex, truth) if best else 0.0
        scores.append(score)
        matches.append(best.latex if best else "")
    exact = sum(canonical(m) == canonical(t) for m, t in zip(matches, truths))
    statuses: dict[str, int] = {}
    for equation in found:
        statuses[equation.validation_status.value] = statuses.get(equation.validation_status.value, 0) + 1
    result = {
        "scenario": name,
        "regions found": len(found),
        "ground truth": len(truths),
        "exact": f"{exact}/{len(truths)}",
        "similar>=0.95": f"{sum(s >= 0.95 for s in scores)}/{len(truths)}",
        "mean sim": round(sum(scores) / len(truths), 3),
        "validation": statuses,
        "status": outcome.result.status.value,
        "seconds": round(seconds, 1),
    }
    print(result, flush=True)
    for match, truth, score in zip(matches, truths, scores):
        flag = "  " if canonical(match) == canonical(truth) else "!!"
        print(f"  {flag} {score:.2f} truth={truth!r}\n          got  ={match!r}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    args = parser.parse_args()
    for name in args.scenario or SCENARIOS:
        print(f"\n=== {name}", flush=True)
        run(name, **SCENARIOS[name])


if __name__ == "__main__":
    main()
