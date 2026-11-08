"""Compare math OCR engines on isolated equation images with known ground truth.

    python -m tools.compare_math_engines --engine docling
    python -m tools.compare_math_engines --engine pix2text    # needs `pip install pix2text`

Each equation of tests/factories.MATH_CORPUS is rendered on its own (matplotlib mathtext, 300 dpi)
and passed straight to the engine through the same adapter the application uses. This isolates
recognition quality from layout detection. Pix2Text's dependencies conflict with Docling's, so
run it from a separate virtual environment (see the README).
"""

import argparse
import io
import time

from PIL import Image

from app.parsing.equation.equation_ocr import (
    DoclingFormulaEngine,
    MathOcrEngine,
    Pix2TextEngine,
)
from tests.factories import MATH_CORPUS
from tools.metrics import canonical, similarity


def render(latex: str) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 1.3))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=26, ha="center", va="center")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["docling", "pix2text"], required=True)
    args = parser.parse_args()
    engine: MathOcrEngine = DoclingFormulaEngine() if args.engine == "docling" else Pix2TextEngine()
    engine.check_available()

    exact, scores, seconds = 0, [], 0.0
    for category, truth in MATH_CORPUS:
        image = render(truth)
        started = time.monotonic()
        predicted = engine.recognize(image).latex
        seconds += time.monotonic() - started
        score = similarity(predicted, truth)
        scores.append(score)
        exact += canonical(predicted) == canonical(truth)
        flag = "  " if canonical(predicted) == canonical(truth) else "!!"
        print(f"{flag} {score:.2f} [{category}] truth={truth!r}\n          got  ={predicted!r}", flush=True)
    count = len(MATH_CORPUS)
    print(
        f"\n{args.engine}: exact {exact}/{count}, similar>=0.95 {sum(s >= 0.95 for s in scores)}/{count}, "
        f"mean similarity {sum(scores) / count:.3f}, {seconds / count:.1f} s per equation"
    )


if __name__ == "__main__":
    main()
