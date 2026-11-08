"""Mathematical OCR: image of a formula -> LaTeX.

Ordinary text OCR (Tesseract) cannot read mathematics, so formula recognition is a separate,
replaceable adapter. Engines are used only for regions the primary route (Docling formula
enrichment, or the document's own markup) could not recover.

Engines:
  docling   Docling's CodeFormula model. No extra dependencies (the model is already used for
            formula enrichment). Gives no confidence score.
  pix2text  Optional. Its dependencies conflict with Docling's (see README), so it is meant for
            an environment where it is installed deliberately.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


class MathOcrUnavailableError(Exception):
    """The engine cannot be used at all (missing package or model). The message is actionable."""


class MathOcrError(Exception):
    """Recognition failed for one image; other images can still be processed."""


@dataclass(frozen=True)
class MathOcrResult:
    latex: str
    confidence: float | None = None  # 0-1, only when the engine reports one


class MathOcrEngine(ABC):
    name: ClassVar[str]

    @abstractmethod
    def check_available(self) -> None:
        """Raise MathOcrUnavailableError, with an actionable message, if the engine can't run."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> MathOcrResult:
        """Recognise one formula image. Raise MathOcrError if it fails."""


class DoclingFormulaEngine(MathOcrEngine):
    """Docling's CodeFormula model applied directly to an image.

    This uses Docling's internal VLM runtime (the same code path as formula enrichment), so it
    is pinned to the installed major version and guarded: if the API is not what we expect, the
    engine reports itself unavailable instead of crashing.
    """

    name = "docling"
    _STRIP_TOKENS = ("</formula>", "<loc_0><loc_0><loc_500><loc_500>")

    def __init__(self, device: str = "cpu", dtype: str = "float32") -> None:
        self.device = device
        self.dtype = dtype
        self._engine: Any = None
        self._input_cls: Any = None
        self._stopper: Any = None

    def _load(self) -> None:
        if self._engine is not None:
            return
        try:
            from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
            from docling.models.inference_engines.vlm import VlmEngineInput, create_vlm_engine
            from docling.models.utils.generation_utils import TailRepetitionStopper

            options = PdfPipelineOptions(do_formula_enrichment=True).code_formula_options
            for config in options.model_spec.engine_overrides.values():
                config.extra_config["torch_dtype"] = self.dtype
            self._engine = create_vlm_engine(
                options=options.engine_options,
                model_spec=options.model_spec,
                accelerator_options=AcceleratorOptions(device=self.device),
                artifacts_path=None,
                enable_remote_services=False,
            )
            self._input_cls, self._stopper = VlmEngineInput, TailRepetitionStopper()
        except Exception as exc:  # missing model, no network on first use, API drift...
            self._engine = None
            raise MathOcrUnavailableError(
                "Docling's formula model could not be loaded "
                f"({type(exc).__name__}: {str(exc)[:200]}). It is downloaded on first use "
                "(about 600 MB): check your internet connection, or see the README section "
                "'Docling models' if a proxy blocks it."
            ) from exc

    def check_available(self) -> None:
        self._load()

    def recognize(self, image: Image.Image) -> MathOcrResult:
        self._load()
        try:
            outputs = self._engine.predict_batch(
                [
                    self._input_cls(
                        image=image.convert("RGB"),
                        prompt="<formula>",
                        temperature=0.0,
                        max_new_tokens=2048,
                        extra_generation_config={
                            "skip_special_tokens": False,
                            "custom_stopping_criteria": [self._stopper],
                        },
                    )
                ]
            )
            text = outputs[0].text
        except Exception as exc:
            raise MathOcrError(f"Formula recognition failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        return MathOcrResult(latex=self._clean(text))

    def _clean(self, text: str) -> str:
        end = text.find("<end_of_utterance")
        if end != -1:
            text = text[:end]
        for token in self._STRIP_TOKENS:
            text = text.replace(token, "")
        text = re.sub(r"<[^>]*>", "", text)  # any stray special token
        return self._stopper.strip(text.lstrip()).strip()


class Pix2TextEngine(MathOcrEngine):
    """Optional adapter for Pix2Text (https://github.com/breezedeus/pix2text)."""

    name = "pix2text"

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from pix2text import Pix2Text
        except ModuleNotFoundError as exc:
            if exc.name != "pix2text":
                raise MathOcrUnavailableError(
                    f"Pix2Text is installed but a dependency is missing or incompatible ({exc}). "
                    "See the README section 'Optional: Pix2Text'."
                ) from exc
            raise MathOcrUnavailableError(
                "Pix2Text is not installed. It is optional and its dependencies conflict with "
                "Docling's pinned stack, so install it only in a dedicated environment "
                "(`pip install pix2text`) or set MATH_OCR_ENGINE=docling."
            ) from exc
        except ImportError as exc:
            raise MathOcrUnavailableError(
                f"Pix2Text is installed but its dependencies are incompatible ({exc}). "
                "See the README section 'Optional: Pix2Text'."
            ) from exc
        try:
            self._model = Pix2Text.from_config()
        except Exception as exc:
            raise MathOcrUnavailableError(
                f"Pix2Text could not start ({type(exc).__name__}: {str(exc)[:200]}). "
                "Its models download on first use; check your internet connection."
            ) from exc

    def check_available(self) -> None:
        self._load()

    def recognize(self, image: Image.Image) -> MathOcrResult:
        self._load()
        try:
            result = self._model.recognize_formula(image.convert("RGB"), return_text=False)
        except Exception as exc:
            raise MathOcrError(f"Pix2Text failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        if isinstance(result, dict):
            latex, score = result.get("text", ""), result.get("score")
        elif isinstance(result, (list, tuple)) and result:
            first = result[0]
            latex = first.get("text", "") if isinstance(first, dict) else str(first)
            score = first.get("score") if isinstance(first, dict) else None
        else:
            latex, score = str(result or ""), None
        confidence = float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None
        return MathOcrResult(latex=str(latex).strip(), confidence=confidence)


class MathOcrService:
    """Applies one configured engine. Checks availability once and remembers the answer."""

    def __init__(self, engine: MathOcrEngine) -> None:
        self.engine = engine
        self._unavailable: str | None = None
        self._checked = False

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def ensure_available(self) -> None:
        if self._checked:
            if self._unavailable:
                raise MathOcrUnavailableError(self._unavailable)
            return
        self._checked = True
        try:
            self.engine.check_available()
        except MathOcrUnavailableError as exc:
            self._unavailable = str(exc)
            raise

    def recognize(self, image: Image.Image) -> MathOcrResult:
        self.ensure_available()
        return self.engine.recognize(image)


def build_math_ocr_service(settings: Settings) -> MathOcrService | None:
    """The configured math OCR service, or None when math OCR is disabled."""
    if not settings.math_ocr_enabled:
        return None
    if settings.math_ocr_engine == "pix2text":
        return MathOcrService(Pix2TextEngine())
    return MathOcrService(
        DoclingFormulaEngine(settings.formula_enrichment_device, settings.formula_enrichment_dtype)
    )
