from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, read from environment variables and the project .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: SecretStr
    storage_root: Path = PROJECT_ROOT / "storage"
    max_upload_size_mb: int = Field(default=100, ge=1)
    log_level: str = "INFO"

    # Document processing
    min_content_characters: int = Field(default=20, ge=1)
    processing_stale_minutes: int = Field(default=120, ge=1)

    # OCR (Tesseract) and scanned-page detection
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    ocr_min_text_chars: int = Field(default=25, ge=0)
    ocr_max_garbled_ratio: float = Field(default=0.3, ge=0, le=1)
    ocr_min_confidence: float = Field(default=30.0, ge=0, le=100)
    ocr_page_timeout_seconds: int = Field(default=120, ge=1)
    ocr_dehyphenate: bool = True

    # Equation extraction
    equation_extraction_enabled: bool = True
    # Docling's formula model (LaTeX from the page image). Slow, so it only runs on pages
    # that contain formulas. cpu/float32 was ~13x faster than Docling's default on test hardware.
    formula_enrichment_enabled: bool = True
    formula_enrichment_device: str = "cpu"
    formula_enrichment_dtype: str = "float32"
    # Fallback for formula regions the primary route could not recover (and for EPUB images).
    math_ocr_enabled: bool = True
    math_ocr_engine: Literal["docling", "pix2text"] = "docling"
    # Recognition confidence (0-1) below which a result is flagged "uncertain".
    equation_min_confidence: float = Field(default=0.70, ge=0, le=1)
    equation_store_crops: bool = True
    equation_crop_dpi: int = Field(default=200, ge=72, le=600)
    # Try math OCR on small, wide picture regions (layout analysis often calls equations that).
    equation_scan_pictures: bool = True
    equation_scan_pictures_max: int = Field(default=300, ge=0)

    # Structure detection (chapters, sections, front/back matter)
    # A candidate heading scoring at least `accept` is a confident heading; between `reject`
    # and `accept` it is kept but flagged ambiguous (and sent to the optional LLM); below
    # `reject` it stays body text.
    structure_heading_accept_threshold: float = Field(default=0.70, ge=0, le=1)
    structure_heading_reject_threshold: float = Field(default=0.45, ge=0, le=1)
    # Detected subtitle/author are only copied into *empty* book fields, and only at or above
    # this confidence. Metadata entered by the user is never overwritten.
    structure_fill_empty_book_metadata: bool = False
    structure_metadata_min_confidence: float = Field(default=0.8, ge=0, le=1)

    # Optional LLM help for ambiguous headings (any OpenAI-compatible chat endpoint).
    structure_llm_enabled: bool = False
    structure_llm_base_url: str = "https://api.openai.com/v1"
    structure_llm_model: str = ""
    structure_llm_api_key: SecretStr | None = None
    structure_llm_timeout_seconds: int = Field(default=60, ge=1)
    structure_llm_max_candidates: int = Field(default=60, ge=1)
    structure_llm_batch_size: int = Field(default=15, ge=1)

    # Passage segmentation (independent of structure detection; can be regenerated at any time)
    passage_target_tokens: int = Field(default=750, ge=1)
    passage_min_tokens: int = Field(default=500, ge=1)
    passage_max_tokens: int = Field(default=1000, ge=1)
    # Share of the target size repeated at the start of the next passage (default 12%; the
    # largest overlap ever taken is 25% above this, i.e. 15%).
    passage_overlap_ratio: float = Field(default=0.12, ge=0, le=0.4)
    # approx (characters / chars-per-token), words, whitespace, or tiktoken (if installed).
    passage_tokenizer: Literal["approx", "words", "whitespace", "tiktoken"] = "approx"
    passage_chars_per_token: float = Field(default=4.0, gt=0)
    tiktoken_encoding: str = "cl100k_base"
    # A table larger than this many times the maximum passage size is split between rows.
    passage_table_split_factor: float = Field(default=2.0, ge=1)

    # Knowledge extraction (LLM). Model definitions live in a JSON file; API keys live only in the
    # environment variables those definitions name (never in the file or the database).
    extraction_models_file: Path = PROJECT_ROOT / "config" / "llm_models.json"
    extraction_model_id: str | None = None
    # Different model per knowledge category, e.g. {"equation": "local-ollama"}.
    extraction_model_overrides: dict[str, str] = Field(default_factory=dict)
    extraction_prompt_versions: dict[str, str] = Field(default_factory=dict)
    extraction_max_retries: int = Field(default=2, ge=0, le=10)
    extraction_retry_backoff_seconds: float = Field(default=1.0, ge=0)
    extraction_max_backoff_seconds: float = Field(default=30.0, ge=0)
    extraction_raw_response_max_chars: int = Field(default=20_000, ge=1_000)
    extraction_context_passages: int = Field(default=1, ge=0, le=10)
    extraction_context_safety_margin: float = Field(default=0.1, ge=0, le=0.5)
    extraction_max_consecutive_failures: int = Field(default=5, ge=0)
    extraction_low_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    extraction_relationship_context_units: int = Field(default=10, ge=0, le=50)

    @model_validator(mode="after")
    def _check_structure_settings(self) -> "Settings":
        if not self.passage_min_tokens <= self.passage_target_tokens <= self.passage_max_tokens:
            raise ValueError(
                "Passage sizes must satisfy PASSAGE_MIN_TOKENS <= PASSAGE_TARGET_TOKENS "
                "<= PASSAGE_MAX_TOKENS"
            )
        if self.structure_heading_reject_threshold > self.structure_heading_accept_threshold:
            raise ValueError(
                "STRUCTURE_HEADING_REJECT_THRESHOLD must not exceed STRUCTURE_HEADING_ACCEPT_THRESHOLD"
            )
        return self

    @field_validator("storage_root")
    @classmethod
    def _resolve_storage_root(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
