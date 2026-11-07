from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
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
