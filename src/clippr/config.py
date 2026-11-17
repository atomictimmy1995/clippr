"""Runtime settings, read from environment variables (prefix ``CLIPPR_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
FONTS_DIR = PACKAGE_DIR / "assets" / "fonts"

SelectorName = Literal["auto", "claude", "local", "rules"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPPR_", env_file=".env", extra="ignore")

    # Storage
    data_dir: Path = Path("data")
    retention_days: int = 7

    # Download
    max_video_min: int = 180
    max_height: int = 1080
    ytdlp_cookies: Path | None = None

    # Transcription (faster-whisper). "auto" picks large-v3-turbo on CUDA, small on CPU.
    whisper_model: str = "auto"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str = "auto"
    language: str | None = None

    # Moment selection. "auto": Claude if an API key is set, else a local Ollama model if
    # one is running, else the free rules-based picker.
    selector: SelectorName = "auto"
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 16384
    ollama_timeout_s: float = 900

    # Clips
    clip_min_s: float = 30
    clip_max_s: float = 60
    max_clips: int = 5
    output_width: int = 1080
    output_height: int = 1920
    fonts_dir: Path = FONTS_DIR
    caption_font: str = "Inter ExtraBold"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clippr.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
