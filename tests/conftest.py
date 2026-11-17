from __future__ import annotations

from pathlib import Path

import pytest

from clippr.config import Settings
from clippr.schemas import Transcript, load_json

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", selector="rules", anthropic_api_key=None,
                    ollama_url="http://127.0.0.1:9", _env_file=None)


@pytest.fixture
def transcript() -> Transcript:
    return load_json(Transcript, FIXTURES / "talk.transcript.json")


@pytest.fixture
def fixture_video() -> Path:
    return FIXTURES / "talk.mp4"
