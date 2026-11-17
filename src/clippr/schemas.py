"""Data passed between pipeline stages. Each one is saved as JSON in the job directory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")


class Word(BaseModel):
    w: str
    start: float
    end: float
    p: float = 1.0

    @property
    def ends_sentence(self) -> bool:
        return bool(_SENTENCE_END.search(self.w.strip()))


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str = "en"
    duration: float
    segments: list[Segment]

    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]


class Chapter(BaseModel):
    title: str
    start: float
    end: float


class VideoInfo(BaseModel):
    source: str
    title: str = ""
    uploader: str = ""
    duration: float
    width: int
    height: int
    fps: float
    chapters: list[Chapter] = Field(default_factory=list)


class Signals(BaseModel):
    loudness_db: list[float]  # one value per second of audio
    loudness_z: list[float]
    scene_cuts: list[float]  # timestamps in seconds


class Moment(BaseModel):
    """One clip suggested by a picker. Times are seconds from the start of the video."""

    start_s: float = Field(description="Clip start time in seconds from the start of the video")
    end_s: float = Field(description="Clip end time in seconds from the start of the video")
    title: str = Field(description="Short, punchy title for the clip (under 60 characters)")
    hook: str = Field(description="The opening line or idea that grabs attention in the first seconds")
    rationale: str = Field(description="One sentence on why this moment works as a standalone short clip")
    score: int = Field(description="How strong this clip is as a standalone short, 0 to 100")


class MomentList(BaseModel):
    moments: list[Moment]


class SelectedMoments(BaseModel):
    picker: str
    notes: list[str] = Field(default_factory=list)
    moments: list[Moment]


M = TypeVar("M", bound=BaseModel)


def save_json(model: BaseModel, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(model.model_dump_json(indent=2))
    tmp.replace(path)


def load_json(cls: type[M], path: Path) -> M:
    return cls.model_validate_json(path.read_text())
