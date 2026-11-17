"""Runs the pipeline stages in order. A stage whose output file already exists is skipped,
so a failed or interrupted job resumes where it stopped."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from clippr.config import Settings
from clippr.pipeline.captions import write_captions
from clippr.pipeline.download import fetch
from clippr.pipeline.moments import select_moments
from clippr.pipeline.reframe import CropPlan, FaceDetector, plan_crop
from clippr.pipeline.render import render_clip
from clippr.pipeline.signals import compute_signals
from clippr.pipeline.transcribe import transcribe
from clippr.schemas import (
    Moment, SelectedMoments, Signals, Transcript, VideoInfo, load_json, save_json,
)

log = logging.getLogger(__name__)

# (stage, label, share of overall progress)
STAGES: list[tuple[str, str, float]] = [
    ("download", "Downloading", 0.15),
    ("transcribe", "Transcribing", 0.35),
    ("signals", "Analysing audio and shots", 0.05),
    ("select", "Picking key moments", 0.10),
    ("render", "Rendering clips", 0.35),
]

Report = Callable[[str, float, str], None]  # (stage, fraction of that stage done, message)


def overall_progress(stage: str, fraction: float) -> float:
    done = 0.0
    for name, _, share in STAGES:
        if name == stage:
            return round(done + share * max(0.0, min(1.0, fraction)), 4)
        done += share
    return 1.0


class JobPaths:
    def __init__(self, job_dir: Path):
        self.dir = job_dir
        self.source = job_dir / "source.mp4"
        self.audio = job_dir / "audio.wav"
        self.info = job_dir / "info.json"
        self.transcript = job_dir / "transcript.json"
        self.signals = job_dir / "signals.json"
        self.moments = job_dir / "moments.json"
        self.clips = job_dir / "clips"

    def clip(self, n: int, ext: str) -> Path:
        return self.clips / f"clip_{n:02d}.{ext}"

    def clear_selection(self) -> None:
        """Forget the picked moments and rendered clips (used by "regenerate")."""
        self.moments.unlink(missing_ok=True)
        shutil.rmtree(self.clips, ignore_errors=True)

    def delete_source_media(self) -> None:
        self.source.unlink(missing_ok=True)
        self.audio.unlink(missing_ok=True)


@dataclass
class ClipResult:
    index: int
    moment: Moment
    video: Path
    srt: Path
    thumb: Path


@dataclass
class PipelineResult:
    info: VideoInfo
    selected: SelectedMoments
    clips: list[ClipResult]


def run_pipeline(source: str, job_dir: Path, settings: Settings, *, max_clips: int,
                 focus: str | None = None, report: Report = lambda s, f, m: None) -> PipelineResult:
    p = JobPaths(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    if p.info.exists() and p.audio.exists() and p.source.exists():
        info = load_json(VideoInfo, p.info)
    else:
        info = fetch(source, job_dir, settings, lambda f, m: report("download", f, m))
        save_json(info, p.info)
    report("download", 1.0, "Downloaded")

    if p.transcript.exists():
        transcript = load_json(Transcript, p.transcript)
    else:
        transcript = transcribe(p.audio, info.duration, settings, lambda f, m: report("transcribe", f, m))
        save_json(transcript, p.transcript)
    report("transcribe", 1.0, "Transcribed")

    if p.signals.exists():
        signals = load_json(Signals, p.signals)
    else:
        report("signals", 0.0, "Measuring loudness and finding camera cuts")
        signals = compute_signals(p.source, p.audio)
        save_json(signals, p.signals)
    report("signals", 1.0, "Analysed")

    if p.moments.exists():
        selected = load_json(SelectedMoments, p.moments)
    else:
        selected = select_moments(transcript, signals, info, settings, n=max_clips, focus=focus,
                                  progress=lambda f, m: report("select", f, m))
        save_json(selected, p.moments)
    report("select", 1.0, f"Picked {len(selected.moments)} moments")

    p.clips.mkdir(exist_ok=True)
    detector: FaceDetector | None = None  # created on first use: it may download a model
    words = transcript.words()
    results: list[ClipResult] = []
    total = len(selected.moments)
    for i, m in enumerate(selected.moments, start=1):
        video, srt, ass, thumb, crop = (p.clip(i, e) for e in ("mp4", "srt", "ass", "jpg", "crop.json"))
        results.append(ClipResult(i, m, video, srt, thumb))
        if video.exists():
            continue

        def sub(frac: float, msg: str, i: int = i) -> None:
            report("render", ((i - 1) + frac) / total, f"Clip {i} of {total}: {msg}")

        write_captions(words, m.start_s, m.end_s, ass, srt, width=settings.output_width,
                       height=settings.output_height, font=settings.caption_font)
        if crop.exists():
            plan = load_json(CropPlan, crop)
        else:
            detector = detector or FaceDetector(settings.models_dir)
            plan = plan_crop(p.source, m.start_s, m.end_s, signals.scene_cuts, detector,
                             out_w=settings.output_width, out_h=settings.output_height,
                             progress=lambda f: sub(0.3 * f, "tracking the speaker"))
            save_json(plan, crop)
        render_clip(p.source, m.start_s, m.end_s, plan, ass, video, thumb,
                    out_w=settings.output_width, out_h=settings.output_height, fps=info.fps,
                    fonts_dir=settings.fonts_dir,
                    progress=lambda f: sub(0.3 + 0.7 * f, "rendering"))
    report("render", 1.0, f"Rendered {total} clips")
    return PipelineResult(info=info, selected=selected, clips=results)
