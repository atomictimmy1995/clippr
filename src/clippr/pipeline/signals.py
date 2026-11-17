"""Stage 3: cheap audio/video signals that hint at where the interesting moments are."""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np

from clippr.media import run
from clippr.schemas import Segment, Signals

LAUGH_RE = re.compile(r"[\[(](laugh\w*|applause|cheer\w*|chuckl\w*)[\])]", re.IGNORECASE)
_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def loudness_per_second(wav: Path) -> list[float]:
    """RMS loudness in dBFS for each whole second of a 16-bit mono WAV."""
    with wave.open(str(wav), "rb") as f:
        rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(samples) // rate
    if n == 0:
        return []
    per_sec = samples[: n * rate].reshape(n, rate)
    rms = np.sqrt(np.mean(per_sec**2, axis=1)) + 1e-9
    return [round(float(v), 2) for v in 20 * np.log10(rms)]


def zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    a = np.asarray(values, dtype=np.float64)
    std = a.std()
    if std < 1e-6:
        return [0.0] * len(values)
    return [round(float(v), 3) for v in (a - a.mean()) / std]


def scene_cuts(video: Path, threshold: float = 0.3) -> list[float]:
    """Timestamps of hard shot changes, using ffmpeg's scene-change score on a downscaled copy."""
    proc = run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-an", "-sn",
        "-vf", f"scale=240:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-",
    ])
    return [round(float(m), 3) for m in _PTS_RE.findall(proc.stderr)]


def compute_signals(video: Path, wav: Path) -> Signals:
    db = loudness_per_second(wav)
    return Signals(loudness_db=db, loudness_z=zscores(db), scene_cuts=scene_cuts(video))


def window_values(series: list[float], start: float, end: float) -> list[float]:
    a, b = max(0, int(start)), min(len(series), int(np.ceil(end)))
    return series[a:b]


def segment_tags(seg: Segment, signals: Signals) -> list[str]:
    """Short labels shown next to each transcript line in the prompt."""
    tags: list[str] = []
    z = window_values(signals.loudness_z, seg.start, seg.end)
    if z and max(z) > 1.5:
        tags.append("loud")
    if any(seg.start <= c < seg.end for c in signals.scene_cuts):
        tags.append("cut")
    dur = max(seg.end - seg.start, 0.1)
    if len(seg.words) / dur > 3.5:
        tags.append("fast")
    if LAUGH_RE.search(seg.text):
        tags.append("laughter")
    return tags
