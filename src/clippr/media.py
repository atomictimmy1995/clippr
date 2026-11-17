"""Small wrappers around the ffmpeg and ffprobe command-line tools."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str
    audio_codec: str | None


def run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command (argument list, never a shell string) and raise MediaError on failure."""
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise MediaError(f"{cmd[0]} failed (exit {proc.returncode}): {tail}")
    return proc


def probe(path: Path) -> Probe:
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    num, _, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    duration = float(data["format"].get("duration") or video.get("duration") or 0)
    return Probe(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 30.0,
        has_audio=audio is not None,
        video_codec=video.get("codec_name", ""),
        audio_codec=audio.get("codec_name") if audio else None,
    )


def extract_audio(video: Path, wav: Path) -> None:
    """16 kHz mono 16-bit WAV: the format Whisper and the loudness analysis use."""
    tmp = wav.with_suffix(".tmp.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(wav)
