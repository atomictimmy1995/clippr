"""Stage 2: transcribe the audio locally with faster-whisper, keeping word-level timestamps."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from clippr.config import Settings
from clippr.schemas import Segment, Transcript, Word

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def resolve_whisper(settings: Settings) -> tuple[str, str, str]:
    """Return (model, device, compute_type), filling in the "auto" settings."""
    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001 - any failure here just means "no usable GPU"
            device = "cpu"
    model = settings.whisper_model
    if model == "auto":
        model = "large-v3-turbo" if device == "cuda" else "small"
    compute = settings.whisper_compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return model, device, compute


def transcribe(wav: Path, duration: float, settings: Settings, progress: Progress) -> Transcript:
    from faster_whisper import WhisperModel  # heavy import, keep it local

    model_name, device, compute = resolve_whisper(settings)
    progress(0.0, f"Loading Whisper model '{model_name}' on {device}")
    log.info("whisper model=%s device=%s compute=%s", model_name, device, compute)
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute,
        download_root=str(settings.models_dir / "whisper"),
    )
    segments_iter, info = model.transcribe(
        str(wav),
        language=settings.language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(w=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3), p=round(w.probability, 3))
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(Segment(start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text.strip(), words=words))
        if duration:
            progress(min(0.99, seg.end / duration), "Transcribing")
    return Transcript(language=info.language, duration=duration, segments=segments)
