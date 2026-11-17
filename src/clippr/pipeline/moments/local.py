"""Free picker using a local open-weight model served by Ollama (https://ollama.com)."""

from __future__ import annotations

import logging
import math

import httpx
from pydantic import ValidationError

from clippr.config import Settings
from clippr.pipeline.moments.errors import PickerError
from clippr.pipeline.moments.prompt import SYSTEM_PROMPT, format_transcript, request_text
from clippr.schemas import Moment, MomentList, Signals, Transcript, VideoInfo

log = logging.getLogger(__name__)

WINDOW_S = 600.0  # local models handle ~10 minutes of transcript well
OVERLAP_S = 60.0


def available(settings: Settings) -> bool:
    """True if an Ollama server is reachable and has the configured model pulled."""
    try:
        r = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2.0)
        r.raise_for_status()
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return False
    model = settings.ollama_model
    return model in names or f"{model}:latest" in names


def windows(duration: float, size: float = WINDOW_S, overlap: float = OVERLAP_S) -> list[tuple[float, float]]:
    if duration <= size * 1.2:
        return [(0.0, duration)]
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + size)
        out.append((start, end))
        if end >= duration:
            break
        start = end - overlap
    return out


def _ask(client: httpx.Client, settings: Settings, user_text: str) -> MomentList:
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "format": MomentList.model_json_schema(),
        "stream": False,
        "options": {"num_ctx": settings.ollama_num_ctx, "temperature": 0.2},
    }
    r = client.post(f"{settings.ollama_url.rstrip('/')}/api/chat", json=payload)
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "")
    return MomentList.model_validate_json(content)


def pick(transcript: Transcript, signals: Signals, info: VideoInfo, settings: Settings, *,
         n: int, min_s: float, max_s: float, focus: str | None = None) -> list[Moment]:
    spans = windows(transcript.duration)
    per_window = max(2, min(6, math.ceil(n * 1.5 * WINDOW_S / max(transcript.duration, 1.0))))
    if len(spans) == 1:
        per_window = math.ceil(n * 1.5)

    found: list[Moment] = []
    with httpx.Client(timeout=settings.ollama_timeout_s) as client:
        for ws, we in spans:
            whole = len(spans) == 1
            text = format_transcript(transcript, signals, info, None if whole else ws, None if whole else we)
            req = request_text(per_window, min_s, max_s, focus, None if whole else ws, None if whole else we)
            result = None
            for attempt in (1, 2):
                try:
                    result = _ask(client, settings, f"{text}\n\n{req}")
                    break
                except (httpx.HTTPError, ValidationError, ValueError) as e:
                    log.warning("local model window %.0f-%.0f attempt %d failed: %s", ws, we, attempt, e)
            if result is None:
                raise PickerError(f"local model '{settings.ollama_model}' returned no usable answer")
            found += [m for m in result.moments if ws - 5 <= m.start_s <= we]
    return found
