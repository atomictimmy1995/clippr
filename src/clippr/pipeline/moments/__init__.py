"""Stage 4: pick the key moments.

Three pickers share one interface. With ``selector="auto"`` the app uses Claude when an API key is
set, otherwise a local Ollama model when one is running, otherwise the free rules-based picker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from clippr.config import Settings
from clippr.pipeline.moments import claude, local, rules
from clippr.pipeline.moments.errors import PickerError
from clippr.pipeline.moments.postprocess import finalize
from clippr.schemas import Moment, SelectedMoments, Signals, Transcript, VideoInfo

log = logging.getLogger(__name__)

__all__ = ["PickerError", "picker_chain", "select_moments"]

PICKER_LABELS = {
    "claude": "Claude (paid API)",
    "local": "Local AI model (free)",
    "rules": "Rules (free, no AI)",
}


def picker_chain(settings: Settings) -> list[str]:
    """Pickers to try, in order. The last one's failure is fatal."""
    if settings.selector == "claude":
        return ["claude"]
    if settings.selector == "local":
        return ["local", "rules"]
    if settings.selector == "rules":
        return ["rules"]
    chain = []
    if settings.anthropic_api_key:
        chain.append("claude")
    if local.available(settings):
        chain.append("local")
    chain.append("rules")
    return chain


def select_moments(transcript: Transcript, signals: Signals, info: VideoInfo, settings: Settings, *,
                   n: int, focus: str | None = None,
                   progress: Callable[[float, str], None] = lambda f, m: None) -> SelectedMoments:
    min_s, max_s = settings.clip_min_s, settings.clip_max_s
    if not transcript.words():
        raise PickerError("No speech was found in the video, so there is nothing to clip")

    notes: list[str] = []
    chain = picker_chain(settings)
    for i, name in enumerate(chain):
        progress(0.1, f"Picking moments with {PICKER_LABELS[name]}")
        try:
            candidates: list[Moment]
            if name == "rules":
                candidates = rules.pick(transcript, signals, min_s=min_s, max_s=max_s, focus=focus)
            else:
                module = claude if name == "claude" else local
                candidates = module.pick(transcript, signals, info, settings, n=n, min_s=min_s,
                                         max_s=max_s, focus=focus)
            moments = finalize(candidates, transcript, n=n, min_s=min_s, max_s=max_s)
            if not moments:
                raise PickerError("no usable clips were suggested")
        except Exception as e:  # noqa: BLE001 - any picker failure falls through to the next one
            if i == len(chain) - 1:
                raise PickerError(str(e)) from e
            log.warning("%s picker failed, falling back: %s", name, e)
            notes.append(f"{PICKER_LABELS[name]} failed ({e}); used the next picker instead.")
            continue
        return SelectedMoments(picker=name, notes=notes, moments=moments)
    raise PickerError("no picker available")  # unreachable: "rules" is always last
