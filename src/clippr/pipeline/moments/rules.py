"""Free picker with no AI: scores sentence-aligned windows using the measured signals."""

from __future__ import annotations

import re

import numpy as np

from clippr.pipeline.moments.postprocess import Sentence, sentences
from clippr.pipeline.signals import LAUGH_RE, window_values
from clippr.schemas import Moment, Signals, Transcript

_WORD_RE = re.compile(r"[a-z0-9']{4,}")


def _z(values: list[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    std = a.std()
    return np.zeros_like(a) if std < 1e-9 else (a - a.mean()) / std


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def pick(transcript: Transcript, signals: Signals, *, min_s: float, max_s: float,
         focus: str | None = None, limit: int = 200) -> list[Moment]:
    words = transcript.words()
    sents = sentences(words)
    if not sents:
        return []

    # Each sentence start opens the longest window that ends on a sentence end and fits max_s.
    spans: list[tuple[int, int]] = []
    for i, s in enumerate(sents):
        best = None
        for j in range(i, len(sents)):
            length = sents[j].end - s.start
            if length > max_s - 0.6:
                break
            if length >= min_s:
                best = j
        if best is not None:
            spans.append((i, best))
    if not spans:  # video (or speech) shorter than one clip
        spans = [(0, len(sents) - 1)]

    focus_terms = set(_WORD_RE.findall((focus or "").lower()))
    energy, rate, dead, laughs, cuts, hook, topical = [], [], [], [], [], [], []
    for i, j in spans:
        st, en = sents[i].start, sents[j].end
        dur = max(en - st, 1.0)
        z = window_values(signals.loudness_z, st, en)
        energy.append(float(np.mean(sorted(z)[-5:])) if z else 0.0)
        w0, w1 = sents[i].i0, sents[j].i1
        rate.append((w1 - w0 + 1) / dur)
        gaps = sum(max(0.0, words[k + 1].start - words[k].end - 0.3) for k in range(w0, w1))
        dead.append(gaps / dur)
        text = " ".join(s.text for s in sents[i : j + 1])
        laughs.append(min(3, len(LAUGH_RE.findall(text))))
        cuts.append(sum(1 for c in signals.scene_cuts if st <= c < en) / dur * 60)
        first = sents[i]
        n_first = first.i1 - first.i0 + 1
        hook.append((1.0 if first.text.rstrip().endswith("?") else 0.0) + (0.5 if 4 <= n_first <= 20 else 0.0))
        topical.append(len(focus_terms & set(_WORD_RE.findall(text.lower()))) if focus_terms else 0)

    score = (
        0.35 * _z(energy)
        + 0.25 * _z(rate)
        - 0.30 * _z(dead)
        + 0.20 * np.asarray(laughs, dtype=np.float64)
        + 0.10 * _z(cuts)
        + 0.15 * np.asarray(hook)
        + 0.60 * np.asarray(topical, dtype=np.float64)
    )
    order = np.argsort(-score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(len(score))
    percentile = 100 * (1 - ranks / max(1, len(score) - 1)) if len(score) > 1 else np.array([75.0])

    moments: list[Moment] = []
    for idx in order[:limit]:
        i, j = spans[idx]
        window: list[Sentence] = sents[i : j + 1]
        peak = max(window, key=lambda s: max(window_values(signals.loudness_z, s.start, s.end) or [0.0]))
        moments.append(Moment(
            start_s=window[0].start,
            end_s=window[-1].end,
            title=_shorten(window[0].text, 60),
            hook=_shorten(peak.text, 120),
            rationale="Picked by the free rules-based picker from audio energy, speech pace and pauses.",
            score=int(round(percentile[idx] * 0.9)),  # rules can't be as sure as a model
        ))
    return moments
