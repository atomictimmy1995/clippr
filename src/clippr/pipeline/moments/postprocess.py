"""Shared clean-up for every picker: snap to sentence boundaries, enforce length, drop overlaps, rank."""

from __future__ import annotations

from dataclasses import dataclass

from clippr.schemas import Moment, Transcript, Word

PAD_START = 0.2  # seconds of lead-in before the first word
PAD_END = 0.35  # seconds of tail after the last word
SNAP_WINDOW = 6.0  # how far a start may move to reach a sentence boundary
MAX_OVERLAP = 0.2  # fraction of the shorter clip


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    i0: int  # index of first word
    i1: int  # index of last word (inclusive)


def sentences(words: list[Word], pause: float = 0.8) -> list[Sentence]:
    """Split words into sentences on end punctuation, or on a long pause when there is none."""
    out: list[Sentence] = []
    i0 = 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        gap = (words[i + 1].start - w.end) if not last else 0.0
        if last or w.ends_sentence or gap > pause:
            chunk = words[i0 : i + 1]
            out.append(Sentence(chunk[0].start, chunk[-1].end, " ".join(x.w for x in chunk), i0, i))
            i0 = i + 1
    return out


def _nearest(values: list[float], target: float) -> float | None:
    return min(values, key=lambda v: abs(v - target)) if values else None


def snap(m: Moment, words: list[Word], sents: list[Sentence], duration: float,
         min_s: float, max_s: float) -> Moment:
    start, end = sorted((float(m.start_s), float(m.end_s)))
    start = min(max(0.0, start), duration)
    end = min(max(0.0, end), duration)
    if end - start < 1.0:  # picker gave a point, not a range: aim for a full-length clip
        end = min(duration, start + max_s * 0.9)

    if duration <= min_s or not words:
        s, e = (0.0, duration) if duration <= min_s else (start, min(end, start + max_s))
        return m.model_copy(update={"start_s": round(s, 2), "end_s": round(e, 2)})

    # Leave room for a full-length clip before the end of the video.
    start = min(start, max(0.0, duration - min_s))
    starts = [s.start for s in sents if abs(s.start - start) <= SNAP_WINDOW and s.start <= duration - min_s]
    new_start = _nearest(starts, start)
    if new_start is None:
        new_start = _nearest([w.start for w in words if w.start <= duration - min_s], start)
    if new_start is None:
        new_start = start

    lo = new_start + min_s
    hi = min(new_start + max_s - PAD_START - PAD_END, duration)
    new_end = _nearest([s.end for s in sents if lo <= s.end <= hi], end)
    if new_end is None:
        new_end = _nearest([w.end for w in words if lo <= w.end <= hi], end)
    if new_end is None:
        new_end = min(max(end, lo), hi)

    s = max(0.0, new_start - PAD_START)
    e = min(duration, new_end + PAD_END, s + max_s)
    return m.model_copy(update={"start_s": round(s, 2), "end_s": round(e, 2)})


def overlap_fraction(a: Moment, b: Moment) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    shorter = max(1e-6, min(a.end_s - a.start_s, b.end_s - b.start_s))
    return inter / shorter


def finalize(candidates: list[Moment], transcript: Transcript, *, n: int, min_s: float,
             max_s: float) -> list[Moment]:
    words = transcript.words()
    sents = sentences(words)
    snapped = [snap(c, words, sents, transcript.duration, min_s, max_s) for c in candidates]
    snapped = [m.model_copy(update={"score": max(0, min(100, int(m.score)))}) for m in snapped]
    kept: list[Moment] = []
    for m in sorted(snapped, key=lambda x: x.score, reverse=True):
        if m.end_s - m.start_s < 1.0:
            continue
        if all(overlap_fraction(m, k) <= MAX_OVERLAP for k in kept):
            kept.append(m)
        if len(kept) >= n:
            break
    return kept
