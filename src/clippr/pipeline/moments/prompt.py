"""Prompt text shared by the Claude and local-model pickers."""

from __future__ import annotations

from clippr.pipeline.signals import segment_tags
from clippr.schemas import Signals, Transcript, VideoInfo

SYSTEM_PROMPT = """\
You are an experienced short-form video editor. From the transcript of a long video, you pick the \
moments that will work best as standalone vertical clips for YouTube Shorts, TikTok and Reels.

What makes a good clip:
- It makes sense to someone who has not seen the rest of the video.
- It opens with a hook in the first few seconds: a bold claim, a question, a surprising fact, or \
the start of a story.
- It ends on a complete thought or a payoff, never mid-sentence.
- Strong candidates: emotional peaks, strong opinions, humour, clear explanations of one idea, \
surprising facts, quotable lines, and stories with a punchline.
- Avoid intros, greetings, sponsor reads, housekeeping, calls to subscribe, and outros.

About the transcript:
- Each line is `[start–end] text`, with times in seconds from the start of the video.
- Tags in braces after a line are measured signals, to use as hints rather than rules: \
loud = spike in audio energy, laughter = laughter or applause, cut = camera cut, fast = rapid speech.

Rules for your answer:
- start_s and end_s are seconds, taken from the transcript timestamps.
- Clips must not overlap each other.
- Score honestly from 0 to 100. Reserve 90+ for clips you expect to perform exceptionally well.
- Titles are short and punchy (under 60 characters). The hook is the opening line or idea that \
grabs attention.\
"""


def _ts(t: float) -> str:
    return f"{t:.1f}"


def format_transcript(transcript: Transcript, signals: Signals, info: VideoInfo,
                      start: float | None = None, end: float | None = None) -> str:
    lines = [f"Video title: {info.title or 'unknown'}", f"Duration: {info.duration:.0f} seconds"]
    if info.chapters:
        lines.append("Chapters:")
        lines += [f"- {_ts(c.start)}: {c.title}" for c in info.chapters]
    lines.append("")
    lines.append("Transcript:")
    for seg in transcript.segments:
        if start is not None and seg.end <= start:
            continue
        if end is not None and seg.start >= end:
            continue
        tags = segment_tags(seg, signals)
        tag_txt = f"  {{{','.join(tags)}}}" if tags else ""
        lines.append(f"[{_ts(seg.start)}–{_ts(seg.end)}] {seg.text}{tag_txt}")
    return "\n".join(lines)


def request_text(n: int, min_s: float, max_s: float, focus: str | None,
                 start: float | None = None, end: float | None = None) -> str:
    parts = [
        f"Find the {n} best moments for clips between {min_s:.0f} and {max_s:.0f} seconds long."
    ]
    if start is not None and end is not None:
        parts.append(
            f"This is one part of a longer video. Only pick clips that lie between {start:.0f} "
            f"and {end:.0f} seconds. If nothing in this part is worth clipping, return fewer."
        )
    if focus:
        parts.append(f"The person asking wants clips about: {focus.strip()}")
    return "\n".join(parts)
