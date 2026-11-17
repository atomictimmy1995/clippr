"""Stage 6: caption files for a clip. ASS (burned in, active word highlighted) and SRT (sidecar)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clippr.schemas import Word

HIGHLIGHT = "&H0000D7FF&"  # #FFD700 gold, in ASS's BGR order
MAX_WORDS = 4
MAX_CHARS = 18
MAX_GAP = 0.6  # never keep a caption on screen across a pause longer than this
HOLD = 0.25  # how long the last chunk lingers after its last word


@dataclass
class Chunk:
    start: float
    end: float
    words: list[Word]  # times relative to the clip start

    @property
    def text(self) -> str:
        return " ".join(w.w for w in self.words)


def clip_words(words: list[Word], clip_start: float, clip_end: float) -> list[Word]:
    """Words inside the clip, with times shifted so the clip starts at 0."""
    dur = clip_end - clip_start
    out = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        s = max(0.0, w.start - clip_start)
        e = min(dur, w.end - clip_start)
        if e > s:
            out.append(Word(w=w.w, start=round(s, 3), end=round(e, 3), p=w.p))
    return out


def chunk_words(words: list[Word], clip_len: float) -> list[Chunk]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            chars = len(" ".join(x.w for x in cur + [w]))
            gap = w.start - cur[-1].end
            if len(cur) >= MAX_WORDS or chars > MAX_CHARS or gap > MAX_GAP or cur[-1].ends_sentence:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    chunks: list[Chunk] = []
    for i, g in enumerate(groups):
        end = g[-1].end + HOLD
        if i + 1 < len(groups):
            nxt = groups[i + 1][0].start
            end = nxt if nxt - g[-1].end <= MAX_GAP else min(end, nxt)
        chunks.append(Chunk(start=g[0].start, end=min(end, clip_len), words=g))
    return chunks


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(chunks: list[Chunk], width: int, height: int, font: str) -> str:
    size = round(width * 0.074)  # ~80 px at 1080 wide
    x, y = width // 2, round(height * 0.68)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{max(3, size // 13)},{max(1, size // 40)},5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for chunk in chunks:
        # One event per word, so the word being spoken is highlighted while the rest stay white.
        for i, w in enumerate(chunk.words):
            start = w.start
            end = chunk.words[i + 1].start if i + 1 < len(chunk.words) else chunk.end
            if end <= start:
                continue
            parts = []
            for j, other in enumerate(chunk.words):
                t = _ass_escape(other.w)
                parts.append(f"{{\\c{HIGHLIGHT}\\fscx108\\fscy108}}{t}{{\\r}}" if j == i else t)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                f"{{\\an5\\pos({x},{y})}}{' '.join(parts)}"
            )
    return header + "\n".join(events) + "\n"


def build_srt(chunks: list[Chunk]) -> str:
    blocks = [
        f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n".join(blocks)


def write_captions(words: list[Word], clip_start: float, clip_end: float, ass_path: Path,
                   srt_path: Path, *, width: int, height: int, font: str) -> list[Chunk]:
    chunks = chunk_words(clip_words(words, clip_start, clip_end), clip_end - clip_start)
    ass_path.write_text(build_ass(chunks, width, height, font), encoding="utf-8")
    srt_path.write_text(build_srt(chunks), encoding="utf-8")
    return chunks
