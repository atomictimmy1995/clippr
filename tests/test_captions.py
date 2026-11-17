import re

from clippr.pipeline.captions import (
    MAX_CHARS, MAX_GAP, MAX_WORDS, _ass_time, _srt_time, build_ass, build_srt, chunk_words, clip_words,
)
from clippr.schemas import Word


def words(spec):
    return [Word(w=w, start=s, end=e) for w, s, e in spec]


def test_clip_words_shifts_and_trims_to_the_clip():
    ws = words([("before", 0, 1), ("edge", 9.8, 10.3), ("inside", 11, 11.5), ("after", 21, 22)])
    out = clip_words(ws, 10, 20)
    assert [w.w for w in out] == ["edge", "inside"]
    assert out[0].start == 0 and out[0].end == 0.3
    assert out[1].start == 1.0


def test_chunks_respect_word_char_gap_and_sentence_limits():
    ws = words([
        ("one", 0, .2), ("two", .25, .4), ("three", .45, .6), ("four", .65, .8), ("five", .85, 1.0),
        ("pause", 2.0, 2.2), ("end.", 2.3, 2.5), ("New", 2.6, 2.8), ("supercalifragilistic", 2.9, 3.6),
    ])
    chunks = chunk_words(ws, 10)
    texts = [c.text for c in chunks]
    assert all(len(c.words) <= MAX_WORDS for c in chunks)
    assert all(len(c.text) <= MAX_CHARS or len(c.words) == 1 for c in chunks)
    assert "pause" not in texts[0]  # long pause starts a new chunk
    assert any(t.endswith("end.") for t in texts)  # sentence end closes a chunk
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-9  # never two captions on screen at once
        if b.start - a.words[-1].end <= MAX_GAP:
            assert a.end == b.start  # no flicker between close chunks


def test_time_formats():
    assert _ass_time(3723.456) == "1:02:03.46"
    assert _srt_time(3723.456) == "01:02:03,456"


def test_ass_highlights_one_word_per_event_and_escapes():
    ws = words([("Hello", 0, .4), ("{world}", .5, 1.0)])
    chunks = chunk_words(ws, 5)
    ass = build_ass(chunks, 1080, 1920, "Inter ExtraBold")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 2
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert all(line.count("\\c&H0000D7FF&") == 1 for line in events)
    assert "{world}" not in ass and "(world)" in ass


def test_srt_blocks():
    chunks = chunk_words(words([("Hi", 0, .3), ("there.", .4, .8), ("Bye", 3, 3.3)]), 5)
    srt = build_srt(chunks)
    assert re.findall(r"^\d+$", srt, re.M) == ["1", "2"]
    assert "00:00:00,000 --> " in srt
