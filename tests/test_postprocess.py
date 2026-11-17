from clippr.pipeline.moments.postprocess import (
    PAD_END, PAD_START, finalize, overlap_fraction, sentences, snap,
)
from clippr.schemas import Moment, Word


def m(start, end, score=50, title="t"):
    return Moment(start_s=start, end_s=end, title=title, hook="h", rationale="r", score=score)


def test_sentences_split_on_punctuation_and_long_pauses():
    words = [Word(w="Hello", start=0, end=0.4), Word(w="there.", start=0.5, end=0.9),
             Word(w="No", start=1.0, end=1.2), Word(w="punctuation", start=1.3, end=1.8),
             Word(w="after", start=3.0, end=3.3), Word(w="pause", start=3.4, end=3.8)]
    s = sentences(words)
    assert [x.text for x in s] == ["Hello there.", "No punctuation", "after pause"]
    assert (s[1].i0, s[1].i1) == (2, 3)


def test_snap_moves_to_sentence_boundaries_and_respects_length(transcript):
    words = transcript.words()
    sents = sentences(words)
    sentence_starts = {s.start for s in sents}
    sentence_ends = {s.end for s in sents}
    # A sloppy suggestion that starts and ends mid-sentence.
    out = snap(m(13.3, 33.1), words, sents, transcript.duration, 15, 30)
    assert round(out.start_s + PAD_START, 2) in {round(x, 2) for x in sentence_starts}
    assert round(out.end_s - PAD_END, 2) in {round(x, 2) for x in sentence_ends}
    assert 15 <= out.end_s - out.start_s <= 30


def test_snap_clamps_overlong_and_reversed_ranges(transcript):
    words = transcript.words()
    sents = sentences(words)
    long = snap(m(2, 50), words, sents, transcript.duration, 15, 30)
    assert long.end_s - long.start_s <= 30
    reversed_ = snap(m(30, 10), words, sents, transcript.duration, 15, 30)
    assert reversed_.start_s < reversed_.end_s
    near_end = snap(m(transcript.duration - 2, transcript.duration + 5), words, sents, transcript.duration, 15, 30)
    assert near_end.end_s <= transcript.duration + 1e-6
    assert near_end.end_s - near_end.start_s >= 15


def test_overlap_fraction():
    assert overlap_fraction(m(0, 10), m(5, 15)) == 0.5
    assert overlap_fraction(m(0, 10), m(10, 20)) == 0
    assert overlap_fraction(m(0, 30), m(10, 20)) == 1.0


def test_finalize_drops_overlaps_ranks_and_limits(transcript):
    cands = [m(1, 25, 60, "a"), m(3, 27, 90, "b"), m(30, 55, 70, "c"), m(40, 58, 10, "d"), m(12, 33, 150, "e")]
    out = finalize(cands, transcript, n=3, min_s=15, max_s=30)
    scores = [x.score for x in out]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100  # clamped to 0-100
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            assert overlap_fraction(a, b) <= 0.2
    assert len(out) <= 3


def test_snap_without_sentences_falls_back_to_word_boundaries():
    words = [Word(w=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(120)]
    out = snap(m(10.3, 31), words, [], 60, 15, 30)  # no sentence boundaries at all
    assert out.start_s == 10.5 - PAD_START  # nearest word start to 10.3
    assert round(out.end_s - PAD_END, 2) in {round(w.end, 2) for w in words}
    assert 15 <= out.end_s - out.start_s <= 30
