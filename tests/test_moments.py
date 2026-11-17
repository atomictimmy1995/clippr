from types import SimpleNamespace

import pytest

from clippr.pipeline import moments
from clippr.pipeline.moments import PickerError, claude, local, picker_chain, rules, select_moments
from clippr.schemas import Moment, MomentList, Signals, VideoInfo


@pytest.fixture
def signals(transcript):
    n = int(transcript.duration)
    loud = [0.0] * n
    for t in range(30, 33):  # the "loud" line in the fixture
        loud[t] = 2.5
    return Signals(loudness_db=[-30.0] * n, loudness_z=loud, scene_cuts=[37.2])


@pytest.fixture
def info(transcript):
    return VideoInfo(source="x", title="Rockets", duration=transcript.duration, width=960, height=540, fps=25)


def test_auto_chain_prefers_claude_then_local_then_rules(settings, monkeypatch):
    s = settings.model_copy(update={"selector": "auto"})
    monkeypatch.setattr(local, "available", lambda _s: False)
    assert picker_chain(s) == ["rules"]
    monkeypatch.setattr(local, "available", lambda _s: True)
    assert picker_chain(s) == ["local", "rules"]
    s_key = s.model_copy(update={"anthropic_api_key": "sk-test"})
    assert picker_chain(s_key) == ["claude", "local", "rules"]
    assert picker_chain(s.model_copy(update={"selector": "claude"})) == ["claude"]
    assert picker_chain(s.model_copy(update={"selector": "local"})) == ["local", "rules"]


def test_local_unavailable_when_nothing_listens(settings):
    assert local.available(settings) is False


def test_rules_picker_returns_sentence_aligned_windows(transcript, signals):
    out = rules.pick(transcript, signals, min_s=15, max_s=30)
    assert out
    for mo in out:
        assert 15 <= mo.end_s - mo.start_s <= 30
        assert 0 <= mo.score <= 100
    assert out[0].score >= out[-1].score


def test_rules_picker_focus_boosts_matching_windows(transcript, signals):
    plain = rules.pick(transcript, signals, min_s=15, max_s=30)
    focused = rules.pick(transcript, signals, min_s=15, max_s=30, focus="payload kilogram")
    assert "kilogram" not in (plain[0].title + plain[0].hook).lower() or plain[0] == focused[0]
    best = focused[0]
    text = " ".join(s.text for s in transcript.segments if best.start_s <= s.start < best.end_s).lower()
    assert "kilogram" in text or "payload" in text


def test_local_windows_cover_long_videos_with_overlap():
    assert local.windows(500) == [(0.0, 500)]
    w = local.windows(2000)
    assert w[0][0] == 0 and w[-1][1] == 2000
    for (a0, a1), (b0, _) in zip(w, w[1:]):
        assert b0 == a1 - local.OVERLAP_S


def test_falls_back_to_rules_when_claude_fails(transcript, signals, info, settings, monkeypatch):
    s = settings.model_copy(update={"selector": "auto", "anthropic_api_key": "sk-test",
                                    "clip_min_s": 15, "clip_max_s": 30})
    monkeypatch.setattr(local, "available", lambda _s: False)

    def boom(*a, **k):
        raise PickerError("Claude rejected the API key")

    monkeypatch.setattr(claude, "pick", boom)
    out = select_moments(transcript, signals, info, s, n=2)
    assert out.picker == "rules"
    assert out.notes and "Claude" in out.notes[0]
    assert 1 <= len(out.moments) <= 2


def test_explicit_claude_failure_is_an_error(transcript, signals, info, settings, monkeypatch):
    s = settings.model_copy(update={"selector": "claude", "anthropic_api_key": "sk-test"})
    monkeypatch.setattr(claude, "pick", lambda *a, **k: (_ for _ in ()).throw(PickerError("nope")))
    with pytest.raises(PickerError):
        select_moments(transcript, signals, info, s, n=2)


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(monkeypatch, response):
    msgs = FakeMessages(response)
    fake = SimpleNamespace(beta=SimpleNamespace(messages=msgs), messages=SimpleNamespace())
    monkeypatch.setattr(claude.anthropic, "Anthropic", lambda **kw: fake)
    return msgs


def _usage():
    return SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0)


def test_claude_request_shape_and_result(transcript, signals, info, settings, monkeypatch):
    s = settings.model_copy(update={"anthropic_api_key": "sk-test"})
    parsed = MomentList(moments=[Moment(start_s=12, end_s=40, title="t", hook="h", rationale="r", score=80)])
    msgs = _fake_client(monkeypatch, SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, usage=_usage()))
    out = claude.pick(transcript, signals, info, s, n=2, min_s=15, max_s=30, focus="orbits")
    assert out == parsed.moments
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is MomentList
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default" and claude.FALLBACK_BETA in call["betas"]
    first, second = call["messages"][0]["content"]
    assert first["cache_control"] == {"type": "ephemeral"} and "[12." in first["text"]
    assert "orbits" in second["text"] and "orbits" not in first["text"]  # cached prefix stays stable


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_bad_stop_reasons_raise(transcript, signals, info, settings, monkeypatch, stop):
    s = settings.model_copy(update={"anthropic_api_key": "sk-test"})
    _fake_client(monkeypatch, SimpleNamespace(stop_reason=stop, parsed_output=None, usage=_usage()))
    with pytest.raises(PickerError):
        claude.pick(transcript, signals, info, s, n=2, min_s=15, max_s=30)


def test_no_speech_is_a_clear_error(signals, info, settings):
    from clippr.schemas import Transcript

    with pytest.raises(PickerError, match="No speech"):
        select_moments(Transcript(duration=10, segments=[]), signals, info, settings, n=2)


def test_module_exports():
    assert set(moments.PICKER_LABELS) == {"claude", "local", "rules"}
