"""End-to-end runs on the fixture video. The Whisper, Ollama and Claude variants run only when
those are available (a downloaded model, a running server, an API key)."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest
from sqlmodel import Session

from clippr.db import Clip, Job, get_engine, job_clips
from clippr.media import probe
from clippr.pipeline.moments import local
from clippr.pipeline.reframe import CropPlan
from clippr.pipeline.runner import run_pipeline
from clippr.schemas import load_json
from clippr.worker import worker_loop

FIXTURES = Path(__file__).resolve().parent / "fixtures"
pytestmark = pytest.mark.integration


def _seed_transcript(job_dir):
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "talk.transcript.json", job_dir / "transcript.json")


SRT_TIME = re.compile(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)")


def _srt_times(text):
    out = []
    for g in SRT_TIME.findall(text):
        v = [int(x) for x in g]
        out.append((v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1000, v[4] * 3600 + v[5] * 60 + v[6] + v[7] / 1000))
    return out


def check_clips(result, max_s):
    assert result.clips
    for c in result.clips:
        p = probe(c.video)
        assert (p.width, p.height) == (1080, 1920)
        assert p.video_codec == "h264" and p.audio_codec == "aac"
        assert p.duration <= max_s + 0.15
        assert abs(p.duration - (c.moment.end_s - c.moment.start_s)) < 0.3
        for start, end in _srt_times(c.srt.read_text()):
            assert 0 <= start < end <= p.duration + 0.05
        assert c.thumb.exists()


def test_pipeline_with_free_rules_picker(fixture_video, settings, tmp_path):
    s = settings.model_copy(update={"clip_min_s": 15, "clip_max_s": 30})
    job_dir = tmp_path / "job"
    _seed_transcript(job_dir)  # skip Whisper: its model can't be downloaded in every environment
    result = run_pipeline(str(fixture_video), job_dir, s, max_clips=2)
    assert result.selected.picker == "rules"
    check_clips(result, 30)

    # The clip that spans the camera cut should follow the face, then switch to the "fit" layout.
    plans = [load_json(CropPlan, c.video.with_suffix(".crop.json")) for c in result.clips]
    modes = {sh.mode for p in plans for sh in p.shots}
    assert modes == {"crop", "fit"}
    crop_shot = next(sh for p in plans for sh in p.shots if sh.mode == "crop")
    assert crop_shot.cx[-1] - crop_shot.cx[0] > 150  # the face drifts right in the fixture

    # Re-running skips finished stages (resume) and re-uses the rendered clips.
    mtime = result.clips[0].video.stat().st_mtime
    again = run_pipeline(str(fixture_video), job_dir, s, max_clips=2)
    assert again.clips[0].video.stat().st_mtime == mtime


def test_worker_runs_a_queued_job(fixture_video, settings):
    s = settings.model_copy(update={"clip_min_s": 15, "clip_max_s": 30})
    engine = get_engine(s.db_path)
    with Session(engine) as session:
        job = Job(source=str(fixture_video), max_clips=1)
        session.add(job)
        session.commit()
        job_id = job.id
    _seed_transcript(s.jobs_dir / job_id)
    worker_loop(s, once=True)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == "done", job.error
        assert job.picker == "rules" and job.progress == 1.0
        clips: list[Clip] = job_clips(session, job_id)
        assert len(clips) == 1
    assert (s.jobs_dir / job_id / "clips" / "clip_01.mp4").exists()


@pytest.mark.skipif(not os.environ.get("CLIPPR_TEST_WHISPER"), reason="set CLIPPR_TEST_WHISPER=1 to run Whisper")
def test_whisper_transcribes_fixture(fixture_video, settings, tmp_path):
    from clippr.media import extract_audio
    from clippr.pipeline.transcribe import transcribe

    wav = tmp_path / "a.wav"
    extract_audio(fixture_video, wav)
    t = transcribe(wav, probe(fixture_video).duration, settings, lambda f, m: None)
    text = " ".join(seg.text for seg in t.segments).lower()
    assert "orbit" in text and "kilogram" in text
    assert t.words() and all(w.end >= w.start for w in t.words())


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_pipeline_with_claude(fixture_video, settings, tmp_path):
    s = settings.model_copy(update={"selector": "claude", "anthropic_api_key": os.environ["ANTHROPIC_API_KEY"],
                                    "clip_min_s": 15, "clip_max_s": 30})
    _seed_transcript(tmp_path / "job")
    result = run_pipeline(str(fixture_video), tmp_path / "job", s, max_clips=1)
    assert result.selected.picker == "claude"
    check_clips(result, 30)


def test_pipeline_with_local_model(fixture_video, settings, tmp_path):
    s = settings.model_copy(update={"selector": "local", "ollama_url": "http://localhost:11434",
                                    "clip_min_s": 15, "clip_max_s": 30})
    if not local.available(s):
        pytest.skip(f"Ollama with model {s.ollama_model} is not running")
    _seed_transcript(tmp_path / "job")
    result = run_pipeline(str(fixture_video), tmp_path / "job", s, max_clips=1)
    assert result.selected.picker == "local"
    check_clips(result, 30)
