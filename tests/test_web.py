import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from clippr.db import Clip, Job, get_engine
from clippr.web.app import create_app

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Make clips" in r.text and "Rules (free, no AI)" in r.text


def test_submit_requires_rights_confirmation(client):
    r = client.post("/jobs", data={"url": URL, "max_clips": 3})
    assert r.status_code == 400 and "rights" in r.text


def test_submit_rejects_non_youtube(client):
    r = client.post("/jobs", data={"url": "https://example.com/v", "rights": "true"})
    assert r.status_code == 400 and "Only YouTube links" in r.text


def test_submit_creates_job_and_shows_progress(client, settings):
    r = client.post("/jobs", data={"url": URL, "max_clips": 3, "focus": "jokes", "rights": "true"},
                    follow_redirects=False)
    assert r.status_code == 303
    job_id = r.headers["location"].rsplit("/", 1)[1]
    page = client.get(f"/jobs/{job_id}")
    assert page.status_code == 200 and 'hx-trigger="every 2s"' in page.text
    with Session(get_engine(settings.db_path)) as s:
        job = s.get(Job, job_id)
        assert (job.status, job.max_clips, job.focus) == ("queued", 3, "jokes")


def test_done_job_lists_clips_and_serves_files(client, settings):
    engine = get_engine(settings.db_path)
    with Session(engine) as s:
        job = Job(source=URL, status="done", picker="rules", title="My video")
        s.add(job)
        s.commit()
        s.add(Clip(job_id=job.id, idx=1, start_s=10, end_s=40, title="Great bit", hook="Wow", rationale="r", score=88))
        s.commit()
        job_id = job.id
    clips_dir = settings.jobs_dir / job_id / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "clip_01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    status = client.get(f"/jobs/{job_id}/status")
    assert "Great bit" in status.text and "every 2s" not in status.text
    assert client.get(f"/jobs/{job_id}/clips/clip_01.srt").text.startswith("1\n")
    dl = client.get(f"/jobs/{job_id}/clips/clip_01.srt?download=true")
    assert "attachment" in dl.headers["content-disposition"]
    api = client.get(f"/api/jobs/{job_id}").json()
    assert api["clips"][0]["video_url"].endswith("clip_01.mp4") and api["picker"] == "rules"


@pytest.mark.parametrize("path", [
    "/jobs/{id}/clips/..%2Fclippr.db",
    "/jobs/{id}/clips/clip_01.ass",
    "/jobs/{id}/clips/source.mp4",
    "/jobs/not-a-job/clips/clip_01.mp4",
])
def test_file_route_only_serves_clip_files(client, path):
    r = client.get(path.replace("{id}", "a" * 32))
    assert r.status_code == 404


def test_api_create_get_regenerate_delete(client, settings):
    assert client.post("/api/jobs", json={"url": URL}).status_code == 400  # rights not confirmed
    r = client.post("/api/jobs", json={"url": URL, "confirm_rights": True, "max_clips": 2})
    assert r.status_code == 201
    job_id = r.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    assert client.post(f"/api/jobs/{job_id}/regenerate").status_code == 409  # still queued
    with Session(get_engine(settings.db_path)) as s:
        job = s.get(Job, job_id)
        job.status = "done"
        s.add(job)
        s.commit()
    r = client.post(f"/api/jobs/{job_id}/regenerate", json={"focus": "space"})
    assert r.status_code == 200
    got = client.get(f"/api/jobs/{job_id}").json()
    assert (got["status"], got["focus"]) == ("queued", "space")
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
