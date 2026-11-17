"""Job actions shared by the HTML pages and the JSON API."""

from __future__ import annotations

import shutil

from sqlalchemy.engine import Engine
from sqlmodel import Session, delete

from clippr.config import Settings
from clippr.db import Clip, Job, utcnow
from clippr.pipeline.download import DownloadError, validate_youtube_url
from clippr.pipeline.runner import JobPaths


class JobError(ValueError):
    pass


def create_job(engine: Engine, settings: Settings, url: str, *, max_clips: int | None = None,
               focus: str | None = None) -> Job:
    try:
        url = validate_youtube_url(url)
    except DownloadError as e:
        raise JobError(str(e)) from e
    n = max_clips or settings.max_clips
    if not 1 <= n <= 20:
        raise JobError("Number of clips must be between 1 and 20")
    focus = (focus or "").strip()[:300] or None
    job = Job(source=url, max_clips=n, focus=focus)
    with Session(engine) as s:
        s.add(job)
        s.commit()
        s.refresh(job)
    return job


def regenerate(engine: Engine, settings: Settings, job_id: str, *, focus: str | None = None,
               reselect: bool = True) -> Job:
    """Queue the job again. With ``reselect`` the moments are picked afresh (the transcript is
    reused); without it the job just resumes from the last stage that finished (a retry)."""
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            raise JobError("Job not found")
        if job.status in ("queued", "running"):
            raise JobError("This job is still in progress")
        if job.source_deleted:
            raise JobError("The source video was cleaned up; start a new job to clip it again")
        if focus is not None:
            job.focus = focus.strip()[:300] or None
        job.status, job.rerun_select, job.progress = "queued", reselect, 0.0
        job.message, job.error, job.updated_at = "Waiting for the worker", None, utcnow()
        s.add(job)
        s.commit()
        s.refresh(job)
        return job


def delete_job(engine: Engine, settings: Settings, job_id: str) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            raise JobError("Job not found")
        if job.status == "running":
            raise JobError("Wait for this job to finish before deleting it")
        s.exec(delete(Clip).where(Clip.job_id == job_id))
        s.delete(job)
        s.commit()
    shutil.rmtree(job_dir(settings, job_id), ignore_errors=True)


def job_dir(settings: Settings, job_id: str):
    return settings.jobs_dir / job_id


def job_paths(settings: Settings, job_id: str) -> JobPaths:
    return JobPaths(job_dir(settings, job_id))
