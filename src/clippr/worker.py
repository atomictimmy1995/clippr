"""The worker process: claims queued jobs one at a time and runs the pipeline for each."""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta

from sqlalchemy.engine import Engine
from sqlmodel import Session, delete, select

from clippr.config import Settings
from clippr.db import Clip, Job, claim_next_job, get_engine, requeue_stale, utcnow
from clippr.jobs import job_paths
from clippr.pipeline.runner import STAGES, overall_progress, run_pipeline

log = logging.getLogger(__name__)

STAGE_LABELS = {name: label for name, label, _ in STAGES}


class ProgressWriter:
    """Writes progress to the job row, at most about once a second."""

    def __init__(self, engine: Engine, job_id: str):
        self.engine, self.job_id = engine, job_id
        self._last = 0.0
        self._stage = ""

    def __call__(self, stage: str, fraction: float, message: str) -> None:
        now = time.monotonic()
        if stage == self._stage and now - self._last < 1.0 and fraction < 1.0:
            return
        self._stage, self._last = stage, now
        with Session(self.engine) as s:
            job = s.get(Job, self.job_id)
            if job is None:
                return
            job.stage = stage
            job.message = message or STAGE_LABELS.get(stage, stage)
            job.progress = overall_progress(stage, fraction)
            job.updated_at = utcnow()
            s.add(job)
            s.commit()


def run_job(engine: Engine, settings: Settings, job_id: str) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        source, max_clips, focus, rerun = job.source, job.max_clips, job.focus, job.rerun_select

    paths = job_paths(settings, job_id)
    if rerun:
        paths.clear_selection()
    log.info("job %s started source=%s", job_id, source)
    try:
        result = run_pipeline(source, paths.dir, settings, max_clips=max_clips, focus=focus,
                              report=ProgressWriter(engine, job_id))
    except Exception as e:  # noqa: BLE001 - every failure must be recorded on the job
        log.exception("job %s failed", job_id)
        with Session(engine) as s:
            job = s.get(Job, job_id)
            if job:
                job.status, job.error, job.updated_at = "failed", str(e)[:1000] or type(e).__name__, utcnow()
                job.message = "Failed"
                s.add(job)
                s.commit()
        return

    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        s.exec(delete(Clip).where(Clip.job_id == job_id))
        for c in result.clips:
            m = c.moment
            s.add(Clip(job_id=job_id, idx=c.index, start_s=m.start_s, end_s=m.end_s, title=m.title,
                       hook=m.hook, rationale=m.rationale, score=m.score))
        job.status, job.stage, job.progress = "done", "render", 1.0
        job.message = f"{len(result.clips)} clips ready"
        job.title = result.info.title or job.title
        job.picker = result.selected.picker
        job.notes = json.dumps(result.selected.notes)
        job.rerun_select = False
        job.updated_at = utcnow()
        s.add(job)
        s.commit()
    log.info("job %s done clips=%d picker=%s", job_id, len(result.clips), result.selected.picker)


def cleanup_old_media(engine: Engine, settings: Settings) -> None:
    """Delete the big source files of finished jobs older than the retention period. Clips stay."""
    cutoff = utcnow() - timedelta(days=settings.retention_days)
    with Session(engine) as s:
        old = s.exec(select(Job).where(Job.status.in_(["done", "failed"]), Job.updated_at < cutoff,
                                       Job.source_deleted == False)).all()  # noqa: E712
        for job in old:
            job_paths(settings, job.id).delete_source_media()
            job.source_deleted = True
            s.add(job)
            log.info("deleted source media for job %s", job.id)
        s.commit()


def worker_loop(settings: Settings, *, poll_s: float = 1.0, once: bool = False) -> None:
    engine = get_engine(settings.db_path)
    n = requeue_stale(engine)
    if n:
        log.info("requeued %d interrupted job(s)", n)
    last_cleanup = 0.0
    log.info("worker ready; data dir %s", settings.data_dir.resolve())
    while True:
        if time.monotonic() - last_cleanup > 3600:
            cleanup_old_media(engine, settings)
            last_cleanup = time.monotonic()
        job_id = claim_next_job(engine)
        if job_id:
            run_job(engine, settings, job_id)
        elif once:
            return
        else:
            time.sleep(poll_s)
