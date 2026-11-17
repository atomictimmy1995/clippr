"""FastAPI app: HTML pages (Jinja2 + HTMX) and a small JSON API over the same job actions."""

from __future__ import annotations

import re
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import Session

from clippr.config import Settings, get_settings
from clippr.db import Job, get_engine, job_clips, recent_jobs
from clippr.jobs import JobError, create_job, delete_job, job_paths, regenerate
from clippr.pipeline.moments import PICKER_LABELS, picker_chain
from clippr.pipeline.runner import STAGES

WEB_DIR = Path(__file__).resolve().parent
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CLIP_FILE_RE = re.compile(r"^clip_(\d{2})\.(mp4|srt|jpg)$")
MEDIA_TYPES = {"mp4": "video/mp4", "srt": "application/x-subrip", "jpg": "image/jpeg"}


class NewJob(BaseModel):
    url: str
    max_clips: int | None = None
    focus: str | None = None
    confirm_rights: bool = False


class Regenerate(BaseModel):
    focus: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = get_engine(settings.db_path)
    app = FastAPI(title="Clippr")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    templates.env.globals.update(stage_labels={n: label for n, label, _ in STAGES}, picker_labels=PICKER_LABELS)

    picker_cache: dict[str, tuple[float, str]] = {}

    def default_picker() -> str:
        cached = picker_cache.get("p")
        if cached and time.monotonic() - cached[0] < 30:
            return cached[1]
        name = picker_chain(settings)[0]
        picker_cache["p"] = (time.monotonic(), name)
        return name

    def load_job(session: Session, job_id: str) -> Job:
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def job_json(session: Session, job: Job) -> dict:
        return {
            "id": job.id, "source": job.source, "title": job.title, "status": job.status,
            "stage": job.stage, "message": job.message, "progress": job.progress, "error": job.error,
            "picker": job.picker, "notes": job.notes_list, "focus": job.focus, "max_clips": job.max_clips,
            "clips": [
                {
                    "index": c.idx, "start_s": c.start_s, "end_s": c.end_s, "title": c.title,
                    "hook": c.hook, "rationale": c.rationale, "score": c.score,
                    "video_url": f"/jobs/{job.id}/clips/clip_{c.idx:02d}.mp4",
                    "srt_url": f"/jobs/{job.id}/clips/clip_{c.idx:02d}.srt",
                    "thumbnail_url": f"/jobs/{job.id}/clips/clip_{c.idx:02d}.jpg",
                }
                for c in job_clips(session, job.id)
            ],
        }

    # ---------- HTML pages ----------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with Session(engine) as s:
            jobs = recent_jobs(s)
        return templates.TemplateResponse(request, "index.html", {
            "jobs": jobs, "settings": settings, "picker": default_picker(), "error": None, "form": {},
        })

    @app.post("/jobs")
    def submit(request: Request, url: str = Form(...), max_clips: int = Form(5), focus: str = Form(""),
               rights: bool = Form(False)):
        error = None
        if not rights:
            error = "Please confirm you own this video or have the rights to clip it."
        else:
            try:
                job = create_job(engine, settings, url, max_clips=max_clips, focus=focus)
                return RedirectResponse(f"/jobs/{job.id}", status_code=303)
            except JobError as e:
                error = str(e)
        with Session(engine) as s:
            jobs = recent_jobs(s)
        return templates.TemplateResponse(request, "index.html", {
            "jobs": jobs, "settings": settings, "picker": default_picker(), "error": error,
            "form": {"url": url, "max_clips": max_clips, "focus": focus},
        }, status_code=400)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str):
        with Session(engine) as s:
            job = load_job(s, job_id)
            clips = job_clips(s, job_id)
        return templates.TemplateResponse(request, "job.html", {"job": job, "clips": clips})

    @app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
    def job_status(request: Request, job_id: str):
        with Session(engine) as s:
            job = load_job(s, job_id)
            clips = job_clips(s, job_id)
        return templates.TemplateResponse(request, "_status.html", {"job": job, "clips": clips})

    @app.post("/jobs/{job_id}/regenerate")
    def regenerate_page(job_id: str, focus: str = Form("")):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        try:
            regenerate(engine, settings, job_id, focus=focus)
        except JobError as e:
            raise HTTPException(409, str(e)) from e
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/jobs/{job_id}/retry")
    def retry_page(job_id: str):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        try:
            regenerate(engine, settings, job_id, reselect=False)
        except JobError as e:
            raise HTTPException(409, str(e)) from e
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/jobs/{job_id}/delete")
    def delete_page(job_id: str):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        try:
            delete_job(engine, settings, job_id)
        except JobError as e:
            raise HTTPException(409, str(e)) from e
        return RedirectResponse("/", status_code=303)

    @app.get("/jobs/{job_id}/clips/{name}")
    def clip_file(job_id: str, name: str, download: bool = False):
        m = CLIP_FILE_RE.match(name)
        if not JOB_ID_RE.match(job_id) or not m:
            raise HTTPException(404, "File not found")
        clips_dir = job_paths(settings, job_id).clips.resolve()
        path = (clips_dir / name).resolve()
        if path.parent != clips_dir or not path.is_file():
            raise HTTPException(404, "File not found")
        filename = f"clippr-{job_id[:8]}-{name}" if download else None
        return FileResponse(path, media_type=MEDIA_TYPES[m.group(2)], filename=filename)

    # ---------- JSON API ----------

    @app.post("/api/jobs", status_code=201)
    def api_create(body: NewJob):
        if not body.confirm_rights:
            raise HTTPException(400, "Set confirm_rights to true to confirm you may clip this video")
        try:
            job = create_job(engine, settings, body.url, max_clips=body.max_clips, focus=body.focus)
        except JobError as e:
            raise HTTPException(400, str(e)) from e
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def api_get(job_id: str):
        with Session(engine) as s:
            return job_json(s, load_job(s, job_id))

    @app.post("/api/jobs/{job_id}/regenerate")
    def api_regenerate(job_id: str, body: Regenerate | None = None):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        try:
            regenerate(engine, settings, job_id, focus=body.focus if body else None)
        except JobError as e:
            raise HTTPException(409, str(e)) from e
        return {"job_id": job_id}

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def api_delete(job_id: str):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Job not found")
        try:
            delete_job(engine, settings, job_id)
        except JobError as e:
            raise HTTPException(409 if "finish" in str(e) else 404, str(e)) from e

    @app.get("/healthz")
    def healthz():
        with Session(engine) as s:
            recent_jobs(s, limit=1)
        return {"ok": True}

    return app
