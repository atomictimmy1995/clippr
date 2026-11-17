"""Job storage in SQLite. The web process creates jobs; the worker process claims and runs them."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, update
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select


def utcnow() -> datetime:
    return datetime.now(UTC)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    source: str
    status: str = Field(default="queued", index=True)  # queued | running | done | failed
    stage: str = ""
    message: str = "Waiting for the worker"
    progress: float = 0.0
    error: str | None = None
    title: str = ""
    max_clips: int = 5
    focus: str | None = None
    picker: str | None = None
    notes: str = "[]"  # JSON list of strings
    rerun_select: bool = False
    source_deleted: bool = False
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def notes_list(self) -> list[str]:
        return json.loads(self.notes or "[]")


class Clip(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    idx: int
    start_s: float
    end_s: float
    title: str
    hook: str
    rationale: str
    score: int

    @property
    def duration(self) -> float:
        return round(self.end_s - self.start_s, 1)


@lru_cache
def get_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record) -> None:  # noqa: ANN001
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # lets the web app read while the worker writes
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    SQLModel.metadata.create_all(engine)
    return engine


def claim_next_job(engine: Engine) -> str | None:
    """Atomically move the oldest queued job to "running". Only one caller can win a given job."""
    oldest = select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(1).scalar_subquery()
    stmt = (
        update(Job)
        .where(Job.id == oldest, Job.status == "queued")
        .values(status="running", message="Starting", error=None, updated_at=utcnow())
        .returning(Job.id)
    )
    with Session(engine) as s:
        row = s.exec(stmt).first()
        s.commit()
    return row[0] if row else None


def requeue_stale(engine: Engine, older_than: timedelta = timedelta(minutes=10)) -> int:
    """Put back jobs left "running" by a worker that died (no progress update for a while)."""
    stmt = (
        update(Job)
        .where(Job.status == "running", Job.updated_at < utcnow() - older_than)
        .values(status="queued", message="Resuming after restart")
    )
    with Session(engine) as s:
        res = s.exec(stmt)
        s.commit()
    return res.rowcount or 0


def recent_jobs(session: Session, limit: int = 20) -> list[Job]:
    return list(session.exec(select(Job).order_by(Job.created_at.desc()).limit(limit)))


def job_clips(session: Session, job_id: str) -> list[Clip]:
    return list(session.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.idx)))
