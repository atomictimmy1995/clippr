from datetime import timedelta

from sqlmodel import Session

from clippr.db import Job, claim_next_job, get_engine, requeue_stale, utcnow


def test_each_job_is_claimed_once_oldest_first(settings):
    engine = get_engine(settings.db_path)
    with Session(engine) as s:
        a = Job(source="a", created_at=utcnow() - timedelta(minutes=2))
        b = Job(source="b", created_at=utcnow() - timedelta(minutes=1))
        s.add_all([a, b])
        s.commit()
        a_id, b_id = a.id, b.id
    assert claim_next_job(engine) == a_id
    assert claim_next_job(engine) == b_id
    assert claim_next_job(engine) is None
    with Session(engine) as s:
        assert s.get(Job, a_id).status == "running"


def test_requeue_only_stale_running_jobs(settings):
    engine = get_engine(settings.db_path)
    with Session(engine) as s:
        old = Job(source="old", status="running", updated_at=utcnow() - timedelta(hours=1))
        fresh = Job(source="fresh", status="running")
        s.add_all([old, fresh])
        s.commit()
        old_id, fresh_id = old.id, fresh.id
    assert requeue_stale(engine) == 1
    with Session(engine) as s:
        assert s.get(Job, old_id).status == "queued"
        assert s.get(Job, fresh_id).status == "running"
