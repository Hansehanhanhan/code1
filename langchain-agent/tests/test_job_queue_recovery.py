from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.job_queue import JobQueueRunner, SqliteJobStore
from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings
from tests.conftest import make_settings as _base_settings


def make_settings(job_db_path: str, **overrides: Any) -> Settings:
    return _base_settings(
        job_db_path=job_db_path,
        run_retry_attempts=0,
        retry_backoff_ms=0,
        **overrides,
    )


def make_response(answer: str = "ok") -> RunResponse:
    return RunResponse(
        final_answer=answer,
        steps=[
            StepRecord(
                name="Agent",
                input={"query": "q", "context": {}, "session_id": "s1"},
                output={"result": answer},
                duration_ms=10,
            )
        ],
        metrics=Metrics(latency_ms=10, fallback_used=False),
    )


def wait_for_status(runner: JobQueueRunner, job_id: str, terminal: set[str], timeout_s: float = 3.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = runner.get_job(job_id)
        if job is not None:
            status = str(job["status"])
            if status in terminal:
                return status
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for terminal status for job {job_id}")


def seed_jobs(db_path: Path) -> None:
    store = SqliteJobStore(str(db_path))
    store.init()
    now = time.time()
    store.create_job(
        job_id="job-queued",
        request_id="req-queued",
        idempotency_key=None,
        query="queued query",
        context={},
        session_id="s1",
        owner_tenant_id=None,
        owner_user_id=None,
        created_at=now,
    )
    store.create_job(
        job_id="job-running",
        request_id="req-running",
        idempotency_key=None,
        query="running query",
        context={},
        session_id="s2",
        owner_tenant_id=None,
        owner_user_id=None,
        created_at=now + 0.01,
    )
    store.set_status("job-running", "running")
    store.create_job(
        job_id="job-cancel-req",
        request_id="req-cancel",
        idempotency_key=None,
        query="cancel req query",
        context={},
        session_id="s3",
        owner_tenant_id=None,
        owner_user_id=None,
        created_at=now + 0.02,
    )
    store.set_status("job-cancel-req", "cancel_requested", error_message="cancel requested before restart")
    store.close()


def test_runner_recovers_incomplete_jobs_on_start(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "jobs_recovery.db"
    seed_jobs(db_path)

    def fake_run_agent(
        query: str,
        context: dict[str, Any],
        session_id: str | None,
        event_sink=None,
        request_id: str | None = None,
    ) -> RunResponse:
        del context, session_id, request_id
        if event_sink is not None:
            event_sink({"type": "agent_action", "content": {"tool_loop_index": 1, "action": "traffic_analyze"}})
        return make_response(f"done:{query}")

    import backend.job_queue as job_queue

    monkeypatch.setattr(job_queue, "run_agent", fake_run_agent)
    runner = JobQueueRunner(make_settings(str(db_path)), SqliteJobStore(str(db_path)))
    try:
        runner.start()
        status_queued = wait_for_status(runner, "job-queued", {"succeeded", "degraded", "failed"})
        status_running = wait_for_status(runner, "job-running", {"succeeded", "degraded", "failed"})
        status_cancel = wait_for_status(runner, "job-cancel-req", {"cancelled"})

        assert status_queued == "succeeded"
        assert status_running == "succeeded"
        assert status_cancel == "cancelled"

        running_events = runner.list_events_since("job-running")
        assert any(event["type"] == "recovered" for event in running_events)
        cancel_events = runner.list_events_since("job-cancel-req")
        assert any(event["type"] == "cancelled" for event in cancel_events)
    finally:
        runner.stop()
