from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.job_queue import JobQueueRunner, SqliteJobStore
from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings


def make_settings(job_db_path: str, **overrides: Any) -> Settings:
    base = Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.example.com",
        openai_model="test-model",
        allow_rule_fallback=True,
        rag_enabled=False,
        rag_docs_dir="knowledge/seed",
        rag_vector_backend="chroma",
        rag_top_k=3,
        rag_fetch_k=12,
        rag_embedding_model="BAAI/bge-small-zh-v1.5",
        rag_embedding_device="cpu",
        session_backend="memory",
        session_ttl_seconds=3600,
        redis_url=None,
        rate_limit_enabled=True,
        rate_limit_window_seconds=60,
        rate_limit_max_requests=30,
        rate_limit_max_requests_run=20,
        rate_limit_max_requests_stream=10,
        rate_limit_max_requests_ip=60,
        trust_x_forwarded_for=False,
        trusted_proxy_ips=[],
        request_timeout_seconds=120,
        request_timeout_seconds_stream=150,
        run_retry_attempts=0,
        retry_backoff_ms=0,
        degrade_on_timeout=True,
        degrade_on_error=True,
        app_auth_enabled=False,
        app_api_key=None,
        max_query_chars=2000,
        max_context_chars=8000,
        prompt_injection_guard_enabled=True,
        app_cors_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        app_cors_allow_credentials=False,
        agent_verbose=False,
        job_db_path=job_db_path,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


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
        created_at=now,
    )
    store.create_job(
        job_id="job-running",
        request_id="req-running",
        idempotency_key=None,
        query="running query",
        context={},
        session_id="s2",
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
            event_sink({"type": "agent_action", "content": {"loop_index": 1, "action": "traffic_analyze"}})
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
