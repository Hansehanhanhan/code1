from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from agent.agent import run_agent
from backend.governance import build_degraded_response, is_timeout_error, run_with_governance
from backend.models import RunRequest
from backend.settings import Settings

logger = logging.getLogger("merchant_ops.job_queue")

TERMINAL_STATUSES = {"succeeded", "failed", "degraded", "cancelled"}


def _coerce_event_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if content is None:
        return {}
    return {"message": str(content)}


class SqliteJobStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._lock = RLock()
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT,
                    query TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    session_id TEXT,
                    response_json TEXT,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id
                ON job_events(job_id, id);
                """
            )
            columns = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "idempotency_key" not in columns:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN idempotency_key TEXT")
            self._conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key_unique
                ON jobs(idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Job store not initialized")
        return self._conn

    def create_job(
        self,
        *,
        job_id: str,
        request_id: str,
        idempotency_key: str | None,
        query: str,
        context: dict[str, Any],
        session_id: str | None,
        created_at: float,
    ) -> None:
        with self._lock:
            conn = self._require_conn()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, request_id, status, idempotency_key, query, context_json, session_id, response_json, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    job_id,
                    request_id,
                    "queued",
                    idempotency_key,
                    query,
                    json.dumps(context, ensure_ascii=False),
                    session_id,
                    created_at,
                    created_at,
                ),
            )
            conn.commit()

    def set_status(self, job_id: str, status: str, *, error_message: str | None = None) -> None:
        with self._lock:
            conn = self._require_conn()
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? WHERE job_id = ?",
                (status, error_message, time(), job_id),
            )
            conn.commit()

    def set_status_if_in(
        self,
        job_id: str,
        status: str,
        *,
        allowed_current_statuses: tuple[str, ...],
        error_message: str | None = None,
    ) -> bool:
        if not allowed_current_statuses:
            return False
        placeholders = ",".join("?" for _ in allowed_current_statuses)
        with self._lock:
            conn = self._require_conn()
            cursor = conn.execute(
                f"""
                UPDATE jobs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE job_id = ? AND status IN ({placeholders})
                """,
                (status, error_message, time(), job_id, *allowed_current_statuses),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0

    def set_response(self, job_id: str, response: dict[str, Any], *, status: str) -> None:
        with self._lock:
            conn = self._require_conn()
            conn.execute(
                "UPDATE jobs SET status = ?, response_json = ?, updated_at = ?, error_message = NULL WHERE job_id = ?",
                (status, json.dumps(response, ensure_ascii=False), time(), job_id),
            )
            conn.commit()

    def append_event(self, job_id: str, event_type: str, content: dict[str, Any]) -> int:
        with self._lock:
            conn = self._require_conn()
            cursor = conn.execute(
                "INSERT INTO job_events (job_id, event_type, content_json, created_at) VALUES (?, ?, ?, ?)",
                (job_id, event_type, json.dumps(content, ensure_ascii=False), time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._require_conn()
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "job_id": str(row["job_id"]),
            "request_id": str(row["request_id"]),
            "status": str(row["status"]),
            "idempotency_key": row["idempotency_key"],
            "query": str(row["query"]),
            "context": json.loads(str(row["context_json"] or "{}")),
            "session_id": row["session_id"],
            "response": json.loads(str(row["response_json"])) if row["response_json"] else None,
            "error_message": row["error_message"],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def list_events_since(
        self,
        job_id: str,
        *,
        last_event_id: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT id, event_type, content_json, created_at
                FROM job_events
                WHERE job_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (job_id, max(0, last_event_id), max(1, limit)),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "type": str(row["event_type"]),
                "content": json.loads(str(row["content_json"] or "{}")),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_jobs_by_statuses(self, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                f"""
                SELECT *
                FROM jobs
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                """,
                statuses,
            ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            jobs.append(
                {
                    "job_id": str(row["job_id"]),
                    "request_id": str(row["request_id"]),
                    "status": str(row["status"]),
                    "idempotency_key": row["idempotency_key"],
                    "query": str(row["query"]),
                    "context": json.loads(str(row["context_json"] or "{}")),
                    "session_id": row["session_id"],
                    "response": json.loads(str(row["response_json"])) if row["response_json"] else None,
                    "error_message": row["error_message"],
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                }
            )
        return jobs

    def get_job_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        normalized = (idempotency_key or "").strip()
        if not normalized:
            return None
        with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": str(row["job_id"]),
            "request_id": str(row["request_id"]),
            "status": str(row["status"]),
            "idempotency_key": row["idempotency_key"],
            "query": str(row["query"]),
            "context": json.loads(str(row["context_json"] or "{}")),
            "session_id": row["session_id"],
            "response": json.loads(str(row["response_json"])) if row["response_json"] else None,
            "error_message": row["error_message"],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }


class JobQueueRunner:
    def __init__(self, settings: Settings, store: SqliteJobStore) -> None:
        self._settings = settings
        self._store = store
        self._queue: Queue[str | None] = Queue()
        self._worker_thread: Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._store.init()
        self._recover_incomplete_jobs()
        self._started = True
        self._worker_thread = Thread(target=self._worker_loop, name="job-queue-worker", daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None
        self._store.close()

    def submit(self, request: RunRequest, *, request_id: str) -> dict[str, Any]:
        idempotency_key = (request.idempotency_key or "").strip() or None
        if idempotency_key:
            existing = self._store.get_job_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._store.append_event(
                    str(existing["job_id"]),
                    "idempotent_reused",
                    {"request_id": request_id, "idempotency_key": idempotency_key},
                )
                return existing

        created_at = time()
        job_id = str(uuid4())
        self._store.create_job(
            job_id=job_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            query=request.query,
            context=request.context,
            session_id=request.session_id,
            created_at=created_at,
        )
        self._queue.put(job_id)
        job = self._store.get_job(job_id)
        if job is None:
            raise RuntimeError("Failed to create job")
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._store.get_job(job_id)

    def list_events_since(self, job_id: str, *, last_event_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return self._store.list_events_since(job_id, last_event_id=last_event_id, limit=limit)

    def _recover_incomplete_jobs(self) -> None:
        recovered_running = 0
        recovered_cancel_requested = 0
        requeued = 0

        # Crash recovery: running jobs are considered incomplete and should be retried.
        running_jobs = self._store.list_jobs_by_statuses(("running",))
        for job in running_jobs:
            job_id = str(job["job_id"])
            self._store.set_status(
                job_id,
                "queued",
                error_message="Recovered after restart from running state.",
            )
            self._store.append_event(
                job_id,
                "recovered",
                {"from": "running", "to": "queued", "message": "Recovered running job after restart."},
            )
            recovered_running += 1

        # Pending cancellation is finalized to avoid hanging in non-terminal state.
        cancel_requested_jobs = self._store.list_jobs_by_statuses(("cancel_requested",))
        for job in cancel_requested_jobs:
            job_id = str(job["job_id"])
            self._store.set_status(
                job_id,
                "cancelled",
                error_message="Recovered after restart from cancel_requested state.",
            )
            self._store.append_event(
                job_id,
                "cancelled",
                {"message": "Recovered cancellation after restart."},
            )
            recovered_cancel_requested += 1

        queued_jobs = self._store.list_jobs_by_statuses(("queued",))
        for job in queued_jobs:
            self._queue.put(str(job["job_id"]))
            requeued += 1

        if recovered_running or recovered_cancel_requested or requeued:
            logger.info(
                json.dumps(
                    {
                        "event": "job_recovery_completed",
                        "requeued": requeued,
                        "running_to_queued": recovered_running,
                        "cancel_requested_to_cancelled": recovered_cancel_requested,
                    },
                    ensure_ascii=False,
                )
            )

    def cancel(self, job_id: str) -> tuple[dict[str, Any], bool, str] | None:
        job = self._store.get_job(job_id)
        if job is None:
            return None

        status = str(job["status"])
        if status in TERMINAL_STATUSES:
            return job, False, f"Job already terminal: {status}"

        if status == "queued":
            changed = self._store.set_status_if_in(
                job_id,
                "cancelled",
                allowed_current_statuses=("queued",),
                error_message="Cancelled by user before execution.",
            )
            if changed:
                self._store.append_event(job_id, "cancelled", {"message": "Job cancelled before execution."})
                latest = self._store.get_job(job_id)
                if latest is not None:
                    return latest, True, "Job cancelled."
            latest = self._store.get_job(job_id)
            if latest is None:
                return None
            return latest, False, f"Job status changed concurrently to {latest['status']}"

        if status == "running":
            changed = self._store.set_status_if_in(
                job_id,
                "cancel_requested",
                allowed_current_statuses=("running",),
                error_message="Cancel requested by user; waiting for current attempt to return.",
            )
            if changed:
                self._store.append_event(
                    job_id,
                    "cancel_requested",
                    {"message": "Cancel requested. The running attempt will stop after current execution returns."},
                )
                latest = self._store.get_job(job_id)
                if latest is not None:
                    return latest, True, "Cancel requested."
            latest = self._store.get_job(job_id)
            if latest is None:
                return None
            return latest, False, f"Job status changed concurrently to {latest['status']}"

        if status == "cancel_requested":
            return job, False, "Cancel already requested."

        return job, False, f"Job cannot be cancelled in status: {status}"

    def retry(self, job_id: str, *, request_id: str) -> dict[str, Any] | None:
        job = self._store.get_job(job_id)
        if job is None:
            return None

        status = str(job["status"])
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Only terminal jobs can be retried, current status: {status}")

        request = RunRequest(
            query=str(job["query"]),
            context=dict(job["context"] or {}),
            session_id=(str(job["session_id"]) if job.get("session_id") else None),
        )
        retried = self.submit(request, request_id=request_id)
        self._store.append_event(
            job_id,
            "retried",
            {"new_job_id": str(retried["job_id"]), "new_request_id": request_id},
        )
        return retried

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                asyncio.run(self._process_job(job_id))
            finally:
                self._queue.task_done()

    async def _process_job(self, job_id: str) -> None:
        job = self._store.get_job(job_id)
        if job is None:
            return
        status = str(job["status"])
        if status == "cancelled":
            return
        if status == "cancel_requested":
            self._store.set_status(job_id, "cancelled", error_message="Cancelled before worker execution.")
            self._store.append_event(job_id, "cancelled", {"message": "Job cancelled before worker execution."})
            return
        if status != "queued":
            return

        request = RunRequest(
            query=str(job["query"]),
            context=dict(job["context"] or {}),
            session_id=job["session_id"],
        )
        request_id = str(job["request_id"])
        self._store.set_status(job_id, "running")
        started_at = perf_counter()
        stream_state: dict[str, Any] = {
            "first_event_at": None,
            "event_count": 0,
            "has_final_response": False,
            "has_error": False,
        }

        def emit(event: dict[str, Any]) -> None:
            event_type = str(event.get("type", "unknown"))
            content = _coerce_event_content(event.get("content"))
            if stream_state["first_event_at"] is None:
                stream_state["first_event_at"] = perf_counter()
            stream_state["event_count"] = int(stream_state["event_count"]) + 1
            if event_type == "final_response":
                stream_state["has_final_response"] = True
            if event_type == "error":
                stream_state["has_error"] = True
            self._store.append_event(job_id, event_type, content)

        def _invoke():
            return run_agent(
                request.query,
                request.context,
                request.session_id,
                event_sink=emit,
                request_id=request_id,
            )

        try:
            response, attempts_used = await run_with_governance(
                _invoke,
                timeout_seconds=self._settings.request_timeout_seconds,
                retry_attempts=self._settings.run_retry_attempts,
                retry_backoff_ms=self._settings.retry_backoff_ms,
            )
            response_payload = response.model_dump()
            emit({"type": "final_response", "content": response_payload})
            latency_ms = max(0, int((perf_counter() - started_at) * 1000))
            ttfb_ms = (
                max(0, int((float(stream_state["first_event_at"]) - started_at) * 1000))
                if stream_state["first_event_at"] is not None
                else latency_ms
            )
            event_count = int(stream_state["event_count"])
            event_completeness = bool(stream_state["has_final_response"]) and not bool(stream_state["has_error"])
            stream_metrics = {
                "ttfb_ms": ttfb_ms,
                "event_count": event_count,
                "event_completeness": event_completeness,
                "attempts_used": attempts_used,
            }
            emit({"type": "stream_metrics", "content": stream_metrics})
            merged_metrics = dict(response_payload.get("metrics") or {})
            merged_metrics.update(stream_metrics)
            response_payload["metrics"] = merged_metrics
            latest_job = self._store.get_job(job_id)
            if latest_job is not None and str(latest_job["status"]) == "cancel_requested":
                self._store.set_status(job_id, "cancelled", error_message="Cancelled by user after execution returned.")
                self._store.append_event(
                    job_id,
                    "cancelled",
                    {"message": "Job execution returned but response was discarded due to cancellation."},
                )
                return
            self._store.set_response(job_id, response_payload, status="succeeded")
            logger.info(
                json.dumps(
                    {
                        "event": "job_succeeded",
                        "job_id": job_id,
                        "request_id": request_id,
                        "latency_ms": latency_ms,
                        "attempts_used": attempts_used,
                    },
                    ensure_ascii=False,
                )
            )
            return
        except Exception as exc:  # noqa: BLE001
            timeout_error = is_timeout_error(exc)
            should_degrade = self._settings.degrade_on_timeout if timeout_error else self._settings.degrade_on_error
            latency_ms = max(0, int((perf_counter() - started_at) * 1000))
            ttfb_ms = (
                max(0, int((float(stream_state["first_event_at"]) - started_at) * 1000))
                if stream_state["first_event_at"] is not None
                else latency_ms
            )
            if should_degrade:
                reason = "timeout" if timeout_error else f"error:{type(exc).__name__}"
                degraded = build_degraded_response(
                    request.query,
                    request.context,
                    request.session_id or "default",
                    reason=reason,
                    latency_ms=latency_ms,
                )
                payload = degraded.model_dump()
                emit({"type": "degraded_response", "content": {"reason": reason}})
                emit({"type": "final_response", "content": payload})
                event_count = int(stream_state["event_count"])
                event_completeness = bool(stream_state["has_final_response"]) and not bool(stream_state["has_error"])
                stream_metrics = {
                    "ttfb_ms": ttfb_ms,
                    "event_count": event_count,
                    "event_completeness": event_completeness,
                    "degraded": True,
                }
                emit({"type": "stream_metrics", "content": stream_metrics})
                payload_metrics = dict(payload.get("metrics") or {})
                payload_metrics.update(stream_metrics)
                payload["metrics"] = payload_metrics
                latest_job = self._store.get_job(job_id)
                if latest_job is not None and str(latest_job["status"]) == "cancel_requested":
                    self._store.set_status(job_id, "cancelled", error_message="Cancelled by user after degraded run.")
                    self._store.append_event(
                        job_id,
                        "cancelled",
                        {"message": "Job degraded response discarded due to cancellation."},
                    )
                    return
                self._store.set_response(job_id, payload, status="degraded")
                return

            emit({"type": "error", "content": str(exc)})
            event_count = int(stream_state["event_count"])
            event_completeness = bool(stream_state["has_final_response"]) and not bool(stream_state["has_error"])
            emit(
                {
                    "type": "stream_metrics",
                    "content": {
                        "ttfb_ms": ttfb_ms,
                        "event_count": event_count,
                        "event_completeness": event_completeness,
                    },
                }
            )
            latest_job = self._store.get_job(job_id)
            if latest_job is not None and str(latest_job["status"]) == "cancel_requested":
                self._store.set_status(job_id, "cancelled", error_message="Cancelled by user after failed run.")
                self._store.append_event(
                    job_id,
                    "cancelled",
                    {"message": "Job failed but marked cancelled due to user cancellation."},
                )
                return
            self._store.set_status(job_id, "failed", error_message=str(exc))


_runner_lock = RLock()
_cached_runner: JobQueueRunner | None = None
_cached_runner_fingerprint: tuple[str, str, int, int, int] | None = None


def _runner_fingerprint(settings: Settings) -> tuple[str, str, int, int, int]:
    return (
        settings.job_db_path,
        settings.openai_model,
        settings.request_timeout_seconds,
        settings.run_retry_attempts,
        settings.retry_backoff_ms,
    )


def get_job_runner(settings: Settings) -> JobQueueRunner:
    global _cached_runner
    global _cached_runner_fingerprint
    fp = _runner_fingerprint(settings)
    with _runner_lock:
        if _cached_runner is not None and _cached_runner_fingerprint == fp:
            return _cached_runner
        store = SqliteJobStore(settings.job_db_path)
        _cached_runner = JobQueueRunner(settings, store)
        _cached_runner_fingerprint = fp
        return _cached_runner
