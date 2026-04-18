from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from threading import RLock
from time import perf_counter
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent.agent import run_agent
from backend.governance import build_degraded_response, is_timeout_error, run_with_governance
from backend.models import RunRequest, RunResponse
from backend.rate_limit import get_rate_limiter
from backend.security import ensure_request_auth_from_key, validate_request_security
from backend.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("merchant_ops.backend")
_error_stats_lock = RLock()
_error_stats: Counter[str] = Counter()
_stability_stats_lock = RLock()
_stability_stats: Counter[str] = Counter()


def _log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def _log_request_started(endpoint: str, request_id: str, session_id: str, context: dict[str, object]) -> None:
    _log_event(
        "request_started",
        endpoint=endpoint,
        request_id=request_id,
        session_id=session_id,
        context_keys=sorted(context.keys()),
    )


def _log_request_finished(
    endpoint: str,
    request_id: str,
    session_id: str,
    status: str,
    latency_ms: int,
    *,
    fallback_used: bool | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    extra_fields: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "endpoint": endpoint,
        "request_id": request_id,
        "session_id": session_id,
        "status": status,
        "latency_ms": latency_ms,
    }
    if fallback_used is not None:
        payload["fallback_used"] = fallback_used
    if error_type:
        payload["error_type"] = error_type
    if error_message:
        payload["error_message"] = error_message
    if extra_fields:
        payload.update(extra_fields)
    _log_event("request_finished", **payload)


def _record_error(endpoint: str, error_type: str) -> None:
    with _error_stats_lock:
        _error_stats[error_type] += 1
        _error_stats[f"{endpoint}:{error_type}"] += 1


def _error_stats_snapshot() -> dict[str, int]:
    with _error_stats_lock:
        return dict(_error_stats)


def _record_stability(metric: str, endpoint: str) -> None:
    with _stability_stats_lock:
        _stability_stats[metric] += 1
        _stability_stats[f"{endpoint}:{metric}"] += 1


def _stability_stats_snapshot() -> dict[str, int]:
    with _stability_stats_lock:
        return dict(_stability_stats)


def _build_cors_config(current_settings: Settings) -> tuple[list[str], bool]:
    origins = list(current_settings.app_cors_origins or [])
    if not origins:
        origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
    allow_credentials = bool(current_settings.app_cors_allow_credentials)
    if "*" in origins and allow_credentials:
        allow_credentials = False
        _log_event(
            "cors_credentials_downgraded",
            reason="wildcard_origin_with_credentials_is_not_safe",
        )
    return origins, allow_credentials


settings = Settings.from_env()
_cors_origins, _cors_allow_credentials = _build_cors_config(settings)

app = FastAPI(
    title="Merchant Ops Copilot (LangChain ReAct)",
    description="LangChain ReAct backend for merchant operations assistant.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_api_key(current_settings: Settings) -> None:
    if not current_settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")


def _endpoint_rate_limit_max(current_settings: Settings, endpoint: str) -> int:
    if endpoint == "run_stream":
        return current_settings.rate_limit_max_requests_stream
    if endpoint == "run":
        return current_settings.rate_limit_max_requests_run
    return current_settings.rate_limit_max_requests


def _resolve_client_ip(request: Request, current_settings: Settings) -> str:
    direct_ip = request.client.host if request.client else "unknown"
    if not current_settings.trust_x_forwarded_for:
        return direct_ip

    trusted_proxy_ips = {item.strip() for item in current_settings.trusted_proxy_ips if item.strip()}
    if trusted_proxy_ips and direct_ip not in trusted_proxy_ips:
        return direct_ip

    forwarded_for = request.headers.get("x-forwarded-for", "")
    first = forwarded_for.split(",")[0].strip() if forwarded_for else ""
    return first or direct_ip


def _build_rate_limit_keys(
    request: Request,
    session_id: str | None,
    endpoint: str,
    current_settings: Settings,
) -> tuple[str, str]:
    client_ip = _resolve_client_ip(request, current_settings)
    sid = (session_id or "").strip() or "default"
    key_with_session = f"endpoint:{endpoint}|ip:{client_ip}|sid:{sid}"
    key_ip_only = f"endpoint:{endpoint}|ip:{client_ip}"
    return key_with_session, key_ip_only


def _validate_run_request_security(
    endpoint: str,
    request_id: str,
    http_request: Request,
    request: RunRequest,
    current_settings: Settings,
) -> None:
    try:
        validate_request_security(
            request,
            current_settings,
            provided_api_key=http_request.headers.get("x-api-key"),
        )
    except HTTPException as exc:
        error_type = f"HTTP{exc.status_code}"
        _record_error(endpoint, error_type)
        _record_stability("input_rejected_total", endpoint)
        _log_event(
            "request_rejected",
            endpoint=endpoint,
            request_id=request_id,
            status_code=exc.status_code,
            reason=str(exc.detail),
        )
        raise


def _check_rate_limit(
    request: Request,
    session_id: str | None,
    current_settings: Settings,
    request_id: str,
    endpoint: str,
) -> tuple[int, int]:
    limiter = get_rate_limiter(current_settings)
    endpoint_limit = _endpoint_rate_limit_max(current_settings, endpoint)
    ip_limit = max(1, current_settings.rate_limit_max_requests_ip)
    key_with_session, key_ip_only = _build_rate_limit_keys(request, session_id, endpoint, current_settings)

    allowed_session, remaining_session = limiter.allow(key_with_session, max_requests=endpoint_limit)
    allowed_ip, remaining_ip = limiter.allow(key_ip_only, max_requests=ip_limit)
    allowed = allowed_session and allowed_ip
    remaining = min(remaining_session, remaining_ip)

    if allowed:
        _log_event(
            "rate_limit_passed",
            request_id=request_id,
            key=key_with_session,
            ip_key=key_ip_only,
            limit=endpoint_limit,
            ip_limit=ip_limit,
            remaining=remaining,
        )
        return max(0, remaining), endpoint_limit

    failed_bucket = "session_bucket" if not allowed_session else "ip_bucket"
    _log_event(
        "rate_limit_exceeded",
        request_id=request_id,
        key=key_with_session,
        ip_key=key_ip_only,
        limit=endpoint_limit,
        ip_limit=ip_limit,
        remaining=remaining,
        failed_bucket=failed_bucket,
    )
    _record_error(endpoint, "RateLimitExceeded")
    _record_stability("rate_limit_exceeded_total", endpoint)
    raise HTTPException(
        status_code=429,
        detail="Too many requests, please retry later.",
        headers={
            "Retry-After": str(current_settings.rate_limit_window_seconds),
            "X-RateLimit-Limit": str(endpoint_limit),
            "X-RateLimit-Remaining": "0",
        },
    )


async def _run_agent_with_governance(
    request: RunRequest,
    request_id: str,
    *,
    endpoint: str,
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_ms: int,
    event_sink=None,
) -> tuple[RunResponse, int]:
    def _invoke() -> RunResponse:
        return run_agent(
            request.query,
            request.context,
            request.session_id,
            event_sink,
            request_id,
        )

    def _on_attempt_failed(attempt: int, max_attempts: int, retryable: bool, exc: Exception) -> None:
        _log_event(
            "governance_attempt_failed",
            request_id=request_id,
            endpoint=endpoint,
            attempt=attempt,
            max_attempts=max_attempts,
            retryable=retryable,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        if is_timeout_error(exc):
            _record_stability("timeout_total", endpoint)
        if retryable and attempt < max_attempts:
            _record_stability("retry_total", endpoint)

    return await run_with_governance(
        _invoke,
        timeout_seconds=timeout_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_ms=retry_backoff_ms,
        on_attempt_failed=_on_attempt_failed,
    )


def _ensure_metrics_auth(http_request: Request, current_settings: Settings) -> None:
    ensure_request_auth_from_key(http_request.headers.get("x-api-key"), current_settings)


@app.get("/")
async def root() -> dict:
    return {
        "message": "Merchant Ops Copilot (LangChain ReAct)",
        "version": "0.4.0",
        "framework": "LangChain",
        "agent_type": "ReAct",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/metrics/error_types")
async def metrics_error_types(http_request: Request) -> dict[str, dict[str, int]]:
    current_settings = Settings.from_env()
    _ensure_metrics_auth(http_request, current_settings)
    return {"error_counts": _error_stats_snapshot()}


@app.get("/metrics/stability")
async def metrics_stability(http_request: Request) -> dict[str, dict[str, int]]:
    current_settings = Settings.from_env()
    _ensure_metrics_auth(http_request, current_settings)
    return {"stability_counts": _stability_stats_snapshot()}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest, http_request: Request, http_response: Response) -> RunResponse:
    current_settings = Settings.from_env()
    _ensure_api_key(current_settings)

    request_id = str(uuid4())
    _validate_run_request_security("run", request_id, http_request, request, current_settings)
    session_id = request.session_id or "default"
    remaining, limit = _check_rate_limit(http_request, request.session_id, current_settings, request_id, "run")
    http_response.headers["X-RateLimit-Limit"] = str(limit)
    http_response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    started_at = perf_counter()
    _log_request_started("run", request_id, session_id, request.context or {})

    try:
        response, attempts_used = await _run_agent_with_governance(
            request,
            request_id,
            endpoint="run",
            timeout_seconds=current_settings.request_timeout_seconds,
            retry_attempts=current_settings.run_retry_attempts,
            retry_backoff_ms=current_settings.retry_backoff_ms,
            event_sink=None,
        )
        latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        _log_request_finished(
            "run",
            request_id,
            session_id,
            "success",
            latency_ms,
            fallback_used=response.metrics.fallback_used,
            extra_fields={
                "llm_latency_ms": response.metrics.llm_latency_ms,
                "tool_latency_ms": response.metrics.tool_latency_ms,
                "loop_count": response.metrics.loop_count,
                "retrieve_hits": response.metrics.retrieve_hits,
                "attempts_used": attempts_used,
                "degraded": False,
            },
        )
        http_response.headers["X-Attempts-Used"] = str(attempts_used)
        return response
    except Exception as exc:  # noqa: BLE001
        latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        _record_error("run", type(exc).__name__)
        timeout_error = is_timeout_error(exc)
        should_degrade = current_settings.degrade_on_timeout if timeout_error else current_settings.degrade_on_error
        if should_degrade:
            reason = "timeout" if timeout_error else f"error:{type(exc).__name__}"
            _record_stability("degraded_total", "run")
            degraded = build_degraded_response(
                request.query,
                request.context or {},
                session_id,
                reason=reason,
                latency_ms=latency_ms,
            )
            _log_request_finished(
                "run",
                request_id,
                session_id,
                "degraded",
                latency_ms,
                fallback_used=True,
                error_type=type(exc).__name__,
                error_message=str(exc),
                extra_fields={
                    "degraded": True,
                    "degrade_reason": reason,
                    "error_counts": _error_stats_snapshot(),
                },
            )
            http_response.headers["X-Degraded"] = "1"
            return degraded

        _log_request_finished(
            "run",
            request_id,
            session_id,
            "error",
            latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
            extra_fields={"error_counts": _error_stats_snapshot()},
        )
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc


@app.post("/run_stream")
async def run_stream(request: RunRequest, http_request: Request) -> StreamingResponse:
    current_settings = Settings.from_env()
    _ensure_api_key(current_settings)

    request_id = str(uuid4())
    _validate_run_request_security("run_stream", request_id, http_request, request, current_settings)
    session_id = request.session_id or "default"
    remaining, limit = _check_rate_limit(http_request, request.session_id, current_settings, request_id, "run_stream")
    started_at = perf_counter()
    _log_request_started("run_stream", request_id, session_id, request.context or {})

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stream_state: dict[str, object] = {
        "first_event_at": None,
        "event_count": 0,
        "has_final_response": False,
        "has_error": False,
    }

    def emit(event: dict) -> None:
        if stream_state["first_event_at"] is None:
            stream_state["first_event_at"] = perf_counter()
        stream_state["event_count"] = int(stream_state["event_count"]) + 1
        event_type = str(event.get("type", ""))
        if event_type == "final_response":
            stream_state["has_final_response"] = True
        if event_type == "error":
            stream_state["has_error"] = True
        payload = json.dumps({"request_id": request_id, **event}, ensure_ascii=False)
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    async def worker() -> None:
        try:
            response, attempts_used = await _run_agent_with_governance(
                request,
                request_id,
                endpoint="run_stream",
                timeout_seconds=current_settings.request_timeout_seconds_stream,
                retry_attempts=current_settings.run_retry_attempts,
                retry_backoff_ms=current_settings.retry_backoff_ms,
                event_sink=emit,
            )
            emit({"type": "final_response", "content": response.model_dump()})
            latency_ms = max(0, int((perf_counter() - started_at) * 1000))
            ttfb_ms = (
                max(0, int((float(stream_state["first_event_at"]) - started_at) * 1000))
                if stream_state["first_event_at"] is not None
                else latency_ms
            )
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
            _log_request_finished(
                "run_stream",
                request_id,
                session_id,
                "success",
                latency_ms,
                fallback_used=response.metrics.fallback_used,
                extra_fields={
                    "ttfb_ms": ttfb_ms,
                    "event_count": event_count,
                    "event_completeness": event_completeness,
                    "llm_latency_ms": response.metrics.llm_latency_ms,
                    "tool_latency_ms": response.metrics.tool_latency_ms,
                    "loop_count": response.metrics.loop_count,
                    "retrieve_hits": response.metrics.retrieve_hits,
                    "attempts_used": attempts_used,
                    "degraded": False,
                },
            )
        except Exception as exc:  # noqa: BLE001
            _record_error("run_stream", type(exc).__name__)
            latency_ms = max(0, int((perf_counter() - started_at) * 1000))
            ttfb_ms = (
                max(0, int((float(stream_state["first_event_at"]) - started_at) * 1000))
                if stream_state["first_event_at"] is not None
                else latency_ms
            )
            timeout_error = is_timeout_error(exc)
            should_degrade = current_settings.degrade_on_timeout if timeout_error else current_settings.degrade_on_error
            if should_degrade:
                reason = "timeout" if timeout_error else f"error:{type(exc).__name__}"
                _record_stability("degraded_total", "run_stream")
                degraded = build_degraded_response(
                    request.query,
                    request.context or {},
                    session_id,
                    reason=reason,
                    latency_ms=latency_ms,
                )
                emit({"type": "degraded_response", "content": {"reason": reason}})
                emit({"type": "final_response", "content": degraded.model_dump()})
                event_count = int(stream_state["event_count"])
                event_completeness = bool(stream_state["has_final_response"]) and not bool(stream_state["has_error"])
                emit(
                    {
                        "type": "stream_metrics",
                        "content": {
                            "ttfb_ms": ttfb_ms,
                            "event_count": event_count,
                            "event_completeness": event_completeness,
                            "degraded": True,
                        },
                    }
                )
                _log_request_finished(
                    "run_stream",
                    request_id,
                    session_id,
                    "degraded",
                    latency_ms,
                    fallback_used=True,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    extra_fields={
                        "ttfb_ms": ttfb_ms,
                        "event_count": event_count,
                        "event_completeness": event_completeness,
                        "degraded": True,
                        "degrade_reason": reason,
                        "error_counts": _error_stats_snapshot(),
                    },
                )
            else:
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
                _log_request_finished(
                    "run_stream",
                    request_id,
                    session_id,
                    "error",
                    latency_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    extra_fields={
                        "ttfb_ms": ttfb_ms,
                        "event_count": event_count,
                        "event_completeness": event_completeness,
                        "error_counts": _error_stats_snapshot(),
                    },
                )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_generator():
        task = asyncio.create_task(worker())
        while True:
            data = await queue.get()
            if data is None:
                break
            yield f"data: {data}\n\n"
        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(max(0, remaining)),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

