from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi import HTTPException

from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings
from tests.conftest import make_settings as _base_settings
from mcp_server import server as mcp_server


def make_settings(**overrides: Any) -> Settings:
    return _base_settings(rag_enabled=True, **overrides)


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


def test_mcp_run_agent_requires_api_key_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server.Settings,
        "from_env",
        classmethod(lambda cls: make_settings(app_auth_enabled=True, app_api_key="secret-key")),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(mcp_server._tool_run_agent({"query": "test", "context": {}}))
    assert excinfo.value.status_code == 401


def test_mcp_run_agent_timeout_degrades(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server.Settings,
        "from_env",
        classmethod(
            lambda cls: make_settings(
                request_timeout_seconds=1,
                run_retry_attempts=0,
                degrade_on_timeout=True,
                app_auth_enabled=False,
            )
        ),
    )

    def slow_run_agent(
        query: str,
        context: dict[str, Any],
        session_id: str | None,
        event_sink=None,
        request_id: str | None = None,
    ) -> RunResponse:
        del query, context, session_id, event_sink, request_id
        time.sleep(1.2)
        return make_response("late")

    monkeypatch.setattr(mcp_server, "run_agent", slow_run_agent)
    out = asyncio.run(mcp_server._tool_run_agent({"query": "test", "context": {}}))
    assert out["metrics"]["fallback_used"] is True
    assert out["steps"][0]["name"] == "DegradedFallback"


def test_mcp_retrieve_knowledge_applies_input_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server.Settings,
        "from_env",
        classmethod(lambda cls: make_settings(max_query_chars=4)),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(mcp_server._tool_retrieve_knowledge({"query": "12345", "context": {}}))
    assert excinfo.value.status_code == 413


def test_mcp_rate_limit_applies(monkeypatch) -> None:
    class CountingLimiter:
        def __init__(self) -> None:
            self._counts: dict[str, int] = {}

        def allow(self, key: str, *, max_requests: int | None = None) -> tuple[bool, int]:
            limit = max_requests or 1
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            remaining = max(0, limit - count)
            return count <= limit, remaining

    limiter = CountingLimiter()
    monkeypatch.setattr(
        mcp_server.Settings,
        "from_env",
        classmethod(
            lambda cls: make_settings(
                rate_limit_max_requests_run=1,
                rate_limit_max_requests_ip=1,
                run_retry_attempts=0,
            )
        ),
    )
    monkeypatch.setattr(mcp_server, "get_rate_limiter", lambda _settings: limiter)
    monkeypatch.setattr(mcp_server, "run_agent", lambda *args, **kwargs: make_response("ok"))

    ok = asyncio.run(
        mcp_server._tool_run_agent({"query": "test", "context": {}, "session_id": "s1", "client_id": "c1"})
    )
    assert ok["metrics"]["fallback_used"] is False

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            mcp_server._tool_run_agent({"query": "test", "context": {}, "session_id": "s2", "client_id": "c1"})
        )
    assert excinfo.value.status_code == 429
