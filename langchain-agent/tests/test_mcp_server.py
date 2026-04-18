from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi import HTTPException

from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings
from mcp_server import server as mcp_server


def make_settings(**overrides: Any) -> Settings:
    base = Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.example.com",
        openai_model="test-model",
        allow_rule_fallback=True,
        rag_enabled=True,
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
        run_retry_attempts=1,
        retry_backoff_ms=300,
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
