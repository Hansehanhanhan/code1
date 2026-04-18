from __future__ import annotations

import asyncio
import json
import logging
import sys
from time import perf_counter
from typing import Any, Awaitable, Callable

from agent.agent import run_agent
from backend.governance import build_degraded_response, is_timeout_error, run_with_governance
from backend.models import RunRequest
from backend.rate_limit import get_rate_limiter
from backend.security import validate_request_security
from backend.settings import Settings
from fastapi import HTTPException
from rag import retrieve_knowledge

logger = logging.getLogger("merchant_ops.mcp")


def _to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


def _resolve_mcp_client_id(arguments: dict[str, Any], api_key: str | None) -> str:
    if api_key and isinstance(api_key, str) and api_key.strip():
        # Prefer stable principal when api_key is present.
        return f"apikey:{api_key.strip()[:8]}"
    client_id = arguments.get("client_id")
    if isinstance(client_id, str) and client_id.strip():
        return client_id.strip()
    return "mcp-default"


def _check_mcp_rate_limit(
    *,
    settings: Settings,
    endpoint: str,
    session_id: str | None,
    client_id: str,
) -> None:
    limiter = get_rate_limiter(settings)
    if endpoint == "mcp_run_agent":
        endpoint_limit = settings.rate_limit_max_requests_run
    else:
        endpoint_limit = settings.rate_limit_max_requests
    ip_limit = max(1, settings.rate_limit_max_requests_ip)
    sid = (session_id or "").strip() or "default"
    key_with_session = f"endpoint:{endpoint}|client:{client_id}|sid:{sid}"
    key_client_only = f"endpoint:{endpoint}|client:{client_id}"

    if endpoint_limit <= ip_limit:
        first_label, first_key, first_limit = "session_bucket", key_with_session, endpoint_limit
        second_label, second_key, second_limit = "client_bucket", key_client_only, ip_limit
    else:
        first_label, first_key, first_limit = "client_bucket", key_client_only, ip_limit
        second_label, second_key, second_limit = "session_bucket", key_with_session, endpoint_limit

    allowed_first, _ = limiter.allow(first_key, max_requests=first_limit)
    if not allowed_first:
        raise HTTPException(status_code=429, detail=f"Too many MCP requests ({first_label})")

    allowed_second, _ = limiter.allow(second_key, max_requests=second_limit)
    if not allowed_second:
        raise HTTPException(status_code=429, detail=f"Too many MCP requests ({second_label})")


async def _tool_run_agent(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required and must be non-empty")

    context = arguments.get("context")
    if not isinstance(context, dict):
        context = {}

    session_id = arguments.get("session_id")
    if not isinstance(session_id, str):
        session_id = None
    api_key = arguments.get("api_key")
    if not isinstance(api_key, str):
        api_key = None
    client_id = _resolve_mcp_client_id(arguments, api_key)

    settings = Settings.from_env()
    request = RunRequest(query=query.strip(), context=context, session_id=session_id)
    validate_request_security(request, settings, provided_api_key=api_key)
    _check_mcp_rate_limit(
        settings=settings,
        endpoint="mcp_run_agent",
        session_id=request.session_id,
        client_id=client_id,
    )

    started_at = perf_counter()

    def _invoke():
        return run_agent(request.query, request.context, request.session_id)

    try:
        response, _ = await run_with_governance(
            _invoke,
            timeout_seconds=settings.request_timeout_seconds,
            retry_attempts=settings.run_retry_attempts,
            retry_backoff_ms=settings.retry_backoff_ms,
        )
        return response.model_dump()
    except Exception as exc:  # noqa: BLE001
        timeout_error = is_timeout_error(exc)
        should_degrade = settings.degrade_on_timeout if timeout_error else settings.degrade_on_error
        if not should_degrade:
            raise
        reason = "timeout" if timeout_error else f"error:{type(exc).__name__}"
        degraded = build_degraded_response(
            request.query,
            request.context,
            request.session_id or "default",
            reason=reason,
            latency_ms=max(0, int((perf_counter() - started_at) * 1000)),
        )
        return degraded.model_dump()


async def _tool_retrieve_knowledge(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required and must be non-empty")

    context = arguments.get("context")
    if not isinstance(context, dict):
        context = {}
    api_key = arguments.get("api_key")
    if not isinstance(api_key, str):
        api_key = None
    client_id = _resolve_mcp_client_id(arguments, api_key)

    settings = Settings.from_env()
    request = RunRequest(query=query.strip(), context=context, session_id="mcp-retrieve")
    validate_request_security(request, settings, provided_api_key=api_key)
    _check_mcp_rate_limit(
        settings=settings,
        endpoint="mcp_retrieve_knowledge",
        session_id=request.session_id,
        client_id=client_id,
    )

    def _invoke():
        return retrieve_knowledge(request.query, request.context, settings)

    result, _ = await run_with_governance(
        _invoke,
        timeout_seconds=settings.request_timeout_seconds,
        retry_attempts=settings.run_retry_attempts,
        retry_backoff_ms=settings.retry_backoff_ms,
    )
    return result


async def _tool_health(arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    settings = Settings.from_env()
    return {
        "status": "healthy",
        "rag_enabled": settings.rag_enabled,
        "session_backend": settings.session_backend,
        "rate_limit_enabled": settings.rate_limit_enabled,
    }


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

TOOLS: dict[str, dict[str, Any]] = {
    "run_agent": {
        "description": "Run merchant ops ReAct agent and return structured result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User query"},
                "context": {"type": "object", "description": "Structured context", "default": {}},
                "session_id": {"type": "string", "description": "Optional session id"},
                "api_key": {"type": "string", "description": "Optional API key when APP_AUTH_ENABLED=true"},
                "client_id": {"type": "string", "description": "Optional caller id for MCP rate limit partition"},
            },
            "required": ["query"],
        },
        "handler": _tool_run_agent,
    },
    "retrieve_knowledge": {
        "description": "Retrieve relevant snippets from local RAG knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "context": {"type": "object", "description": "Optional context", "default": {}},
                "api_key": {"type": "string", "description": "Optional API key when APP_AUTH_ENABLED=true"},
                "client_id": {"type": "string", "description": "Optional caller id for MCP rate limit partition"},
            },
            "required": ["query"],
        },
        "handler": _tool_retrieve_knowledge,
    },
    "health": {
        "description": "Get backend health and runtime feature flags.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": _tool_health,
    },
}


def create_mcp_server() -> Any:
    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server("merchant-ops-copilot-mcp")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=meta["description"],
                inputSchema=meta["input_schema"],
            )
            for name, meta in TOOLS.items()
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        meta = TOOLS.get(name)
        if meta is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Tool not found: {name}")],
                isError=True,
            )

        handler: ToolHandler = meta["handler"]
        try:
            payload = await handler(arguments or {})
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=_to_text(payload))],
                isError=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool failed: %s", name)
            if isinstance(exc, HTTPException):
                text = f"HTTP {exc.status_code}: {exc.detail}"
            else:
                text = f"Tool execution failed: {exc}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                isError=True,
            )

    return server


async def run_stdio_server() -> int:
    import mcp.server.stdio

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = create_mcp_server()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    return 0


def main() -> int:
    try:
        return asyncio.run(run_stdio_server())
    except ImportError as exc:
        print(
            "MCP dependency is missing. Install requirements first: "
            "pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
