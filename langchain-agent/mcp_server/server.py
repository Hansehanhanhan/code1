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
from backend.security import validate_request_security
from backend.settings import Settings
from rag import retrieve_knowledge

logger = logging.getLogger("merchant_ops.mcp")


def _to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


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

    settings = Settings.from_env()
    request = RunRequest(query=query.strip(), context=context, session_id=session_id)
    validate_request_security(request, settings, provided_api_key=api_key)

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

    settings = Settings.from_env()
    request = RunRequest(query=query.strip(), context=context, session_id="mcp-retrieve")
    validate_request_security(request, settings, provided_api_key=api_key)

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
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Tool execution failed: {exc}")],
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
