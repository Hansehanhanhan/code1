from __future__ import annotations

import hmac
import json
import re
from typing import Any

from fastapi import HTTPException

from backend.models import RunRequest
from backend.settings import Settings

# 基础注入特征（规则级拦截，不依赖模型判断）。
_PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+all\s+previous\s+instructions",
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"dan\b",
    r"<\s*script\b",
    r"```system",
    r"忽略(以上|之前).*(指令|要求)",
)
_PROMPT_INJECTION_REGEX = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), flags=re.IGNORECASE)


def build_context_text(context: dict[str, Any]) -> str:
    """将 context 统一序列化为字符串，便于长度检查和安全检查。"""
    return json.dumps(context or {}, ensure_ascii=False, default=str)


def ensure_request_auth_from_key(provided_api_key: str | None, current_settings: Settings) -> None:
    """简单 API Key 鉴权（开启时强制校验）。"""
    if not current_settings.app_auth_enabled:
        return
    expected = (current_settings.app_api_key or "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="APP_API_KEY is not configured")
    provided = (provided_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


def ensure_input_limits(request: RunRequest, current_settings: Settings) -> None:
    """控制 query/context 输入体量，避免超长输入拖垮服务。"""
    query_length = len(request.query)
    if query_length > current_settings.max_query_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Query too long: {query_length} chars (max {current_settings.max_query_chars})",
        )
    context_text = build_context_text(request.context or {})
    context_length = len(context_text)
    if context_length > current_settings.max_context_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Context too long: {context_length} chars (max {current_settings.max_context_chars})",
        )


def ensure_prompt_safety(request: RunRequest, current_settings: Settings) -> None:
    """按规则做基础 Prompt 注入拦截。"""
    if not current_settings.prompt_injection_guard_enabled:
        return
    context_text = build_context_text(request.context or {})
    candidate = f"{request.query}\n{context_text}"
    if _PROMPT_INJECTION_REGEX.search(candidate):
        raise HTTPException(
            status_code=400,
            detail="Potential prompt injection pattern detected in input.",
        )


def validate_request_security(
    request: RunRequest,
    current_settings: Settings,
    *,
    provided_api_key: str | None,
) -> None:
    """统一安全校验入口：鉴权 -> 长度限制 -> 注入检查。"""
    ensure_request_auth_from_key(provided_api_key, current_settings)
    ensure_input_limits(request, current_settings)
    ensure_prompt_safety(request, current_settings)
