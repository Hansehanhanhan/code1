from __future__ import annotations

import hmac
import json
import re
from typing import Any

from fastapi import HTTPException

from backend.identities import Identity, build_identity_from_key, get_identity_registry
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


_MANAGER_API_KEY_PREFIX = "__admin__"


def _authorize_valid_key(provided_api_key: str | None, current_settings: Settings) -> None:
    """鉴权：身份注册表中任意有效 key 或管理员 key 均放行（不解析具体身份）。"""
    if not current_settings.app_auth_enabled:
        return
    provided = (provided_api_key or "").strip()
    if not provided:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")
    if get_identity_registry(current_settings.identity_keys_path).resolve(provided) is not None:
        return
    expected_admin = (current_settings.app_api_key or "").strip()
    if expected_admin and hmac.compare_digest(provided, expected_admin):
        return
    raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


def authorize_identity(provided_api_key: str | None, current_settings: Settings) -> Identity | None:
    """把 API Key 解析为身份；鉴权关闭时返回 None（表示不强制）。"""
    if not current_settings.app_auth_enabled:
        return None

    provided = (provided_api_key or "").strip()
    if not provided:
        raise HTTPException(status_code=401, detail="Unauthorized: missing API key")

    registry_identity = get_identity_registry(current_settings.identity_keys_path).resolve(provided)
    if registry_identity is not None:
        return registry_identity

    expected_admin = (current_settings.app_api_key or "").strip()
    if expected_admin and hmac.compare_digest(provided, expected_admin):
        return build_identity_from_key(provided, user_id=_MANAGER_API_KEY_PREFIX, tenant_id=_MANAGER_API_KEY_PREFIX)

    raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


def ensure_merchant_access(identity: Identity | None, context: dict[str, Any]) -> None:
    """校验用户是否有权访问 context 中声明的 merchant_id。"""
    if identity is None:
        return
    raw_merchant_id = context.get("merchant_id")
    merchant_id = str(raw_merchant_id).strip() if raw_merchant_id is not None else ""
    if merchant_id and not identity.can_access_merchant(merchant_id):
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: no access to merchant_id '{merchant_id}'",
        )


def _identity_owner(identity: Identity) -> tuple[str, str]:
    return identity.user_id, identity.tenant_id


def ensure_session_ownership(
    session_store,
    session_id: str,
    identity: Identity | None,
    *,
    ttl_seconds: int,
) -> None:
    """校验 session_id 是否属于当前身份（首次访问时绑定）。"""
    if identity is None:
        return
    owner = _identity_owner(identity)
    existing_owner = session_store.get_owner(session_id)
    if existing_owner is None:
        session_store.bind_owner(session_id, owner, ttl_seconds=ttl_seconds)
        return
    if existing_owner != owner:
        raise HTTPException(status_code=403, detail="Forbidden: session_id belongs to another user")


def ensure_session_read_access(
    session_store,
    session_id: str,
    identity: Identity | None,
) -> None:
    """校验 session_id 读取权限；不绑定归属，仅当已有归属时校验。"""
    if identity is None:
        return
    existing_owner = session_store.get_owner(session_id)
    if existing_owner is None:
        return
    if existing_owner != _identity_owner(identity):
        raise HTTPException(status_code=403, detail="Forbidden: session_id belongs to another user")


def ensure_job_ownership(identity: Identity | None, job: dict[str, Any] | None) -> None:
    """校验任务是否属于当前身份。"""
    if identity is None:
        return
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    owner_tenant_id = str(job.get("owner_tenant_id") or "")
    owner_user_id = str(job.get("owner_user_id") or "")
    identity_tenant_id = str(identity.tenant_id or "")
    identity_user_id = str(identity.user_id or "")
    if not owner_tenant_id or not owner_user_id:
        raise HTTPException(status_code=403, detail="Forbidden: job has no owner")
    if owner_tenant_id != identity_tenant_id or owner_user_id != identity_user_id:
        raise HTTPException(status_code=403, detail="Forbidden: job belongs to another user")


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
    _authorize_valid_key(provided_api_key, current_settings)
    ensure_input_limits(request, current_settings)
    ensure_prompt_safety(request, current_settings)
