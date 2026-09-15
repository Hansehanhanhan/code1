from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

logger = logging.getLogger("merchant_ops.identities")


@dataclass(frozen=True)
class Identity:
    """API Key 对应的身份与权限范围。"""

    api_key: str
    user_id: str
    tenant_id: str
    merchant_ids: tuple[str, ...] = field(default_factory=tuple)

    def can_access_merchant(self, merchant_id: str) -> bool:
        # 未声明商家（空/None）表示无约束。
        if not merchant_id:
            return True
        # 空列表 = 允许访问所有商家。
        if not self.merchant_ids:
            return True
        return merchant_id in self.merchant_ids


def _load_keys(identity_keys_path: str) -> list[dict]:
    path = Path(identity_keys_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("invalid identity keys file %s: %s", identity_keys_path, exc)
        return []
    raw_keys = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(raw_keys, list):
        return []
    parsed: list[dict] = []
    for entry in raw_keys:
        if not isinstance(entry, dict):
            continue
        api_key = str(entry.get("api_key") or "").strip()
        user_id = str(entry.get("user_id") or "").strip()
        tenant_id = str(entry.get("tenant_id") or "").strip()
        if not api_key or not user_id or not tenant_id:
            continue
        raw_merchants = entry.get("merchant_ids") or []
        merchant_ids = tuple(str(item).strip() for item in raw_merchants if str(item).strip()) if isinstance(
            raw_merchants, list
        ) else tuple()
        parsed.append(
            {
                "api_key": api_key,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "merchant_ids": tuple(dict.fromkeys(merchant_ids)),
            }
        )
    return parsed


class IdentityRegistry:
    """身份注册表：从 api_keys.json 加载（线程安全，可刷新）。"""

    def __init__(self, identity_keys_path: str) -> None:
        self._identity_keys_path = identity_keys_path
        self._lock = RLock()
        self._reload()

    def _reload(self) -> None:
        self._by_key: dict[str, dict] = {}
        for entry in _load_keys(self._identity_keys_path):
            self._by_key[entry["api_key"]] = entry

    def resolve(self, api_key: str | None) -> Identity | None:
        if not api_key:
            return None
        key = str(api_key).strip()
        with self._lock:
            entry = self._by_key.get(key)
        if entry is None:
            return None
        return Identity(
            api_key=key,
            user_id=entry["user_id"],
            tenant_id=entry["tenant_id"],
            merchant_ids=entry["merchant_ids"],
        )


_registry_lock = RLock()
_cached_registry: IdentityRegistry | None = None
_cached_registry_path = ""


def get_identity_registry(identity_keys_path: str) -> IdentityRegistry:
    global _cached_registry
    global _cached_registry_path
    with _registry_lock:
        if _cached_registry is not None and _cached_registry_path == identity_keys_path:
            return _cached_registry
        registry = IdentityRegistry(identity_keys_path)
        _cached_registry = registry
        _cached_registry_path = identity_keys_path
        return registry


def build_identity_from_key(api_key: str, *, user_id: str, tenant_id: str) -> Identity:
    """构造身份（用于管理员 key 等预定义映射）。"""
    return Identity(
        api_key=api_key,
        user_id=user_id,
        tenant_id=tenant_id,
        merchant_ids=tuple(),
    )