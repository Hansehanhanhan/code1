from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from backend.identities import Identity, IdentityRegistry, build_identity_from_key, get_identity_registry
from backend.security import (
    authorize_identity,
    ensure_merchant_access,
    ensure_session_ownership,
)
from backend.settings import Settings


def write_keys(tmp_path, keys: list[dict]) -> str:
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"keys": keys}), encoding="utf-8")
    return str(path)


from tests.conftest import make_settings


DEMO_KEYS = [
    {"api_key": "key-a", "user_id": "user-a", "tenant_id": "t1", "merchant_ids": ["demo-001"]},
    {"api_key": "key-b", "user_id": "user-b", "tenant_id": "t2", "merchant_ids": ["demo-002", "demo-003"]},
    {"api_key": "key-all", "user_id": "user-all", "tenant_id": "t3", "merchant_ids": []},
]


def test_registry_resolves_known_key(tmp_path) -> None:
    registry = IdentityRegistry(write_keys(tmp_path, DEMO_KEYS))
    identity = registry.resolve("key-b")
    assert identity is not None
    assert identity.user_id == "user-b"
    assert identity.tenant_id == "t2"
    assert identity.merchant_ids == ("demo-002", "demo-003")


def test_registry_missing_file_returns_empty(tmp_path) -> None:
    registry = IdentityRegistry(str(tmp_path / "missing.json"))
    assert registry.resolve("key-a") is None


def test_registry_resolve_unknown_key(tmp_path) -> None:
    registry = IdentityRegistry(write_keys(tmp_path, DEMO_KEYS))
    assert registry.resolve("unknown") is None


def test_registry_skips_invalid_entries(tmp_path) -> None:
    registry = IdentityRegistry(
        write_keys(
            tmp_path,
            [
                {"api_key": "missing-user", "tenant_id": "t1"},
                {"api_key": "ok", "user_id": "u", "tenant_id": "t", "merchant_ids": ["m1"]},
            ],
        )
    )
    assert registry.resolve("missing-user") is None
    assert registry.resolve("ok") is not None
    assert registry.resolve("ok").merchant_ids == ("m1",)


def test_can_access_merchant() -> None:
    limited = Identity(api_key="k", user_id="u", tenant_id="t", merchant_ids=("m1",))
    assert limited.can_access_merchant("m1") is True
    assert limited.can_access_merchant("m2") is False
    assert limited.can_access_merchant("") is True

    all_access = Identity(api_key="k", user_id="u", tenant_id="t", merchant_ids=())
    assert all_access.can_access_merchant("anything") is True
    assert all_access.can_access_merchant("") is True


def test_authorize_identity_returns_none_when_disabled(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=False, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    assert authorize_identity(None, settings) is None
    assert authorize_identity("garbage", settings) is None


def test_authorize_identity_missing_key_401(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    with pytest.raises(HTTPException) as exc_info:
        authorize_identity(None, settings)
    assert exc_info.value.status_code == 401


def test_authorize_identity_invalid_key_401(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    with pytest.raises(HTTPException) as exc_info:
        authorize_identity("bogus", settings)
    assert exc_info.value.status_code == 401


def test_authorize_identity_resolves_registry(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    identity = authorize_identity("key-a", settings)
    assert identity is not None
    assert identity.user_id == "user-a"
    assert identity.tenant_id == "t1"


def test_authorize_identity_admin_key_fallback(tmp_path) -> None:
    settings = make_settings(
        app_auth_enabled=True,
        app_api_key="admin-key",
        identity_keys_path=write_keys(tmp_path, DEMO_KEYS),
    )
    identity = authorize_identity("admin-key", settings)
    assert identity is not None
    assert identity.merchant_ids == ()


def test_ensure_merchant_access_granted(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    identity = authorize_identity("key-a", settings)
    ensure_merchant_access(identity, {"merchant_id": "demo-001"})
    ensure_merchant_access(identity, {"time_range": "last_7_days"})


def test_ensure_merchant_access_forbidden(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    identity = authorize_identity("key-a", settings)
    with pytest.raises(HTTPException) as exc_info:
        ensure_merchant_access(identity, {"merchant_id": "demo-002"})
    assert exc_info.value.status_code == 403


def test_ensure_merchant_access_all_merchants(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    identity = authorize_identity("key-all", settings)
    ensure_merchant_access(identity, {"merchant_id": "random-001"})


class MemoryStore:
    def __init__(self) -> None:
        self._owners: dict[str, tuple[str, str]] = {}

    def get_owner(self, session_id: str) -> tuple[str, str] | None:
        return self._owners.get(session_id)

    def bind_owner(self, session_id: str, owner: tuple[str, str], *, ttl_seconds: int) -> None:
        del ttl_seconds
        self._owners[session_id] = owner


def test_session_ownership_binds_on_first_use(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    identity = authorize_identity("key-a", settings)
    store = MemoryStore()
    ensure_session_ownership(store, "s1", identity, ttl_seconds=3600)
    assert store.get_owner("s1") == ("user-a", "t1")
    ensure_session_ownership(store, "s1", identity, ttl_seconds=3600)


def test_session_ownership_rejects_other_user(tmp_path) -> None:
    settings = make_settings(app_auth_enabled=True, identity_keys_path=write_keys(tmp_path, DEMO_KEYS))
    store = MemoryStore()
    owner_a = authorize_identity("key-a", settings)
    owner_b = authorize_identity("key-b", settings)
    ensure_session_ownership(store, "shared", owner_a, ttl_seconds=3600)
    with pytest.raises(HTTPException) as exc_info:
        ensure_session_ownership(store, "shared", owner_b, ttl_seconds=3600)
    assert exc_info.value.status_code == 403


def test_session_ownership_skipped_when_identity_none() -> None:
    store = MemoryStore()
    ensure_session_ownership(store, "s1", None, ttl_seconds=3600)
    assert store.get_owner("s1") is None


def test_get_identity_registry_cached(tmp_path) -> None:
    path = write_keys(tmp_path, DEMO_KEYS)
    assert get_identity_registry(path) is get_identity_registry(path)


def test_build_identity_from_key_grants_all_merchants() -> None:
    identity = build_identity_from_key("admin", user_id="u", tenant_id="t")
    assert identity.can_access_merchant("anything") is True