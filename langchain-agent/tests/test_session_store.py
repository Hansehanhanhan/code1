from __future__ import annotations

from typing import Any

import pytest

from backend.settings import Settings
from backend.diagnostic_state import DiagnosticPatch, StateConflictError
from tests.conftest import make_settings as _base_settings

import backend.session_store as session_store


def make_settings(**overrides: Any) -> Settings:
    return _base_settings(rag_enabled=True, **overrides)


def reset_session_store_cache() -> None:
    session_store._cached_store = None
    session_store._cached_store_mode = ""


def test_in_memory_session_store_trim_history() -> None:
    store = session_store.InMemorySessionStore()
    store.append_turn("s1", "q1", "a1", max_history_turns=2, ttl_seconds=100)
    store.append_turn("s1", "q2", "a2", max_history_turns=2, ttl_seconds=100)
    store.append_turn("s1", "q3", "a3", max_history_turns=2, ttl_seconds=100)

    history = store.get_history("s1")
    assert history == [("q2", "a2"), ("q3", "a3")]


def test_in_memory_get_history_returns_copy() -> None:
    store = session_store.InMemorySessionStore()
    store.append_turn("s1", "q1", "a1", max_history_turns=5, ttl_seconds=100)

    history = store.get_history("s1")
    history.append(("q2", "a2"))

    latest = store.get_history("s1")
    assert latest == [("q1", "a1")]


def test_in_memory_context_slots_are_merged_and_copied() -> None:
    store = session_store.InMemorySessionStore()
    store.update_context_slots(
        "s1",
        {"merchant_id": "demo-001", "time_range": "last_7_days"},
        ttl_seconds=100,
    )
    store.update_context_slots("s1", {"time_range": "yesterday"}, ttl_seconds=100)

    slots = store.get_context_slots("s1")
    assert slots == {"merchant_id": "demo-001", "time_range": "yesterday"}

    slots["merchant_id"] = "mutated"
    assert store.get_context_slots("s1")["merchant_id"] == "demo-001"


def test_in_memory_state_update_is_versioned_and_atomic() -> None:
    store = session_store.InMemorySessionStore()
    patch = DiagnosticPatch(reason="test", expected_version=0, context_slots={"merchant_id": "demo-001"})

    updated = store.update_state("s1", patch, ttl_seconds=100)

    assert updated.version == 1
    assert updated.context_slots == {"merchant_id": "demo-001"}
    assert store.get_context_slots("s1") == {"merchant_id": "demo-001"}

    with pytest.raises(StateConflictError):
        store.update_state("s1", patch, ttl_seconds=100)


def test_in_memory_state_patch_adds_constraints_and_dedupes() -> None:
    store = session_store.InMemorySessionStore()
    first = DiagnosticPatch(
        reason="request_finalize",
        expected_version=0,
        add_constraints=["库存数据缺失，无法评估"],
        add_candidates=["检查广告位"],
    )
    second = DiagnosticPatch(
        reason="request_finalize",
        expected_version=1,
        add_constraints=["库存数据缺失，无法评估", "缺少转化分环节数据"],
    )

    updated = store.update_state("s1", first, ttl_seconds=100)
    updated = store.update_state("s1", second, ttl_seconds=100)

    assert updated.version == 2
    assert updated.unresolved_constraints == ["库存数据缺失，无法评估", "缺少转化分环节数据"]
    assert updated.current_candidates == ["检查广告位"]


def test_get_session_store_falls_back_to_memory(monkeypatch) -> None:
    reset_session_store_cache()

    class FailingRedisStore:
        def __init__(self, redis_url: str) -> None:
            raise RuntimeError(f"cannot connect: {redis_url}")

    monkeypatch.setattr(session_store, "RedisSessionStore", FailingRedisStore)

    settings = make_settings(session_backend="redis", redis_url="redis://127.0.0.1:6379/0")
    store = session_store.get_session_store(settings)

    assert isinstance(store, session_store.InMemorySessionStore)


def test_get_session_store_uses_cached_instance(monkeypatch) -> None:
    del monkeypatch
    reset_session_store_cache()

    settings = make_settings(session_backend="memory")
    store1 = session_store.get_session_store(settings)
    store2 = session_store.get_session_store(settings)

    assert store1 is store2


def test_in_memory_owner_bind_and_get() -> None:
    store = session_store.InMemorySessionStore()
    assert store.get_owner("s1") is None

    store.bind_owner("s1", ("user-a", "t1"), ttl_seconds=100)
    assert store.get_owner("s1") == ("user-a", "t1")


def test_in_memory_owner_bind_overwrites_and_is_scoped() -> None:
    store = session_store.InMemorySessionStore()
    store.bind_owner("s1", ("user-a", "t1"), ttl_seconds=100)
    store.bind_owner("s2", ("user-b", "t2"), ttl_seconds=100)

    assert store.get_owner("s1") == ("user-a", "t1")
    assert store.get_owner("s2") == ("user-b", "t2")


def test_redis_owner_bind_and_get_uses_json_key() -> None:
    class FakeRedisClient:
        def __init__(self) -> None:
            self._data: dict[str, object] = {}

        def get(self, key: str):
            return self._data.get(key)

        def set(self, key: str, value: str, *, ex: int) -> None:
            self._data[key] = value

    client = FakeRedisClient()
    store = session_store.RedisSessionStore.__new__(session_store.RedisSessionStore)
    store._client = client  # type: ignore[attr-defined]

    assert store.get_owner("s1") is None

    store.bind_owner("s1", ("user-a", "t1"), ttl_seconds=3600)

    assert store.get_owner("s1") == ("user-a", "t1")
    assert "merchant_ops:session:s1:owner" in client._data
