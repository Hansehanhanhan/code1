from __future__ import annotations

import time
from threading import Thread

import pytest

from backend.diagnostic_state import DiagnosticPatch, StateConflictError
from backend.session_store import InMemorySessionStore, RedisSessionStore

REDIS_URL = "redis://127.0.0.1:6379/15"
TTL = 3600

try:
    _probe = RedisSessionStore(REDIS_URL)
    _probe._client.flushdb()
    REDIS_AVAILABLE = True
except Exception:  # noqa: BLE001
    REDIS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="Redis/Memurai not reachable at 127.0.0.1:6379",
)


@pytest.fixture
def redis_store():
    store = RedisSessionStore(REDIS_URL)
    store._client.flushdb()
    try:
        yield store
    finally:
        store._client.flushdb()


def _slot_patch(expected_version: int, **extra) -> DiagnosticPatch:
    return DiagnosticPatch(
        reason="test",
        expected_version=expected_version,
        context_slots={"merchant_id": "m-1"},
        **extra,
    )


def _normalize(state) -> dict:
    dump = state.model_dump()
    for finding in dump["verified_findings"]:
        finding.pop("created_at", None)
    return dump


def _update_with_retry(store, session_id: str, patch: DiagnosticPatch, retries: int = 100) -> None:
    import random as _random

    for attempt in range(retries):
        try:
            store.update_state(session_id, patch, ttl_seconds=TTL)
            return
        except StateConflictError:
            current = store.get_state(session_id)
            patch = patch.model_copy(update={"expected_version": current.version})
            if attempt < retries - 1:
                time.sleep(_random.uniform(0.002, 0.01))
    raise StateConflictError("update failed after retries")


def test_redis_stale_version_patch_raises_state_conflict(redis_store) -> None:
    store = redis_store
    store.update_state("s1", _slot_patch(0, add_candidates=["a"]), ttl_seconds=TTL)
    assert store.get_state("s1").version == 1

    with pytest.raises(StateConflictError):
        store.update_state("s1", _slot_patch(0, add_candidates=["b"]), ttl_seconds=TTL)


def test_redis_state_sequence_matches_inmemory() -> None:
    memory = InMemorySessionStore()
    redis = RedisSessionStore(REDIS_URL)
    sequence = [
        _slot_patch(
            0,
            add_candidates=["c1"],
            add_findings=[
                {
                    "id": "f-1",
                    "claim": "流量下滑",
                    "confidence": "medium",
                    "evidence": {
                        "evidence_id": "req-1:tool_0001",
                        "request_id": "req-1",
                        "observation_hash": "sha256:abc",
                    },
                }
            ],
        ),
        _slot_patch(1, add_candidates=["c2"], reject_candidates=["c1"], add_constraints=["需要近14天数据"]),
        _slot_patch(2, add_candidates=["c3"], replace_next_step_plan=["复盘广告位", "排查详情页"]),
    ]
    for step in sequence:
        memory.update_state("same", step, ttl_seconds=TTL)
        redis.update_state("same", step, ttl_seconds=TTL)

    assert _normalize(memory.get_state("same")) == _normalize(redis.get_state("same"))


def test_redis_cross_instance_share_state() -> None:
    first = RedisSessionStore(REDIS_URL)
    second = RedisSessionStore(REDIS_URL)
    first.update_state("shared", _slot_patch(0, add_candidates=["跨实例候选"]), ttl_seconds=TTL)
    assert "跨实例候选" in second.get_state("shared").current_candidates
    assert second.get_state("shared").version == 1


def test_redis_owner_ttl_expires(redis_store) -> None:
    store = redis_store
    store.bind_owner("ttl", ("u-1", "t-1"), ttl_seconds=1)
    assert store.get_owner("ttl") == ("u-1", "t-1")
    time.sleep(1.2)
    assert store.get_owner("ttl") is None


def test_redis_state_ttl_expires(redis_store) -> None:
    store = redis_store
    store.update_state("ttl", _slot_patch(0), ttl_seconds=1)
    assert store.get_state("ttl").version == 1
    time.sleep(1.2)
    assert store.get_state("ttl").version == 0


def test_redis_history_capacity_eviction() -> None:
    store = RedisSessionStore(REDIS_URL)
    for index in range(5):
        store.append_turn(
            "hist",
            f"q{index}",
            f"a{index}",
            max_history_turns=3,
            ttl_seconds=TTL,
        )
    history = store.get_history("hist")
    assert [query for query, _answer in history] == ["q2", "q3", "q4"]


def test_redis_concurrent_updates_no_lost_versions() -> None:
    store = RedisSessionStore(REDIS_URL)
    errors: list[BaseException] = []
    barrier = {"go": False}

    def worker(label: str) -> None:
        try:
            while not barrier["go"]:
                time.sleep(0.001)
            for index in range(30):
                current = store.get_state("race")
                _update_with_retry(store, "race", _slot_patch(current.version, add_candidates=[f"{label}{index}"]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [Thread(target=worker, args=("w1-",)), Thread(target=worker, args=("w2-",))]
    for thread in threads:
        thread.start()
    barrier["go"] = True
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    final = store.get_state("race")
    assert final.version == 60
    assert len(final.current_candidates) == 10


def test_redis_get_state_merges_context_when_state_missing() -> None:
    store = RedisSessionStore(REDIS_URL)
    store.update_context_slots("orphan", {"merchant_id": "m-x"}, ttl_seconds=TTL)
    store._client.delete(RedisSessionStore._state_key("orphan"))
    state = store.get_state("orphan")
    assert state.context_slots == {"merchant_id": "m-x"}
    assert state.version == 0


def test_redis_finding_with_evidence_roundtrip(redis_store) -> None:
    store = redis_store
    store.update_state(
        "s2",
        _slot_patch(
            0,
            add_findings=[
                {
                    "id": "f-1",
                    "claim": "广告ROI下降",
                    "confidence": "high",
                    "evidence": {
                        "evidence_id": "req-1:tool_0003",
                        "request_id": "req-1",
                        "observation_hash": "sha256:9f86d0",
                    },
                }
            ],
        ),
        ttl_seconds=TTL,
    )
    finding = store.get_state("s2").verified_findings[0]
    assert finding.id == "f-1"
    assert finding.evidence.evidence_id == "req-1:tool_0003"
    assert finding.evidence.request_id == "req-1"
    assert finding.evidence.observation_hash == "sha256:9f86d0"