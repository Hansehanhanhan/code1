from __future__ import annotations

import json
import logging
from collections import defaultdict
from threading import RLock
from typing import Protocol

from backend.settings import Settings
from backend.diagnostic_state import DiagnosticPatch, DiagnosticState, apply_diagnostic_patch

logger = logging.getLogger("merchant_ops.session_store")


class SessionStore(Protocol):
    """会话存储统一协议（用于短期对话记忆）。"""

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        ...

    def get_owner(self, session_id: str) -> tuple[str, str] | None:
        ...

    def bind_owner(
        self,
        session_id: str,
        owner: tuple[str, str],
        *,
        ttl_seconds: int,
    ) -> None:
        ...

    def append_turn(
        self,
        session_id: str,
        query: str,
        final_answer: str,
        *,
        max_history_turns: int,
        ttl_seconds: int,
    ) -> None:
        ...

    def get_context_slots(self, session_id: str) -> dict[str, object]:
        ...

    def update_context_slots(
        self,
        session_id: str,
        slots: dict[str, object],
        *,
        ttl_seconds: int,
    ) -> None:
        ...

    def get_state(self, session_id: str) -> DiagnosticState:
        ...

    def update_state(
        self,
        session_id: str,
        patch: DiagnosticPatch,
        *,
        ttl_seconds: int,
    ) -> DiagnosticState:
        ...


class InMemorySessionStore:
    """内存会话存储：实现简单，但服务重启会丢失。"""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, object]] = defaultdict(
            lambda: {"history": [], "context_slots": {}, "state": DiagnosticState()}
        )
        self._owners: dict[str, tuple[str, str]] = {}
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            record = self._data.get(session_id)
            if record is None:
                return []
            return list(record["history"])

    def get_owner(self, session_id: str) -> tuple[str, str] | None:
        with self._lock:
            return self._owners.get(session_id)

    def bind_owner(
        self,
        session_id: str,
        owner: tuple[str, str],
        *,
        ttl_seconds: int,
    ) -> None:
        del ttl_seconds  # In-memory backend does not support TTL.
        with self._lock:
            self._owners[session_id] = owner

    def get_context_slots(self, session_id: str) -> dict[str, object]:
        with self._lock:
            record = self._data.get(session_id)
            if record is None:
                return {}
            return dict(record["context_slots"])

    def append_turn(
        self,
        session_id: str,
        query: str,
        final_answer: str,
        *,
        max_history_turns: int,
        ttl_seconds: int,
    ) -> None:
        del ttl_seconds  # In-memory backend does not support TTL.
        with self._lock:
            record = self._data[session_id]
            history = record["history"]
            history.append((query, final_answer))
            # 仅保留最近 N 轮，防止上下文无上限增长。
            if len(history) > max_history_turns:
                record["history"] = history[-max_history_turns:]

    def update_context_slots(
        self,
        session_id: str,
        slots: dict[str, object],
        *,
        ttl_seconds: int,
    ) -> None:
        current = self.get_state(session_id)
        patch = DiagnosticPatch(
            reason="context_slots_updated",
            expected_version=current.version,
            context_slots=slots,
        )
        self.update_state(session_id, patch, ttl_seconds=ttl_seconds)

    def get_state(self, session_id: str) -> DiagnosticState:
        with self._lock:
            record = self._data.get(session_id)
            if record is None:
                return DiagnosticState()
            return record["state"].model_copy(deep=True)

    def update_state(
        self,
        session_id: str,
        patch: DiagnosticPatch,
        *,
        ttl_seconds: int,
    ) -> DiagnosticState:
        del ttl_seconds  # In-memory backend does not support TTL.
        with self._lock:
            record = self._data[session_id]
            current = record["state"]
            updated = apply_diagnostic_patch(current, patch)
            record["state"] = updated
            record["context_slots"] = dict(updated.context_slots)
            return updated.model_copy(deep=True)


class RedisSessionStore:
    """Redis 会话存储：支持跨实例共享，并可设置 TTL。"""

    def __init__(self, redis_url: str) -> None:
        from redis import Redis

        # decode_responses=True makes Redis return str instead of bytes.
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"merchant_ops:session:{session_id}:history"

    @staticmethod
    def _context_key(session_id: str) -> str:
        return f"merchant_ops:session:{session_id}:context"

    @staticmethod
    def _state_key(session_id: str) -> str:
        return f"merchant_ops:session:{session_id}:state"

    @staticmethod
    def _owner_key(session_id: str) -> str:
        return f"merchant_ops:session:{session_id}:owner"

    def _decode_state(self, payload: str | None, session_id: str) -> DiagnosticState:
        if payload:
            try:
                return DiagnosticState.model_validate_json(payload)
            except ValueError:
                logger.warning("invalid diagnostic state payload for session %s", session_id)
        return DiagnosticState(context_slots=self.get_context_slots(session_id))

    def get_state(self, session_id: str) -> DiagnosticState:
        payload = self._client.get(self._state_key(session_id))
        return self._decode_state(payload, session_id)

    def get_owner(self, session_id: str) -> tuple[str, str] | None:
        payload = self._client.get(self._owner_key(session_id))
        if not payload:
            return None
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, list) or len(value) != 2:
            return None
        user_id, tenant_id = value
        if not isinstance(user_id, str) or not isinstance(tenant_id, str):
            return None
        return user_id, tenant_id

    def bind_owner(
        self,
        session_id: str,
        owner: tuple[str, str],
        *,
        ttl_seconds: int,
    ) -> None:
        self._client.set(
            self._owner_key(session_id),
            json.dumps([owner[0], owner[1]], ensure_ascii=False),
            ex=ttl_seconds,
        )

    def get_context_slots(self, session_id: str) -> dict[str, object]:
        payload = self._client.get(self._context_key(session_id))
        if not payload:
            return {}
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        key = self._key(session_id)
        # 历史按 list 顺序读取（左旧右新）。
        items = self._client.lrange(key, 0, -1)
        history: list[tuple[str, str]] = []
        for item in items:
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                continue
            query = payload.get("q")
            answer = payload.get("a")
            if isinstance(query, str) and isinstance(answer, str):
                history.append((query, answer))
        return history

    def append_turn(
        self,
        session_id: str,
        query: str,
        final_answer: str,
        *,
        max_history_turns: int,
        ttl_seconds: int,
    ) -> None:
        key = self._key(session_id)
        payload = json.dumps({"q": query, "a": final_answer}, ensure_ascii=False)
        pipeline = self._client.pipeline()
        # 追加 + 截断 + TTL 在一个 pipeline 中执行，减少网络往返。
        pipeline.rpush(key, payload)
        pipeline.ltrim(key, -max_history_turns, -1)
        pipeline.expire(key, ttl_seconds)
        pipeline.execute()

    def update_context_slots(
        self,
        session_id: str,
        slots: dict[str, object],
        *,
        ttl_seconds: int,
    ) -> None:
        current = self.get_state(session_id)
        patch = DiagnosticPatch(
            reason="context_slots_updated",
            expected_version=current.version,
            context_slots=slots,
        )
        self.update_state(session_id, patch, ttl_seconds=ttl_seconds)

    def update_state(
        self,
        session_id: str,
        patch: DiagnosticPatch,
        *,
        ttl_seconds: int,
    ) -> DiagnosticState:
        from redis.exceptions import WatchError

        state_key = self._state_key(session_id)
        context_key = self._context_key(session_id)
        max_retries = 3
        for _ in range(max_retries):
            pipeline = self._client.pipeline(transaction=True)
            try:
                pipeline.watch(state_key)
                current = self._decode_state(pipeline.get(state_key), session_id)
                updated = apply_diagnostic_patch(current, patch)
                pipeline.multi()
                payload = updated.model_dump_json()
                pipeline.set(state_key, payload, ex=ttl_seconds)
                pipeline.set(
                    context_key,
                    json.dumps(updated.context_slots, ensure_ascii=False),
                    ex=ttl_seconds,
                )
                pipeline.execute()
                return updated
            except WatchError:
                continue
            finally:
                pipeline.reset()
        raise StateConflictError(f"Redis state update conflicted after {max_retries} retries")


_store_lock = RLock()
_cached_store: SessionStore | None = None
_cached_store_mode = ""


def get_session_store(settings: Settings) -> SessionStore:
    """按配置返回会话存储，并支持 Redis 不可用时回退内存。"""

    global _cached_store
    global _cached_store_mode

    preferred_mode = settings.session_backend
    if preferred_mode == "redis" and settings.redis_url:
        preferred_mode = f"redis:{settings.redis_url}"
    elif preferred_mode == "redis":
        preferred_mode = "memory"

    with _store_lock:
        if _cached_store is not None and _cached_store_mode == preferred_mode:
            return _cached_store

        if settings.session_backend == "redis" and settings.redis_url:
            try:
                _cached_store = RedisSessionStore(settings.redis_url)
                _cached_store_mode = preferred_mode
                logger.info(json.dumps({"event": "session_store_ready", "backend": "redis"}, ensure_ascii=False))
                return _cached_store
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    json.dumps(
                        {
                            "event": "session_store_fallback",
                            "backend": "memory",
                            "reason": f"redis_unavailable:{exc}",
                        },
                        ensure_ascii=False,
                    )
                )

        _cached_store = InMemorySessionStore()
        _cached_store_mode = "memory"
        logger.info(json.dumps({"event": "session_store_ready", "backend": "memory"}, ensure_ascii=False))
        return _cached_store
