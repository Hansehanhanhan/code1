from __future__ import annotations

import json
import logging
from collections import defaultdict
from threading import RLock
from typing import Protocol

from backend.settings import Settings

logger = logging.getLogger("merchant_ops.session_store")


class SessionStore(Protocol):
    def get_history(self, session_id: str) -> list[tuple[str, str]]:
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


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._data.get(session_id, []))

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
            history = self._data[session_id]
            history.append((query, final_answer))
            if len(history) > max_history_turns:
                self._data[session_id] = history[-max_history_turns:]


class RedisSessionStore:
    def __init__(self, redis_url: str) -> None:
        from redis import Redis

        # decode_responses=True makes Redis return str instead of bytes.
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"merchant_ops:session:{session_id}:history"

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        key = self._key(session_id)
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
        pipeline.rpush(key, payload)
        pipeline.ltrim(key, -max_history_turns, -1)
        pipeline.expire(key, ttl_seconds)
        pipeline.execute()


_store_lock = RLock()
_cached_store: SessionStore | None = None
_cached_store_mode = ""


def get_session_store(settings: Settings) -> SessionStore:
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

