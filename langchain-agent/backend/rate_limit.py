from __future__ import annotations

import json
import logging
import time
from threading import RLock
from typing import Protocol

from backend.settings import Settings

logger = logging.getLogger("merchant_ops.rate_limit")


class RateLimiter(Protocol):
    """限流器统一协议：返回 (是否放行, 剩余配额)。"""

    def allow(self, key: str, *, max_requests: int | None = None) -> tuple[bool, int]:
        ...


class NoopRateLimiter:
    """关闭限流时使用的空实现。"""

    def allow(self, key: str, *, max_requests: int | None = None) -> tuple[bool, int]:
        del key
        del max_requests
        return True, 0


class InMemoryFixedWindowRateLimiter:
    """内存版固定窗口限流（单进程有效）。"""

    def __init__(self, window_seconds: int, max_requests: int) -> None:
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._lock = RLock()
        # key -> (window_id, count)
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, key: str, *, max_requests: int | None = None) -> tuple[bool, int]:
        limit = self._max_requests if max_requests is None else max(1, int(max_requests))
        now = int(time.time())
        current_window = now // self._window_seconds
        with self._lock:
            existing = self._windows.get(key)
            if existing is None or existing[0] != current_window:
                self._windows[key] = (current_window, 1)
                return True, limit - 1

            _, count = existing
            if count >= limit:
                return False, 0

            count += 1
            self._windows[key] = (current_window, count)
            return True, limit - count


class RedisFixedWindowRateLimiter:
    """Redis 版固定窗口限流（多实例可共享计数）。"""

    def __init__(self, redis_url: str, window_seconds: int, max_requests: int) -> None:
        from redis import Redis

        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    def allow(self, key: str, *, max_requests: int | None = None) -> tuple[bool, int]:
        limit = self._max_requests if max_requests is None else max(1, int(max_requests))
        now = int(time.time())
        window_id = now // self._window_seconds
        # 把用户 key 和窗口编号拼成 redis key，实现按窗口隔离计数。
        redis_key = f"merchant_ops:rl:{key}:{window_id}"

        # fixed-window 的基本写法：INCR + EXPIRE。
        pipeline = self._client.pipeline()
        pipeline.incr(redis_key)
        pipeline.expire(redis_key, self._window_seconds)
        count, _ = pipeline.execute()
        count_int = int(count)

        if count_int > limit:
            return False, 0
        return True, limit - count_int


_limiter_lock = RLock()
_cached_limiter: RateLimiter | None = None
_cached_mode = ""


def get_rate_limiter(settings: Settings) -> RateLimiter:
    """按配置返回限流器，并带有 Redis->Memory 的自动降级。"""

    global _cached_limiter
    global _cached_mode

    # 配置签名变化时才重建，避免每次请求重复初始化。
    mode = f"enabled:{settings.rate_limit_enabled}|backend:{settings.session_backend}|redis:{settings.redis_url}|window:{settings.rate_limit_window_seconds}|max:{settings.rate_limit_max_requests}"
    with _limiter_lock:
        if _cached_limiter is not None and _cached_mode == mode:
            return _cached_limiter

        if not settings.rate_limit_enabled:
            _cached_limiter = NoopRateLimiter()
            _cached_mode = mode
            logger.info(json.dumps({"event": "rate_limiter_ready", "backend": "disabled"}, ensure_ascii=False))
            return _cached_limiter

        if settings.redis_url:
            try:
                _cached_limiter = RedisFixedWindowRateLimiter(
                    settings.redis_url,
                    settings.rate_limit_window_seconds,
                    settings.rate_limit_max_requests,
                )
                _cached_mode = mode
                logger.info(json.dumps({"event": "rate_limiter_ready", "backend": "redis"}, ensure_ascii=False))
                return _cached_limiter
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    json.dumps(
                        {
                            "event": "rate_limiter_fallback",
                            "backend": "memory",
                            "reason": f"redis_unavailable:{exc}",
                        },
                        ensure_ascii=False,
                    )
                )

        _cached_limiter = InMemoryFixedWindowRateLimiter(
            window_seconds=settings.rate_limit_window_seconds,
            max_requests=settings.rate_limit_max_requests,
        )
        _cached_mode = mode
        logger.info(json.dumps({"event": "rate_limiter_ready", "backend": "memory"}, ensure_ascii=False))
        return _cached_limiter
