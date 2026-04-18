from __future__ import annotations

from typing import Any

from backend.settings import Settings

import backend.rate_limit as rate_limit


def make_settings(**overrides: Any) -> Settings:
    base = Settings(
        openai_api_key="k",
        openai_base_url="u",
        openai_model="m",
        allow_rule_fallback=True,
        rag_enabled=True,
        rag_docs_dir="knowledge/seed",
        rag_vector_backend="chroma",
        rag_top_k=3,
        rag_fetch_k=12,
        rag_embedding_model="BAAI/bge-small-zh-v1.5",
        rag_embedding_device="cpu",
        session_backend="memory",
        session_ttl_seconds=3600,
        redis_url=None,
        rate_limit_enabled=True,
        rate_limit_window_seconds=60,
        rate_limit_max_requests=2,
        rate_limit_max_requests_run=2,
        rate_limit_max_requests_stream=1,
        rate_limit_max_requests_ip=10,
        trust_x_forwarded_for=False,
        trusted_proxy_ips=[],
        request_timeout_seconds=120,
        request_timeout_seconds_stream=150,
        run_retry_attempts=1,
        retry_backoff_ms=300,
        degrade_on_timeout=True,
        degrade_on_error=True,
        app_auth_enabled=False,
        app_api_key=None,
        max_query_chars=2000,
        max_context_chars=8000,
        prompt_injection_guard_enabled=True,
        app_cors_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        app_cors_allow_credentials=False,
        agent_verbose=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def reset_rate_limiter_cache() -> None:
    rate_limit._cached_limiter = None
    rate_limit._cached_mode = ""


def test_in_memory_fixed_window() -> None:
    limiter = rate_limit.InMemoryFixedWindowRateLimiter(window_seconds=60, max_requests=2)

    assert limiter.allow("k1") == (True, 1)
    assert limiter.allow("k1") == (True, 0)
    assert limiter.allow("k1") == (False, 0)


def test_in_memory_fixed_window_with_override_limit() -> None:
    limiter = rate_limit.InMemoryFixedWindowRateLimiter(window_seconds=60, max_requests=10)
    assert limiter.allow("k2", max_requests=1) == (True, 0)
    assert limiter.allow("k2", max_requests=1) == (False, 0)


def test_noop_rate_limiter_when_disabled() -> None:
    reset_rate_limiter_cache()
    settings = make_settings(rate_limit_enabled=False)

    limiter = rate_limit.get_rate_limiter(settings)
    assert isinstance(limiter, rate_limit.NoopRateLimiter)
    assert limiter.allow("any") == (True, 0)


def test_get_rate_limiter_falls_back_to_memory(monkeypatch) -> None:
    reset_rate_limiter_cache()

    class FailingRedisLimiter:
        def __init__(self, redis_url: str, window_seconds: int, max_requests: int) -> None:
            del window_seconds, max_requests
            raise RuntimeError(f"cannot connect: {redis_url}")

    monkeypatch.setattr(rate_limit, "RedisFixedWindowRateLimiter", FailingRedisLimiter)

    settings = make_settings(redis_url="redis://127.0.0.1:6379/0")
    limiter = rate_limit.get_rate_limiter(settings)

    assert isinstance(limiter, rate_limit.InMemoryFixedWindowRateLimiter)


def test_get_rate_limiter_uses_cached_instance() -> None:
    reset_rate_limiter_cache()

    settings = make_settings()
    limiter1 = rate_limit.get_rate_limiter(settings)
    limiter2 = rate_limit.get_rate_limiter(settings)

    assert limiter1 is limiter2
