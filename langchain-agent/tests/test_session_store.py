from __future__ import annotations

from typing import Any

from backend.settings import Settings

import backend.session_store as session_store


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
        rate_limit_max_requests=30,
        rate_limit_max_requests_run=20,
        rate_limit_max_requests_stream=10,
        rate_limit_max_requests_ip=60,
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
