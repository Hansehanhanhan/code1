from __future__ import annotations

from typing import Any

from backend.settings import Settings


def make_settings(**overrides: Any) -> Settings:
    base = Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.example.com",
        openai_model="test-model",
        allow_rule_fallback=True,
        rag_enabled=False,
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
        identity_keys_path="api_keys.json",
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