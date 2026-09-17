from __future__ import annotations

from backend.settings import Settings


def _clear_env(monkeypatch, keys: tuple[str, ...]) -> None:
    for key in keys:
        monkeypatch.delenv(key, raising=False)


DEFAULT_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "ALLOW_RULE_FALLBACK",
    "RAG_ENABLED",
    "RAG_DOCS_DIR",
    "RAG_VECTOR_BACKEND",
    "RAG_TOP_K",
    "RAG_FETCH_K",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_DEVICE",
    "SESSION_BACKEND",
    "SESSION_TTL_SECONDS",
    "REDIS_URL",
    "RATE_LIMIT_ENABLED",
    "RATE_LIMIT_WINDOW_SECONDS",
    "RATE_LIMIT_MAX_REQUESTS",
    "RATE_LIMIT_MAX_REQUESTS_RUN",
    "RATE_LIMIT_MAX_REQUESTS_STREAM",
    "RATE_LIMIT_MAX_REQUESTS_IP",
    "TRUST_X_FORWARDED_FOR",
    "TRUSTED_PROXY_IPS",
    "REQUEST_TIMEOUT_SECONDS",
    "REQUEST_TIMEOUT_SECONDS_STREAM",
    "RUN_RETRY_ATTEMPTS",
    "RETRY_BACKOFF_MS",
    "DEGRADE_ON_TIMEOUT",
    "DEGRADE_ON_ERROR",
    "APP_AUTH_ENABLED",
    "APP_API_KEY",
    "IDENTITY_KEYS_PATH",
    "MAX_QUERY_CHARS",
    "MAX_CONTEXT_CHARS",
    "PROMPT_INJECTION_GUARD_ENABLED",
    "APP_CORS_ORIGINS",
    "APP_CORS_ALLOW_CREDENTIALS",
    "AGENT_VERBOSE",
    "OUTER_LOOP_ENABLED",
    "JOB_DB_PATH",
)


def test_defaults_with_no_env(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    s = Settings.from_env()
    assert s.openai_base_url == "https://api.deepseek.com"
    assert s.openai_model == "deepseek-chat"
    assert s.rag_enabled is True
    assert s.rag_top_k == 3
    assert s.rag_fetch_k == 12
    assert s.rag_vector_backend == "chroma"
    assert s.rag_embedding_device == "cpu"
    assert s.session_backend == "memory"
    assert s.session_ttl_seconds == 86400
    assert s.rate_limit_enabled is True
    assert s.rate_limit_max_requests == 30
    assert s.trust_x_forwarded_for is False
    assert s.trusted_proxy_ips == []
    assert s.request_timeout_seconds == 120
    assert s.request_timeout_seconds_stream == 150
    assert s.run_retry_attempts == 1
    assert s.degrade_on_timeout is True
    assert s.app_auth_enabled is False
    assert s.max_query_chars == 2000
    assert s.app_cors_origins == ["http://127.0.0.1:3000", "http://localhost:3000"]
    assert s.app_cors_allow_credentials is False
    assert s.agent_verbose is False
    assert s.outer_loop_enabled is True
    assert s.job_db_path == ".run/jobs.db"


def test_bool_parsing_true_variants(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    for raw in ("1", "true", "TRUE", "Yes", "on", " On "):
        monkeypatch.setenv("APP_AUTH_ENABLED", raw)
        assert Settings.from_env().app_auth_enabled is True


def test_bool_parsing_false_and_garbage(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    for raw in ("0", "false", "FALSE", "no", "off", "abc", ""):
        monkeypatch.setenv("APP_AUTH_ENABLED", raw)
        assert Settings.from_env().app_auth_enabled is False


def test_int_parsing_clamps_to_minimum(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    monkeypatch.setenv("RAG_TOP_K", "0")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "30")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("RUN_RETRY_ATTEMPTS", "x")
    monkeypatch.setenv("RETRY_BACKOFF_MS", "-5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "5")
    monkeypatch.setenv("RUN_RETRY_ATTEMPTS", "3")
    s = Settings.from_env()
    assert s.rag_top_k == 1
    assert s.session_ttl_seconds == 60
    assert s.request_timeout_seconds == 5
    assert s.run_retry_attempts == 3
    assert s.retry_backoff_ms == 0
    assert s.rate_limit_window_seconds == 5


def test_csv_parsing(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    monkeypatch.setenv("TRUSTED_PROXY_IPS", ",10.0.0.1, ,10.0.0.2,")
    assert Settings.from_env().trusted_proxy_ips == ["10.0.0.1", "10.0.0.2"]
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "")
    assert Settings.from_env().trusted_proxy_ips == []
    monkeypatch.setenv("APP_CORS_ORIGINS", " http://a.example ")
    assert Settings.from_env().app_cors_origins == ["http://a.example"]


def test_string_normalization(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    monkeypatch.setenv("SESSION_BACKEND", " Redis ")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "CHROMA")
    monkeypatch.setenv("RAG_EMBEDDING_DEVICE", " CPU ")
    s = Settings.from_env()
    assert s.session_backend == "redis"
    assert s.rag_vector_backend == "chroma"
    assert s.rag_embedding_device == "cpu"


def test_openai_overrides(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setenv("OPENAI_MODEL", "fake-stub")
    monkeypatch.setenv("OPENAI_API_KEY", "ci-dummy-key")
    s = Settings.from_env()
    assert s.openai_base_url == "http://127.0.0.1:18080/v1"
    assert s.openai_model == "fake-stub"
    assert s.openai_api_key == "ci-dummy-key"


def test_auth_flags(monkeypatch) -> None:
    _clear_env(monkeypatch, DEFAULT_KEYS)
    monkeypatch.setenv("APP_AUTH_ENABLED", "true")
    monkeypatch.setenv("APP_API_KEY", "admin-secret")
    monkeypatch.setenv("IDENTITY_KEYS_PATH", "api_keys.test.json")
    s = Settings.from_env()
    assert s.app_auth_enabled is True
    assert s.app_api_key == "admin-secret"
    assert s.identity_keys_path == "api_keys.test.json"