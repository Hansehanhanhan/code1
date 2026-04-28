from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from project root.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _to_bool(value: str | None, default: bool) -> bool:
    """把环境变量解析为布尔值，解析失败时使用默认值。"""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int, minimum: int = 1) -> int:
    """把环境变量解析为整数，并约束最小值。"""
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _to_csv(value: str | None, default: list[str]) -> list[str]:
    """把逗号分隔环境变量解析为字符串列表。"""
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",")]
    parsed = [item for item in items if item]
    return parsed if parsed else list(default)


@dataclass
class Settings:
    """运行时配置集合（统一由环境变量加载）。"""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    allow_rule_fallback: bool
    rag_enabled: bool
    rag_docs_dir: str
    rag_vector_backend: str
    rag_top_k: int
    rag_fetch_k: int
    rag_embedding_model: str
    rag_embedding_device: str
    session_backend: str
    session_ttl_seconds: int
    redis_url: str | None
    rate_limit_enabled: bool
    rate_limit_window_seconds: int
    rate_limit_max_requests: int
    rate_limit_max_requests_run: int
    rate_limit_max_requests_stream: int
    rate_limit_max_requests_ip: int
    trust_x_forwarded_for: bool
    trusted_proxy_ips: list[str]
    request_timeout_seconds: int
    request_timeout_seconds_stream: int
    run_retry_attempts: int
    retry_backoff_ms: int
    degrade_on_timeout: bool
    degrade_on_error: bool
    app_auth_enabled: bool
    app_api_key: str | None
    max_query_chars: int
    max_context_chars: int
    prompt_injection_guard_enabled: bool
    app_cors_origins: list[str]
    app_cors_allow_credentials: bool
    agent_verbose: bool
    job_db_path: str = ".run/jobs.db"

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构造 Settings；所有默认值都在这里集中定义。"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            allow_rule_fallback=_to_bool(os.getenv("ALLOW_RULE_FALLBACK"), default=True),
            rag_enabled=_to_bool(os.getenv("RAG_ENABLED"), default=True),
            rag_docs_dir=os.getenv("RAG_DOCS_DIR", "knowledge/seed"),
            rag_vector_backend=(os.getenv("RAG_VECTOR_BACKEND", "chroma").strip().lower() or "chroma"),
            rag_top_k=_to_int(os.getenv("RAG_TOP_K"), default=3, minimum=1),
            rag_fetch_k=_to_int(os.getenv("RAG_FETCH_K"), default=12, minimum=1),
            rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            rag_embedding_device=(os.getenv("RAG_EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"),
            session_backend=(os.getenv("SESSION_BACKEND", "memory").strip().lower() or "memory"),
            session_ttl_seconds=_to_int(os.getenv("SESSION_TTL_SECONDS"), default=86400, minimum=60),
            redis_url=os.getenv("REDIS_URL"),
            rate_limit_enabled=_to_bool(os.getenv("RATE_LIMIT_ENABLED"), default=True),
            rate_limit_window_seconds=_to_int(os.getenv("RATE_LIMIT_WINDOW_SECONDS"), default=60, minimum=1),
            rate_limit_max_requests=_to_int(os.getenv("RATE_LIMIT_MAX_REQUESTS"), default=30, minimum=1),
            rate_limit_max_requests_run=_to_int(os.getenv("RATE_LIMIT_MAX_REQUESTS_RUN"), default=20, minimum=1),
            rate_limit_max_requests_stream=_to_int(os.getenv("RATE_LIMIT_MAX_REQUESTS_STREAM"), default=10, minimum=1),
            rate_limit_max_requests_ip=_to_int(os.getenv("RATE_LIMIT_MAX_REQUESTS_IP"), default=60, minimum=1),
            trust_x_forwarded_for=_to_bool(os.getenv("TRUST_X_FORWARDED_FOR"), default=False),
            trusted_proxy_ips=_to_csv(os.getenv("TRUSTED_PROXY_IPS"), default=[]),
            request_timeout_seconds=_to_int(os.getenv("REQUEST_TIMEOUT_SECONDS"), default=120, minimum=5),
            request_timeout_seconds_stream=_to_int(os.getenv("REQUEST_TIMEOUT_SECONDS_STREAM"), default=150, minimum=5),
            run_retry_attempts=_to_int(os.getenv("RUN_RETRY_ATTEMPTS"), default=1, minimum=0),
            retry_backoff_ms=_to_int(os.getenv("RETRY_BACKOFF_MS"), default=300, minimum=0),
            degrade_on_timeout=_to_bool(os.getenv("DEGRADE_ON_TIMEOUT"), default=True),
            degrade_on_error=_to_bool(os.getenv("DEGRADE_ON_ERROR"), default=True),
            app_auth_enabled=_to_bool(os.getenv("APP_AUTH_ENABLED"), default=False),
            app_api_key=os.getenv("APP_API_KEY"),
            max_query_chars=_to_int(os.getenv("MAX_QUERY_CHARS"), default=2000, minimum=1),
            max_context_chars=_to_int(os.getenv("MAX_CONTEXT_CHARS"), default=8000, minimum=1),
            prompt_injection_guard_enabled=_to_bool(os.getenv("PROMPT_INJECTION_GUARD_ENABLED"), default=True),
            app_cors_origins=_to_csv(
                os.getenv("APP_CORS_ORIGINS"),
                default=["http://127.0.0.1:3000", "http://localhost:3000"],
            ),
            app_cors_allow_credentials=_to_bool(os.getenv("APP_CORS_ALLOW_CREDENTIALS"), default=False),
            agent_verbose=_to_bool(os.getenv("AGENT_VERBOSE"), default=False),
            job_db_path=os.getenv("JOB_DB_PATH", ".run/jobs.db"),
        )
