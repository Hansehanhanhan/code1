from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from project root.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int, minimum: int = 1) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


@dataclass
class Settings:
    """Runtime settings."""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    allow_rule_fallback: bool
    rag_enabled: bool
    rag_docs_dir: str
    rag_vector_backend: str
    rag_top_k: int
    rag_embedding_model: str
    rag_embedding_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            allow_rule_fallback=_to_bool(os.getenv("ALLOW_RULE_FALLBACK"), default=True),
            rag_enabled=_to_bool(os.getenv("RAG_ENABLED"), default=True),
            rag_docs_dir=os.getenv("RAG_DOCS_DIR", "knowledge/seed"),
            rag_vector_backend=(os.getenv("RAG_VECTOR_BACKEND", "chroma").strip().lower() or "chroma"),
            rag_top_k=_to_int(os.getenv("RAG_TOP_K"), default=3, minimum=1),
            rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            rag_embedding_device=(os.getenv("RAG_EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"),
        )
