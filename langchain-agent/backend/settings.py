from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """运行时配置。"""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    allow_rule_fallback: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            allow_rule_fallback=_to_bool(os.getenv("ALLOW_RULE_FALLBACK"), default=True),
        )
