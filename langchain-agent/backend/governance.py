from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Callable, TypeVar

from backend.models import Metrics, RunResponse, StepRecord

T = TypeVar("T")
AttemptFailedHook = Callable[[int, int, bool, Exception], None]


def is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def is_retryable_error(exc: Exception) -> bool:
    if is_timeout_error(exc):
        return True
    return isinstance(exc, (ConnectionError, RuntimeError))


async def run_with_governance(
    fn: Callable[[], T],
    *,
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_ms: int,
    on_attempt_failed: AttemptFailedHook | None = None,
) -> tuple[T, int]:
    last_exc: Exception | None = None
    max_attempts = max(1, retry_attempts + 1)

    for attempt in range(1, max_attempts + 1):
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(fn),
                timeout=float(timeout_seconds),
            )
            return result, attempt
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            retryable = is_retryable_error(exc)
            if on_attempt_failed is not None:
                on_attempt_failed(attempt, max_attempts, retryable, exc)
            if attempt >= max_attempts or not retryable:
                break
            backoff_seconds = max(0.0, (retry_backoff_ms / 1000.0) * (2 ** (attempt - 1)))
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds)

    assert last_exc is not None
    raise last_exc


def build_degraded_response(
    query: str,
    context: dict[str, Any],
    session_id: str,
    *,
    reason: str,
    latency_ms: int,
) -> RunResponse:
    fallback_text = (
        "问题摘要：当前请求触发了稳定性降级路径，已返回安全兜底建议。\n"
        f"降级原因：{reason}\n"
        "行动计划：\n"
        "1. 先检查核心指标（流量、转化、广告ROI）是否持续恶化。\n"
        "2. 暂停低效投放，优先保障高转化商品和高价值关键词。\n"
        "3. 待服务恢复后重新发起分析，获取完整多工具诊断结果。"
    )
    return RunResponse(
        final_answer=fallback_text,
        steps=[
            StepRecord(
                name="DegradedFallback",
                input={"query": query, "context": context, "session_id": session_id},
                output={"reason": reason},
                duration_ms=max(0, latency_ms),
            )
        ],
        metrics=Metrics(
            latency_ms=max(0, latency_ms),
            fallback_used=True,
            llm_latency_ms=0,
            tool_latency_ms=0,
            loop_count=0,
            retrieve_hits=0,
        ),
    )


def elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))

