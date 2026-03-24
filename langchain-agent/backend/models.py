from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Agent 运行请求。"""

    query: str = Field(..., min_length=1, description="用户输入的问题描述")
    context: Dict[str, Any] = Field(default_factory=dict, description="可选结构化上下文")
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="可选会话 ID。相同 session_id 会复用内存中的短期历史。",
    )


class StepRecord(BaseModel):
    """单个执行步骤的输入输出快照。"""

    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    duration_ms: int = Field(ge=0)


class Metrics(BaseModel):
    """本次请求的关键运行指标。"""

    latency_ms: int = Field(ge=0)
    fallback_used: bool


class RunResponse(BaseModel):
    """Agent 返回结果。"""

    final_answer: str
    steps: List[StepRecord]
    metrics: Metrics
