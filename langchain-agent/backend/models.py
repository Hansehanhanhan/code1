from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Agent 运行请求。"""

    # 用户主问题，后端会对其进行长度与安全规则校验。
    query: str = Field(..., min_length=1, description="用户输入的问题描述")
    # 结构化上下文（如 merchant_id/category/time_range）。
    context: Dict[str, Any] = Field(default_factory=dict, description="可选结构化上下文")
    # 会话键：相同 session_id 会命中同一段短期记忆历史。
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="可选会话 ID。相同 session_id 会复用内存中的短期历史。",
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="可选幂等键。相同 key 的重复提交会复用已创建任务，避免重复入队。",
    )


class StepRecord(BaseModel):
    """单个执行步骤的输入输出快照。"""

    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    # 该步骤耗时（毫秒）。
    duration_ms: int = Field(ge=0)


class Metrics(BaseModel):
    """本次请求的关键运行指标。"""

    # 端到端请求耗时。
    latency_ms: int = Field(ge=0)
    fallback_used: bool
    # 分阶段指标（便于做性能归因）。
    llm_latency_ms: int = Field(default=0, ge=0)
    tool_latency_ms: int = Field(default=0, ge=0)
    loop_count: int = Field(default=0, ge=0)
    retrieve_hits: int = Field(default=0, ge=0)
    # 流式场景指标：首包延迟、事件数、完整性。
    ttfb_ms: int | None = Field(default=None, ge=0)
    event_count: int | None = Field(default=None, ge=0)
    event_completeness: bool | None = None


class RunResponse(BaseModel):
    """Agent 返回结果。"""

    final_answer: str
    steps: List[StepRecord]
    metrics: Metrics


class JobCreateResponse(BaseModel):
    """异步任务创建返回。"""

    job_id: str
    status: str
    created_at: float
    retry_of: str | None = None


class JobCancelResponse(BaseModel):
    """异步任务取消返回。"""

    job_id: str
    status: str
    cancelled: bool
    message: str


class JobEvent(BaseModel):
    """任务事件（用于回放/流式转发）。"""

    id: int
    type: str
    content: Dict[str, Any]
    created_at: float


class JobStatusResponse(BaseModel):
    """任务状态查询返回。"""

    job_id: str
    status: str
    created_at: float
    updated_at: float
    error_message: str | None = None
    response: RunResponse | None = None
