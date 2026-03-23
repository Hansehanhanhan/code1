from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.model_client import OpenAIModelClient
from agent.state_machine import SimpleAgentStateMachine
from backend.models import RunRequest, RunResponse
from backend.settings import Settings

app = FastAPI(title="Merchant Ops Copilot", version="0.2.0")
settings = Settings.from_env()

# 允许前端开发环境跨域访问后端接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_client = (
    OpenAIModelClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
    if settings.openai_api_key
    else None
)

state_machine = SimpleAgentStateMachine(
    model_client=model_client,
    allow_rule_fallback=settings.allow_rule_fallback,
)


@app.get("/health")
def health() -> dict:
    # 健康检查接口，用于本地联调和部署探活。
    return {
        "status": "ok",
        "model_enabled": state_machine.model_enabled,
        "allow_rule_fallback": settings.allow_rule_fallback,
        "model": settings.openai_model,
    }


@app.post("/run", response_model=RunResponse)
def run_agent(request: RunRequest) -> RunResponse:
    # 统一入口：接收请求后交给状态机执行完整 Agent 流程。
    try:
        return state_machine.run(request.query, request.context)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
