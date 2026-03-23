from __future__ import annotations

from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.settings import Settings
from backend.models import RunRequest, RunResponse
from agent.agent import run_agent

# 初始化 FastAPI
app = FastAPI(
    title="Merchant Ops Copilot (LangChain 版)",
    description="基于 LangChain 的商家运营助手",
    version="0.2.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载配置
settings = Settings.from_env()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Merchant Ops Copilot (LangChain 版)",
        "version": "0.2.0",
        "framework": "LangChain"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest):
    """运行 Agent（LangChain 版本）。"""

    try:
        # 调用 LangChain Agent
        response = run_agent(request.query, request.context)
        return response

    except Exception as e:
        # 错误处理
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent 执行失败: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
