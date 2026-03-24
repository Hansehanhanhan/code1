from __future__ import annotations

import asyncio
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent.agent import run_agent
from backend.models import RunRequest, RunResponse
from backend.settings import Settings

app = FastAPI(
    title="Merchant Ops Copilot (LangChain ReAct)",
    description="LangChain ReAct backend for merchant operations assistant.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings.from_env()


def _ensure_api_key() -> None:
    current_settings = Settings.from_env()
    if not current_settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")


@app.get("/")
async def root() -> dict:
    return {
        "message": "Merchant Ops Copilot (LangChain ReAct)",
        "version": "0.3.0",
        "framework": "LangChain",
        "agent_type": "ReAct",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    _ensure_api_key()

    try:
        return await run_in_threadpool(
            run_agent,
            request.query,
            request.context,
            request.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc


@app.post("/run_stream")
async def run_stream(request: RunRequest) -> StreamingResponse:
    _ensure_api_key()

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    async def worker() -> None:
        try:
            response = await run_in_threadpool(
                run_agent,
                request.query,
                request.context,
                request.session_id,
                emit,
            )
            emit({"type": "final_response", "content": response.model_dump()})
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "content": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_generator():
        task = asyncio.create_task(worker())
        while True:
            data = await queue.get()
            if data is None:
                break
            yield f"data: {data}\n\n"
        await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
