from __future__ import annotations

import json
import re
from collections import defaultdict
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Dict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI

from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings
from tools.tools import ads_analyze, inventory_check, product_diagnose, traffic_analyze

DEFAULT_SESSION_ID = "default"
MAX_HISTORY_TURNS = 8
_session_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
_session_lock = RLock()
EventSink = Callable[[Dict[str, Any]], None]


def _extract_thought(action_log: str) -> str:
    match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\Z)", action_log, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


class ReActTraceCallbackHandler(BaseCallbackHandler):
    """Collect ReAct loop traces and convert them to StepRecord."""

    def __init__(self, event_sink: EventSink | None = None) -> None:
        self.steps: list[StepRecord] = []
        self._loop_index = 0
        self._pending: Dict[str, Any] | None = None
        self._event_sink = event_sink

    def _emit(self, event_type: str, content: Dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, "content": content})
        except Exception:
            # Never fail the agent run because downstream streaming failed.
            return

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        self._loop_index += 1
        thought = _extract_thought(getattr(action, "log", ""))
        action_name = getattr(action, "tool", "")
        action_input = _normalize_value(getattr(action, "tool_input", ""))
        self._pending = {
            "loop_index": self._loop_index,
            "thought": thought,
            "action": action_name,
            "action_input": action_input,
            "started_at": perf_counter(),
        }
        self._emit(
            "agent_action",
            {
                "loop_index": self._loop_index,
                "thought": thought,
                "action": action_name,
                "action_input": action_input,
            },
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        if self._pending is None:
            self._loop_index += 1
            self._pending = {
                "loop_index": self._loop_index,
                "thought": "",
                "action": "",
                "action_input": {},
                "started_at": perf_counter(),
            }

        duration_ms = max(0, int((perf_counter() - self._pending["started_at"]) * 1000))
        observation = _normalize_value(output)
        self.steps.append(
            StepRecord(
                name=f"ReAct Loop {self._pending['loop_index']}",
                input={
                    "thought": self._pending["thought"],
                    "action": self._pending["action"],
                    "action_input": self._pending["action_input"],
                },
                output={"observation": observation},
                duration_ms=duration_ms,
            )
        )
        self._emit(
            "tool_observation",
            {
                "loop_index": self._pending["loop_index"],
                "observation": observation,
                "duration_ms": duration_ms,
            },
        )
        self._pending = None


def _cleanup_markdown(text: str) -> str:
    # Keep plain-text readability in UI by removing bold markers.
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    return cleaned.replace("**", "")


def _normalize_session_id(session_id: str | None) -> str:
    if session_id is None:
        return DEFAULT_SESSION_ID
    normalized = session_id.strip()
    return normalized if normalized else DEFAULT_SESSION_ID


def _get_history_text(session_id: str) -> str:
    with _session_lock:
        turns = list(_session_histories.get(session_id, []))
    if not turns:
        return ""
    lines: list[str] = []
    for user_query, assistant_answer in turns:
        lines.append(f"Human: {user_query}")
        lines.append(f"Assistant: {assistant_answer}")
    return "\n".join(lines)


def _append_history(session_id: str, query: str, final_answer: str) -> None:
    with _session_lock:
        history = _session_histories[session_id]
        history.append((query, final_answer))
        if len(history) > MAX_HISTORY_TURNS:
            _session_histories[session_id] = history[-MAX_HISTORY_TURNS:]


def _parse_tool_input(tool_input: str) -> tuple[str, Dict[str, Any]]:
    raw = tool_input.strip()
    if not raw:
        return "", {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}

    if isinstance(payload, dict):
        query = payload.get("query")
        context = payload.get("context")
        normalized_query = query if isinstance(query, str) else raw
        normalized_context = context if isinstance(context, dict) else {}
        return normalized_query, normalized_context

    return raw, {}


def _build_react_tool(
    name: str,
    description: str,
    tool_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> StructuredTool:
    def _runner(tool_input: str) -> str:
        query, context = _parse_tool_input(tool_input)
        result = tool_fn(query, context)
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_runner,
        name=name,
        description=description,
    )


def _build_tools() -> list[StructuredTool]:
    return [
        _build_react_tool(
            name="traffic_analyze",
            description=(
                "流量趋势分析工具。入参应为 JSON 字符串，包含 `query`，可选 `context`。"
            ),
            tool_fn=traffic_analyze,
        ),
        _build_react_tool(
            name="ads_analyze",
            description=(
                "广告效率与 ROI 分析工具。入参应为 JSON 字符串，包含 `query`，可选 `context`。"
            ),
            tool_fn=ads_analyze,
        ),
        _build_react_tool(
            name="inventory_check",
            description=(
                "库存风险检查工具。入参应为 JSON 字符串，包含 `query`，可选 `context`。"
            ),
            tool_fn=inventory_check,
        ),
        _build_react_tool(
            name="product_diagnose",
            description=(
                "商品转化诊断工具。入参应为 JSON 字符串，包含 `query`，可选 `context`。"
            ),
            tool_fn=product_diagnose,
        ),
    ]


def create_agent(settings: Settings, callbacks: list[BaseCallbackHandler] | None = None) -> AgentExecutor:
    """Create a ReAct Agent executor."""

    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.3,
    )

    tools = _build_tools()
    template = """你是电商运营分析助手。
你必须优先调用工具获取证据，严禁编造 Observation。
请使用中文进行 Thought 和 Final Answer。
最终输出请使用纯文本，不要使用 Markdown 加粗符号 **。
Final Answer 必须有清晰结构，按下面模板输出，每一项单独换行：
问题判断：
核心原因：
1. ...
2. ...
3. ...
行动建议：
1. ...
2. ...
3. ...
风险与复盘：
...

你可以使用如下工具：
{tools}

会话历史（可能为空）：
{chat_history}

请严格使用如下格式：
Question: 用户问题
Thought: 你的思考（中文）
Action: 从 [{tool_names}] 里选择一个工具
Action Input: JSON 字符串，例如 {{"query": "本周流量下滑", "context": {{"merchant_id": "demo-001"}}}}
Observation: 工具输出结果
...（可重复 Thought/Action/Action Input/Observation）
Thought: 我现在知道最终答案
Final Answer: 给用户的最终回答（中文）

开始！

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks or [],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )


def run_agent(
    query: str,
    context: Dict[str, Any],
    session_id: str | None = None,
    event_sink: EventSink | None = None,
) -> RunResponse:
    """Run ReAct agent and return structured response."""

    settings = Settings.from_env()
    trace_callback = ReActTraceCallbackHandler(event_sink=event_sink)
    agent_executor = create_agent(settings, callbacks=[trace_callback])

    sid = _normalize_session_id(session_id)
    history_text = _get_history_text(sid)
    enhanced_input = (
        f"User question: {query}\n"
        f"Context JSON: {json.dumps(context, ensure_ascii=False)}"
    )

    started_at = perf_counter()
    result = agent_executor.invoke(
        {
            "input": enhanced_input,
            "chat_history": history_text,
        }
    )
    latency_ms = int((perf_counter() - started_at) * 1000)

    final_answer = _cleanup_markdown(str(result["output"]))
    _append_history(sid, query, final_answer)

    steps: list[StepRecord] = trace_callback.steps + [
        StepRecord(
            name="Agent",
            input={"query": query, "context": context, "session_id": sid},
            output={"result": final_answer},
            duration_ms=latency_ms,
        )
    ]

    return RunResponse(
        final_answer=final_answer,
        steps=steps,
        metrics=Metrics(latency_ms=latency_ms, fallback_used=False),
    )
