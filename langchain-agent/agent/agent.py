from __future__ import annotations

import json
import re
from collections import defaultdict
from threading import RLock
from time import perf_counter
from typing import Any, Callable

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from backend.models import Metrics, RunResponse, StepRecord
from backend.settings import Settings
from rag import retrieve_knowledge
from tools.tools import ads_analyze, inventory_check, product_diagnose, traffic_analyze

DEFAULT_SESSION_ID = "default"
MAX_HISTORY_TURNS = 8
MAX_AGENT_ITERATIONS = 12
MAX_AGENT_EXECUTION_SECONDS = 90
_session_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
_session_lock = RLock()
EventSink = Callable[[dict[str, Any]], None]
ToolFn = Callable[[str, dict[str, Any]], dict[str, Any]]


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
        self._pending: dict[str, Any] | None = None
        self._event_sink = event_sink

    def _emit(self, event_type: str, content: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, "content": content})
        except Exception:
            # Never fail the run if SSE push fails.
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


def _parse_tool_input(tool_input: str) -> tuple[str, dict[str, Any]]:
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


def _build_react_tool(name: str, description: str, tool_fn: ToolFn) -> StructuredTool:
    def _runner(tool_input: str) -> str:
        query, context = _parse_tool_input(tool_input)
        result = tool_fn(query, context)
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_runner,
        name=name,
        description=description,
    )


def _build_tools(settings: Settings) -> list[StructuredTool]:
    tools: list[StructuredTool] = [
        _build_react_tool(
            name="traffic_analyze",
            description="Analyze traffic trend. Input JSON must contain query and can include context.",
            tool_fn=traffic_analyze,
        ),
        _build_react_tool(
            name="ads_analyze",
            description="Analyze ad efficiency and ROI. Input JSON must contain query and can include context.",
            tool_fn=ads_analyze,
        ),
        _build_react_tool(
            name="inventory_check",
            description="Check inventory risk. Input JSON must contain query and can include context.",
            tool_fn=inventory_check,
        ),
        _build_react_tool(
            name="product_diagnose",
            description="Diagnose product conversion. Input JSON must contain query and can include context.",
            tool_fn=product_diagnose,
        ),
    ]

    if settings.rag_enabled:
        tools.append(
            _build_react_tool(
                name="retrieve_knowledge",
                description=(
                    "Retrieve SOP and policy snippets from local knowledge base. "
                    "Input JSON must contain query and can include context."
                ),
                tool_fn=lambda query, context: retrieve_knowledge(query, context, settings),
            )
        )

    return tools


def create_agent(settings: Settings, callbacks: list[BaseCallbackHandler] | None = None) -> AgentExecutor:
    """Create a ReAct Agent executor."""

    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.3,
    )
    tools = _build_tools(settings)

    knowledge_hint = (
        "If the question needs SOP or policy knowledge, call retrieve_knowledge first."
        if settings.rag_enabled
        else "Knowledge retrieval tool is disabled. Use only available analysis tools."
    )

    template = """You are an ecommerce operations analyst assistant.
You must call tools to gather evidence before concluding.
Do not fabricate any Observation.
{knowledge_hint}
Thought and Final Answer must be in Chinese.
Output plain text only, do not use Markdown bold markers (**).
Final Answer should include:
Problem Summary:
Root Causes:
1. ...
2. ...
3. ...
Action Plan:
1. ...
2. ...
3. ...
Risks and Follow-up:
...

Available tools:
{tools}

Chat history (may be empty):
{chat_history}

Use this exact ReAct format:
Question: user question
Thought: your reasoning in Chinese
Action: one of [{tool_names}]
Action Input: a JSON string, e.g. {{"query":"traffic dropped this week","context":{{"merchant_id":"demo-001"}}}}
Observation: tool output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: If evidence is already sufficient, stop tool calls and provide final answer.
Thought: I now know the final answer
Final Answer: final response to user in Chinese

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(knowledge_hint=knowledge_hint)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks or [],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=MAX_AGENT_ITERATIONS,
        max_execution_time=MAX_AGENT_EXECUTION_SECONDS,
        early_stopping_method="generate",
    )


def run_agent(
    query: str,
    context: dict[str, Any],
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
