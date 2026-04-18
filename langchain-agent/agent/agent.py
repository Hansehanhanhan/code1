from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, Callable

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from backend.models import Metrics, RunResponse, StepRecord
from backend.session_store import SessionStore, get_session_store
from backend.settings import Settings
from rag import retrieve_knowledge
from tools.tools import ads_analyze, inventory_check, product_diagnose, traffic_analyze

# 会话相关运行参数（当前实现为进程内短期记忆）。
DEFAULT_SESSION_ID = "default"
MAX_HISTORY_TURNS = 8
# ReAct 执行安全阈值，避免无限循环或超长占用。
MAX_AGENT_ITERATIONS = 12
MAX_AGENT_EXECUTION_SECONDS = 90
ROUTED_AGENT_ITERATIONS = 8

ANALYSIS_TOOL_NAMES = [
    "traffic_analyze",
    "ads_analyze",
    "inventory_check",
    "product_diagnose",
]

TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "traffic_analyze": ("流量", "曝光", "点击", "ctr", "visit", "traffic", "impression"),
    "ads_analyze": ("广告", "投放", "roi", "cpc", "cpm", "ad", "campaign"),
    "inventory_check": ("库存", "补货", "缺货", "积压", "周转", "stock", "inventory"),
    "product_diagnose": ("转化", "详情页", "主图", "标题", "定价", "conversion", "cvr"),
    "retrieve_knowledge": ("sop", "政策", "规则", "手册", "规范", "指南", "policy", "knowledge"),
}

BROAD_QUERY_KEYWORDS = ("综合", "整体", "全面", "排查", "诊断", "分析全部", "全链路", "all", "overall")

# 会话记忆：{session_id: [(user_query, assistant_answer), ...]}
# 事件回调类型：用于 SSE 逐步推送 Agent 执行事件。
EventSink = Callable[[dict[str, Any]], None]
# 工具函数类型：输入 query/context，输出结构化字典结果。
ToolFn = Callable[[str, dict[str, Any]], dict[str, Any]]

logger = logging.getLogger("merchant_ops.agent")


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def _preview(value: Any, max_len: int = 200) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _extract_thought(action_log: str) -> str:
    # 从 ReAct 日志中提取 Thought 文本，便于前端可视化展示。
    match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\Z)", action_log, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _normalize_value(value: Any) -> Any:
    # 工具输入/输出可能是 JSON 字符串，这里统一尝试反序列化。
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

    def __init__(
        self,
        event_sink: EventSink | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        request_id: str | None = None,
    ) -> None:
        self.steps: list[StepRecord] = []
        self._loop_index = 0
        self._pending: dict[str, Any] | None = None
        self._event_sink = event_sink
        self._session_id = session_id
        self._request_id = request_id
        self.total_tool_latency_ms = 0
        self.total_llm_latency_ms = 0
        self.retrieve_hits = 0
        self._llm_started_at: dict[Any, float] = {}

    def _emit(self, event_type: str, content: dict[str, Any]) -> None:
        # 通过事件回调把中间过程推送给 SSE 流。
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, "content": content})
        except Exception:
            # Never fail the run if SSE push fails.
            return

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        # 记录一轮 ReAct 的起点：Thought + Action + Action Input。
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
        _log_event(
            "react_agent_action",
            request_id=self._request_id,
            session_id=self._session_id,
            loop_index=self._loop_index,
            action=action_name,
            thought=_preview(thought, max_len=160),
            action_input=_preview(action_input, max_len=240),
        )

    def _mark_llm_start(self, run_id: Any) -> None:
        if run_id is None:
            return
        self._llm_started_at[run_id] = perf_counter()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        del serialized, prompts
        self._mark_llm_start(kwargs.get("run_id"))

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any) -> Any:
        del serialized, messages
        self._mark_llm_start(kwargs.get("run_id"))

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        del response
        run_id = kwargs.get("run_id")
        if run_id is None:
            return
        started_at = self._llm_started_at.pop(run_id, None)
        if started_at is None:
            return
        duration_ms = max(0, int((perf_counter() - started_at) * 1000))
        self.total_llm_latency_ms += duration_ms
        self._emit(
            "llm_observation",
            {
                "loop_index": self._loop_index,
                "duration_ms": duration_ms,
                "llm_latency_ms": self.total_llm_latency_ms,
            },
        )
        _log_event(
            "react_llm_observation",
            request_id=self._request_id,
            session_id=self._session_id,
            loop_index=self._loop_index,
            duration_ms=duration_ms,
            total_llm_latency_ms=self.total_llm_latency_ms,
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        # 在工具执行结束时补全 Observation 与耗时。
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
        self.total_tool_latency_ms += duration_ms
        observation = _normalize_value(output)
        if self._pending["action"] == "retrieve_knowledge" and isinstance(observation, dict):
            matches = observation.get("data", {}).get("matches")
            if isinstance(matches, list):
                self.retrieve_hits += len(matches)
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
                "tool_latency_ms": self.total_tool_latency_ms,
                "retrieve_hits": self.retrieve_hits,
            },
        )
        _log_event(
            "react_tool_observation",
            request_id=self._request_id,
            session_id=self._session_id,
            loop_index=self._pending["loop_index"],
            action=self._pending["action"],
            duration_ms=duration_ms,
            total_tool_latency_ms=self.total_tool_latency_ms,
            retrieve_hits=self.retrieve_hits,
            observation_preview=_preview(observation, max_len=240),
        )
        self._pending = None


def _cleanup_markdown(text: str) -> str:
    # 为前端展示做兜底清洗：去掉可能出现的 Markdown 粗体标记。
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    return cleaned.replace("**", "")


def _normalize_session_id(session_id: str | None) -> str:
    # 统一 session_id，空值时回退到默认会话。
    if session_id is None:
        return DEFAULT_SESSION_ID
    normalized = session_id.strip()
    return normalized if normalized else DEFAULT_SESSION_ID


def _get_history_text(session_store: SessionStore, session_id: str) -> str:
    # 将历史对话拼成纯文本，注入到当前 ReAct 提示中。
    turns = session_store.get_history(session_id)
    if not turns:
        return ""

    lines: list[str] = []
    for user_query, assistant_answer in turns:
        lines.append(f"Human: {user_query}")
        lines.append(f"Assistant: {assistant_answer}")
    return "\n".join(lines)


def _append_history(
    session_store: SessionStore,
    settings: Settings,
    session_id: str,
    query: str,
    final_answer: str,
) -> None:
    # 只保留最近 N 轮，避免上下文无限膨胀。
    session_store.append_turn(
        session_id,
        query,
        final_answer,
        max_history_turns=MAX_HISTORY_TURNS,
        ttl_seconds=settings.session_ttl_seconds,
    )


def _parse_tool_input(tool_input: str) -> tuple[str, dict[str, Any]]:
    # ReAct 工具输入约定是 JSON 字符串：{"query": "...", "context": {...}}
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


def _compose_routing_text(query: str, context: dict[str, Any]) -> str:
    return f"{query} {json.dumps(context, ensure_ascii=False)}".lower()


def _route_tools(query: str, context: dict[str, Any], rag_enabled: bool) -> tuple[list[str], str]:
    text = _compose_routing_text(query, context)
    scores: dict[str, int] = {}
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name == "retrieve_knowledge" and not rag_enabled:
            continue
        scores[tool_name] = sum(1 for kw in keywords if kw.lower() in text)

    selected = [name for name in ANALYSIS_TOOL_NAMES if scores.get(name, 0) > 0]
    route_reason = "all_tools_default"
    if selected:
        route_reason = f"keyword_matched:{','.join(selected)}"
    else:
        selected = list(ANALYSIS_TOOL_NAMES)

    if rag_enabled and scores.get("retrieve_knowledge", 0) > 0:
        selected.append("retrieve_knowledge")
        route_reason = f"{route_reason}+knowledge"

    return selected, route_reason


def _should_short_circuit(query: str, selected_tool_names: list[str]) -> bool:
    if len(selected_tool_names) != 1:
        return False
    tool_name = selected_tool_names[0]
    if tool_name == "retrieve_knowledge":
        return False

    lowered = query.lower()
    if any(token in lowered for token in BROAD_QUERY_KEYWORDS):
        return False
    return True


def _build_react_tool(
    name: str,
    description: str,
    tool_fn: ToolFn,
    request_cache: dict[str, dict[str, Any]] | None = None,
) -> StructuredTool:
    # 把本地 Python 函数包装成 LangChain 可调用的 StructuredTool。
    def _runner(tool_input: str) -> str:
        query, context = _parse_tool_input(tool_input)
        cache_key = f"{name}|{query}|{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
        if request_cache is not None and cache_key in request_cache:
            cached = dict(request_cache[cache_key])
            cached["cached"] = True
            return json.dumps(cached, ensure_ascii=False)

        result = tool_fn(query, context)
        if request_cache is not None:
            request_cache[cache_key] = result
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_runner,
        name=name,
        description=description,
    )


def _build_tools(
    settings: Settings,
    selected_tool_names: list[str] | None = None,
    request_cache: dict[str, dict[str, Any]] | None = None,
) -> list[StructuredTool]:
    # 工具注册表：基础业务工具 + 可选 RAG 检索工具。
    selected = set(selected_tool_names or [*ANALYSIS_TOOL_NAMES, "retrieve_knowledge"])
    tools: list[StructuredTool] = []
    if "traffic_analyze" in selected:
        tools.append(
            _build_react_tool(
                name="traffic_analyze",
                description="Analyze traffic trend. Input JSON must contain query and can include context.",
                tool_fn=traffic_analyze,
                request_cache=request_cache,
            )
        )
    if "ads_analyze" in selected:
        tools.append(
            _build_react_tool(
                name="ads_analyze",
                description="Analyze ad efficiency and ROI. Input JSON must contain query and can include context.",
                tool_fn=ads_analyze,
                request_cache=request_cache,
            )
        )
    if "inventory_check" in selected:
        tools.append(
            _build_react_tool(
                name="inventory_check",
                description="Check inventory risk. Input JSON must contain query and can include context.",
                tool_fn=inventory_check,
                request_cache=request_cache,
            )
        )
    if "product_diagnose" in selected:
        tools.append(
            _build_react_tool(
                name="product_diagnose",
                description="Diagnose product conversion. Input JSON must contain query and can include context.",
                tool_fn=product_diagnose,
                request_cache=request_cache,
            )
        )

    if settings.rag_enabled and "retrieve_knowledge" in selected:
        # 开启 RAG 时，允许 Agent 主动检索 SOP/策略知识片段。
        tools.append(
            _build_react_tool(
                name="retrieve_knowledge",
                description=(
                    "Retrieve SOP and policy snippets from local knowledge base. "
                    "Input JSON must contain query and can include context."
                ),
                tool_fn=lambda query, context: retrieve_knowledge(query, context, settings),
                request_cache=request_cache,
            )
        )

    return tools


def _run_single_tool(tool_name: str, settings: Settings, query: str, context: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "traffic_analyze":
        return traffic_analyze(query, context)
    if tool_name == "ads_analyze":
        return ads_analyze(query, context)
    if tool_name == "inventory_check":
        return inventory_check(query, context)
    if tool_name == "product_diagnose":
        return product_diagnose(query, context)
    if tool_name == "retrieve_knowledge":
        return retrieve_knowledge(query, context, settings)
    raise ValueError(f"Unknown tool: {tool_name}")


def create_agent(
    settings: Settings,
    callbacks: list[BaseCallbackHandler] | None = None,
    *,
    selected_tool_names: list[str] | None = None,
    request_cache: dict[str, dict[str, Any]] | None = None,
) -> AgentExecutor:
    """Create a ReAct Agent executor."""

    # LLM 客户端：支持 OpenAI 兼容接口。
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.3,
    )
    tools = _build_tools(settings, selected_tool_names=selected_tool_names, request_cache=request_cache)

    # 根据配置动态提示模型是否可用知识检索工具。
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

    # 先注入动态提示，再构建 ReAct Agent。
    prompt = PromptTemplate.from_template(template).partial(knowledge_hint=knowledge_hint)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks or [],
        verbose=settings.agent_verbose,
        handle_parsing_errors=True,
        max_iterations=ROUTED_AGENT_ITERATIONS if selected_tool_names else MAX_AGENT_ITERATIONS,
        max_execution_time=MAX_AGENT_EXECUTION_SECONDS,
        early_stopping_method="generate",
    )


def run_agent(
    query: str,
    context: dict[str, Any],
    session_id: str | None = None,
    event_sink: EventSink | None = None,
    request_id: str | None = None,
) -> RunResponse:
    """Run ReAct agent and return structured response."""

    settings = Settings.from_env()
    session_store = get_session_store(settings)

    sid = _normalize_session_id(session_id)
    trace_callback = ReActTraceCallbackHandler(
        event_sink=event_sink,
        session_id=sid,
        request_id=request_id,
    )
    selected_tool_names, route_reason = _route_tools(query, context, settings.rag_enabled)
    request_cache: dict[str, dict[str, Any]] = {}
    agent_executor = create_agent(
        settings,
        callbacks=[trace_callback],
        selected_tool_names=selected_tool_names,
        request_cache=request_cache,
    )

    history_text = _get_history_text(session_store, sid)
    enhanced_input = (
        f"User question: {query}\n"
        f"Context JSON: {json.dumps(context, ensure_ascii=False)}"
    )

    _log_event(
        "agent_run_started",
        request_id=request_id,
        session_id=sid,
        query_preview=_preview(query, max_len=120),
        context_keys=sorted((context or {}).keys()),
        selected_tools=selected_tool_names,
        route_reason=route_reason,
    )

    started_at = perf_counter()
    if _should_short_circuit(query, selected_tool_names):
        tool_name = selected_tool_names[0]
        tool_started = perf_counter()
        tool_output = _run_single_tool(tool_name, settings, query, context)
        tool_duration_ms = max(0, int((perf_counter() - tool_started) * 1000))
        final_answer = _cleanup_markdown(
            "问题摘要：命中单一高相关工具，使用快速模式直达输出。\n"
            f"核心发现：{tool_output.get('summary', '暂无')}\n"
            "行动计划：\n"
            "1. 先执行工具返回的高优先建议。\n"
            "2. 观察 24-72 小时关键指标变化。\n"
            "3. 若未改善，再进入多工具综合诊断。"
        )
        _append_history(session_store, settings, sid, query, final_answer)
        latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        steps: list[StepRecord] = [
            StepRecord(
                name="Early Stop Route",
                input={"query": query, "context": context, "selected_tool": tool_name},
                output={"observation": tool_output},
                duration_ms=tool_duration_ms,
            ),
            StepRecord(
                name="Agent",
                input={"query": query, "context": context, "session_id": sid},
                output={"result": final_answer},
                duration_ms=latency_ms,
            ),
        ]
        _log_event(
            "agent_run_early_stopped",
            request_id=request_id,
            session_id=sid,
            selected_tool=tool_name,
            route_reason=route_reason,
            latency_ms=latency_ms,
        )
        return RunResponse(
            final_answer=final_answer,
            steps=steps,
            metrics=Metrics(
                latency_ms=latency_ms,
                fallback_used=False,
                llm_latency_ms=0,
                tool_latency_ms=tool_duration_ms,
                loop_count=1,
                retrieve_hits=len(tool_output.get("data", {}).get("matches", []))
                if isinstance(tool_output, dict)
                else 0,
            ),
        )

    try:
        result = agent_executor.invoke(
            {
                "input": enhanced_input,
                "chat_history": history_text,
            }
        )
    except Exception as exc:
        latency_ms = int((perf_counter() - started_at) * 1000)
        logger.exception(
            json.dumps(
                {
                    "event": "agent_run_failed",
                    "request_id": request_id,
                    "session_id": sid,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                },
                ensure_ascii=False,
                default=str,
            )
        )
        raise

    latency_ms = int((perf_counter() - started_at) * 1000)

    final_answer = _cleanup_markdown(str(result["output"]))
    _append_history(session_store, settings, sid, query, final_answer)

    # steps = ReAct 每轮轨迹 + 一条总览 Agent 结果。
    steps: list[StepRecord] = trace_callback.steps + [
        StepRecord(
            name="Agent",
            input={"query": query, "context": context, "session_id": sid},
            output={"result": final_answer},
            duration_ms=latency_ms,
        )
    ]

    _log_event(
        "agent_run_succeeded",
        request_id=request_id,
        session_id=sid,
        latency_ms=latency_ms,
        step_count=len(trace_callback.steps),
        llm_latency_ms=trace_callback.total_llm_latency_ms,
        tool_latency_ms=trace_callback.total_tool_latency_ms,
        loop_count=len(trace_callback.steps),
        retrieve_hits=trace_callback.retrieve_hits,
        selected_tools=selected_tool_names,
        route_reason=route_reason,
        final_answer_preview=_preview(final_answer, max_len=160),
    )

    return RunResponse(
        final_answer=final_answer,
        steps=steps,
        metrics=Metrics(
            latency_ms=latency_ms,
            fallback_used=False,
            llm_latency_ms=trace_callback.total_llm_latency_ms,
            tool_latency_ms=trace_callback.total_tool_latency_ms,
            loop_count=len(trace_callback.steps),
            retrieve_hits=trace_callback.retrieve_hits,
        ),
    )
