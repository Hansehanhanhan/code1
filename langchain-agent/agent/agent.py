from __future__ import annotations

import hashlib
import json
import logging
import re
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.models import Metrics, RunResponse, StepRecord
from backend.diagnostic_state import (
    DiagnosticPatch,
    EvidenceRef,
    EvidenceRecord,
    EvidenceValidationError,
    FindingProposal,
    StateConflictError,
    validate_patch_evidence,
)
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
ROUTED_AGENT_ITERATIONS = 12
# AREX 外环 refine 预算：单个请求内最多额外执行的定向查证次数。
MAX_REFINE_ROUNDS = 2

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
REQUIRED_CONTEXT_KEYS = ("merchant_id", "time_range")

REACT_TEMPLATE = """You are an ecommerce operations analyst assistant.
You must call tools to gather evidence before concluding.
Do not fabricate any Observation.
Persist conclusions via update_context, not only into the Final Answer: the server arbitrates and merges them into the session diagnostic state.
Call update_context with an incremental JSON patch when:
1. You complete a diagnostic sub-step (e.g. one tool just returned a conclusion).
2. You detect conflicting evidence between tools.
3. You reject or rule out a candidate direction.
4. Before writing the final answer (submit at least once per request).
Only reference evidence_id and observation_hash values copied verbatim from the current request's tool Observations. Never invent them.
Do not delete historical findings; use supersede_findings when newer evidence replaces one.
Example (evidence fields are illustrative only - copy them from the real Observation):
Action: update_context
Action Input: {{"reason":"traffic_analysis_completed","context_slots":{{"merchant_id":"demo-001"}},"add_findings":[{{"id":"f-1","claim":"近7天流量下滑22%","confidence":"high","evidence":{{"evidence_id":"req_0001:tool_0003","request_id":"req_0001","observation_hash":"sha256:9f86d0..."}}}}],"add_candidates":["广告投放效率下降"],"reject_candidates":["平台流量规则变更（证据不足）"]}}
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

TOOL_CALLING_SYSTEM_PROMPT = """You are an ecommerce operations analyst assistant.
{knowledge_hint}
You must back every conclusion with real tool Observations. Never fabricate evidence.
Persist conclusions via the update_context tool: the server parses, validates and merges your incremental DiagnosticPatch into the session diagnostic state. Call update_context at least once per request, before the final response.

Call update_context when:
1. You complete a diagnostic sub-step (a tool just returned a conclusion).
2. Conflicting evidence appears between tools.
3. A candidate direction is rejected or ruled out.
4. Before writing the final response (submit at least once per request).

update_context input schema (pass the whole patch as the tool_input JSON string):
{{
  "reason": "traffic_analysis_completed",
  "expected_version": 0,
  "context_slots": {{"merchant_id": "demo-001"}},
  "add_findings": [{{
    "id": "f-1",
    "claim": "近7天流量下滑22%",
    "confidence": "high",
    "evidence": {{
      "evidence_id": "req_0001:tool_0003",
      "request_id": "req_0001",
      "observation_hash": "sha256:9f86d0..."
    }}
  }}],
  "add_candidates": ["广告投放效率下降"],
  "reject_candidates": ["平台流量规则变更（证据不足）"],
  "update_constraint_status": {{"merchant_id": "satisfied", "time_range": "satisfied"}},
  "set_answer_confidence": 0.8
}}

Rules:
- Only reference evidence_id / request_id / observation_hash values copied verbatim from the current request's tool Observations. Never invent them.
- Never delete historical findings; use supersede_findings when newer evidence replaces one.
- For each required constraint (merchant_id, time_range and any declared concern) report its status via update_constraint_status (satisfied / unsatisfied / unchecked) and set set_answer_confidence to a 0..1 confidence score.
- Thought and Final Response must be in Chinese.
Final Response must include:
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
..."""

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


def _build_evidence_record(
    observation: Any,
    *,
    request_id: str,
    tool_call_id: str,
    tool_name: str,
) -> dict[str, Any]:
    """Build a request-scoped, deterministic reference for a real observation."""
    normalized = _normalize_value(observation)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    observation_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        "evidence_id": f"{request_id}:{tool_call_id}",
        "request_id": request_id,
        "tool_call_id": tool_call_id,
        "tool": tool_name,
        "observation_hash": f"sha256:{observation_hash}",
        "observation_preview": _preview(normalized, max_len=500),
    }


class ReActTraceCallbackHandler(BaseCallbackHandler):
    """Collect ReAct loop traces and convert them to StepRecord."""

    def __init__(
        self,
        event_sink: EventSink | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        request_id: str | None = None,
        evidence_counter: dict[str, int] | None = None,
    ) -> None:
        self.steps: list[StepRecord] = []
        self._loop_index = 0
        self._pending: dict[str, Any] | None = None
        self._event_sink = event_sink
        self._session_id = session_id
        self._request_id = request_id or f"req_{uuid4().hex}"
        self._evidence_counter = evidence_counter if evidence_counter is not None else {"index": 0}
        self.evidence_records: list[dict[str, Any]] = []
        self.total_tool_latency_ms = 0
        self.total_llm_latency_ms = 0
        self.retrieve_hits = 0
        self._llm_started_at: dict[Any, float] = {}
        self._first_evidence_emitted = False

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
        # 分析工具已将证据块嵌入观察输出（哈希基于原始结果计算），直接复用；
        # 其余工具（如 update_context）由回调兜底生成证据记录。
        embedded = observation.get("evidence") if isinstance(observation, dict) else None
        if isinstance(embedded, dict):
            # 分析工具已将证据块嵌入观察输出（哈希基于原始结果计算）：
            # 直接复用其证据字段；tool_call_id/tool 从 evidence_id 后缀或 action 兜底推导。
            embedded_id = embedded.get("evidence_id", "")
            embedded_call_id = embedded.get("tool_call_id") or (
                embedded_id.split(":", 1)[-1] if ":" in embedded_id else "tool_0000"
            )
            evidence = {
                "evidence_id": embedded_id or f"{self._request_id}:tool_0000",
                "request_id": embedded.get("request_id", self._request_id),
                "tool_call_id": embedded_call_id,
                "tool": embedded.get("tool", self._pending["action"]),
                "observation_hash": embedded.get("observation_hash", ""),
            }
        else:
            self._evidence_counter["index"] += 1
            evidence = _build_evidence_record(
                observation,
                request_id=self._request_id,
                tool_call_id=f"tool_{self._evidence_counter['index']:04d}",
                tool_name=self._pending["action"],
            )
        self.evidence_records.append(evidence)
        if self._pending["action"] == "retrieve_knowledge" and isinstance(observation, dict):
            matches = observation.get("data", {}).get("matches")
            if isinstance(matches, list):
                self.retrieve_hits += len(matches)
        if (
            not self._first_evidence_emitted
            and self._pending["action"] in ANALYSIS_TOOL_NAMES
            and isinstance(observation, dict)
            and observation.get("summary")
            and evidence.get("evidence_id")
        ):
            self._first_evidence_emitted = True
            self._emit(
                "key_step",
                {
                    "kind": "first_evidence",
                    "action": self._pending["action"],
                    "evidence_id": evidence["evidence_id"],
                    "observation_hash": evidence["observation_hash"],
                },
            )
            _log_event(
                "key_step",
                request_id=self._request_id,
                session_id=self._session_id,
                kind="first_evidence",
                action=self._pending["action"],
                evidence_id=evidence["evidence_id"],
            )
        self.steps.append(
            StepRecord(
                name=f"ReAct Loop {self._pending['loop_index']}",
                input={
                    "thought": self._pending["thought"],
                    "action": self._pending["action"],
                    "action_input": self._pending["action_input"],
                },
                output={"observation": observation, "evidence": evidence},
                duration_ms=duration_ms,
            )
        )
        self._emit(
            "tool_observation",
            {
                "loop_index": self._pending["loop_index"],
                "observation": observation,
                "evidence": evidence,
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
            evidence_id=evidence["evidence_id"],
            observation_hash=evidence["observation_hash"],
        )
        self._pending = None

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        # 工具报错也记录到 steps，避免中断时无 action/observation。
        if self._pending is None:
            return
        duration_ms = max(0, int((perf_counter() - self._pending["started_at"]) * 1000))
        self.total_tool_latency_ms += duration_ms
        error_message = str(error)
        self.steps.append(
            StepRecord(
                name=f"ReAct Loop {self._pending['loop_index']}",
                input={
                    "thought": self._pending["thought"],
                    "action": self._pending["action"],
                    "action_input": self._pending["action_input"],
                },
                output={"observation_error": error_message},
                duration_ms=duration_ms,
            )
        )
        self._emit(
            "tool_observation",
            {
                "loop_index": self._pending["loop_index"],
                "observation": {"error": error_message},
                "duration_ms": duration_ms,
                "tool_latency_ms": self.total_tool_latency_ms,
                "retrieve_hits": self.retrieve_hits,
            },
        )
        _log_event(
            "react_tool_error",
            request_id=self._request_id,
            session_id=self._session_id,
            loop_index=self._pending["loop_index"],
            action=self._pending["action"],
            duration_ms=duration_ms,
            total_tool_latency_ms=self.total_tool_latency_ms,
            error_type=type(error).__name__,
            error_message=error_message,
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
    # 将当前状态和历史对话拼成纯文本，注入到当前 ReAct 提示中。
    state = session_store.get_state(session_id)
    turns = session_store.get_history(session_id)
    lines: list[str] = [
        "Current Diagnostic State:",
        state.model_dump_json(ensure_ascii=False),
        "Recent Conversation:",
    ]
    for user_query, assistant_answer in turns:
        lines.append(f"Human: {user_query}")
        lines.append(f"Assistant: {assistant_answer}")
    return "\n".join(lines)


def _merge_context_with_slots(
    session_store: SessionStore,
    settings: Settings,
    session_id: str,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge explicit request context over previously remembered session slots."""
    remembered = session_store.get_context_slots(session_id)
    effective_context = dict(remembered)
    explicit_context = dict(context or {})
    effective_context.update(explicit_context)
    if explicit_context:
        session_store.update_context_slots(
            session_id,
            explicit_context,
            ttl_seconds=settings.session_ttl_seconds,
        )
    return effective_context


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


def _missing_context_keys(context: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_CONTEXT_KEYS:
        value = context.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def _build_clarification_response(
    query: str,
    context: dict[str, Any],
    session_id: str,
    missing_keys: list[str],
) -> RunResponse:
    field_labels = {
        "merchant_id": "商家ID（merchant_id）",
        "time_range": "分析时间范围（time_range）",
    }
    asks = [field_labels.get(key, key) for key in missing_keys]
    ask_text = "、".join(asks)
    final_answer = (
        "为保证分析结论可执行，我需要先补齐关键信息。\n"
        f"请补充：{ask_text}。\n"
        "建议示例：\n"
        "1. merchant_id: demo-001\n"
        "2. time_range: last_7_days\n"
        "补充后我会继续输出根因与行动方案。"
    )
    return RunResponse(
        final_answer=final_answer,
        steps=[
            StepRecord(
                name="Clarification",
                input={"query": query, "context": context, "session_id": session_id},
                output={"missing_context_keys": missing_keys},
                duration_ms=0,
            )
        ],
        metrics=Metrics(
            latency_ms=0,
            fallback_used=False,
            llm_latency_ms=0,
            tool_latency_ms=0,
            loop_count=0,
            retrieve_hits=0,
        ),
    )


def _extract_evidence_from_observation(observation: Any) -> list[str]:
    if not isinstance(observation, dict):
        return []
    lines: list[str] = []
    tool_name = str(observation.get("tool", "")).strip()
    summary = str(observation.get("summary", "")).strip()
    if tool_name and summary:
        lines.append(f"{tool_name}: {summary}")
    data = observation.get("data")
    if tool_name == "retrieve_knowledge" and isinstance(data, dict):
        matches = data.get("matches")
        if isinstance(matches, list):
            sources: list[str] = []
            for item in matches:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if isinstance(source, str) and source.strip() and source not in sources:
                    sources.append(source.strip())
                if len(sources) >= 3:
                    break
            if sources:
                lines.append(f"retrieve_knowledge sources: {', '.join(sources)}")
    return lines


def _collect_evidence_lines(steps: list[StepRecord]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for step in steps:
        observation = step.output.get("observation")
        if observation is None:
            continue
        for line in _extract_evidence_from_observation(observation):
            normalized = line.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            lines.append(normalized)
    return lines[:5]


def _append_evidence_block(final_answer: str, evidence_lines: list[str]) -> str:
    if not evidence_lines:
        return final_answer
    block = ["证据来源："]
    for idx, line in enumerate(evidence_lines, start=1):
        block.append(f"{idx}. {line}")
    return f"{final_answer}\n" + "\n".join(block)


def _extract_candidates_from_steps(steps: list[StepRecord]) -> list[str]:
    # 从工具真实输出中确定性提取候选方向（recommendations），不依赖 LLM 提议。
    candidates: list[str] = []
    for step in steps:
        observation = step.output.get("observation")
        if not isinstance(observation, dict):
            continue
        recommendations = observation.get("recommendations")
        if not isinstance(recommendations, list):
            continue
        for item in recommendations:
            text = str(item).strip()
            if text and text not in candidates:
                candidates.append(text)
    return candidates


def _extract_constraints_from_steps(steps: list[StepRecord]) -> list[str]:
    # 工具执行失败/异常时，把 summary 确认为阻碍诊断的未解决约束。
    constraints: list[str] = []
    for step in steps:
        observation = step.output.get("observation")
        if not isinstance(observation, dict):
            continue
        if observation.get("status") in ("ok", None):
            continue
        text = str(observation.get("summary", "")).strip()
        if text and text not in constraints:
            constraints.append(text)
    return constraints


def _extract_findings_from_steps(
    steps: list[StepRecord], request_id: str
) -> tuple[list[FindingProposal], list[str]]:
    # AREX 兜底：把"带证据的分析工具结论"确定性提升为已验证 findings，
    # 证据引用（evidence_id/observation_hash）直接复制自回调沉淀的 evidence 块。
    proposals: list[FindingProposal] = []
    citations: list[str] = []
    for step in steps:
        if len(proposals) >= 5:
            break
        output = step.output
        observation = output.get("observation")
        evidence = output.get("evidence")
        if not isinstance(observation, dict) or not isinstance(evidence, dict):
            continue
        if str(observation.get("tool", "")) not in ANALYSIS_TOOL_NAMES:
            continue
        recommendations = observation.get("recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            continue
        evidence_id = evidence.get("evidence_id")
        observation_hash = evidence.get("observation_hash")
        if not evidence_id or not observation_hash:
            continue
        for item in recommendations:
            claim = str(item).strip()[:200]
            if not claim:
                continue
            proposal_id = (
                f"auto-{hashlib.sha1(f'{request_id}:{claim}'.encode('utf-8')).hexdigest()[:12]}"
            )
            proposals.append(
                FindingProposal(
                    id=proposal_id,
                    claim=claim,
                    confidence="medium",
                    evidence=EvidenceRef(
                        evidence_id=evidence_id,
                        request_id=request_id,
                        observation_hash=observation_hash,
                    ),
                )
            )
            citations.append(evidence_id)
            if len(proposals) >= 5:
                break
    return proposals, citations


def _audit_constraint_status(
    context: dict[str, Any] | None, unresolved: list[str]
) -> tuple[dict[str, str], float | None]:
    # AREX 审计：确定性逐约束检查。必需上下文缺失/未解决约束 => unsatisfied。
    effective = context or {}
    status: dict[str, str] = {}
    for key in REQUIRED_CONTEXT_KEYS:
        value = effective.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            status[key] = "unsatisfied"
        else:
            status[key] = "satisfied"
    for constraint in unresolved:
        status[constraint] = "unsatisfied"
    total = len(status)
    confidence = (
        round(sum(1 for value in status.values() if value == "satisfied") / total, 2)
        if total
        else None
    )
    return status, confidence


def _build_request_finalize_patch(
    steps: list[StepRecord], expected_version: int, request_id: str = ""
) -> DiagnosticPatch | None:
    add_candidates = _extract_candidates_from_steps(steps)
    add_constraints = _extract_constraints_from_steps(steps)
    add_findings, add_citations = _extract_findings_from_steps(steps, request_id)
    if not add_candidates and not add_constraints and not add_findings:
        return None
    return DiagnosticPatch(
        reason="request_finalize",
        expected_version=expected_version,
        add_candidates=add_candidates,
        add_constraints=add_constraints,
        add_findings=add_findings,
        add_citations=add_citations,
    )


def _finalize_request_state(
    session_store: SessionStore,
    settings: Settings,
    session_id: str,
    request_id: str,
    steps: list[StepRecord],
    context: dict[str, Any] | None = None,
) -> bool:
    """P3 服务端兜底：请求结束时确定性沉淀本轮工具证据，不依赖 LLM 调用 update_context。"""
    current = session_store.get_state(session_id)
    patch = _build_request_finalize_patch(
        steps, expected_version=current.version, request_id=request_id
    )
    if patch is None:
        return False
    current_status, audit_confidence = _audit_constraint_status(context, current.unresolved_constraints)
    patch.update_constraint_status = current_status
    if audit_confidence is not None and current.answer_confidence is None:
        patch.set_answer_confidence = audit_confidence
    existing_finding_ids = {finding.id for finding in current.verified_findings}
    has_new = bool(
        [item for item in patch.add_candidates if item not in current.current_candidates]
        or [item for item in patch.add_constraints if item not in current.unresolved_constraints]
        or [item for item in patch.add_findings if item.id not in existing_finding_ids]
        or [item for item in patch.add_citations if item not in current.verified_citations]
        or any(
            current.constraint_status.get(key) != value
            for key, value in patch.update_constraint_status.items()
        )
        or (
            patch.set_answer_confidence is not None
            and patch.set_answer_confidence != current.answer_confidence
        )
    )
    if not has_new:
        return False
    try:
        updated = session_store.update_state(session_id, patch, ttl_seconds=settings.session_ttl_seconds)
    except StateConflictError:
        current = session_store.get_state(session_id)
        patch.expected_version = current.version
        updated = session_store.update_state(session_id, patch, ttl_seconds=settings.session_ttl_seconds)
    _log_event(
        "state_finalized",
        request_id=request_id,
        session_id=session_id,
        new_version=updated.version,
        added_candidates=patch.add_candidates,
        added_constraints=patch.add_constraints,
        added_findings=[item.id for item in patch.add_findings],
        constraint_status=patch.update_constraint_status,
    )
    return True


def _refine_target_for_constraint(constraint: str) -> str | None:
    text = constraint.lower()
    best_name: str | None = None
    best_score = 0
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name == "retrieve_knowledge":
            continue
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_name, best_score = tool_name, score
    return best_name if best_score > 0 else None


def _run_refine_pass(
    session_store: SessionStore,
    settings: Settings,
    session_id: str,
    request_id: str,
    query: str,
    context: dict[str, Any],
    request_cache: dict[str, dict[str, Any]],
    evidence_counter: dict[str, int],
    steps: list[StepRecord],
) -> int:
    """AREX 外环 refine：针对 unresolved_constraints 定向补查工具证据，预算受限。"""
    refined = 0
    previous_unsatisfied: set[str] | None = None
    for _round in range(MAX_REFINE_ROUNDS):
        state = session_store.get_state(session_id)
        unsatisfied = [
            constraint
            for constraint, status in state.constraint_status.items()
            if status == "unsatisfied"
        ]
        if not unsatisfied:
            break
        current_unsatisfied = set(unsatisfied)
        if previous_unsatisfied is not None and current_unsatisfied == previous_unsatisfied:
            break
        previous_unsatisfied = current_unsatisfied
        target = _refine_target_for_constraint(unsatisfied[0])
        if target is None:
            _log_event(
                "refine_pass_no_target",
                request_id=request_id,
                session_id=session_id,
                constraint_preview=_preview(unsatisfied[0], max_len=120),
            )
            break
        cache_key = f"{target}|{unsatisfied[0]}|{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
        tool_started = perf_counter()
        if cache_key in request_cache:
            tool_output = dict(request_cache[cache_key])
        else:
            tool_output = _run_single_tool(target, settings, unsatisfied[0], context) or {}
            request_cache[cache_key] = dict(tool_output)
        tool_duration_ms = max(0, int((perf_counter() - tool_started) * 1000))
        evidence_counter["index"] += 1
        evidence = _build_evidence_record(
            tool_output,
            request_id=request_id,
            tool_call_id=f"tool_{evidence_counter['index']:04d}",
            tool_name=target,
        )
        steps.append(
            StepRecord(
                name=f"Refine Loop {_round + 1}",
                input={
                    "thought": f"针对未解决约束定向补查 {target}",
                    "action": target,
                    "action_input": {"query": unsatisfied[0], "context": context},
                },
                output={"observation": tool_output, "evidence": evidence},
                duration_ms=tool_duration_ms,
            )
        )
        _log_event(
            "refine_pass_tool_called",
            request_id=request_id,
            session_id=session_id,
            refine_round=_round + 1,
            tool=target,
            constraint_preview=_preview(unsatisfied[0], max_len=120),
            duration_ms=tool_duration_ms,
            evidence_id=evidence["evidence_id"],
        )
        _finalize_request_state(
            session_store,
            settings,
            session_id,
            request_id,
            steps,
            context=context,
        )
        refined += 1
    return refined


def _build_react_tool(
    name: str,
    description: str,
    tool_fn: ToolFn,
    request_cache: dict[str, dict[str, Any]] | None = None,
    request_id: str = "",
    evidence_counter: dict[str, int] | None = None,
) -> StructuredTool:
    # 把本地 Python 函数包装成 LangChain 可调用的 StructuredTool。
    def _build_payload(result: dict[str, Any]) -> dict[str, Any]:
        # 注入请求级证据引用：只嵌入 EvidenceRef 三字段（evidence_id/request_id/observation_hash），
        # 与 update_context 校验所需的证据形状完全一致，避免模型复制到
        # observation_preview / tool_call_id / tool 等额外字段被 pydantic 拒绝。
        if evidence_counter is not None:
            evidence_counter["index"] += 1
            record = _build_evidence_record(
                result,
                request_id=request_id,
                tool_call_id=f"tool_{evidence_counter['index']:04d}",
                tool_name=name,
            )
            evidence_ref = {
                "evidence_id": record["evidence_id"],
                "request_id": record["request_id"],
                "observation_hash": record["observation_hash"],
            }
            return {"evidence": evidence_ref, **dict(result)}
        return dict(result)

    def _runner(query: str, context: dict[str, Any] | None = None) -> str:
        normalized_context = context or {}
        cache_key = (
            f"{name}|{query}|{json.dumps(normalized_context, ensure_ascii=False, sort_keys=True)}"
        )
        if request_cache is not None and cache_key in request_cache:
            # 缓存命中同样嵌入证据引用：观察内容与首次完全一致（同一 evidence_id 语义），
            # 只是 tool_call_id 序号递增，保证模型当前读到的 Observation 总携带合法引用。
            payload = _build_payload(dict(request_cache[cache_key]))
            payload["cached"] = True
            return json.dumps(payload, ensure_ascii=False)

        result = tool_fn(query, normalized_context)
        if request_cache is not None:
            request_cache[cache_key] = result
        payload = _build_payload(result)
        return json.dumps(payload, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_runner,
        name=name,
        description=description,
    )


def _build_update_context_tool(
    *,
    session_store: SessionStore,
    session_id: str,
    request_id: str,
    evidence_records: list[dict[str, Any]],
    ttl_seconds: int,
) -> StructuredTool:
    def _update_context(tool_input: str) -> str:
        try:
            if isinstance(tool_input, str):
                patch = DiagnosticPatch.model_validate_json(tool_input)
            else:
                patch = DiagnosticPatch.model_validate(tool_input)
            evidence = [EvidenceRecord.model_validate(item) for item in evidence_records]
            validate_patch_evidence(patch, evidence, request_id=request_id)
            updated = session_store.update_state(session_id, patch, ttl_seconds=ttl_seconds)
            _log_event(
                "context_updated",
                request_id=request_id,
                session_id=session_id,
                reason=patch.reason,
                new_version=updated.version,
                added_finding_ids=[item.id for item in patch.add_findings],
            )
            _log_event(
                "key_step",
                request_id=request_id,
                session_id=session_id,
                kind="context_update",
                reason=patch.reason,
                new_version=updated.version,
            )
            if patch.reject_candidates:
                _log_event(
                    "key_step",
                    request_id=request_id,
                    session_id=session_id,
                    kind="direction_repair",
                    candidate_previews=[_preview(item, max_len=80) for item in patch.reject_candidates],
                )
            return json.dumps(
                {"ok": True, "message": "Context updated.", "new_version": updated.version},
                ensure_ascii=False,
            )
        except (ValueError, EvidenceValidationError, StateConflictError) as exc:
            _log_event(
                "context_update_rejected",
                request_id=request_id,
                session_id=session_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            hint = (
                " Allowed evidence reference fields: evidence_id, request_id, observation_hash only "
                "(copy the evidence block from the latest tool Observation verbatim)."
            )
            return json.dumps(
                {"ok": False, "error": f"{exc} {hint}".strip(), "retryable": True},
                ensure_ascii=False,
            )

    return StructuredTool.from_function(
        func=_update_context,
        name="update_context",
        description=(
            "Submit an incremental DiagnosticPatch as a JSON string in tool_input. "
            "Fields: reason(required), expected_version, context_slots, "
            "add_findings[{id,claim,confidence,evidence{evidence_id,request_id,observation_hash}}], "
            "add_candidates, reject_candidates, add_validity_concerns, "
            "update_constraint_status{constraint: satisfied|unsatisfied|unchecked}, "
            "set_answer_confidence, replace_next_step_plan, "
            "supersede_findings[{finding_id,replacement}]. "
            "Only reference evidence_id/request_id/observation_hash from the current request. "
            "The evidence field inside add_findings MUST contain ONLY these three keys — "
            "do not copy tool_call_id, tool, or observation_preview. "
            "Never delete historical findings; use supersede_findings."
        ),
    )


def _build_tools(
    settings: Settings,
    selected_tool_names: list[str] | None = None,
    request_cache: dict[str, dict[str, Any]] | None = None,
    session_store: SessionStore | None = None,
    session_id: str = DEFAULT_SESSION_ID,
    request_id: str = "",
    evidence_records: list[dict[str, Any]] | None = None,
    evidence_counter: dict[str, int] | None = None,
    ttl_seconds: int = 3600,
) -> list[StructuredTool]:
    # 工具注册表：基础业务工具 + 可选 RAG 检索工具。
    selected = set(selected_tool_names or [*ANALYSIS_TOOL_NAMES, "retrieve_knowledge"])
    tools: list[StructuredTool] = []
    if "traffic_analyze" in selected:
        tools.append(
            _build_react_tool(
                name="traffic_analyze",
                description="Analyze traffic trend. Args: query (investigation question, str), context (optional dict with merchant_id/time_range).",
                tool_fn=traffic_analyze,
                request_cache=request_cache,
                request_id=request_id,
                evidence_counter=evidence_counter,
            )
        )
    if "ads_analyze" in selected:
        tools.append(
            _build_react_tool(
                name="ads_analyze",
                description="Analyze ad efficiency and ROI. Args: query (investigation question, str), context (optional dict with merchant_id/time_range).",
                tool_fn=ads_analyze,
                request_cache=request_cache,
                request_id=request_id,
                evidence_counter=evidence_counter,
            )
        )
    if "inventory_check" in selected:
        tools.append(
            _build_react_tool(
                name="inventory_check",
                description="Check inventory risk. Args: query (investigation question, str), context (optional dict with merchant_id/time_range).",
                tool_fn=inventory_check,
                request_cache=request_cache,
                request_id=request_id,
                evidence_counter=evidence_counter,
            )
        )
    if "product_diagnose" in selected:
        tools.append(
            _build_react_tool(
                name="product_diagnose",
                description="Diagnose product conversion. Args: query (investigation question, str), context (optional dict with merchant_id/time_range).",
                tool_fn=product_diagnose,
                request_cache=request_cache,
                request_id=request_id,
                evidence_counter=evidence_counter,
            )
        )

    if settings.rag_enabled and "retrieve_knowledge" in selected:
        # 开启 RAG 时，允许 Agent 主动检索 SOP/策略知识片段。
        tools.append(
            _build_react_tool(
                name="retrieve_knowledge",
                description=(
                    "Retrieve SOP and policy snippets from local knowledge base. "
                    "Args: query (investigation question, str), context (optional dict with merchant_id/time_range)."
                ),
                tool_fn=lambda query, context: retrieve_knowledge(query, context, settings),
                request_cache=request_cache,
                request_id=request_id,
                evidence_counter=evidence_counter,
            )
        )

    if session_store is not None:
        tools.append(
            _build_update_context_tool(
                session_store=session_store,
                session_id=session_id,
                request_id=request_id,
                evidence_records=evidence_records if evidence_records is not None else [],
                ttl_seconds=ttl_seconds,
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
    session_store: SessionStore | None = None,
    session_id: str = DEFAULT_SESSION_ID,
    request_id: str = "",
    evidence_records: list[dict[str, Any]] | None = None,
    evidence_counter: dict[str, int] | None = None,
) -> AgentExecutor:
    """Create a tool-calling Agent executor."""

    callback_list = callbacks or []
    # LLM 客户端：支持 OpenAI 兼容接口。
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.3,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
        callbacks=callback_list,
    )
    tools = _build_tools(
        settings,
        selected_tool_names=selected_tool_names,
        request_cache=request_cache,
        session_store=session_store,
        session_id=session_id,
        request_id=request_id,
        evidence_records=evidence_records,
        evidence_counter=evidence_counter,
        ttl_seconds=settings.session_ttl_seconds,
    )

    # 根据配置动态提示模型是否可用知识检索工具。
    knowledge_hint = (
        "If the question needs SOP or policy knowledge, call retrieve_knowledge first."
        if settings.rag_enabled
        else "Knowledge retrieval tool is disabled. Use only available analysis tools."
    )

    template = TOOL_CALLING_SYSTEM_PROMPT
    # 先注入动态提示，再构建 tool-calling Agent。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(knowledge_hint=knowledge_hint)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callback_list,
        verbose=settings.agent_verbose,
        handle_parsing_errors=True,
        max_iterations=ROUTED_AGENT_ITERATIONS if selected_tool_names else MAX_AGENT_ITERATIONS,
        max_execution_time=MAX_AGENT_EXECUTION_SECONDS,
        early_stopping_method="force",
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
    execution_request_id = request_id or f"req_{uuid4().hex}"
    evidence_counter: dict[str, int] = {"index": 0}
    effective_context = _merge_context_with_slots(session_store, settings, sid, context)
    missing_keys = _missing_context_keys(effective_context)
    if missing_keys:
        response = _build_clarification_response(query, effective_context, sid, missing_keys)
        _log_event(
            "agent_clarification_requested",
            request_id=request_id,
            session_id=sid,
            missing_context_keys=missing_keys,
        )
        return response
    trace_callback = ReActTraceCallbackHandler(
        event_sink=event_sink,
        session_id=sid,
        request_id=execution_request_id,
        evidence_counter=evidence_counter,
    )
    selected_tool_names, route_reason = _route_tools(query, effective_context, settings.rag_enabled)
    request_cache: dict[str, dict[str, Any]] = {}
    agent_executor = create_agent(
        settings,
        callbacks=[trace_callback],
        selected_tool_names=selected_tool_names,
        request_cache=request_cache,
        session_store=session_store,
        session_id=sid,
        request_id=execution_request_id,
        evidence_records=trace_callback.evidence_records,
        evidence_counter=evidence_counter,
    )

    history_text = _get_history_text(session_store, sid)
    enhanced_input = (
        f"User question: {query}\n"
        f"Context JSON: {json.dumps(effective_context, ensure_ascii=False)}"
    )

    _log_event(
        "agent_run_started",
        request_id=request_id,
        session_id=sid,
        query_preview=_preview(query, max_len=120),
        context_keys=sorted(effective_context.keys()),
        selected_tools=selected_tool_names,
        route_reason=route_reason,
    )

    started_at = perf_counter()
    if _should_short_circuit(query, selected_tool_names):
        tool_name = selected_tool_names[0]
        tool_started = perf_counter()
        tool_output = _run_single_tool(tool_name, settings, query, effective_context)
        tool_duration_ms = max(0, int((perf_counter() - tool_started) * 1000))
        early_evidence = _build_evidence_record(
            tool_output,
            request_id=execution_request_id,
            tool_call_id="tool_0001",
            tool_name=tool_name,
        )
        final_answer = _cleanup_markdown(
            "问题摘要：命中单一高相关工具，使用快速模式直达输出。\n"
            f"核心发现：{tool_output.get('summary', '暂无')}\n"
            "行动计划：\n"
            "1. 先执行工具返回的高优先建议。\n"
            "2. 观察 24-72 小时关键指标变化。\n"
            "3. 若未改善，再进入多工具综合诊断。"
        )
        final_answer = _append_evidence_block(final_answer, _extract_evidence_from_observation(tool_output))
        _append_history(session_store, settings, sid, query, final_answer)
        latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        steps: list[StepRecord] = [
            StepRecord(
                name="Early Stop Route",
                input={"query": query, "context": effective_context, "selected_tool": tool_name},
                output={"observation": tool_output, "evidence": early_evidence},
                duration_ms=tool_duration_ms,
            ),
            StepRecord(
                name="Agent",
                input={"query": query, "context": effective_context, "session_id": sid},
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
        _finalize_request_state(session_store, settings, sid, execution_request_id, steps, context=effective_context)
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
        chat_history_messages = [HumanMessage(content=history_text)] if history_text.strip() else []
        result = agent_executor.invoke(
            {
                "input": enhanced_input,
                "chat_history": chat_history_messages,
            },
            config={"callbacks": [trace_callback]},
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
    final_answer = _append_evidence_block(final_answer, _collect_evidence_lines(trace_callback.steps))
    _append_history(session_store, settings, sid, query, final_answer)
    _finalize_request_state(session_store, settings, sid, execution_request_id, trace_callback.steps, context=effective_context)
    refine_rounds = _run_refine_pass(
        session_store,
        settings,
        sid,
        execution_request_id,
        query,
        effective_context,
        request_cache,
        evidence_counter,
        trace_callback.steps,
    )
    if refine_rounds:
        _log_event(
            "agent_run_refined",
            request_id=request_id,
            session_id=sid,
            refine_rounds=refine_rounds,
            state_version=session_store.get_state(sid).version,
        )

    # steps = ReAct 每轮轨迹 + 一条总览 Agent 结果。
    steps: list[StepRecord] = trace_callback.steps + [
        StepRecord(
            name="Agent",
            input={"query": query, "context": effective_context, "session_id": sid},
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
