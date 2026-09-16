from __future__ import annotations

import json
import re as _re

from agent.agent import (
    TOOL_CALLING_SYSTEM_PROMPT,
    _append_evidence_block,
    _audit_constraint_status,
    _build_evidence_record,
    _build_request_finalize_patch,
    _build_tool,
    _build_update_context_tool,
    _extract_evidence_from_observation,
    _finalize_request_state,
    _get_history_text,
    _merge_context_with_slots,
    _missing_context_keys,
    _route_tools,
    _run_refine_pass,
    _should_short_circuit,
    run_agent,
)
from backend.models import StepRecord
from backend.session_store import InMemorySessionStore
from backend.diagnostic_state import DiagnosticPatch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class SessionSettings:
    session_ttl_seconds = 3600


def test_route_tools_selects_ads_tool() -> None:
    selected, reason = _route_tools("请分析广告ROI下降原因", {"merchant_id": "demo-001"}, rag_enabled=False)
    assert "ads_analyze" in selected
    assert reason.startswith("keyword_matched")


def test_route_tools_adds_retrieve_when_policy_query() -> None:
    selected, reason = _route_tools("请给我平台政策SOP", {"merchant_id": "demo-001"}, rag_enabled=True)
    assert "retrieve_knowledge" in selected
    assert "knowledge" in reason


def test_short_circuit_for_single_tool_query() -> None:
    assert _should_short_circuit("只看广告ROI", ["ads_analyze"]) is True


def test_no_short_circuit_for_broad_query() -> None:
    assert _should_short_circuit("请做一次综合排查", ["ads_analyze"]) is False


def test_missing_context_keys_detected() -> None:
    missing = _missing_context_keys({"merchant_id": "demo-001"})
    assert missing == ["time_range"]


def test_merge_context_with_slots_fills_missing_and_explicit_values_win() -> None:
    store = InMemorySessionStore()
    store.update_context_slots(
        "s1",
        {"merchant_id": "demo-001", "time_range": "last_7_days"},
        ttl_seconds=100,
    )

    effective = _merge_context_with_slots(
        store,
        SessionSettings(),
        "s1",
        {"time_range": "yesterday"},
    )

    assert effective == {
        "merchant_id": "demo-001",
        "time_range": "yesterday",
    }
    assert store.get_context_slots("s1") == effective


def test_history_context_includes_diagnostic_state_version_and_slots() -> None:
    store = InMemorySessionStore()
    store.update_context_slots("s1", {"merchant_id": "demo-001"}, ttl_seconds=100)

    history_text = _get_history_text(store, "s1")

    assert "Current Diagnostic State:" in history_text
    assert '"version":1' in history_text
    assert '"merchant_id":"demo-001"' in history_text


def test_extract_evidence_from_observation() -> None:
    observation = {
        "tool": "retrieve_knowledge",
        "summary": "命中2条SOP。",
        "data": {"matches": [{"source": "knowledge/seed/ads_sop.md"}, {"source": "knowledge/seed/inventory_sop.md"}]},
    }
    lines = _extract_evidence_from_observation(observation)
    assert any(line.startswith("retrieve_knowledge:") for line in lines)
    assert any("ads_sop.md" in line for line in lines)


def test_append_evidence_block() -> None:
    answer = _append_evidence_block("这是结论。", ["ads_analyze: ROI下降"])
    assert "证据来源：" in answer
    assert "ads_analyze: ROI下降" in answer


def test_evidence_record_hash_is_stable_and_request_scoped() -> None:
    first = _build_evidence_record(
        {"summary": "流量下降", "value": 22},
        request_id="req-1",
        tool_call_id="tool-0001",
        tool_name="traffic_analyze",
    )
    second = _build_evidence_record(
        {"value": 22, "summary": "流量下降"},
        request_id="req-1",
        tool_call_id="tool-0001",
        tool_name="traffic_analyze",
    )

    assert first["evidence_id"] == "req-1:tool-0001"
    assert first["observation_hash"] == second["observation_hash"]
    assert first["tool"] == "traffic_analyze"


def test_trace_callback_records_evidence_with_tool_observation() -> None:
    from agent.agent import ToolCallTraceCallbackHandler

    callback = ToolCallTraceCallbackHandler(session_id="s1", request_id="req-1")
    callback._pending = {
        "tool_loop_index": 1,
        "thought": "检查流量",
        "action": "traffic_analyze",
        "action_input": {"query": "流量"},
        "started_at": 0.0,
    }
    callback.on_tool_end({"summary": "流量下降"})

    assert len(callback.evidence_records) == 1
    assert callback.evidence_records[0]["evidence_id"] == "req-1:tool_0001"
    assert callback.steps[0].output["evidence"] == callback.evidence_records[0]


def test_update_context_accepts_valid_evidence_and_updates_state() -> None:
    import json

    store = InMemorySessionStore()
    evidence = {
        "evidence_id": "req-1:tool-0001",
        "request_id": "req-1",
        "tool_call_id": "tool-0001",
        "tool": "traffic_analyze",
        "observation_hash": "sha256:abc",
    }
    tool = _build_update_context_tool(
        session_store=store,
        session_id="s1",
        request_id="req-1",
        evidence_records=[evidence],
        ttl_seconds=100,
    )
    result = json.loads(
        tool.invoke(
            json.dumps(
                {
                    "reason": "traffic_analysis_completed",
                    "expected_version": 0,
                    "add_findings": [
                        {
                            "id": "f-1",
                            "claim": "流量下降",
                            "confidence": "high",
                            "evidence": {
                                "evidence_id": "req-1:tool-0001",
                                "request_id": "req-1",
                                "observation_hash": "sha256:abc",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )
    )

    assert result["ok"] is True
    assert store.get_state("s1").verified_findings[0].id == "f-1"


def test_update_context_tool_emits_key_steps_via_sink() -> None:
    store = InMemorySessionStore()
    evidence = {
        "evidence_id": "req-1:tool-0001",
        "request_id": "req-1",
        "tool_call_id": "tool-0001",
        "tool": "traffic_analyze",
        "observation_hash": "sha256:abc",
    }
    emitted: list[dict] = []
    tool = _build_update_context_tool(
        session_store=store,
        session_id="s1",
        request_id="req-1",
        evidence_records=[evidence],
        ttl_seconds=100,
        event_sink=emitted.append,
    )
    tool.invoke(
        json.dumps(
            {
                "reason": "traffic_analysis_completed",
                "expected_version": 0,
                "add_findings": [
                    {
                        "id": "f-1",
                        "claim": "流量下降",
                        "confidence": "high",
                        "evidence": {
                            "evidence_id": "req-1:tool-0001",
                            "request_id": "req-1",
                            "observation_hash": "sha256:abc",
                        },
                    }
                ],
                "reject_candidates": ["候选A：流量下降归因于大促"],
            },
            ensure_ascii=False,
        )
    )

    key_steps = [event for event in emitted if event.get("type") == "key_step"]
    kinds = [event["content"]["kind"] for event in key_steps]
    assert "context_update" in kinds
    assert "direction_repair" in kinds
    context_update = next(
        event["content"] for event in key_steps if event["content"]["kind"] == "context_update"
    )
    assert context_update["new_version"] == 1
    direction_repair = next(
        event["content"] for event in key_steps if event["content"]["kind"] == "direction_repair"
    )
    assert any("候选A" in preview for preview in direction_repair["candidate_previews"])

    rejected_tool = _build_update_context_tool(
        session_store=store,
        session_id="s1",
        request_id="req-1",
        evidence_records=[],
        ttl_seconds=100,
        event_sink=emitted.append,
    )
    rejected_tool.invoke(
        json.dumps(
            {
                "reason": "invalid_reference",
                "expected_version": 1,
                "add_findings": [
                    {
                        "id": "f-2",
                        "claim": "无证据结论",
                        "confidence": "low",
                        "evidence": {
                            "evidence_id": "missing",
                            "request_id": "req-1",
                            "observation_hash": "sha256:wrong",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
    )

    kinds_after_reject = [event["content"]["kind"] for event in emitted if event.get("type") == "key_step"]
    assert "context_update_rejected" in kinds_after_reject


def test_update_context_rejects_invalid_evidence_without_state_change() -> None:
    import json

    store = InMemorySessionStore()
    tool = _build_update_context_tool(
        session_store=store,
        session_id="s1",
        request_id="req-1",
        evidence_records=[],
        ttl_seconds=100,
    )
    result = json.loads(
        tool.invoke(
            json.dumps(
                {
                    "reason": "invalid_test",
                    "expected_version": 0,
                    "add_findings": [
                        {
                            "id": "f-1",
                            "claim": "未经证据验证",
                            "confidence": "low",
                            "evidence": {
                                "evidence_id": "missing",
                                "request_id": "req-1",
                                "observation_hash": "sha256:wrong",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )
    )

    assert result["ok"] is False
    assert store.get_state("s1").version == 0


def test_update_context_rejects_extra_evidence_field_with_hint() -> None:
    import json

    store = InMemorySessionStore()
    evidence = {
        "evidence_id": "req-1:tool-0001",
        "request_id": "req-1",
        "tool_call_id": "tool-0001",
        "tool": "traffic_analyze",
        "observation_hash": "sha256:abc",
    }
    tool = _build_update_context_tool(
        session_store=store,
        session_id="s1",
        request_id="req-1",
        evidence_records=[evidence],
        ttl_seconds=100,
    )
    result = json.loads(
        tool.invoke(
            json.dumps(
                {
                    "reason": "invalid_test",
                    "expected_version": 0,
                    "add_findings": [
                        {
                            "id": "f-1",
                            "claim": "流量下降",
                            "confidence": "high",
                            "evidence": {
                                **{k: evidence[k] for k in ("evidence_id", "request_id", "observation_hash")},
                                "observation_preview": "流量下降",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )
    )

    assert result["ok"] is False
    assert result["retryable"] is True
    assert "Allowed evidence reference fields" in result["error"]
    assert store.get_state("s1").version == 0


def test_run_agent_returns_clarification_when_context_missing() -> None:
    response = run_agent("请分析最近流量波动", {})
    assert response.steps[0].name == "Clarification"
    assert "merchant_id" in response.final_answer


def _observation_step(tool: str, observation: dict) -> StepRecord:
    import json

    return StepRecord(
        name=f"Tool Loop {tool}",
        input={"action": tool, "action_input": {}},
        output={"observation": observation},
        duration_ms=5,
    )


def test_build_request_finalize_patch_extracts_candidates_and_constraints() -> None:
    steps = [
        _observation_step(
            "traffic_analyze",
            {
                "tool": "traffic_analyze",
                "status": "ok",
                "summary": "流量下降",
                "recommendations": ["检查广告位", "排查详情页"],
            },
        ),
        _observation_step(
            "inventory_check",
            {
                "tool": "inventory_check",
                "status": "error",
                "summary": "库存数据缺失，无法评估缺货风险",
            },
        ),
    ]

    patch = _build_request_finalize_patch(steps, expected_version=3)

    assert patch is not None
    assert patch.reason == "request_finalize"
    assert patch.expected_version == 3
    assert patch.add_candidates == ["检查广告位", "排查详情页"]
    assert patch.add_constraints == ["库存数据缺失，无法评估缺货风险"]


def test_finalize_request_state_writes_candidates_then_noops_on_duplicate() -> None:
    store = InMemorySessionStore()
    steps = [
        _observation_step(
            "ads_analyze",
            {"tool": "ads_analyze", "status": "ok", "recommendations": ["优化预算结构"]},
        )
    ]

    assert _finalize_request_state(store, SessionSettings(), "s1", "req-1", steps) is True
    state = store.get_state("s1")
    assert state.version == 1
    assert "优化预算结构" in state.current_candidates

    assert _finalize_request_state(store, SessionSettings(), "s1", "req-2", steps) is False
    assert store.get_state("s1").version == 1


def test_finalize_request_state_noop_without_grounded_observations() -> None:
    store = InMemorySessionStore()
    steps = [
        _observation_step(
            "retrieve_knowledge",
            {"tool": "retrieve_knowledge", "status": "ok", "summary": "命中 SOP"},
        )
    ]

    assert _finalize_request_state(store, SessionSettings(), "s1", "req-1", steps) is False
    assert store.get_state("s1").version == 0


def test_tool_calling_template_includes_update_context_few_shot() -> None:
    assert "Call update_context at least once per request" in TOOL_CALLING_SYSTEM_PROMPT
    assert "submit at least once per request" in TOOL_CALLING_SYSTEM_PROMPT
    assert '"add_findings"' in TOOL_CALLING_SYSTEM_PROMPT
    assert '"evidence_id"' in TOOL_CALLING_SYSTEM_PROMPT
    assert '"observation_hash"' in TOOL_CALLING_SYSTEM_PROMPT
    assert "supersede_findings" in TOOL_CALLING_SYSTEM_PROMPT
    assert "copied verbatim" in TOOL_CALLING_SYSTEM_PROMPT
    assert "update_constraint_status" in TOOL_CALLING_SYSTEM_PROMPT
    assert "set_answer_confidence" in TOOL_CALLING_SYSTEM_PROMPT


class ScriptedToolCallingModel(BaseChatModel):
    """脚本化 tool-calling 模型：先查流量，再从 ToolMessage 提取真实证据提交 update_context，最后给结论。"""

    def _agenerate(self, messages, *, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(self, messages, *, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self._produce(messages))])

    @property
    def model_name(self) -> str:
        return "fake-tool-calling"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _produce(self, messages) -> AIMessage:
        import json as _json

        self._produce_calls = getattr(self, "_produce_calls", 0) + 1
        if self._produce_calls == 1:
            return _ai_message(
                "Thought: 需要先查流量数据。",
                tool_calls=[
                    {
                        "name": "traffic_analyze",
                        "args": {
                            "query": "流量下滑原因",
                            "context": {"merchant_id": "demo-001", "time_range": "last_7_days"},
                        },
                        "id": "call_traffic_1",
                        "type": "tool_call",
                    }
                ],
            )
        if self._produce_calls == 2:
            tool_content = self._last_tool_content(messages)
            payload = _json.loads(tool_content) if tool_content else {}
            evidence = payload.get("evidence") if isinstance(payload, dict) else {}
            version_match = _re.search(r'"version":(\d+)', _joined(messages))
            expected_version = int(version_match.group(1)) if version_match else 0
            patch = {
                "reason": "traffic_analysis_completed",
                "expected_version": expected_version,
                "context_slots": {"merchant_id": "demo-001"},
                "add_findings": [
                    {
                        "id": "f-1",
                        "claim": "近7天流量下滑",
                        "confidence": "medium",
                        "evidence": {
                            "evidence_id": evidence.get("evidence_id"),
                            "request_id": evidence.get("request_id"),
                            "observation_hash": evidence.get("observation_hash"),
                        },
                    }
                ],
                "add_candidates": ["广告投放效率下降"],
                "update_constraint_status": {"merchant_id": "satisfied", "time_range": "satisfied"},
            }
            return _ai_message(
                "Thought: 拿到证据，提交增量 patch 沉淀结论。",
                tool_calls=[
                    {
                        "name": "update_context",
                        "args": {"tool_input": _json.dumps(patch, ensure_ascii=False)},
                        "id": "call_update_1",
                        "type": "tool_call",
                    }
                ],
            )
        return _ai_message("Thought: 证据已充分。\nFinal Answer: 流量下滑，建议检查广告位与详情页转化。")

    @staticmethod
    def _last_tool_content(messages) -> str:
        for message in reversed(messages):
            if getattr(message, "type", "") == "tool":
                return str(getattr(message, "content", ""))
        return ""


def _ai_message(content: str, tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls if tool_calls else [])


def _joined(messages) -> str:
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


def test_run_agent_fake_llm_persists_finding_with_injected_evidence(monkeypatch) -> None:
    import agent.agent as agent_module

    store = InMemorySessionStore()
    monkeypatch.setattr(agent_module, "get_session_store", lambda _settings: store)
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: ScriptedToolCallingModel())

    response = run_agent(
        "请综合排查最近流量下滑原因",
        {"merchant_id": "demo-001", "time_range": "last_7_days"},
        session_id="s-e2e",
        request_id="req-e2e",
    )

    assert response.steps[0].name != "Clarification"
    state = store.get_state("s-e2e")
    finding_ids = {finding.id for finding in state.verified_findings}
    assert "f-1" in finding_ids
    f1 = next(finding for finding in state.verified_findings if finding.id == "f-1")
    assert f1.evidence.evidence_id == "req-e2e:tool_0001"
    assert f1.evidence.request_id == "req-e2e"
    assert "广告投放效率下降" in state.current_candidates
    assert state.version >= 2
    assert state.constraint_status.get("merchant_id") == "satisfied"
    assert 0 <= state.answer_confidence <= 1


def test_tool_observation_embeds_evidence_block() -> None:
    tool = _build_tool(
        name="traffic_analyze",
        description="test",
        tool_fn=lambda query, context: {
            "tool": "traffic_analyze",
            "status": "ok",
            "summary": "流量下降",
            "data": {"trend": "declining"},
        },
        request_id="req-1",
        evidence_counter={"index": 0},
    )
    payload = json.loads(tool.invoke({"query": "流量", "context": {"merchant_id": "m"}}))
    assert payload["evidence"]["evidence_id"] == "req-1:tool_0001"
    assert payload["evidence"]["request_id"] == "req-1"
    assert payload["evidence"]["observation_hash"].startswith("sha256:")
    assert payload["tool"] == "traffic_analyze"
    # 嵌入块必须是纯 EvidenceRef 三字段，避免模型复制多余的
    # observation_preview / tool_call_id / tool 被 pydantic 拒绝。
    assert set(payload["evidence"].keys()) == {"evidence_id", "request_id", "observation_hash"}


def test_tool_cache_hit_embeds_evidence_block() -> None:
    tool = _build_tool(
        name="traffic_analyze",
        description="test",
        tool_fn=lambda query, context: {
            "tool": "traffic_analyze",
            "status": "ok",
            "summary": "流量下降",
            "data": {"trend": "declining"},
        },
        request_cache={},
        request_id="req-cache",
        evidence_counter={"index": 0},
    )
    args = {"query": "流量", "context": {"merchant_id": "m"}}
    first = json.loads(tool.invoke(args))
    second = json.loads(tool.invoke(args))
    assert first["evidence"]["evidence_id"] == "req-cache:tool_0001"
    assert second["cached"] is True
    # 缓存命中同样携带证据引用，且哈希与首次一致（同一观察内容）。
    assert second["evidence"]["evidence_id"] == "req-cache:tool_0002"
    assert second["evidence"]["observation_hash"] == first["evidence"]["observation_hash"]
    assert set(second["evidence"].keys()) == {"evidence_id", "request_id", "observation_hash"}


def test_audit_constraint_status_is_deterministic() -> None:
    status, confidence = _audit_constraint_status(
        {"merchant_id": "demo-001", "time_range": ""},
        ["库存数据缺失"],
    )
    assert status == {
        "merchant_id": "satisfied",
        "time_range": "unsatisfied",
        "库存数据缺失": "unsatisfied",
    }
    assert confidence == round(1 / 3, 2)


def test_finalize_promotes_evidence_backed_findings_then_noops() -> None:
    store = InMemorySessionStore()
    evidence = _build_evidence_record(
        {"tool": "ads_analyze", "status": "ok", "recommendations": ["优化预算结构"]},
        request_id="req-x",
        tool_call_id="tool_0001",
        tool_name="ads_analyze",
    )
    step = StepRecord(
        name="Tool Loop 1",
        input={"action": "ads_analyze", "action_input": {}},
        output={
            "observation": {"tool": "ads_analyze", "status": "ok", "recommendations": ["优化预算结构"]},
            "evidence": evidence,
        },
        duration_ms=5,
    )

    assert _finalize_request_state(
        store, SessionSettings(), "s1", "req-x", [step], context={"merchant_id": "m"}
    ) is True
    state = store.get_state("s1")
    assert any(finding.claim == "优化预算结构" for finding in state.verified_findings)
    assert any(
        finding.evidence.evidence_id == evidence["evidence_id"] for finding in state.verified_findings
    )
    assert state.verified_citations == [evidence["evidence_id"]]
    assert state.constraint_status == {"merchant_id": "satisfied", "time_range": "unsatisfied"}
    assert state.answer_confidence == 0.5
    assert state.version == 1

    assert _finalize_request_state(
        store, SessionSettings(), "s1", "req-x", [step], context={"merchant_id": "m"}
    ) is False
    assert store.get_state("s1").version == 1


def test_refine_pass_targets_unsatisfied_constraint_and_persists_evidence() -> None:
    store = InMemorySessionStore()
    initial = store.get_state("s-r")
    store.update_state(
        "s-r",
        DiagnosticPatch(
            reason="seed_constraint",
            expected_version=initial.version,
            update_constraint_status={"库存数据缺失": "unsatisfied"},
        ),
        ttl_seconds=100,
    )
    steps: list[StepRecord] = []

    refined = _run_refine_pass(
        store,
        SessionSettings(),
        "s-r",
        "req-r",
        "请排查库存问题",
        {"merchant_id": "m", "time_range": "t"},
        {},
        {"index": 0},
        steps,
    )

    assert refined >= 1
    assert any(step.name.startswith("Refine Loop") for step in steps)
    state = store.get_state("s-r")
    assert any(
        finding.evidence.evidence_id == "req-r:tool_0001" for finding in state.verified_findings
    )


def test_refine_pass_stops_when_no_unsatisfied_constraint() -> None:
    store = InMemorySessionStore()
    initial = store.get_state("s-ok")
    store.update_state(
        "s-ok",
        DiagnosticPatch(
            reason="seed_status",
            expected_version=initial.version,
            update_constraint_status={"merchant_id": "satisfied"},
        ),
        ttl_seconds=100,
    )
    steps: list[StepRecord] = []

    refined = _run_refine_pass(
        store,
        SessionSettings(),
        "s-ok",
        "req-n",
        "请排查",
        {"merchant_id": "m", "time_range": "t"},
        {},
        {"index": 0},
        steps,
    )

    assert refined == 0
    assert steps == []
