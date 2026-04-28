from __future__ import annotations

from agent.agent import (
    _append_evidence_block,
    _extract_evidence_from_observation,
    _missing_context_keys,
    _route_tools,
    _should_short_circuit,
    run_agent,
)


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


def test_run_agent_returns_clarification_when_context_missing() -> None:
    response = run_agent("请分析最近流量波动", {})
    assert response.steps[0].name == "Clarification"
    assert "merchant_id" in response.final_answer
