from __future__ import annotations

from agent.agent import _route_tools, _should_short_circuit


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
