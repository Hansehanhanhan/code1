from __future__ import annotations

from tools.tools import ads_analyze, inventory_check, product_diagnose, traffic_analyze

CTX = {"merchant_id": "demo-002", "time_range": "近7天"}


def test_tool_output_stable_across_query_phrasing() -> None:
    # 同一商家上下文下，任何查询措辞都必须返回完全一致的结果，
    # 避免模型换措辞反复查询时拿到相互矛盾的证据并陷入排查循环。
    for tool_fn in (traffic_analyze, ads_analyze, inventory_check, product_diagnose):
        a = tool_fn("分析最近的坏情况？下滑、下降、积压、转化差都查查", CTX)
        b = tool_fn("看一下整体趋势和效率表现", CTX)
        assert a == b, tool_fn.__name__


def test_demo_merchant_uses_weak_branch() -> None:
    demo_ctx = {"merchant_id": "demo-001", "time_range": "近7天"}
    assert traffic_analyze("尽可能中性的话术", demo_ctx)["data"]["trend"] == "declining"
    assert ads_analyze("尽量中性的话术", demo_ctx)["data"]["roi"] < 2.5
    assert inventory_check("尽量中性的话术", demo_ctx)["data"]["stock_level"] == "overstock"
    assert product_diagnose("尽量中性的话术", demo_ctx)["data"]["conversion_rate"] < 0.02


def test_handles_missing_context_deterministically() -> None:
    for tool_fn in (traffic_analyze, ads_analyze, inventory_check, product_diagnose):
        out = tool_fn("趋势", {})
        assert out["status"] == "ok"
        assert out["tool"] == tool_fn.__name__