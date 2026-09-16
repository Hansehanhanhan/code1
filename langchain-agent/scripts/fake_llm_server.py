#!/usr/bin/env python3
"""OpenAI 兼容的 Fake LLM 服务，用于 CI 坏例回归的真实执行。

按关键词把请求路由到固定的工具调用序列（先调分析工具、再调检索工具或互补工具），
满足条件后返回结构化最终答案。服务地址经 OPENAI_BASE_URL 注入后端，让 agent 真实走
工具调用 + 证据闭环 + update_context 确定性兜底，而不是触发降级路径。
仅用于测试环境。
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TOOL_ARGS: dict[str, str] = {
    "traffic_analyze": json.dumps(
        {"query": "{query}", "context": {"merchant_id": "demo-001", "time_range": "last_7_days", "category": "retail"}},
        ensure_ascii=False,
    ),
    "ads_analyze": json.dumps(
        {"query": "{query}", "context": {"merchant_id": "demo-001", "time_range": "last_7_days", "category": "retail"}},
        ensure_ascii=False,
    ),
    "inventory_check": json.dumps(
        {"query": "{query}", "context": {"merchant_id": "demo-001", "time_range": "last_7_days", "category": "retail"}},
        ensure_ascii=False,
    ),
    "product_diagnose": json.dumps(
        {"query": "{query}", "context": {"merchant_id": "demo-001", "time_range": "last_7_days", "category": "retail"}},
        ensure_ascii=False,
    ),
    "retrieve_knowledge": json.dumps(
        {"query": "{query}", "context": {"merchant_id": "demo-001", "time_range": "last_7_days", "category": "retail"}},
        ensure_ascii=False,
    ),
}

# (关键字, (工具序列), 最终答案需包含的提示语)
PLANS: list[tuple[tuple[str, ...], tuple[str, ...], str]] = [
    (("流量", "下滑", "先排查", "flow", "decline"), ("traffic_analyze", "retrieve_knowledge"), "先分析流量再给行动建议"),
    (("roi", "预算", "低效", "1.8"), ("ads_analyze", "traffic_analyze"), "给出预算倾斜与暂停低效位建议"),
    (("库存", "覆盖", "补货", "清仓"), ("inventory_check", "product_diagnose"), "识别库存风险并给补货/清仓策略"),
    (("转化", "详情页", "定价", "商品"), ("product_diagnose", "ads_analyze"), "识别详情页与定价问题并给A/B方案"),
]

DEFAULT_PLAN: tuple[str, ...] = ("traffic_analyze", "ads_analyze")


def pick_plan(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    for keywords, plan, _hint in PLANS:
        if any(kw in query or kw.lower() in lowered for kw in keywords):
            return plan
    return DEFAULT_PLAN


def build_hint(query: str) -> str:
    for _keywords, _plan, hint in PLANS:
        if any(kw in query for kw in _keywords):
            return hint
    return "输出结构化建议且遵守格式约束"


class FakeLlmHandler(BaseHTTPRequestHandler):
    server_version = "FakeLlm/1.0"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_GET(self) -> None:
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json({"error": "not found"}, status=404)
            return

        body = self._read_body()
        messages: list[dict[str, Any]] = (
            body.get("messages") if isinstance(body.get("messages"), list) else []
        )

        query = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    query = content.strip()
                    break

        tool_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        plan = pick_plan(query)
        n_tools = len(plan)
        prompt = query[:128]

        message: dict[str, Any]
        if tool_count < n_tools:
            tool = plan[tool_count]
            args = TOOL_ARGS[tool].replace("{query}", json.dumps(query, ensure_ascii=False)[1:-1])
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_fake_{tool_count + 1}",
                        "type": "function",
                        "function": {"name": tool, "arguments": args},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            hint = build_hint(query)
            message = {
                "role": "assistant",
                "content": (
                    f"已完成基于工具证据的结构化诊断（调用工具：{', '.join(plan)}）。"
                    f"结论要点：{hint}。\n"
                    f"行动建议：\n"
                    f"1. 优先处理流量与转化瓶颈，定位高潜力关键词与商品。\n"
                    f"2. 控制预算倾斜与库存节奏，暂停低效投放位。\n"
                    f"3. 风险与后续跟进：持续监控核心指标，定期回放证据闭环，避免重复建议。\n"
                    f"（FakeLlm 渲染，query={prompt}）"
                ),
            }
            finish_reason = "stop"

        self._send_json(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake-stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 16, "total_tokens": 32},
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI compatible Fake LLM stub for CI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FakeLlmHandler)
    print(f"FakeLlm serving on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())