from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI


@dataclass
class ModelPlanResult:
    scenario: str
    objective: str
    tool_names: List[str]


@dataclass
class ModelVerifyResult:
    final_answer: str
    recommendations: List[str]
    risk_level: str


class OpenAIModelClient:
    """调用真实模型完成规划和校验。"""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def plan(self, query: str, context: Dict[str, Any], tool_names: List[str]) -> ModelPlanResult:
        prompt = (
            "你是电商运营智能体的规划器。"
            "请根据用户问题和上下文，选择最需要调用的工具并给出目标。"
            "输出必须是 JSON，字段：scenario(字符串), objective(字符串), tool_names(字符串数组)。"
            f"可选工具：{tool_names}。"
            "tool_names 只能从可选工具里选择，且至少返回 1 个。"
        )
        payload = self._call_json(
            system_prompt=prompt,
            user_payload={"query": query, "context": context},
        )

        chosen = payload.get("tool_names") if isinstance(payload.get("tool_names"), list) else []
        valid = [name for name in chosen if isinstance(name, str) and name in tool_names]
        deduped = list(dict.fromkeys(valid))
        if not deduped:
            deduped = tool_names[:2] if len(tool_names) >= 2 else tool_names

        scenario = payload.get("scenario") if isinstance(payload.get("scenario"), str) else "general_diagnosis"
        objective = payload.get("objective") if isinstance(payload.get("objective"), str) else "完成商家问题诊断并输出行动建议。"

        return ModelPlanResult(scenario=scenario, objective=objective, tool_names=deduped)

    def verify(
        self,
        query: str,
        scenario: str,
        objective: str,
        execution_results: Dict[str, Any],
        fallback_used: bool,
    ) -> ModelVerifyResult:
        prompt = (
            "你是电商运营智能体的校验器。"
            "请基于工具结果输出最终结论和行动建议。"
            "输出必须是 JSON，字段：final_answer(字符串), recommendations(字符串数组), risk_level(字符串: low/medium/high)。"
            "final_answer 要精炼，recommendations 建议 2-4 条。"
        )
        payload = self._call_json(
            system_prompt=prompt,
            user_payload={
                "query": query,
                "scenario": scenario,
                "objective": objective,
                "fallback_used": fallback_used,
                "tool_results": execution_results,
            },
        )

        final_answer = payload.get("final_answer") if isinstance(payload.get("final_answer"), str) else "模型未返回有效结论，请检查输入后重试。"
        recs = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
        recommendations = [x for x in recs if isinstance(x, str)]
        if not recommendations:
            recommendations = ["优先处理影响最大的异常项。", "调整后重新执行相关工具验证效果。"]

        risk_level = payload.get("risk_level") if isinstance(payload.get("risk_level"), str) else "medium"
        risk_level = risk_level.lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"

        return ModelVerifyResult(final_answer=final_answer, recommendations=recommendations, risk_level=risk_level)

    def _call_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=30.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
