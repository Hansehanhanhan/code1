from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

from agent.model_client import OpenAIModelClient
from backend.models import Metrics, RunResponse, StepRecord
from tools.mock_tools import ads_analyze, inventory_check, product_diagnose, traffic_analyze


@dataclass
class PlannerOutput:
    """规划阶段输出。"""

    scenario: str
    tool_names: List[str]
    objective: str


class SimpleAgentStateMachine:
    """简化版流程：Planner -> Executor -> Verifier。"""

    def __init__(
        self,
        tool_registry: Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]] | None = None,
        model_client: OpenAIModelClient | None = None,
        allow_rule_fallback: bool = False,
    ) -> None:
        self.tool_registry = tool_registry or {
            "traffic_analyze": traffic_analyze,
            "ads_analyze": ads_analyze,
            "inventory_check": inventory_check,
            "product_diagnose": product_diagnose,
        }
        self.model_client = model_client
        self.allow_rule_fallback = allow_rule_fallback

    @property
    def model_enabled(self) -> bool:
        return self.model_client is not None

    def run(self, query: str, context: Dict[str, Any]) -> RunResponse:
        if not self.model_enabled and not self.allow_rule_fallback:
            raise RuntimeError("未配置真实模型。请设置 OPENAI_API_KEY 后重试。")

        started_at = perf_counter()
        steps: List[StepRecord] = []
        fallback_used = False

        planner_started = perf_counter()
        planner_output, planner_source = self._plan(query, context)
        if planner_source != "llm":
            fallback_used = True
        steps.append(
            StepRecord(
                name="Planner",
                input={"query": query, "context": context},
                output={
                    "scenario": planner_output.scenario,
                    "tool_names": planner_output.tool_names,
                    "objective": planner_output.objective,
                    "source": planner_source,
                },
                duration_ms=self._elapsed_ms(planner_started),
            )
        )

        executor_started = perf_counter()
        execution_results = self._execute(query, context, planner_output.tool_names)
        steps.append(
            StepRecord(
                name="Executor",
                input={"tool_names": planner_output.tool_names},
                output={"tool_results": execution_results},
                duration_ms=self._elapsed_ms(executor_started),
            )
        )

        verifier_started = perf_counter()
        final_answer, verifier_output, verifier_source = self._verify(query, planner_output, execution_results)
        if verifier_source != "llm":
            fallback_used = True
        if verifier_output.get("missing_tools"):
            fallback_used = True
        steps.append(
            StepRecord(
                name="Verifier",
                input={"planner": planner_output.__dict__, "tool_results": execution_results},
                output={**verifier_output, "source": verifier_source},
                duration_ms=self._elapsed_ms(verifier_started),
            )
        )

        return RunResponse(
            final_answer=final_answer,
            steps=steps,
            metrics=Metrics(latency_ms=self._elapsed_ms(started_at), fallback_used=fallback_used),
        )

    def _plan(self, query: str, context: Dict[str, Any]) -> Tuple[PlannerOutput, str]:
        if self.model_client is not None:
            try:
                result = self.model_client.plan(query=query, context=context, tool_names=list(self.tool_registry.keys()))
                return PlannerOutput(
                    scenario=result.scenario,
                    tool_names=result.tool_names,
                    objective=result.objective,
                ), "llm"
            except Exception as exc:  # noqa: BLE001
                if not self.allow_rule_fallback:
                    raise RuntimeError(f"模型规划阶段失败：{exc}") from exc

        return self._rule_plan(query, context), "rule"

    def _rule_plan(self, query: str, context: Dict[str, Any]) -> PlannerOutput:
        lowered = f"{query} {context}".lower()

        if any(keyword in lowered for keyword in ["traffic", "exposure", "impression", "click", "流量", "曝光", "点击"]):
            return PlannerOutput(
                scenario="traffic_drop",
                tool_names=["traffic_analyze", "product_diagnose"],
                objective="诊断流量下滑并识别商品层面的原因。",
            )
        if any(keyword in lowered for keyword in ["ad", "ads", "roi", "cpc", "广告", "投放"]):
            return PlannerOutput(
                scenario="ads_efficiency",
                tool_names=["ads_analyze", "product_diagnose"],
                objective="分析广告效率并检查商品转化问题。",
            )
        if any(keyword in lowered for keyword in ["inventory", "stock", "overstock", "库存", "积压", "缺货"]):
            return PlannerOutput(
                scenario="inventory_risk",
                tool_names=["inventory_check", "product_diagnose"],
                objective="检查库存风险并给出备货建议。",
            )
        return PlannerOutput(
            scenario="general_diagnosis",
            tool_names=["traffic_analyze", "ads_analyze", "inventory_check", "product_diagnose"],
            objective="执行商家全局诊断，覆盖流量、投放、库存和商品健康度。",
        )

    def _execute(self, query: str, context: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
        # 逐个调用工具，聚合结构化结果。
        results: Dict[str, Any] = {}
        for tool_name in tool_names:
            tool = self.tool_registry[tool_name]
            results[tool_name] = tool(query, context)
        return results

    def _verify(
        self,
        query: str,
        planner_output: PlannerOutput,
        execution_results: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any], str]:
        missing_tools = [name for name, result in execution_results.items() if result.get("status") != "ok"]
        fallback_used = bool(missing_tools)

        if self.model_client is not None:
            try:
                verify_result = self.model_client.verify(
                    query=query,
                    scenario=planner_output.scenario,
                    objective=planner_output.objective,
                    execution_results=execution_results,
                    fallback_used=fallback_used,
                )
                verifier_output = {
                    "query": query,
                    "scenario": planner_output.scenario,
                    "validated": not missing_tools,
                    "missing_tools": missing_tools,
                    "risk_level": verify_result.risk_level,
                    "recommendations": verify_result.recommendations,
                }
                return verify_result.final_answer, verifier_output, "llm"
            except Exception as exc:  # noqa: BLE001
                if not self.allow_rule_fallback:
                    raise RuntimeError(f"模型校验阶段失败：{exc}") from exc

        # 规则兜底输出。
        if missing_tools:
            final_answer = (
                f"目标：{planner_output.objective} 但以下工具返回异常状态：{', '.join(missing_tools)}。"
                "请先按规则兜底建议处理，再修复参数后重试。"
            )
        else:
            highlights = [result["summary"].rstrip("。.") for result in execution_results.values()]
            final_answer = (
                f"场景：{planner_output.scenario}。"
                f"目标：{planner_output.objective}"
                f"关键发现：{'；'.join(highlights)}。"
                "建议先处理影响最大的项，再重新执行相关工具验证结果。"
            )

        verifier_output = {
            "query": query,
            "scenario": planner_output.scenario,
            "validated": not missing_tools,
            "missing_tools": missing_tools,
            "risk_level": "medium" if missing_tools else "low",
            "recommendations": [
                "优先处理执行器返回的最高优先级问题。",
                "记录下一步验证动作，便于团队复盘。",
            ],
        }
        return final_answer, verifier_output, "rule"

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))
