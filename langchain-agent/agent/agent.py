from __future__ import annotations

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool

from backend.settings import Settings
from backend.models import RunResponse, StepRecord, Metrics
from tools.tools import (
    traffic_analyze,
    ads_analyze,
    inventory_check,
    product_diagnose,
)
from time import perf_counter
from typing import Dict, Any


def create_agent(settings: Settings):
    """创建 LangChain Agent。"""

    # 初始化 LLM
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.7,
    )

    # 定义工具列表
    tools = [traffic_analyze, ads_analyze, inventory_check, product_diagnose]

    # 创建记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的商家运营助手，帮助商家诊断和解决运营问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        verbose=True,
    )

    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor


def run_agent(query: str, context: Dict[str, Any]) -> RunResponse:
    """运行 Agent 并返回结构化响应。"""

    settings = Settings.from_env()

    # 创建 Agent
    agent_executor = create_agent(settings)

    # 注入上下文到输入
    enhanced_input = f"商家信息：{context}\n\n问题：{query}"

    # 记录开始时间
    started_at = perf_counter()

    # 执行 Agent
    result = agent_executor.invoke({"input": enhanced_input})

    # 计算耗时
    latency_ms = int((perf_counter() - started_at) * 1000)

    # 构建 StepRecord
    steps: list[StepRecord] = [
        StepRecord(
            name="Agent",
            input={"query": query, "context": context},
            output={"result": str(result["output"])},
            duration_ms=latency_ms,
        )
    ]

    # 构建响应
    response = RunResponse(
        final_answer=str(result["output"]),
        steps=steps,
        metrics=Metrics(
            latency_ms=latency_ms,
            fallback_used=False,
        ),
    )

    return response
