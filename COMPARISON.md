# Agent 实现方案对比：原生 vs LangChain

本文档对比了商家运营 Copilot 项目的两种实现方案，帮助理解两种方式的优劣。

## 方案概览

| 方案 | 版本 | 代码行数 | 核心框架 |
|------|------|----------|----------|
| 原生实现 | v1 | ~300 | OpenAI API + 自定义状态机 |
| LangChain | v2 | ~150 | LangChain + OpenAI |

## 核心差异对比

### 1. Agent 实现

#### 原生版本 (code1)
```python
# agent/state_machine.py
class SimpleAgentStateMachine:
    def run(self, query: str, context: Dict) -> RunResponse:
        # 1. Planner 规划
        planner_output = self._plan(query, context)

        # 2. Executor 执行
        executor_output = self._execute(planner_output)

        # 3. Verifier 校验
        verifier_output = self._verify(executor_output)

        return self._build_response(verifier_output)
```

**特点**：
- 需要自己实现状态流转逻辑
- 每个阶段可以独立定制
- 灵活性高，但代码量大

#### LangChain 版本 (langchain-agent)
```python
# agent/agent.py
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
)
```

**特点**：
- 框架封装好了工作流
- 代码简洁
- 功能开箱即用

### 2. 工具定义

#### 原生版本
```python
# tools/mock_tools.py
def traffic_analyze(query: str, context: Dict) -> Dict:
    """流量分析工具"""
    return {
        "tool": "traffic_analyze",
        "data": {...}
    }
```

#### LangChain 版本
```python
# tools/tools.py
from langchain.tools import tool

@tool
def traffic_analyze(query: str, context: dict) -> dict:
    """流量分析工具

    Args:
        query: 用户的问题描述
        context: 商家上下文信息

    Returns:
        包含分析结果的字典
    """
    return {
        "tool": "traffic_analyze",
        "data": {...}
    }
```

**差异**：
- LangChain 自动处理参数解析
- 支持类型提示和文档字符串
- 自动生成工具 schema

### 3. 记忆管理

#### 原生版本
- ❌ 无内置记忆功能
- 需要自己实现上下文管理

#### LangChain 版本
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# Agent 自动维护对话历史
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,  # 注入记忆
)
```

### 4. Prompt 管理

#### 原生版本
- Prompt 硬编码在代码中
- 修改需要改代码

#### LangChain 版本
```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的商家运营助手..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

### 5. 错误处理

#### 原生版本
```python
def run(self, query, context):
    try:
        planner_output = self._plan(query, context)
    except Exception as e:
        if self.allow_rule_fallback:
            return self._fallback(query, context)
        raise
```

#### LangChain 版本
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 自动处理解析错误
)
```

## 性能对比

| 指标 | 原生版本 | LangChain 版本 |
|------|---------|---------------|
| 首次响应时间 | 较快 | 略慢（框架开销） |
| 代码量 | ~300 行 | ~150 行 |
| 依赖数量 | 少 | 多 |
| 调试难度 | 容易（自己写的） | 稍难（框架封装） |

## 适用场景

| 场景 | 推荐方案 |
|------|----------|
| 学习 Agent 原理 | 原生版本 |
| 快速开发 MVP | LangChain 版本 |
| 需要深度定制 | 原生版本 |
| 需要多轮对话 | LangChain 版本 |
| 面试展示技术深度 | 两个版本都展示 |

## 学习价值

### 通过原生版本可以学到

- [ ] Agent 工作流的原理（Planner → Executor → Verifier）
- [ ] LLM API 调用和 Prompt 设计
- [ ] 工具调用的实现机制
- [ ] 状态机设计模式

### 通过 LangChain 版本可以学到

- [ ] LangChain 框架的使用
- [ ] 记忆管理的实现
- [ ] Agent 抽象的理解
- [ ] 工程化实践

## 面试回答示例

**Q: 为什么不直接用 LangChain，还要自己实现？**

> "我先实现了原生版本，目的是深入理解 Agent 的工作原理。通过自己写状态机，我清楚了 Planner、Executor、Verifier 各个环节的作用。后来用 LangChain 重构，体验了框架的便利，也理解了两者的优劣。"

**Q: LangChain 和原生实现的区别？**

> "原生实现灵活性高，可以精细控制每个环节，适合需要深度定制的场景。LangChain 代码简洁，开箱即用，适合快速开发。但框架有学习成本，且增加了依赖。"

**Q: 项目中如何选择技术方案？**

> "技术选型要基于实际需求。这个项目是 MVP，原生实现能让我深入理解原理；后续如果需要快速迭代或增加复杂功能（如多轮对话、知识库），会考虑 LangChain。"

## 总结对比表

| 维度 | 原生实现 | LangChain | 胜出 |
|------|---------|-----------|------|
| 代码简洁性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LangChain |
| 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 原生 |
| 学习曲线 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 原生 |
| 功能完备性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LangChain |
| 依赖大小 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 原生 |
| 面试展示价值 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平手 |

## 建议

1. **学习阶段**：先看原生版本理解原理
2. **开发阶段**：用 LangChain 快速实现
3. **面试阶段**：两个版本都准备好，能讲清楚差异和选型理由
