# Merchant Ops Copilot (LangChain 版)

基于 LangChain 的商家运营 AI Agent 项目，展示了如何使用 LangChain 框架简化 Agent 开发。

## 项目简介

这是项目的第二个版本，使用 LangChain 重构了 Agent 实现。相比原生版本，LangChain 提供了：

- ✅ 开箱即用的 Agent 类型
- ✅ 记忆管理（ConversationBufferMemory）
- ✅ 工具装饰器（@tool）
- ✅ Prompt 模板管理
- ✅ 更简洁的代码结构

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 后端 | FastAPI | >=0.115 |
| Agent 框架 | LangChain | >=0.3 |
| 模型 | OpenAI API | >=1.45.0 |
| 部署 | Uvicorn | >=0.30 |

## vs 原生版本

| 特性 | 原生版本 | LangChain 版本 |
|------|---------|------------|
| 代码量 | 约 300 行 | 约 150 行 |
| 记忆管理 | ❌ | ✅ ConversationBufferMemory |
| Agent 抽象 | 自己写状态机 | LangChain 提供的 |
| 工具定义 | 普通函数 | @tool 装饰器 |
| 扩展性 | 需要手动实现 | 框架内置 |

## 项目结构

```
langchain-agent/
├── agent/
│   ├── agent.py           # LangChain Agent 实现
│   └── __init__.py
├── backend/
│   ├── main.py           # FastAPI 入口
│   ├── settings.py        # 配置
│   ├── models.py          # 数据模型
│   └── __init__.py
├── tools/
│   ├── tools.py          # 工具定义（@tool）
│   └── __init__.py
├── prompts/
│   ├── agent_prompt.py   # Agent Prompt 模板
│   └── __init__.py
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```powershell
$env:OPENAI_API_KEY="<YOUR_KEY>"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_MODEL="deepseek-chat"
```

### 启动服务

```bash
uvicorn backend.main:app --reload
```

### 测试 API

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "本周流量下滑严重",
    "context": {"merchant_id": "demo-001"}
  }'
```

## 核心功能

### 1. LangChain Agent

使用 `create_openai_functions_agent` 创建 Agent，自动处理：
- 工具选择
- 执行结果解析
- 错误重试

### 2. 记忆管理

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

支持多轮对话，自动管理历史上下文。

### 3. 工具定义

```python
from langchain.tools import tool

@tool
def traffic_analyze(query: str, context: dict) -> dict:
    """流量分析工具"""
    # 工具逻辑
    return {"result": ...}
```

LangChain 自动处理参数解析和类型转换。

## API 接口

### POST /run

执行 Agent 任务。

**请求：**
```json
{
  "query": "本周流量下滑严重",
  "context": {"merchant_id": "demo-001"}
}
```

**响应：**
```json
{
  "final_answer": "...",
  "steps": [...],
  "metrics": {...}
}
```

## 学习要点

通过这个项目，你可以掌握：

1. **LangChain 核心**
   - Agent 的创建和执行
   - 工具的定义和使用
   - Prompt 模板

2. **记忆管理**
   - ConversationBufferMemory 的使用
   - 多轮对话的实现

3. **工程实践**
   - FastAPI 集成
   - 错误处理和日志
   - 环境配置管理

## 后续改进

- [ ] 加入 RAG 知识库检索
- [ ] 实现反思纠错机制
- [ ] 增加更多业务工具
- [ ] 优化 Prompt 提升稳定性

## 对比总结

| 维度 | 原生实现 | LangChain 实现 |
|------|---------|------------|
| 代码简洁性 | 中 | 高 |
| 开发效率 | 低 | 高 |
| 可维护性 | 中 | 高 |
| 学习曲线 | 高 | 低 |

**LangChain 让 Agent 开发变得更简单和标准化。**
