# Merchant Ops Copilot (LangChain 版)

基于 LangChain 的商家运营 AI Agent 项目，展示了如何使用 LangChain 框架简化 Agent 开发。

> 当前代码实现已切换为 `ReAct Agent`，并支持通过 `session_id` 的内存短期会话记忆（进程内，重启后清空）。

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

### 1. 安装依赖

建议在虚拟环境中运行：

```bash
# 创建并激活虚拟环境 (Windows)
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录下创建 `.env` 文件（已支持自动加载）：

```env
OPENAI_API_KEY=your_sk_key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

### 3. 启动服务

**后端 (FastAPI):**

```powershell
# Windows 环境下建议设置 PYTHONPATH
$env:PYTHONPATH="."
python backend/main.py
```

**前端 (Next.js):**

如果您需要启动 UI 界面，请进入前端目录：

```bash
cd ../code1/frontend
npm install
npm run dev
```

> **注意**: 前端默认访问地址为 `http://localhost:3000`，后端地址已配置为 `http://localhost:8000`。

---

## 注意事项 (Troubleshooting)

如果在启动或运行中遇到问题，请参考以下条目：

1.  **500 Internal Server Error (API Key)**:
    - 确保根目录下的 `.env` 文件配置正确。
    - 服务必须由支持 `load_dotenv` 的入口启动（目前已在 `settings.py` 中修复）。

2.  **TypeError: got an unexpected keyword argument 'verbose'**:
    - LangChain 的 `create_openai_functions_agent` 不再支持直接传入 `verbose` 参数。
    - 如需开启详细日志，应在 `AgentExecutor` 级别设置 `verbose=True`。

3.  **AttributeError: 'function' object has no attribute 'get'**:
    - LangChain 要求工具必须被 `StructuredTool.from_function` 包装，不能直接传入 Python 原生函数。

4.  **Windows 下的 Hot Reload 故障**:
    - 如果修改代码后 `uvicorn` 重载报错，请手动 `Ctrl+C` 强行终止进程并重新冷启动。


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
