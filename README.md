# AI Agent 商家运营助手项目

本仓库包含两个实现版本：

- `code1`：原生状态机版（Planner -> Executor -> Verifier）
- `langchain-agent`：LangChain ReAct 版（当前主线）

## 项目目录结构

```text
E:\CODE
├── code1/                    # 原生版本 (Native MVP)
│   ├── agent/                # 核心逻辑：状态机、模型客户端
│   ├── backend/              # FastAPI 后端实现
│   ├── frontend/             # Next.js 前端 UI（统一入口）
│   ├── tools/                # 商家运营模拟工具 (Mock Tools)
│   └── README.md             # 原生版详细文档
├── langchain-agent/          # LangChain 版本 (Framework Optimized)
│   ├── agent/                # 基于 LangChain 的 Agent 实现
│   ├── backend/              # FastAPI 后端实现
│   ├── tools/                # 工具集实现
│   ├── rag/                  # RAG 检索模块
│   ├── knowledge/seed/       # 知识库文档（示例）
│   └── README.md             # LangChain 版详细文档
├── COMPARISON.md             # 两个版本实现的技术对比
└── README.md                 # 本文件（项目主说明文档）
```

## 本次改动（2026-03）

### 1) LangChain Agent 切换为 ReAct 架构

- 从 `create_openai_functions_agent` 迁移为 `create_react_agent`
- 使用 ReAct 推理格式：`Question -> Thought -> Action -> Action Input -> Observation -> Final Answer`
- 工具调用输入统一为 JSON 字符串，兼容 `query/context`

### 2) 增加会话记忆（session_id）

- 请求模型新增 `session_id`（可选）
- 相同 `session_id` 复用进程内短期历史
- 服务重启后记忆清空（内存实现）

### 3) 增加流式接口（SSE）

- 新增接口：`POST /run_stream`
- 实时事件：
  - `agent_action`
  - `tool_observation`
  - `final_response`
  - `error`

### 4) 执行步骤可视化

- `steps` 支持每一轮 ReAct 轨迹（`ReAct Loop N`）
- 包含每轮的 `thought/action/action_input/observation/duration_ms`

### 5) 前端联动升级

- 默认调用 `/run_stream`，边执行边渲染
- 新增 `session_id` 输入框并随请求提交
- 运行指标中文化（如 `latency_ms`、`fallback_used`）
- 最终建议文本优化（去除 `**`、自动断行）

### 6) 新增最小可演示 RAG 能力（本次重点）

- 新增工具：`retrieve_knowledge`
- 知识库目录：`langchain-agent/knowledge/seed/*.md`
- 检索流程：文档读取 -> 切片 -> 向量化 -> 相似度检索 -> 返回来源片段
- 向量后端：`chroma`（不可用时回退 `in_memory`）
- Embedding：本地开源模型（`sentence-transformers`），默认 `BAAI/bge-small-zh-v1.5`

## 快速启动

> 环境约定：统一使用根目录虚拟环境 `E:\code\.venv`，不再使用 `langchain-agent/.venv`。

### 后端（LangChain ReAct）

```powershell
cd E:\code
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements.txt

cd E:\code\langchain-agent
$env:PYTHONPATH="."
& "..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

### 前端

```powershell
cd E:\code\code1\frontend
npm install
npm run dev
```

访问：`http://127.0.0.1:3000`

## 联调请求示例

### 同步接口

`POST /run`

```json
{
  "query": "本周流量下滑严重，请给出排查方案",
  "context": {
    "merchant_id": "demo-001",
    "time_range": "last_7_days",
    "category": "retail"
  },
  "session_id": "demo-session-001"
}
```

### 流式接口

`POST /run_stream`，`Content-Type: application/json`，响应为 `text/event-stream`。

## RAG 关键配置

- `RAG_ENABLED`：是否启用 RAG（默认 `true`）
- `RAG_DOCS_DIR`：知识库目录（默认 `knowledge/seed`）
- `RAG_VECTOR_BACKEND`：向量后端（默认 `chroma`）
- `RAG_TOP_K`：每次检索返回数量（默认 `3`）
- `RAG_EMBEDDING_MODEL`：默认 `BAAI/bge-small-zh-v1.5`
- `RAG_EMBEDDING_DEVICE`：默认 `cpu`（可改 `cuda`）
