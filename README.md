# AI Agent 商家运营助手项目

本仓库包含两个实现版本：

- `code1`：原生状态机版（Planner -> Executor -> Verifier）
- `langchain-agent`：LangChain ReAct 工程版（当前主线）

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
│   ├── rag/                  # 检索与知识库模块
│   ├── mcp_server/           # 轻量 MCP Server
│   ├── tools/                # 工具集
│   ├── tests/                # 测试
│   └── README.md             # LangChain 版详细文档
├── COMPARISON.md             # 两个版本实现技术对比
└── README.md                 # 本文件（项目主说明文档）
```

## 本次更新（2026-04-18）

### 1) RAG 升级为混合检索
- 新增 `BM25 + 向量检索` 混合召回。
- 采用 `RRF` 融合两路候选，再进行 metadata 过滤与重排。
- 返回新增观测字段：`retrieval_mode`、`vector_candidates_count`、`bm25_candidates_count`、`fused_candidates_count`。

### 2) 安全与鉴权增强
- 增加简单 API Key 鉴权（`APP_AUTH_ENABLED` + `APP_API_KEY`）。
- 增加输入长度限制（`MAX_QUERY_CHARS`、`MAX_CONTEXT_CHARS`）。
- 增加基础 Prompt 注入拦截（`PROMPT_INJECTION_GUARD_ENABLED`）。
- `/metrics/*` 接口在鉴权开启时也受 `X-API-Key` 保护。

### 3) 限流与代理信任策略加固
- 限流从单桶扩展为“双桶”：会话桶 + 纯 IP 硬桶（`RATE_LIMIT_MAX_REQUESTS_IP`）。
- `X-Forwarded-For` 仅在受信代理场景使用（`TRUST_X_FORWARDED_FOR` + `TRUSTED_PROXY_IPS`）。
- 当开启 `TRUST_X_FORWARDED_FOR=true` 且未配置 `TRUSTED_PROXY_IPS` 时，自动回退 `request.client.host`。
- 双桶判定改为短路策略，减少次桶的非必要配额扣减。

### 4) MCP 治理对齐
- MCP 的 `run_agent` / `retrieve_knowledge` 接入与 HTTP 一致的治理：
  - 鉴权
  - 输入校验
  - 超时 / 重试 / 降级
  - 限流
- MCP 新增 `client_id` 参数用于限流分桶。

### 5) 工程与可运维性
- Agent `verbose` 改为环境变量控制（`AGENT_VERBOSE`，默认 `false`）。
- CORS 改为白名单配置（`APP_CORS_ORIGINS`），并避免 `* + credentials` 风险组合。
- 文档已与实现对齐（metadata 过滤行为、MCP 限流与鉴权说明）。

### 6) 测试覆盖
- 新增/扩展 MCP 治理、XFF 信任、双桶限流短路、metrics 鉴权、CORS 配置测试。
- 当前主线测试结果：`47 passed`（2026-04-18）。

## 历史更新（2026-03）

- LangChain Agent 从 `create_openai_functions_agent` 迁移到 `create_react_agent`。
- 新增 `session_id` 会话记忆（内存/Redis）。
- 新增 SSE 流式接口 `POST /run_stream`。
- 前端支持流式渲染和执行步骤可视化。
- 增加最小可演示 RAG：本地开源 embedding + 向量检索链路。

## 快速开始

优先参考各子项目 README：

- `langchain-agent/README.md`（主线，建议先看）
- `code1/README.md`（原生版本）

常用启动方式（主线）：

```powershell
cd E:\code
.\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements.txt

cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

前端（可选）：

```powershell
cd E:\code\code1\frontend
npm install
npm run dev
```

## 下一步计划
可观测性完善（分阶段指标、错误聚合、报表）
持久化改造（Redis 持久化 + 运行记录落库）
队列化与异步任务化
多实例/分布式部署

## 说明

- 该仓库用于 AI Agent 工程化实践。
- 如需查看版本差异，请阅读 `COMPARISON.md`。
