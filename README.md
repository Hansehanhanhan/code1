# AI Agent 商家运营助手项目

一个面向商家的 AI 运营诊断 Agent：用户以自然语言描述经营问题（如"近 7 天流量下滑，请排查"），Agent 通过功能调用（function-calling）自主编排分析工具、收集证据，输出可追溯的诊断结论与优化建议。

核心特点：

- **LangChain function-calling Agent**：模型主动调用分析工具（流量/转化/库存/广告等模拟工具）收集证据后作答，支持 SSE 流式输出执行轨迹。
- **多轮诊断记忆（AREX 式 LLM 提议 + 服务端仲裁）**：结论沉淀为结构化诊断状态，模型提交增量提议、服务端校验证据后合并，长会话不丢信息、结论可溯源。
- **异步任务（/jobs）**：长耗时诊断可异步提交、回放事件、取消与重试。
- **RAG 知识检索**：混合检索 + metadata 过滤，支持行业知识辅助。
- **工程化治理**：API Key 鉴权与租户隔离、限流、超时重试降级、结构化日志与观测指标。

本仓库为单一主线实现：

- `langchain-agent`：LangChain function-calling 工程版（当前主线，前端统一入口位于 `langchain-agent/frontend`）

## 项目目录结构

```text
E:\CODE
├── langchain-agent/          # LangChain 版本 (Framework Optimized)
│   ├── agent/                # 基于 LangChain 的 Agent 实现
│   ├── backend/              # FastAPI 后端实现
│   ├── frontend/             # Next.js 前端 UI（统一入口）
│   ├── rag/                  # 检索与知识库模块
│   ├── mcp_server/           # 轻量 MCP Server
│   ├── tools/                # 工具集
│   ├── tests/                # 测试
│   └── README.md             # LangChain 版详细文档
└── README.md                 # 本文件（项目主说明文档）
```

## 本次更新（2026-09-15）

### 1) 版本收敛：移除原生 code1，前端统一入口
- 删除原生状态机版 `code1`（Planner -> Executor -> Verifier）及技术对比文档 `COMPARISON.md`。
- 唯一前端 UI 迁移至 `langchain-agent/frontend`（保留 git 历史；`node_modules` 随目录一并保留，本地可免重装直接 `npm run dev`）。

### 2) AREX 记忆闭环定稿（LLM 提议 + 服务端仲裁）
- Agent 全文切换为 function-calling tool-calling（`create_tool_calling_agent`），工具参数结构化 `{query, context}`，`update_context` 以 JSON patch 提交增量提议。
- 证据引用收敛为纯三字段 `{evidence_id, request_id, observation_hash}`，服务端拒绝 `tool`/`observation_preview`/`tool_call_id` 等冗余字段；`tool_call_id` 由 `evidence_id` 后缀推导。
- 请求内工具缓存命中分支同样嵌入证据引用（同一 `evidence_id` 语义、序号递增），模型读到的每条 Observation 都携带合法引用。
- 模拟工具信号改为按商家上下文确定性输出（`_weak_signal`），同一商家不同措辞结果恒定。

### 3) 端到端验证与测试
- 新增 `scripts/e2e_verify.py`：真实 LLM × 真实 HTTP 端到端验证（默认自动拉起后端子进程），覆盖 `/run_stream` SSE 事件序列、`/sessions` 状态演化、`/run` 同步、`/jobs` 异步与 `/jobs/{id}/stream` 回放，全部断言通过（32/32）。
- 新增 `scripts/real_multi_round_probe.py`：两轮真实 API 探针，验收模型自主调用 `update_context`、`finding_count >= 1`、`constraint_status` 非空（本轮 `force_stopped=false`）。
- 测试基线：`131 passed`。

### 4) 可运行性
- 本机可直接启动前后端联调：后端 `uvicorn backend.main:app`（:8000，端点 `/run_stream`、`/run`、`/jobs`、`/sessions`），前端 `npm run dev`（:3000，Next.js 15）。

## 历史更新（2026-04-28）

### 1) 异步任务能力增强（Jobs）
- 新增 `POST /jobs/{job_id}/cancel`，支持排队中任务取消与运行中任务“取消请求”。
- 新增 `POST /jobs/{job_id}/retry`，支持对终态任务一键重试并生成新任务。
- 任务终态新增 `cancelled`，并完善事件回放中的取消/重试事件。

### 2) 任务恢复与一致性增强
- JobQueue 启动时新增恢复流程：
  - `running -> queued`（重启后自动补偿）
  - `cancel_requested -> cancelled`（避免悬挂状态）
  - 自动重入队 `queued` 任务
- 增加恢复日志事件，便于排障和审计。

### 3) 幂等提交（防重复入队）
- `POST /jobs` 支持请求体字段 `idempotency_key`。
- 相同 `idempotency_key` 的重复提交会复用已创建任务，不重复入队。
- 落库新增幂等字段与唯一索引（兼容旧库自动迁移）。

### 4) Agent 可用性体验优化
- 当请求缺少关键上下文（如 `merchant_id`、`time_range`）时，优先返回澄清问题而非直接泛化结论。
- 最终答案自动追加“证据来源”区块，提升可追溯性。

### 5) 测试覆盖
- 新增/扩展 jobs 取消、重试、幂等复用、重启恢复、澄清与证据提取相关测试。
- 当前主线测试结果：`58 passed`（2026-04-28）。

## 历史更新（2026-04-18）

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

常用启动方式（主线）：

```powershell
cd E:\code
.\langchain-agent\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements.txt

cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

前端（可选）：

```powershell
cd E:\code\langchain-agent\frontend
npm install
npm run dev
```

默认访问 `http://127.0.0.1:3000`（前端默认指向 `http://127.0.0.1:8000`）。本机测试建议保持 Redis 可用（`SESSION_BACKEND=redis` 时状态跨请求闭环）。

## 下一步计划
- 前端消费 `key_step` 事件（`first_evidence`/`context_update`/`direction_repair`）做执行步骤可视化突出与证据展示
- Agent 自省外环：verify/restart（对置信不足的结论触发生成-验证-修订）
- 命名清理：`ReActTraceCallbackHandler`、`loop_count` 等 ReAct 残留统一为 tool-calling 语义

## 说明

- 该仓库用于 AI Agent 工程化实践。
