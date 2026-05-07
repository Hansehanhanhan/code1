# Merchant Ops Copilot (LangChain ReAct)

基于 LangChain 的商家运营 Agent，当前实现为：
- ReAct 推理
- 工具路由 + 早停（减少无效多轮调用）
- SSE 流式事件
- `session_id` 会话记忆（支持内存/Redis）
- RAG 检索工具（`retrieve_knowledge`）
- RAG 质量升级：中文切片 + 混合检索（BM25 + 向量）+ 召回后重排 + metadata 过滤
- 稳定性治理：超时 + 重试 + 降级 + 按接口限流
- 安全治理：简单 API Key 鉴权 + 输入长度限制 + 基础注入防护
- 仅本地开源 embedding（sentence-transformers）
- 基础限流（按 `ip+session_id` 固定窗口）

## 最新更新（2026-04-28）

### 1) Jobs 能力升级
- 新增 `POST /jobs/{job_id}/cancel`：支持取消排队任务与运行中任务的取消请求。
- 新增 `POST /jobs/{job_id}/retry`：支持终态任务重试，返回新的 `job_id`。
- `POST /jobs` 支持 `idempotency_key`，重复提交同 key 会复用已有任务（防重复入队）。

### 2) 任务恢复与一致性
- JobQueue 启动时自动恢复未完成任务：
  - `running -> queued`
  - `cancel_requested -> cancelled`
  - 自动重入队 `queued` 任务
- 补充恢复日志与事件，提升可观测性与排障效率。

### 3) Agent 体验增强
- 缺少关键上下文（`merchant_id`、`time_range`）时优先返回澄清问题。
- 最终答案自动附加“证据来源”区块，提升结论可追溯性。

## 目录

```text
langchain-agent/
├── agent/
│   └── agent.py
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── rate_limit.py
│   ├── session_store.py
│   └── settings.py
├── mcp_server/
│   └── server.py
├── tools/
│   └── tools.py
├── rag/
│   ├── __init__.py
│   └── knowledge_base.py
├── tests/
│   ├── test_main_api.py
│   ├── test_rag_quality.py
│   ├── test_rate_limit.py
│   └── test_session_store.py
├── scripts/
│   ├── load_test.py
│   ├── load_test_matrix.py
│   ├── eval_badcase.py
│   └── run_badcase_regression.py
├── docs/
│   ├── load_test_cases.json
│   ├── perf_report.md
│   └── bad_case_regression.md
├── Dockerfile
├── docker-compose.yml
├── knowledge/
│   └── seed/
└── requirements.txt
```

## 启动

```powershell
cd E:\code
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements.txt

cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## 前端联调（可选）

```powershell
cd E:\code\code1\frontend
npm install
npm run dev
```

默认访问：`http://127.0.0.1:3000`。  
若后端地址不是 `http://127.0.0.1:8000`，请先设置：

```powershell
$env:NEXT_PUBLIC_BACKEND_URL='http://127.0.0.1:8000'
```

## API
- `POST /run`
- `POST /run_stream`
- `POST /jobs`（异步提交任务）
- `GET /jobs/{job_id}`（查询任务状态与结果）
- `POST /jobs/{job_id}/cancel`（取消任务）
- `POST /jobs/{job_id}/retry`（重试任务）
- `GET /jobs/{job_id}/events`（拉取任务事件）
- `GET /jobs/{job_id}/stream`（SSE 订阅任务事件）
- `GET /metrics/error_types`
- `GET /metrics/stability`

限流相关响应头：
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `Retry-After`（仅 429）

降级相关响应头：
- `X-Degraded: 1`（仅降级返回时）

鉴权请求头（开启后必填）：
- `X-API-Key`

## API 调用示例（含鉴权）

同步调用 `/run`：

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace_with_strong_secret" \
  -d '{
    "query": "本周流量下滑，请给排查建议",
    "context": {
      "merchant_id": "demo-001",
      "category": "retail",
      "time_range": "last_7_days"
    },
    "session_id": "demo-session-001"
  }'
```

流式调用 `/run_stream`：

```bash
curl -N -X POST "http://127.0.0.1:8000/run_stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace_with_strong_secret" \
  -d '{
    "query": "请分步骤输出本周优化策略",
    "context": {"merchant_id":"demo-001"},
    "session_id": "demo-session-002"
  }'
```

异步提交 `/jobs`：

```bash
curl -X POST "http://127.0.0.1:8000/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace_with_strong_secret" \
  -d '{
    "query": "请给我一份本周流量下滑的分步排查方案",
    "context": {"merchant_id":"demo-001","category":"retail"},
    "session_id": "demo-session-job-001",
    "idempotency_key": "job-demo-001-last7d-v1"
  }'
```

查询任务状态 `/jobs/{job_id}`：

```bash
curl "http://127.0.0.1:8000/jobs/<job_id>" \
  -H "X-API-Key: replace_with_strong_secret"
```

取消任务 `/jobs/{job_id}/cancel`：

```bash
curl -X POST "http://127.0.0.1:8000/jobs/<job_id>/cancel" \
  -H "X-API-Key: replace_with_strong_secret"
```

重试任务 `/jobs/{job_id}/retry`：

```bash
curl -X POST "http://127.0.0.1:8000/jobs/<job_id>/retry" \
  -H "X-API-Key: replace_with_strong_secret"
```

## 前端鉴权联动

前端请求后端时，需要把 `X-API-Key` 一并带上（仅当 `APP_AUTH_ENABLED=true` 时）。

```ts
const headers: Record<string, string> = {
  "Content-Type": "application/json",
};

if (apiKey?.trim()) {
  headers["X-API-Key"] = apiKey.trim();
}

await fetch(`${backendUrl}/run`, {
  method: "POST",
  headers,
  body: JSON.stringify(payload),
});
```

说明：
- 本地联调可以在前端输入框中临时填写 key。
- 不建议把真实密钥写入 `NEXT_PUBLIC_*` 环境变量（会暴露到浏览器）。
- 生产环境建议由后端网关/BFF 统一注入鉴权信息。

## Agent 决策效率优化

- 工具路由：根据 query/context 的关键词仅开放相关工具，减少无关工具的尝试。
- 早停策略：当路由高置信命中单工具且非综合型问题时，直接走单工具快速路径。
- 请求内工具缓存：同一请求内重复的同工具同参数调用直接命中缓存，避免重复耗时。

## 测试

```powershell
cd E:\code
.\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements-dev.txt

cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" -m pytest -q
```

覆盖范围：
- `/health`、`/run`、`/run_stream` 接口行为
- 会话存储（memory/redis fallback）逻辑
- 限流器（fixed window + fallback）逻辑
- Jobs 取消/重试/幂等复用/重启恢复逻辑
- Agent 澄清追问与证据来源附加逻辑

当前测试结果：`58 passed`（2026-04-28）。

## 结构化日志

后端日志采用 JSON 结构，核心字段包括：
- `request_id`
- `session_id`
- `endpoint`
- `status`（`success` / `error`）
- `latency_ms`
- `error_type`（失败时）
- `error_message`（失败时）
- `llm_latency_ms`
- `tool_latency_ms`
- `loop_count`
- `retrieve_hits`
- `ttfb_ms`（`/run_stream`）
- `event_count`（`/run_stream`）
- `event_completeness`（`/run_stream`）

字段字典：

| 字段 | 类型 | 说明 |
|---|---|---|
| `event` | string | 日志事件名（如 `request_started`/`request_finished`） |
| `request_id` | string | 单次请求唯一 ID，用于全链路关联 |
| `session_id` | string | 会话 ID，用于记忆与问题排查 |
| `endpoint` | string | 接口名（`run` / `run_stream`） |
| `status` | string | 请求状态（`success` / `error`） |
| `latency_ms` | int | 端到端耗时（毫秒） |
| `llm_latency_ms` | int | LLM 阶段累计耗时（毫秒） |
| `tool_latency_ms` | int | 工具阶段累计耗时（毫秒） |
| `loop_count` | int | ReAct 循环轮数 |
| `retrieve_hits` | int | RAG 命中片段总数 |
| `ttfb_ms` | int | 流式首包延迟（仅 `run_stream`） |
| `event_count` | int | 流式事件总数（仅 `run_stream`） |
| `event_completeness` | bool | 流式是否完整（有 `final_response` 且无 `error`） |
| `error_type` | string | 错误类型（仅失败） |
| `error_message` | string | 错误信息（仅失败） |

示例日志：

```json
{"event":"request_started","endpoint":"run","request_id":"...","session_id":"demo-001","context_keys":["merchant_id"]}
{"event":"request_finished","endpoint":"run","request_id":"...","session_id":"demo-001","status":"success","latency_ms":1820,"fallback_used":false,"llm_latency_ms":1240,"tool_latency_ms":390,"loop_count":3,"retrieve_hits":2}
{"event":"request_finished","endpoint":"run","request_id":"...","session_id":"demo-001","status":"error","latency_ms":731,"error_type":"RuntimeError","error_message":"Agent execution failed: ..."}
{"event":"request_finished","endpoint":"run_stream","request_id":"...","session_id":"demo-001","status":"success","latency_ms":2143,"ttfb_ms":182,"event_count":6,"event_completeness":true}
```

## 流式指标（/run_stream）

`/run_stream` 除了业务事件外，还会额外发送 `stream_metrics` 事件，包含：
- `ttfb_ms`：首包延迟（请求开始到首个 SSE 事件）
- `event_count`：本次流式过程产生的事件数
- `event_completeness`：是否完整（有 `final_response` 且无 `error`）

同时，`run_agent` 返回的 `metrics` 中包含分阶段指标：
- `llm_latency_ms`
- `tool_latency_ms`
- `loop_count`
- `retrieve_hits`

## 错误聚合统计

新增接口 `GET /metrics/error_types`，返回按错误类型聚合的计数结果，便于快速定位问题趋势。

## 稳定性指标统计

新增接口 `GET /metrics/stability`，返回稳定性治理相关计数，当前包含：
- `rate_limit_exceeded_total`
- `retry_total`
- `timeout_total`
- `degraded_total`

同时会附带按接口维度的子计数（例如 `run:retry_total`、`run_stream:degraded_total`），方便区分同步与流式链路表现。

说明：
- 当 `APP_AUTH_ENABLED=true` 时，`/metrics/*` 也需要 `X-API-Key`。

## MCP 接入（最小版）
本项目提供轻量 MCP Server（stdio），复用现有 Agent 和 RAG 逻辑。

可用工具：
- `run_agent`：执行 ReAct Agent
- `retrieve_knowledge`：调用本地 RAG 检索
- `health`：查看运行状态和关键开关

提示：
- 当启用 `APP_AUTH_ENABLED=true` 时，MCP 的 `run_agent/retrieve_knowledge` 参数里也需要传 `api_key`。
- MCP 已接入基础限流，建议调用方传 `client_id` 以获得稳定的限流分桶语义。

启动命令：

```powershell
cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" -m mcp_server.server
```

若报 `mcp` 缺失，请先安装依赖：

```powershell
cd E:\code
.\.venv\Scripts\python.exe -m pip install -r .\langchain-agent\requirements.txt
```

## RAG 配置
- `RAG_ENABLED`：是否启用 RAG（默认 `true`）
- `RAG_DOCS_DIR`：知识库目录（默认 `knowledge/seed`）
- `RAG_VECTOR_BACKEND`：向量后端（默认 `chroma`）
- `RAG_TOP_K`：检索返回数量（默认 `3`）
- `RAG_FETCH_K`：初始召回候选数（默认 `12`，重排后截断为 `RAG_TOP_K`）
- `RAG_EMBEDDING_MODEL`：默认 `BAAI/bge-small-zh-v1.5`
- `RAG_EMBEDDING_DEVICE`：默认 `cpu`（可改 `cuda`）

说明：
- 仅支持本地开源 embedding。
- 检索链路为混合检索：向量召回 + BM25 稀疏召回，使用 RRF 融合后再重排。
- 模型加载失败时不会回退 hash，会直接返回错误。
- metadata 过滤键：`merchant_id`、`category`、`time_range`（文档存在对应 metadata 时生效）。
- 当请求携带 metadata 过滤条件时：对“存在该 metadata 字段”的文档做严格匹配；缺失该字段的文档会保留参与重排。
- `knowledge/seed` 内置示例文档已带 front matter，可直接体验 metadata 过滤。
- Markdown 文档支持简单 front matter，例如：

```text
---
merchant_id: demo-001
category: retail
time_range: last_7_days
---
```

## 会话与限流配置
- `SESSION_BACKEND`：`memory` / `redis`（默认 `memory`）
- `SESSION_TTL_SECONDS`：会话过期秒数（Redis 下生效，默认 `86400`）
- `REDIS_URL`：Redis 连接地址（如 `redis://127.0.0.1:6379/0`）
- `JOB_DB_PATH`：异步任务持久化 SQLite 路径（默认 `.run/jobs.db`）
- `RATE_LIMIT_ENABLED`：是否启用限流（默认 `true`）
- `RATE_LIMIT_WINDOW_SECONDS`：限流窗口秒数（默认 `60`）
- `RATE_LIMIT_MAX_REQUESTS`：窗口内最大请求数（默认 `30`）
- `RATE_LIMIT_MAX_REQUESTS_RUN`：`/run` 每窗口上限（默认 `20`）
- `RATE_LIMIT_MAX_REQUESTS_STREAM`：`/run_stream` 每窗口上限（默认 `10`）
- `RATE_LIMIT_MAX_REQUESTS_IP`：纯 IP 硬限流桶每窗口上限（默认 `60`）
- `TRUST_X_FORWARDED_FOR`：是否信任 `X-Forwarded-For`（默认 `false`）
- `TRUSTED_PROXY_IPS`：受信代理 IP 列表（逗号分隔，仅在开启 `TRUST_X_FORWARDED_FOR` 时生效）

说明：
- 当 `TRUST_X_FORWARDED_FOR=true` 且 `TRUSTED_PROXY_IPS` 为空时，系统会忽略 `X-Forwarded-For` 并回退到 `request.client.host`。

## 稳定性治理配置
- `REQUEST_TIMEOUT_SECONDS`：`/run` 超时秒数（默认 `120`）
- `REQUEST_TIMEOUT_SECONDS_STREAM`：`/run_stream` 超时秒数（默认 `150`）
- `RUN_RETRY_ATTEMPTS`：失败重试次数（默认 `1`，总尝试次数=重试+1）
- `RETRY_BACKOFF_MS`：重试退避基准毫秒（指数退避，默认 `300`）
- `DEGRADE_ON_TIMEOUT`：超时是否降级返回（默认 `true`）
- `DEGRADE_ON_ERROR`：非超时错误是否降级返回（默认 `true`）

## 安全与鉴权配置
- `APP_AUTH_ENABLED`：是否启用简单 API Key 鉴权（默认 `false`）
- `APP_API_KEY`：服务鉴权密钥（`APP_AUTH_ENABLED=true` 时必填）
- `MAX_QUERY_CHARS`：`query` 最大长度（默认 `2000`）
- `MAX_CONTEXT_CHARS`：`context` JSON 序列化后的最大长度（默认 `8000`）
- `PROMPT_INJECTION_GUARD_ENABLED`：是否启用基础注入关键词拦截（默认 `true`）
- `APP_CORS_ORIGINS`：CORS 白名单（逗号分隔，默认本地前端域名）
- `APP_CORS_ALLOW_CREDENTIALS`：是否允许凭证（默认 `false`；当 origin 为 `*` 时会强制关闭）
- `AGENT_VERBOSE`：是否开启 Agent verbose 日志（默认 `false`）

说明：
- 鉴权失败返回 `401 Unauthorized`。
- 输入超限返回 `413`。
- 命中注入关键词规则返回 `400`。

## 非单机部署（推荐）
先复制配置：

```powershell
cd E:\code\langchain-agent
Copy-Item .env.example .env -Force
# 然后编辑 .env，填入 OPENAI_API_KEY
```

使用 Docker Compose 启动后端 + Redis：

```powershell
cd E:\code\langchain-agent
docker compose up --build
```

说明：
- 该模式已将会话状态外置到 Redis，支持后续多实例扩展。
- 如果 Redis 不可用，系统会自动回退到内存会话存储，保证可用性。

## 性能压测（单机）

项目提供两个压测脚本：
- `scripts/load_test.py`：单场景压测（支持输出 JSON、TTFB 指标、header/body-file）。
- `scripts/load_test_matrix.py`：多场景压测（读取 `docs/load_test_cases.json`，自动产出 JSON + Markdown 报告）。

单场景示例：

```powershell
cd E:\code\langchain-agent
& "..\.venv\Scripts\python.exe" .\scripts\load_test.py --url http://127.0.0.1:8000/health --requests 500 --concurrency 50 --output .\docs\load_test_single.json
```

矩阵场景示例：

```powershell
cd E:\code\langchain-agent
& "..\.venv\Scripts\python.exe" .\scripts\load_test_matrix.py --cases-file .\docs\load_test_cases.json --output-json .\docs\load_test_results.json --output-md .\docs\load_test_results.md
```

报告模板与口径说明见：`docs/perf_report.md`。

## Bad Case 回归

固定回归样例表：`docs/bad_case_regression.md`。  
可执行脚本：`scripts/run_badcase_regression.py`（会输出 `docs/bad_case_results.json`）。

示例：

```powershell
cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" .\scripts\run_badcase_regression.py
```

可选参数示例（适合 CI 或鉴权开启场景）：

```powershell
cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" .\scripts\run_badcase_regression.py --base-url http://127.0.0.1:8000 --output .\docs\bad_case_results.json --ready-timeout 60 --request-timeout 15 --api-key your_api_key
```

## 自动化回归（CI）

仓库已提供 GitHub Actions 工作流：`.github/workflows/ci.yml`，在每次 `push` / `pull_request` 自动执行：
- `pytest -q`
- 启动后端（CI 配置）
- 运行 bad case 回归脚本
- 运行门禁评估脚本 `scripts/eval_badcase.py`
- 上传 `bad_case_results.json` 与后端日志产物

本地可手动执行门禁评估：

```powershell
cd E:\code\langchain-agent
& "..\.venv\Scripts\python.exe" .\scripts\eval_badcase.py --input .\docs\bad_case_results.json --min-pass-rate 0.9 --max-avg-latency-ms 30000 --require-status 200 --require-expected false
```
