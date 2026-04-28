# 电商运营 AI Agent 架构设计文档

## 1. 文档信息
- 项目名称：Merchant Ops Copilot（LangChain ReAct）
- 技术栈：Python、FastAPI、LangChain、Redis、Next.js、SSE、Chroma/InMemory、Sentence-Transformers
- 版本：v1（当前实现）
- 目标读者：后端开发、AI 应用工程师、面试评审

## 2. 背景与目标
本项目面向电商运营场景，用户输入业务问题（如流量下滑、ROI 下降、库存风险），系统通过 ReAct Agent 与工具调用完成归因和建议输出。

核心目标：
1. 构建可运行的 Agent 工程闭环，而不仅是单次 LLM 调用。
2. 支持流式执行过程可视化（Thought/Action/Observation）。
3. 支持 RAG 检索增强，提升策略类问答的准确性。
4. 提供基础稳定性治理（限流、超时、重试、降级）与安全治理（鉴权、输入限制、注入拦截）。
5. 支持从单机演示平滑演进到多实例部署。

## 3. 范围与非目标
范围：
1. `/run` 同步接口与 `/run_stream` 流式接口。
2. ReAct Agent 多轮工具调用。
3. 会话短期记忆（内存/Redis）。
4. 混合检索 RAG（向量 + BM25 + RRF + 重排）。
5. 基础可观测性与回归测试机制。

非目标：
1. 不实现复杂 RBAC 与多租户权限系统。
2. 不实现全量任务队列调度平台（当前仍以同步执行为主）。
3. 不做训练/微调流水线。

## 4. 总体架构

```text
[Next.js Frontend]
        |
        | HTTP / SSE
        v
[FastAPI Backend]
  |  校验层: 鉴权/限流/输入安全/超时重试降级
  |  编排层: ReAct Agent + Tool Routing + Early Stop
  v
[Tools Layer]
  | traffic_analyze / ads_analyze / inventory_check / product_diagnose / retrieve_knowledge
  v
[RAG Layer]
  | 文档加载 -> 切片 -> 向量召回 + BM25召回 -> RRF融合 -> metadata过滤 -> 重排
  v
[Stores]
  | SessionStore(memory/redis)
  | RateLimiter(memory/redis)
  | VectorStore(chroma/in_memory)
```

## 5. 模块划分

### 5.1 前端层（Next.js）
职责：
1. 输入 query/context/session_id。
2. 默认走 `/run_stream`，实时渲染执行步骤与最终建议。
3. 展示关键指标（latency、fallback、loop 等）。

### 5.2 API 层（FastAPI）
职责：
1. 提供 `POST /run`、`POST /run_stream`、`GET /health`、`GET /metrics/*`。
2. 执行通用治理逻辑：鉴权、限流、输入长度限制、注入检查。
3. 调用 `run_agent` 并封装响应模型。

### 5.3 Agent 编排层
职责：
1. ReAct 多轮推理与工具调用。
2. 工具路由（按 query/context 关键词筛选工具）。
3. 早停策略（单工具高置信场景快速返回）。
4. 请求内工具缓存（同请求重复工具调用去重）。

### 5.4 工具层
职责：
1. 承载业务诊断工具（流量、广告、库存、商品）。
2. 暴露统一输入输出，便于 Agent 调用。
3. `retrieve_knowledge` 对接 RAG。

### 5.5 RAG 层
职责：
1. 文档解析（支持 front matter metadata）。
2. 中文友好切片。
3. 双路召回 + 融合 + 重排。
4. 返回结构化证据片段用于 Agent 生成结论。

### 5.6 存储与缓存层
职责：
1. SessionStore 管理会话短期记忆。
2. RateLimiter 管理请求配额。
3. 向量索引与稀疏索引缓存，减少重复构建开销。

## 6. 核心流程设计

### 6.1 `/run` 同步流程
1. 接收请求（query/context/session_id）。
2. 安全校验（API Key、长度限制、注入规则）。
3. 限流判定（会话桶 + IP 桶）。
4. 执行 `run_agent`（ReAct + 工具调用）。
5. 返回 `RunResponse(final_answer, steps, metrics)`。

### 6.2 `/run_stream` 流式流程
1. 前置校验同 `/run`。
2. 创建事件 sink，运行 Agent 时推送事件。
3. 持续输出 `agent_action/tool_observation/final_response/error/stream_metrics`。
4. 前端边收边渲染执行过程。

### 6.3 ReAct 执行流程
1. 构建 prompt（包含工具清单、格式约束、历史会话）。
2. 进入循环：Thought -> Action -> Action Input -> Observation。
3. 达到停止条件后输出 Final Answer。
4. 记录步骤轨迹和阶段耗时。

## 7. RAG 检索流程与参数

### 7.1 文档处理
1. 读取 `RAG_DOCS_DIR`（默认 `knowledge/seed`）下 `.md/.txt`。
2. 解析 front matter metadata。
3. 切片参数：`chunk_size=450`，`chunk_overlap=80`。

### 7.2 检索流程
1. 向量召回：`similarity_search(k=fetch_k)`。
2. BM25 召回：基于本地 tf/df/avgdl 稀疏索引。
3. RRF 融合：`score += 1 / (RRF_K + rank)`，`RRF_K=60`。
4. metadata 过滤：`merchant_id/category/time_range`。
5. 轻量重排：query/context 词重叠 + metadata boost。
6. 输出 top_k（默认 `RAG_TOP_K=3`）。

### 7.3 关键配置
1. `RAG_FETCH_K=12`
2. `RAG_VECTOR_BACKEND=chroma`（失败回退 `in_memory`）
3. `RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
4. `RAG_EMBEDDING_DEVICE=cpu`

## 8. 会话记忆设计
1. 默认保留最近 `MAX_HISTORY_TURNS=8` 轮。
2. 内存实现：进程内字典 + 锁。
3. Redis 实现：`RPUSH + LTRIM + EXPIRE`。
4. 会话键：`session_id`，未传则用 `default`。

## 9. 稳定性与安全治理

### 9.1 稳定性
1. 超时：`REQUEST_TIMEOUT_SECONDS` 与流式超时配置。
2. 重试：失败后有限次重试 + backoff。
3. 降级：超时或错误时返回可解释降级结果。
4. 限流：固定窗口限流，支持 Redis/内存 fallback。

### 9.2 安全
1. API Key 鉴权（可开关）。
2. 输入长度限制（query/context）。
3. Prompt 注入规则拦截。
4. CORS 白名单配置。

## 10. 可观测性设计
1. 结构化日志（JSON）统一字段：`request_id/session_id/endpoint/status/latency/error_type`。
2. 分阶段指标：`llm_latency_ms/tool_latency_ms/loop_count/retrieve_hits`。
3. 流式特有指标：`ttfb_ms/event_count/event_completeness`。
4. 统计接口：`/metrics/error_types`、`/metrics/stability`。

## 11. 部署架构

### 11.1 本地开发
1. 单实例 FastAPI + 可选前端。
2. 会话默认内存。

### 11.2 Compose 模式
1. FastAPI + Redis。
2. 会话与限流状态外置，支持多实例演进。

## 12. 测试与质量保障
1. 单元与接口测试：`pytest`。
2. bad case 回归：固定样例脚本。
3. CI 自动化：push/PR 自动跑测试与回归门禁。
4. 回归产物归档：结果 JSON + 后端日志。

## 13. 已知限制
1. 当前主链路仍是请求内同步执行，复杂请求延迟受 LLM 与工具调用影响较大。
2. 会话记忆为短期窗口策略，长期记忆未做摘要持久化。
3. 任务队列、优先级调度、死信重试尚未完整引入。

## 14. 演进路线（建议）
1. 引入队列化与异步任务执行（API 入队 + Worker 消费）。
2. 会话记忆升级为“短窗 + 摘要”混合策略。
3. 检索评测体系升级（离线评测集 + 指标看板）。
4. 多实例扩容与更细粒度限流熔断策略。

## 15. 验收标准
1. 功能验收：同步/流式接口稳定可用，RAG 可返回结构化证据。
2. 稳定性验收：超时、限流、降级路径可验证。
3. 安全验收：鉴权与输入防护可触发并返回预期状态码。
4. 质量验收：CI 绿灯，回归样例通过率达到预设阈值。

