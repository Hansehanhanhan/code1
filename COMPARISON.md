# 商家运营 Agent 两版实现对比

本文对比仓库内两套实现：
- `code1`：原生状态机版（Planner -> Executor -> Verifier）
- `langchain-agent`：LangChain ReAct 版（当前主线）

## 一览表

| 维度 | `code1` 原生状态机 | `langchain-agent` ReAct |
|---|---|---|
| 核心架构 | 手写状态机（三阶段） | `create_react_agent` + `AgentExecutor` |
| 执行轨迹 | 固定 3 步：Planner/Executor/Verifier | 多轮 ReAct Loop（Thought/Action/Observation） |
| 后端接口 | `POST /run` | `POST /run` + `POST /run_stream`（SSE） |
| 记忆能力 | 无会话记忆字段 | 支持 `session_id`，进程内短期记忆 |
| 流式能力 | 无 | 有，实时推送工具调用与观察结果 |
| 工具接入 | 手动工具注册与调度 | LangChain `StructuredTool` 封装 |
| 兜底策略 | 支持规则兜底（可配置） | 解析错误自动处理；`fallback_used=false`（当前实现） |
| 代码控制力 | 高，可精细控制每个阶段 | 更偏框架标准化，开发速度更快 |
| 依赖复杂度 | 低 | 中（增加 LangChain 相关依赖） |

## 架构差异

### 1) 原生状态机版（`code1`）
- 主流程固定：`_plan -> _execute -> _verify`
- 规划与校验阶段可走 LLM，也可走规则兜底
- 工具统一签名：`tool(query, context) -> dict`
- 输出结构稳定，适合教学和可控实验

### 2) LangChain ReAct 版（`langchain-agent`）
- 使用 ReAct 模式动态决定每轮工具调用
- 通过回调收集每轮轨迹，落到 `steps` 中
- 支持流式事件，前端可边跑边显示
- 支持同 `session_id` 的会话短期记忆（内存存储，重启丢失）

## API 与数据模型对比

### 请求模型

| 字段 | `code1` | `langchain-agent` |
|---|---|---|
| `query` | 必填 | 必填 |
| `context` | 可选 | 可选 |
| `session_id` | 不支持 | 支持（可选） |

### 响应模型

两版都返回：
- `final_answer`
- `steps`
- `metrics`（`latency_ms`、`fallback_used`）

差异在 `steps` 语义：
- `code1`：步骤名固定为 `Planner/Executor/Verifier`
- `langchain-agent`：步骤名以 `ReAct Loop N` 为主，最后补一条 `Agent` 汇总

### 流式能力

| 维度 | `code1` | `langchain-agent` |
|---|---|---|
| SSE 接口 | 无 | `POST /run_stream` |
| 事件类型 | 无 | `agent_action` / `tool_observation` / `final_response` / `error` |
| 前端体验 | 等待整包结果 | 实时展示每轮抉择与工具结果 |

## 前端联动现状

当前前端位于 `code1/frontend`，但已按 ReAct 版后端联动升级：
- 默认请求 `langchain-agent` 的 `/run_stream`
- 支持输入并提交 `session_id`
- 实时显示每一轮 ReAct 执行步骤
- 指标和步骤字段已中文化显示

## 选型建议

### 选 `code1`（原生状态机）如果你希望
- 深入理解 Agent 分层执行原理
- 强控制每个阶段逻辑与兜底策略
- 依赖更轻、调试路径更直接

### 选 `langchain-agent`（ReAct）如果你希望
- 快速得到可用的多轮工具调用 Agent
- 直接获得流式可视化能力
- 更容易扩展为多轮对话产品形态

## 当前结论

仓库目前推荐主线是 `langchain-agent`：
- 功能完整度更高（ReAct + 流式 + session 记忆）
- 与现有前端展示能力匹配
- 更适合对外演示“可观察的 Agent 决策过程”
