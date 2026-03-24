# AI Agent 商家运营助手项目

本仓库包含两个版本：
- `code1`：原生状态机版（Planner -> Executor -> Verifier）
- `langchain-agent`：LangChain ReAct 版（本次重点更新）

## 本次改动（2026-03）

### 1) LangChain Agent 切换为 ReAct 架构
- 将 `create_openai_functions_agent` 迁移为 `create_react_agent`
- 使用 ReAct 推理格式：`Question -> Thought -> Action -> Action Input -> Observation -> Final Answer`
- 工具调用输入统一为 JSON 字符串，兼容 `query/context`

### 2) 增加会话记忆（session_id）
- 后端请求模型新增 `session_id`（可选）
- 相同 `session_id` 复用进程内短期历史（最近若干轮）
- 服务重启后记忆清空（当前为内存实现）

### 3) 增加流式接口（SSE）
- 新增接口：`POST /run_stream`
- 实时事件：
  - `agent_action`：每轮 Thought/Action/Action Input
  - `tool_observation`：工具观察结果与耗时
  - `final_response`：最终完整响应
  - `error`：错误信息

### 4) 执行步骤可见化
- `steps` 不再只有最终聚合
- 现在会返回每一轮 ReAct 轨迹（`ReAct Loop N`）+ 最终 `Agent` 汇总

### 5) 前端联动升级
- 默认调用 `/run_stream`，边执行边展示步骤
- 新增 `session_id` 输入框并随请求提交
- 运行指标中文化显示：
  - `latency_ms -> 延迟(毫秒)`
  - `fallback_used -> 是否使用兜底（是/否）`
- 最终建议文本优化：
  - 去除 `**` 粗体符号
  - 自动断行提升可读性

## 快速启动

### 后端（LangChain ReAct）
```powershell
cd langchain-agent
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 前端
```powershell
cd code1/frontend
npm run dev
```

访问：`http://localhost:3000`

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
`POST /run_stream`，`Content-Type: application/json`，响应类型为 `text/event-stream`。

