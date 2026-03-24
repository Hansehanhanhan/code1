# Merchant Ops Copilot (LangChain ReAct)

基于 LangChain 的商家运营 Agent，当前实现为 **ReAct + 流式事件 + 会话记忆（session_id）**。

## 本次改动说明

### ReAct 架构
- 已切换为 `create_react_agent`
- Prompt 使用 ReAct 规范流程
- 工具输入约定为 JSON 字符串（含 `query/context`）

### 会话记忆
- 请求体新增可选字段：`session_id`
- 相同 `session_id` 复用短期历史（进程内）
- 重启服务后历史清空

### 流式接口
- 新增 `POST /run_stream`（SSE）
- 事件类型：
  - `agent_action`
  - `tool_observation`
  - `final_response`
  - `error`

### 执行步骤增强
- API 返回 `steps` 支持每轮 ReAct 轨迹
- 每轮记录：
  - `thought`
  - `action`
  - `action_input`
  - `observation`
  - `duration_ms`

## API

### 1) `POST /run`
同步返回完整结果。

请求示例：
```json
{
  "query": "本周流量下滑，请诊断原因",
  "context": {
    "merchant_id": "demo-001"
  },
  "session_id": "demo-session-001"
}
```

### 2) `POST /run_stream`
流式返回事件，`Content-Type: text/event-stream`。

每条事件格式：
```text
data: {"type":"agent_action","content":{...}}
```

## 启动

```powershell
cd langchain-agent
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

## 目录

```text
langchain-agent/
├── agent/
│   └── agent.py
├── backend/
│   ├── main.py
│   ├── models.py
│   └── settings.py
├── tools/
│   └── tools.py
└── requirements.txt
```

