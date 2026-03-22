# Merchant Ops Copilot

面向商家运营场景的 AI Agent 项目，采用 `Planner → Executor → Verifier` 工作流架构，支持通过 OpenAI API（或兼容接口）驱动智能规划与校验。

## 项目简介

这是一个 Day 1 MVP 项目，展示了如何构建一个面向商家运营场景的 AI 助手系统。系统能够接收用户的运营问题，自动规划工具调用、执行分析并校验结果，最终给出可执行的建议。

### 适用场景

- 电商平台商家运营诊断（流量下滑、广告效率、库存风险等）
- 多商家平台的智能客服或运营助手
- 商家数据洞察与决策支持

## 技术架构

### 系统组成

```
┌─────────────┐
│   前端界面    │ Next.js
└─────────────┘
       │
       │ HTTP POST /run
       ▼
┌─────────────┐
│  FastAPI    │ 后端 API
└─────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Planner → Executor → Verifier   │  Agent 工作流
└──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  工具函数层   │ Mock 实现
└─────────────┘
```

### 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 后端 | FastAPI | >=0.115 |
| 前端 | Next.js | 最新 |
| 模型 | OpenAI API | >=1.45.0 |
| 部署 | Uvicorn | >=0.30 |

### 工作流设计

**Planner（规划器）**
- 接收用户问题和上下文
- 判断场景类型，规划需要调用的工具
- 输出工具列表和执行目标

**Executor（执行器）**
- 按照规划调用工具函数
- 收集工具返回的分析结果
- 汇总数据供下一步处理

**Verifier（校验器）**
- 校验工具输出是否合理
- 生成最终的建议文本

### 工具接口

| 工具名称 | 功能 | 当前状态 |
|---------|------|----------|
| `traffic_analyze` | 流量分析 | Mock 实现 |
| `ads_analyze` | 广告效率分析 | Mock 实现 |
| `inventory_check` | 库存风险检查 | Mock 实现 |
| `product_diagnose` | 商品转化诊断 | Mock 实现 |

> 注：当前工具函数使用 Mock 数据演示工作流。实际落地时可对接真实数据源（如电商平台的生意参谋、商智等 API）。

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 18+
- OpenAI API Key（或兼容接口，如 DeepSeek）

### 安装依赖

```bash
# 后端
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 配置环境变量

在 Windows PowerShell 中设置：

```powershell
# 必填：OpenAI API Key
$env:OPENAI_API_KEY="<YOUR_KEY>"

# 选填：模型配置
$env:OPENAI_MODEL="deepseek-chat"
$env:OPENAI_BASE_URL="https://api.deepseek.com"

# 选填：是否允许规则兜底（默认 false）
$env:ALLOW_RULE_FALLBACK="false"
```

### 启动服务

**启动后端**

```bash
uvicorn backend.main:app --reload --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

**启动前端**

```bash
cd frontend
npm run dev
```

访问 `http://localhost:3000`

### 测试 API

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "本周流量下滑且转化偏低，请给出排查和行动方案",
    "context": {
      "merchant_id": "demo-001",
      "time_range": "last_7_days",
      "category": "retail"
    }
  }'
```

## API 文档

### POST /run

执行 Agent 任务，返回分析建议。

**请求参数**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 用户输入的问题描述 |
| context | object | 否 | 上下文信息，可包含 merchant_id、time_range 等 |

**响应示例**

```json
{
  "final_answer": "建议如下：1. 检查关键词匹配度...",
  "steps": [
    {
      "name": "Planner",
      "input": {"query": "...", "context": {...}},
      "output": {"scenario": "...", "tool_names": [...]},
      "duration_ms": 1500
    },
    {
      "name": "Executor",
      "input": {...},
      "output": {...},
      "duration_ms": 2300
    },
    {
      "name": "Verifier",
      "input": {...},
      "output": {...},
      "duration_ms": 800
    }
  ],
  "metrics": {
    "latency_ms": 4600,
    "fallback_used": false
  }
}
```

## 项目结构

```
e:\code/
├── backend/           # 后端 API
│   ├── main.py        # FastAPI 应用入口
│   ├── models.py      # Pydantic 数据模型
│   └── settings.py    # 配置管理
├── agent/             # Agent 核心逻辑
│   ├── state_machine.py  # 状态机实现
│   └── model_client.py   # OpenAI 客户端
├── tools/             # 工具函数层
│   └── mock_tools.py  # Mock 工具实现
├── frontend/          # 前端界面
│   └── app/
│       ├── page.tsx   # 主页面
│       └── globals.css # 全局样式
├── eval/              # 评测脚本
│   └── run-eval.mjs   # 快速离线评测
├── requirements.txt   # Python 依赖
└── README.md
```

## 技术亮点

### 1. 清晰的工作流分离

将 Agent 逻辑拆分为 Planner、Executor、Verifier 三个独立阶段，便于：
- 单独优化每个环节
- 增加或减少工具调用
- 调试和问题排查

### 2. 可扩展的工具系统

工具函数采用统一接口 `tool(query: str, context: Dict) -> Dict`，便于：
- 添加新的分析工具
- 替换 Mock 实现为真实数据查询
- 灵活组合不同工具

### 3. 兜底机制设计

支持模型失败时的规则兜底（通过 `ALLOW_RULE_FALLBACK` 配置），保证系统可用性。

### 4. 前后端分离

Next.js 前端 + FastAPI 后端，展示全栈开发能力。

## 后续扩展规划

### 数据接入方案

| 方案 | 数据来源 | 复杂度 | 适用场景 |
|------|----------|--------|----------|
| 方案一 | 用户上传 Excel/CSV | 低 | 个人工具 |
| 方案二 | API 对接第三方平台 | 中 | 商家授权接入 |
| 方案三 | 平台内置数据 | 高 | 电商平台自研 |

### 功能增强

- 支持更多运营场景（竞品分析、定价建议等）
- 优化 Prompt，提升规划稳定性
- 增加多轮对话能力
- 添加数据可视化展示

### 部署优化

- Docker 容器化
- 添加用户认证和权限管理
- 支持批量商家分析

## 评测

运行快速评测脚本：

```bash
node eval/run-eval.mjs
```

输出示例：
```
Summary: 3/3 cases returned suggestions.
```

## 常见问题

### Q: 工具函数返回的是 Mock 数据吗？

A: 是的。当前 MVP 阶段使用 Mock 数据演示工作流。实际落地时需要对接真实数据源。

### Q: 商家 ID 是如何使用的？

A: 商家 ID 是多商家平台架构的一部分，用于区分不同商家的数据。当前演示中可以填任意值。

### Q: 如何切换到其他模型？

A: 通过环境变量 `OPENAI_MODEL` 和 `OPENAI_BASE_URL` 配置即可，支持任何 OpenAI 兼容接口。

## License

MIT

## 作者

用于实习面试技术展示
