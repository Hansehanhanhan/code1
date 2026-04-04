# Merchant Ops Copilot (LangChain ReAct)

基于 LangChain 的商家运营 Agent，当前实现为：
- ReAct 推理
- SSE 流式事件
- `session_id` 会话短期记忆
- RAG 检索工具（`retrieve_knowledge`）
- 仅本地开源 embedding（sentence-transformers）

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
├── rag/
│   ├── __init__.py
│   └── knowledge_base.py
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

## RAG 配置
- `RAG_ENABLED`：是否启用 RAG（默认 `true`）
- `RAG_DOCS_DIR`：知识库目录（默认 `knowledge/seed`）
- `RAG_VECTOR_BACKEND`：向量后端（默认 `chroma`）
- `RAG_TOP_K`：检索返回数量（默认 `3`）
- `RAG_EMBEDDING_MODEL`：默认 `BAAI/bge-small-zh-v1.5`
- `RAG_EMBEDDING_DEVICE`：默认 `cpu`（可改 `cuda`）

说明：
- 仅支持本地开源 embedding。
- 模型加载失败时不会回退 hash，会直接返回错误。
