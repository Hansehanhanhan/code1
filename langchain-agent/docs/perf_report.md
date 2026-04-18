# 性能压测报告（单机）

## 1. 测试目标
- 记录当前版本在单机场景下的吞吐、延迟、首包延迟（TTFB）与状态码分布。
- 沉淀可重复执行的压测流程，便于面试展示“优化前后对比”。

## 2. 测试环境
- OS: Windows
- 服务: FastAPI + Uvicorn
- 脚本：
  - 单场景：`scripts/load_test.py`
  - 矩阵场景：`scripts/load_test_matrix.py`
- 默认场景配置：`docs/load_test_cases.json`

启动后端：

```powershell
cd E:\code\langchain-agent
$env:PYTHONPATH='.'
& "..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

执行压测矩阵：

```powershell
cd E:\code\langchain-agent
& "..\.venv\Scripts\python.exe" .\scripts\load_test_matrix.py --cases-file .\docs\load_test_cases.json --output-json .\docs\load_test_results.json --output-md .\docs\load_test_results.md
```

## 3. 指标说明
- `success_rate`: 2xx 响应占比。
- `duration_s`: 整轮压测总时长。
- `achieved_rps`: 实际吞吐（请求数 / 总时长）。
- `p50/p95/p99`: 端到端延迟分位数（毫秒）。
- `ttfb_p50/ttfb_p95`: 首包延迟分位数（毫秒），对流式接口更关键。
- `status_counts`: 状态码分布（如 `200`、`429`、`500`）。

## 4. 结果记录模板

| 场景 | 请求数 | 并发 | 成功率 | RPS | P50(ms) | P95(ms) | P99(ms) | TTFB P95(ms) | 主要状态码 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| health_baseline | 500 | 50 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| run_light | 50 | 5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| run_stream_light | 30 | 3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 5. 当前瓶颈与改进方向
- `run` / `run_stream` 的主要延迟通常来自外部 LLM 与工具链路，而非 Web 框架本身。
- 若出现 `429` 增多，说明限流生效，可结合 `GET /metrics/stability` 看 `rate_limit_exceeded_total`。
- 若 `degraded_total` 升高，说明下游不稳定或超时偏多，需要优先排查外部依赖与超时策略。
