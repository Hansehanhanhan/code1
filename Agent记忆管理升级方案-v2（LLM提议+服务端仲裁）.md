# 电商运营 AI 助手 · 记忆管理升级方案 v2（LLM 提议 + 服务端仲裁）

> 设计来源：BAAI AREX《Towards a Recursively Self-Improving Agent for Deep Research》（arXiv:2607.21461v2）
> 修订记录：v1（架构概念稿，LLM 完整覆盖状态）→ v2（吸收代码审查，改为 **LLM 提交增量提议，服务端验证并合并**）
> v2 核心转变：**模型不拥有状态覆盖权，服务端是状态唯一权威**。

## 0. v2 相对 v1 的修订总览

| # | v1 问题（审查指出） | v2 修正 |
|---|---|---|
| 1 | 澄清检查在 run_agent 开头，未读历史，状态帮不上忙 | 澄清流程重排：请求覆盖 → context_slots 补全 → 仍缺才澄清 |
| 2 | 快速路径绕过 Agent 主循环，不更新状态 | 快速路径做**确定性状态更新**（服务端写，不依赖 LLM） |
| 3 | LLM 提交完整状态，旧事实可能被覆盖丢失 | 改为**增量 patch**：add/remove/reject，服务端合并去重 |
| 4 | source 只是字符串，可伪造 | 证据绑定 request_id + evidence_id + observation_hash，服务端校验真实执行轨迹 |
| 5 | 并发请求用旧状态覆盖，丢状态 | version + 乐观锁 CAS（Redis WATCH/Lua，内存全事务锁） |
| 6 | 状态无限增长 | 服务端硬限制：各字段条数/claim 长度/总字节，按规则淘汰 |
| 7 | SessionStore 接口改动被低估 | 明确新接口三件套：get_state / update_state / get_snapshot，双后端 + 兼容测试 |
| 8 | session_id 可越权读他人状态 | key 加租户范围 + owner/merchant_id 校验 |

## 0.1 落地状态（2026-09-15 核对）

P1 / P2 / P3 三个阶段均已实现，与 v2 设计一致；当前测试基线 `150 passed`，真实 API 两轮探针与 `scripts/e2e_verify.py`（32/32）通过。原“初审 verify/restart 外环”已落地为自省外环。实现核对差异如下：

| 设计点 | v2 原述 | 实现核对 |
|---|---|---|
| Agent 引擎 | Tool-Calling 可执行工具 | 已切换 `create_tool_calling_agent`（function-calling）；`ToolCallTraceCallbackHandler` 作为轨迹回调实现 |
| patch 载体 | `update_context` 工具 | 工具参数为结构化 `{query, context}`，`update_context` 以 JSON 字符串提交增量 patch，服务端解析校验并 CAS 合并 |
| 证据引用字段 | `evidence_id` / `observation_hash` 等 | 收敛为**纯三字段** `{evidence_id, request_id, observation_hash}`；`tool` / `observation_preview` / `tool_call_id` 不被接受；`tool_call_id` 由 `evidence_id` 后缀推导 |
| 请求内缓存命中 | 未提及 | 缓存命中分支同样嵌入证据引用（同一 `evidence_id` 语义、序号递增），保证模型读到的每条 Observation 都携带合法引用 |
| 澄清流程重排 | §5 新流程 | `_merge_context_with_slots`（agent.py:517）→ `_missing_context_keys`（agent.py:1293）：请求覆盖 → context_slots 补全 → 澄清，第二轮省略上下文不重复澄清已验证 |
| 服务端兜底 | §12 P3 | `_finalize_request_state` 确定性沉淀（带证据结论提升为 findings 并回填 citations、recommendations → candidates、异常 summary → unresolved_constraints）+ AREX 外环 `_run_refine_pass`（`MAX_REFINE_ROUNDS=2`） |
| 自省外环 | v2“初审 verify/restart” | `_run_outer_loop` 已落地：确定性置信三分 `accept≥0.7/verify 0.4~0.7/restart<0.4`；verify 以 LLM 复核（`_run_llm_verify` 逐约束 verdict、失败回退确定性交叉验证+refine）为主；restart 为保守版保留已验证进度（`MAX_RESTART_ROUNDS=1`）；总预算 `MAX_OUTER_ROUNDS=3`、会话级开关 `OUTER_LOOP_ENABLED`；决策经 SSE `outer_decision` 透出并进入 `metrics.outer_rounds/outer_decisions` |
| 工具输出确定性 | 未提及 | 模拟工具信号按商家上下文确定性输出（`_weak_signal`），同一商家不同措辞恒定 |
| 并发与双后端 | §7 / §9 | Redis WATCH+CAS、内存全事务锁；`tests/test_redis_store.py` 独立 `/15` 库专项覆盖（CAS 冲突、双后端一致、并发不丢、TTL、淘汰） |
| 身份隔离 §8 | "未完成，P0 前置" | P0 已落地：`api_keys.json` 身份映射（`backend/identities.py`）+ session 所有者绑定 + merchant 权限 + jobs 归属校验（401 / 403） |
| session 快照 | `get_snapshot` | `GET /sessions/{session_id}` 返回状态快照（含 `state_version`、findings 数等） |

> 正文保留 v2 的设计原述作为评审基线，上表为实现核对记录。原正文中的行号引用（`agent.py:691` 澄清位置、`agent.py:730` 快速路径）为旧版代码，已随重构失效，以核对表为准。

---

## 1. 设计目标（不变）

| 目标 | 升级后效果 |
|---|---|
| 长会话不丢信息 | 结论沉淀进状态，服务端合并，跨轮保留 |
| 不重复问上下文 | context_slots 自动补全（P1 先落地） |
| 结论可溯源 | 每条 finding 绑定不可伪造的工具执行证据 |
| 不重走弯路 | rejected_candidates 显式记录 |
| 可靠工程而非概念稿 | LLM 提议 + 服务端仲裁，测试可覆盖 |

---

## 2. 状态设计（DiagnosticState v2）

```json
{
  "version": 7,
  "context_slots": {
    "merchant_id": "demo-001",
    "time_range": "last_7_days"
  },
  "verified_findings": [
    {
      "id": "f-001",
      "claim": "近 7 天流量下滑 22%，主要来自搜索渠道",
      "evidence": {
        "evidence_id": "req_0001:tool_0003",
        "tool": "traffic_analyze",
        "request_id": "req_0001",
        "tool_call_id": "tool_0003",
        "observation_hash": "sha256:9f86d0..."
      },
      "confidence": "high",
      "status": "active",
      "created_at": "2026-09-13T10:00:00Z"
    }
  ],
  "current_candidates": ["广告投放效率下降", "详情页转化率偏低"],
  "unresolved_constraints": ["缺转化漏斗分环节数据"],
  "validity_concerns": ["库存数据与流量结论可能冲突"],
  "rejected_candidates": ["平台流量规则变更（证据不足，已排除）"],
  "next_step_plan": ["查转化漏斗 → 对比历史同期 → 输出诊断报告"]
}
```

### 关键区分（审查第 1 点）

**context_slots 与 verified_findings 是两类东西，必须分开：**

| | context_slots | verified_findings |
|---|---|---|
| 性质 | 输入参数（会话上下文） | 诊断结论 |
| 例子 | merchant_id、time_range | "流量下滑 22%" |
| 谁来写 | 请求显式传入 + 服务端落槽 | LLM 提议 + 服务端校验证据后合并 |
| 用于 | 澄清流程补全 | 回答复用、溯源 |

**禁止**把"time_range 已确认"写进 unresolved_constraints——已确认条件和未解决约束是两个字段。

---

## 3. 核心机制：update_context（增量提议，非完整覆盖）

### 3.1 LLM 提交物（patch schema）

```json
{
  "reason": "traffic_analysis_completed",
  "context_slots": { "merchant_id": "demo-001", "time_range": "last_7_days" },
  "add_findings": [
    {
      "claim": "近 7 天流量下滑 22%",
      "confidence": "high",
      "evidence": {
        "evidence_id": "req_0001:tool_0003",
        "request_id": "req_0001",
        "observation_hash": "sha256:9f86d0..."
      }
    }
  ],
  "supersede_findings": [],
  "add_candidates": ["广告投放效率下降"],
  "reject_candidates": ["平台流量规则变更"],
  "add_validity_concerns": ["库存数据与流量结论可能冲突"],
  "replace_next_step_plan": ["查转化漏斗 → 输出诊断报告"]
}
```

> 说明：`step_id` 仅用于展示，不作为证据主键。正式 patch 必须提交服务端生成的 `evidence_id` 和 `observation_hash`；模型不能自行生成可信的证据标识。

### 3.2 服务端处理管线（update_state）

```
收到 patch + expected_version
  → 校验 reason 非空
  → 校验每条 add_findings.evidence 命中"本次请求真实执行过的工具观察"（证据表）
  → 合并 + 去重（按 id / claim 归一）
  → 容量限制 + 淘汰（见 §6）
  → version + 1，CAS 写回
  → 记录 before/after 日志
  → 返回 {ok, new_version} 或冲突错误（调用方按 patch 类型重试）
```

**职责划分：**

| 动作 | 谁负责 |
|---|---|
| 提议"想记住/忘记什么" | LLM（patch） |
| 合并、去重、限长、淘汰 | 服务端 |
| 校验证据真实性 | 服务端（证据表） |
| 版本控制、并发安全 | 服务端 |
| 日志、可观测 | 服务端 |

---

## 4. 证据引用机制（防伪造，审查第 4 点）

### 4.1 轨迹证据表

服务端在**每次工具调用结束时**记录一条证据：

```json
{
  "request_id": "req_0001",
  "tool_call_id": "tool_0003",
  "evidence_id": "req_0001:tool_0003",
  "tool": "traffic_analyze",
  "observation_hash": "sha256:9f86d0...",
  "observation_preview": "traffic dropped 22%...",
  "ts": "..."
}
```

- `observation_hash = sha256(json.dumps(observation, sort_keys=True))`
- `evidence_id` 和 `tool_call_id` 由服务端生成，`tool_loop_index` 只用于前端展示，不作为证据主键
- 证据表是**本次请求的轨迹**，请求结束后再清理或归档；状态更新重试完成前不能提前删除
- `evidence_record` 只证明工具真实执行过，不等于已验证诊断结论

### 4.2 校验规则

- `add_findings` 每条必须带 `evidence_id` 和 `observation_hash`，且 `evidence_id` 必须命中本次请求的证据表
- 命中的 observation_hash 与证据表一致才算合法
- 不合法 → **整条 finding 拒绝**（不是静默忽略），并返回错误让 LLM 修正
- 服务端**只允许**状态引用本次真实执行过的工具观察——模型不能声称"traffic_analyze 说 X"除非真的发生过

### 4.3 Finding 的生命周期

- `verified_findings` 默认只允许新增或标记为 `superseded`，不允许模型直接物理删除历史 finding
- `remove_finding_ids` 改为 `supersede_findings`，新旧 finding 都必须引用真实证据
- 被替代的 finding 保留在审计记录中，默认不注入有效上下文
- 工具 observation 自动进入证据表，但只有通过 patch 仲裁的结论才能进入 `verified_findings`

---

## 5. 澄清流程重排（审查第 1 点）

### 5.1 现状问题

`run_agent()` 开头先 `_missing_context_keys()`，缺了就返回澄清——**还没读历史状态**（agent.py:691）。所以"状态里有 merchant_id，第二轮不澄清"不会自动发生。

### 5.2 新流程

```
请求 context
  ↓
① 当前请求显式传入的 context 覆盖（最高优先）
  ↓
② 从会话 context_slots 补全缺失键（会话记忆）
  ↓
③ 仍缺失 → 澄清
```

对应代码改动：`run_agent()` 里先 `get_state(sid).context_slots` 合并进 context，再做 `_missing_context_keys()`。

### 5.3 效果

- 第一轮："分析 demo-001 上周流量" → context_slots 写入 {merchant_id, time_range}
- 第二轮："那广告呢？" → 缺的键从槽位补全 → **不澄清，直接分析**

---

## 6. 容量硬限制（审查第 6 点）

"keep it minimal"只是提示词约束，不是容量控制。服务端硬限制：

| 字段 | 上限 |
|---|---|
| verified_findings | 20 条 |
| rejected_candidates | 20 条 |
| validity_concerns | 10 条 |
| current_candidates | 10 条 |
| claim 长度 | 200 字符 |
| evidence.observation_preview | 500 字符 |
| 整个状态 JSON | 8 KB |

**超限淘汰策略（不是简单截断）：**

1. 先淘汰 confidence=low 且最旧的
2. 再淘汰无引用价值的（孤立的、被后续结论取代的——按 created_at + 关联度）
3. 最后才截断数组尾部
4. 淘汰动作进日志（可审计）

---

## 7. 并发一致性（审查第 5 点）

### 7.1 场景

```
请求 A 读 version 3 ──┐
请求 B 读 version 3 ──┤→ A 写 version 4 → B 用旧状态写 version 4（覆盖丢状态）
```

### 7.2 方案

- **Redis**：state 存 `string(JSON) + version`，用 **WATCH/MULTI 或 Lua 脚本**做 CAS（`update_state(..., expected_version)` 不匹配则返回冲突）
- **内存**：RLock 覆盖**"读取-合并-写入"整个事务**（现状锁只锁 append_turn 单个方法，不够）
- **按 session 串行**（备选）：同 session 请求进队列，实现简单但牺牲并发
- 冲突处理：调用方（run_agent）收到冲突 → 重新读最新状态 → 重新合并 → 重试（有上限）

### 7.3 Patch 的合并语义

不同字段不能统一采用“最后写入覆盖”：

| Patch 类型 | 合并规则 |
|---|---|
| `add_findings`、`add_validity_concerns`、`reject_candidates` | 追加并去重，适合自动重试 |
| `context_slots` | 按键合并；显式请求值覆盖历史值，推断值不能覆盖显式值 |
| `replace_next_step_plan` | 冲突时重新读取状态并由服务端拒绝旧计划，不能静默覆盖 |
| `supersede_findings` | 必须引用新证据，并校验旧 finding 仍为 active |

每次 patch 都必须带 `expected_version`。CAS 重试只能自动重试追加型 patch；计划替换和 finding 替代发生冲突时，返回明确错误交给 Agent 重新判断。

---

## 8. 越权与隔离（审查第 8 点）

- **key 加租户范围**：`merchant_ops:{tenant_id}:session:{session_id}:state`
- **校验**：
  - `session.owner == current_user`（APIKey 鉴权已存在，需把 key → 用户/租户绑定）
  - `session.merchant_id == request.merchant_id`
- 至少先做 key 隔离 + merchant_id 校验；owner 校验紧随（改动小）

### 8.1 前置条件

当前 `RunRequest` 只有 `session_id` 和业务 `context`，因此租户隔离不能只靠拼接 Redis key 完成。P0 阶段必须先确定：

- API key 如何映射到 `user_id` / `tenant_id`
- `session_id` 的所有者如何持久化和校验
- 用户是否有权访问请求中的 `merchant_id`

在身份映射未完成前，不应把“key 加租户范围”描述为已完成的安全能力。

---

## 9. SessionStore 接口（审查第 7 点）

### 9.1 新协议

```python
class SessionStore(Protocol):
    # 现有
    get_history(session_id) -> list[tuple[str, str]]
    append_turn(session_id, query, answer, *, max_history_turns, ttl_seconds) -> None
    # 新增
    get_state(session_id) -> DiagnosticState | None          # 旧会话返回 None
    update_state(session_id, patch, expected_version, *, ttl_seconds)
        -> StateUpdateResult                                 # {ok, new_version} / 冲突
    get_snapshot(session_id) -> SessionSnapshot              # state + history + version
```

- `DiagnosticState` / `StateUpdateResult` 用 pydantic/dataclass + JSON schema 校验
- **Redis 和内存两个后端都要实现**，行为一致性测试覆盖
- 旧会话无 state → `get_state` 返回 None → run_agent 走初始化分支（兼容）

### 9.2 上下文组装（有效上下文）

```
h_eff = DiagnosticState(z) ⊕ recent_raw_turns(last 4)
```

`_get_history_text()` 改为 `_compose_context()`：状态 JSON + 最近 4 轮原文。

---

## 10. 快速路径处理（审查第 2 点）

现状：单工具问题走 `_should_short_circuit()` 直接返回（agent.py:730），绕过 Agent 主循环，不会更新状态。

### 方案：快速路径也做确定性状态更新

- 执行工具后，**服务端直接写状态**（不依赖 LLM）：
  - 工具输出 → evidence_records（绑定本次 request_id/evidence_id/observation_hash）
  - 只有经过确定性规则确认的输入槽位才直接写入 context_slots
  - 工具 summary 不直接晋升为 verified_findings，避免把普通 observation 误认为诊断结论
  - 请求 context → context_slots
- 同时明确：快速路径适用于"不需要跨轮记忆的单一查询"；涉及多工具/综合诊断的请求走完整 Agent 路径
- 效果：任何请求结束，状态都有基础沉淀（LLM 没调 update_context 也不丢）

---

## 11. 可观测

- 每次 update_state 记日志：`{request_id, session_id, expected_version, new_version, reason, before, after}`
- 证据表本身可查询（"这条 finding 是哪次工具调用来的"）
- 状态演化轨迹可回放

生产日志默认记录版本、事件类型、finding ID、状态 hash 和变更摘要，不直接打印完整业务状态；完整 before/after 仅在脱敏后的调试模式保存。证据记录和诊断状态分别设置 TTL，状态更新重试完成前不得清理证据。

---

## 12. 四阶段落地计划

### P0：身份映射与租户隔离前置（见 §8.1）

改动：
1. 确定 API key 到 `user_id` / `tenant_id` 的映射
2. 持久化并校验 `session_id` 所有者
3. 校验用户是否有权访问请求中的 `merchant_id`
4. 完成后再启用带租户范围的状态 key 和越权测试

验证：不同用户、租户和商家之间的 session 访问隔离测试。

### P1：上下文槽位（先解决"不重复澄清"）

改动：
1. DiagnosticState 最小版：仅 `context_slots` + `version`
2. get_state / update_state（只支持 slots patch）双后端实现
3. 澄清流程重排：请求覆盖 → slots 补全 → 澄清
4. 快速路径写 slots
5. 明确 context_slots 的来源：显式 API context 优先；自然语言提取或模型推断必须标记来源，推断值不能伪装成用户确认值
6. 测试：首轮给上下文、次轮缺上下文不澄清；旧会话兼容

验证：离线可测，**不需要 API key**。

### P2：服务端增量状态（核心）

改动：
1. 六字段完整状态 + patch schema + 合并/去重/限长/淘汰
2. 证据表 + observation_hash 校验（复用 ToolCallTraceCallbackHandler）
3. version + CAS 并发（Redis Lua / 内存全事务锁）
4. update_context 工具注册（接受 patch，返回合并结果）
5. 快速路径确定性更新（evidence + slots，不自动生成 findings）
6. 测试：patch 不覆盖旧 finding / 非法 source 拒绝 / 并发不丢 / 上限淘汰 / 双后端一致

验证：mock LLM（工具调用可离线模拟），证据校验部分不需 key。

### P3：LLM 自主触发 + 服务端兜底

改动：
1. 提示词：触发时机（完成子环节/证据冲突/排除方向/最终结论前）+ few-shot
2. 服务端兜底：工具调用成功后自动记证据；**请求结束时生成状态快照**（LLM 没调用 update_context 时，基础结论不丢）。快照仅作为审计归档，不绕过 evidence 校验直接写入诊断状态；如需将快照内容沉淀进状态，必须从本次请求证据表生成 evidence 引用，并重新走 patch 校验、合并和 CAS 管线
3. 端到端 3 个验收场景（延续追问/证据冲突/排除方向不重查）

验证：真实 API key 端到端 + 状态演化日志人工检查。

---

## 13. 优先测试清单（对应审查建议）

| # | 测试 | 阶段 |
|---|---|---|
| 1 | 首轮提供上下文，次轮缺少上下文时不澄清 | P1 |
| 2 | 快速路径也能更新状态 | P1/P2 |
| 3 | 两次状态 patch 不会覆盖旧 finding | P2 |
| 4 | 非法 source 无法写入 verified finding | P2 |
| 5 | 并发更新不会丢状态 | P2 |
| 6 | Redis 和内存行为一致 | P2 |
| 7 | 旧会话没有 state 时可以正常运行 | P1 |
| 8 | 状态达到上限时按规则淘汰 | P2 |
| 9 | 不同用户不能读取同一 session 的状态 | P0（依赖身份映射） |

---

## 14. 风险与对策（v2 增补）

| 风险 | 对策 |
|---|---|
| LLM 不调 update_context | 服务端兜底：工具证据自动记录 + 请求结束状态快照 |
| patch 证据校验过严导致模型频繁失败 | 错误信息明确（指出哪条缺 evidence），允许重试；兜底路径不依赖模型 |
| 状态/证据表增长 | 状态 8KB 硬限 + 淘汰；证据表随请求生命周期，请求结束归档 |
| 为什么服务端不信任模型 | 这是设计亮点：**"模型提议、服务端仲裁"**——防止幻觉污染状态，这是资深工程观 |

---

## 15. 前置条件与资源

- [x] P1/P2 可离线开发测试（不需要 API key）
- [x] P3 需要 OpenAI 兼容 API key 端到端（`real_multi_round_probe.py` / `e2e_verify.py` 已通过）
- [x] 现有 58 项测试回归基线已更新为 `131 passed`
- [x] 租户/用户与 APIKey 的绑定关系（越权校验前置，`api_keys.json` + `identities.py`）
#（注：内容由AI生成）
