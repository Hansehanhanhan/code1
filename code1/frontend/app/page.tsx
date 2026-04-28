"use client";

import { FormEvent, useMemo, useRef, useState } from "react";

type StepRecord = {
  name: string;
  input?: unknown;
  output?: unknown;
  duration_ms?: number;
};

type RunResponse = {
  final_answer: string;
  steps: StepRecord[];
  metrics?: Record<string, unknown>;
};

type JobCreateResponse = {
  job_id: string;
  status: string;
  created_at: number;
  retry_of?: string | null;
};

type JobStatusResponse = {
  job_id: string;
  status: string;
  created_at: number;
  updated_at: number;
  error_message?: string | null;
  response?: RunResponse | null;
};

type JobCancelResponse = {
  job_id: string;
  status: string;
  cancelled: boolean;
  message: string;
};

type JobState = {
  job_id: string;
  status: string;
  created_at: number;
  updated_at?: number;
  error_message?: string;
  retry_of?: string | null;
};

type StreamEvent =
  | {
      type: "agent_action";
      content: {
        loop_index: number;
        thought: string;
        action: string;
        action_input: unknown;
      };
    }
  | {
      type: "tool_observation";
      content: {
        loop_index: number;
        observation: unknown;
        duration_ms: number;
      };
    }
  | {
      type: "stream_metrics";
      content: {
        ttfb_ms?: number;
        event_count?: number;
        event_completeness?: boolean;
        degraded?: boolean;
      };
    }
  | {
      type: "final_response";
      content: RunResponse;
    }
  | {
      type: "degraded_response";
      content: {
        reason: string;
      };
    }
  | {
      type: "error";
      content: string;
    };

const SAMPLE_QUERY = "本周店铺流量明显下滑，请诊断原因并给出可执行建议。";
const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

const STEP_LABELS: Record<string, string> = {
  Planner: "规划器",
  Executor: "执行器",
  Verifier: "校验器",
  Clarification: "澄清问题",
};

const TOOL_LABELS: Record<string, string> = {
  traffic_analyze: "流量分析",
  ads_analyze: "广告分析",
  inventory_check: "库存检查",
  product_diagnose: "商品诊断",
  retrieve_knowledge: "知识检索",
};

const FIELD_LABELS: Record<string, string> = {
  latency_ms: "延迟(毫秒)",
  fallback_used: "是否使用兜底",
  ttfb_ms: "首包延迟(毫秒)",
  event_count: "事件数",
  event_completeness: "事件完整性",
  llm_latency_ms: "LLM耗时(毫秒)",
  tool_latency_ms: "工具耗时(毫秒)",
  loop_count: "循环轮数",
  retrieve_hits: "检索命中数",
  duration_ms: "耗时(毫秒)",
  thought: "思考",
  action: "动作",
  action_input: "动作输入",
  observation: "观察结果",
  result: "最终结果",
  query: "问题",
  context: "上下文",
  session_id: "会话ID",
  missing_context_keys: "缺失字段",
};

const METRIC_PRIORITY_KEYS = [
  "latency_ms",
  "ttfb_ms",
  "loop_count",
  "llm_latency_ms",
  "tool_latency_ms",
  "retrieve_hits",
  "event_count",
  "event_completeness",
  "fallback_used",
];

const TERMINAL_JOB_STATUSES = new Set(["succeeded", "failed", "degraded", "cancelled"]);

type ContextFormData = {
  merchant_id: string;
  time_range: string;
  category: string;
};

const TEMPLATES: Record<string, ContextFormData> = {
  retail: { merchant_id: "demo-001", time_range: "last_7_days", category: "retail" },
  wholesale: { merchant_id: "demo-002", time_range: "last_30_days", category: "wholesale" },
};

function toolLabel(name: string) {
  return TOOL_LABELS[name] ? `${TOOL_LABELS[name]} (${name})` : name;
}

function fieldLabel(key: string) {
  return FIELD_LABELS[key] ?? key;
}

function stripMarkdownBold(text: string) {
  return text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*\*/g, "");
}

function formatFinalAnswer(text: string) {
  let formatted = stripMarkdownBold(text).replace(/\r\n/g, "\n").trim();
  formatted = formatted.replace(/建议采取以下措施[:：]/g, "建议采取以下措施：\n");
  formatted = formatted.replace(/([。；])(?=\d+[\.、])/g, "$1\n");
  formatted = formatted.replace(/([。；])(?=[一二三四五六七八九十]+是)/g, "$1\n");
  formatted = formatted.replace(/(^|[^\n])(核心原因[:：]|行动建议[:：]|风险与复盘[:：]|问题判断[:：])/g, "$1\n$2");
  formatted = formatted.replace(/(^|[^\n])(\d+[\.、])/g, "$1\n$2");
  formatted = formatted.replace(/\n{3,}/g, "\n\n").trim();
  return formatted;
}

function splitAnswerAndEvidence(text: string) {
  const marker = "证据来源：";
  const idx = text.indexOf(marker);
  if (idx === -1) {
    return { answerText: text, evidenceLines: [] as string[] };
  }
  const answerText = text.slice(0, idx).trim();
  const tail = text.slice(idx + marker.length).trim();
  const evidenceLines = tail
    .split("\n")
    .map((line) => line.replace(/^\d+[\.、]\s*/, "").trim())
    .filter((line) => line.length > 0);
  return { answerText, evidenceLines };
}

function renderPrimitiveByKey(key: string, value: unknown) {
  if (key === "action" && typeof value === "string") {
    return <span className="text-sm leading-relaxed">{toolLabel(value)}</span>;
  }
  if (key === "fallback_used" && typeof value === "boolean") {
    return <span className="text-sm leading-relaxed">{value ? "是" : "否"}</span>;
  }
  if ((key === "latency_ms" || key === "duration_ms") && typeof value === "number") {
    return <span className="text-sm leading-relaxed">{value} 毫秒</span>;
  }
  return <span className="text-sm leading-relaxed">{String(value)}</span>;
}

function formatValue(value: unknown) {
  if (value === undefined || value === null) {
    return "无";
  }

  if (Array.isArray(value)) {
    return (
      <ul className="list-disc ml-5 space-y-1.5 mt-2">
        {value.map((item, i) => (
          <li key={i} className="data-list-item">
            {typeof item === "object" ? formatValue(item) : String(item)}
          </li>
        ))}
      </ul>
    );
  }

  if (typeof value === "object") {
    return (
      <div className="data-kv-list">
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} className="data-kv-item">
            <span className="data-kv-key">{fieldLabel(k)}:</span>
            <div className="data-kv-value">{typeof v === "object" ? formatValue(v) : renderPrimitiveByKey(k, v)}</div>
          </div>
        ))}
      </div>
    );
  }

  return <span className="text-sm leading-relaxed">{String(value)}</span>;
}

function formatStepName(name: string) {
  if (name.startsWith("ReAct Loop")) {
    const index = name.replace("ReAct Loop", "").trim();
    return `ReAct 第 ${index} 轮`;
  }
  return STEP_LABELS[name] ?? name;
}

function upsertLoopStep(steps: StepRecord[], loopIndex: number, patch: Partial<StepRecord>): StepRecord[] {
  const name = `ReAct Loop ${loopIndex}`;
  const idx = steps.findIndex((s) => s.name === name);
  if (idx === -1) {
    return [...steps, { name, ...patch }];
  }
  const copy = [...steps];
  copy[idx] = { ...copy[idx], ...patch, name };
  return copy;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function formatMetricValue(key: string, value: unknown) {
  if (value === undefined || value === null) {
    return "未上报";
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (typeof value === "number" && key.endsWith("_ms")) {
    return `${value} ms`;
  }
  return String(value);
}

function buildHeaders(apiKey: string) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey.trim()) {
    headers["X-API-Key"] = apiKey.trim();
  }
  return headers;
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function Home() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_BACKEND_URL;
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [sessionId, setSessionId] = useState("demo-session-001");
  const [apiKey, setApiKey] = useState("");
  const [context, setContext] = useState<ContextFormData>(TEMPLATES.retail);
  const [loading, setLoading] = useState(false);
  const [jobActionLoading, setJobActionLoading] = useState(false);
  const [error, setError] = useState("");
  const [jobNotice, setJobNotice] = useState("");
  const [response, setResponse] = useState<RunResponse | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const pollingTokenRef = useRef(0);

  const endpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/run`, [backendUrl]);
  const streamEndpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/run_stream`, [backendUrl]);
  const jobsEndpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/jobs`, [backendUrl]);

  const payload = useMemo(
    () => ({
      query,
      session_id: sessionId.trim() ? sessionId.trim() : undefined,
      context: {
        merchant_id: context.merchant_id,
        time_range: context.time_range,
        category: context.category,
      },
    }),
    [query, sessionId, context]
  );

  const answerView = useMemo(() => {
    const raw = response?.final_answer?.trim();
    if (!raw) {
      return { answerText: "等待结果...", evidenceLines: [] as string[] };
    }
    return splitAnswerAndEvidence(formatFinalAnswer(raw));
  }, [response]);

  const canCancelJob = Boolean(job && (job.status === "queued" || job.status === "running" || job.status === "cancel_requested"));
  const canRetryJob = Boolean(job && TERMINAL_JOB_STATUSES.has(job.status));

  function applyTemplate(templateKey: string) {
    setContext({ ...TEMPLATES[templateKey] });
  }

  function stopPolling() {
    pollingTokenRef.current += 1;
  }

  async function runSyncFallback(runPayload: Record<string, unknown>) {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: buildHeaders(apiKey),
      body: JSON.stringify(runPayload),
    });

    if (!res.ok) {
      throw new Error(`请求失败：${res.status} ${res.statusText}`);
    }

    const data = (await res.json()) as RunResponse;
    setResponse(data);
  }

  async function pollJobUntilTerminal(jobId: string, token: number) {
    while (token === pollingTokenRef.current) {
      const res = await fetch(`${jobsEndpoint}/${jobId}`, {
        method: "GET",
        headers: apiKey.trim() ? { "X-API-Key": apiKey.trim() } : undefined,
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(`任务查询失败：${res.status} ${res.statusText}`);
      }
      const data = (await res.json()) as JobStatusResponse;
      if (token !== pollingTokenRef.current) {
        return;
      }
      setJob({
        job_id: data.job_id,
        status: data.status,
        created_at: data.created_at,
        updated_at: data.updated_at,
        error_message: data.error_message ?? undefined,
      });

      if (data.response) {
        setResponse(data.response);
      }

      if (TERMINAL_JOB_STATUSES.has(data.status)) {
        if (data.status === "cancelled") {
          setJobNotice("任务已取消。");
        }
        return;
      }
      await delay(900);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    stopPolling();
    setLoading(true);
    setError("");
    setJobNotice("");
    setResponse({ final_answer: "", steps: [], metrics: {} });

    try {
      const res = await fetch(streamEndpoint, {
        method: "POST",
        headers: buildHeaders(apiKey),
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`请求失败：${res.status} ${res.statusText}`);
      }

      if (!res.body) {
        await runSyncFallback(payload);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let liveSteps: StepRecord[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        while (true) {
          const splitIndex = buffer.indexOf("\n\n");
          if (splitIndex === -1) {
            break;
          }
          const eventBlock = buffer.slice(0, splitIndex);
          buffer = buffer.slice(splitIndex + 2);

          const dataLine = eventBlock.split("\n").find((line) => line.startsWith("data:"));
          if (!dataLine) {
            continue;
          }

          const raw = dataLine.slice(5).trim();
          if (!raw) {
            continue;
          }

          const parsed = JSON.parse(raw) as StreamEvent;

          if (parsed.type === "agent_action") {
            liveSteps = upsertLoopStep(liveSteps, parsed.content.loop_index, {
              input: {
                thought: parsed.content.thought,
                action: parsed.content.action,
                action_input: parsed.content.action_input,
              },
            });
            setResponse((prev) => ({
              final_answer: prev?.final_answer ?? "",
              steps: [...liveSteps],
              metrics: prev?.metrics ?? {},
            }));
            continue;
          }

          if (parsed.type === "tool_observation") {
            const step = liveSteps.find((s) => s.name === `ReAct Loop ${parsed.content.loop_index}`);
            liveSteps = upsertLoopStep(liveSteps, parsed.content.loop_index, {
              input: step?.input,
              output: { observation: parsed.content.observation },
              duration_ms: parsed.content.duration_ms,
            });
            setResponse((prev) => ({
              final_answer: prev?.final_answer ?? "",
              steps: [...liveSteps],
              metrics: prev?.metrics ?? {},
            }));
            continue;
          }

          if (parsed.type === "final_response") {
            setResponse((prev) => ({
              ...parsed.content,
              metrics: {
                ...(parsed.content.metrics ?? {}),
                ...(prev?.metrics ?? {}),
              },
            }));
            continue;
          }

          if (parsed.type === "stream_metrics") {
            setResponse((prev) => ({
              final_answer: prev?.final_answer ?? "",
              steps: prev?.steps ?? [...liveSteps],
              metrics: {
                ...(prev?.metrics ?? {}),
                ...(parsed.content ?? {}),
              },
            }));
            continue;
          }

          if (parsed.type === "error") {
            throw new Error(parsed.content || "流式请求失败。");
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败，请稍后重试。");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateJob() {
    stopPolling();
    setError("");
    setJobNotice("");
    setJobActionLoading(true);
    try {
      const res = await fetch(jobsEndpoint, {
        method: "POST",
        headers: buildHeaders(apiKey),
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error(`提交任务失败：${res.status} ${res.statusText}`);
      }
      const created = (await res.json()) as JobCreateResponse;
      setJob({
        job_id: created.job_id,
        status: created.status,
        created_at: created.created_at,
        retry_of: created.retry_of ?? null,
      });
      const token = pollingTokenRef.current;
      void pollJobUntilTerminal(created.job_id, token).catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "任务轮询失败");
      });
      setJobNotice(`任务已提交：${created.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "任务提交失败，请稍后重试。");
    } finally {
      setJobActionLoading(false);
    }
  }

  async function handleCancelJob() {
    if (!job) {
      return;
    }
    setError("");
    setJobNotice("");
    setJobActionLoading(true);
    try {
      const res = await fetch(`${jobsEndpoint}/${job.job_id}/cancel`, {
        method: "POST",
        headers: apiKey.trim() ? { "X-API-Key": apiKey.trim() } : undefined,
      });
      if (!res.ok) {
        throw new Error(`取消失败：${res.status} ${res.statusText}`);
      }
      const data = (await res.json()) as JobCancelResponse;
      setJob((prev) =>
        prev
          ? {
              ...prev,
              status: data.status,
            }
          : null
      );
      setJobNotice(data.message);
      if (data.status === "cancel_requested") {
        const token = pollingTokenRef.current;
        void pollJobUntilTerminal(job.job_id, token).catch((err: unknown) => {
          setError(err instanceof Error ? err.message : "任务轮询失败");
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "取消任务失败，请稍后重试。");
    } finally {
      setJobActionLoading(false);
    }
  }

  async function handleRetryJob() {
    if (!job) {
      return;
    }
    stopPolling();
    setError("");
    setJobNotice("");
    setJobActionLoading(true);
    try {
      const res = await fetch(`${jobsEndpoint}/${job.job_id}/retry`, {
        method: "POST",
        headers: apiKey.trim() ? { "X-API-Key": apiKey.trim() } : undefined,
      });
      if (!res.ok) {
        throw new Error(`重试失败：${res.status} ${res.statusText}`);
      }
      const created = (await res.json()) as JobCreateResponse;
      setJob({
        job_id: created.job_id,
        status: created.status,
        created_at: created.created_at,
        retry_of: created.retry_of ?? null,
      });
      const token = pollingTokenRef.current;
      void pollJobUntilTerminal(created.job_id, token).catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "任务轮询失败");
      });
      setJobNotice(`已创建重试任务：${created.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "重试任务失败，请稍后重试。");
    } finally {
      setJobActionLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">ReAct 流式诊断 + 异步任务</p>
          <h1>商家运营 Copilot</h1>
          <p className="hero-text">
            支持 <code>/run_stream</code> 实时决策轨迹，以及 <code>/jobs</code> 的提交、取消、重试。
          </p>
        </div>

        <form className="workspace" onSubmit={handleSubmit}>
          <label className="field">
            <span>问题描述</span>
            <textarea
              rows={4}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="请描述商家当前遇到的问题"
            />
          </label>

          <label className="field">
            <span>会话 ID（用于记忆）</span>
            <input
              type="text"
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value)}
              placeholder="demo-session-001"
            />
          </label>

          <label className="field">
            <span>API Key（可选）</span>
            <input
              type="text"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="APP_AUTH_ENABLED=true 时必填"
            />
          </label>

          <div className="context-section">
            <div className="template-bar">
              <span className="template-label">快速模板：</span>
              <button type="button" className="template-btn" onClick={() => applyTemplate("retail")}>
                零售商家
              </button>
              <button type="button" className="template-btn" onClick={() => applyTemplate("wholesale")}>
                批发商家
              </button>
            </div>

            <div className="context-fields">
              <label className="field">
                <span>商家 ID</span>
                <input
                  type="text"
                  value={context.merchant_id}
                  onChange={(e) => setContext({ ...context, merchant_id: e.target.value })}
                  placeholder="demo-001"
                />
              </label>

              <label className="field">
                <span>时间范围</span>
                <select value={context.time_range} onChange={(e) => setContext({ ...context, time_range: e.target.value })}>
                  <option value="last_7_days">最近 7 天</option>
                  <option value="last_30_days">最近 30 天</option>
                  <option value="last_90_days">最近 90 天</option>
                </select>
              </label>

              <label className="field">
                <span>店铺类型</span>
                <select value={context.category} onChange={(e) => setContext({ ...context, category: e.target.value })}>
                  <option value="retail">零售</option>
                  <option value="wholesale">批发</option>
                  <option value="franchise">连锁</option>
                </select>
              </label>
            </div>
          </div>

          <div className="actions">
            <button type="submit" disabled={loading || jobActionLoading}>
              {loading ? "流式运行中..." : "流式运行 Agent"}
            </button>
            <button
              type="button"
              className="secondary-btn"
              disabled={loading || jobActionLoading}
              onClick={handleCreateJob}
            >
              {jobActionLoading ? "处理中..." : "提交异步任务"}
            </button>
            <span className="endpoint">POST {streamEndpoint}</span>
          </div>
        </form>
      </section>

      <section className="results-grid">
        <article className="result-card">
          <h2>最终建议</h2>
          {error ? <p className="error">{error}</p> : null}
          {jobNotice ? <p className="notice">{jobNotice}</p> : null}
          <pre>{answerView.answerText}</pre>
          {answerView.evidenceLines.length ? (
            <div className="evidence-panel">
              <h3>证据来源</h3>
              <ul>
                {answerView.evidenceLines.map((line, idx) => (
                  <li key={`${line}-${idx}`}>{line}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </article>

        <article className="result-card">
          <h2>执行步骤</h2>
          <div className="step-list">
            {response?.steps?.length ? (
              response.steps.map((step, index) => (
                <div className="step-item" key={`${step.name}-${index}`}>
                  <div className="step-header">
                    <strong>{formatStepName(step.name)}</strong>
                    {typeof step.duration_ms === "number" ? <span>{step.duration_ms} 毫秒</span> : null}
                  </div>
                  {step.name.startsWith("ReAct Loop") ? (
                    (() => {
                      const input = asRecord(step.input) ?? {};
                      const output = asRecord(step.output) ?? {};
                      const thought = typeof input.thought === "string" ? input.thought : "";
                      const action = typeof input.action === "string" ? input.action : "";
                      const actionInput = input.action_input;
                      const observation = output.observation !== undefined ? output.observation : step.output ?? "暂无";
                      return (
                        <div className="step-main">
                          <div className="step-block">
                            <span>思考</span>
                            <p className="thought-text">{thought || "（无）"}</p>
                          </div>
                          <div className="step-block">
                            <span>动作</span>
                            <div className="action-chip">{action ? toolLabel(action) : "（无）"}</div>
                          </div>
                          <details className="detail-card">
                            <summary>动作输入</summary>
                            <div className="detail-body">{formatValue(actionInput)}</div>
                          </details>
                          <details className="detail-card" open>
                            <summary>观察结果</summary>
                            <div className="detail-body">{formatValue(observation)}</div>
                          </details>
                        </div>
                      );
                    })()
                  ) : (
                    <>
                      <div className="step-block">
                        <span>输入</span>
                        <div className="detail-body">{formatValue(step.input)}</div>
                      </div>
                      <div className="step-block">
                        <span>输出</span>
                        <div className="detail-body">{formatValue(step.output)}</div>
                      </div>
                    </>
                  )}
                </div>
              ))
            ) : (
              <p className="muted">暂无步骤数据，流式执行后会显示每一轮 ReAct 轨迹。</p>
            )}
          </div>
        </article>

        <article className="result-card">
          <h2>任务与指标</h2>
          <div className="job-card">
            <div className="job-row">
              <span>任务 ID</span>
              <code>{job?.job_id ?? "暂无"}</code>
            </div>
            <div className="job-row">
              <span>状态</span>
              <strong>{job?.status ?? "未提交"}</strong>
            </div>
            {job?.retry_of ? (
              <div className="job-row">
                <span>重试来源</span>
                <code>{job.retry_of}</code>
              </div>
            ) : null}
            {job?.error_message ? (
              <div className="job-row">
                <span>错误</span>
                <span>{job.error_message}</span>
              </div>
            ) : null}
            <div className="job-actions">
              <button type="button" className="danger-btn" disabled={!canCancelJob || jobActionLoading} onClick={handleCancelJob}>
                取消任务
              </button>
              <button type="button" className="secondary-btn" disabled={!canRetryJob || jobActionLoading} onClick={handleRetryJob}>
                重试任务
              </button>
            </div>
          </div>

          {response?.metrics ? (
            <>
              <div className="metrics-grid">
                {METRIC_PRIORITY_KEYS.filter((key) => key in (response.metrics ?? {})).map((key) => (
                  <div className="metric-card" key={key}>
                    <div className="metric-key">{fieldLabel(key)}</div>
                    <div className="metric-value">{formatMetricValue(key, response.metrics?.[key])}</div>
                  </div>
                ))}
              </div>
              <details className="detail-card metrics-detail">
                <summary>查看完整指标</summary>
                <div className="detail-body">{formatValue(response.metrics)}</div>
              </details>
            </>
          ) : (
            <p className="muted">等待指标...</p>
          )}
        </article>
      </section>
    </main>
  );
}
