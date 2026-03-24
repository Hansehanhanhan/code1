"use client";

import { FormEvent, useMemo, useState } from "react";

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
      type: "final_response";
      content: RunResponse;
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
};

const TOOL_LABELS: Record<string, string> = {
  traffic_analyze: "流量分析",
  ads_analyze: "广告分析",
  inventory_check: "库存检查",
  product_diagnose: "商品诊断",
};

const FIELD_LABELS: Record<string, string> = {
  latency_ms: "延迟(毫秒)",
  fallback_used: "是否使用兜底",
  duration_ms: "耗时(毫秒)",
  thought: "思考",
  action: "动作",
  action_input: "动作输入",
  observation: "观察结果",
  result: "最终结果",
  query: "问题",
  context: "上下文",
  session_id: "会话ID",
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
          <li
            key={i}
            className="text-sm text-gray-800 leading-relaxed shadow-sm p-2 bg-white/40 rounded-lg border border-blue-50/50"
          >
            {typeof item === "object" ? JSON.stringify(item) : String(item)}
          </li>
        ))}
      </ul>
    );
  }

  if (typeof value === "object") {
    return (
      <div className="flex flex-col gap-3 mt-2">
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} className="flex items-start gap-2 text-sm group">
            <span className="text-[#555f6a] font-semibold min-w-[95px] pt-0.5 select-none">
              {fieldLabel(k)}:
            </span>
            <div className="flex-1 break-words transition-colors text-[#12171e]">
              {typeof v === "object" ? formatValue(v) : renderPrimitiveByKey(k, v)}
            </div>
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

type ContextFormData = {
  merchant_id: string;
  time_range: string;
  category: string;
};

const TEMPLATES: Record<string, ContextFormData> = {
  retail: { merchant_id: "demo-001", time_range: "last_7_days", category: "retail" },
  wholesale: { merchant_id: "demo-002", time_range: "last_30_days", category: "wholesale" },
};

function upsertLoopStep(
  steps: StepRecord[],
  loopIndex: number,
  patch: Partial<StepRecord>
): StepRecord[] {
  const name = `ReAct Loop ${loopIndex}`;
  const idx = steps.findIndex((s) => s.name === name);
  if (idx === -1) {
    return [...steps, { name, ...patch }];
  }
  const copy = [...steps];
  copy[idx] = { ...copy[idx], ...patch, name };
  return copy;
}

export default function Home() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_BACKEND_URL;
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [sessionId, setSessionId] = useState("demo-session-001");
  const [context, setContext] = useState<ContextFormData>(TEMPLATES.retail);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<RunResponse | null>(null);

  const endpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/run`, [backendUrl]);
  const streamEndpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/run_stream`, [backendUrl]);

  function applyTemplate(templateKey: string) {
    setContext({ ...TEMPLATES[templateKey] });
  }

  async function runSyncFallback(payload: Record<string, unknown>) {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`请求失败：${res.status} ${res.statusText}`);
    }

    const data = (await res.json()) as RunResponse;
    setResponse(data);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResponse({ final_answer: "", steps: [], metrics: {} });

    const payload = {
      query,
      session_id: sessionId.trim() ? sessionId.trim() : undefined,
      context: context.merchant_id
        ? {
            merchant_id: context.merchant_id,
            time_range: context.time_range,
            category: context.category,
          }
        : undefined,
    };

    try {
      const res = await fetch(streamEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
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

          const dataLine = eventBlock
            .split("\n")
            .find((line) => line.startsWith("data:"));
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
            setResponse(parsed.content);
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

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">ReAct 流式诊断</p>
          <h1>商家运营 Copilot</h1>
          <p className="hero-text">
            页面会调用后端 <code>/run_stream</code>，实时展示每一轮 ReAct 抉择和工具调用结果。
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
                <select
                  value={context.time_range}
                  onChange={(e) => setContext({ ...context, time_range: e.target.value })}
                >
                  <option value="last_7_days">最近 7 天</option>
                  <option value="last_30_days">最近 30 天</option>
                  <option value="last_90_days">最近 90 天</option>
                </select>
              </label>

              <label className="field">
                <span>店铺类型</span>
                <select
                  value={context.category}
                  onChange={(e) => setContext({ ...context, category: e.target.value })}
                >
                  <option value="retail">零售</option>
                  <option value="wholesale">批发</option>
                  <option value="franchise">连锁</option>
                </select>
              </label>
            </div>
          </div>

          <div className="actions">
            <button type="submit" disabled={loading}>
              {loading ? "运行中..." : "运行 Agent"}
            </button>
            <span className="endpoint">POST {streamEndpoint}</span>
          </div>
        </form>
      </section>

      <section className="results-grid">
        <article className="result-card">
          <h2>最终建议</h2>
          {error ? <p className="error">{error}</p> : null}
          <pre>{response ? formatFinalAnswer(response.final_answer || "流式生成中...") : "等待结果..."}</pre>
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
                  <div className="step-block">
                    <span>输入</span>
                    <pre>{formatValue(step.input)}</pre>
                  </div>
                  <div className="step-block">
                    <span>输出</span>
                    <pre>{formatValue(step.output)}</pre>
                  </div>
                </div>
              ))
            ) : (
              <p className="muted">暂无步骤数据，流式执行后会显示每一轮 ReAct 轨迹。</p>
            )}
          </div>
        </article>

        <article className="result-card">
          <h2>运行指标</h2>
          <pre>{response?.metrics ? formatValue(response.metrics) : "等待指标..."}</pre>
        </article>
      </section>
    </main>
  );
}
