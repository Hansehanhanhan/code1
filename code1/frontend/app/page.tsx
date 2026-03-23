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

const SAMPLE_QUERY = "本周店铺流量明显下滑，请诊断原因并给出可执行建议。";
const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

const STEP_LABELS: Record<string, string> = {
  Planner: "规划器",
  Executor: "执行器",
  Verifier: "校验器"
};

function formatValue(value: unknown) {
  if (value === undefined || value === null) {
    return "无";
  }

  // 渲染建议/列表
  if (Array.isArray(value)) {
    return (
      <ul className="list-disc ml-5 space-y-1.5 mt-2">
        {value.map((item, i) => (
          <li key={i} className="text-sm text-gray-800 leading-relaxed shadow-sm p-2 bg-white/40 rounded-lg border border-blue-50/50">
            {typeof item === "object" ? JSON.stringify(item) : String(item)}
          </li>
        ))}
      </ul>
    );
  }

  // 渲染对象（如工具结果卡片）
  if (typeof value === "object") {
    return (
      <div className="flex flex-col gap-3 mt-2">
        {Object.entries(value as Record<string, any>).map(([k, v]) => {
          // 核心：处理工具执行结果的具体卡片
          const isToolResult = v && typeof v === 'object' && v.tool;
          
          if (isToolResult) {
            return (
              <div key={k} className="p-3 bg-white/80 rounded-xl border border-gray-200/50 shadow-sm backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-[#234f8d] text-xs px-2 py-0.5 bg-blue-50 rounded-md ring-1 ring-blue-100">{v.tool}</span>
                  <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${v.status === 'ok' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    {v.status}
                  </span>
                </div>
                <p className="text-sm font-semibold text-[#12171e] mb-2">{v.summary}</p>
                {v.data && (
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 p-2 bg-gray-50/50 rounded-lg text-[11px] border border-gray-100">
                    {Object.entries(v.data).map(([dk, dv]) => (
                      <div key={dk} className="flex justify-between border-b border-gray-100/50 pb-0.5">
                        <span className="text-[#555f6a] font-medium">{dk}</span>
                        <span className="font-mono text-[#12171e]">{String(dv)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {v.recommendations && (
                  <div className="mt-2 text-[11px] text-[#555f6a] italic">
                    💡 {v.recommendations.join(" | ")}
                  </div>
                )}
              </div>
            );
          }

          // 常规字段字段渲染
          const isHighlight = ["scenario", "risk_level", "source", "objective"].includes(k);
          return (
            <div key={k} className="flex items-start gap-2 text-sm group">
              <span className="text-[#555f6a] font-semibold min-w-[85px] pt-0.5 select-none">{k}:</span>
              <div className={`flex-1 break-words transition-colors ${isHighlight ? 'font-bold text-[#234f8d]' : 'text-[#12171e]'}`}>
                {typeof v === "object" ? formatValue(v) : String(v)}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return <span className="text-sm leading-relaxed">{String(value)}</span>;
}

function formatStepName(name: string) {
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

export default function Home() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_BACKEND_URL;
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [context, setContext] = useState<ContextFormData>(TEMPLATES.retail);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<RunResponse | null>(null);

  const endpoint = useMemo(() => `${backendUrl.replace(/\/$/, "")}/run`, [backendUrl]);

  function applyTemplate(templateKey: string) {
    setContext({ ...TEMPLATES[templateKey] });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResponse(null);

    try {
      // 前端仅负责组装请求和展示结果，Agent 逻辑在后端完成。
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query,
          context: context.merchant_id ? {
            merchant_id: context.merchant_id,
            time_range: context.time_range,
            category: context.category,
          } : undefined
        })
      });

      if (!res.ok) {
        throw new Error(`请求失败：${res.status} ${res.statusText}`);
      }

      const data = (await res.json()) as RunResponse;
      setResponse(data);
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
          <p className="eyebrow">Day 1 骨架</p>
          <h1>商家运营 Copilot</h1>
          <p className="hero-text">
            输入运营问题后会调用后端 <code>/run</code>，页面将展示最终建议、执行轨迹和运行指标。
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

          <div className="context-section">
            <div className="template-bar">
              <span className="template-label">快速模板：</span>
              <button type="button" className="template-btn" onClick={() => applyTemplate("retail")}>零售商家</button>
              <button type="button" className="template-btn" onClick={() => applyTemplate("wholesale")}>批发商家</button>
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
            <span className="endpoint">POST {endpoint}</span>
          </div>
        </form>
      </section>

      <section className="results-grid">
        <article className="result-card">
          <h2>最终建议</h2>
          {error ? <p className="error">{error}</p> : null}
          <pre>{response ? response.final_answer : "等待结果中..."}</pre>
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
              <p className="muted">暂无步骤数据，后端返回后会显示规划器/执行器/校验器轨迹。</p>
            )}
          </div>
        </article>

        <article className="result-card">
          <h2>运行指标</h2>
          <pre>{response?.metrics ? formatValue(response.metrics) : "等待指标中..."}</pre>
        </article>
      </section>
    </main>
  );
}
