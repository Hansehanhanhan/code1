import type { FormEvent } from "react";
import type { ContextFormData } from "@/types";

const TEMPLATES: Record<string, ContextFormData> = {
  retail: { merchant_id: "demo-001", time_range: "last_7_days", category: "retail" },
  wholesale: { merchant_id: "demo-002", time_range: "last_30_days", category: "wholesale" },
};

export function InputPanel({
  query,
  onQueryChange,
  sessionId,
  onSessionIdChange,
  apiKey,
  onApiKeyChange,
  context,
  onContextChange,
  loading,
  jobActionLoading,
  onSubmit,
  onCreateJob,
  streamEndpoint,
}: {
  query: string;
  onQueryChange: (value: string) => void;
  sessionId: string;
  onSessionIdChange: (value: string) => void;
  apiKey: string;
  onApiKeyChange: (value: string) => void;
  context: ContextFormData;
  onContextChange: (value: ContextFormData) => void;
  loading: boolean;
  jobActionLoading: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onCreateJob: () => void;
  streamEndpoint: string;
}) {
  return (
    <section className="hero-card">
      <div className="hero-copy">
        <p className="eyebrow">工具调用流式诊断 + 记忆闭环</p>
        <h1>商家运营 Copilot</h1>
        <p className="hero-text">
          支持 <code>/run_stream</code> 实时决策轨迹（含 <code>key_step</code> 记忆里程碑），以及{" "}
          <code>/jobs</code> 的提交、取消、重试与 SSE 实时订阅。
        </p>
      </div>

      <form className="workspace" onSubmit={onSubmit}>
        <label className="field">
          <span>问题描述</span>
          <textarea
            rows={4}
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="请描述商家当前遇到的问题"
          />
        </label>

        <label className="field">
          <span>会话 ID（用于记忆）</span>
          <input
            type="text"
            value={sessionId}
            onChange={(event) => onSessionIdChange(event.target.value)}
            placeholder="demo-session-001"
          />
        </label>

        <label className="field">
          <span>API Key（可选）</span>
          <input
            type="password"
            value={apiKey}
            onChange={(event) => onApiKeyChange(event.target.value)}
            placeholder="APP_AUTH_ENABLED=true 时必填"
            autoComplete="off"
          />
        </label>

        <div className="context-section">
          <div className="template-bar">
            <span className="template-label">快速模板：</span>
            <button
              type="button"
              className="template-btn"
              onClick={() => onContextChange({ ...TEMPLATES.retail })}
            >
              零售商家
            </button>
            <button
              type="button"
              className="template-btn"
              onClick={() => onContextChange({ ...TEMPLATES.wholesale })}
            >
              批发商家
            </button>
          </div>

          <div className="context-fields">
            <label className="field">
              <span>商家 ID</span>
              <input
                type="text"
                value={context.merchant_id}
                onChange={(e) => onContextChange({ ...context, merchant_id: e.target.value })}
                placeholder="demo-001"
              />
            </label>

            <label className="field">
              <span>时间范围</span>
              <select
                value={context.time_range}
                onChange={(e) => onContextChange({ ...context, time_range: e.target.value })}
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
                onChange={(e) => onContextChange({ ...context, category: e.target.value })}
              >
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
            onClick={onCreateJob}
          >
            {jobActionLoading ? "处理中..." : "提交异步任务"}
          </button>
          <span className="endpoint">POST {streamEndpoint}</span>
        </div>
      </form>
    </section>
  );
}