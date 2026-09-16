import type { JobState, StreamEvent } from "@/types";
import { JOB_STATUS_LABELS, fieldLabel } from "@/lib/labels";
import { formatMetricValue } from "@/lib/format";
import { KeyStepBadge } from "@/components/shared/KeyStepBadge";
import { ValueView } from "@/components/shared/ValueView";

const METRIC_PRIORITY_KEYS = [
  "latency_ms",
  "ttfb_ms",
  "tool_loop_count",
  "llm_latency_ms",
  "tool_latency_ms",
  "retrieve_hits",
  "event_count",
  "event_completeness",
  "fallback_used",
];

function eventSummary(evt: StreamEvent) {
  switch (evt.type) {
    case "agent_action":
      return `工具调用 第 ${evt.content.tool_loop_index} 轮`;
    case "tool_observation":
      return `观察 第 ${evt.content.tool_loop_index} 轮`;
    case "llm_observation":
      return `LLM 思考 ${evt.content.duration_ms} 毫秒`;
    case "key_step":
      return null;
    case "degraded_response":
      return `服务降级：${evt.content.reason}`;
    default:
      return String(evt.content);
  }
}

export function JobPanel({
  job,
  jobEvents,
  notice,
  error,
  actionLoading,
  metrics,
  onCancel,
  onRetry,
}: {
  job: JobState | null;
  jobEvents: StreamEvent[];
  notice: string;
  error: string;
  actionLoading: boolean;
  metrics?: Record<string, unknown>;
  onCancel: () => void;
  onRetry: () => void;
}) {
  const status = job?.status ?? "未提交";
  const statusLabel = JOB_STATUS_LABELS[status] ?? status;
  const canCancel = Boolean(job && ["queued", "running", "cancel_requested"].includes(job.status));
  const canRetry = Boolean(job && ["succeeded", "failed", "degraded", "cancelled"].includes(job.status));

  return (
    <article className="result-card">
      <h2>任务与指标</h2>

      {error ? <p className="error">{error}</p> : null}
      {notice ? <p className="notice">{notice}</p> : null}

      <div className="job-card">
        <div className="job-row">
          <span>任务 ID</span>
          <code>{job?.job_id ?? "暂无"}</code>
        </div>
        <div className="job-row">
          <span>状态</span>
          <strong>{job ? statusLabel : "未提交"}</strong>
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
          <button type="button" className="danger-btn" disabled={!canCancel || actionLoading} onClick={onCancel}>
            取消任务
          </button>
          <button type="button" className="secondary-btn" disabled={!canRetry || actionLoading} onClick={onRetry}>
            重试任务
          </button>
        </div>
      </div>

      <h3 className="state-h3">实时事件（SSE 回放）</h3>
      <div className="job-event-list">
        {jobEvents.length ? (
          jobEvents.map((evt, idx) => {
            if (evt.type === "key_step") {
              return <KeyStepBadge key={`${idx}`} event={evt.content} />;
            }
            const summary = eventSummary(evt);
            if (summary === null) {
              return null;
            }
            if (evt.type === "tool_observation") {
              return (
                <details className="detail-card job-event" key={idx}>
                  <summary>{summary}</summary>
                  <div className="detail-body">
                    <ValueView value={evt.content.observation} />
                  </div>
                </details>
              );
            }
            return (
              <div className="job-event" key={idx}>
                <span className="job-event-label">{evt.type}</span>
                <span className="job-event-summary">{summary}</span>
              </div>
            );
          })
        ) : (
          <p className="muted">暂无事件流，提交任务后通过 /jobs/{`{id}`}/stream 实时展示。</p>
        )}
      </div>

      {metrics && Object.keys(metrics).length ? (
        <>
          <div className="metrics-grid">
            {METRIC_PRIORITY_KEYS.filter((key) => key in metrics).map((key) => (
              <div className="metric-card" key={key}>
                <div className="metric-key">{fieldLabel(key)}</div>
                <div className="metric-value">{formatMetricValue(key, metrics[key])}</div>
              </div>
            ))}
          </div>
          <details className="detail-card metrics-detail">
            <summary>查看完整指标</summary>
            <div className="detail-body">
              <ValueView value={metrics} />
            </div>
          </details>
        </>
      ) : (
        <p className="muted">等待指标...</p>
      )}
    </article>
  );
}