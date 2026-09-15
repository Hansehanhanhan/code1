import type { SessionState } from "@/types";
import { ConstraintBadge } from "@/components/shared/ConstraintBadge";
import { ValueView } from "@/components/shared/ValueView";

function FindingRow({ finding }: { finding: NonNullable<SessionState["verified_findings"]>[number] }) {
  return (
    <div className="finding-row">
      <div className="finding-head">
        <code>{finding.id}</code>
        <span className="confidence-chip">{finding.confidence ?? "n/a"}</span>
      </div>
      <p className="finding-claim">{finding.claim}</p>
      {finding.evidence?.evidence_id ? (
        <code className="finding-evidence">
          {finding.evidence.evidence_id}
          {finding.evidence.request_id ? ` · ${finding.evidence.request_id}` : ""}
        </code>
      ) : null}
    </div>
  );
}

function listOrMuted(items: string[], empty: string) {
  if (!items.length) {
    return <p className="muted">{empty}</p>;
  }
  return (
    <ul className="plain-list">
      {items.map((item, idx) => (
        <li key={`${item}-${idx}`}>{item}</li>
      ))}
    </ul>
  );
}

export function SessionStatePanel({ state, error }: { state: SessionState | null; error: string }) {
  if (error) {
    return (
      <article className="result-card">
        <h2>诊断状态</h2>
        <p className="muted">{error}</p>
        <p className="muted">运行一次流式对话或任务后，此处会展示记忆闭环的实时状态。</p>
      </article>
    );
  }

  if (!state) {
    return (
      <article className="result-card">
        <h2>诊断状态</h2>
        <p className="muted">暂无会话状态。提交一次运行后自动拉取 /sessions/{`{session_id}`}。</p>
      </article>
    );
  }

  const entries = Object.entries(state.constraint_status ?? {});
  const confidence = state.answer_confidence;

  return (
    <article className="result-card">
      <div className="state-head">
        <h2>诊断状态</h2>
        <code className="version-chip">v{state.version}</code>
        <span className="muted-state">{state.history_turns} 轮历史</span>
      </div>

      <h3 className="state-h3">上下文槽位</h3>
      <div className="slot-chips">
        {Object.entries(state.context_slots ?? {}).map(([key, value]) => (
          <span className="slot-chip" key={key}>
            {key}: {String(value ?? "无")}
          </span>
        ))}
      </div>

      <h3 className="state-h3">已验证发现（{state.verified_findings.length}）</h3>
      {state.verified_findings?.length ? (
        <div className="findings-list">
          {state.verified_findings.map((finding) => (
            <FindingRow key={finding.id} finding={finding} />
          ))}
        </div>
      ) : (
        <p className="muted">暂无已验证发现。</p>
      )}

      <h3 className="state-h3">当前候选</h3>
      <div className="detail-body">
        <ValueView value={state.current_candidates} />
      </div>

      <h3 className="state-h3">约束状态</h3>
      {entries.length ? (
        <div className="constraint-grid">
          {entries.map(([key, status]) => (
            <div className="constraint-row" key={key}>
              <span>{key}</span>
              <ConstraintBadge status={status} />
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">暂无约束记录。</p>
      )}

      <h3 className="state-h3">未解决约束</h3>
      {listOrMuted(state.unresolved_constraints ?? [], "暂无未解决约束。")}

      <h3 className="state-h3">有效性关切</h3>
      {listOrMuted(state.validity_concerns ?? [], "暂无有效性关切。")}

      <h3 className="state-h3">已排除候选</h3>
      {listOrMuted(state.rejected_candidates ?? [], "暂无已排除候选。")}

      {confidence !== null && confidence !== undefined ? (
        <>
          <h3 className="state-h3">答案置信度</h3>
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${Math.round(confidence * 100)}%` }} />
          </div>
          <span className="confidence-label">{Math.round(confidence * 100)}%</span>
        </>
      ) : null}

      <h3 className="state-h3">已验证引用（{state.verified_citations?.length ?? 0}）</h3>
      {listOrMuted(state.verified_citations ?? [], "暂无已验证引用。")}

      {state.next_step_plan?.length ? (
        <>
          <h3 className="state-h3">下一步计划</h3>
          {listOrMuted(state.next_step_plan, "暂无下一步计划。")}
        </>
      ) : null}
    </article>
  );
}