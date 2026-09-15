import type { KeyStepEvent, StepRecord } from "@/types";
import { asRecord, formatStepName } from "@/lib/format";
import { toolLabel } from "@/lib/labels";
import { KeyStepBadge } from "@/components/shared/KeyStepBadge";
import { ValueView } from "@/components/shared/ValueView";

export function StepTrace({
  steps,
  milestones,
}: {
  steps: StepRecord[];
  milestones: KeyStepEvent[];
}) {
  return (
    <article className="result-card">
      <h2>执行轨迹</h2>

      {milestones.length ? (
        <div className="milestone-strip">
          {milestones.map((milestone, idx) => (
            <KeyStepBadge key={`${milestone.kind}-${idx}`} event={milestone} />
          ))}
        </div>
      ) : null}

      <div className="step-list">
        {steps.length ? (
          steps.map((step, index) => (
            <div className="step-item" key={`${step.name}-${index}`}>
              <div className="step-header">
                <strong>{formatStepName(step.name)}</strong>
                <div className="step-ms">
                  {typeof step.duration_ms === "number" ? <span>{step.duration_ms} 毫秒</span> : null}
                  {typeof step.llm_duration_ms === "number" ? (
                    <span>思考 {step.llm_duration_ms} 毫秒</span>
                  ) : null}
                </div>
              </div>

              {(() => {
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
                      <div className="detail-body">
                        <ValueView value={actionInput} />
                      </div>
                    </details>
                    <details className="detail-card" open>
                      <summary>观察结果</summary>
                      <div className="detail-body">
                        <ValueView value={observation} />
                      </div>
                    </details>
                  </div>
                );
              })()}
            </div>
          ))
        ) : (
          <p className="muted">
            {milestones.length
              ? "里程碑已出现，等待完整轨迹数据..."
              : "暂无轨迹数据，流式执行后会显示每一步工具调用轨迹。"}
          </p>
        )}
      </div>
    </article>
  );
}