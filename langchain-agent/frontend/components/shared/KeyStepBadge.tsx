import type { KeyStepEvent } from "@/types";
import { KEY_STEP_KIND_LABELS } from "@/lib/labels";
import { toolLabel } from "@/lib/labels";

function kindTone(kind: KeyStepEvent["kind"]) {
  switch (kind) {
    case "context_update":
      return "tone-green";
    case "direction_repair":
      return "tone-amber";
    case "context_update_rejected":
      return "tone-red";
    default:
      return "tone-blue";
  }
}

export function KeyStepBadge({ event }: { event: KeyStepEvent }) {
  const title = KEY_STEP_KIND_LABELS[event.kind] ?? event.kind;
  return (
    <div className={`key-step-badge ${kindTone(event.kind)}`}>
      <span className="key-step-kind">{title}</span>
      {event.kind === "first_evidence" ? (
        <code className="key-step-detail">
          {event.action ? `${toolLabel(event.action)} · ` : ""}
          {event.evidence_id ?? ""}
        </code>
      ) : null}
      {event.kind === "context_update" ? (
        <code className="key-step-detail">
          记忆已更新至 v{event.new_version}
          {event.reason ? ` · ${event.reason}` : ""}
        </code>
      ) : null}
      {event.kind === "direction_repair" ? (
        <code className="key-step-detail">
          {event.candidate_previews?.join("；") || "已排除候选"}
        </code>
      ) : null}
      {event.kind === "context_update_rejected" ? (
        <code className="key-step-detail">
          {event.error_type ?? ""}
          {event.error ? `：${event.error}` : ""}
        </code>
      ) : null}
    </div>
  );
}