import { CONSTRAINT_STATUS_LABELS } from "@/lib/labels";

function constraintTone(status: string) {
  if (status === "satisfied") {
    return "cs-satisfied";
  }
  if (status === "unsatisfied") {
    return "cs-unsatisfied";
  }
  return "cs-unchecked";
}

export function ConstraintBadge({ status }: { status: string }) {
  return (
    <span className={`constraint-badge ${constraintTone(status)}`}>
      {CONSTRAINT_STATUS_LABELS[status] ?? status}
    </span>
  );
}