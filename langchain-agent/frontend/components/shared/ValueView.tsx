import type { ReactNode } from "react";
import { fieldLabel, toolLabel } from "@/lib/labels";

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

export function ValueView({ value }: { value: unknown }): ReactNode {
  if (value === undefined || value === null) {
    return "无";
  }

  if (Array.isArray(value)) {
    return (
      <ul className="list-disc ml-5 space-y-1.5 mt-2">
        {value.map((item, i) => (
          <li key={i} className="data-list-item">
            {typeof item === "object" ? <ValueView value={item} /> : String(item)}
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
            <div className="data-kv-value">{typeof v === "object" ? <ValueView value={v} /> : renderPrimitiveByKey(k, v)}</div>
          </div>
        ))}
      </div>
    );
  }

  return <span className="text-sm leading-relaxed">{String(value)}</span>;
}