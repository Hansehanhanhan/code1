import type { ReactNode } from "react";

const TABS: { id: string; label: string }[] = [
  { id: "dialogue", label: "对话" },
  { id: "trace", label: "轨迹" },
  { id: "state", label: "诊断状态" },
  { id: "jobs", label: "任务与指标" },
];

export function Tabs({
  active,
  onChange,
  children,
}: {
  active: string;
  onChange: (id: string) => void;
  children: ReactNode;
}) {
  return (
    <div className="tabs-shell">
      <div className="tabs-bar" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active === tab.id}
            className={`tab-btn ${active === tab.id ? "tab-btn-active" : ""}`}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tabs-panel" role="tabpanel">
        {children}
      </div>
    </div>
  );
}