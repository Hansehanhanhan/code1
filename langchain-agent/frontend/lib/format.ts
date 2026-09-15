import type { StepRecord } from "@/types";

const LOOP_STEP_PREFIX = "Tool Loop";

export function stripMarkdownBold(text: string) {
  return text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*\*/g, "");
}

export function formatFinalAnswer(text: string) {
  let formatted = stripMarkdownBold(text).replace(/\r\n/g, "\n").trim();
  formatted = formatted.replace(/建议采取以下措施[:：]/g, "建议采取以下措施：\n");
  formatted = formatted.replace(/([。；])(?=\d+[\.、])/g, "$1\n");
  formatted = formatted.replace(/([。；])(?=[一二三四五六七八九十]+是)/g, "$1\n");
  formatted = formatted.replace(/(^|[^\n])(核心原因[:：]|行动建议[:：]|风险与复盘[:：]|问题判断[:：])/g, "$1\n$2");
  formatted = formatted.replace(/(^|[^\n])(\d+[\.、])/g, "$1\n$2");
  formatted = formatted.replace(/\n{3,}/g, "\n\n").trim();
  return formatted;
}

export function splitAnswerAndEvidence(text: string) {
  const marker = "证据来源：";
  const idx = text.indexOf(marker);
  if (idx === -1) {
    return { answerText: text, evidenceLines: [] as string[] };
  }
  const answerText = text.slice(0, idx).trim();
  const tail = text.slice(idx + marker.length).trim();
  const evidenceLines = tail
    .split("\n")
    .map((line) => line.replace(/^\d+[\.、]\s*/, "").trim())
    .filter((line) => line.length > 0);
  return { answerText, evidenceLines };
}

export function formatStepName(name: string) {
  if (name.startsWith(LOOP_STEP_PREFIX)) {
    const index = name.replace(LOOP_STEP_PREFIX, "").trim();
    return `工具轨迹 第 ${index} 步`;
  }
  return name;
}

export function upsertLoopStep(steps: StepRecord[], loopIndex: number, patch: Partial<StepRecord>): StepRecord[] {
  const name = `${LOOP_STEP_PREFIX} ${loopIndex}`;
  const idx = steps.findIndex((s) => s.name === name);
  if (idx === -1) {
    return [...steps, { name, ...patch }];
  }
  const copy = [...steps];
  copy[idx] = { ...copy[idx], ...patch, name };
  return copy;
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function formatMetricValue(key: string, value: unknown) {
  if (value === undefined || value === null) {
    return "未上报";
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (typeof value === "number" && key.endsWith("_ms")) {
    return `${value} ms`;
  }
  return String(value);
}