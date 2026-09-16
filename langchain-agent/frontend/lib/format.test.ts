import { describe, expect, it } from "vitest";
import {
  formatFinalAnswer,
  formatMetricValue,
  formatStepName,
  splitAnswerAndEvidence,
  upsertLoopStep,
} from "./format";

describe("format", () => {
  it("formatStepName 把 Tool Loop 序号翻译成中文轨迹", () => {
    expect(formatStepName("Tool Loop 2")).toBe("工具轨迹 第 2 步");
    expect(formatStepName("Clarification")).toBe("Clarification");
  });

  it("upsertLoopStep 首次插入、二次合并且保持顺序", () => {
    let steps = upsertLoopStep([], 0, { input: { thought: "t" } });
    expect(steps).toHaveLength(1);
    expect(steps[0].name).toBe("Tool Loop 0");

    steps = upsertLoopStep(steps, 0, { output: { observation: "o" } });
    expect(steps).toHaveLength(1);
    expect(steps[0].output).toEqual({ observation: "o" });
    expect(steps[0].input).toEqual({ thought: "t" });

    steps = upsertLoopStep(steps, 1, { input: { thought: "t2" } });
    expect(steps).toHaveLength(2);
    expect(steps[1].name).toBe("Tool Loop 1");
  });

  it("splitAnswerAndEvidence 拆分答案与证据行", () => {
    const text = "结论：A\n证据来源：\n1. ev-1\n2. ev-2\n";
    const { answerText, evidenceLines } = splitAnswerAndEvidence(text);
    expect(answerText).toBe("结论：A");
    expect(evidenceLines).toEqual(["ev-1", "ev-2"]);
  });

  it("splitAnswerAndEvidence 无证据标记时原样返回", () => {
    const { answerText, evidenceLines } = splitAnswerAndEvidence("只有回答");
    expect(answerText).toBe("只有回答");
    expect(evidenceLines).toEqual([]);
  });

  it("formatFinalAnswer 整理硬换行与多空行", () => {
    const out = formatFinalAnswer("**开头**。\n\n\n行动建议：a\n1. 一\n2. 二");
    expect(out).not.toContain("\n\n\n");
    expect(out).not.toContain("**");
  });

  it("formatMetricValue 处理布尔与毫秒与空值", () => {
    expect(formatMetricValue("latency_ms", 120)).toBe("120 ms");
    expect(formatMetricValue("degraded", true)).toBe("是");
    expect(formatMetricValue("x", null)).toBe("未上报");
    expect(formatMetricValue("outer_rounds", 2)).toBe("2");
  });
});