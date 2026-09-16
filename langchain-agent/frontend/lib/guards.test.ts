import { describe, expect, it } from "vitest";
import { parseEventFrame, sanitizeStreamEvent } from "./guards";

describe("parseEventFrame", () => {
  it("接受带 content 的合法帧", () => {
    expect(parseEventFrame({ type: "heartbeat" })).toEqual({ type: "heartbeat", content: undefined });
    expect(parseEventFrame({ type: "error", content: { x: 1 } })).toEqual({
      type: "error",
      content: { x: 1 },
    });
  });

  it("拒绝非对象 / 缺 type / 空 type / content 非法", () => {
    expect(parseEventFrame(null)).toBeNull();
    expect(parseEventFrame("hello")).toBeNull();
    expect(parseEventFrame({})).toBeNull();
    expect(parseEventFrame({ type: "" })).toBeNull();
    expect(parseEventFrame({ type: "x", content: "string" })).toBeNull();
  });
});

describe("sanitizeStreamEvent", () => {
  it("关键词序字段缺失的帧被丢弃", () => {
    expect(sanitizeStreamEvent({ type: "agent_action", content: { thought: "t" } })).toBeNull();
    expect(sanitizeStreamEvent({ type: "outer_decision", content: { decision: "verify" } })).toBeNull();
  });

  it("未知事件类型返回 null（run_stream 不允许未知类型）", () => {
    expect(sanitizeStreamEvent({ type: "idempotent_reused", content: { job_id: "1" } })).toBeNull();
  });

  it("合法 final_response 通过", () => {
    const evt = sanitizeStreamEvent({
      type: "final_response",
      content: { final_answer: "ok", steps: [], metrics: {} },
    });
    expect(evt).not.toBeNull();
    expect(evt?.type).toBe("final_response");
  });

  it("合法 tool_observation 通过并保留字段", () => {
    const evt = sanitizeStreamEvent({
      type: "tool_observation",
      content: { tool_loop_index: 1, observation: { x: 1 }, duration_ms: 12 },
    });
    expect(evt).not.toBeNull();
    expect((evt as { content: { tool_loop_index: number } }).content.tool_loop_index).toBe(1);
  });
});