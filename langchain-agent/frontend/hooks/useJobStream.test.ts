import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJobStream } from "./useJobStream";

const BASE = "http://127.0.0.1:8000";

function sseResponse(frames: unknown[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
      }
      controller.close();
    },
  });
  return { ok: true, status: 200, statusText: "OK", body } as unknown as Response;
}

function jsonResponse(value: unknown) {
  return { ok: true, status: 200, statusText: "OK", json: async () => value } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useJobStream", () => {
  it("degraded 状态在直播中不被 final_response 覆盖", async () => {
    const frames = [
      { job_id: "job-1", event_id: 1, type: "degraded_response", content: { reason: "LLM 超时" } },
      {
        job_id: "job-1",
        event_id: 2,
        type: "final_response",
        content: { final_answer: "兜底", steps: [], metrics: {} },
      },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/jobs")) {
          return Promise.resolve(jsonResponse({ job_id: "job-1", status: "queued", created_at: 1 }));
        }
        if (u.endsWith("/stream")) {
          return Promise.resolve(sseResponse(frames));
        }
        return Promise.resolve(
          jsonResponse({
            job_id: "job-1",
            status: "degraded",
            created_at: 1,
            updated_at: 2,
            response: null,
          })
        );
      })
    );

    const { result } = renderHook(() => useJobStream(BASE, ""));
    const onEvent = vi.fn();
    await act(async () => {
      await result.current.createJob({ query: "q" }, onEvent);
    });

    expect(result.current.job?.status).toBe("degraded");
    expect(result.current.jobEvents.map((e) => e.type)).toEqual(["degraded_response"]);
  });

  it("jobEvents 只收集非 final_response 事件", async () => {
    const frames = [
      { job_id: "job-1", event_id: 1, type: "key_step", content: { kind: "context_update" } },
      {
        job_id: "job-1",
        event_id: 2,
        type: "final_response",
        content: { final_answer: "ok", steps: [], metrics: {} },
      },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/jobs")) {
          return Promise.resolve(jsonResponse({ job_id: "job-1", status: "queued", created_at: 1 }));
        }
        if (u.endsWith("/stream")) {
          return Promise.resolve(sseResponse(frames));
        }
        return Promise.resolve(
          jsonResponse({ job_id: "job-1", status: "succeeded", created_at: 1, updated_at: 2 })
        );
      })
    );

    const { result } = renderHook(() => useJobStream(BASE, ""));
    await act(async () => {
      await result.current.createJob({ query: "q" }, vi.fn());
    });

    expect(result.current.jobEvents.map((e) => e.type)).toEqual(["key_step"]);
    expect(result.current.job?.status).toBe("succeeded");
  });

  it("heartbeat 帧被忽略、非法帧被丢弃", async () => {
    const frames = [
      { job_id: "job-1", type: "heartbeat" },
      "garbage",
      { job_id: "job-1", event_id: 1, type: "key_step", content: { kind: "first_evidence" } },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/jobs")) {
          return Promise.resolve(jsonResponse({ job_id: "job-1", status: "queued", created_at: 1 }));
        }
        if (u.endsWith("/stream")) {
          return Promise.resolve(sseResponse(frames));
        }
        return Promise.resolve(
          jsonResponse({ job_id: "job-1", status: "succeeded", created_at: 1, updated_at: 2 })
        );
      })
    );

    const { result } = renderHook(() => useJobStream(BASE, ""));
    await act(async () => {
      await result.current.createJob({ query: "q" }, vi.fn());
    });

    expect(result.current.jobEvents.map((e) => e.type)).toEqual(["key_step"]);
  });
});