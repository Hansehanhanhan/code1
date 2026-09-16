import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useRunSession } from "./useRunSession";

type Deferred = { resolve: (value: unknown) => void };

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

function mockFetchResolvable() {
  const deferreds: Deferred[] = [];
  const fn = vi.fn(
    () =>
      new Promise<unknown>((resolve) => {
        deferreds.push({ resolve });
      })
  );
  vi.stubGlobal("fetch", fn);
  return deferreds;
}

const finalEvent = (answer: string) => ({
  type: "final_response",
  content: { final_answer: answer, steps: [], metrics: {} },
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useRunSession.runStream", () => {
  it("loading 在请求进行中为 true、结束后为 false", async () => {
    const { result } = renderHook(() => useRunSession());
    const deferreds = mockFetchResolvable();

    let running!: Promise<void>;
    await act(async () => {
      running = result.current.runStream("http://x/run_stream", {}, {});
    });
    expect(result.current.loading).toBe(true);

    await act(async () => {
      deferreds[0].resolve(sseResponse([finalEvent("ok")]));
      await running;
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.response?.final_answer).toBe("ok");
  });

  it("并发提交时旧流的迟到事件被丢弃", async () => {
    const { result } = renderHook(() => useRunSession());
    const deferreds = mockFetchResolvable();

    let first!: Promise<void>;
    await act(async () => {
      first = result.current.runStream("http://x/run_stream", {}, {});
    });
    let second!: Promise<void>;
    await act(async () => {
      second = result.current.runStream("http://x/run_stream", {}, {});
    });

    await act(async () => {
      deferreds[1].resolve(sseResponse([finalEvent("second")]));
      await second;
    });
    expect(result.current.response?.final_answer).toBe("second");

    await act(async () => {
      deferreds[0].resolve(sseResponse([finalEvent("stale")]));
      await first;
    });
    expect(result.current.response?.final_answer).toBe("second");
  });

  it("degraded_response 记录降级备注，stream_metrics 置 degraded 标志", async () => {
    const { result } = renderHook(() => useRunSession());
    const deferreds = mockFetchResolvable();

    let running!: Promise<void>;
    await act(async () => {
      running = result.current.runStream("http://x/run_stream", {}, {});
    });
    await act(async () => {
      deferreds[0].resolve(
        sseResponse([
          { type: "degraded_response", content: { reason: "LLM 超时" } },
          { type: "stream_metrics", content: { degraded: true, ttfb_ms: 5 } },
          finalEvent("兜底"),
        ])
      );
      await running;
    });
    expect(result.current.degraded).toBe(true);
    expect(result.current.notes.join(" ")).toContain("服务降级");
    expect(result.current.response?.metrics?.ttfb_ms).toBe(5);
  });

  it("错误帧被 sanitize 丢弃且不中断流", async () => {
    const { result } = renderHook(() => useRunSession());
    const deferreds = mockFetchResolvable();

    let running!: Promise<void>;
    await act(async () => {
      running = result.current.runStream("http://x/run_stream", {}, {});
    });
    await act(async () => {
      deferreds[0].resolve(
        sseResponse([
          { type: "agent_action", content: { no_index: true } },
          "garbage",
          finalEvent("ok"),
        ])
      );
      await running;
    });
    expect(result.current.response?.final_answer).toBe("ok");
  });
});