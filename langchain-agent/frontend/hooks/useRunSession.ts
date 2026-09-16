import { useCallback, useRef, useState } from "react";
import type { KeyStepEvent, RunResponse, StepRecord, StreamEvent } from "@/types";
import { readSseStream } from "@/lib/sse";
import { sanitizeStreamEvent } from "@/lib/guards";
import { upsertLoopStep } from "@/lib/format";

export function useRunSession() {
  const [response, setResponse] = useState<RunResponse | null>(null);
  const [error, setError] = useState("");
  const [notes, setNotes] = useState<string[]>([]);
  const [milestones, setMilestones] = useState<KeyStepEvent[]>([]);
  const [degraded, setDegraded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);
  const stepsRef = useRef<StepRecord[]>([]);
  const finalRef = useRef<RunResponse | null>(null);
  const genRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const isActive = useCallback((gen: number) => gen === genRef.current, []);

  const setLive = useCallback((steps: StepRecord[]) => {
    setResponse((prev) => ({
      final_answer: prev?.final_answer ?? "",
      steps: [...steps],
      metrics: prev?.metrics ?? {},
    }));
  }, []);

  const handleStreamEvent = useCallback(
    (evt: StreamEvent) => {
      switch (evt.type) {
        case "agent_action":
          stepsRef.current = upsertLoopStep(stepsRef.current, evt.content.tool_loop_index, {
            input: {
              thought: evt.content.thought,
              action: evt.content.action,
              action_input: evt.content.action_input,
            },
          });
          setLive(stepsRef.current);
          break;
        case "llm_observation":
          stepsRef.current = upsertLoopStep(stepsRef.current, evt.content.tool_loop_index, {
            llm_duration_ms: evt.content.duration_ms,
          });
          setLive(stepsRef.current);
          break;
        case "tool_observation": {
          const prev = stepsRef.current.find((s) => s.name === `Tool Loop ${evt.content.tool_loop_index}`);
          stepsRef.current = upsertLoopStep(stepsRef.current, evt.content.tool_loop_index, {
            input: prev?.input,
            output: { observation: evt.content.observation },
            duration_ms: evt.content.duration_ms,
          });
          setLive(stepsRef.current);
          break;
        }
        case "key_step":
          setMilestones((prev) => [...prev, evt.content]);
          break;
        case "outer_decision":
          setNotes((prev) => [
            ...prev,
            `自省外环 R${evt.content.round} 决策 ${evt.content.decision}（置信 ${evt.content.confidence ?? "—"}）：${evt.content.reason}`,
          ]);
          break;
        case "degraded_response":
          setNotes((prev) => [...prev, `服务降级：${evt.content.reason}`]);
          break;
        case "stream_metrics":
          setResponse((prev) => ({
            final_answer: prev?.final_answer ?? "",
            steps: prev?.steps ?? [...stepsRef.current],
            metrics: { ...(prev?.metrics ?? {}), ...evt.content },
          }));
          setDegraded(Boolean(evt.content.degraded));
          break;
        case "final_response":
          finalRef.current = evt.content;
          setResponse((prev) => ({
            ...evt.content,
            metrics: { ...(evt.content.metrics ?? {}), ...(prev?.metrics ?? {}) },
          }));
          break;
        case "error":
          setError(evt.content || "流式请求失败。");
          break;
      }
    },
    [setLive]
  );

  const reset = useCallback(() => {
    genRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    stepsRef.current = [];
    finalRef.current = null;
    setDegraded(false);
    setMilestones([]);
    setNotes([]);
    setError("");
    setResponse({ final_answer: "", steps: [], metrics: {} });
  }, []);

  const stopRun = useCallback(() => {
    genRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const bumpRefresh = useCallback(() => {
    setTick((n) => n + 1);
  }, []);

  async function runStream(
    url: string,
    payload: Record<string, unknown>,
    headers: Record<string, string>,
    onFinished?: (response: RunResponse | null) => void
  ) {
    reset();
    setLoading(true);
    const gen = genRef.current;
    const controller = new AbortController();
    abortRef.current = controller;

    const dispatch = (obj: unknown) => {
      if (!isActive(gen)) {
        return;
      }
      const evt = sanitizeStreamEvent(obj);
      if (!evt) {
        console.warn("[run-stream] 忽略非法事件帧", obj);
        return;
      }
      handleStreamEvent(evt);
    };

    try {
      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!isActive(gen)) {
        return;
      }
      if (!res.ok) {
        throw new Error(`请求失败：${res.status} ${res.statusText}`);
      }

      if (!res.body) {
        const data = (await res.json()) as RunResponse;
        if (isActive(gen)) {
          setResponse(data);
          onFinished?.(data);
        }
        return;
      }

      await readSseStream(res.body.getReader(), dispatch);
      if (isActive(gen)) {
        onFinished?.(finalRef.current);
      }
    } catch (err) {
      if (!isActive(gen) || controller.signal.aborted) {
        return;
      }
      setError(err instanceof Error ? err.message : "请求失败，请稍后重试。");
      onFinished?.(null);
    } finally {
      if (isActive(gen)) {
        setLoading(false);
      }
    }
  }

  return {
    response,
    error,
    notes,
    milestones,
    degraded,
    loading,
    refreshToken: tick,
    handleStreamEvent,
    reset,
    stopRun,
    runStream,
    bumpRefresh,
  };
}