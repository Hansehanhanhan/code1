import { useCallback, useRef, useState } from "react";
import type {
  JobCancelResponse,
  JobCreateResponse,
  JobState,
  JobStatusResponse,
  JobStreamFrame,
  StreamEvent,
} from "@/types";
import { buildAuthHeaders, buildHeaders, joinUrl } from "@/lib/api";
import { readSseStream } from "@/lib/sse";

export function useJobStream(baseUrl: string, apiKey: string) {
  const [job, setJob] = useState<JobState | null>(null);
  const [jobEvents, setJobEvents] = useState<StreamEvent[]>([]);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState(false);
  const tokenRef = useRef(0);
  const aborterRef = useRef<AbortController | null>(null);

  const jobsEndpoint = `${baseUrl.replace(/\/$/, "")}`;

  const stopSubscription = useCallback(() => {
    tokenRef.current += 1;
    aborterRef.current?.abort();
    aborterRef.current = null;
  }, []);

  const subscribe = useCallback(
    async (jobId: string, onEvent: (evt: StreamEvent) => void) => {
      const token = ++tokenRef.current;
      aborterRef.current?.abort();
      const controller = new AbortController();
      aborterRef.current = controller;
      setJobEvents([]);

      try {
        const res = await fetch(joinUrl(jobsEndpoint, "jobs", jobId, "stream"), {
          headers: buildAuthHeaders(apiKey) as Record<string, string> | undefined,
          cache: "no-store",
          signal: controller.signal,
        });
        if (!res.ok) {
          throw new Error(`任务流订阅失败：${res.status} ${res.statusText}`);
        }
        if (!res.body) {
          return;
        }

        await readSseStream(res.body.getReader(), (obj) => {
          if (token !== tokenRef.current) {
            return;
          }
          const frame = obj as JobStreamFrame;
          if (frame.type === "heartbeat") {
            return;
          }
          const evt = { type: frame.type, content: frame.content } as StreamEvent;
          if (evt.type === "final_response") {
            setJob((prev) => (prev ? { ...prev, status: "succeeded" } : prev));
          } else if (evt.type === "degraded_response") {
            setJob((prev) => (prev ? { ...prev, status: "degraded" } : prev));
          } else if (evt.type === "error") {
            setJob((prev) => (prev ? { ...prev, status: "failed" } : prev));
          }
          if (evt.type !== "final_response") {
            setJobEvents((prev) => [...prev, evt]);
          }
          onEvent(evt);
        });

        if (token !== tokenRef.current) {
          return;
        }

        const statusRes = await fetch(joinUrl(jobsEndpoint, "jobs", jobId), {
          headers: buildAuthHeaders(apiKey) as Record<string, string> | undefined,
          cache: "no-store",
        });
        if (!statusRes.ok) {
          return;
        }
        const data = (await statusRes.json()) as JobStatusResponse;
        if (token !== tokenRef.current) {
          return;
        }
        setJob({
          job_id: data.job_id,
          status: data.status,
          created_at: data.created_at,
          updated_at: data.updated_at,
          error_message: data.error_message ?? undefined,
        });
        if (data.response) {
          onEvent({ type: "final_response", content: data.response } as StreamEvent);
        }
      } catch {
        if (token !== tokenRef.current) {
          return;
        }
        setError("任务流订阅中断，请重新提交或查看任务状态。");
      }
    },
    [apiKey, jobsEndpoint]
  );

  const createJob = useCallback(
    async (payload: Record<string, unknown>, onEvent: (evt: StreamEvent) => void) => {
      stopSubscription();
      setError("");
      setNotice("");
      setActionLoading(true);
      try {
        const res = await fetch(joinUrl(jobsEndpoint, "jobs"), {
          method: "POST",
          headers: buildHeaders(apiKey),
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          throw new Error(`提交任务失败：${res.status} ${res.statusText}`);
        }
        const created = (await res.json()) as JobCreateResponse;
        setJob({
          job_id: created.job_id,
          status: created.status,
          created_at: created.created_at,
          retry_of: created.retry_of ?? null,
        });
        setNotice(`任务已提交：${created.job_id}`);
        await subscribe(created.job_id, onEvent);
      } catch (err) {
        setError(err instanceof Error ? err.message : "任务提交失败，请稍后重试。");
      } finally {
        setActionLoading(false);
      }
    },
    [apiKey, jobsEndpoint, stopSubscription, subscribe]
  );

  const cancelJob = useCallback(
    async (jobId: string, onEvent: (evt: StreamEvent) => void) => {
      if (!jobId) {
        return;
      }
      setError("");
      setNotice("");
      setActionLoading(true);
      try {
        const res = await fetch(joinUrl(jobsEndpoint, "jobs", jobId, "cancel"), {
          method: "POST",
          headers: buildAuthHeaders(apiKey) as Record<string, string> | undefined,
        });
        if (!res.ok) {
          throw new Error(`取消失败：${res.status} ${res.statusText}`);
        }
        const data = (await res.json()) as JobCancelResponse;
        setJob((prev) => (prev ? { ...prev, status: data.status } : prev));
        setNotice(data.message);
        if (data.status === "cancel_requested") {
          await subscribe(jobId, onEvent);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "取消任务失败，请稍后重试。");
      } finally {
        setActionLoading(false);
      }
    },
    [apiKey, jobsEndpoint, subscribe]
  );

  const retryJob = useCallback(
    async (jobId: string, onEvent: (evt: StreamEvent) => void) => {
      if (!jobId) {
        return;
      }
      stopSubscription();
      setError("");
      setNotice("");
      setActionLoading(true);
      try {
        const res = await fetch(joinUrl(jobsEndpoint, "jobs", jobId, "retry"), {
          method: "POST",
          headers: buildAuthHeaders(apiKey) as Record<string, string> | undefined,
        });
        if (!res.ok) {
          throw new Error(`重试失败：${res.status} ${res.statusText}`);
        }
        const created = (await res.json()) as JobCreateResponse;
        setJob({
          job_id: created.job_id,
          status: created.status,
          created_at: created.created_at,
          retry_of: created.retry_of ?? null,
        });
        setNotice(`已创建重试任务：${created.job_id}`);
        await subscribe(created.job_id, onEvent);
      } catch (err) {
        setError(err instanceof Error ? err.message : "重试任务失败，请稍后重试。");
      } finally {
        setActionLoading(false);
      }
    },
    [apiKey, jobsEndpoint, stopSubscription, subscribe]
  );

  return {
    job,
    jobEvents,
    notice,
    error,
    actionLoading,
    createJob,
    cancelJob,
    retryJob,
    stopSubscription,
  };
}