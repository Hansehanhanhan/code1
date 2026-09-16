"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { ContextFormData, StreamEvent, TabId } from "@/types";
import { buildHeaders, joinUrl } from "@/lib/api";
import { useRunSession } from "@/hooks/useRunSession";
import { useJobStream } from "@/hooks/useJobStream";
import { useSessionState } from "@/hooks/useSessionState";
import { Tabs } from "@/components/shared/Tabs";
import { InputPanel } from "@/components/InputPanel";
import { FinalAnswerPanel } from "@/components/FinalAnswerPanel";
import { StepTrace } from "@/components/StepTrace";
import { SessionStatePanel } from "@/components/SessionStatePanel";
import { JobPanel } from "@/components/JobPanel";

const SAMPLE_QUERY = "本周店铺流量明显下滑，请诊断原因并给出可执行建议。";
const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

export default function Home() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_BACKEND_URL;
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [sessionId, setSessionId] = useState("demo-session-001");
  const [apiKey, setApiKey] = useState("");
  const [context, setContext] = useState<ContextFormData>({
    merchant_id: "demo-001",
    time_range: "last_7_days",
    category: "retail",
  });
  const [activeTab, setActiveTab] = useState<TabId>("dialogue");

  const runSession = useRunSession();
  const jobHook = useJobStream(backendUrl, apiKey);
  const sessionState = useSessionState(
    backendUrl,
    apiKey,
    sessionId,
    runSession.refreshToken
  );

  const streamEndpoint = useMemo(() => `${joinUrl(backendUrl, "run_stream")}`, [backendUrl]);

  const payload = useMemo(
    () => ({
      query,
      session_id: sessionId.trim() ? sessionId.trim() : undefined,
      context: {
        merchant_id: context.merchant_id,
        time_range: context.time_range,
        category: context.category,
      },
    }),
    [query, sessionId, context]
  );

  const onStreamEvent = useCallback(
    (evt: StreamEvent) => {
      runSession.handleStreamEvent(evt);
      if (evt.type === "final_response" || evt.type === "error") {
        runSession.bumpRefresh();
      }
    },
    [runSession]
  );

  useEffect(() => {
    return () => {
      jobHook.stopSubscription();
      runSession.stopRun();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    jobHook.stopSubscription();
    void runSession.runStream(
      streamEndpoint,
      payload,
      buildHeaders(apiKey),
      () => runSession.bumpRefresh()
    );
    setActiveTab("dialogue");
  }

  function handleCreateJob() {
    void jobHook.createJob(payload, onStreamEvent);
  }

  function handleCancelJob() {
    if (jobHook.job?.job_id) {
      void jobHook.cancelJob(jobHook.job.job_id, onStreamEvent);
    }
  }

  function handleRetryJob() {
    if (jobHook.job?.job_id) {
      void jobHook.retryJob(jobHook.job.job_id, onStreamEvent);
    }
  }

  return (
    <main className="page-shell">
      <InputPanel
        query={query}
        onQueryChange={setQuery}
        sessionId={sessionId}
        onSessionIdChange={setSessionId}
        apiKey={apiKey}
        onApiKeyChange={setApiKey}
        context={context}
        onContextChange={setContext}
        loading={runSession.loading}
        jobActionLoading={jobHook.actionLoading}
        onSubmit={handleSubmit}
        onCreateJob={handleCreateJob}
        streamEndpoint={streamEndpoint}
      />

      <Tabs active={activeTab} onChange={(id) => setActiveTab(id as TabId)}>
        {activeTab === "dialogue" ? (
          <FinalAnswerPanel
            response={runSession.response}
            error={runSession.error}
            notes={runSession.notes}
          />
        ) : null}
        {activeTab === "trace" ? (
          <StepTrace steps={runSession.response?.steps ?? []} milestones={runSession.milestones} />
        ) : null}
        {activeTab === "state" ? (
          <SessionStatePanel state={sessionState.state} error={sessionState.error} />
        ) : null}
        {activeTab === "jobs" ? (
          <JobPanel
            job={jobHook.job}
            jobEvents={jobHook.jobEvents}
            notice={jobHook.notice}
            error={jobHook.error}
            actionLoading={jobHook.actionLoading}
            metrics={runSession.response?.metrics}
            onCancel={handleCancelJob}
            onRetry={handleRetryJob}
          />
        ) : null}
      </Tabs>
    </main>
  );
}