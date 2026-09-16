export type StepRecord = {
  name: string;
  input?: unknown;
  output?: unknown;
  duration_ms?: number;
  llm_duration_ms?: number;
};

export type RunResponse = {
  final_answer: string;
  steps: StepRecord[];
  metrics?: Record<string, unknown>;
};

export type JobCreateResponse = {
  job_id: string;
  status: string;
  created_at: number;
  retry_of?: string | null;
};

export type JobStatusResponse = {
  job_id: string;
  status: string;
  created_at: number;
  updated_at: number;
  error_message?: string | null;
  response?: RunResponse | null;
};

export type JobCancelResponse = {
  job_id: string;
  status: string;
  cancelled: boolean;
  message: string;
};

export type JobState = {
  job_id: string;
  status: string;
  created_at: number;
  updated_at?: number;
  error_message?: string;
  retry_of?: string | null;
};

export type KeyStepKind =
  | "first_evidence"
  | "context_update"
  | "direction_repair"
  | "context_update_rejected";

export type KeyStepEvent = {
  kind: KeyStepKind;
  action?: string;
  evidence_id?: string;
  reason?: string;
  new_version?: number;
  candidate_previews?: string[];
  error_type?: string;
  error?: string;
};

export type AgentActionContent = {
  tool_loop_index: number;
  thought: string;
  action: string;
  action_input: unknown;
};

export type ToolObservationContent = {
  tool_loop_index: number;
  observation: unknown;
  duration_ms: number;
};

export type LlmObservationContent = {
  tool_loop_index: number;
  duration_ms: number;
  llm_latency_ms?: number;
};

export type StreamMetricsContent = {
  ttfb_ms?: number;
  event_count?: number;
  event_completeness?: boolean;
  degraded?: boolean;
};

export type OuterDecisionContent = {
  decision: "accept" | "verify" | "restart";
  confidence: number | null;
  reason: string;
  round: number;
};

export type StreamEvent =
  | { type: "agent_action"; content: AgentActionContent }
  | { type: "tool_observation"; content: ToolObservationContent }
  | { type: "llm_observation"; content: LlmObservationContent }
  | { type: "key_step"; content: KeyStepEvent }
  | { type: "outer_decision"; content: OuterDecisionContent }
  | { type: "stream_metrics"; content: StreamMetricsContent }
  | { type: "final_response"; content: RunResponse }
  | { type: "degraded_response"; content: { reason: string } }
  | { type: "error"; content: string };

export type JobStreamFrame =
  | { job_id: string; event_id?: number; type: "heartbeat" }
  | { job_id: string; event_id: number; type: StreamEvent["type"]; content: unknown };

export type VerifiedFinding = {
  id: string;
  claim: string;
  confidence?: string;
  evidence?: {
    evidence_id?: string;
    request_id?: string;
    observation_hash?: string;
  };
};

export type SessionState = {
  session_id: string;
  version: number;
  context_slots: Record<string, unknown>;
  verified_findings: VerifiedFinding[];
  current_candidates: Record<string, unknown>;
  unresolved_constraints: string[];
  validity_concerns: string[];
  rejected_candidates: string[];
  next_step_plan: string[];
  constraint_status: Record<string, string>;
  answer_confidence: number | null;
  verified_citations: string[];
  history_turns: number;
};

export type ContextFormData = {
  merchant_id: string;
  time_range: string;
  category: string;
};

export type TabId = "dialogue" | "trace" | "state" | "jobs";