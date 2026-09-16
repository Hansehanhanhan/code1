import type { StreamEvent } from "@/types";

export type EventFrame = { type: string; content: unknown };

export function parseEventFrame(obj: unknown): EventFrame | null {
  if (!obj || typeof obj !== "object") return null;
  const raw = obj as Record<string, unknown>;
  if (typeof raw.type !== "string" || raw.type.length === 0) return null;
  const content = raw.content;
  if (content !== undefined && (content === null || typeof content !== "object")) {
    return null;
  }
  return { type: raw.type, content };
}

export function sanitizeStreamEvent(obj: unknown): StreamEvent | null {
  const frame = parseEventFrame(obj);
  if (!frame) {
    return null;
  }
  switch (frame.type) {
    case "agent_action":
    case "tool_observation":
    case "llm_observation": {
      const content = frame.content as { tool_loop_index?: unknown };
      if (typeof content.tool_loop_index !== "number") return null;
      break;
    }
    case "key_step": {
      const content = frame.content as { kind?: unknown };
      if (typeof content.kind !== "string") return null;
      break;
    }
    case "outer_decision": {
      const content = frame.content as { decision?: unknown; round?: unknown };
      if (typeof content.decision !== "string" || typeof content.round !== "number") {
        return null;
      }
      break;
    }
    case "final_response": {
      const content = frame.content as { final_answer?: unknown };
      if (typeof content.final_answer !== "string") return null;
      break;
    }
    case "stream_metrics":
    case "degraded_response":
      break;
    default:
      return null;
  }
  return frame as unknown as StreamEvent;
}