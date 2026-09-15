import { useEffect, useState } from "react";
import type { SessionState } from "@/types";
import { buildAuthHeaders, joinUrl } from "@/lib/api";

export function useSessionState(
  baseUrl: string,
  apiKey: string,
  sessionId: string,
  refreshToken: number
) {
  const [state, setState] = useState<SessionState | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const sid = sessionId.trim();
    if (!sid) {
      setState(null);
      return;
    }

    (async () => {
      try {
        const res = await fetch(joinUrl(baseUrl, "sessions", encodeURIComponent(sid)), {
          headers: buildAuthHeaders(apiKey) as Record<string, string> | undefined,
          cache: "no-store",
        });
        if (cancelled) {
          return;
        }
        if (res.ok) {
          setState((await res.json()) as SessionState);
          setError("");
          return;
        }
        setError(`会话状态不可用：${res.status}`);
        setState(null);
      } catch {
        if (!cancelled) {
          setError("会话状态拉取失败");
          setState(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [baseUrl, apiKey, sessionId, refreshToken]);

  return { state, error };
}