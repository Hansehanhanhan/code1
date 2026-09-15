export function buildHeaders(apiKey: string) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey.trim()) {
    headers["X-API-Key"] = apiKey.trim();
  }
  return headers;
}

export function buildAuthHeaders(apiKey: string) {
  return apiKey.trim() ? { "X-API-Key": apiKey.trim() } : undefined;
}

export function joinUrl(base: string, ...segments: string[]) {
  const root = base.replace(/\/$/, "");
  return [root, ...segments.map((s) => s.replace(/^\/+/, ""))].join("/");
}