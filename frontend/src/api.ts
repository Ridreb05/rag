import type { QueryRequest, QueryResponse, VoiceQueryResponse } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function api<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || "Something went wrong. Please try again.", response.status);
  }
  return response.json() as Promise<T>;
}

export const submitText = (body: QueryRequest) =>
  api<QueryResponse>("/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/** Phase two of a two-phase answer: upgrades a within-budget extractive
 *  answer to the generated one. Only call when the first response set
 *  `refinement_available`; a 404 means it expired or was already claimed, in
 *  which case the fast answer already on screen stays as-is. */
export const refineAnswer = (traceId: string) =>
  api<QueryResponse>("/v1/query/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: traceId }),
  });

/** Translates an answer to English for display. Deliberately a separate call:
 *  it must never enter the latency budget the query itself is measured against. */
export const translateText = (text: string) =>
  api<{ text: string; latency_ms: Record<string, number> }>("/v1/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

export const submitVoice = (audio: Blob) => {
  const data = new FormData();
  data.append("audio", audio, `voice-query.${audio.type.includes("mp4") ? "mp4" : "webm"}`);
  data.append("top_k", "10");
  return api<VoiceQueryResponse>("/v1/voice-query", { method: "POST", body: data });
};
