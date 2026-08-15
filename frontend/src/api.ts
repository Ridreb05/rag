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

export const submitVoice = (audio: Blob) => {
  const data = new FormData();
  data.append("audio", audio, `voice-query.${audio.type.includes("mp4") ? "mp4" : "webm"}`);
  data.append("top_k", "10");
  return api<VoiceQueryResponse>("/v1/voice-query", { method: "POST", body: data });
};
