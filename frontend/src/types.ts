export interface QueryRequest {
  query: string;
  language?: string;
  top_k?: number;
}

export type AnswerMode = "refused" | "extractive" | "generative";

export interface EvidenceItem {
  chunk_id: string;
  text: string;
  rerank_score: number | null;
}

export interface LatencyMs {
  embedding_ms?: number;
  retrieval_ms?: number;
  fusion_ms?: number;
  rerank_ms?: number;
  generation_ms?: number;
  total_ms: number;
}

export interface QueryResponse {
  trace_id: string;
  answer_text: string;
  mode: AnswerMode;
  confidence: number;
  guardrail_flags: string[];
  evidence: EvidenceItem[];
  latency_ms: LatencyMs;
}

export interface VoiceQueryResponse extends QueryResponse {
  transcript: string;
}

export type Result = QueryResponse | VoiceQueryResponse;

export interface HealthResponse {
  status: "ok";
}

export interface HistoryItem {
  query: string;
  result: Result;
  voice: boolean;
}

export const isVoice = (result: Result): result is VoiceQueryResponse => "transcript" in result;

export const formatMs = (value: number) => (value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`);
