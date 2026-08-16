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
  /** Sarvam speech-to-text round trip. Voice queries only. */
  stt_ms?: number;
  /** total_ms minus stt_ms — this system's own work, what the budget governs. */
  pipeline_ms?: number;
  /** The <200ms target, served by the API so it is never hardcoded here. */
  budget_ms?: number;
  total_ms: number;
}

/** Stage keys that are accounting totals rather than pipeline stages — shown
 *  as the headline figure and budget, so they must not repeat in the per-stage
 *  breakdown. */
export const SUMMARY_LATENCY_KEYS = ["total_ms", "pipeline_ms", "budget_ms"];

export interface QueryResponse {
  trace_id: string;
  answer_text: string;
  mode: AnswerMode;
  confidence: number;
  guardrail_flags: string[];
  evidence: EvidenceItem[];
  latency_ms: LatencyMs;
  /** This answer met the latency budget, and a generated one is available via
   *  POST /v1/query/refine. Absent on older API builds, hence optional. */
  refinement_available?: boolean;
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
