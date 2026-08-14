import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation } from "@tanstack/react-query";
import * as Accordion from "@radix-ui/react-accordion";
import * as Collapsible from "@radix-ui/react-collapsible";
import * as Toast from "@radix-ui/react-toast";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowUpRight,
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  Database,
  FileText,
  Mic,
  Radio,
  RotateCcw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Square,
  Volume2,
  Waves,
  X,
  Zap,
} from "lucide-react";
import "@fontsource/imbue/500.css";
import "@fontsource/imbue/600.css";
import "@fontsource/imbue/700.css";
import "@fontsource/victor-mono/500.css";
import "@fontsource/victor-mono/600.css";
import "@fontsource/noto-sans-devanagari/400.css";
import "@fontsource/noto-sans-devanagari/500.css";
import "@fontsource/noto-sans-devanagari/700.css";
import "./styles.css";

interface QueryRequest {
  query: string;
  language?: string;
  top_k?: number;
}

type AnswerMode = "refused" | "extractive" | "generative";

interface EvidenceItem {
  chunk_id: string;
  text: string;
  rerank_score: number | null;
}

interface LatencyMs {
  embedding_ms?: number;
  retrieval_ms?: number;
  fusion_ms?: number;
  rerank_ms?: number;
  generation_ms?: number;
  total_ms: number;
}

interface QueryResponse {
  trace_id: string;
  answer_text: string;
  mode: AnswerMode;
  confidence: number;
  guardrail_flags: string[];
  evidence: EvidenceItem[];
  latency_ms: LatencyMs;
}

interface VoiceQueryResponse extends QueryResponse {
  transcript: string;
}

type Result = QueryResponse | VoiceQueryResponse;

interface HealthResponse {
  status: "ok";
}

interface HistoryItem {
  query: string;
  result: Result;
  voice: boolean;
}

class ApiError extends Error {
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

const submitText = (body: QueryRequest) =>
  api<QueryResponse>("/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const submitVoice = (audio: Blob) => {
  const data = new FormData();
  data.append("audio", audio, `voice-query.${audio.type.includes("mp4") ? "mp4" : "webm"}`);
  data.append("language", "hi");
  data.append("top_k", "10");
  return api<VoiceQueryResponse>("/v1/voice-query", { method: "POST", body: data });
};

const modeCopy: Record<AnswerMode, { label: string; eyebrow: string }> = {
  refused: { label: "No grounded answer", eyebrow: "Signal not found" },
  extractive: { label: "Direct match", eyebrow: "Corpus extract" },
  generative: { label: "Grounded synthesis", eyebrow: "Answer assembled" },
};

const prompts = [
  "भारत का राष्ट्रीय खेल क्या है?",
  "मैनहट्टन परियोजना का प्रभाव क्या था?",
  "What is the purpose of a firewall?",
];

const isVoice = (result: Result): result is VoiceQueryResponse => "transcript" in result;
const formatMs = (value: number) => (value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`);

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [micDisabled, setMicDisabled] = useState(false);
  const [micMessage, setMicMessage] = useState("");
  const [health, setHealth] = useState<"checking" | "ready" | "offline">("checking");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const presentError = (error: unknown) => {
    setToast(
      error instanceof ApiError
        ? error.status === 429
          ? "You’re sending requests quickly—please slow down a little."
          : error.message
        : "Something went wrong. Please try again.",
    );
  };

  const recordResult = (next: Result, sourceQuery: string, voice: boolean) => {
    setResult(next);
    setHistory((current) => [
      { query: sourceQuery, result: next, voice },
      ...current.filter((item) => item.result.trace_id !== next.trace_id),
    ].slice(0, 5));
  };

  const textMutation = useMutation({
    mutationFn: submitText,
    onSuccess: (next) => recordResult(next, query.trim(), false),
    onError: presentError,
  });

  const voiceMutation = useMutation({
    mutationFn: submitVoice,
    onSuccess: (next) => recordResult(next, next.transcript, true),
    onError: (error) => {
      if (error instanceof ApiError && error.status === 503) {
        setMicDisabled(true);
        setMicMessage("Voice input is not configured on this deployment.");
      }
      presentError(error);
    },
  });

  const pending = textMutation.isPending || voiceMutation.isPending;

  useEffect(() => {
    let mounted = true;
    fetch("/v1/health")
      .then((response) => (response.ok ? (response.json() as Promise<HealthResponse>) : Promise.reject()))
      .then(() => mounted && setHealth("ready"))
      .catch(() => mounted && setHealth("offline"));

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
    };
  }, []);

  const runText = (event: React.FormEvent) => {
    event.preventDefault();
    const clean = query.trim();
    if (!clean || pending) return;
    setResult(null);
    textMutation.mutate({ query: clean, language: "hi", top_k: 10 });
  };

  const stopRecording = () => {
    if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
    recorderRef.current?.stop();
  };

  const toggleRecording = async () => {
    if (recording) {
      stopRecording();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setMicMessage("Audio recording is not supported by this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const chunks: BlobPart[] = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        const audio = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        if (audio.size) {
          setResult(null);
          voiceMutation.mutate(audio);
        } else {
          setToast("No audio was captured. Please try again.");
        }
      };
      recorder.start();
      setMicMessage("");
      setRecording(true);
      timeoutRef.current = window.setTimeout(stopRecording, 29_500);
    } catch {
      setMicMessage("Microphone permission was denied. Allow access in your browser settings to speak a question.");
    }
  };

  return (
    <Toast.Provider swipeDirection="right">
      <div className="page-noise" aria-hidden="true" />
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Vaani home">
          <span className="brand-mark"><AudioLines size={25} /></span>
          <span><strong>VĀNI</strong><small>वाणी</small></span>
        </a>
        <div className="header-meta">
          <span>VOICE-ENABLED RAG</span>
          <span className={`health ${health}`}><i />{health === "ready" ? "SYSTEM LIVE" : health === "checking" ? "CHECKING" : "OFFLINE"}</span>
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-stamp" aria-hidden="true">बोलो</div>
          <div className="hero-copy">
            <p className="kicker"><span>Task 02</span> / Hindi knowledge engine</p>
            <h1>ASK THE CORPUS.<br /><em>GET THE SIGNAL.</em></h1>
            <div className="hero-bottom">
              <p>Speak or type a question. Vāni retrieves the strongest evidence, checks the answer, and shows exactly what it found.</p>
              <div className="hero-index"><span>01</span><small>VOICE → EVIDENCE<br />NO NOISE. NO GUESSING.</small></div>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="sun-disc"><Waves size={108} strokeWidth={1.1} /></div>
            <span className="orbit orbit-one" />
            <span className="orbit orbit-two" />
            <i className="leaf leaf-one" /><i className="leaf leaf-two" /><i className="leaf leaf-three" />
          </div>
        </section>

        <section className="query-zone" aria-label="Ask the knowledge corpus">
          <div className="query-label"><span>LIVE CONSOLE</span><i /><small>HINDI CORPUS · TOP 10</small></div>
          <form className="query-card" onSubmit={runText}>
            <label htmlFor="question">What do you want to know?</label>
            <div className="input-shell">
              <textarea
                id="question"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    runText(event);
                  }
                }}
                maxLength={2000}
                placeholder="मुझसे भारत, इतिहास, विज्ञान या तकनीक के बारे में पूछें…"
                rows={3}
                disabled={pending}
              />
              <span className="character-count">{query.length}/2000</span>
            </div>
            <div className="query-actions">
              <button
                type="button"
                className={`voice-button ${recording ? "recording" : ""}`}
                onClick={toggleRecording}
                disabled={pending || micDisabled}
              >
                <span>{recording ? <Square size={20} fill="currentColor" /> : <Mic size={23} />}</span>
                <div><strong>{recording ? "STOP & SEARCH" : "ASK WITH VOICE"}</strong><small>{recording ? "Listening now…" : "Up to 30 seconds"}</small></div>
              </button>
              <button className="search-button" disabled={!query.trim() || pending}>
                <span>{pending ? "SEARCHING" : "SEARCH CORPUS"}</span><ArrowUpRight size={24} />
              </button>
            </div>
            {micMessage && <p className="mic-note"><CircleAlert size={15} />{micMessage}</p>}
          </form>

          <div className="prompt-strip">
            <span>TRY A SIGNAL</span>
            <div>{prompts.map((prompt, index) => <button key={prompt} onClick={() => setQuery(prompt)}><b>0{index + 1}</b>{prompt}</button>)}</div>
          </div>
        </section>

        <AnimatePresence mode="wait">
          {pending && <LoadingCard voice={voiceMutation.isPending} />}
          {result && !pending && <ResultCard result={result} />}
        </AnimatePresence>

        <section className="system-section">
          <div className="section-heading">
            <p className="kicker"><span>Under the hood</span> / evidence first</p>
            <h2>ONE QUESTION.<br /><em>FOUR CHECKPOINTS.</em></h2>
          </div>
          <Pipeline health={health} />
          <History history={history} onReplay={(item) => { setQuery(item.query); setResult(item.result); window.scrollTo({ top: 700, behavior: "smooth" }); }} />
        </section>
      </main>

      <footer>
        <div className="brand footer-brand"><span className="brand-mark"><AudioLines size={22} /></span><span><strong>VĀNI</strong><small>वाणी</small></span></div>
        <p>BUILT FOR <b>#RAGInGoa</b> · GROUNDED ANSWERS ONLY</p>
        <span>GHOST PACKET / 2026</span>
      </footer>

      <Toast.Root className="toast" open={Boolean(toast)} onOpenChange={(open) => !open && setToast(null)}>
        <Toast.Title>REQUEST UPDATE</Toast.Title>
        <Toast.Description>{toast}</Toast.Description>
        <Toast.Close aria-label="Dismiss"><X size={18} /></Toast.Close>
      </Toast.Root>
      <Toast.Viewport className="toast-viewport" />
    </Toast.Provider>
  );
}

function LoadingCard({ voice }: { voice: boolean }) {
  return (
    <motion.section className="loading-card" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}>
      <div className="loading-radar"><Search size={33} /><i /><i /><i /></div>
      <div><span className="section-tag">PROCESSING SIGNAL</span><h2>{voice ? "Transcribing your voice…" : "Searching the corpus…"}</h2><p>Retrieving, reranking, and checking the strongest passages.</p></div>
      <span className="loading-code">RAG / LIVE</span>
    </motion.section>
  );
}

function ResultCard({ result }: { result: Result }) {
  const copy = modeCopy[result.mode];
  const confidence = Math.round(result.confidence * 100);
  const stages = Object.entries(result.latency_ms).filter(([, value]) => typeof value === "number");
  const ModeIcon = result.mode === "generative" ? Sparkles : result.mode === "extractive" ? Zap : ShieldCheck;

  return (
    <motion.section
      className={`result-section ${result.mode}`}
      initial="hidden"
      animate="visible"
      variants={{ hidden: { opacity: 0, y: 28 }, visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.07 } } }}
    >
      <motion.header variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}>
        <div className="result-title"><span className="section-tag">{copy.eyebrow}</span><h2>{copy.label}</h2></div>
        <div className="mode-seal"><ModeIcon size={24} /><span>{result.mode}</span></div>
      </motion.header>

      {isVoice(result) && (
        <motion.div className="transcript" variants={{ hidden: { opacity: 0, x: -12 }, visible: { opacity: 1, x: 0 } }}>
          <span><Volume2 size={19} /></span><div><small>WHAT WE HEARD</small><p>{result.transcript}</p></div>
        </motion.div>
      )}

      <motion.div className="answer-panel" variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}>
        <div className="answer-index">A.</div>
        <p>{result.answer_text}</p>
      </motion.div>

      <motion.div className="result-metrics" variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}>
        <div className="confidence-block"><span>CONFIDENCE</span><strong>{confidence}%</strong><div><i style={{ width: `${confidence}%` }} /></div></div>
        <div><span>SOURCES</span><strong>{String(result.evidence.length).padStart(2, "0")}</strong><small>passages cited</small></div>
        <div><span>RESPONSE</span><strong>{formatMs(result.latency_ms.total_ms)}</strong><small>end to end</small></div>
        <div><span>TRACE</span><strong>#{result.trace_id.slice(0, 7)}</strong><small>request id</small></div>
      </motion.div>

      {result.guardrail_flags.length > 0 && <div className="flags">{result.guardrail_flags.map((flag) => <span key={flag}><ShieldCheck size={13} />{flag}</span>)}</div>}

      <div className="result-expanders">
        <Accordion.Root type="single" collapsible className="sources">
          <Accordion.Item value="sources">
            <Accordion.Header><Accordion.Trigger><span><FileText size={18} />SOURCE PASSAGES</span><span>{result.evidence.length} CITED <ChevronDown size={19} /></span></Accordion.Trigger></Accordion.Header>
            <Accordion.Content>
              {result.evidence.length ? result.evidence.map((evidence, index) => (
                <motion.article className="source" key={evidence.chunk_id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.05 }}>
                  <header><span>PASSAGE {String(index + 1).padStart(2, "0")}</span><code>{evidence.chunk_id}</code></header>
                  <p>{evidence.text}</p>
                  {evidence.rerank_score !== null && <small>RELEVANCE · {Math.round(evidence.rerank_score * 100)}%</small>}
                </motion.article>
              )) : <p className="empty-sources">No source passages were returned for this response.</p>}
            </Accordion.Content>
          </Accordion.Item>
        </Accordion.Root>

        <Collapsible.Root className="details">
          <Collapsible.Trigger><span><Clock3 size={18} />PIPELINE TIMING</span><span>{formatMs(result.latency_ms.total_ms)} <ChevronDown size={19} /></span></Collapsible.Trigger>
          <Collapsible.Content><div className="timings">{stages.map(([name, value], index) => <div key={name}><span>0{index + 1}</span><p>{name.replace("_ms", "").replace(/\b\w/g, (letter) => letter.toUpperCase())}</p><b>{formatMs(value as number)}</b></div>)}</div></Collapsible.Content>
        </Collapsible.Root>
      </div>
    </motion.section>
  );
}

function Pipeline({ health }: { health: "checking" | "ready" | "offline" }) {
  const steps = [
    { icon: Radio, title: "Capture", copy: "Native browser audio or typed input enters the pipeline." },
    { icon: Database, title: "Retrieve", copy: "Dense and sparse indexes surface the strongest evidence." },
    { icon: Search, title: "Rerank", copy: "The best passages are reordered by relevance." },
    { icon: ShieldCheck, title: "Ground", copy: "Guardrails decide whether to answer or refuse." },
  ];
  return (
    <article className="pipeline-card">
      <header><span>PIPELINE MAP</span><span className={`health ${health}`}><i />{health === "ready" ? "READY" : health === "checking" ? "CHECKING" : "OFFLINE"}</span></header>
      <ol>{steps.map(({ icon: Icon, title, copy }, index) => <li key={title}><span className="step-number">0{index + 1}</span><div className="step-icon"><Icon size={24} /></div><h3>{title}</h3><p>{copy}</p>{index < steps.length - 1 && <ArrowUpRight className="step-arrow" size={21} />}</li>)}</ol>
      <div className="pipeline-note"><Check size={16} /><span>ADAPTIVE CHUNKING</span><p>Sentence-aware windows preserve context before overlap is ever used.</p></div>
    </article>
  );
}

function History({ history, onReplay }: { history: HistoryItem[]; onReplay: (item: HistoryItem) => void }) {
  return (
    <article className="history-card">
      <header><span>SESSION LOG</span><RotateCcw size={19} /></header>
      {history.length ? <div className="history-list">{history.map((item, index) => (
        <button key={item.result.trace_id} onClick={() => onReplay(item)}>
          <span className={`history-mode ${item.result.mode}`}>{item.voice ? <Volume2 size={16} /> : <Send size={15} />}</span>
          <div><small>0{index + 1} / {item.result.mode}</small><strong>{item.query}</strong></div>
          <span className="history-time">{formatMs(item.result.latency_ms.total_ms)}<ArrowUpRight size={16} /></span>
        </button>
      ))}</div> : <div className="history-empty"><AudioLines size={38} /><h3>No signals yet.</h3><p>Your questions will appear here for one-click replay.</p></div>}
    </article>
  );
}

createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>,
);
