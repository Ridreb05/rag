import { ArrowUpRight, AudioLines, Check, Database, Radio, RotateCcw, Search, Send, ShieldCheck, Volume2 } from "lucide-react";
import { formatMs, type HistoryItem } from "../types";

type HealthState = "checking" | "ready" | "offline";

const healthCopy: Record<HealthState, string> = { ready: "Ready", checking: "Checking", offline: "Offline" };
const healthDotClasses: Record<HealthState, string> = {
  ready: "bg-clay-success animate-clay-breathe",
  checking: "bg-clay-warning",
  offline: "bg-clay-accentAlt",
};

const steps = [
  { icon: Radio, title: "Capture", copy: "Native browser audio or Hindi text enters the pipeline." },
  { icon: Database, title: "Retrieve", copy: "The deployment’s language index surfaces its strongest evidence." },
  { icon: Search, title: "Rerank", copy: "The best passages are reordered by relevance." },
  { icon: ShieldCheck, title: "Ground", copy: "Guardrails decide whether to answer or refuse." },
];

const iconOrbGradients = [
  "from-blue-400 to-blue-600",
  "from-purple-400 to-purple-600",
  "from-pink-400 to-pink-600",
  "from-emerald-400 to-emerald-600",
];

function Pipeline({ health }: { health: HealthState }) {
  return (
    <article className="overflow-hidden rounded-[32px] bg-white/70 shadow-clayCard backdrop-blur-xl">
      <header className="flex items-center justify-between px-6 py-5">
        <span className="font-body text-xs font-bold uppercase tracking-widest text-clay-foreground">Pipeline map</span>
        <span className="inline-flex items-center gap-2 rounded-full bg-clay-pressed px-3 py-1.5 font-body text-[10px] font-bold uppercase tracking-wide text-clay-foreground shadow-clayPressed">
          <i className={`block h-1.5 w-1.5 rounded-full ${healthDotClasses[health]}`} />
          {healthCopy[health]}
        </span>
      </header>
      <ol className="grid grid-cols-1 gap-4 px-6 pb-6 sm:grid-cols-2">
        {steps.map(({ icon: Icon, title, copy }, index) => (
          <li key={title} className="relative rounded-[24px] bg-white p-5 shadow-clayCard">
            <span className="font-body text-[10px] font-bold text-clay-accentAlt">0{index + 1}</span>
            <div className={`my-3 grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br text-white shadow-clayButton ${iconOrbGradients[index]}`}>
              <Icon size={22} />
            </div>
            <h3 className="font-heading text-xl font-extrabold text-clay-foreground">{title}</h3>
            <p className="mt-1.5 max-w-[220px] font-body text-xs leading-relaxed text-clay-muted">{copy}</p>
            {index < steps.length - 1 && <ArrowUpRight className="absolute right-4 top-4 text-clay-accent/40" size={18} />}
          </li>
        ))}
      </ol>
      <div className="mx-6 mb-6 flex items-center gap-3 rounded-2xl bg-clay-success/10 px-5 py-4">
        <Check size={16} className="shrink-0 text-clay-success" />
        <span className="font-body text-[10px] font-bold uppercase tracking-wide text-clay-foreground">Adaptive chunking</span>
        <p className="ml-auto text-right font-body text-[10px] text-clay-muted">
          Sentence-aware windows preserve context before overlap is ever used.
        </p>
      </div>
    </article>
  );
}

function History({ history, onReplay }: { history: HistoryItem[]; onReplay: (item: HistoryItem) => void }) {
  return (
    <article className="flex min-h-[420px] flex-col overflow-hidden rounded-[32px] bg-white/70 shadow-clayCard backdrop-blur-xl">
      <header className="flex items-center justify-between px-6 py-5">
        <span className="font-body text-xs font-bold uppercase tracking-widest text-clay-foreground">Session log</span>
        <RotateCcw size={18} className="text-clay-muted" />
      </header>
      {history.length ? (
        <div className="flex-1 px-4 pb-4">
          {history.map((item, index) => (
            <button
              key={item.result.trace_id}
              onClick={() => onReplay(item)}
              className="grid w-full grid-cols-[auto_1fr_auto] items-center gap-3 rounded-2xl px-3 py-3 text-left transition-colors duration-150 hover:bg-clay-accent/5"
            >
              <span
                className={`grid h-10 w-10 place-items-center rounded-full text-white ${
                  item.result.mode === "generative" ? "bg-gradient-to-br from-[#FBBF24] to-clay-warning" : "bg-gradient-to-br from-[#A78BFA] to-[#7C3AED]"
                }`}
              >
                {item.voice ? <Volume2 size={15} /> : <Send size={14} />}
              </span>
              <div className="min-w-0">
                <small className="block font-body text-[9px] font-bold uppercase tracking-wide text-clay-accentAlt">
                  0{index + 1} / {item.result.mode}
                </small>
                <strong className="block truncate font-script text-sm font-medium text-clay-foreground [direction:auto] [unicode-bidi:plaintext]">
                  {item.query}
                </strong>
              </div>
              <span className="flex items-center gap-1.5 font-body text-xs text-clay-muted">
                {formatMs(item.result.latency_ms.pipeline_ms ?? item.result.latency_ms.total_ms)}
                <ArrowUpRight size={14} />
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div className="grid flex-1 place-items-center gap-3 px-6 py-10 text-center">
          <AudioLines size={36} className="text-clay-accentAlt" />
          <h3 className="font-heading text-2xl font-extrabold text-clay-foreground">No signals yet.</h3>
          <p className="max-w-[220px] font-body text-xs leading-relaxed text-clay-muted">
            Your questions will appear here for one-click replay.
          </p>
        </div>
      )}
    </article>
  );
}

export function SystemSection({
  health,
  history,
  onReplay,
}: {
  health: HealthState;
  history: HistoryItem[];
  onReplay: (item: HistoryItem) => void;
}) {
  return (
    <section className="mx-auto w-[min(100%-32px,1280px)] py-16 sm:py-24">
      <div className="mb-10 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <p className="inline-flex w-fit items-center gap-2 rounded-full bg-white/70 px-4 py-1.5 font-body text-xs font-bold uppercase tracking-widest text-clay-accent shadow-clayCard backdrop-blur-xl">
          Under the hood <span className="text-clay-muted">/ evidence first</span>
        </p>
        <h2 className="max-w-lg font-heading text-3xl font-black leading-[1.05] tracking-tight text-clay-foreground sm:text-right sm:text-5xl">
          One question. <span className="text-clay-accentAlt">Four checkpoints.</span>
        </h2>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Pipeline health={health} />
        <History history={history} onReplay={onReplay} />
      </div>
    </section>
  );
}
