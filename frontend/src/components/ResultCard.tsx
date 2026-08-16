import * as Accordion from "@radix-ui/react-accordion";
import * as Collapsible from "@radix-ui/react-collapsible";
import { motion } from "framer-motion";
import { ChevronDown, Clock3, FileText, ShieldCheck, Sparkles, Volume2, Zap } from "lucide-react";
import { formatMs, isVoice, SUMMARY_LATENCY_KEYS, type AnswerMode, type Result } from "../types";

const modeCopy: Record<AnswerMode, { label: string; eyebrow: string }> = {
  refused: { label: "No grounded answer", eyebrow: "Signal not found" },
  extractive: { label: "Direct match", eyebrow: "Corpus extract" },
  generative: { label: "Grounded synthesis", eyebrow: "Answer assembled" },
};

const modeSealClasses: Record<AnswerMode, string> = {
  refused: "bg-gradient-to-br from-[#9CA3AF] to-[#6B7280]",
  extractive: "bg-gradient-to-br from-[#A78BFA] to-[#7C3AED]",
  generative: "bg-gradient-to-br from-[#FBBF24] to-clay-warning",
};

export function ResultCard({ result, refining = false }: { result: Result; refining?: boolean }) {
  const copy = modeCopy[result.mode];
  const confidence = Math.round(result.confidence * 100);
  // Only `_ms` keys are durations. latency_ms also carries diagnostic counts
  // (bm25_recovered), which rendered through formatMs would invent a timing
  // that does not exist — "2 ms" for what is actually "2 chunks".
  const stages = Object.entries(result.latency_ms).filter(
    ([name, value]) => typeof value === "number" && name.endsWith("_ms") && !SUMMARY_LATENCY_KEYS.includes(name),
  );
  const ModeIcon = result.mode === "generative" ? Sparkles : result.mode === "extractive" ? Zap : ShieldCheck;

  // The task's <200ms target covers this system's own work; Sarvam's STT round
  // trip is a third-party call, reported alongside rather than inside it.
  const budgetMs = result.latency_ms.budget_ms;
  const pipelineMs = result.latency_ms.pipeline_ms ?? result.latency_ms.total_ms;
  const sttMs = result.latency_ms.stt_ms;
  const withinBudget = budgetMs !== undefined && pipelineMs <= budgetMs;
  const budgetUsedPct = budgetMs ? Math.min(100, (pipelineMs / budgetMs) * 100) : 0;

  return (
    <motion.section
      initial="hidden"
      animate="visible"
      variants={{ hidden: { opacity: 0, y: 28 }, visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.07 } } }}
      className="mx-auto my-10 w-[min(100%-32px,1200px)] overflow-hidden rounded-[32px] bg-white/80 shadow-clayCard backdrop-blur-xl sm:rounded-[40px]"
    >
      <motion.header
        variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}
        className="flex items-center justify-between gap-4 p-6 sm:p-8"
      >
        <div>
          <span className="font-body text-xs font-bold uppercase tracking-widest text-clay-accentAlt">{copy.eyebrow}</span>
          <h2 className="mt-1 font-heading text-2xl font-extrabold text-clay-foreground sm:text-4xl">{copy.label}</h2>
        </div>
        <div className={`grid h-16 w-16 shrink-0 place-items-center gap-1 rounded-full text-white shadow-clayButton sm:h-20 sm:w-20 ${modeSealClasses[result.mode]}`}>
          <ModeIcon size={22} />
          <span className="font-body text-[8px] font-bold uppercase tracking-widest">{result.mode}</span>
        </div>
      </motion.header>

      {isVoice(result) && (
        <motion.div
          variants={{ hidden: { opacity: 0, x: -12 }, visible: { opacity: 1, x: 0 } }}
          className="mx-6 flex items-start gap-3 rounded-2xl bg-clay-accent/8 p-4 sm:mx-8"
        >
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white">
            <Volume2 size={18} />
          </span>
          <div>
            <small className="font-body text-[10px] font-bold uppercase tracking-widest text-clay-accent">What we heard</small>
            <p className="mt-1 font-script text-base text-clay-foreground [direction:auto] [unicode-bidi:plaintext]">
              {result.transcript}
            </p>
          </div>
        </motion.div>
      )}

      <motion.div
        variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}
        className="flex gap-4 p-6 sm:gap-6 sm:p-8"
      >
        <div className="shrink-0 font-heading text-3xl font-black text-clay-accentAlt sm:text-4xl">A.</div>
        <p className="max-w-3xl font-script text-xl leading-relaxed text-clay-foreground [direction:auto] [unicode-bidi:plaintext] sm:text-2xl">
          {result.answer_text}
        </p>
      </motion.div>

      <motion.div
        variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}
        className="grid grid-cols-2 gap-px overflow-hidden rounded-[24px] bg-clay-accent/10 sm:mx-8 sm:mb-2 sm:grid-cols-4"
      >
        <div className="bg-white p-5">
          <span className="font-body text-[9px] font-bold uppercase tracking-widest text-clay-muted">Confidence</span>
          <strong className="mt-2 block font-heading text-3xl font-black text-clay-foreground">{confidence}%</strong>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-clay-accent/10">
            <div className="h-full rounded-full bg-gradient-to-r from-[#A78BFA] to-[#7C3AED]" style={{ width: `${confidence}%` }} />
          </div>
        </div>
        <div className="bg-white p-5">
          <span className="font-body text-[9px] font-bold uppercase tracking-widest text-clay-muted">Sources</span>
          <strong className="mt-2 block font-heading text-3xl font-black text-clay-foreground">
            {String(result.evidence.length).padStart(2, "0")}
          </strong>
          <small className="font-body text-xs text-clay-muted">passages cited</small>
        </div>
        <div className="bg-white p-5">
          <span className="font-body text-[9px] font-bold uppercase tracking-widest text-clay-muted">Pipeline</span>
          <strong
            className={`mt-2 block font-heading text-3xl font-black ${
              budgetMs === undefined ? "text-clay-foreground" : withinBudget ? "text-clay-success" : "text-clay-warning"
            }`}
          >
            {formatMs(pipelineMs)}
          </strong>
          {budgetMs === undefined ? (
            <small className="font-body text-xs text-clay-muted">end to end</small>
          ) : (
            <>
              <small className="font-body text-xs text-clay-muted">
                {withinBudget ? "within" : "over"} {formatMs(budgetMs)} budget
              </small>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-clay-accent/10">
                <div
                  className={`h-full rounded-full ${withinBudget ? "bg-clay-success" : "bg-clay-warning"}`}
                  style={{ width: `${budgetUsedPct}%` }}
                />
              </div>
            </>
          )}
          {sttMs !== undefined && (
            <small className="mt-2 block font-body text-[10px] text-clay-muted">
              + {formatMs(sttMs)} Sarvam STT (external)
            </small>
          )}
        </div>
        <div className="bg-white p-5">
          <span className="font-body text-[9px] font-bold uppercase tracking-widest text-clay-muted">Trace</span>
          <strong className="mt-2 block font-heading text-3xl font-black text-clay-foreground">
            #{result.trace_id.slice(0, 7)}
          </strong>
          <small className="font-body text-xs text-clay-muted">request id</small>
        </div>
      </motion.div>

      {refining && (
        <div className="mx-6 mt-6 flex items-center gap-3 rounded-2xl bg-clay-accent/10 px-5 py-3 sm:mx-8">
          <Sparkles size={16} className="shrink-0 animate-pulse text-clay-accent" />
          <p className="font-body text-xs text-clay-foreground">
            <b>Answered within budget.</b> Generating a synthesized answer from the same evidence — this
            replaces the extract above when it arrives.
          </p>
        </div>
      )}

      {result.guardrail_flags.length > 0 && (
        <div className="flex flex-wrap gap-2 px-6 pt-6 sm:px-8">
          {result.guardrail_flags.map((flag) => (
            <span
              key={flag}
              className="inline-flex items-center gap-1.5 rounded-full bg-clay-warning/15 px-3 py-1.5 font-body text-[10px] font-bold uppercase text-[#92400E]"
            >
              <ShieldCheck size={12} />
              {flag}
            </span>
          ))}
        </div>
      )}

      <div className="p-6 sm:p-8">
        <Accordion.Root type="single" collapsible className="overflow-hidden rounded-2xl bg-clay-pressed shadow-clayPressed">
          <Accordion.Item value="sources">
            <Accordion.Header>
              <Accordion.Trigger className="group flex w-full items-center justify-between gap-4 px-5 py-4 font-body text-xs font-bold uppercase tracking-widest text-clay-foreground">
                <span className="flex items-center gap-2">
                  <FileText size={16} />
                  Source passages
                </span>
                <span className="flex items-center gap-2 text-clay-accent">
                  {result.evidence.length} cited
                  <ChevronDown size={16} className="transition-transform duration-200 group-data-[state=open]:rotate-180" />
                </span>
              </Accordion.Trigger>
            </Accordion.Header>
            <Accordion.Content className="px-5 pb-5">
              {result.evidence.length ? (
                result.evidence.map((evidence, index) => (
                  <motion.article
                    key={evidence.chunk_id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="mb-3 rounded-2xl bg-white p-5 shadow-clayCard last:mb-0"
                  >
                    <header className="flex flex-wrap items-center justify-between gap-2 font-body text-[10px] font-bold uppercase tracking-wide text-clay-accent">
                      <span>Passage {String(index + 1).padStart(2, "0")}</span>
                      <code className="max-w-full truncate text-clay-muted">{evidence.chunk_id}</code>
                    </header>
                    <p className="mt-3 font-script text-base leading-relaxed text-clay-foreground [direction:auto] [unicode-bidi:plaintext]">
                      {evidence.text}
                    </p>
                    {evidence.rerank_score !== null && (
                      <small className="mt-3 block font-body text-[10px] font-bold uppercase text-clay-accentAlt">
                        Relevance · {Math.round(evidence.rerank_score * 100)}%
                      </small>
                    )}
                  </motion.article>
                ))
              ) : (
                <p className="font-body text-sm text-clay-muted">No source passages were returned for this response.</p>
              )}
            </Accordion.Content>
          </Accordion.Item>
        </Accordion.Root>

        <Collapsible.Root className="mt-4 overflow-hidden rounded-2xl bg-clay-pressed shadow-clayPressed">
          <Collapsible.Trigger className="group flex w-full items-center justify-between gap-4 px-5 py-4 font-body text-xs font-bold uppercase tracking-widest text-clay-foreground">
            <span className="flex items-center gap-2">
              <Clock3 size={16} />
              Pipeline timing
            </span>
            <span className="flex items-center gap-2 text-clay-accent">
              <span className={budgetMs === undefined ? "" : withinBudget ? "text-clay-success" : "text-clay-warning"}>
                {formatMs(pipelineMs)}
                {budgetMs !== undefined && ` / ${formatMs(budgetMs)}`}
              </span>
              <ChevronDown size={16} className="transition-transform duration-200 group-data-[state=open]:rotate-180" />
            </span>
          </Collapsible.Trigger>
          <Collapsible.Content className="grid grid-cols-2 gap-3 px-5 pb-5 sm:grid-cols-3 lg:grid-cols-6">
            {stages.map(([name, value], index) => (
              <div key={name} className="rounded-xl bg-white p-4 shadow-clayCard">
                <span className="font-body text-[9px] font-bold text-clay-accentAlt">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <p className="mt-1 font-body text-[10px] font-bold uppercase text-clay-muted">
                  {name.replace(/_ms$/, "").replace(/_/g, " ")}
                </p>
                <b className="mt-1 block font-heading text-lg font-black text-clay-foreground">{formatMs(value as number)}</b>
              </div>
            ))}
          </Collapsible.Content>
        </Collapsible.Root>
      </div>
    </motion.section>
  );
}
