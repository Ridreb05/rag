import { MapPin } from "lucide-react";

export function ChallengeSection() {
  return (
    <section
      aria-label="Hacker House Goa challenge"
      className="mx-auto my-16 grid w-[min(100%-32px,1280px)] grid-cols-1 overflow-hidden rounded-[32px] bg-white/70 shadow-clayCard backdrop-blur-xl sm:rounded-[48px] lg:grid-cols-[1.1fr_0.9fr]"
    >
      <div className="relative flex min-h-[260px] items-center justify-center overflow-hidden bg-gradient-to-br from-clay-accentAlt via-clay-accentAlt/80 to-clay-accent transition-transform duration-700 hover:scale-105 lg:min-h-[510px]">
        <MapPin className="h-20 w-20 text-white/90 sm:h-28 sm:w-28" strokeWidth={1.5} aria-hidden="true" />
        <span className="absolute bottom-4 left-4 rounded-full bg-white/90 px-4 py-2 font-body text-[10px] font-bold uppercase tracking-widest text-clay-foreground shadow-clayCard backdrop-blur-xl">
          Built by the beach
        </span>
      </div>
      <div className="flex flex-col justify-center p-8 sm:p-12">
        <div className="mb-8 flex items-center justify-between gap-6">
          <strong className="font-heading text-2xl font-black tracking-tight text-clay-foreground">Hacker House</strong>
          <small className="font-body text-[10px] font-bold uppercase tracking-widest text-clay-muted">2:47 PM Studio</small>
        </div>
        <p className="mb-4 inline-flex w-fit items-center gap-2 rounded-full bg-clay-accent/10 px-4 py-1.5 font-body text-xs font-bold uppercase tracking-widest text-clay-accent">
          Challenge build <span className="text-clay-muted">/ task 02</span>
        </p>
        <h2 className="font-heading text-4xl font-black leading-[1.05] tracking-tight text-clay-foreground sm:text-5xl">
          Voice in. <span className="text-clay-accentAlt">Proof out.</span>
        </h2>
        <p className="mt-5 max-w-md font-body text-sm leading-relaxed text-clay-muted">
          ClearAsk was designed for multilingual voice RAG: script-aware input, fast retrieval, visible evidence, measured
          latency, and guardrails that know when to stay quiet.
        </p>
        <div className="mt-8 grid grid-cols-3 gap-4 border-t border-clay-accent/10 pt-6">
          <span className="font-body text-[9px] font-bold uppercase leading-relaxed tracking-wide text-clay-muted">
            <b className="mb-1 block font-heading text-2xl font-black text-clay-accentAlt">01</b>
            Voice query
          </span>
          <span className="font-body text-[9px] font-bold uppercase leading-relaxed tracking-wide text-clay-muted">
            <b className="mb-1 block font-heading text-2xl font-black text-clay-accentAlt">10</b>
            Top passages
          </span>
          <span className="font-body text-[9px] font-bold uppercase leading-relaxed tracking-wide text-clay-muted">
            <b className="mb-1 block font-heading text-2xl font-black text-clay-accentAlt">04</b>
            Checkpoints
          </span>
        </div>
      </div>
    </section>
  );
}
