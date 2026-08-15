import { AudioLines } from "lucide-react";

type HealthState = "checking" | "ready" | "offline";

const healthCopy: Record<HealthState, string> = {
  ready: "System live",
  checking: "Checking",
  offline: "Offline",
};

const healthDotClasses: Record<HealthState, string> = {
  ready: "bg-clay-success shadow-[0_0_0_5px_rgba(16,185,129,0.18)] animate-clay-breathe",
  checking: "bg-clay-warning",
  offline: "bg-clay-accentAlt",
};

export function Header({ health }: { health: HealthState }) {
  return (
    <header className="sticky top-4 z-20 mx-auto flex h-16 w-[min(100%-32px,1200px)] items-center justify-between rounded-[32px] bg-white/70 px-4 shadow-clayCard backdrop-blur-xl sm:h-20 sm:rounded-[40px] sm:px-8">
      <a href="#top" aria-label="ClearAsk home" className="inline-flex items-center gap-3 text-clay-foreground no-underline">
        <span className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white shadow-clayButton">
          <AudioLines size={22} />
        </span>
        <span className="hidden flex-col leading-none sm:flex">
          <strong className="font-heading text-lg font-black tracking-tight">ClearAsk</strong>
          <small className="font-body text-[10px] font-bold uppercase tracking-widest text-clay-muted">Voice RAG</small>
        </span>
      </a>
      <div className="flex items-center gap-3 sm:gap-6">
        <span className="hidden font-body text-xs font-bold uppercase tracking-widest text-clay-muted md:inline">
          Hindi voice RAG
        </span>
        <span className="inline-flex items-center gap-2 rounded-full bg-clay-pressed px-4 py-2 font-body text-xs font-bold uppercase tracking-wide text-clay-foreground shadow-clayPressed">
          <i className={`block h-2 w-2 rounded-full ${healthDotClasses[health]}`} />
          {healthCopy[health]}
        </span>
      </div>
    </header>
  );
}
