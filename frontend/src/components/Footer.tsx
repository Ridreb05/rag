import { AudioLines } from "lucide-react";

export function Footer() {
  return (
    <footer className="relative mt-16 overflow-hidden bg-gradient-to-r from-clay-accent/10 via-white/40 to-clay-accentAlt/10">
      <div className="mx-auto flex w-[min(100%-32px,1200px)] flex-wrap items-center justify-between gap-4 py-8">
        <div className="inline-flex items-center gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white shadow-clayButton">
            <AudioLines size={20} />
          </span>
          <span className="flex flex-col leading-none">
            <strong className="font-heading text-base font-black text-clay-foreground">ClearAsk</strong>
            <small className="font-body text-[9px] font-bold uppercase tracking-widest text-clay-muted">Voice RAG</small>
          </span>
        </div>
        <p className="font-body text-xs font-bold uppercase tracking-wide text-clay-foreground">
          Built for <b className="text-clay-accentAlt">#RAGInGoa</b> · Grounded answers only
        </p>
        <span className="font-body text-[10px] font-bold uppercase tracking-widest text-clay-muted">Ghost Packet / 2026</span>
      </div>
    </footer>
  );
}
