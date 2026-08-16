import { motion } from "framer-motion";
import { Search } from "lucide-react";

export function LoadingCard({ voice }: { voice: boolean }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      className="mx-auto my-10 flex w-[min(100%-32px,1200px)] flex-col items-center gap-6 rounded-[32px] bg-white/70 p-8 text-center shadow-clayCard backdrop-blur-xl sm:flex-row sm:gap-8 sm:p-10 sm:text-left"
    >
      <div className="relative grid h-24 w-24 shrink-0 place-items-center rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white shadow-clayButton">
        <Search size={30} />
        <i className="absolute inset-3 rounded-full border border-white/60 [animation:clay-radar_1.6s_ease-out_infinite]" />
        <i className="absolute inset-3 rounded-full border border-white/60 [animation:clay-radar_1.6s_ease-out_infinite] [animation-delay:0.4s]" />
        <i className="absolute inset-3 rounded-full border border-white/60 [animation:clay-radar_1.6s_ease-out_infinite] [animation-delay:0.8s]" />
      </div>
      <div>
        <span className="font-body text-xs font-bold uppercase tracking-widest text-clay-accentAlt">Processing signal</span>
        <h2 className="mt-2 font-heading text-3xl font-extrabold text-clay-foreground sm:text-4xl">
          {voice ? "Transcribing your voice…" : "Searching the corpus…"}
        </h2>
        <p className="mt-2 font-body text-sm text-clay-muted">Retrieving, reranking, and checking the strongest passages.</p>
      </div>
    </motion.section>
  );
}
