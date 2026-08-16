import { ArrowUpRight, CircleAlert, Mic, Square } from "lucide-react";
import { ClayButton } from "./ui/ClayButton";

const prompts = [
  { language: "Hindi", text: "भारत का राष्ट्रीय खेल क्या है?" },
  { language: "Hindi", text: "इंटरनेट कैसे काम करता है?" },
  { language: "Hindi", text: "फ़ायरवॉल का उद्देश्य क्या है?" },
  { language: "Hindi", text: "भारत की राजधानी क्या है?" },
];

interface QueryConsoleProps {
  query: string;
  setQuery: (value: string) => void;
  pending: boolean;
  recording: boolean;
  micDisabled: boolean;
  micMessage: string;
  onSubmit: (event: React.FormEvent) => void;
  onToggleRecording: () => void;
}

export function QueryConsole({
  query,
  setQuery,
  pending,
  recording,
  micDisabled,
  micMessage,
  onSubmit,
  onToggleRecording,
}: QueryConsoleProps) {
  return (
    <section className="w-full" aria-label="Ask the knowledge corpus">
      <div className="mb-4 flex items-center gap-3 font-body text-xs font-bold uppercase tracking-widest text-clay-muted">
        <span className="text-clay-accent">Live console</span>
        <i className="h-px flex-1 bg-clay-accent/15" />
        <small>Hindi input · Top 10</small>
      </div>

      <form
        onSubmit={onSubmit}
        className="rounded-[32px] bg-white/70 p-6 shadow-clayCard backdrop-blur-xl sm:rounded-[40px] sm:p-10"
      >
        <label htmlFor="question" className="mb-4 block font-heading text-2xl font-extrabold text-clay-foreground sm:text-3xl">
          What do you want to know?
        </label>
        <div className="relative rounded-2xl bg-clay-pressed shadow-clayPressed transition-all duration-200 focus-within:bg-white focus-within:shadow-clayCard focus-within:ring-4 focus-within:ring-clay-accent/20">
          <textarea
            id="question"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSubmit(event);
              }
            }}
            maxLength={2000}
            placeholder="Ask in Hindi…"
            rows={3}
            disabled={pending}
            className="block w-full resize-y rounded-2xl border-0 bg-transparent px-6 py-5 pr-20 font-script text-lg text-clay-foreground outline-none placeholder:text-clay-muted disabled:opacity-60"
            style={{ minHeight: "128px" }}
          />
          <span className="absolute bottom-3 right-5 font-body text-[10px] font-bold text-clay-muted">{query.length}/2000</span>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <button
            type="button"
            onClick={onToggleRecording}
            disabled={pending || micDisabled}
            className={`flex min-h-[76px] items-center gap-4 rounded-2xl px-5 text-left font-body transition-all duration-200 disabled:opacity-50 ${
              recording
                ? "bg-gradient-to-br from-clay-accentAlt to-[#BE185D] text-white shadow-clayButton"
                : "bg-white text-clay-foreground shadow-clayCard hover:-translate-y-1 hover:shadow-clayCardHover"
            }`}
          >
            <span
              className={`grid h-12 w-12 shrink-0 place-items-center rounded-full ${
                recording ? "animate-clay-mic-pulse bg-white text-clay-accentAlt" : "bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white"
              }`}
            >
              {recording ? <Square size={18} fill="currentColor" /> : <Mic size={22} />}
            </span>
            <span>
              <strong className="block font-heading text-sm font-extrabold uppercase tracking-wide">
                {recording ? "Stop & search" : "Ask with voice"}
              </strong>
              <small className="mt-1 block text-xs opacity-75">{recording ? "Listening now…" : "Up to 30 seconds"}</small>
            </span>
          </button>

          <ClayButton type="submit" size="lg" disabled={!query.trim() || pending} className="min-h-[76px] justify-between">
            <span>{pending ? "Searching" : "Search corpus"}</span>
            <ArrowUpRight size={22} />
          </ClayButton>
        </div>

        {micMessage && (
          <p className="mt-4 flex items-start gap-2 font-body text-sm leading-relaxed text-clay-accentAlt">
            <CircleAlert size={16} className="mt-0.5 shrink-0" />
            {micMessage}
          </p>
        )}
      </form>

      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-[140px_1fr] sm:items-start">
        <span className="pt-2 font-body text-xs font-bold uppercase tracking-widest text-clay-accentAlt">Try a signal</span>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {prompts.map((prompt, index) => (
            <button
              key={prompt.text}
              onClick={() => setQuery(prompt.text)}
              className="flex min-h-[84px] items-start gap-3 rounded-[20px] bg-white/70 p-4 text-left shadow-clayCard backdrop-blur-xl transition-all duration-200 hover:-translate-y-1 hover:shadow-clayCardHover"
            >
              <b className="font-heading text-sm font-black text-clay-accentAlt">0{index + 1}</b>
              <span className="font-script text-sm text-clay-foreground [direction:auto] [unicode-bidi:plaintext]">
                {prompt.text}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 flex items-center gap-3 border-t border-clay-accent/10 pt-6">
        <span className="shrink-0 font-body text-xs font-bold uppercase tracking-widest text-clay-muted">Live language</span>
        <b className="rounded-full bg-clay-accent/8 px-4 py-1.5 font-script text-sm font-normal text-clay-foreground shadow-[inset_0_0_0_1px_rgba(124,58,237,0.12)]">
          हिन्दी · Hindi
        </b>
      </div>
    </section>
  );
}
