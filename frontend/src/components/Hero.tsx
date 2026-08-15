import { Languages, Waves } from "lucide-react";

export function Hero() {
  return (
    <section className="relative mx-auto grid w-[min(100%-32px,1280px)] grid-cols-1 items-center gap-10 py-16 sm:py-20 lg:grid-cols-[1.1fr_0.9fr] lg:gap-16 lg:py-28">
      <div className="relative z-10">
        <p className="mb-6 inline-flex items-center gap-2 rounded-full bg-white/70 px-5 py-2 font-body text-xs font-bold uppercase tracking-widest text-clay-accent shadow-clayCard backdrop-blur-xl">
          Task 02 <span className="text-clay-muted">/ Multilingual knowledge engine</span>
        </p>
        <h1
          className="font-heading text-5xl font-black leading-[1.05] tracking-tight text-clay-foreground sm:text-6xl md:text-7xl lg:text-8xl"
        >
          Ask the corpus.{" "}
          <span className="bg-gradient-to-br from-clay-foreground via-clay-accent to-clay-accentAlt bg-clip-text text-transparent">
            Get the signal.
          </span>
        </h1>
        <div className="mt-10 flex flex-col items-start gap-8 sm:flex-row sm:items-end">
          <p className="max-w-md font-body text-base leading-relaxed text-clay-muted sm:text-lg">
            Speak or type naturally. ClearAsk searches the configured language corpus, checks the answer, and shows exactly
            what it found.
          </p>
          <div className="flex items-start gap-3 rounded-[24px] bg-white/70 px-5 py-4 shadow-clayCard backdrop-blur-xl">
            <span className="font-heading text-3xl font-black text-clay-accentAlt">01</span>
            <small className="pt-1 font-body text-[10px] font-bold uppercase leading-relaxed tracking-wide text-clay-muted">
              Voice → evidence
              <br />
              No noise. No guessing.
            </small>
          </div>
        </div>
      </div>

      <figure className="relative mx-auto w-full max-w-md lg:max-w-none">
        <div className="animate-clay-float-slow flex h-[280px] w-full items-center justify-center overflow-hidden rounded-[32px] bg-gradient-to-br from-clay-accent via-clay-accent/80 to-clay-accentAlt shadow-clayCard sm:h-[360px] sm:rounded-[40px] lg:h-[430px]">
          <Waves className="h-20 w-20 text-white/90 sm:h-28 sm:w-28" strokeWidth={1.5} aria-hidden="true" />
        </div>
        <figcaption className="absolute bottom-4 left-4 right-4 flex items-center justify-between gap-3 rounded-[20px] bg-white/85 px-4 py-3 font-body text-[10px] font-bold uppercase tracking-widest text-clay-foreground shadow-clayCard backdrop-blur-xl">
          <span>Goa / India</span>
          <small className="text-clay-muted">Voice → evidence</small>
        </figcaption>
        <div className="absolute -left-6 -top-6 hidden h-24 w-24 animate-clay-float place-items-center rounded-full bg-white p-3 shadow-clayButton sm:grid" aria-hidden="true">
          <Languages className="h-full w-full text-clay-accent" strokeWidth={1.5} />
        </div>
      </figure>
    </section>
  );
}
