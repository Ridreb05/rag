import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation } from "@tanstack/react-query";
import * as Toast from "@radix-ui/react-toast";
import { AnimatePresence, motion } from "framer-motion";
import "@fontsource/nunito/700.css";
import "@fontsource/nunito/800.css";
import "@fontsource/nunito/900.css";
import "@fontsource/dm-sans/400.css";
import "@fontsource/dm-sans/500.css";
import "@fontsource/dm-sans/700.css";
import "@fontsource/noto-sans-devanagari/400.css";
import "@fontsource/noto-sans-devanagari/500.css";
import "@fontsource/noto-sans-devanagari/700.css";
import "@fontsource/noto-sans-bengali/400.css";
import "@fontsource/noto-sans-gujarati/400.css";
import "@fontsource/noto-sans-gurmukhi/400.css";
import "@fontsource/noto-sans-kannada/400.css";
import "@fontsource/noto-sans-malayalam/400.css";
import "@fontsource/noto-sans-oriya/400.css";
import "@fontsource/noto-sans-tamil/400.css";
import "@fontsource/noto-sans-telugu/400.css";
import "@fontsource/noto-nastaliq-urdu/400.css";
import "./styles.css";

import { ApiError, refineAnswer, submitText, submitVoice } from "./api";
import type { HealthResponse, HistoryItem, Result, VoiceQueryResponse } from "./types";
import { Background } from "./components/Background";
import { Header } from "./components/Header";
import { QueryConsole } from "./components/QueryConsole";
import { LoadingCard } from "./components/LoadingCard";
import { ResultCard } from "./components/ResultCard";
import { SystemSection } from "./components/SystemSection";
import { Footer } from "./components/Footer";
import { ToastLayer } from "./components/ToastLayer";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [micDisabled, setMicDisabled] = useState(false);
  const [micMessage, setMicMessage] = useState("");
  const [health, setHealth] = useState<"checking" | "ready" | "offline">("checking");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [refining, setRefining] = useState(false);
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

    // Two-phase answering: the response above already met the latency budget,
    // so the generated upgrade is fetched afterwards and swapped in place.
    // Nothing is blocked on it — a failure or expiry silently leaves the
    // grounded answer already on screen, which is a complete answer in its
    // own right, not a placeholder.
    if (!next.refinement_available) return;
    const { trace_id: traceId } = next;
    setRefining(true);
    refineAnswer(traceId)
      .then((refined) => {
        const merged: Result = voice
          ? { ...refined, transcript: (next as VoiceQueryResponse).transcript }
          : refined;
        setResult((current) => (current?.trace_id === traceId ? merged : current));
        setHistory((current) =>
          current.map((item) => (item.result.trace_id === traceId ? { ...item, result: merged } : item)),
        );
      })
      .catch(() => undefined)
      .finally(() => setRefining(false));
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
    textMutation.mutate({ query: clean, top_k: 10 });
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
      <Background />
      <Header health={health} />

      <main id="top">
        {/* Ask and answer sit side by side from `lg` up, so a result is visible
            without scrolling away from the input that produced it. Below `lg`
            the columns collapse and the original stacked order is preserved. */}
        <div className="mx-auto grid w-[min(100%-32px,1500px)] grid-cols-1 items-start gap-6 py-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          {/* Sticky so the console stays put while a long answer is read. */}
          <div className="lg:sticky lg:top-6">
            <QueryConsole
              query={query}
              setQuery={setQuery}
              pending={pending}
              recording={recording}
              micDisabled={micDisabled}
              micMessage={micMessage}
              onSubmit={runText}
              onToggleRecording={toggleRecording}
            />
          </div>

          <div className="min-w-0">
            <AnimatePresence mode="wait">
              {pending && <LoadingCard voice={voiceMutation.isPending} />}
              {result && !pending && <ResultCard result={result} refining={refining} />}
              {!pending && !result && (
                <motion.div
                  key="answer-placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="grid min-h-[320px] place-items-center rounded-[32px] border-2 border-dashed border-clay-accent/20 bg-white/40 p-10 text-center backdrop-blur-xl sm:rounded-[40px]"
                >
                  <div className="max-w-sm">
                    <p className="font-heading text-xl font-extrabold text-clay-foreground">Answers appear here</p>
                    <p className="mt-2 font-body text-sm text-clay-muted">
                      Ask a question in Hindi — type it or use the mic. Every answer cites the passages it came
                      from, and the pipeline timing is shown against the 200&nbsp;ms budget.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <SystemSection
          health={health}
          history={history}
          onReplay={(item) => {
            setQuery(item.query);
            setResult(item.result);
            // Back to the console/answer pair. A fixed offset was right when
            // these were stacked; side by side, both live at the top.
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      </main>

      <Footer />

      <ToastLayer message={toast} onOpenChange={(open) => !open && setToast(null)} />
    </Toast.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>,
);
