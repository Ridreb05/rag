# Frontend build spec — Voice RAG

Build a real, polished single-page web app for a working Voice-Enabled RAG
system. The backend is already built, tested, and running — this spec
describes its real, exact API contract. Do not invent or guess
endpoints/fields; use what's documented below.

## Stack

- **React 18** + **TypeScript** (strict mode on)
- **Vite 6** — build tool, dev server
- **Tailwind CSS** — styling, with `darkMode: "media"` (respect
  `prefers-color-scheme`, no manual toggle needed)
- **Radix UI primitives** (`@radix-ui/react-*`) for accessible unstyled
  components — `Tooltip`, `Accordion` (evidence/sources list), `Collapsible`
  (latency details), `Toast` (errors), `Dialog` if useful — styled with
  Tailwind, not a pre-built theme
- **Framer Motion** — real, purposeful animation: response reveal,
  mode-badge transitions, a live "recording" pulse on the mic button,
  staggered evidence-list entry. Not decoration for its own sake, but this
  should feel like a crafted product, not a form with a submit button.
- **TanStack Query** (`@tanstack/react-query`) — `useMutation` for both
  `/v1/query` and `/v1/voice-query` (these are user-triggered actions, not
  cached GETs — `useMutation` is the correct primitive, not `useQuery`).
  Use it for loading/error/success state instead of hand-rolled `useState`
  juggling.
- Self-host any font you need (e.g. `@fontsource/noto-sans-devanagari` or
  similar npm package) — no `<link>` to Google Fonts or any other CDN.
  Same for icons: use an npm-installed icon set (e.g. `lucide-react`), not
  a CDN icon font.
- Package manager: npm (a plain `package-lock.json` is expected by the
  Docker build — see below).

## Where this lives and how it's built/served

- Project root: `frontend/` at the repo root (`frontend/package.json`,
  `frontend/vite.config.ts`, `frontend/src/`, `frontend/index.html` at the
  Vite-conventional location, etc.) — **replace everything currently in
  `frontend/` except this spec file isn't in there; the current
  `frontend/package.json` and `frontend/placeholder.html` are a stub, not
  a starting point — overwrite them.**
- Build command: `npm run build` must produce static output at
  `frontend/dist/` (Vite's default `outDir`, don't change it).
- The backend serves that build output directly: FastAPI mounts
  `StaticFiles(directory=".../frontend/dist", html=True)` at `/` — see the
  bottom of `src/voice_rag/apps/api_gateway/main.py`. No separate frontend
  host, no separate dev server in production.
- `Dockerfile` (already updated) builds the frontend in a `node:20-slim`
  stage (`npm ci && npm run build`) and copies only `frontend/dist` into
  the final CUDA/Python image — Node itself never ships. This means:
  **`npm ci` must succeed from `package.json` + `package-lock.json` alone**
  (commit the lockfile), and **`npm run build` must not require network
  access** beyond what `npm ci` already fetched (no runtime CDN calls, no
  build-time API calls).
- For local development, `npm run dev` (Vite dev server) should proxy
  `/v1/*` to the FastAPI backend (`vite.config.ts` → `server.proxy`) so
  `fetch("/v1/query")` works identically in dev and in the built app.
  Assume the backend runs on `http://localhost:8000` in dev.

## What this UI is for

The task is "Voice-Enabled RAG" — a query (typed or spoken) goes through
retrieval over an indexed Hindi corpus and comes back as a grounded,
cited answer, or an honest refusal if nothing relevant was found. The UI's
job is to make that loop demoable to someone who has never seen the code:
type or speak a question, see exactly what came back and why, and feel
like a real product while doing it.

No auth, no multi-user state, no persisted history/database — a
single-session demo tool. It's fine to keep an in-memory list of this
session's past queries in React state if that helps the UX (e.g. a small
history sidebar), but nothing needs to survive a page reload.

## API contract (real, from `src/voice_rag/apps/api_gateway/main.py`)

Base URL: same origin the page is served from (relative paths, e.g.
`fetch("/v1/query")`, or `http://localhost:8000` via the Vite dev proxy)
— never hardcode a production host.

Define these as TypeScript types and share them across the query and
voice-query hooks:

```ts
interface QueryRequest {
  query: string;
  language?: string; // default "hi" — the only language actually indexed
  top_k?: number;     // default 10
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
  confidence: number; // 0-1
  guardrail_flags: string[];
  evidence: EvidenceItem[];
  latency_ms: LatencyMs;
}

interface VoiceQueryResponse extends QueryResponse {
  transcript: string;
}
```

### `POST /v1/query` — text query

`application/json` body matching `QueryRequest` above. Returns
`QueryResponse`, `200`.

- `mode` is **the single most important field to surface visually** — it
  tells the user what kind of answer they got:
  - `refused` — no good answer found (off-topic, nothing relevant in the
    corpus, or unsafe input). `answer_text` is a refusal message. This is
    a correct, honest outcome, not an error — style it as neutral/muted,
    not as a failure state.
  - `extractive` — high-confidence direct match, answered without an LLM
    call (fast — near-retrieval-latency response).
  - `generative` — answer synthesized by an LLM from retrieved passages;
    this path has real, sometimes multi-second latency. Loading state
    must cover this comfortably (see UX flow).
- `evidence` — cited source passages backing the answer. Render as a
  Radix `Accordion` ("Sources", collapsed by default): passage text +
  rerank score per item. Empty array is normal for `refused`.
- `guardrail_flags` — empty when nothing triggered; show as small tags
  when present (e.g. `generation_declined`).
- `confidence` — 0–1, show as a percentage or a small visual indicator
  (e.g. a thin bar), not a raw decimal dumped on the page.
- `latency_ms` — per-stage timing breakdown. Put behind a Radix
  `Collapsible` ("Details" / a small expandable row) — this is real
  measured evidence of the system's speed and a specific grading
  criterion for this project, so make it available, not omitted, but it's
  secondary to the answer itself.

Errors:
- `400` — empty query or >2000 chars. Body: `{"detail": "..."}` — surface
  via a Radix `Toast`, not a browser `alert()`.
- `429` — rate limited (20 req/min/IP). Friendly "slow down" toast.
- `5xx` — generic "something went wrong, try again" toast.

### `POST /v1/voice-query` — spoken query

`multipart/form-data` (not JSON):
- `audio` (file, required) — send whatever `MediaRecorder` produces as-is
  (`audio/webm` in Chrome/Edge, `audio/mp4` in Safari — both are valid,
  the backend passes the real content-type through to STT; **do not
  transcode client-side**).
- `language` (form field, optional, default `"hi"`)
- `top_k` (form field, optional, default `10`)

Returns `VoiceQueryResponse` (= `QueryResponse` + `transcript: string`),
`200`. **Always show `transcript`** before/above the answer — voice
recognition is imperfect and the user needs to see what the system heard,
not just what it answered.

Errors: same as `/v1/query`, plus:
- `503` — voice input not configured on this deployment. Detail:
  `"Voice input unavailable: SARVAM_API_KEY not configured"`. On first
  503, disable the mic control for the rest of the session with a small
  explanatory note rather than letting the user retry into the same wall.
- `400` — empty audio upload or non-audio content-type.

### `GET /v1/health`

`{"status": "ok"}`. Optional: a small connectivity indicator, not
required.

## UX flow

1. **Text input** — a text box + submit (Enter-to-submit too). Drives a
   TanStack Query `useMutation` calling `POST /v1/query`.
2. **Voice input** — a mic button:
   - `navigator.mediaDevices.getUserMedia({ audio: true })`, record via
     `MediaRecorder` (default mime type — don't force one).
   - Framer Motion pulse/glow on the button while recording, so it's
     unambiguous the mic is live.
   - Click again (or auto-stop at 30s — Sarvam's sync endpoint is
     documented for files under 30s) to stop and submit via a second
     `useMutation` calling `POST /v1/voice-query`.
   - Handle mic-permission-denied with a clear inline message, not a
     silent failure.
3. **Loading state** — anywhere from ~100ms (`refused`/`extractive`) to
   several seconds (`generative`). Show immediately on submit
   (`isPending` from the mutation), animate it (Framer Motion), and keep
   it until settled. No fake progress bar with an implied ETA — an
   honest indeterminate indicator.
4. **Result display**, in priority order, animated in with Framer Motion
   (e.g. a staggered reveal, not everything popping in at once):
   - Mode badge (`refused` / `extractive` / `generative`), visually
     distinct per mode via color + label.
   - (Voice path only) the transcript, clearly labeled ("Heard:" or
     similar).
   - The answer text, prominent, largest text on the result.
   - Confidence indicator, small.
   - Guardrail flags, small tags, only if non-empty.
   - Evidence/sources — Radix `Accordion`, collapsed by default.
   - Latency breakdown — Radix `Collapsible`, collapsed by default.
5. **Errors** — Radix `Toast`, dismissible, inline near the input; never a
   browser `alert()`.

## Visual direction

A real, considered visual identity — not a bootstrap-default form. Some
latitude is yours; the following are constraints, not a full spec:
- Distinct, deliberate color treatment per `mode`: `refused` reads as
  calm/neutral (it's a correct outcome, not an error — don't use red for
  it), `extractive` and `generative` both read as "answered" but can be
  differentiated with a small visual distinction (icon or label is
  enough).
- Respect `prefers-color-scheme` for light/dark; both must be genuinely
  legible, not an inverted afterthought.
- Devanagari must render natively well — verify your chosen
  font stack actually covers it (test with real Hindi text, not just
  Latin placeholder copy).
- Responsive down to a single-column mobile layout — assume this gets
  tried from a phone.
- Motion should feel purposeful and quick (short durations, easing that
  matches a fast/technical product) — not slow, bouncy, or gratuitous.

## What NOT to build

- No login/auth, no user accounts, no server-persisted history/database.
- No language picker beyond acknowledging the deployment is Hindi-only.
- No client-side audio transcoding — send the browser's native recording
  format as-is.
- No CDN dependency of any kind at runtime (fonts, icons, scripts) —
  everything ships in the Vite bundle.
- No mocking of the API — this backend is real and running; build against
  the exact contract above, and if the running server's actual behavior
  ever disagrees with this spec, trust the running server.
