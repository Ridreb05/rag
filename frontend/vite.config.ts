import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy target. Defaults to a local backend; set VITE_API_PROXY to point
// the dev server at a running GPU Pod instead, which is the only way to see
// real answers (and therefore real result-card layout) without a local GPU:
//   VITE_API_PROXY=https://<pod-id>-8000.proxy.runpod.net npm run dev
// changeOrigin is required for a remote target — RunPod's proxy routes on the
// Host header and rejects the request without it.
export default defineConfig(({ mode }) => {
  // loadEnv rather than process.env: no @types/node dependency, and it picks up
  // .env.local too, so the pod URL can live in a gitignored file instead of
  // being retyped on every run.
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_PROXY || "http://localhost:8000";

  return {
    plugins: [react()],
    server: { proxy: { "/v1": { target, changeOrigin: true } } },
  };
});
