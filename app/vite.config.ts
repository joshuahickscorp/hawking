import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// HIDE is a pure web app talking to hide-serve over localhost HTTP/WS.
// No external network at build or runtime: fonts are vendored via npm (air-gap ethos).
export default defineConfig({
  plugins: [react()],
  server: { port: 5273, strictPort: false },
  // Monaco's editor + language services run off-thread; the workers are imported via `?worker`
  // (see app/src/surfaces/ide/monacoTheme.ts) and bundled as ES-module workers, no CDN fetch.
  worker: { format: "es" },
  // Bundle the editor core; exclude the worker entry subpaths from prebundling so the `?worker`
  // imports are handled as real worker modules, not flattened into the optimize step.
  optimizeDeps: {
    include: ["monaco-editor"],
    exclude: [
      "monaco-editor/esm/vs/editor/editor.worker",
      "monaco-editor/esm/vs/language/typescript/ts.worker",
      "monaco-editor/esm/vs/language/json/json.worker",
      "monaco-editor/esm/vs/language/css/css.worker",
      "monaco-editor/esm/vs/language/html/html.worker",
    ],
  },
  build: { target: "es2022", sourcemap: true },
});
