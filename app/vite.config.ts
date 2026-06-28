import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// HIDE is a pure web app talking to hide-serve over localhost HTTP/WS.
// No external network at build or runtime: fonts are vendored via npm (air-gap ethos).
export default defineConfig({
  plugins: [react()],
  server: { port: 5273, strictPort: false },
  // Monaco ships many language workers; we only bundle the editor core for the skeleton.
  optimizeDeps: { include: ["monaco-editor"] },
  build: { target: "es2022", sourcemap: true },
});
