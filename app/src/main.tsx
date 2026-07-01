import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import "./styles/chat.css";
import "./styles/ide.css";
import "./styles/panels.css";
import { App } from "./App";
import { initGlass } from "./shell/glass";
import { ErrorBoundary } from "./shell/ErrorBoundary";

initGlass(); // engine-aware glass: sets data-glass and injects the refraction filter on Chromium-class

const root = document.getElementById("root");
if (!root) throw new Error("root element missing"); // surface the failure, never silently no-op
createRoot(root).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
