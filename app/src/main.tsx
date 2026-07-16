import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import "./chat.css";
import "./ide.css";
import "./panels.css";
import "./home.css";
import { App } from "./App";
import { useStore } from "./store";
import { initGlass } from "./shell/policies";
import { ErrorBoundary } from "./shell/components";

// Errors must never vanish, but framework-internal noise must never scream at the user either. In a
// bundled .app there is no console, so surface *actionable* uncaught errors in the notices strip and
// drop the benign churn (Monaco model-disposal races, ResizeObserver loops, cross-origin script errors).
if (typeof window !== "undefined") {
  // Known-benign framework noise: not user-actionable, so it must not become a red notice.
  const BENIGN =
    /TextModel got disposed|DiffEditorWidget|ResizeObserver loop|Non-Error promise rejection|^Script error\.?$|Load failed$/i;
  let last = "";
  const notice = (message: string) => {
    if (!message || BENIGN.test(message)) return;
    if (message === last) return; // dedupe an error that fires in a tight loop
    last = message;
    try {
      useStore.getState().pushNotice({ kind: "error", code: "uncaught", message: message.slice(0, 160) });
    } catch {
      /* notice bus not ready yet */
    }
  };
  window.addEventListener("error", (e) => notice(e.message || "uncaught error"));
  window.addEventListener("unhandledrejection", (e) => {
    const r = e.reason as unknown;
    notice(typeof r === "string" ? r : (r as { message?: string })?.message || "unhandled rejection");
  });
}

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
