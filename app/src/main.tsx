import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
import "./styles/chat.css";
import "./styles/ide.css";
import "./styles/panels.css";
import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("root element missing"); // surface the failure, never silently no-op
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
