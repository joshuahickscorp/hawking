/*
  App.tsx: the workbench shell. Six regions (01-surfaces §A.1): title bar, activity/mode rail,
  the main stage that swaps the three surface frames, the persistent Context Stack right rail,
  the bottom status bar, and the Cmd+K command palette. Observation-first: the Workstation is the
  default front door. The shell connects the store to the live UiEvent stream on mount.
*/
import { useEffect, useMemo, useState } from "react";
import { connectStore, useStore } from "./store";
import { TRANSPORT_KIND } from "./ipc";
import { Chat } from "./surfaces/Chat";
import { ContextStack } from "./surfaces/ContextStack";
import { Ide } from "./surfaces/Ide";
import { Workstation } from "./surfaces/Workstation";
import {
  CommandPalette,
  EventHorizon,
  ModeSwitcher,
  StatusPill,
  type Command,
  type SurfaceMode,
} from "./ui";

export function App() {
  const [mode, setMode] = useState<SurfaceMode>("workstation"); // front door
  const [paletteOpen, setPaletteOpen] = useState(false);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const runPhase = useStore((s) => s.runPhase);
  const notices = useStore((s) => s.notices);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);

  // Connect the store to the live stream once, on mount.
  useEffect(() => connectStore(), []);

  // Cmd+K palette.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const commands = useMemo<Command[]>(
    () => [
      { id: "go.workstation", label: "Go to Workstation", run: () => setMode("workstation") },
      { id: "go.ide", label: "Go to IDE", run: () => setMode("ide") },
      { id: "go.chat", label: "Go to Chat", run: () => setMode("chat") },
    ],
    [],
  );

  const degraded = runtimeStatus === "degraded" || runtimeStatus === "failed" || runtimeStatus === "down";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: "var(--titlebar-h) 1fr var(--statusbar-h)",
        height: "100%",
      }}
    >
      {/* TITLE BAR */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s3)",
          padding: "0 var(--s4)",
          borderBottom: "1px solid var(--rim)",
          background: "var(--surface-0)",
          backgroundImage: "var(--panel-grad)",
        }}
      >
        <EventHorizon size={15} />
        {/* the wordmark is the ONLY place Geist appears */}
        <span style={{ fontFamily: "Geist, var(--font-mono)", fontWeight: 600, letterSpacing: "0.04em" }}>HIDE</span>
        <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>hawking</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "var(--s3)" }}>
          <StatusPill status={runtimeStatus} detail={runtimeDetail} />
          <button
            onClick={() => setPaletteOpen(true)}
            style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", padding: "2px 8px", boxShadow: "inset 0 0 0 1px var(--rim)", borderRadius: "var(--radius)" }}
          >
            ⌘K
          </button>
        </div>
      </header>

      {/* MAIN ROW: activity rail | stage | context rail */}
      <div style={{ display: "grid", gridTemplateColumns: "var(--activity-w) 1fr var(--rail-w)", minHeight: 0 }}>
        <aside style={{ borderRight: "1px solid var(--rim)", background: "var(--surface-0)" }}>
          <ModeSwitcher mode={mode} onMode={setMode} />
        </aside>

        <main style={{ minWidth: 0, minHeight: 0, position: "relative" }}>
          {degraded ? <DegradedBanner status={runtimeStatus} detail={runtimeDetail} /> : null}
          {mode === "workstation" ? <Workstation /> : mode === "ide" ? <Ide /> : <Chat />}
        </main>

        <aside style={{ borderLeft: "1px solid var(--rim)", background: "var(--surface-0)", minHeight: 0 }}>
          <ContextStack />
        </aside>
      </div>

      {/* STATUS BAR */}
      <footer
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s4)",
          padding: "0 var(--s4)",
          borderTop: "1px solid var(--rim)",
          background: "var(--surface-0)",
          fontSize: "var(--text-xs)",
          color: "var(--text-low)",
        }}
      >
        <span>agent: {runPhase}</span>
        <span style={{ marginLeft: "auto" }}>{TRANSPORT_KIND} transport</span>
        {notices.length ? (
          <span style={{ color: notices[notices.length - 1].kind === "error" ? "var(--danger)" : "var(--text-mid)" }}>
            {notices[notices.length - 1].message.slice(0, 80)}
          </span>
        ) : null}
      </footer>

      {/* SECURITY GATE: blocking, never auto-dismissed (FE-5) */}
      {gate ? <GatePrompt gate={gate.gate} message={gate.message} onDismiss={dismissGate} /> : null}

      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}

function DegradedBanner({ status, detail }: { status: string; detail: string | null }) {
  return (
    <div
      role="status"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 10,
        padding: "var(--s2) var(--s4)",
        background: "var(--surface-1)",
        boxShadow: "inset 0 0 0 1px var(--warning)",
        color: "var(--text-mid)",
        fontSize: "var(--text-xs)",
      }}
    >
      Local engine is {status}. {detail ?? "It may not be running."} Auto-restarting.
    </div>
  );
}

function GatePrompt({ gate, message, onDismiss }: { gate: string; message: string; onDismiss: () => void }) {
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(6,6,6,0.6)", display: "grid", placeItems: "center", zIndex: 200 }}>
      <div
        className="panel"
        style={{
          padding: "var(--s5)",
          width: "min(440px, 90vw)",
          animation: "radiation-breathe 2.6s ease-in-out infinite",
        }}
      >
        <div style={{ fontSize: "var(--text-xs)", color: "var(--radiation-bright)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Approval needed
        </div>
        <div style={{ margin: "var(--s3) 0", color: "var(--text-hi)" }}>{message}</div>
        <div style={{ display: "flex", gap: "var(--s2)", justifyContent: "flex-end" }}>
          <button onClick={onDismiss} style={{ padding: "6px 14px", color: "var(--text-mid)", boxShadow: "inset 0 0 0 1px var(--rim)", borderRadius: "var(--radius)" }}>
            Deny
          </button>
          <button
            onClick={onDismiss}
            title={gate}
            style={{ padding: "6px 14px", color: "var(--void)", background: "var(--radiation-bright)", borderRadius: "var(--radius)", boxShadow: "0 0 16px -4px var(--radiation-bloom)" }}
          >
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
