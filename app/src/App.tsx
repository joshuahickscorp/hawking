/*
  App.tsx: the Shell (Doctrine v3). One grid: west wall | active chamber | east light-well.
  The west wall is the ModeRail (the three surfaces). The center Stage mounts the active
  surface. The east ContextStack is the light well, always present. Observation-first: the
  Workstation is the default front door. Volumes float in generous void; nothing touches an
  edge. The shell connects the store to the live UiEvent stream on mount.
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
  Gate,
  Mark,
  ModeRail,
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

  const degraded =
    runtimeStatus === "degraded" || runtimeStatus === "failed" || runtimeStatus === "down";

  return (
    <div
      className="shell"
      style={{
        display: "grid",
        gridTemplateColumns: "56px 1fr clamp(320px, 26vw, 380px)",
        height: "100vh",
        background: "var(--void)",
        color: "var(--text-1)",
        fontFamily: "var(--font)",
      }}
    >
      {/* WEST WALL: the quiet mode rail. A single shadow-line separates it from the chamber. */}
      <aside
        style={{
          display: "flex",
          flexDirection: "column",
          boxShadow: "inset -1px 0 0 0 var(--line)",
        }}
      >
        <div
          style={{
            height: 56,
            display: "grid",
            placeItems: "center",
          }}
          title="HIDE"
        >
          <Mark size={16} />
        </div>
        <ModeRail mode={mode} onMode={setMode} />
      </aside>

      {/* ACTIVE CHAMBER: the stage. Generous hero air; the surface floats in the void. */}
      <div style={{ display: "grid", gridTemplateRows: "1fr auto", minWidth: 0, minHeight: 0 }}>
        <main
          style={{
            position: "relative",
            minWidth: 0,
            minHeight: 0,
            overflow: "auto",
            padding: "var(--ma-14) var(--ma-18)",
          }}
        >
          {degraded ? <DegradedBanner status={runtimeStatus} detail={runtimeDetail} /> : null}
          {mode === "workstation" ? <Workstation /> : mode === "ide" ? <Ide /> : <Chat />}
        </main>

        {/* STATUS BAR: a low chalk line resting under the chamber. */}
        <footer
          className="t-micro"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--ma-6)",
            padding: "0 var(--ma-18)",
            height: 32,
            boxShadow: "inset 0 1px 0 0 var(--line)",
            color: "var(--text-3)",
          }}
        >
          <span>agent: {runPhase}</span>
          <span style={{ marginLeft: "auto" }}>{TRANSPORT_KIND} transport</span>
          {notices.length ? (
            <span
              style={{
                color:
                  notices[notices.length - 1].kind === "error" ? "var(--bad)" : "var(--text-2)",
              }}
            >
              {notices[notices.length - 1].message.slice(0, 80)}
            </span>
          ) : null}
        </footer>
      </div>

      {/* EAST LIGHT-WELL: the context stack, always present. */}
      <aside
        style={{
          minHeight: 0,
          overflow: "auto",
          boxShadow: "inset 1px 0 0 0 var(--line)",
          padding: "var(--ma-8) var(--ma-6)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--ma-3)",
            marginBottom: "var(--ma-8)",
          }}
        >
          {/* the wordmark is the ONLY place Geist Sans appears */}
          <span
            style={{
              fontFamily: '"Geist Sans", var(--font)',
              fontWeight: 600,
              letterSpacing: "0.04em",
              color: "var(--text-1)",
            }}
          >
            HIDE
          </span>
          <span className="t-micro">hawking</span>
          <div style={{ marginLeft: "auto" }}>
            <StatusPill status={runtimeStatus} detail={runtimeDetail} />
          </div>
        </div>
        <ContextStack />
      </aside>

      {/* SECURITY GATE: blocking, never auto-dismissed (FE-5) */}
      {gate ? <GatePrompt message={gate.message} gateId={gate.gate} onDismiss={dismissGate} /> : null}

      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}

function DegradedBanner({ status, detail }: { status: string; detail: string | null }) {
  return (
    <div
      role="status"
      className="t-micro"
      style={{
        marginBottom: "var(--ma-8)",
        padding: "var(--ma-3) var(--ma-4)",
        borderRadius: "var(--radius)",
        background: "var(--concrete-2)",
        boxShadow: "var(--hairline-strong)",
        color: "var(--text-2)",
      }}
    >
      Local engine is {status}. {detail ?? "It may not be running."} Auto restarting.
    </div>
  );
}

function GatePrompt({
  message,
  gateId,
  onDismiss,
}: {
  message: string;
  gateId: string;
  onDismiss: () => void;
}) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(7,7,7,0.62)",
        display: "grid",
        placeItems: "center",
        zIndex: 200,
      }}
    >
      <div
        className="volume alive"
        style={{ padding: "var(--ma-8)", width: "min(440px, 90vw)" }}
      >
        <div className="t-label" style={{ color: "var(--light)" }}>
          Approval needed
        </div>
        <div className="t-body" style={{ margin: "var(--ma-4) 0", color: "var(--text-1)" }}>
          {message}
        </div>
        <div style={{ display: "flex", gap: "var(--ma-3)", justifyContent: "flex-end" }}>
          <button
            className="t-body"
            onClick={onDismiss}
            style={{
              padding: "var(--ma-2) var(--ma-4)",
              color: "var(--text-2)",
              boxShadow: "var(--hairline)",
              borderRadius: "var(--radius-pill)",
            }}
          >
            Deny
          </button>
          <Gate onClick={onDismiss} title={gateId}>
            Approve
          </Gate>
        </div>
      </div>
    </div>
  );
}
