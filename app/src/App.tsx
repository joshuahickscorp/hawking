import { useEffect, useMemo, useState } from "react";
import { connectStore, useStore } from "./store";
import { TRANSPORT_KIND } from "./ipc";
import { Chat } from "./surfaces/Chat";
import { ContextStack } from "./surfaces/ContextStack";
import { Ide } from "./surfaces/Ide";
import { Workstation } from "./surfaces/Workstation";
import { CommandPalette, Gate, Mark, ModeRail, StatusPill, Volume, type Command, type SurfaceMode } from "./ui";

export function App() {
  const [mode, setMode] = useState<SurfaceMode>("workstation");
  const [paletteOpen, setPaletteOpen] = useState(false);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const runPhase = useStore((s) => s.runPhase);
  const notices = useStore((s) => s.notices);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);

  useEffect(() => connectStore(), []);

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
      { id: "go.workstation", label: "Open Workstation", run: () => setMode("workstation") },
      { id: "go.ide", label: "Open IDE", run: () => setMode("ide") },
      { id: "go.chat", label: "Open Chat", run: () => setMode("chat") },
    ],
    [],
  );

  const degraded = runtimeStatus === "degraded" || runtimeStatus === "failed" || runtimeStatus === "down";
  const latestNotice = notices[notices.length - 1];

  return (
    <div className="app-shell">
      <aside className="mode-wall">
        <div className="brand-mark" title="Hawking">
          <Mark size={16} />
        </div>
        <ModeRail mode={mode} onMode={setMode} />
      </aside>

      <div className="stage-shell">
        <main className="stage">
          {degraded ? <DegradedBanner status={runtimeStatus} detail={runtimeDetail} /> : null}
          {mode === "workstation" ? <Workstation /> : mode === "ide" ? <Ide /> : <Chat />}
        </main>

        <footer className="statusbar t-micro">
          <span>phase: {runPhase}</span>
          <span>{TRANSPORT_KIND} transport</span>
          {latestNotice ? (
            <span
              className="statusbar__notice"
              style={{ color: latestNotice.kind === "error" ? "var(--bad)" : undefined }}
            >
              {latestNotice.message}
            </span>
          ) : null}
        </footer>
      </div>

      <aside className="lightwell">
        <div className="lightwell__head">
          <span className="wordmark">HIDE</span>
          <span className="t-micro">hawking</span>
          <div style={{ marginLeft: "auto", minWidth: 0 }}>
            <StatusPill status={runtimeStatus} detail={runtimeDetail} />
          </div>
        </div>
        <ContextStack />
      </aside>

      {gate ? <GatePrompt message={gate.message} gateId={gate.gate} onDismiss={dismissGate} /> : null}

      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}

function DegradedBanner({ status, detail }: { status: string; detail: string | null }) {
  return (
    <div role="status" className="degraded-banner t-micro">
      Local engine is {status}, {detail ?? "start hide-serve to connect"}
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
    <div className="gate-overlay">
      <Volume alive className="gate-dialog">
        <div className="t-label" style={{ color: "var(--light)" }}>Approval</div>
        <div className="t-body" style={{ margin: "var(--ma-4) 0", color: "var(--text-1)" }}>
          {message}
        </div>
        <div style={{ display: "flex", gap: "var(--ma-3)", justifyContent: "flex-end" }}>
          <button className="ghost-button t-body" onClick={onDismiss}>Deny</button>
          <Gate onClick={onDismiss} title={gateId}>Approve</Gate>
        </div>
      </Volume>
    </div>
  );
}
