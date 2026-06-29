import { useEffect, useMemo, useState } from "react";
import { connectStore, useStore, type FleetRun } from "./store";
import { TRANSPORT_KIND, sendIntent } from "./ipc";
import { intent } from "./wire";
import { ActivityBar, type SideView } from "./shell/ActivityBar";
import { SideBar } from "./shell/SideBar";
import { EditorArea } from "./shell/EditorArea";
import { ChatPane } from "./shell/ChatPane";
import { FloatingChat } from "./shell/FloatingChat";
import { Toolbar } from "./shell/Toolbar";
import { StatusBar } from "./shell/StatusBar";
import { CommandPalette, Gate, type Command } from "./ui";
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./surfaces/ide/types";

const INITIAL_FILE = "crates/pool/src/guard.rs";

export function App() {
  const [view, setView] = useState<SideView>("explorer");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatOpen, setChatOpen] = useState(true);
  const [chatFloating, setChatFloating] = useState(true);
  const [chatPos, setChatPos] = useState(() => ({
    x: Math.max(24, (typeof window !== "undefined" ? window.innerWidth : 1280) - 384),
    y: 52,
  }));
  const [panelOpen, setPanelOpen] = useState(true);
  const [paletteOpen, setPaletteOpen] = useState(false);

  const [openPath, setOpenPath] = useState<string | null>(INITIAL_FILE);
  const [tabs, setTabs] = useState<string[]>([INITIAL_FILE]);
  const [diff, setDiff] = useState<DiffDoc | null>(null);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const activeRunId = useStore((s) => s.activeRunId);
  const notices = useStore((s) => s.notices);
  const pushNotice = useStore((s) => s.pushNotice);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);
  const diffPatch = useStore((s) => s.projections.diff);

  useEffect(() => connectStore(), []);

  // Lift the IDE's diff lifecycle into the shell so the editor area and SCM view share it.
  const hostDiff = useMemo(() => parseDiff(diffPatch as Record<string, unknown> | undefined), [diffPatch]);
  useEffect(() => {
    if (hostDiff) {
      setDiff(hostDiff);
      return;
    }
    if (TRANSPORT_KIND === "mock") {
      const t = setTimeout(() => setDiff((d) => d ?? MOCK_DIFF), 1500);
      return () => clearTimeout(t);
    }
  }, [hostDiff]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "p") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (mod && e.key.toLowerCase() === "j") {
        e.preventDefault();
        setPanelOpen((v) => !v);
      } else if (mod && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setSidebarOpen((v) => !v);
      } else if (mod && e.key.toLowerCase() === "i") {
        e.preventDefault();
        setChatOpen((v) => !v);
      } else if (e.key === "Escape" && chatFloating) {
        setChatOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatFloating]);

  const openFile = (path: string) => {
    setOpenPath(path);
    setTabs((t) => (t.includes(path) ? t : [...t, path]));
  };

  const closeTab = (path: string) => {
    setTabs((t) => {
      const next = t.filter((p) => p !== path);
      if (path === openPath) setOpenPath(next[next.length - 1] ?? null);
      return next;
    });
  };

  const selectView = (v: SideView) => {
    if (v === view && sidebarOpen) setSidebarOpen(false);
    else {
      setView(v);
      setSidebarOpen(true);
    }
  };

  // Fork the agent state into N parallel attempts (free, local). Backend forks state in plan 2;
  // here we dispatch the real intent and surface it. The Fleet view (Part 2) renders the branches.
  const tryN = (n: number) => {
    const msgs = useStore.getState().messages;
    const task = [...msgs].reverse().find((m) => m.role === "user")?.text ?? "explore approaches for the current task";
    void sendIntent(intent.custom("fleet_run", { task, n }));
    // Optimistic local preview of the forked branches until the backend forks RWKV state (plan 2).
    const cycle: FleetRun["state"][] = ["active", "active", "waiting", "active", "done", "active", "active", "waiting"];
    const runs: FleetRun[] = Array.from({ length: n }, (_, i) => ({
      id: `try_${i + 1}`,
      objective: `${task.slice(0, 60)} — approach ${i + 1}`,
      state: i === 0 ? "active" : cycle[i % cycle.length],
      step: 1 + (i % 5),
      steps: 6,
    }));
    useStore.getState().apply({ seq: 0, session_id: null, kind: { type: "projection_patch", data: { projection: "fleet", patch: { runs } } } });
    setView("agents");
    setSidebarOpen(true);
    pushNotice({ kind: "info", code: "fleet", message: `forked ${n} attempts, free, local` });
  };

  const cancelRun = () => {
    if (activeRunId) void sendIntent(intent.cancelRun(activeRunId));
  };

  const commands = useMemo<Command[]>(
    () => [
      { id: "view.explorer", label: "View: Explorer", run: () => selectView("explorer") },
      { id: "view.agents", label: "View: Agents", run: () => selectView("agents") },
      { id: "view.context", label: "View: Context", run: () => selectView("context") },
      { id: "toggle.chat", label: "Toggle Assistant", run: () => setChatOpen((v) => !v) },
      { id: "toggle.float", label: "Assistant: Float / Dock", run: () => setChatFloating((v) => !v) },
      { id: "toggle.panel", label: "Toggle Terminal", run: () => setPanelOpen((v) => !v) },
      { id: "toggle.sidebar", label: "Toggle Navigator", run: () => setSidebarOpen((v) => !v) },
      { id: "try.5", label: "Fork & Try 5", run: () => tryN(5) },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [view, sidebarOpen],
  );

  const degraded = runtimeStatus === "degraded" || runtimeStatus === "failed" || runtimeStatus === "down";

  return (
    <div className="vsc-shell">
      <Toolbar
        chatOpen={chatOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onTogglePanel={() => setPanelOpen((v) => !v)}
        onToggleChat={() => setChatOpen((v) => !v)}
        onTryN={tryN}
        onCancel={cancelRun}
      />

      <div className="vsc-body">
        <ActivityBar
          view={view}
          sidebarOpen={sidebarOpen}
          onView={selectView}
          onSettings={() => setPaletteOpen(true)}
        />

        {sidebarOpen ? <SideBar view={view} openPath={openPath} onOpen={openFile} /> : null}

        <EditorArea
          openPath={openPath}
          tabs={tabs}
          diff={diff}
          panelOpen={panelOpen}
          onSelectTab={setOpenPath}
          onCloseTab={closeTab}
          onDiffChange={setDiff}
          onTogglePanel={() => setPanelOpen((v) => !v)}
        />

        {chatOpen && !chatFloating ? <ChatPane onClose={() => setChatOpen(false)} onFloat={() => setChatFloating(true)} /> : null}
      </div>

      <StatusBar />

      {chatOpen && chatFloating ? (
        <FloatingChat pos={chatPos} onPos={setChatPos} onClose={() => setChatOpen(false)} onDock={() => setChatFloating(false)} />
      ) : null}

      {degraded ? <DegradedToast status={runtimeStatus} detail={runtimeDetail} /> : null}
      {gate ? <GatePrompt message={gate.message} gateId={gate.gate} onDismiss={dismissGate} /> : null}
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
      {/* notices surface in the status bar */}
      <span hidden>{notices.length}</span>
    </div>
  );
}

function DegradedToast({ status, detail }: { status: string; detail: string | null }) {
  return (
    <div role="status" className="degraded-toast t-body glass">
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
    <div className="gate-overlay" role="dialog" aria-modal="true">
      <div className="gate-dialog">
        <div className="t-label" style={{ color: "var(--text-strong)" }}>Approval required</div>
        <div className="t-body" style={{ margin: "var(--ma-4) 0", color: "var(--text)" }}>
          {message}
        </div>
        <div style={{ display: "flex", gap: "var(--ma-2)", justifyContent: "flex-end" }}>
          <button className="text-button" onClick={onDismiss}>Deny</button>
          <Gate onClick={onDismiss} title={gateId}>Approve</Gate>
        </div>
      </div>
    </div>
  );
}
