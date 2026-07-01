import { useEffect, useMemo, useState } from "react";
import { connectStore, useStore } from "./store";
import { TRANSPORT_KIND, sendIntent } from "./ipc";
import { intent } from "./wire";
import { SideBar } from "./shell/SideBar";
import { EditorArea } from "./shell/EditorArea";
import { ChatPane } from "./shell/ChatPane";
import { FloatingChat } from "./shell/FloatingChat";
import { Toolbar } from "./shell/Toolbar";
import { StatusBar } from "./shell/StatusBar";
import { Settings } from "./surfaces/Settings";
import { Onboarding } from "./surfaces/Onboarding";
import { CommandPalette, Gate, type Command } from "./ui";
import { useFocusTrap } from "./shell/a11y";
import { shouldShowOnboarding } from "./shell/onboarding";
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./surfaces/ide/types";

const INITIAL_FILE = "crates/pool/src/guard.rs";

// Lightweight UI-state persistence: the shell layout survives a restart. Wrapped in try/catch so a
// disabled localStorage never breaks boot.
function persisted<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem("hide." + key);
    return v == null ? fallback : (JSON.parse(v) as T);
  } catch {
    return fallback;
  }
}
function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem("hide." + key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => persisted("sidebarOpen", true));
  const [chatOpen, setChatOpen] = useState(() => persisted("chatOpen", true));
  const [chatFloating, setChatFloating] = useState(() => persisted("chatFloating", true));
  const [chatPos, setChatPos] = useState(() =>
    persisted("chatPos", {
      x: Math.max(24, (typeof window !== "undefined" ? window.innerWidth : 1280) - 384),
      y: 52,
    }),
  );
  const [panelOpen, setPanelOpen] = useState(() => persisted("panelOpen", true));
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // First-run: show the no-folder onboarding until a project is opened (or the sample is chosen).
  const [folderOpened, setFolderOpened] = useState(() => persisted("folderOpened", false));

  const [openPath, setOpenPath] = useState<string | null>(() => persisted("openPath", INITIAL_FILE));
  const [tabs, setTabs] = useState<string[]>(() => persisted("tabs", [INITIAL_FILE]));
  const [diff, setDiff] = useState<DiffDoc | null>(null);

  // Persist the layout whenever it changes.
  useEffect(() => persist("sidebarOpen", sidebarOpen), [sidebarOpen]);
  useEffect(() => persist("chatOpen", chatOpen), [chatOpen]);
  useEffect(() => persist("chatFloating", chatFloating), [chatFloating]);
  useEffect(() => persist("chatPos", chatPos), [chatPos]);
  useEffect(() => persist("panelOpen", panelOpen), [panelOpen]);
  useEffect(() => persist("openPath", openPath), [openPath]);
  useEffect(() => persist("tabs", tabs), [tabs]);
  useEffect(() => persist("folderOpened", folderOpened), [folderOpened]);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const activeRunId = useStore((s) => s.activeRunId);
  const notices = useStore((s) => s.notices);
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

  const cancelRun = () => {
    if (activeRunId) void sendIntent(intent.cancelRun(activeRunId));
  };

  // Approve-and-proceed: tell the engine the gate is cleared (it releases + runs the held command),
  // then drop the prompt. Deny tells the engine to drop the held command. Mirrors the Executor's
  // inline gate. Both carry the gate id the SecurityGate was emitted with.
  const approveGate = () => {
    if (gate) void sendIntent(intent.custom("approve_gate", { gate: gate.gate }));
    dismissGate();
  };
  const denyGate = () => {
    if (gate) void sendIntent(intent.custom("deny_gate", { gate: gate.gate }));
    dismissGate();
  };

  // First-run folder choice: a real path (from the desktop folder picker) tells the engine to switch
  // its workspace root; a null (the sample-workspace fallback) just dismisses first-run. Either way it
  // is recorded so onboarding shows once.
  const chooseFolder = (folder: string | null) => {
    if (folder) void sendIntent(intent.custom("open_folder", { path: folder }));
    setFolderOpened(true);
  };

  const commands = useMemo<Command[]>(
    () => [
      { id: "toggle.chat", label: "Toggle Executor", run: () => setChatOpen((v) => !v) },
      { id: "toggle.float", label: "Executor: Float / Dock", run: () => setChatFloating((v) => !v) },
      { id: "toggle.panel", label: "Toggle Terminal", run: () => setPanelOpen((v) => !v) },
      { id: "toggle.sidebar", label: "Toggle Navigator", run: () => setSidebarOpen((v) => !v) },
    ],
    [],
  );

  const degraded = runtimeStatus === "degraded" || runtimeStatus === "failed" || runtimeStatus === "down";

  return (
    <div className="vsc-shell">
      <Toolbar
        chatOpen={chatOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onTogglePanel={() => setPanelOpen((v) => !v)}
        onToggleChat={() => setChatOpen((v) => !v)}
        onSettings={() => setSettingsOpen(true)}
        onCancel={cancelRun}
      />

      <div className="vsc-body">
        {sidebarOpen ? <SideBar openPath={openPath} onOpen={openFile} /> : null}

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
      {gate ? <GatePrompt message={gate.message} gateId={gate.gate} onApprove={approveGate} onDeny={denyGate} /> : null}
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
      {settingsOpen ? <Settings onClose={() => setSettingsOpen(false)} /> : null}
      {shouldShowOnboarding(folderOpened) ? <Onboarding onChoose={chooseFolder} /> : null}
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
  onApprove,
  onDeny,
}: {
  message: string;
  gateId: string;
  onApprove: () => void;
  onDeny: () => void;
}) {
  const trapRef = useFocusTrap<HTMLDivElement>();
  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDeny();
    };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [onDeny]);
  return (
    <div className="gate-overlay" role="presentation">
      <div
        className="gate-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Approval required"
        tabIndex={-1}
        ref={trapRef}
      >
        <div className="t-label" style={{ color: "var(--text-strong)" }}>Approval required</div>
        <div className="t-body" style={{ margin: "var(--ma-4) 0", color: "var(--text)" }}>
          {message}
        </div>
        <div style={{ display: "flex", gap: "var(--ma-2)", justifyContent: "flex-end" }}>
          <button className="text-button" onClick={onDeny}>Deny</button>
          <Gate onClick={onApprove} title={gateId}>Approve</Gate>
        </div>
      </div>
    </div>
  );
}
