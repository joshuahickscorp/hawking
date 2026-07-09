import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { connectStore, useStore } from "./store";
import { TRANSPORT_KIND, sendIntent } from "./ipc";
import { intent } from "./wire";
import { SideBar } from "./shell/SideBar";
// The Code chamber carries Monaco (~4.5MB). Lazy so the Chat front door never parses the editor at
// boot; it loads on first entry to Code. Prefetched on Toolbar hover of the Code tab.
const EditorArea = lazy(() => import("./shell/EditorArea").then((m) => ({ default: m.EditorArea })));
import { ChatPane } from "./shell/ChatPane";
import { FloatingChat } from "./shell/FloatingChat";
import { Toolbar } from "./shell/Toolbar";
import { StatusBar } from "./shell/StatusBar";
import { Settings } from "./surfaces/Settings";
import { CommandPalette, Gate, type Command } from "./ui";
import { useFocusTrap } from "./shell/a11y";
import { useAutoCompact } from "./shell/autocompact";
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./surfaces/ide/types";
import { Home } from "./surfaces/home/Home";
import type { PermMode } from "./surfaces/home/HomeComposer";

// The two chambers: Chat (Claude Code style, the front door) and Code (the IDE, Cursor style).
type Mode = "chat" | "code";

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
  // Boot into Chat (the front door); walk to Code (the IDE) via the pop-out. Migrate any legacy "home".
  const [mode, setMode] = useState<Mode>(() => (persisted<string>("mode", "chat") === "code" ? "code" : "chat"));
  // Permission mode governs the security gate: bypass auto-approves, ask/auto prompt (see below).
  // Never silently resume Bypass across a restart: it auto-approves every gated command, so a
  // persisted bypass would run unattended on the next launch. Boot such a session back to "ask".
  const [permMode, setPermMode] = useState<PermMode>(() => {
    const saved = persisted<PermMode>("permMode", "ask");
    return saved === "bypass" ? "ask" : saved;
  });
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
  const [openPath, setOpenPath] = useState<string | null>(() => persisted("openPath", INITIAL_FILE));
  const [tabs, setTabs] = useState<string[]>(() => persisted("tabs", [INITIAL_FILE]));
  const [diff, setDiff] = useState<DiffDoc | null>(null);

  // Persist the layout whenever it changes.
  useEffect(() => persist("mode", mode), [mode]);
  useEffect(() => persist("permMode", permMode), [permMode]);
  useEffect(() => persist("sidebarOpen", sidebarOpen), [sidebarOpen]);
  useEffect(() => persist("chatOpen", chatOpen), [chatOpen]);
  useEffect(() => persist("chatFloating", chatFloating), [chatFloating]);
  useEffect(() => persist("chatPos", chatPos), [chatPos]);
  useEffect(() => persist("panelOpen", panelOpen), [panelOpen]);
  useEffect(() => persist("openPath", openPath), [openPath]);
  useEffect(() => persist("tabs", tabs), [tabs]);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const activeRunId = useStore((s) => s.activeRunId);
  const notices = useStore((s) => s.notices);
  const gate = useStore((s) => s.gate);
  const dismissGate = useStore((s) => s.dismissGate);
  const diffPatch = useStore((s) => s.projections.diff);

  useEffect(() => connectStore(), []);

  // Secret, proactive context compaction: the condenser mindset applied to the live window. Watches the
  // engine's watermark and compacts ahead of the cliff during work, silently. No cap ever reaches the UI.
  useAutoCompact();

  // Bypass permissions: when the operator has opted into full autonomy, a security gate is auto-approved
  // instead of prompting. ask/auto still surface the lit approval capsule (the human-in-the-loop default).
  useEffect(() => {
    if (gate && permMode === "bypass") {
      void sendIntent(intent.custom("approve_gate", { gate: gate.gate }));
      dismissGate();
    }
  }, [gate, permMode, dismissGate]);

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

  const commands = useMemo<Command[]>(
    () => [
      { id: "go.chat", label: "Go to Chat", run: () => setMode("chat") },
      { id: "go.code", label: "Go to Code", run: () => setMode("code") },
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
        mode={mode}
        onMode={setMode}
        chatOpen={chatOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onTogglePanel={() => setPanelOpen((v) => !v)}
        onToggleChat={() => setChatOpen((v) => !v)}
        onSettings={() => setSettingsOpen(true)}
        onCancel={cancelRun}
      />

      {mode === "chat" ? (
        <Home
          mode={mode}
          onMode={setMode}
          onPopToCode={() => {
            // Picture in picture out: open the SAME conversation in the Code chamber to watch code
            // (Cursor style). Dock beside the editor when wide; float when narrow (the docked pane sheds
            // below 1100px, so floating guarantees the chat stays visible).
            setMode("code");
            const wide = typeof window !== "undefined" && window.innerWidth >= 1180;
            setChatFloating(!wide);
            setChatOpen(true);
          }}
          onSettings={() => setSettingsOpen(true)}
          permMode={permMode}
          onPermMode={setPermMode}
        />
      ) : (
        <div className="vsc-body">
          {sidebarOpen ? <SideBar openPath={openPath} onOpen={openFile} /> : null}

          <Suspense fallback={<div className="editor-area editor-loading" />}>
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
          </Suspense>

          {chatOpen && !chatFloating ? (
            <ChatPane
              onClose={() => setChatOpen(false)}
              onFloat={() => setChatFloating(true)}
              onPopToChat={() => setMode("chat")}
            />
          ) : null}
        </div>
      )}

      <StatusBar />

      {mode === "code" && chatOpen && chatFloating ? (
        <FloatingChat
          pos={chatPos}
          onPos={setChatPos}
          onClose={() => setChatOpen(false)}
          onDock={() => setChatFloating(false)}
          onPopToChat={() => setMode("chat")}
        />
      ) : null}

      {degraded ? <DegradedToast status={runtimeStatus} detail={runtimeDetail} /> : null}
      {gate ? <GatePrompt message={gate.message} gateId={gate.gate} onApprove={approveGate} onDeny={denyGate} /> : null}
      <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />
      {settingsOpen ? <Settings onClose={() => setSettingsOpen(false)} /> : null}
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
