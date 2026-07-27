import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  boundShortcuts,
  commandById,
  connectStore,
  hasSessionActivity,
  matchesShortcut,
  paletteCommands,
  runCommand,
  SHELL_COMMANDS,
  useStore,
} from "./store";
import { ackState, heldNote } from "./wire";
import { TRANSPORT_KIND } from "./ipc";
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
import { MOCK_DIFF, parseDiff, type DiffDoc } from "./surfaces/ide/types";
import { Home } from "./surfaces/home/Home";
import type { ChatPanelKind } from "./surfaces/home/ChatPanel";
import type { PermMode } from "./surfaces/home/HomeComposer";

// The two chambers: Chat (Claude Code style, the front door) and Code (the IDE, Cursor style).
type Mode = "chat" | "code";

// The boot tab is a MOCK fixture path (it exists only in surfaces/ide/types.ts). On a live host
// there is no file this app may assume, so nothing is opened until the user opens one.
const INITIAL_FILE = TRANSPORT_KIND === "mock" ? "crates/pool/src/guard.rs" : null;

// Lightweight UI-state persistence: the shell layout survives a restart. Wrapped in try/catch so a
// disabled localStorage never breaks boot.
//
// The namespace carries the transport. Persisted state includes workspace-shaped content (the open
// tab list and the active path), and the mock transport's content is FIXTURE content that does not
// exist on a live host. Sharing one namespace let a dev session's fixture tabs reopen against a real
// workspace, which produced real user.intent.open_file events for files that were never there and an
// error panel the user did not ask for. Keying by transport keeps the two worlds apart.
// ponytail: transport-scoped, not workspace-scoped. Switching a live host between two repositories
// still carries tabs across. Upgrade is keying on the workspace root once the shell knows it at boot.
const PERSIST_NS = "hide." + TRANSPORT_KIND + ".";

function persisted<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(PERSIST_NS + key);
    return v == null ? fallback : (JSON.parse(v) as T);
  } catch {
    return fallback;
  }
}
function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem(PERSIST_NS + key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

export function App() {
  // Boot into Chat (the front door); walk to Code (the IDE) via the pop-out. Migrate any legacy "home".
  const [mode, setMode] = useState<Mode>(() => (persisted<string>("mode", "chat") === "code" ? "code" : "chat"));
  // Permission mode governs the security gate: bypass auto-approves, ask prompts (see below).
  // Never resumed across a restart: bypass auto-approves every gated command, so a persisted one
  // would run unattended on the next launch. Every session therefore boots into "ask", which is why
  // this is the one piece of shell state that is not persisted.
  const [permMode, setPermMode] = useState<PermMode>("ask");
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
  const [tabs, setTabs] = useState<string[]>(() => persisted("tabs", INITIAL_FILE ? [INITIAL_FILE] : []));
  const [diff, setDiff] = useState<DiffDoc | null>(null);
  // The conversation's side panel. Lifted out of Home so the ONE command spine owns the toggles (a
  // panel that only a mouse could reach had no palette or keyboard path at all).
  const [panel, setPanel] = useState<ChatPanelKind | null>(null);

  // Persist the layout whenever it changes.
  useEffect(() => persist("mode", mode), [mode]);
  useEffect(() => persist("sidebarOpen", sidebarOpen), [sidebarOpen]);
  useEffect(() => persist("chatOpen", chatOpen), [chatOpen]);
  useEffect(() => persist("chatFloating", chatFloating), [chatFloating]);
  useEffect(() => persist("chatPos", chatPos), [chatPos]);
  useEffect(() => persist("panelOpen", panelOpen), [panelOpen]);
  useEffect(() => persist("openPath", openPath), [openPath]);
  useEffect(() => persist("tabs", tabs), [tabs]);

  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const notices = useStore((s) => s.notices);
  const gate = useStore((s) => s.gate);
  const approveGate = useStore((s) => s.approveGate);
  const denyGate = useStore((s) => s.denyGate);
  const pushNotice = useStore((s) => s.pushNotice);
  const diffPatch = useStore((s) => s.projections.diff);
  const sessions = useStore((s) => s.sessions);
  const openSession = useStore((s) => s.openSession);
  // The same condition Home gates its side-panel bar on, read from the ONE selector both share,
  // because the palette rows for those panels are only honest while it holds.
  const hasConversation = useStore(hasSessionActivity);

  useEffect(() => connectStore(), []);

  // RETIRED: the proactive auto-compaction hook. It fired `compact_context`, whose host arm was
  // empty, and it re-armed only on an occupancy drop that only a real compaction produces, so it was
  // a one-shot no-op per session. Real compaction is budget-driven span admission inside the context
  // compiler (crates/hawking-context compiler.rs); nothing in the app has to ask for it.

  // Bypass permissions: when the operator has opted into full autonomy, a security gate is auto-approved
  // instead of prompting. ask/auto still surface the lit approval capsule (the human-in-the-loop default).
  // Once per gate: the decision now stays pending until the host records it, and a host that
  // REFUSES the approval puts the prompt back. Re-firing on that would spin, so a refused bypass
  // approval falls through to the human, which is the safe direction.
  const autoApproved = useRef<string | null>(null);
  useEffect(() => {
    if (!gate) {
      autoApproved.current = null;
      return;
    }
    if (permMode !== "bypass" || autoApproved.current === gate.gate) return;
    autoApproved.current = gate.gate;
    approveGate();
  }, [gate, permMode, approveGate]);

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

  // The side panels live on the Chat stage, so opening one from the palette walks there rather than
  // flipping state under a chamber that does not render it. Like every run-scoped command, it shows
  // nothing until there is a conversation to show it beside; that is the same condition the icon bar
  // itself appears under, not a dead entry.
  const togglePanelFace = useCallback((k: ChatPanelKind) => {
    setMode("chat");
    setPanel((cur) => (cur === k ? null : k));
  }, []);

  // The Navigator, the bottom panel and the Executor pane are Code-chamber furniture: nothing
  // renders them in Chat. Their chords are advertised in Settings and beside their palette rows, so
  // they walk to the chamber that shows them rather than flipping state under one that does not,
  // exactly as togglePanelFace walks to Chat. Advertised means live, everywhere it is advertised.
  const inCode = useCallback((f: () => void) => () => {
    setMode("code");
    f();
  }, []);

  // The handlers for the local-only shell commands (SHELL_COMMANDS carries their ids and keys).
  const shellHandlers = useMemo<Record<string, () => void>>(
    () => ({
      "go.chat": () => setMode("chat"),
      "go.code": () => setMode("code"),
      "toggle.chat": inCode(() => setChatOpen((v) => !v)),
      "toggle.float": inCode(() => setChatFloating((v) => !v)),
      "toggle.panel": inCode(() => setPanelOpen((v) => !v)),
      "toggle.sidebar": inCode(() => setSidebarOpen((v) => !v)),
      "toggle.palette": () => setPaletteOpen((v) => !v),
      "open.settings": () => setSettingsOpen(true),
      "perm.ask": () => setPermMode("ask"),
      "perm.bypass": () => setPermMode("bypass"),
      // Same toggle the panel bar performs: pressing the open panel closes it.
      "panel.terminal": () => togglePanelFace("terminal"),
      "panel.diff": () => togglePanelFace("diff"),
      "panel.preview": () => togglePanelFace("preview"),
      "panel.tools": () => togglePanelFace("tools"),
      "panel.artifacts": () => togglePanelFace("artifacts"),
      "panel.context": () => togglePanelFace("context"),
    }),
    [togglePanelFace, inCode],
  );

  // ONE resolver for every gesture: a shell id runs locally, anything else is a catalog command and
  // goes through the spine. The palette and every catalog chord land here, and this is the only
  // feedback either of them gets, so the ACK is read, not discarded: a refusal (unreachable binding,
  // missing argument) and a hold at the approval gate both surface as a notice. Dropping the ack
  // meant a keyboard user pressing a gated chord saw nothing at all happen.
  const runFromSpine = useCallback(
    (id: string) => {
      const local = shellHandlers[id];
      if (local) return local();
      void runCommand(id)
        .then((ack) => {
          const state = ackState(ack);
          if (state === "accepted") return;
          pushNotice(
            state === "held"
              ? { kind: "info", code: "command", message: heldNote(commandById(id)?.title ?? id) }
              : { kind: "error", code: "command", message: ack.message ?? `${id} was refused` },
          );
        })
        .catch((e) =>
          pushNotice({ kind: "error", code: "command", message: e instanceof Error ? e.message : String(e) }),
        );
    },
    [shellHandlers, pushNotice],
  );

  // Keyboard bindings are DERIVED (shell keys plus every catalog keyboard_shortcut the shell owns),
  // never a second hand-written list.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      for (const b of boundShortcuts()) {
        if (!matchesShortcut(b.shortcut, e)) continue;
        e.preventDefault();
        runFromSpine(b.id);
        return;
      }
      // Escape closes ONE thing: the innermost open overlay. This listener is the outermost of
      // several, so it stands down whenever something nearer the user is open (the palette, Settings
      // and the gate each close themselves), which is what kept a single Escape from closing the
      // palette and the Executor at once.
      if (e.key === "Escape" && chatFloating && !paletteOpen && !settingsOpen && !gate) setChatOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatFloating, runFromSpine, paletteOpen, settingsOpen, gate]);

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

  const cancelRun = () => runFromSpine("cancel_run");

  // Palette entries are DERIVED: the local-only shell commands merged with every catalog command the
  // palette can actually run. No second command list. The shortcut on a row is read from the SAME
  // bound-key table the shell binds from, so the palette can only show a chord that really fires.
  const commands = useMemo<Command[]>(() => {
    const keys = new Map(boundShortcuts().map((b) => [b.id, b.shortcut]));
    // The side panels hang off a live conversation (Home renders the panel bar and the panel itself
    // only when there are messages), so with none there is nothing for the row to open. It is
    // withheld rather than offered as a silent no-op.
    const shell = SHELL_COMMANDS.filter((c) => hasConversation || !c.id.startsWith("panel."));
    return [
      ...[...shell, ...paletteCommands()].map((c) => ({
        id: c.id,
        label: c.title,
        shortcut: keys.get(c.id) ?? null,
        run: () => runFromSpine(c.id),
      })),
      // `open_session` needs the id of a session, which a bare palette gesture cannot invent, so it
      // is offered once per RECENT instead: the argument comes from the row, not from the user.
      ...sessions.slice(0, 8).map((s) => ({
        id: `open_session:${s.id}`,
        label: `Open session: ${s.title}`,
        shortcut: null,
        run: () => openSession(s.id),
      })),
    ];
  }, [runFromSpine, sessions, openSession, hasConversation]);

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
          panel={panel}
          onPanel={togglePanelFace}
          onClosePanel={() => setPanel(null)}
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

      {/* The open editor tabs are the file list the Problems counter offers to the analyzer. */}
      <StatusBar openPaths={tabs} />

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
      {gate ? (
        <GatePrompt
          message={gate.message}
          gateId={gate.gate}
          deciding={!!gate.deciding}
          onApprove={approveGate}
          onDeny={denyGate}
        />
      ) : null}
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

// The security gate. Escape is DELIBERATELY not bound: it used to deny, which made the one key every
// overlay in this app uses to close itself into a decision on a paused step of a live turn (one press
// meant "close the palette", "close the Executor" AND "deny"). A gate is answered by its two buttons
// and by nothing else, and it stays up until the host records the answer (store.decideGate).
function GatePrompt({
  message,
  gateId,
  deciding,
  onApprove,
  onDeny,
}: {
  message: string;
  gateId: string;
  deciding: boolean;
  onApprove: () => void;
  onDeny: () => void;
}) {
  const trapRef = useFocusTrap<HTMLDivElement>();
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
          <button className="text-button" onClick={onDeny} disabled={deciding}>Deny</button>
          <Gate onClick={onApprove} title={gateId} disabled={deciding}>Approve</Gate>
        </div>
        {deciding ? (
          <div role="status" aria-live="polite" className="t-micro" style={{ marginTop: "var(--ma-2)", color: "var(--text-dim)" }}>
            recording your decision
          </div>
        ) : null}
      </div>
    </div>
  );
}
