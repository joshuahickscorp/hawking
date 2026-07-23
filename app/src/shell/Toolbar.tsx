/*
  Toolbar.tsx — the Xcode-style unified top bar (Liquid Glass), kept deliberately spare. Left:
  navigator toggle + wordmark + summon/stop. Right: the ✦ Executor summon + the panel toggle. The
  model + status + live context now live on the Explorer workcard, and forking is auto-invoked by the
  workflows, so the old center status chip and the Try-N control are gone.
*/
import { boundShortcuts, useStore } from "../store";
import { keyLabel } from "../surfaces/chat/actions";
import { Icon } from "./icons";
import { LogoH } from "./Mark";

export function Toolbar({
  mode,
  onMode,
  chatOpen,
  onToggleSidebar,
  onTogglePanel,
  onToggleChat,
  onSettings,
  onCancel,
}: {
  mode: "chat" | "code";
  onMode: (m: "chat" | "code") => void;
  chatOpen: boolean;
  onToggleSidebar: () => void;
  onTogglePanel: () => void;
  onToggleChat: () => void;
  onSettings: () => void;
  onCancel: () => void;
}) {
  const runPhase = useStore((s) => s.runPhase);
  // Chord labels are READ from the bound-key table, never spelled out here: the hand-written
  // "Cmd B" / "Cmd I" / "Cmd J" were wrong wherever Mod is Ctrl, and the bound Mod+. for cancel_run
  // was shown nowhere at all.
  const chord = (id: string) => {
    const b = boundShortcuts().find((k) => k.id === id);
    return b ? ` (${keyLabel(b.shortcut)})` : "";
  };
  const ws = useStore((s) => s.home?.workspace);
  const working = runPhase === "executing" || runPhase === "planning" || runPhase === "awaiting";
  const inCode = mode === "code";

  return (
    <header className="toolbar glass">
      <div className="toolbar__group toolbar__left">
        {inCode ? (
          <button className="toolbar__icon" title={`Toggle navigator${chord("toggle.sidebar")}`} aria-label="Toggle navigator" onClick={onToggleSidebar}>
            <Icon name="sidebar-toggle" size={17} strokeWidth={1.5} />
          </button>
        ) : null}
        <span className="toolbar__brand" title="HIDE"><LogoH size={13} /></span>
        {/* Chat/Code lives in the sidebar in chat mode (Claude Code geometry); the toolbar keeps it in
            the Code chamber (which has no rail), and re-shows it as a fallback in chat mode only when
            the rail is hidden at narrow widths, so chamber switching is never unreachable. */}
        <div
          className={"toolbar__switch" + (inCode ? "" : " toolbar__switch--rail-fallback")}
          role="tablist"
          aria-label="Chamber"
        >
          <button
            role="tab"
            aria-selected={mode === "chat"}
            className={"toolbar__switchbtn" + (mode === "chat" ? " toolbar__switchbtn--on" : "")}
            onClick={() => onMode("chat")}
          >
            Chat
          </button>
          <button
            role="tab"
            aria-selected={mode === "code"}
            className={"toolbar__switchbtn" + (mode === "code" ? " toolbar__switchbtn--on" : "")}
            onClick={() => onMode("code")}
          >
            Code
          </button>
        </div>
        {inCode ? (
          <button className="toolbar__icon" title={`Cancel run${chord("cancel_run")}`} aria-label="Stop" onClick={onCancel} disabled={!working}>
            <Icon name="stop" size={12} />
          </button>
        ) : null}
        {/* Session identity in the title bar (Claude Code geometry): project name + branch tag. */}
        {!inCode ? (
          <span className="toolbar__session" title={ws?.root}>
            <span className="toolbar__session-name">{ws?.repo ?? "workspace"}</span>
            <span className="toolbar__session-tag">
              <Icon name="source-control" size={11} strokeWidth={1.5} />
              {ws?.branch ?? "main"}
            </span>
          </span>
        ) : null}
      </div>

      <div className="toolbar__group toolbar__right">
        {inCode ? (
          <button
            className={["toolbar__icon", chatOpen && "toolbar__icon--active"].filter(Boolean).join(" ")}
            title={`Executor${chord("toggle.chat")}`}
            aria-label="Executor"
            aria-pressed={chatOpen}
            onClick={onToggleChat}
          >
            <Icon name="sparkle" size={17} strokeWidth={1.4} />
          </button>
        ) : null}
        <button className="toolbar__icon" title={`Settings${chord("open.settings")}`} aria-label="Settings" onClick={onSettings}>
          <Icon name="settings" size={17} strokeWidth={1.5} />
        </button>
        {inCode ? (
          <button className="toolbar__icon" title={`Toggle panel${chord("toggle.panel")}`} aria-label="Toggle panel" onClick={onTogglePanel}>
            <Icon name="panel-toggle" size={17} strokeWidth={1.5} />
          </button>
        ) : null}
      </div>
    </header>
  );
}
