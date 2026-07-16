/*
  Toolbar.tsx — the Xcode-style unified top bar (Liquid Glass), kept deliberately spare. Left:
  navigator toggle + wordmark + summon/stop. Right: the ✦ Executor summon + the panel toggle. The
  model + status + live context now live on the Explorer workcard, and forking is auto-invoked by the
  workflows, so the old center status chip and the Try-N control are gone.
*/
import { useStore } from "../store";
import { Icon } from "./icons";
import { LogoH } from "./components";

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
  const ws = useStore((s) => s.home?.workspace);
  const working = runPhase === "executing" || runPhase === "planning" || runPhase === "awaiting";
  const inCode = mode === "code";

  return (
    <header className="toolbar glass">
      <div className="toolbar__group toolbar__left">
        {inCode ? (
          <button className="toolbar__icon" title="Toggle navigator (Cmd B)" aria-label="Toggle navigator" onClick={onToggleSidebar}>
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
          <button className="toolbar__icon" title="Cancel run" aria-label="Stop" onClick={onCancel} disabled={!working}>
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
            title="Executor (Cmd I)"
            aria-label="Executor"
            aria-pressed={chatOpen}
            onClick={onToggleChat}
          >
            <Icon name="sparkle" size={17} strokeWidth={1.4} />
          </button>
        ) : null}
        <button className="toolbar__icon" title="Settings" aria-label="Settings" onClick={onSettings}>
          <Icon name="settings" size={17} strokeWidth={1.5} />
        </button>
        {inCode ? (
          <button className="toolbar__icon" title="Toggle panel (Cmd J)" aria-label="Toggle panel" onClick={onTogglePanel}>
            <Icon name="panel-toggle" size={17} strokeWidth={1.5} />
          </button>
        ) : null}
      </div>
    </header>
  );
}
