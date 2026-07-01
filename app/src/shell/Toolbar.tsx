/*
  Toolbar.tsx — the Xcode-style unified top bar (Liquid Glass), kept deliberately spare. Left:
  navigator toggle + wordmark + summon/stop. Right: the ✦ Executor summon + the panel toggle. The
  model + status + live context now live on the Explorer workcard, and forking is auto-invoked by the
  workflows, so the old center status chip and the Try-N control are gone.
*/
import { useStore } from "../store";
import { Icon } from "./icons";
import { LogoH } from "./Mark";

export function Toolbar({
  chatOpen,
  onToggleSidebar,
  onTogglePanel,
  onToggleChat,
  onSettings,
  onCancel,
}: {
  chatOpen: boolean;
  onToggleSidebar: () => void;
  onTogglePanel: () => void;
  onToggleChat: () => void;
  onSettings: () => void;
  onCancel: () => void;
}) {
  const runPhase = useStore((s) => s.runPhase);
  const working = runPhase === "executing" || runPhase === "planning" || runPhase === "awaiting";

  return (
    <header className="toolbar glass">
      <div className="toolbar__group toolbar__left">
        <button className="toolbar__icon" title="Toggle navigator (Cmd B)" aria-label="Toggle navigator" onClick={onToggleSidebar}>
          <Icon name="sidebar-toggle" size={17} strokeWidth={1.5} />
        </button>
        <span className="toolbar__brand" title="HIDE"><LogoH size={13} /></span>
        <button className="toolbar__icon" title="Summon the agent" aria-label="Run" onClick={onToggleChat}>
          <Icon name="play" size={14} />
        </button>
        <button className="toolbar__icon" title="Cancel run" aria-label="Stop" onClick={onCancel} disabled={!working}>
          <Icon name="stop" size={12} />
        </button>
      </div>

      <div className="toolbar__group toolbar__right">
        <button
          className={["toolbar__icon", chatOpen && "toolbar__icon--active"].filter(Boolean).join(" ")}
          title="Executor (Cmd I)"
          aria-label="Executor"
          aria-pressed={chatOpen}
          onClick={onToggleChat}
        >
          <Icon name="sparkle" size={17} strokeWidth={1.4} />
        </button>
        <button className="toolbar__icon" title="Settings" aria-label="Settings" onClick={onSettings}>
          <Icon name="settings" size={17} strokeWidth={1.5} />
        </button>
        <button className="toolbar__icon" title="Toggle panel (Cmd J)" aria-label="Toggle panel" onClick={onTogglePanel}>
          <Icon name="panel-toggle" size={17} strokeWidth={1.5} />
        </button>
      </div>
    </header>
  );
}
