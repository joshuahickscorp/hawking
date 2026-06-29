/*
  Toolbar.tsx — the Xcode-style unified top bar (Liquid Glass). Left: navigator toggle + wordmark +
  run/stop. Center: the build/phase status with the radiate signature + model chip. Right: Fork & Try N
  (the free-fleet move), the ✦ assistant summon, and the panel toggle. Reads live store state.
*/
import { useStore } from "../store";
import { Icon } from "./icons";
import { Radiate } from "./Radiate";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function Toolbar({
  chatOpen,
  onToggleSidebar,
  onTogglePanel,
  onToggleChat,
  onTryN,
  onCancel,
}: {
  chatOpen: boolean;
  onToggleSidebar: () => void;
  onTogglePanel: () => void;
  onToggleChat: () => void;
  onTryN: (n: number) => void;
  onCancel: () => void;
}) {
  const runPhase = useStore((s) => s.runPhase);
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const manifest = useStore((s) => s.manifest);
  const model = manifest?.model?.id ?? "qwen2.5";
  const working = runPhase === "executing" || runPhase === "planning" || runPhase === "awaiting";

  return (
    <header className="toolbar glass">
      <div className="toolbar__group toolbar__left">
        <button className="toolbar__icon" title="Toggle navigator (Cmd B)" aria-label="Toggle navigator" onClick={onToggleSidebar}>
          <Icon name="sidebar-toggle" size={17} strokeWidth={1.5} />
        </button>
        <span className="toolbar__brand">HIDE</span>
        <button className="toolbar__icon" title="Summon the agent" aria-label="Run" onClick={onToggleChat}>
          <Icon name="play" size={14} />
        </button>
        <button className="toolbar__icon" title="Cancel run" aria-label="Stop" onClick={onCancel} disabled={!working}>
          <Icon name="stop" size={12} />
        </button>
      </div>

      <div className="toolbar__center">
        <div className="toolbar__status glass" title={runtimeStatus}>
          <Radiate size={13} active={working} title={working ? "working" : "idle"} />
          <span className="toolbar__phase">{working ? cap(runPhase) : runtimeStatus === "ready" ? "Ready" : cap(runtimeStatus)}</span>
          <span className="toolbar__sep">·</span>
          <span className="toolbar__model">{model}</span>
          <span className="toolbar__sep">·</span>
          <span className="toolbar__local" title="Local only, nothing leaves your machine">local</span>
        </div>
      </div>

      <div className="toolbar__group toolbar__right">
        <div className="toolbar__tryn" title="Fork the agent state into N attempts and run them in parallel. Free, local.">
          <Icon name="fork" size={13} strokeWidth={1.6} />
          <span className="toolbar__tryn-label">try</span>
          {[3, 5, 8].map((n) => (
            <button key={n} className="toolbar__tryn-n" title={`Fork & try ${n}`} onClick={() => onTryN(n)}>
              {n}
            </button>
          ))}
        </div>
        <button
          className={["toolbar__icon", chatOpen && "toolbar__icon--active"].filter(Boolean).join(" ")}
          title="Assistant (Cmd I)"
          aria-label="Assistant"
          aria-pressed={chatOpen}
          onClick={onToggleChat}
        >
          <Icon name="sparkle" size={17} strokeWidth={1.4} />
        </button>
        <button className="toolbar__icon" title="Toggle panel (Cmd J)" aria-label="Toggle panel" onClick={onTogglePanel}>
          <Icon name="panel-toggle" size={17} strokeWidth={1.5} />
        </button>
      </div>
    </header>
  );
}
