/*
  StatusBar.tsx — the 22px VS Code status bar. Left: branch + problem counts. Right: phase, model,
  transport, runtime state. Binds existing store fields; no new state.
*/
import { TRANSPORT_KIND } from "../ipc";
import { useStore } from "../store";
import { Icon } from "./icons";

export function StatusBar() {
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const runPhase = useStore((s) => s.runPhase);
  const manifest = useStore((s) => s.manifest);
  const notices = useStore((s) => s.notices);

  const model = manifest?.model?.id ?? "qwen2.5-7b";
  const latest = notices[notices.length - 1];
  const dotClass =
    runtimeStatus === "ready"
      ? "status-dot status-dot--ok"
      : runtimeStatus === "failed" || runtimeStatus === "down"
        ? "status-dot status-dot--bad"
        : "status-dot status-dot--light";

  return (
    <footer className="vsc-statusbar">
      <button className="vsc-statusbar__item vsc-statusbar__item--button" title="Branch">
        <Icon name="source-control" size={13} strokeWidth={1.8} />
        <span>main</span>
      </button>
      <span className="vsc-statusbar__item" title="Problems">
        <Icon name="error" size={13} strokeWidth={1.8} />
        <span>0</span>
        <Icon name="warning" size={13} strokeWidth={1.8} style={{ marginLeft: 6 }} />
        <span>0</span>
      </span>

      {latest ? (
        <span className="vsc-statusbar__item" style={{ color: latest.kind === "error" ? "var(--red)" : "var(--text-muted)" }}>
          {latest.message}
        </span>
      ) : null}

      <span className="vsc-statusbar__spacer" />

      <span className="vsc-statusbar__item">phase: {runPhase}</span>
      <span className="vsc-statusbar__item">{model}</span>
      <span className="vsc-statusbar__item">{TRANSPORT_KIND} transport</span>
      <span className="vsc-statusbar__item" title={runtimeDetail ?? undefined}>
        <span className={dotClass} style={{ width: 8, height: 8 }} />
        <span style={{ textTransform: "capitalize" }}>{runtimeStatus}</span>
      </span>
    </footer>
  );
}
