import { Component, useState, type ErrorInfo, type ReactNode } from "react";
import { sendIntent, TRANSPORT_KIND } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Icon } from "./icons";

const H_PATH =
  "M56 710V0H250V272C250 348 270 391 319 391C371 391 380 348 380 272V0H575V341C575 459 508 542 397 542C335 542 281 523 250 467V710Z";

export function LogoH({ size = 18 }: { size?: number }) {
  return (
    <svg height={size} viewBox="56 0 519 710" fill="currentColor" role="img" aria-label="HIDE">
      <g transform="translate(0,710) scale(1,-1)">
        <path d={H_PATH} />
      </g>
    </svg>
  );
}

export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="currentColor" role="img" aria-label="HIDE">
      <circle cx="42.41" cy="59.12" r="17.2" />
      <g transform="translate(56.40,46.39) scale(0.03197,-0.03197)">
        <path d={H_PATH} />
      </g>
    </svg>
  );
}

export function Radiate({
  size = 16,
  active = true,
  stage,
  stages = 4,
  title,
}: {
  size?: number;
  active?: boolean;
  stage?: number;
  stages?: number;
  title?: string;
}) {
  const sw = Math.max(1.25, size / 12);
  const r = size / 2 - sw;
  const c = 2 * Math.PI * r;
  const laddered = typeof stage === "number";
  const clamped = laddered ? Math.min(Math.max(stage as number, 0), stages) : 0;
  const frac = laddered ? 0.12 + 0.8 * (clamped / stages) : 0.26;
  const label = title ?? (laddered ? `verifying ${clamped} of ${stages}` : active ? "working" : "idle");
  return (
    <span
      className={["radiate", active && !laddered && "radiate--active", laddered && "radiate--laddered"].filter(Boolean).join(" ")}
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      title={title}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--line-strong)" strokeWidth={sw} />
        <circle
          className="radiate__arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="var(--light)"
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={`${c * frac} ${c}`}
        />
      </svg>
    </span>
  );
}

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

export function StateTimeline() {
  const tools = useStore((s) => s.tools);
  const sessionId = useStore((s) => s.sessionId);
  const [sel, setSel] = useState<number | null>(null);
  const steps = tools.slice(-14);
  if (steps.length === 0) return null;
  const at = sel == null ? steps.length - 1 : Math.min(sel, steps.length - 1);
  const scrub = (i: number) => {
    setSel(i);
    void sendIntent(intent.scrubToEvent(sessionId, steps[i].call_id));
  };
  const forkHere = () => void sendIntent(intent.forkSession(sessionId, steps[at].call_id));

  return (
    <div className="statetl" role="group" aria-label="Agent state timeline">
      <span className="statetl__label" title="The agent's state is a serializable snapshot. Scrubbing is instant.">state</span>
      <div className="statetl__dots">
        {steps.map((step, i) => (
          <button
            key={step.call_id + step.ts}
            className={["statetl__dot", i <= at && "statetl__dot--past", i === at && "statetl__dot--at"].filter(Boolean).join(" ")}
            title={step.message}
            aria-label={step.message}
            onClick={() => scrub(i)}
          />
        ))}
      </div>
      <span className="statetl__msg">{steps[at]?.message}</span>
      <div className="statetl__actions">
        {sel != null ? (
          <button className="statetl__btn" onClick={() => setSel(null)} title="Return to the latest state">live</button>
        ) : null}
        <button className="statetl__btn statetl__fork" onClick={forkHere} title="Fork a new branch from this point. Instant, the state is a snapshot.">
          <Icon name="fork" size={12} strokeWidth={1.6} />
          fork from here
        </button>
      </div>
    </div>
  );
}

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("HIDE render error:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="crash" role="alert">
        <div className="crash__box glass">
          <div className="crash__title">HIDE hit a render error</div>
          <pre className="crash__detail">{this.state.error.message}</pre>
          <button className="crash__btn" onClick={() => location.reload()}>reload</button>
        </div>
      </div>
    );
  }
}
