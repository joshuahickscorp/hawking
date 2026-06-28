import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { RuntimeState } from "./wire";

export type SurfaceMode = "workstation" | "ide" | "chat";

export function Volume({
  children,
  alive = false,
  raised = false,
  quiet = false,
  pad,
  as: As = "div",
  style,
  className,
}: {
  children: ReactNode;
  alive?: boolean;
  raised?: boolean;
  quiet?: boolean;
  pad?: string;
  as?: "div" | "section" | "aside" | "article";
  style?: CSSProperties;
  className?: string;
}) {
  const cls = [
    "volume",
    raised && "volume--raised",
    quiet && "volume--quiet",
    alive && "alive",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <As className={cls} style={{ ...(pad ? { padding: pad } : null), ...style }}>
      {children}
    </As>
  );
}

export function Panel(props: {
  children: ReactNode;
  active?: boolean;
  pad?: string;
  style?: CSSProperties;
  className?: string;
}) {
  return <Volume alive={props.active} pad={props.pad} style={props.style} className={props.className}>{props.children}</Volume>;
}

export function LightEdge({
  mode = "breathe",
  children,
  style,
}: {
  mode?: "breathe" | "travel";
  children: ReactNode;
  style?: CSSProperties;
}) {
  if (mode === "breathe") {
    return (
      <div className="alive" style={{ position: "relative", borderRadius: "var(--radius)", ...style }}>
        {children}
      </div>
    );
  }

  return (
    <div style={{ position: "relative", borderRadius: "var(--radius)", boxShadow: "var(--hairline)", ...style }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          insetInline: 0,
          top: 0,
          height: 1,
          borderRadius: "inherit",
          pointerEvents: "none",
          backgroundImage:
            "linear-gradient(90deg, transparent 0%, var(--light-soft) 38%, var(--light) 50%, var(--light-soft) 62%, transparent 100%)",
          backgroundSize: "55% 1px",
          backgroundRepeat: "no-repeat",
          animation: "light-travel var(--dur-door) var(--ease) forwards",
        }}
      />
      {children}
    </div>
  );
}

const LIGHT_ACTIVE: CSSProperties = { animation: "breathe var(--breathe) var(--ease) infinite" };
export function useLight(active: boolean): CSSProperties {
  return useMemo(() => (active ? LIGHT_ACTIVE : {}), [active]);
}

export function Display({ children, style, className }: { children: ReactNode; style?: CSSProperties; className?: string }) {
  return (
    <h1 className={["t-display", className].filter(Boolean).join(" ")} style={style}>
      {children}
    </h1>
  );
}

export function SectionLabel({ children, count }: { children: ReactNode; count?: number }) {
  return (
    <div className="t-label" style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)" }}>
      <span>{children}</span>
      {count != null ? <span className="t-micro">{count}</span> : null}
    </div>
  );
}

export function SurfaceHeader({
  label,
  title,
  children,
  meta,
}: {
  label: string;
  title: ReactNode;
  children?: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="surface-header">
      <div className="surface-header__row">
        <div>
          <div className="t-label" style={{ marginBottom: "var(--ma-4)" }}>{label}</div>
          <Display>{title}</Display>
        </div>
        {meta ? <div className="surface-header__meta">{meta}</div> : null}
      </div>
      {children ? <div className="t-body surface-header__sub">{children}</div> : null}
    </header>
  );
}

export function Mark({ size = 16 }: { size?: number }) {
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "50%",
        background: "radial-gradient(circle at 50% 42%, var(--concrete-1) 54%, var(--void) 60%)",
        boxShadow: "0 0 0 1px var(--line-strong), var(--light-bloom), inset 0 0 4px rgba(0, 0, 0, 0.9)",
      }}
    />
  );
}

const RUNTIME_STYLE: Record<RuntimeState, { label: string; dot: string; pulse?: boolean; lit?: boolean }> = {
  down: { label: "Down", dot: "status-dot" },
  booting: { label: "Booting", dot: "status-dot status-dot--light", pulse: true },
  ready: { label: "Ready", dot: "status-dot status-dot--light", lit: true },
  degraded: { label: "Degraded", dot: "status-dot" },
  failed: { label: "Failed", dot: "status-dot status-dot--bad" },
};

export function StatusPill({ status, detail }: { status: RuntimeState; detail?: string | null }) {
  const s = RUNTIME_STYLE[status];
  return (
    <span
      title={detail ?? undefined}
      className={["status-pill", s.lit && "status-pill--lit", s.pulse && "alive"].filter(Boolean).join(" ")}
    >
      <span className={s.dot} />
      <span>{s.label}</span>
      {detail ? <span className="t-micro" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{detail}</span> : null}
    </span>
  );
}

const MODES: { id: SurfaceMode; glyph: string; label: string }[] = [
  { id: "workstation", glyph: "▦", label: "Workstation" },
  { id: "ide", glyph: "⌘", label: "IDE" },
  { id: "chat", glyph: "✎", label: "Chat" },
];

export function ModeRail({ mode, onMode }: { mode: SurfaceMode; onMode: (m: SurfaceMode) => void }) {
  return (
    <nav className="mode-rail" aria-label="Surface">
      {MODES.map((m) => (
        <button
          key={m.id}
          className="mode-button"
          title={m.label}
          aria-label={m.label}
          aria-pressed={m.id === mode}
          onClick={() => onMode(m.id)}
        >
          {m.glyph}
        </button>
      ))}
    </nav>
  );
}

export const ModeSwitcher = ModeRail;

export function Gate({
  children,
  onClick,
  title,
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  title?: string;
  style?: CSSProperties;
}) {
  return (
    <button className="gate" onClick={onClick} title={title} style={style}>
      {children}
    </button>
  );
}

export interface Command {
  id: string;
  label: string;
  run: () => void;
}

export function CommandPalette({
  open,
  commands,
  onClose,
}: {
  open: boolean;
  commands: Command[];
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? commands.filter((c) => c.label.toLowerCase().includes(t)) : commands;
  }, [q, commands]);

  useEffect(() => {
    if (!open) return;
    setQ("");
    setSel(0);
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  if (!open) return null;

  return (
    <div role="dialog" aria-label="Command palette" className="palette-overlay" onClick={onClose}>
      <Volume alive pad="0" className="palette">
        <div onClick={(e) => e.stopPropagation()}>
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setSel(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") onClose();
              if (e.key === "ArrowDown") setSel((i) => Math.min(i + 1, filtered.length - 1));
              if (e.key === "ArrowUp") setSel((i) => Math.max(i - 1, 0));
              if (e.key === "Enter" && filtered[sel]) {
                filtered[sel].run();
                onClose();
              }
            }}
            placeholder="Type a command"
            className="t-body palette__input"
          />
          <ul className="palette__list">
            {filtered.length === 0 ? (
              <li className="t-body" style={{ padding: "var(--ma-3)", color: "var(--text-3)" }}>No commands</li>
            ) : (
              filtered.map((c, i) => (
                <li key={c.id}>
                  <button
                    className="ghost-button palette__item t-body"
                    aria-selected={i === sel}
                    onMouseEnter={() => setSel(i)}
                    onClick={() => {
                      c.run();
                      onClose();
                    }}
                  >
                    {c.label}
                  </button>
                </li>
              ))
            )}
          </ul>
        </div>
      </Volume>
    </div>
  );
}
