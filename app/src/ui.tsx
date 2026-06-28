/*
  ui.tsx: the base design primitives. Dense, doctrine-correct (04-design-doctrine.md).
  Everything references the tokens in theme.css; no hardcoded colors here.
  The radiation edge is the single most brand-load-bearing asset: it is the black box that
  radiates, made into a reusable device (breathing glow for active/streaming AND a travelling
  fork/handoff sheen). Static fallback under prefers-reduced-motion is honored in theme.css.
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { RuntimeState } from "./wire";

// ---- Panel: the rim-lit anodized material surface (recess + depth + hairline rim). ----
export function Panel({
  children,
  active = false,
  pad = "var(--s3)",
  style,
  className,
}: {
  children: ReactNode;
  active?: boolean;       // wear the breathing radiation edge when alive/streaming
  pad?: string;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <div
      className={"panel" + (className ? " " + className : "")}
      style={{ padding: pad, ...(active ? RADIATION_ACTIVE : null), ...style }}
    >
      {children}
    </div>
  );
}

// The breathing-glow style applied inline so any element can wear the radiation edge.
const RADIATION_ACTIVE: CSSProperties = {
  animation: "radiation-breathe 2.6s ease-in-out infinite",
};

/*
  RadiationEdge: THE gold primitive. Two modes:
    - "breathe": the heartbeat glow on active/streaming elements.
    - "travel":  a one-shot gold sheen that sweeps along the top edge (fork / handoff).
  Wraps its children and lights their border. No spinner; aliveness is the light.
*/
export function RadiationEdge({
  mode = "breathe",
  children,
  style,
}: {
  mode?: "breathe" | "travel";
  children: ReactNode;
  style?: CSSProperties;
}) {
  const base: CSSProperties = { position: "relative", borderRadius: "var(--radius)" };
  if (mode === "breathe") {
    return <div style={{ ...base, animation: "radiation-breathe 2.6s ease-in-out infinite", ...style }}>{children}</div>;
  }
  // travel: a gradient stripe that sweeps once across the top hairline.
  const travel: CSSProperties = {
    ...base,
    boxShadow: "0 0 0 1px var(--rim)",
    ...style,
  };
  const sheen: CSSProperties = {
    position: "absolute",
    inset: 0,
    borderRadius: "inherit",
    pointerEvents: "none",
    backgroundImage:
      "linear-gradient(90deg, transparent 0%, var(--radiation-bloom) 42%, var(--radiation-bright) 50%, var(--radiation-bloom) 58%, transparent 100%)",
    backgroundSize: "60% 1px",
    backgroundRepeat: "no-repeat",
    backgroundPosition: "-140% 0",
    animation: "radiation-travel 1.4s ease-in-out forwards",
  };
  return (
    <div style={travel}>
      <div style={sheen} />
      {children}
    </div>
  );
}

// useRadiation: a hook returning the inline style for a breathing edge, gated on an `active` flag.
export function useRadiation(active: boolean): CSSProperties {
  return useMemo(() => (active ? RADIATION_ACTIVE : {}), [active]);
}

// ---- Display: Cormorant Garamond, display sizes only. The 032c editorial moment. ----
export function Display({
  children,
  size = 40,
  style,
}: {
  children: ReactNode;
  size?: number;
  style?: CSSProperties;
}) {
  return (
    <h1 className="display" style={{ fontSize: Math.max(28, size), ...style }}>
      {children}
    </h1>
  );
}

// ---- The event-horizon mark: black disk, thin luminous gold rim (the brand glyph). ----
export function EventHorizon({ size = 16 }: { size?: number }) {
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "50%",
        background: "radial-gradient(circle at 50% 42%, #0a0a0b 56%, transparent 60%)",
        boxShadow: "0 0 0 1px var(--radiation), 0 0 6px 0 var(--radiation-bloom), inset 0 0 4px 0 rgba(0,0,0,0.9)",
      }}
    />
  );
}

// ---- StatusPill: bound to RuntimeStatus. Color + label + shape (never color alone). ----
const RUNTIME_STYLE: Record<RuntimeState, { dot: string; label: string }> = {
  down: { dot: "var(--text-low)", label: "Down" },
  booting: { dot: "var(--warning)", label: "Booting" },
  ready: { dot: "var(--success)", label: "Ready" },
  degraded: { dot: "var(--warning)", label: "Degraded" },
  failed: { dot: "var(--danger)", label: "Failed" },
};

export function StatusPill({ status, detail }: { status: RuntimeState; detail?: string | null }) {
  const s = RUNTIME_STYLE[status];
  const breathing = status === "booting";
  return (
    <span
      title={detail ?? undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--s2)",
        padding: "2px 10px",
        borderRadius: 999,
        fontSize: "var(--text-xs)",
        color: "var(--text-mid)",
        boxShadow: "inset 0 0 0 1px var(--rim)",
        background: "var(--surface-1)",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: s.dot,
          boxShadow: status === "ready" ? "0 0 6px 0 var(--success)" : undefined,
          animation: breathing ? "radiation-breathe 1.8s ease-in-out infinite" : undefined,
        }}
      />
      {s.label}
      {detail ? <span style={{ color: "var(--text-low)" }}>{detail}</span> : null}
    </span>
  );
}

// ---- ModeSwitcher: the three surfaces, as a left icon-rail. Glanceable spatial anchor. ----
export type SurfaceMode = "workstation" | "ide" | "chat";
const MODES: { id: SurfaceMode; glyph: string; label: string }[] = [
  { id: "workstation", glyph: "▦", label: "Workstation" },
  { id: "ide", glyph: "‹›", label: "IDE" },
  { id: "chat", glyph: "✎", label: "Chat" },
];

export function ModeSwitcher({ mode, onMode }: { mode: SurfaceMode; onMode: (m: SurfaceMode) => void }) {
  return (
    <nav
      aria-label="Surface"
      style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--s2)", paddingTop: "var(--s3)" }}
    >
      {MODES.map((m) => {
        const on = m.id === mode;
        return (
          <button
            key={m.id}
            title={m.label}
            aria-pressed={on}
            onClick={() => onMode(m.id)}
            style={{
              width: 36,
              height: 36,
              borderRadius: "var(--radius)",
              display: "grid",
              placeItems: "center",
              fontSize: 15,
              color: on ? "var(--radiation-bright)" : "var(--text-low)",
              background: on ? "var(--surface-1)" : "transparent",
              boxShadow: on ? "inset 0 0 0 1px var(--radiation), 0 0 12px -4px var(--radiation-bloom)" : "inset 0 0 0 1px transparent",
            }}
          >
            {m.glyph}
          </button>
        );
      })}
    </nav>
  );
}

// ---- CommandPalette: the keyboard-first spine (Cmd+K). Skeleton: fuzzy list, no commands wired yet. ----
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
    if (!t) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(t));
  }, [q, commands]);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      // focus on next frame so the field is ready
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-label="Command palette"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(6,6,6,0.55)",
        display: "grid",
        placeItems: "start center",
        paddingTop: "12vh",
        zIndex: 100,
      }}
    >
      <Panel
        active
        pad="0"
        style={{ width: "min(620px, 86vw)", overflow: "hidden" }}
      >
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
            style={{
              width: "100%",
              padding: "var(--s4)",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--text-hi)",
              font: "inherit",
              fontSize: "var(--text-lg)",
              boxShadow: "inset 0 -1px 0 0 var(--rim)",
            }}
          />
          <ul style={{ listStyle: "none", margin: 0, padding: "var(--s2)", maxHeight: 320, overflowY: "auto" }}>
            {filtered.length === 0 ? (
              <li style={{ padding: "var(--s3)", color: "var(--text-low)" }}>No commands.</li>
            ) : (
              filtered.map((c, i) => (
                <li key={c.id}>
                  <button
                    onMouseEnter={() => setSel(i)}
                    onClick={() => {
                      c.run();
                      onClose();
                    }}
                    style={{
                      width: "100%",
                      textAlign: "left",
                      padding: "var(--s2) var(--s3)",
                      borderRadius: "var(--radius)",
                      color: i === sel ? "var(--text-hi)" : "var(--text-mid)",
                      background: i === sel ? "var(--surface-2)" : "transparent",
                      boxShadow: i === sel ? "inset 0 0 0 1px var(--radiation)" : undefined,
                    }}
                  >
                    {c.label}
                  </button>
                </li>
              ))
            )}
          </ul>
        </div>
      </Panel>
    </div>
  );
}

// ---- A labeled section header (OP-1 instrument feel): mono caps, low-contrast rule. ----
export function SectionLabel({ children, count }: { children: ReactNode; count?: number }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--s2)",
        fontSize: "var(--text-xs)",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        color: "var(--text-low)",
        padding: "var(--s2) 0",
      }}
    >
      <span>{children}</span>
      {count != null ? <span style={{ color: "var(--text-mid)" }}>({count})</span> : null}
    </div>
  );
}
