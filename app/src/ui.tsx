/*
  ui.tsx: the base design primitives (Doctrine v3, Tadao Ando grayscale concrete).
  Everything references the tokens in theme.css; no hardcoded colors here.
  The single most brand-load-bearing asset is LIGHT entering the dark: a breathing bloom
  for active/streaming and a one-shot travelling sheen for fork/handoff. There is no gold,
  no color, no spinner. Aliveness is light. Borders are shadow-lines, never CSS `border`.
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { RuntimeState } from "./wire";

// ---- Volume / Panel: the poured concrete slab (the .volume recipe). ----
// `alive` wears the light breathe (active/streaming). `raised` is the lighter concrete tier.
export function Volume({
  children,
  alive = false,
  raised = false,
  pad,
  as: As = "div",
  style,
  className,
}: {
  children: ReactNode;
  alive?: boolean;
  raised?: boolean;
  pad?: string;
  as?: "div" | "section" | "aside" | "article";
  style?: CSSProperties;
  className?: string;
}) {
  const cls =
    "volume" +
    (raised ? " volume--raised" : "") +
    (alive ? " alive" : "") +
    (className ? " " + className : "");
  return (
    <As className={cls} style={{ ...(pad != null ? { padding: pad } : null), ...style }}>
      {children}
    </As>
  );
}

// Panel: the historical name the surfaces import; an alias of Volume. `active` -> `alive`.
export function Panel({
  children,
  active = false,
  pad,
  style,
  className,
}: {
  children: ReactNode;
  active?: boolean;
  pad?: string;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <Volume alive={active} pad={pad} style={style} className={className}>
      {children}
    </Volume>
  );
}

/*
  LightEdge: THE accent primitive (replaces RadiationEdge). Two modes:
    - "breathe": the heartbeat bloom on active/streaming elements (light, not gold).
    - "travel":  a one-shot light sheen that sweeps once along the top edge (fork / handoff).
  Wraps its children. No spinner; aliveness is the light. Reduced-motion is handled by
  theme.css (.alive rests already-bloomed); travel falls back to a static lit top edge.
*/
export function LightEdge({
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
    return (
      <div className="alive" style={{ ...base, ...style }}>
        {children}
      </div>
    );
  }
  // travel: a soft light band that sweeps once across the top hairline.
  const sheen: CSSProperties = {
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
    backgroundPosition: "-40% 0",
    animation: "light-travel var(--dur-door) var(--ease) forwards",
  };
  return (
    <div style={{ ...base, boxShadow: "var(--hairline)", ...style }}>
      <div style={sheen} />
      {children}
    </div>
  );
}

// useLight: inline style for a breathing light edge, gated on `active` (replaces useRadiation).
const LIGHT_ACTIVE: CSSProperties = { animation: "breathe var(--breathe) var(--ease) infinite" };
export function useLight(active: boolean): CSSProperties {
  return useMemo(() => (active ? LIGHT_ACTIVE : {}), [active]);
}

// ---- Display: Geist Mono 600 (.t-display). The editorial moment, in concrete, not serif. ----
export function Display({
  children,
  style,
  className,
}: {
  children: ReactNode;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <h1 className={"t-display" + (className ? " " + className : "")} style={{ margin: 0, color: "var(--text-1)", ...style }}>
      {children}
    </h1>
  );
}

// ---- SectionLabel: the OP-1 instrument caps (.t-label). ----
export function SectionLabel({ children, count }: { children: ReactNode; count?: number }) {
  return (
    <div className="t-label" style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)" }}>
      <span>{children}</span>
      {count != null ? <span style={{ color: "var(--text-3)" }}>{count}</span> : null}
    </div>
  );
}

// ---- Mark: the brand glyph. A grayscale concrete disk with a thin LIGHT rim (event horizon). ----
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
        boxShadow: "0 0 0 1px var(--line-strong), var(--light-bloom), inset 0 0 4px 0 rgba(0,0,0,0.9)",
      }}
    />
  );
}

// ---- StatusPill: bound to RuntimeState. Readiness reads as light, never gold. ----
// Each state pairs a glyph + label + tone so meaning never rests on a single channel.
const RUNTIME_STYLE: Record<RuntimeState, { glyph: string; label: string; tone: string }> = {
  down: { glyph: "○", label: "Down", tone: "var(--text-3)" },
  booting: { glyph: "◐", label: "Booting", tone: "var(--text-2)" },
  ready: { glyph: "●", label: "Ready", tone: "var(--light)" },
  degraded: { glyph: "◑", label: "Degraded", tone: "var(--text-2)" },
  failed: { glyph: "✕", label: "Failed", tone: "var(--bad)" },
};

export function StatusPill({ status, detail }: { status: RuntimeState; detail?: string | null }) {
  const s = RUNTIME_STYLE[status];
  const breathing = status === "booting";
  const lit = status === "ready";
  return (
    <span
      title={detail ?? undefined}
      className={breathing ? "alive" : undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "2px 10px",
        borderRadius: "var(--radius-pill)",
        fontSize: "11px",
        fontWeight: 500,
        letterSpacing: "0.04em",
        color: "var(--text-2)",
        boxShadow: lit
          ? "var(--hairline-strong), var(--light-bloom)"
          : "var(--hairline)",
        background: "var(--concrete-2)",
      }}
    >
      <span style={{ color: s.tone, fontSize: "10px", lineHeight: 1 }}>{s.glyph}</span>
      {s.label}
      {detail ? <span style={{ color: "var(--text-3)" }}>{detail}</span> : null}
    </span>
  );
}

// ---- Mode rail: the three surfaces as the quiet west wall. Active mode is marked by light. ----
export type SurfaceMode = "workstation" | "ide" | "chat";
const MODES: { id: SurfaceMode; glyph: string; label: string }[] = [
  { id: "workstation", glyph: "▦", label: "Workstation" }, // the front door / default
  { id: "ide", glyph: "‹›", label: "IDE" },
  { id: "chat", glyph: "✎", label: "Chat" },
];

export function ModeRail({ mode, onMode }: { mode: SurfaceMode; onMode: (m: SurfaceMode) => void }) {
  return (
    <nav
      aria-label="Surface"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "var(--ma-3)",
        paddingTop: "var(--ma-6)",
      }}
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
              color: on ? "var(--light)" : "var(--text-3)",
              background: on ? "var(--concrete-3)" : "transparent",
              boxShadow: on ? "var(--hairline-strong), var(--light-bloom)" : "none",
              transition: "color var(--dur) var(--ease), box-shadow var(--dur) var(--ease)",
            }}
          >
            {m.glyph}
          </button>
        );
      })}
    </nav>
  );
}

// ModeSwitcher: the historical name the surfaces import; an alias of ModeRail.
export const ModeSwitcher = ModeRail;

// ---- Gate: the approval capsule (the .gate recipe). A lit threshold the user must cross. ----
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

// ---- CommandPalette: the keyboard-first threshold (Cmd+K). Fuzzy list over wired commands. ----
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
        background: "rgba(7,7,7,0.6)",
        display: "grid",
        placeItems: "start center",
        paddingTop: "14vh",
        zIndex: 100,
      }}
    >
      <Volume
        alive
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
            className="t-body"
            style={{
              width: "100%",
              padding: "var(--ma-4)",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--text-1)",
              fontFamily: "var(--font)",
              fontSize: "15px",
              boxShadow: "inset 0 -1px 0 0 var(--line)",
            }}
          />
          <ul style={{ listStyle: "none", margin: 0, padding: "var(--ma-2)", maxHeight: 340, overflowY: "auto" }}>
            {filtered.length === 0 ? (
              <li className="t-body" style={{ padding: "var(--ma-3)", color: "var(--text-3)" }}>No commands.</li>
            ) : (
              filtered.map((c, i) => (
                <li key={c.id}>
                  <button
                    className="t-body"
                    onMouseEnter={() => setSel(i)}
                    onClick={() => {
                      c.run();
                      onClose();
                    }}
                    style={{
                      width: "100%",
                      textAlign: "left",
                      padding: "var(--ma-2) var(--ma-3)",
                      borderRadius: "var(--radius)",
                      color: i === sel ? "var(--text-1)" : "var(--text-2)",
                      background: i === sel ? "var(--concrete-3)" : "transparent",
                      boxShadow: i === sel ? "var(--hairline)" : "none",
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
