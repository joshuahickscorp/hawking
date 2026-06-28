/*
  parts.tsx: the tactile hardware of the Context Stack (the OP-1 / patch-bay feel, doctrine C7).
  Every affordance here is a material control: a recessed channel, a hairline rim, a gold lit
  state when engaged. No flat buttons, no neon. These are LOCAL primitives (parallel-safety:
  the surface owns them; ui.tsx is not touched). They re-house the VS Code SCM "tree-row hover
  action" pattern into designed hardware: instead of ghost icons on hover, every row carries
  visible labeled toggles that read as physical switches you can always see and press.
*/
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useRadiation } from "../../ui";

/* A row inside a stratum: a thin recessed channel, content + trailing controls. */
export function Line({ children, onClick, title }: { children: ReactNode; onClick?: () => void; title?: string }) {
  const base: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "var(--s2)",
    padding: "var(--s1) var(--s2)",
    borderRadius: "var(--radius)",
    minWidth: 0,
  };
  if (!onClick) return <div style={base}>{children}</div>;
  return (
    <button
      title={title}
      onClick={onClick}
      style={{ ...base, width: "100%", textAlign: "left", color: "inherit" }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface-1)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      {children}
    </button>
  );
}

/*
  HardwareToggle: the signature control. A small labeled switch that reads as a physical
  module on a patch bay. Off = recessed dark with a hairline rim. On = lit gold with a soft
  bloom, the engaged state unmistakable. Tactile feedback is a quick depress on press, never a bounce.
*/
export function HardwareToggle({
  label,
  on,
  onToggle,
  tone = "gold",
  title,
}: {
  label: string;
  on: boolean;
  onToggle: () => void;
  tone?: "gold" | "mute" | "danger";
  title?: string;
}) {
  const [down, setDown] = useState(false);
  const lit =
    tone === "danger" ? "var(--danger)" : tone === "mute" ? "var(--text-mid)" : "var(--radiation-bright)";
  return (
    <button
      title={title ?? label}
      aria-pressed={on}
      onMouseDown={() => setDown(true)}
      onMouseUp={() => setDown(false)}
      onMouseLeave={() => setDown(false)}
      onClick={onToggle}
      style={{
        flex: "0 0 auto",
        fontSize: "var(--text-xs)",
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        padding: "1px 7px",
        borderRadius: "var(--radius)",
        color: on ? "var(--void)" : "var(--text-low)",
        background: on ? lit : "var(--surface-1)",
        boxShadow: on
          ? `inset 0 1px 0 0 rgba(255,255,255,0.25), 0 0 10px -3px var(--radiation-bloom)`
          : "inset 0 0 0 1px var(--rim), inset 0 1px 0 0 rgba(255,255,255,0.02)",
        transform: down ? "translateY(1px)" : "none",
        transition: "transform 80ms ease, background 120ms ease, color 120ms ease",
      }}
    >
      {label}
    </button>
  );
}

/*
  Stratum: a labeled layer of the stack. Calm summary by default; the whole header is the
  expand control. Expanding opens the inspector body with WEIGHT (height + opacity eased in),
  never a pop (doctrine: "slides and grows with mass"). The live stratum wears the breathing
  gold edge via `live`. Header carries a count and an optional trailing control slot.
*/
export function Stratum({
  label,
  count,
  live = false,
  defaultOpen = false,
  summary,
  trailing,
  children,
}: {
  label: string;
  count?: number;
  live?: boolean;
  defaultOpen?: boolean;
  summary?: ReactNode; // one-line glance shown collapsed
  trailing?: ReactNode; // a control that lives in the header (e.g. a global mute)
  children?: ReactNode; // the inspector body, revealed on expand
}) {
  const [open, setOpen] = useState(defaultOpen);
  const breathe = useRadiation(live);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [h, setH] = useState(0);

  // Measure the natural body height so the expand eases to an exact target (weighted, not jumpy).
  useEffect(() => {
    if (open && bodyRef.current) setH(bodyRef.current.scrollHeight);
  }, [open, children, count]);

  const hasBody = children != null;
  return (
    <section
      className="panel"
      style={{
        padding: "var(--s2) var(--s3)",
        ...breathe,
        ...(live ? { boxShadow: undefined } : null), // let the breathe animation own the edge
      }}
    >
      <header style={{ display: "flex", alignItems: "center", gap: "var(--s2)" }}>
        <button
          onClick={() => hasBody && setOpen((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--s2)",
            flex: 1,
            minWidth: 0,
            color: "var(--text-low)",
            cursor: hasBody ? "pointer" : "default",
          }}
        >
          {hasBody ? (
            <span
              aria-hidden
              style={{
                fontSize: 9,
                color: live ? "var(--radiation-bright)" : "var(--text-low)",
                transform: open ? "rotate(90deg)" : "none",
                transition: "transform 220ms cubic-bezier(0.2, 0.7, 0.2, 1)",
              }}
            >
              ▸
            </span>
          ) : (
            <span style={{ width: 9, color: live ? "var(--radiation-bright)" : "var(--text-low)" }}>▪</span>
          )}
          <span style={{ fontSize: "var(--text-xs)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {label}
          </span>
          {count != null ? <span style={{ color: "var(--text-mid)", fontSize: "var(--text-xs)" }}>({count})</span> : null}
          {!open && summary ? (
            <span
              style={{
                marginLeft: "auto",
                color: "var(--text-low)",
                fontSize: "var(--text-xs)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {summary}
            </span>
          ) : null}
        </button>
        {trailing ? <div style={{ flex: "0 0 auto" }}>{trailing}</div> : null}
      </header>

      {hasBody ? (
        <div
          style={{
            height: open ? h : 0,
            opacity: open ? 1 : 0,
            overflow: "hidden",
            transition: "height 260ms cubic-bezier(0.2, 0.7, 0.2, 1), opacity 220ms ease",
          }}
        >
          <div ref={bodyRef} style={{ paddingTop: "var(--s2)", display: "flex", flexDirection: "column", gap: 1 }}>
            {children}
          </div>
        </div>
      ) : null}
    </section>
  );
}

/*
  NoteField: inject a note into context (the fourth touch verb). A recessed input that commits
  on Enter and clears, the way an instrument latches a value. Calm, no submit button chrome.
*/
export function NoteField({ value, onCommit, placeholder }: { value?: string; onCommit: (text: string) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--s1)" }}>
      {value ? (
        <div
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--text-mid)",
            padding: "var(--s1) var(--s2)",
            borderRadius: "var(--radius)",
            boxShadow: "inset 0 0 0 1px var(--radiation)",
            background: "var(--surface-1)",
          }}
        >
          {value}
        </div>
      ) : null}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && draft.trim()) {
            onCommit(draft.trim());
            setDraft("");
          }
        }}
        placeholder={placeholder}
        style={{
          width: "100%",
          padding: "var(--s1) var(--s2)",
          fontSize: "var(--text-xs)",
          color: "var(--text-hi)",
          background: "var(--void)",
          border: "none",
          borderRadius: "var(--radius)",
          outline: "none",
          boxShadow: "inset 0 0 0 1px var(--rim)",
          font: "inherit",
        }}
      />
    </div>
  );
}

/* A small ok/fail marker, shape + color (never color alone), per doctrine. */
export function Mark({ ok }: { ok: boolean }) {
  return (
    <span
      aria-label={ok ? "ok" : "fail"}
      style={{
        flex: "0 0 auto",
        width: 14,
        textAlign: "center",
        fontSize: "var(--text-xs)",
        color: ok ? "var(--success)" : "var(--danger)",
      }}
    >
      {ok ? "✓" : "✗"}
    </span>
  );
}
