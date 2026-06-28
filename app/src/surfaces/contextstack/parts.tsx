/*
  parts.tsx: the tactile hardware of the Context Stack (Doctrine v3, the OP-1 instrument inside
  an Ando chamber). Every affordance here is a quiet TE line-glyph control on raw concrete: dim at
  rest, brighter on hover, and when engaged it lights with LIGHT, never gold. These are LOCAL
  primitives (parallel-safety: the surface owns them; ui.tsx is not touched). They re-house the
  VS Code SCM "tree-row hover action" pattern into restrained hardware: instead of ghost icons,
  each row carries a labeled toggle you can always read, that reads as a switch you can press.
*/
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useLight } from "../../ui";

/* A row inside a stratum: a thin recessed channel, content + trailing controls. */
export function Line({ children, onClick, title }: { children: ReactNode; onClick?: () => void; title?: string }) {
  const base: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "var(--ma-3)",
    padding: "var(--ma-2) var(--ma-3)",
    borderRadius: "var(--radius)",
    minWidth: 0,
  };
  if (!onClick) return <div style={base}>{children}</div>;
  return (
    <button
      title={title}
      onClick={onClick}
      style={{ ...base, width: "100%", textAlign: "left", color: "inherit" }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--concrete-3)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      {children}
    </button>
  );
}

/*
  HardwareToggle: the signature "let me touch it" control. A small labeled line-glyph switch.
  Off = a recessed dark channel with a hairline rim, the label dim (--mute). Hover lifts the label
  toward --text-2. On = the switch lights with LIGHT (a soft bloom + lit label), the engaged state
  unmistakable without any hue. The lone semantic exception is `tone="bad"`, the oxide pigment used
  only where the action is genuinely destructive (evict), and even then glyph-paired by its label.
  Tactile feedback is a quick depress on press, never a bounce.
*/
export function HardwareToggle({
  label,
  on,
  onToggle,
  tone = "light",
  title,
}: {
  label: string;
  on: boolean;
  onToggle: () => void;
  tone?: "light" | "mute" | "bad";
  title?: string;
}) {
  const [down, setDown] = useState(false);
  const [hover, setHover] = useState(false);
  // engaged tone: LIGHT by default; oxide only where the action is genuinely destructive.
  const litColor = tone === "bad" ? "var(--bad)" : "var(--light)";
  const litBloom = tone === "bad" ? "0 0 10px -3px var(--bad)" : "var(--light-bloom)";
  const restColor = on ? litColor : hover ? "var(--text-2)" : "var(--mute)";
  return (
    <button
      title={title ?? label}
      aria-pressed={on}
      onMouseDown={() => setDown(true)}
      onMouseUp={() => setDown(false)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => {
        setDown(false);
        setHover(false);
      }}
      onClick={onToggle}
      className="t-micro"
      style={{
        flex: "0 0 auto",
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        padding: "1px 7px",
        borderRadius: "var(--radius)",
        color: restColor,
        background: on ? "var(--concrete-4)" : hover ? "var(--concrete-3)" : "transparent",
        boxShadow: on
          ? `var(--hairline-strong), ${litBloom}, var(--inner-glow)`
          : "var(--hairline)",
        transform: down ? "translateY(1px)" : "none",
        transition: "transform 80ms var(--ease), background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease)",
      }}
    >
      {label}
    </button>
  );
}

/*
  Stratum: a labeled ledge of the stack, poured as a .volume slab resting in the void. Calm summary
  by default; the whole header is the expand control. Expanding opens the inspector body with WEIGHT
  (height + opacity eased in), the weighted disclosure of a heavy door, never a pop. The live stratum
  wears the breathing LIGHT via `live` (useLight), not gold. Header carries a count and an optional
  trailing control slot.
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
  const breathe = useLight(live);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [h, setH] = useState(0);

  // Measure the natural body height so the expand eases to an exact target (weighted, not jumpy).
  useEffect(() => {
    if (open && bodyRef.current) setH(bodyRef.current.scrollHeight);
  }, [open, children, count]);

  const hasBody = children != null;
  const glyphColor = live ? "var(--light)" : "var(--text-3)";
  return (
    <section
      className={"volume" + (live ? " alive" : "")}
      style={{
        padding: "var(--ma-6)",
        ...breathe,
      }}
    >
      <header style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)" }}>
        <button
          onClick={() => hasBody && setOpen((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--ma-3)",
            flex: 1,
            minWidth: 0,
            color: "var(--mute)",
            cursor: hasBody ? "pointer" : "default",
          }}
        >
          {hasBody ? (
            <span
              aria-hidden
              style={{
                fontSize: 9,
                color: glyphColor,
                transform: open ? "rotate(90deg)" : "none",
                transition: "transform var(--dur) var(--ease)",
              }}
            >
              ▸
            </span>
          ) : (
            <span aria-hidden style={{ width: 9, fontSize: 9, color: glyphColor }}>▪</span>
          )}
          <span className="t-label">{label}</span>
          {count != null ? <span className="t-micro" style={{ color: "var(--text-3)" }}>{count}</span> : null}
          {!open && summary ? (
            <span
              className="t-micro"
              style={{
                marginLeft: "auto",
                color: "var(--text-3)",
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
            transition: "height var(--dur-slow) var(--ease), opacity var(--dur) var(--ease)",
          }}
        >
          <div ref={bodyRef} style={{ paddingTop: "var(--ma-3)", display: "flex", flexDirection: "column", gap: 1 }}>
            {children}
          </div>
        </div>
      ) : null}
    </section>
  );
}

/*
  NoteField: inject a note into context (the fourth touch verb, the @-add). A recessed input that
  commits on Enter and clears, the way an instrument latches a value. Calm, no submit chrome. The
  already-committed value rests inside a faintly lit channel (LIGHT inner-glow, never a colored rim).
*/
export function NoteField({ value, onCommit, placeholder }: { value?: string; onCommit: (text: string) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
      {value ? (
        <div
          className="t-micro"
          style={{
            color: "var(--text-2)",
            padding: "var(--ma-2) var(--ma-3)",
            borderRadius: "var(--radius)",
            boxShadow: "var(--hairline), var(--inner-glow)",
            background: "var(--concrete-3)",
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
        className="t-micro"
        style={{
          width: "100%",
          padding: "var(--ma-2) var(--ma-3)",
          color: "var(--text-1)",
          background: "var(--void)",
          border: "none",
          borderRadius: "var(--radius)",
          outline: "none",
          boxShadow: "var(--hairline)",
          font: "inherit",
          fontSize: "11px",
        }}
      />
    </div>
  );
}

/*
  OkMark: a small ok/fail marker, shape + pigment (never pigment alone), per doctrine. The only two
  colors, glyph-paired: lichen check for ok, oxide cross for fail. (Named OkMark to leave the brand
  glyph `Mark` in ui.tsx unshadowed; this is the semantic state marker, not the event-horizon disk.)
*/
export function OkMark({ ok }: { ok: boolean }) {
  return (
    <span
      aria-label={ok ? "ok" : "fail"}
      style={{
        flex: "0 0 auto",
        width: 14,
        textAlign: "center",
        fontSize: "12px",
        color: ok ? "var(--ok)" : "var(--bad)",
      }}
    >
      {ok ? "✓" : "✗"}
    </span>
  );
}
