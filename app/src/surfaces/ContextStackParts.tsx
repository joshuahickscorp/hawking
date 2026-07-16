/*
  ContextStackParts.tsx: primitives of the Context tree and Explorer sidebar.
  Each Stratum is a collapsible tree SECTION (a 22px header row with a twistie chevron, an uppercase
  section label, an optional count). Rows (Line) are compact ~22px tree rows. Flat + system-font +
  VS Code-toned: no glows, no breathing, no bloom. Liveness is a small accent dot, nothing more.
  These are LOCAL primitives (the surface owns them; ui.tsx is not touched). Every prop is preserved
  so ContextStack.tsx — and the steer logic — keep working unchanged.
*/
import { useState, type CSSProperties, type ReactNode } from "react";

/* A compact tree row: ~22px, content + trailing controls, hover background. */
export function Line({ children, onClick, title }: { children: ReactNode; onClick?: () => void; title?: string }) {
  const base: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "var(--ma-2)",
    minHeight: 22,
    padding: "0 var(--ma-2) 0 22px",
    fontSize: "var(--fs-ui)",
    minWidth: 0,
  };
  if (!onClick) return <div className="ctx-row" style={base}>{children}</div>;
  return (
    <button
      className="ctx-row ctx-row--btn"
      title={title}
      onClick={onClick}
      style={{ ...base, width: "100%", textAlign: "left", color: "inherit", background: "transparent" }}
    >
      {children}
    </button>
  );
}

/*
  HardwareToggle: a small labeled toggle that sits at the trailing edge of a tree row. Flat VS Code
  toning: dim uppercase label at rest, brighter on hover, accent (or a destructive red for `tone="bad"`)
  when engaged. No bloom, no depress bounce — just a quiet background + color shift.
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
  const litColor = tone === "bad" ? "var(--red)" : "var(--accent)";
  return (
    <button
      title={title ?? label}
      aria-pressed={on}
      onClick={onToggle}
      className="ctx-toggle"
      style={{
        flex: "0 0 auto",
        fontSize: "var(--fs-label)",
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        padding: "1px 6px",
        borderRadius: "var(--radius-sm)",
        color: on ? "var(--accent-text)" : "var(--text-dim)",
        background: on ? litColor : "transparent",
        border: "1px solid",
        borderColor: on ? "transparent" : "var(--input-border)",
      }}
    >
      {label}
    </button>
  );
}

/*
  Stratum: a collapsible Explorer SECTION. The 22px header row carries a twistie chevron
  (▾ open / ▸ collapsed), an uppercase 11px section label, an optional count, and (collapsed) a quiet
  one-line summary. Clicking the header toggles. The live section shows a small accent dot, no breathing.
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
  children?: ReactNode; // the section body, revealed on expand
}) {
  const [open, setOpen] = useState(defaultOpen);
  const hasBody = children != null;

  return (
    <section className="ctx-section">
      <header className="ctx-section__head" style={{ display: "flex", alignItems: "center", gap: "var(--ma-1)" }}>
        <button
          className="ctx-section__toggle"
          onClick={() => hasBody && setOpen((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--ma-1)",
            flex: 1,
            minWidth: 0,
            height: 22,
            color: "var(--text-muted)",
            background: "transparent",
            cursor: hasBody ? "pointer" : "default",
          }}
        >
          <span
            aria-hidden
            className="ctx-twistie"
            style={{
              flex: "0 0 auto",
              width: 16,
              textAlign: "center",
              fontSize: "var(--fs-label)",
              color: "var(--text-muted)",
              visibility: hasBody ? "visible" : "hidden",
            }}
          >
            {open ? "▾" : "▸"}
          </span>
          <span
            className="ctx-section__label"
            style={{
              fontSize: "var(--fs-label)",
              fontWeight: 700,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
            }}
          >
            {label}
          </span>
          {live ? (
            <span
              aria-label="active"
              title="agent active"
              style={{ flex: "0 0 auto", width: 6, height: 6, borderRadius: "50%", background: "var(--accent)" }}
            />
          ) : null}
          {count != null ? (
            <span style={{ fontSize: "var(--fs-label)", color: "var(--text-dim)" }}>{count}</span>
          ) : null}
          {!open && summary ? (
            <span
              style={{
                marginLeft: "auto",
                fontSize: "var(--fs-label)",
                color: "var(--text-dim)",
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

      {hasBody && open ? (
        <div className="ctx-section__body" style={{ display: "flex", flexDirection: "column" }}>
          {children}
        </div>
      ) : null}
    </section>
  );
}

/*
  NoteField: inject a note into context. A flat VS Code input that commits on Enter and clears.
  The already-committed value rests above it in a quiet card.
*/
export function NoteField({ value, onCommit, placeholder }: { value?: string; onCommit: (text: string) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-1)", padding: "var(--ma-1) var(--ma-2) var(--ma-1) 22px" }}>
      {value ? (
        <div
          style={{
            fontSize: "var(--fs-small)",
            color: "var(--text)",
            padding: "var(--ma-1) var(--ma-2)",
            borderRadius: "var(--radius-sm)",
            background: "var(--surface-2)",
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
        className="ctx-note-input"
        style={{
          width: "100%",
          padding: "3px var(--ma-2)",
          color: "var(--text)",
          background: "var(--input-bg)",
          border: "1px solid var(--input-border)",
          borderRadius: "var(--radius-sm)",
          outline: "none",
          font: "inherit",
          fontSize: "var(--fs-small)",
        }}
      />
    </div>
  );
}

/*
  OkMark: a small ok/fail marker, shape + pigment (never pigment alone). Green check for ok,
  red cross for fail.
*/
export function OkMark({ ok }: { ok: boolean }) {
  return (
    <span
      aria-label={ok ? "ok" : "fail"}
      style={{
        flex: "0 0 auto",
        width: 14,
        textAlign: "center",
        fontSize: "var(--fs-small)",
        color: ok ? "var(--green)" : "var(--red)",
      }}
    >
      {ok ? "✓" : "✗"}
    </span>
  );
}
