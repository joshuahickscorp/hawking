import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useFocusTrap } from "./shell/a11y";

// Shared UI primitives still used by the VS Code shell. (The old doctrine primitives —
// Volume/Mark/LightEdge/ModeRail/StatusPill — were retired with the concrete design.)

export function Display({ children, style, className }: { children: ReactNode; style?: CSSProperties; className?: string }) {
  return (
    <h1 className={["t-display", className].filter(Boolean).join(" ")} style={style}>
      {children}
    </h1>
  );
}

// Primary (accent) button — VS Code button.background.
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

// Quick Open / command palette (Cmd+P).
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
  const trapRef = useFocusTrap<HTMLDivElement>(open);

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
    <div role="presentation" className="palette-overlay" onClick={onClose}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette" ref={trapRef} onClick={(e) => e.stopPropagation()}>
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
            <li className="t-body" style={{ padding: "var(--ma-3)", color: "var(--text-dim)" }}>No commands</li>
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
    </div>
  );
}
