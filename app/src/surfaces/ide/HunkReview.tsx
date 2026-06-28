/*
  HunkReview.tsx: THE core gesture, factored as a reusable local component (v3). Per-hunk review of
  an agent's proposed diff, driven by keyboard:
    j / ArrowDown  -> next hunk        k / ArrowUp  -> prev hunk
    a / Cmd+Enter  -> accept hunk      r / Cmd+Backspace -> reject hunk
  Accept dispatches AcceptDiff{run_id,diff_id}; reject dispatches RejectDiff. The accepted hunk settles
  with a brief LIGHT settle ("the change is taken in", light entering then quieting); the rejected one
  dissolves back out. Diff add/del use the .hunk-add/.hunk-del classes (lichen/oxide) AND a +/- marker,
  never color alone (accessibility: color is never the sole signal).

  This same gesture must be identical in the Workstation merge-review (the review gesture must be the
  same everywhere or the seams show), so this component is the single source of it: it takes a DiffDoc +
  an onAct callback and owns nothing surface-specific. Re-housed from the Cline hunk-accept UX, recast
  in v3 grayscale concrete: the selected hunk wears a LightEdge breathe, never a colored rim.
*/
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import type { DiffDoc, Hunk } from "./types";

export type HunkAction = "accept" | "reject";

// add/del carry the diff pigment via class (.hunk-add/.hunk-del = --ok/--bad + bg); ctx is neutral.
// Markers are always present so the meaning never rests on color alone.
const KIND_STYLE = {
  add: { cls: "hunk-add", marker: "+" },
  del: { cls: "hunk-del", marker: "-" },
  ctx: { cls: "", marker: " " },
} as const;

const STATUS_LABEL: Record<Hunk["status"], { label: string; color: string }> = {
  pending: { label: "pending", color: "var(--text-3)" },
  accepted: { label: "accepted", color: "var(--ok)" },
  rejected: { label: "rejected", color: "var(--bad)" },
  applied: { label: "applied", color: "var(--ok)" },
};

export function HunkReview({
  doc,
  onAct,
  active = true,
}: {
  doc: DiffDoc;
  // Called with the hunk + action so the host caller emits AcceptDiff/RejectDiff (or a merge choice).
  onAct: (hunk: Hunk, action: HunkAction) => void;
  active?: boolean; // when false, keyboard handlers detach (the panel is not focused)
}) {
  const [sel, setSel] = useState(0);
  // hunks that just settled, for the one-shot light settle, keyed by id -> action.
  const [settling, setSettling] = useState<Record<string, HunkAction>>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const hunkRefs = useRef<(HTMLDivElement | null)[]>([]);

  const pendingIdx = useMemo(
    () => doc.hunks.map((h, i) => (h.status === "pending" ? i : -1)).filter((i) => i >= 0),
    [doc.hunks],
  );

  const act = useCallback(
    (idx: number, action: HunkAction) => {
      const hunk = doc.hunks[idx];
      if (!hunk || hunk.status !== "pending") return;
      setSettling((s) => ({ ...s, [hunk.id]: action }));
      onAct(hunk, action);
      // advance to the next still-pending hunk so the flow stays on the keyboard.
      const next = pendingIdx.find((i) => i > idx);
      if (next != null) setSel(next);
    },
    [doc.hunks, onAct, pendingIdx],
  );

  const move = useCallback(
    (delta: number) => {
      setSel((i) => Math.min(Math.max(i + delta, 0), doc.hunks.length - 1));
    },
    [doc.hunks.length],
  );

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      // Cmd+Enter / Cmd+Backspace are the doctrine seed-keymap bindings for accept/reject hunk.
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        act(sel, "accept");
        return;
      }
      if ((e.metaKey || e.ctrlKey) && (e.key === "Backspace" || e.key === "Delete")) {
        e.preventDefault();
        act(sel, "reject");
        return;
      }
      if (e.metaKey || e.ctrlKey) return; // leave other chords alone
      switch (e.key) {
        case "j":
        case "ArrowDown":
          e.preventDefault();
          move(1);
          break;
        case "k":
        case "ArrowUp":
          e.preventDefault();
          move(-1);
          break;
        case "a":
          e.preventDefault();
          act(sel, "accept");
          break;
        case "r":
          e.preventDefault();
          act(sel, "reject");
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, sel, act, move]);

  // keep the selected hunk in view as j/k walks the list.
  useEffect(() => {
    hunkRefs.current[sel]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [sel]);

  const remaining = pendingIdx.length;

  return (
    <div ref={rootRef} style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <LocalSettleKeyframes />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-3)",
          padding: "var(--ma-3) var(--ma-4)",
          boxShadow: "inset 0 -1px 0 0 var(--line)",
          fontSize: "12px",
          color: "var(--text-3)",
        }}
      >
        <span className="t-code" style={{ color: "var(--text-2)" }}>{doc.path}</span>
        {doc.stale ? (
          // 'needs you' state with a GLYPH + neutral text (no orange/warning hue exists in v3).
          <span
            title="The file changed under this pending diff. Re-sync before applying."
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "var(--ma-1)",
              color: "var(--text-2)",
              boxShadow: "var(--hairline-strong)",
              borderRadius: "var(--radius)",
              padding: "1px 8px",
            }}
          >
            <span aria-hidden style={{ color: "var(--text-2)" }}>△</span>
            stale
          </span>
        ) : null}
        <span style={{ marginLeft: "auto" }}>
          {remaining} hunk{remaining === 1 ? "" : "s"} to review
        </span>
        <kbd style={kbd}>j</kbd>
        <kbd style={kbd}>k</kbd>
        <span>move</span>
        <kbd style={kbd}>a</kbd>
        <span>accept</span>
        <kbd style={kbd}>r</kbd>
        <span>reject</span>
      </div>

      <div style={{ overflowY: "auto", padding: "var(--ma-4)", display: "flex", flexDirection: "column", gap: "var(--ma-4)", minHeight: 0 }}>
        {doc.hunks.map((h, i) => (
          <HunkCard
            key={h.id}
            hunk={h}
            selected={i === sel}
            settle={settling[h.id]}
            innerRef={(el) => (hunkRefs.current[i] = el)}
            onSelect={() => setSel(i)}
            onAccept={() => act(i, "accept")}
            onReject={() => act(i, "reject")}
          />
        ))}
        {doc.hunks.length === 0 ? (
          <div className="t-body" style={{ color: "var(--text-3)", padding: "var(--ma-6)" }}>No hunks in this change</div>
        ) : null}
      </div>
    </div>
  );
}

function HunkCard({
  hunk,
  selected,
  settle,
  innerRef,
  onSelect,
  onAccept,
  onReject,
}: {
  hunk: Hunk;
  selected: boolean;
  settle?: HunkAction;
  innerRef: (el: HTMLDivElement | null) => void;
  onSelect: () => void;
  onAccept: () => void;
  onReject: () => void;
}) {
  const decided = hunk.status !== "pending";
  const st = STATUS_LABEL[hunk.status];
  // The settle: accept lets light enter then quiet; reject fades the card out (no gold anywhere).
  const settleStyle =
    settle === "accept"
      ? { animation: "hunk-light-settle 620ms var(--ease)" }
      : settle === "reject"
        ? { animation: "hunk-dissolve 480ms var(--ease) forwards" }
        : {};
  // The selected, still-pending hunk breathes light (the .alive heartbeat = LightEdge breathe).
  const breathing = selected && !decided && !settle;

  return (
    <div
      ref={innerRef}
      onClick={onSelect}
      className={"volume" + (breathing ? " alive" : "")}
      style={{
        padding: 0,
        overflow: "hidden",
        opacity: decided && settle !== "accept" ? 0.55 : 1,
        boxShadow: breathing ? undefined : selected ? "var(--hairline-strong), var(--inner-glow)" : "var(--hairline), var(--inner-glow)",
        ...settleStyle,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-2)",
          padding: "var(--ma-2) var(--ma-4)",
          boxShadow: "inset 0 -1px 0 0 var(--line)",
          fontSize: "12px",
          color: "var(--text-3)",
        }}
      >
        <span className="t-code" style={{ color: "var(--text-2)" }}>{hunk.header}</span>
        <span style={{ marginLeft: "auto", color: st.color }}>{st.label}</span>
      </div>

      <div className="t-code" style={{ lineHeight: 1.7 }}>
        {hunk.lines.map((ln, j) => {
          const ks = KIND_STYLE[ln.kind];
          const ctx = ln.kind === "ctx";
          return (
            <div key={j} className={ks.cls} style={{ display: "flex" }}>
              <span style={gutterNo}>{ln.oldNo ?? ""}</span>
              <span style={gutterNo}>{ln.newNo ?? ""}</span>
              <span style={{ width: 16, textAlign: "center", userSelect: "none", color: ctx ? "var(--text-3)" : undefined }}>{ks.marker}</span>
              <span style={{ color: ctx ? "var(--text-2)" : undefined, whiteSpace: "pre", paddingRight: "var(--ma-4)" }}>
                {ln.text || " "}
              </span>
            </div>
          );
        })}
      </div>

      {!decided ? (
        <div style={{ display: "flex", gap: "var(--ma-2)", padding: "var(--ma-3) var(--ma-4)", boxShadow: "inset 0 1px 0 0 var(--line)" }}>
          <ActBtn label="Accept" hint="a" tone="accept" onClick={(e) => { e.stopPropagation(); onAccept(); }} />
          <ActBtn label="Reject" hint="r" tone="reject" onClick={(e) => { e.stopPropagation(); onReject(); }} />
        </div>
      ) : null}
    </div>
  );
}

function ActBtn({
  label,
  hint,
  tone,
  onClick,
}: {
  label: string;
  hint: string;
  tone: "accept" | "reject";
  onClick: (e: MouseEvent) => void;
}) {
  const accent = tone === "accept" ? "var(--ok)" : "var(--bad)";
  return (
    <button
      className="t-label"
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "var(--ma-1) var(--ma-3)",
        borderRadius: "var(--radius)",
        color: "var(--text-2)",
        background: "var(--concrete-3)",
        boxShadow: "var(--hairline)",
        textTransform: "none",
        letterSpacing: "0.02em",
      }}
    >
      <span style={{ color: accent }}>{tone === "accept" ? "+" : "-"}</span>
      {label}
      <kbd style={kbd}>{hint}</kbd>
    </button>
  );
}

/*
  The two one-shot diff-settle keyframes the absorption uses, recast to LIGHT (the retired gold
  hunk-absorb / hunk-dissolve are gone). theme.css owns the shared breathe / light-travel keyframes
  and is off-limits to edit, so these surface-local animations live here. They honor
  prefers-reduced-motion via theme.css's global reduce rule.
*/
function LocalSettleKeyframes() {
  return (
    <style>{`
      @keyframes hunk-light-settle {
        0%   { box-shadow: var(--inner-glow), var(--light-bloom); }
        100% { box-shadow: var(--hairline), var(--inner-glow); }
      }
      @keyframes hunk-dissolve {
        0%   { opacity: 1; transform: translateY(0); }
        100% { opacity: 0.4; transform: translateY(-2px); }
      }
    `}</style>
  );
}

const gutterNo: CSSProperties = {
  width: 36,
  flexShrink: 0,
  textAlign: "right",
  paddingRight: 8,
  color: "var(--text-3)",
  userSelect: "none",
  fontSize: "11px",
};

const kbd: CSSProperties = {
  fontFamily: "var(--font)",
  fontSize: 10,
  padding: "1px 5px",
  borderRadius: 3,
  color: "var(--text-2)",
  boxShadow: "var(--hairline-strong)",
  background: "var(--concrete-3)",
};
