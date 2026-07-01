/*
  HunkReview.tsx: the core per-hunk diff-review gesture, a reusable local component. Keyboard-driven:
    j / ArrowDown  -> next hunk        k / ArrowUp  -> prev hunk
    a / Cmd+Enter  -> accept hunk      r / Cmd+Backspace -> reject hunk
  Accept dispatches AcceptDiff{run_id,diff_id}; reject dispatches RejectDiff and fades the card out.
  Diff add/del use the .hunk-add/.hunk-del classes (VS Code green/red) AND a +/- marker, never color
  alone (accessibility: color is never the sole signal). The selected pending hunk wears a flat 1px
  accent border (VS Code style), no glow or animation. Takes a DiffDoc + an onAct callback and owns
  nothing surface-specific.
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
  pending: { label: "pending", color: "var(--text-dim)" },
  accepted: { label: "accepted", color: "var(--git-add)" },
  rejected: { label: "rejected", color: "var(--git-del)" },
  applied: { label: "applied", color: "var(--git-add)" },
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
    <div
      ref={rootRef}
      style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, width: "100%", minWidth: 0, overflow: "hidden" }}
    >
      <LocalSettleKeyframes />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-2)",
          flexWrap: "wrap",
          rowGap: "var(--ma-1)",
          padding: "var(--ma-2) var(--ma-3)",
          boxShadow: "inset 0 -1px 0 0 var(--border)",
          fontSize: "12px",
          color: "var(--text-muted)",
          minWidth: 0,
        }}
      >
        <span className="t-code" style={{ color: "var(--text)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{doc.path}</span>
        {doc.stale ? (
          <span
            title="The file changed under this pending diff. Re-sync before applying."
            aria-label="Diff is stale; re-sync before applying"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "var(--ma-1)",
              color: "var(--git-mod)",
              border: "1px solid var(--border-strong)",
              borderRadius: "var(--radius-sm)",
              padding: "0 6px",
              flex: "0 0 auto",
            }}
          >
            <span aria-hidden>△</span>
            stale
          </span>
        ) : null}
        <span role="status" aria-live="polite" style={{ marginLeft: "auto", flex: "0 0 auto" }}>
          {remaining} hunk{remaining === 1 ? "" : "s"} to review
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--ma-1)", flex: "0 0 auto", flexWrap: "wrap" }}>
          <kbd style={kbd}>j</kbd>
          <kbd style={kbd}>k</kbd>
          <span>move</span>
          <kbd style={kbd}>a</kbd>
          <span>accept</span>
          <kbd style={kbd}>r</kbd>
          <span>reject</span>
        </span>
      </div>

      <div role="list" aria-label={`Diff hunks for ${doc.path}`} style={{ overflow: "auto", padding: "var(--ma-3)", display: "flex", flexDirection: "column", gap: "var(--ma-3)", minHeight: 0, minWidth: 0 }}>
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
  // Reject fades the card out; accept just settles (no glow/bloom — flat VS Code surface).
  const settleStyle =
    settle === "reject" ? { animation: "hunk-dissolve 480ms var(--ease) forwards" } : {};
  const selectedPending = selected && !decided && !settle;

  return (
    <div
      ref={innerRef}
      onClick={onSelect}
      role="listitem"
      aria-current={selectedPending ? "true" : undefined}
      aria-label={`Hunk ${hunk.header}, ${st.label}`}
      style={{
        padding: 0,
        overflow: "hidden",
        borderRadius: "var(--radius-sm)",
        background: "var(--surface-2)",
        border: selectedPending ? "1px solid var(--accent)" : "1px solid var(--border)",
        opacity: decided && settle !== "accept" ? 0.55 : 1,
        ...settleStyle,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-2)",
          padding: "var(--ma-2) var(--ma-3)",
          boxShadow: "inset 0 -1px 0 0 var(--border)",
          fontSize: "12px",
          color: "var(--text-muted)",
          minWidth: 0,
        }}
      >
        <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{hunk.header}</span>
        <span style={{ marginLeft: "auto", color: st.color, flex: "0 0 auto" }}>{st.label}</span>
      </div>

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "12px", lineHeight: 1.6, overflowX: "auto", minWidth: 0 }}>
        {hunk.lines.map((ln, j) => {
          const ks = KIND_STYLE[ln.kind];
          const ctx = ln.kind === "ctx";
          return (
            <div key={j} className={"vsc-diffline " + ks.cls} style={{ display: "flex" }}>
              <span style={gutterNo}>{ln.oldNo ?? ""}</span>
              <span style={gutterNo}>{ln.newNo ?? ""}</span>
              <span style={{ width: 16, flexShrink: 0, textAlign: "center", userSelect: "none", color: ctx ? "var(--text-dim)" : undefined }}>{ks.marker}</span>
              <span style={{ color: ctx ? "var(--text)" : undefined, whiteSpace: "pre", paddingRight: "var(--ma-4)" }}>
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
  const isAccept = tone === "accept";
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "4px var(--ma-3)",
        borderRadius: "var(--radius-sm)",
        fontSize: "12px",
        color: isAccept ? "var(--accent-text)" : "var(--text)",
        background: isAccept ? "var(--accent)" : "var(--input-bg)",
        border: isAccept ? "none" : "1px solid var(--border-strong)",
      }}
    >
      {label}
      <kbd style={kbd}>{hint}</kbd>
    </button>
  );
}

/*
  The one-shot keyframe a rejected hunk uses to fade out. Kept surface-local; honors
  prefers-reduced-motion via theme.css's global reduce rule.
*/
function LocalSettleKeyframes() {
  return (
    <style>{`
      @keyframes hunk-dissolve {
        0%   { opacity: 1; transform: translateY(0); }
        100% { opacity: 0.4; transform: translateY(-2px); }
      }
    `}</style>
  );
}

const gutterNo: CSSProperties = {
  width: 34,
  flexShrink: 0,
  textAlign: "right",
  paddingRight: 8,
  color: "var(--text-dim)",
  userSelect: "none",
  fontSize: "11px",
};

const kbd: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  padding: "1px 5px",
  borderRadius: 3,
  color: "var(--text-muted)",
  border: "1px solid var(--border-strong)",
  background: "var(--surface-2)",
};
