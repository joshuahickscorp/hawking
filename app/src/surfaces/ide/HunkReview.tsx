/*
  HunkReview.tsx: THE core gesture, factored as a reusable local component (C9/C10/D1.2).
  Per-hunk review of an agent's proposed diff, driven by keyboard:
    j / ArrowDown  -> next hunk        k / ArrowUp  -> prev hunk
    a / Cmd+Enter  -> accept hunk      r / Cmd+Backspace -> reject hunk
  Accept dispatches AcceptDiff{run_id,diff_id}; reject dispatches RejectDiff. The accepted hunk
  settles with a brief gold radiation absorption (C11: "the change is taken in", a clean physical
  action), the rejected one dissolves back out. Diff add/del use the --diff-* tokens AND a +/-
  marker, never color alone (C3 / C14 accessibility).

  This same gesture must be identical in the Workstation merge-review (C9: "the review gesture must
  be identical everywhere or the seams show"), so this component is the single source of it: it takes
  a DiffDoc + an onAct callback and owns nothing surface-specific. Re-housed from the Void/Cline
  hunk-accept UX, re-skinned to near-black material + gold rim + Geist Mono.
*/
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import type { DiffDoc, Hunk } from "./types";

export type HunkAction = "accept" | "reject";

const KIND_STYLE = {
  add: { fg: "var(--diff-add-fg)", bg: "var(--diff-add-bg)", marker: "+" },
  del: { fg: "var(--diff-del-fg)", bg: "var(--diff-del-bg)", marker: "-" },
  ctx: { fg: "var(--text-mid)", bg: "transparent", marker: " " },
} as const;

const STATUS_LABEL: Record<Hunk["status"], { label: string; color: string }> = {
  pending: { label: "pending", color: "var(--text-low)" },
  accepted: { label: "accepted", color: "var(--success)" },
  rejected: { label: "rejected", color: "var(--danger)" },
  applied: { label: "applied", color: "var(--success)" },
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
  // hunks that just settled, for the one-shot absorption animation, keyed by id -> action.
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
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s3)",
          padding: "var(--s2) var(--s3)",
          borderBottom: "1px solid var(--rim)",
          fontSize: "var(--text-xs)",
          color: "var(--text-low)",
        }}
      >
        <span style={{ color: "var(--text-mid)" }}>{doc.path}</span>
        {doc.stale ? (
          <span
            title="The file changed under this pending diff. Re-sync before applying."
            style={{
              color: "var(--warning)",
              boxShadow: "inset 0 0 0 1px var(--warning)",
              borderRadius: "var(--radius)",
              padding: "0 6px",
            }}
          >
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

      <div style={{ overflowY: "auto", padding: "var(--s3)", display: "flex", flexDirection: "column", gap: "var(--s3)", minHeight: 0 }}>
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
          <div style={{ color: "var(--text-low)", padding: "var(--s4)" }}>No hunks in this change.</div>
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
  // The settle absorption: accept glows gold then quiets; reject fades the card out.
  const settleStyle =
    settle === "accept"
      ? { animation: "hunk-absorb 620ms ease-out" }
      : settle === "reject"
        ? { animation: "hunk-dissolve 480ms ease-in forwards" }
        : {};

  return (
    <div
      ref={innerRef}
      onClick={onSelect}
      className="panel"
      style={{
        padding: 0,
        overflow: "hidden",
        opacity: decided && settle !== "accept" ? 0.55 : 1,
        boxShadow: selected
          ? "inset 0 0 0 1px var(--radiation), 0 0 14px -4px var(--radiation-bloom)"
          : "var(--panel-inset)",
        ...settleStyle,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s2)",
          padding: "4px var(--s3)",
          borderBottom: "1px solid var(--rim)",
          fontSize: "var(--text-xs)",
          color: "var(--text-low)",
        }}
      >
        <span style={{ color: "var(--text-mid)" }}>{hunk.header}</span>
        <span style={{ marginLeft: "auto", color: st.color }}>{st.label}</span>
      </div>

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-sm)", lineHeight: 1.6 }}>
        {hunk.lines.map((ln, j) => {
          const ks = KIND_STYLE[ln.kind];
          return (
            <div key={j} style={{ display: "flex", background: ks.bg }}>
              <span style={gutterNo}>{ln.oldNo ?? ""}</span>
              <span style={gutterNo}>{ln.newNo ?? ""}</span>
              <span style={{ width: 14, textAlign: "center", color: ks.fg, userSelect: "none" }}>{ks.marker}</span>
              <span style={{ color: ln.kind === "ctx" ? "var(--text-mid)" : ks.fg, whiteSpace: "pre", paddingRight: "var(--s3)" }}>
                {ln.text || " "}
              </span>
            </div>
          );
        })}
      </div>

      {!decided ? (
        <div style={{ display: "flex", gap: "var(--s2)", padding: "var(--s2) var(--s3)", borderTop: "1px solid var(--rim)" }}>
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
  const accent = tone === "accept" ? "var(--success)" : "var(--danger)";
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--s2)",
        padding: "3px 12px",
        borderRadius: "var(--radius)",
        fontSize: "var(--text-xs)",
        color: "var(--text-mid)",
        background: "var(--surface-1)",
        boxShadow: `inset 0 0 0 1px ${accent}55`,
      }}
    >
      <span style={{ color: accent }}>{tone === "accept" ? "+" : "-"}</span>
      {label}
      <kbd style={kbd}>{hint}</kbd>
    </button>
  );
}

const gutterNo: CSSProperties = {
  width: 34,
  flexShrink: 0,
  textAlign: "right",
  paddingRight: 6,
  color: "var(--text-low)",
  userSelect: "none",
  fontSize: "var(--text-xs)",
};

const kbd: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  padding: "1px 5px",
  borderRadius: 3,
  color: "var(--text-mid)",
  boxShadow: "inset 0 0 0 1px var(--rim-strong)",
  background: "var(--surface-1)",
};
