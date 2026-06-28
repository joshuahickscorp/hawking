/*
  hunkreview.tsx: the hunk-by-hunk diff-review gesture, re-housed for the Workstation
  merge-review queue. THE CONTRACT IS THE IDE's (HIDE_PLAN D §580 + §594): move with
  j/k, accept/reject with a/r (and Cmd+Enter / Cmd+Backspace under focus), so the gesture
  is indistinguishable across the IDE and the merge surface. Accept -> AcceptDiff{run_id,diff_id},
  reject -> RejectDiff, undo a settled hunk -> Custom:revert_diff. The host applies; we only
  send the verdict and render the settling. No spinner: a hunk absorbs (green/red dissolves to
  steady) or dissolves back out, the OP-1 key-resolving feel (C §432).

  This is a LOCAL component (parallel-safety: the IDE stub has no shared hunk primitive yet, so
  the prompt says replicate the identical gesture rather than import across surfaces).
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { sendIntent } from "../../ipc";
import { intent } from "../../wire";
import { RadiationEdge } from "../../ui";

export type HunkVerdict = "pending" | "accepted" | "rejected";

export interface DiffLine {
  kind: "ctx" | "add" | "del";
  text: string;
}

export interface Hunk {
  id: string;
  header: string; // e.g. "@@ guard.rs 42,7 +42,9 @@"
  lines: DiffLine[];
}

export interface ReviewBranch {
  run_id: string;
  diff_id: string;
  label: string; // the branch / approach name
  path: string;
  hunks: Hunk[];
}

// The j/k/a/r keymap, shared verbatim with the IDE diff focus (D §580).
const KEY_HELP = "j / k move   a accept   r reject   u undo";

export function HunkReview({ branch }: { branch: ReviewBranch }) {
  // verdict per hunk id; the user's accept/reject stream IS the personalization corpus (A6.8).
  const [verdicts, setVerdicts] = useState<Record<string, HunkVerdict>>({});
  const [cursor, setCursor] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

  // a fresh branch resets the review (new diff_id, new gesture session).
  useEffect(() => {
    setVerdicts({});
    setCursor(0);
    requestAnimationFrame(() => rootRef.current?.focus());
  }, [branch.diff_id]);

  const verdictOf = (id: string): HunkVerdict => verdicts[id] ?? "pending";

  const accept = (h: Hunk) => {
    setVerdicts((v) => ({ ...v, [h.id]: "accepted" }));
    void sendIntent(intent.acceptDiff(branch.run_id, branch.diff_id));
    advance();
  };
  const reject = (h: Hunk) => {
    setVerdicts((v) => ({ ...v, [h.id]: "rejected" }));
    void sendIntent(intent.rejectDiff(branch.run_id, branch.diff_id));
    advance();
  };
  const undo = (h: Hunk) => {
    setVerdicts((v) => ({ ...v, [h.id]: "pending" }));
    // a settled hunk reverts via a compensating upstream event (D §594).
    void sendIntent(intent.custom("revert_diff", { run_id: branch.run_id, diff_id: branch.diff_id, hunk: h.id }));
  };

  const advance = () => setCursor((c) => Math.min(c + 1, branch.hunks.length - 1));

  // scroll the focused hunk into view as the cursor walks.
  useEffect(() => {
    rowRefs.current[cursor]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [cursor]);

  const onKey = (e: React.KeyboardEvent) => {
    const h = branch.hunks[cursor];
    if (!h) return;
    // Cmd+Enter / Cmd+Backspace mirror the IDE accept/reject hunk binding (D §580).
    if (e.metaKey && e.key === "Enter") return void (e.preventDefault(), accept(h));
    if (e.metaKey && (e.key === "Backspace" || e.key === "Delete")) return void (e.preventDefault(), reject(h));
    if (e.metaKey) return;
    switch (e.key) {
      case "j":
      case "ArrowDown":
        e.preventDefault();
        setCursor((c) => Math.min(c + 1, branch.hunks.length - 1));
        break;
      case "k":
      case "ArrowUp":
        e.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
        break;
      case "a":
        e.preventDefault();
        accept(h);
        break;
      case "r":
        e.preventDefault();
        reject(h);
        break;
      case "u":
        e.preventDefault();
        undo(h);
        break;
    }
  };

  const settled = useMemo(
    () => branch.hunks.filter((h) => verdictOf(h.id) !== "pending").length,
    [branch.hunks, verdicts],
  );
  const accepted = branch.hunks.filter((h) => verdictOf(h.id) === "accepted").length;
  const allSettled = settled === branch.hunks.length;

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onKeyDown={onKey}
      style={{ outline: "none", display: "flex", flexDirection: "column", gap: "var(--s3)", minHeight: 0 }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: "var(--s3)" }}>
        <span style={{ color: "var(--text-hi)" }}>{branch.label}</span>
        <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{branch.path}</span>
        <span style={{ marginLeft: "auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
          {allSettled ? `${accepted} kept` : `${settled}/${branch.hunks.length} reviewed`}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--s3)", overflowY: "auto", minHeight: 0 }}>
        {branch.hunks.map((h, i) => (
          <div key={h.id} ref={(el) => { rowRefs.current[i] = el; }}>
            <HunkBlock
              hunk={h}
              focused={i === cursor}
              verdict={verdictOf(h.id)}
              onFocus={() => setCursor(i)}
              onAccept={() => accept(h)}
              onReject={() => reject(h)}
              onUndo={() => undo(h)}
            />
          </div>
        ))}
      </div>

      <div style={{ color: "var(--text-low)", fontSize: "var(--text-xs)", letterSpacing: "0.04em" }}>{KEY_HELP}</div>
    </div>
  );
}

const VERDICT_RING: Record<HunkVerdict, string> = {
  pending: "var(--rim)",
  accepted: "var(--success)",
  rejected: "var(--rim-strong)",
};

function HunkBlock({
  hunk,
  focused,
  verdict,
  onFocus,
  onAccept,
  onReject,
  onUndo,
}: {
  hunk: Hunk;
  focused: boolean;
  verdict: HunkVerdict;
  onFocus: () => void;
  onAccept: () => void;
  onReject: () => void;
  onUndo: () => void;
}) {
  // a rejected hunk dissolves (drops away, dimmed); an accepted hunk absorbs to steady code.
  const absorbed = verdict === "accepted";
  const dissolved = verdict === "rejected";
  const body: CSSProperties = {
    borderRadius: "var(--radius)",
    background: "var(--surface-0)",
    boxShadow: `inset 0 0 0 1px ${VERDICT_RING[verdict]}`,
    opacity: dissolved ? 0.4 : 1,
    transition: "opacity 220ms ease, box-shadow 220ms ease, transform 220ms ease",
    transform: dissolved ? "translateY(-2px)" : "none",
    overflow: "hidden",
  };
  const inner = (
    <div style={body}>
      <div
        onMouseEnter={onFocus}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s2)",
          padding: "var(--s2) var(--s3)",
          color: "var(--text-low)",
          fontSize: "var(--text-xs)",
          boxShadow: "inset 0 -1px 0 0 var(--rim)",
        }}
      >
        <span>{hunk.header}</span>
        <span style={{ marginLeft: "auto", display: "flex", gap: "var(--s2)" }}>
          {verdict === "pending" ? (
            <>
              <GestureBtn label="a accept" tone="add" onClick={onAccept} />
              <GestureBtn label="r reject" tone="del" onClick={onReject} />
            </>
          ) : (
            <>
              <span style={{ color: absorbed ? "var(--diff-add-fg)" : "var(--text-low)" }}>
                {absorbed ? "kept" : "rejected"}
              </span>
              <GestureBtn label="u undo" tone="ctx" onClick={onUndo} />
            </>
          )}
        </span>
      </div>
      <pre style={{ margin: 0, padding: "var(--s2) 0", fontSize: "var(--text-sm)", lineHeight: 1.5 }}>
        {hunk.lines.map((l, i) => (
          <DiffRow key={i} line={l} muted={absorbed && l.kind !== "ctx"} />
        ))}
      </pre>
    </div>
  );
  // the focused, still-pending hunk wears the breathing gold edge (this is the live cursor).
  return focused && verdict === "pending" ? <RadiationEdge mode="breathe">{inner}</RadiationEdge> : inner;
}

function DiffRow({ line, muted }: { line: DiffLine; muted: boolean }) {
  const sign = line.kind === "add" ? "+" : line.kind === "del" ? "-" : " ";
  // after accept the +/- settles into normal code (muted to context tone): the absorption (C §432).
  const fg =
    muted
      ? "var(--text-mid)"
      : line.kind === "add"
        ? "var(--diff-add-fg)"
        : line.kind === "del"
          ? "var(--diff-del-fg)"
          : "var(--text-mid)";
  const bg =
    muted
      ? "transparent"
      : line.kind === "add"
        ? "var(--diff-add-bg)"
        : line.kind === "del"
          ? "var(--diff-del-bg)"
          : "transparent";
  return (
    <div style={{ display: "flex", gap: "var(--s2)", padding: "0 var(--s3)", background: bg, color: fg }}>
      <span style={{ color: "var(--text-low)", width: 10, flex: "0 0 auto", userSelect: "none" }}>{sign}</span>
      <span style={{ whiteSpace: "pre-wrap" }}>{line.text}</span>
    </div>
  );
}

function GestureBtn({ label, tone, onClick }: { label: string; tone: DiffLine["kind"]; onClick: () => void }) {
  const color = tone === "add" ? "var(--diff-add-fg)" : tone === "del" ? "var(--diff-del-fg)" : "var(--text-mid)";
  return (
    <button
      onClick={onClick}
      style={{
        padding: "1px 8px",
        borderRadius: "var(--radius)",
        fontSize: "var(--text-xs)",
        color,
        boxShadow: "inset 0 0 0 1px var(--rim)",
        background: "var(--surface-1)",
      }}
    >
      {label}
    </button>
  );
}
