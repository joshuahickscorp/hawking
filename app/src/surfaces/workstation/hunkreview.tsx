/*
  hunkreview.tsx: the hunk-by-hunk diff-review gesture, re-housed for the Workstation
  merge-review queue. THE CONTRACT IS THE IDE's: move with j/k, accept/reject with a/r
  (and Cmd+Enter / Cmd+Backspace under focus), so the gesture is indistinguishable across
  the IDE and the merge surface. Accept -> AcceptDiff{run_id,diff_id}, reject -> RejectDiff,
  undo a settled hunk -> Custom:revert_diff. The host applies; we only send the verdict and
  render the settling. No spinner: a hunk absorbs (the +/- quiets to steady code) or dissolves
  back out, the OP-1 key-resolving feel.

  Doctrine v3 (Tadao Ando grayscale concrete): the diff is poured concrete; add/del are the
  ONLY two pigments (--ok lichen, --bad oxide via the .hunk-add / .hunk-del classes, always
  glyph-paired with +/-). The focused, still-pending hunk is the place light enters: it breathes
  (LightEdge mode="breathe"). No gold, no third color, generous --ma-* air.

  This is a LOCAL component (the IDE keeps its own copy; the prompt says replicate the
  identical gesture rather than import across surfaces, so no seam shows).
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { sendIntent } from "../../ipc";
import { intent } from "../../wire";
import { LightEdge } from "../../ui";

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

// The j/k/a/r keymap, shared verbatim with the IDE diff focus.
const KEY_HELP = "j / k move   a accept   r reject   u undo";

export function HunkReview({ branch }: { branch: ReviewBranch }) {
  // verdict per hunk id; the user's accept/reject stream IS the personalization corpus.
  const [verdicts, setVerdicts] = useState<Record<string, HunkVerdict>>({});
  const [cursor, setCursor] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

  // a fresh branch resets the review (new diff_id, new gesture session).
  useEffect(() => {
    setVerdicts({});
    setCursor(0);
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
    // a settled hunk reverts via a compensating upstream event.
    void sendIntent(intent.custom("revert_diff", { run_id: branch.run_id, diff_id: branch.diff_id, hunk: h.id }));
  };

  const advance = () => setCursor((c) => Math.min(c + 1, branch.hunks.length - 1));

  const onKey = (e: React.KeyboardEvent) => {
    const h = branch.hunks[cursor];
    if (!h) return;
    // Cmd+Enter / Cmd+Backspace mirror the IDE accept/reject hunk binding.
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
      style={{ outline: "none", display: "flex", flexDirection: "column", gap: "var(--ma-6)", minHeight: 0 }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: "var(--ma-4)", flexWrap: "wrap", minWidth: 0 }}>
        <span className="t-title" style={{ color: "var(--text-1)" }}>{branch.label}</span>
        <span className="t-code" style={{ color: "var(--text-3)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{branch.path}</span>
        <span className="t-micro" style={{ marginLeft: "auto", flex: "0 0 auto" }}>
          {allSettled ? `${accepted} kept` : `${settled}/${branch.hunks.length} reviewed`}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-4)", overflowY: "auto", minHeight: 0 }}>
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

      <div className="t-micro" style={{ color: "var(--mute)" }}>{KEY_HELP}</div>
    </div>
  );
}

// A settled hunk wears a quiet, steady shadow-line; pending wears the plain hairline.
const VERDICT_SHADOW: Record<HunkVerdict, string> = {
  pending: "var(--hairline)",
  accepted: "var(--hairline-strong), var(--inner-glow)",
  rejected: "var(--hairline)",
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
    background: "var(--concrete-1)",
    boxShadow: VERDICT_SHADOW[verdict],
    opacity: dissolved ? 0.4 : 1,
    transition: "opacity var(--dur) var(--ease), box-shadow var(--dur) var(--ease), transform var(--dur) var(--ease)",
    transform: dissolved ? "translateY(-2px)" : "none",
    overflow: "hidden",
  };
  const inner = (
    <div style={body}>
      <div
        onMouseEnter={onFocus}
        className="t-micro"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-2)",
          flexWrap: "wrap",
          padding: "var(--ma-2) var(--ma-3)",
          color: "var(--text-3)",
          boxShadow: "inset 0 -1px 0 0 var(--line)",
        }}
      >
        <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{hunk.header}</span>
        <span style={{ marginLeft: "auto", display: "flex", gap: "var(--ma-2)", flex: "0 0 auto" }}>
          {verdict === "pending" ? (
            <>
              <GestureBtn label="a accept" tone="add" onClick={onAccept} />
              <GestureBtn label="r reject" tone="del" onClick={onReject} />
            </>
          ) : (
            <>
              <span style={{ color: absorbed ? "var(--ok)" : "var(--text-3)" }}>
                {absorbed ? "kept" : "rejected"}
              </span>
              <GestureBtn label="u undo" tone="ctx" onClick={onUndo} />
            </>
          )}
        </span>
      </div>
      <pre className="t-code" style={{ margin: 0, padding: "var(--ma-2) 0" }}>
        {hunk.lines.map((l, i) => (
          <DiffRow key={i} line={l} muted={absorbed && l.kind !== "ctx"} />
        ))}
      </pre>
    </div>
  );
  // the focused, still-pending hunk is where light enters: it breathes (the live cursor).
  return focused && verdict === "pending" ? <LightEdge mode="breathe">{inner}</LightEdge> : inner;
}

function DiffRow({ line, muted }: { line: DiffLine; muted: boolean }) {
  const sign = line.kind === "add" ? "+" : line.kind === "del" ? "-" : " ";
  // after accept the +/- settles into normal code (quieted to context tone): the absorption.
  // add/del use the .hunk-add / .hunk-del classes (--ok / --bad + bg), the only two pigments.
  const cls = muted ? "" : line.kind === "add" ? "hunk-add" : line.kind === "del" ? "hunk-del" : "";
  const fg = muted || line.kind === "ctx" ? "var(--text-2)" : undefined;
  return (
    <div
      className={cls}
      style={{ display: "flex", gap: "var(--ma-2)", padding: "0 var(--ma-3)", ...(fg ? { color: fg } : null) }}
    >
      <span style={{ color: "var(--text-3)", width: 10, flex: "0 0 auto", userSelect: "none" }}>{sign}</span>
      <span style={{ minWidth: 0, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{line.text}</span>
    </div>
  );
}

function GestureBtn({ label, tone, onClick }: { label: string; tone: DiffLine["kind"]; onClick: () => void }) {
  const color = tone === "add" ? "var(--ok)" : tone === "del" ? "var(--bad)" : "var(--text-2)";
  return (
    <button
      onClick={onClick}
      className="t-micro"
      style={{
        padding: "1px 8px",
        borderRadius: "var(--radius)",
        color,
        boxShadow: "var(--hairline)",
        background: "var(--concrete-2)",
      }}
    >
      {label}
    </button>
  );
}
