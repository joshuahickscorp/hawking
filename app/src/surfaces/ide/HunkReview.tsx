/*
  HunkReview.tsx: the per-hunk diff review gesture AND the review action spine for the whole diff.

  Every gesture here resolves to a catalog command id (src/generated/command_catalog.json) through
  runCommand, so the hunk buttons, the keyboard, the per-hunk menu and the palette all mean the same
  thing. Nothing invents a verb.

  What the host really does (crates/hide-backend/src/host.rs, sec 23):
    accept_diff{hunk_id}  -> apply_hunk       accept_diff{no hunk_id}  -> apply_diff (whole)
    reject_diff{hunk_id}  -> reject_hunk      custom revert_diff       -> revert_diff (whole)
  The whole-diff revert is ONE command with one approval policy. reject_diff without a hunk_id is
  the same host effect, so the host resolves it to revert_diff and holds it at the same gate; this
  surface never sends that shape.
  reject_hunk reverts exactly that file through the verifying applier and appends
  `verify.invalidated` for every verification receipt whose scope covers the file, which is why a
  rejected hunk offers a re-verify.

  Keyboard: j/k or arrows move, a / Mod+Enter accept, r / Mod+Backspace reject, m opens the hunk
  detail (provenance, base hash, the rest of the actions), d opens the review detail (counts and the
  receipt export). These are BARE letters on a window listener, so they act only while the diff
  surface itself holds focus (see reviewKeysActive): exempting text fields alone meant an "r" typed
  with focus on any button anywhere in the app rejected the selected hunk and reverted a file.
  Scoping is also what keeps Mod+Enter unambiguous: the catalog gives that chord to submit_turn on
  the chat surface and to accept_diff on the diff surface, and only one of the two can hold focus.
  Diff add/del carry a +/- marker as well as the pigment, so color is never the sole
  signal, and every hunk shows pending / working / accepted / rejected / failed in words.

  Deliberately ABSENT, with the reason (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md):
    (no longer absent) run the affected checks -> `run_static_analysis` is Custom-bound with a real
                                             host arm that publishes diagnostics, so the rejected
                                             hunk's re-verify RUNS the checks on the reverted file
                                             instead of asking the agent to do it.
    run declared acceptance               -> `goal_evaluate` is reachable (Custom-bound,
                                             host-handled) but it grades the SESSION goal, so it is
                                             bound once on the goal chip in the courtyard composer;
                                             a second copy here would be two controls for one
                                             capability.
    rewind / fork an alternative /
    compare candidates                    -> `checkpoint_rewind|fork|compare` all need a
                                             checkpoint_id and no UiEvent carries one. The whole-diff
                                             undo that IS reachable is `revert_diff`, which the diff
                                             bar already offers once the hunks are decided.
    edit_hunk custom name                 -> RETIRED. It routed to the SAME host apply_hunk as
                                             accept_diff{hunk_id} and carried no edited text, so it
                                             was one action under two ids.
*/
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import { runCommand, useStore } from "../../store";
import { ackState, heldNote, type IntentAck } from "../../wire";
import type { DiffDoc, Hunk, HunkStatus } from "./types";

export type HunkAction = "accept" | "reject";

/** The surface a review key belongs to: the whole diff review region when this panel is mounted
 *  inside one (Editor.tsx renders `.diffreview` around the Monaco pane and this panel), otherwise
 *  the panel itself (the courtyard's Diff side panel). */
export function reviewSurface(root: Element | null): Element | null {
  return root?.closest(".diffreview") ?? root;
}

/** Whether a review key may act. Bare letters (a, r, j, k, m, d) accept and REJECT hunks, so they
 *  require focus inside the diff surface; a text field inside it still keeps its own keystrokes. */
export function reviewKeysActive(surfaceHasFocus: boolean, tagName?: string): boolean {
  return surfaceHasFocus && tagName !== "INPUT" && tagName !== "TEXTAREA";
}

/* ---- The host's hunk record, read defensively -------------------------------------------------
   The host's DiffHunk carries hunk_id / file / base_hash / provenance beside the fields the local
   view model names (ide/types.ts is owned by another stage, so the extra fields are read here rather
   than re-declared there). A projection that has not been enriched yet simply reads as absent. */

export interface HunkProvenance {
  plan_step?: string | null;
  agent?: string;
  turn?: number;
}

type RichHunk = Hunk & {
  hunk_id?: string;
  file?: string;
  base_hash?: string;
  provenance?: HunkProvenance;
};

/** The id the WIRE addresses. Falls back to the local id so a mock diff still targets one hunk. */
export const hunkId = (h: Hunk): string => (h as RichHunk).hunk_id ?? h.id;
export const hunkFile = (h: Hunk, fallback: string): string => (h as RichHunk).file ?? fallback;
export const hunkBaseHash = (h: Hunk): string | null => (h as RichHunk).base_hash ?? null;
export const hunkProvenance = (h: Hunk): HunkProvenance | null => (h as RichHunk).provenance ?? null;

/** The +/- body of one hunk, the text an explain / alternative request quotes. */
export const hunkPatch = (h: Hunk): string =>
  h.lines.map((l) => (l.kind === "add" ? "+" : l.kind === "del" ? "-" : " ") + l.text).join("\n");

/** How a hunk is cited to the agent: file, header and the addressable hunk id. */
export const citeHunk = (h: Hunk, path: string): string =>
  `${hunkFile(h, path)} ${h.header} (hunk ${hunkId(h)})`;

/** What rejecting this hunk costs, stated the way the host implements it. */
export const invalidatedNote = (file: string): string =>
  `Reverting ${file} invalidates every verification receipt whose scope covers it.`;

/* ---- The action tables ----------------------------------------------------------------------- */

export type HunkActionId =
  | "accept"
  | "reject"
  | "revert"
  | "reapply"
  | "explain"
  | "reverify"
  | "side_chat"
  | "alternative"
  | "concern";

export interface HunkActionSpec {
  id: HunkActionId;
  label: string;
  /** Catalog command id, or null when the route is a custom intent (documented in runHunkAction). */
  command: string | null;
  shortcut?: string;
  /** True when the action carries the user's typed note, so it is offered with an input. */
  needsNote?: boolean;
  /** True when the action changes what is on disk, so it reads as explicit. */
  destructive?: boolean;
}

export const HUNK_ACTIONS: HunkActionSpec[] = [
  { id: "accept", label: "Accept this hunk", command: "accept_diff", shortcut: "a" },
  { id: "reject", label: "Reject this hunk, reverting the file", command: "reject_diff", shortcut: "r", destructive: true },
  { id: "revert", label: "Revert this hunk on disk", command: "reject_diff", destructive: true },
  { id: "reapply", label: "Re-apply this hunk", command: "accept_diff" },
  { id: "explain", label: "Explain this hunk", command: "submit_turn" },
  { id: "reverify", label: "Run the checks on this file", command: "run_static_analysis" },
  { id: "side_chat", label: "Review side chat about this hunk", command: "create_side_chat" },
  { id: "alternative", label: "Request an alternative", command: null },
  { id: "concern", label: "Attach a concern", command: null, needsNote: true },
];

export const hunkActionSpec = (id: HunkActionId): HunkActionSpec =>
  HUNK_ACTIONS.find((a) => a.id === id) as HunkActionSpec;

/** Which actions a hunk offers right now. Offered-but-dead is worse than absent: a pending hunk
 *  cannot be reverted, an accepted one cannot be accepted again, and the re-verify offer belongs to
 *  the rejected hunk whose verification the host has just invalidated. */
export function hunkActionsFor(status: HunkStatus): HunkActionId[] {
  const tail: HunkActionId[] = ["explain", "side_chat", "alternative", "concern"];
  if (status === "pending") return ["accept", "reject", ...tail];
  if (status === "rejected") return ["reapply", "reverify", ...tail];
  return ["revert", ...tail];
}

export interface HunkCtx {
  diffId: string;
  runId: string;
  sessionId: string;
  /** The diff's path, used when a hunk carries no file of its own. */
  path: string;
  hunk: Hunk;
  /** The typed note, for `concern`. */
  note?: string;
}

/** Put text in front of the model now: a live run is steered over the same real InterruptHub route
 *  the composer uses, and with no run in flight it opens the turn. */
function steerOrSend(ctx: HunkCtx, text: string): Promise<IntentAck> {
  if (ctx.runId)
    return runCommand("steer", { run_id: ctx.runId, session_id: ctx.sessionId, text });
  return runCommand("submit_turn", { session_id: ctx.sessionId, text });
}

/**
 * THE dispatch point for every per-hunk gesture. Throws (never silently no-ops) so the caller shows
 * the refusal. accept/reapply and reject/revert differ only in the label the current status earns;
 * they are never offered together, so one host route never wears two buttons.
 */
export async function runHunkAction(id: HunkActionId, ctx: HunkCtx): Promise<IntentAck> {
  const hid = hunkId(ctx.hunk);
  const file = hunkFile(ctx.hunk, ctx.path);
  const cite = citeHunk(ctx.hunk, ctx.path);
  switch (id) {
    case "accept":
    case "reapply":
      return runCommand("accept_diff", { run_id: ctx.runId, diff_id: ctx.diffId, hunk_id: hid });

    case "reject":
    case "revert":
      return runCommand("reject_diff", { run_id: ctx.runId, diff_id: ctx.diffId, hunk_id: hid });

    case "explain":
      return runCommand("submit_turn", {
        session_id: ctx.sessionId,
        text: `Explain this change and why it was made:\n\n${cite}\n\n${hunkPatch(ctx.hunk)}`,
      });

    case "reverify":
      // The host's own model-free checker over the reverted file, whose verification receipts the
      // reject just invalidated. A real run, not a request that the agent do one.
      return runCommand("run_static_analysis", { session_id: ctx.sessionId, paths: [file] });

    case "side_chat":
      return runCommand("create_side_chat", { session_id: ctx.sessionId, inherit: true });

    case "alternative":
      return steerOrSend(ctx, `Propose a different change for ${cite}. This hunk is not what I want:\n\n${hunkPatch(ctx.hunk)}`);

    case "concern": {
      const note = ctx.note?.trim();
      if (!note) throw new Error("Attach a concern needs a note");
      return steerOrSend(ctx, `Concern on ${cite}: ${note}`);
    }
  }
}

/** The status an optimistic local flip should show while the host echoes its own patch. */
export function statusAfter(id: HunkActionId, current: HunkStatus): HunkStatus {
  if (id === "accept" || id === "reapply") return "accepted";
  if (id === "reject" || id === "revert") return "rejected";
  return current;
}

export type DiffActionId = "accept_all" | "revert_all";

export interface DiffActionSpec {
  id: DiffActionId;
  label: string;
  command: string | null;
  destructive?: boolean;
}

export const DIFF_ACTIONS: DiffActionSpec[] = [
  // hunk_id omitted on purpose: the host reads that as "the whole diff" and walks its own record, so
  // every hunk it applies keeps the provenance the record holds. There is no provenance-free accept.
  { id: "accept_all", label: "Accept every pending hunk", command: "accept_diff" },
  // ONE whole-diff undo, not two. "Reject all" and "revert all" were two buttons in the same bar
  // for the SAME host revert_diff, one of them gated by the approval policy and one of them not, so
  // the gate could be walked around by pressing the other button. Both phases use this row now.
  { id: "revert_all", label: "Revert the whole diff on disk", command: "revert_diff", destructive: true },
];

export const diffActionSpec = (id: DiffActionId): DiffActionSpec =>
  DIFF_ACTIONS.find((a) => a.id === id) as DiffActionSpec;

export async function runDiffAction(id: DiffActionId, ctx: { diffId: string; runId: string }): Promise<IntentAck> {
  if (id === "accept_all") return runCommand("accept_diff", { run_id: ctx.runId, diff_id: ctx.diffId });
  return runCommand("revert_diff", { diff_id: ctx.diffId, run_id: ctx.runId });
}

/* ---- The review receipt ------------------------------------------------------------------------
   A client-side export of the review as it stands: every hunk with its status, base hash and
   provenance. The host seals the authoritative receipt (DiffReviewReceipt, blake3 sealed); this is
   the copy the reviewer can paste, and it says so rather than posing as the sealed one. */

export interface ReviewReceiptHunk {
  hunk_id: string;
  file: string;
  header: string;
  status: HunkStatus;
  base_hash: string | null;
  provenance: HunkProvenance | null;
}

export interface ReviewReceipt {
  diff_id: string;
  run_id: string;
  path: string;
  counts: Record<string, number>;
  hunks: ReviewReceiptHunk[];
  note: string;
}

export function reviewReceipt(doc: DiffDoc): ReviewReceipt {
  const counts: Record<string, number> = {};
  for (const h of doc.hunks) counts[h.status] = (counts[h.status] ?? 0) + 1;
  return {
    diff_id: doc.diff_id,
    run_id: doc.run_id,
    path: doc.path,
    counts,
    hunks: doc.hunks.map((h) => ({
      hunk_id: hunkId(h),
      file: hunkFile(h, doc.path),
      header: h.header,
      status: h.status,
      base_hash: hunkBaseHash(h),
      provenance: hunkProvenance(h),
    })),
    note: "Client-side export of this review. The host seals the authoritative receipt.",
  };
}

/** Export the receipt. The writer is injected so the export is testable and so a browser without
 *  clipboard access fails loudly instead of pretending it copied. */
export async function exportReviewReceipt(
  doc: DiffDoc,
  write: (text: string) => Promise<void> = (t) => navigator.clipboard.writeText(t),
): Promise<ReviewReceipt> {
  const receipt = reviewReceipt(doc);
  await write(JSON.stringify(receipt, null, 2));
  return receipt;
}

/* ---- The surface ------------------------------------------------------------------------------ */

// add/del carry the diff pigment via class (.hunk-add/.hunk-del = --ok/--bad + bg); ctx is neutral.
// Markers are always present so the meaning never rests on color alone.
const KIND_STYLE = {
  add: { cls: "hunk-add", marker: "+" },
  del: { cls: "hunk-del", marker: "-" },
  ctx: { cls: "", marker: " " },
} as const;

const STATUS_LABEL: Record<HunkStatus, { label: string; color: string }> = {
  pending: { label: "pending", color: "var(--text-dim)" },
  accepted: { label: "accepted", color: "var(--git-add)" },
  rejected: { label: "rejected", color: "var(--git-del)" },
  applied: { label: "applied", color: "var(--git-add)" },
};

export function HunkReview({
  doc,
  onStatus,
  active = true,
}: {
  doc: DiffDoc;
  // The optimistic local flip, by the LOCAL hunk id, before the host echoes its own status patch.
  // This panel owns the dispatch for EVERY caller, so the hunk_id always rides along. There is no
  // second, hunk_id-less route any more: the Home diff panel binds this same prop.
  onStatus?: (hunkId: string, status: HunkStatus) => void;
  active?: boolean; // when false, keyboard handlers detach (the panel is not focused)
}) {
  const [sel, setSel] = useState(0);
  // hunks that just settled, for the one-shot light settle, keyed by id -> action.
  const [settling, setSettling] = useState<Record<string, HunkAction>>({});
  const [openDetail, setOpenDetail] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<Record<string, string>>({});
  const [status, setStatus] = useState<string | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const sessionId = useStore((s) => s.sessionId);
  const activeRunId = useStore((s) => s.activeRunId);
  const hunkRefs = useRef<(HTMLDivElement | null)[]>([]);
  const rootRef = useRef<HTMLDivElement>(null);

  const pendingIdx = useMemo(
    () => doc.hunks.map((h, i) => (h.status === "pending" ? i : -1)).filter((i) => i >= 0),
    [doc.hunks],
  );

  const runId = doc.run_id || activeRunId || "";

  const act = useCallback(
    (idx: number, id: HunkActionId, note?: string) => {
      const hunk = doc.hunks[idx];
      if (!hunk) return;
      if (!hunkActionsFor(hunk.status).includes(id)) return;
      const spec = hunkActionSpec(id);
      const next = statusAfter(id, hunk.status);
      setBusy(hunk.id);
      setFailed((f) => ({ ...f, [hunk.id]: "" }));
      setStatus(`${spec.label}: working`);
      void runHunkAction(id, { diffId: doc.diff_id, runId, sessionId, path: doc.path, hunk, note })
        .then((ack) => {
          setBusy(null);
          const state = ackState(ack);
          if (state === "refused") {
            setFailed((f) => ({ ...f, [hunk.id]: ack.message ?? "The host refused that action" }));
            setStatus(`${spec.label}: refused`);
            return;
          }
          // A hold is not a done. A reject with no hunk id resolves to the approval-gated whole-diff
          // revert (host effect_command), so this surface can be handed a held ack; printing "done"
          // and flipping the row claimed an on-disk revert that has not run yet.
          if (state === "held") {
            setStatus(heldNote(spec.label));
            return;
          }
          setStatus(`${spec.label}: done`);
          if (next !== hunk.status) {
            setSettling((s) => ({ ...s, [hunk.id]: next === "accepted" ? "accept" : "reject" }));
            onStatus?.(hunk.id, next);
          }
        })
        .catch((e) => {
          setBusy(null);
          setFailed((f) => ({ ...f, [hunk.id]: e instanceof Error ? e.message : String(e) }));
          setStatus(`${spec.label}: failed`);
        })
        // Focus never falls to the body when a control disappears with its status flip.
        .finally(() => hunkRefs.current[idx]?.focus());
      // advance to the next still-pending hunk so the flow stays on the keyboard.
      if (next !== hunk.status) {
        const after = pendingIdx.find((i) => i > idx);
        if (after != null) setSel(after);
      }
    },
    [doc.hunks, doc.diff_id, doc.path, onStatus, pendingIdx, runId, sessionId],
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
      const el = e.target as HTMLElement | null;
      const surface = reviewSurface(rootRef.current);
      // Scoped to the diff surface, not to the window: focus decides whether these keys mean
      // anything at all. INPUT / TEXTAREA (the concern input, Monaco's hidden input) keep their own.
      if (!reviewKeysActive(!!surface?.contains(document.activeElement), el?.tagName)) return;
      // Cmd+Enter / Cmd+Backspace are the catalog shortcuts for accept_diff / reject_diff.
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
        case "m": {
          e.preventDefault();
          const h = doc.hunks[sel];
          if (h) setOpenDetail((d) => (d === h.id ? null : h.id));
          break;
        }
        case "d":
          e.preventDefault();
          setReviewOpen((v) => !v);
          break;
        case "Escape":
          if (openDetail) {
            e.preventDefault();
            setOpenDetail(null);
          }
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, sel, act, move, doc.hunks, openDetail]);

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
          fontSize: "var(--fs-small)",
          color: "var(--text-muted)",
          minWidth: 0,
        }}
        onContextMenu={(e) => {
          e.preventDefault();
          setReviewOpen((v) => !v);
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
          {status ?? `${remaining} hunk${remaining === 1 ? "" : "s"} to review`}
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--ma-1)", flex: "0 0 auto", flexWrap: "wrap" }}>
          <kbd style={kbd}>j</kbd>
          <kbd style={kbd}>k</kbd>
          <span>move</span>
          <kbd style={kbd}>a</kbd>
          <span>accept</span>
          <kbd style={kbd}>r</kbd>
          <span>reject</span>
          <kbd style={kbd}>m</kbd>
          <span>hunk detail</span>
          <kbd style={kbd}>d</kbd>
          <span>review</span>
        </span>
      </div>

      {reviewOpen ? <ReviewDetail doc={doc} onClose={() => setReviewOpen(false)} onStatus={setStatus} /> : null}

      <div role="list" aria-label={`Diff hunks for ${doc.path}`} style={{ overflow: "auto", padding: "var(--ma-3)", display: "flex", flexDirection: "column", gap: "var(--ma-3)", minHeight: 0, minWidth: 0 }}>
        {doc.hunks.map((h, i) => (
          <HunkCard
            key={h.id}
            hunk={h}
            path={doc.path}
            selected={i === sel}
            settle={settling[h.id]}
            busy={busy === h.id}
            error={failed[h.id] || null}
            detailOpen={openDetail === h.id}
            innerRef={(el) => (hunkRefs.current[i] = el)}
            onSelect={() => setSel(i)}
            onToggleDetail={() => {
              setSel(i);
              setOpenDetail((d) => (d === h.id ? null : h.id));
            }}
            onAct={(id, note) => act(i, id, note)}
          />
        ))}
        {doc.hunks.length === 0 ? (
          <div className="t-body" style={{ color: "var(--text-3)", padding: "var(--ma-6)" }}>No hunks in this change</div>
        ) : null}
      </div>
    </div>
  );
}

/** The whole-diff detail: what the review currently says, and the receipt export. Opened with d or
 *  a right-click on the header, so the panel gains no permanent control. */
function ReviewDetail({ doc, onClose, onStatus }: { doc: DiffDoc; onClose: () => void; onStatus: (s: string) => void }) {
  const receipt = reviewReceipt(doc);
  const covered = receipt.hunks.filter((h) => h.base_hash).length;
  return (
    <div
      role="group"
      aria-label="Review detail"
      style={{ padding: "var(--ma-2) var(--ma-3)", boxShadow: "inset 0 -1px 0 0 var(--border)", fontSize: "var(--fs-small)", color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: "var(--ma-1)" }}
    >
      <span className="t-code" style={{ color: "var(--text)" }}>{doc.diff_id}</span>
      <span>
        {receipt.hunks.length} hunk{receipt.hunks.length === 1 ? "" : "s"}
        {Object.entries(receipt.counts).map(([k, n]) => `, ${n} ${k}`)}
      </span>
      <span>{covered} of {receipt.hunks.length} carry a base hash</span>
      <div style={{ display: "flex", gap: "var(--ma-2)", paddingTop: "var(--ma-1)" }}>
        <MenuBtn
          label="Copy review receipt"
          onClick={() =>
            void exportReviewReceipt(doc)
              .then(() => onStatus("Review receipt copied"))
              .catch((e) => onStatus(`Receipt export failed: ${e instanceof Error ? e.message : String(e)}`))
          }
        />
        <MenuBtn label="Close detail" onClick={onClose} />
      </div>
    </div>
  );
}

function HunkCard({
  hunk,
  path,
  selected,
  settle,
  busy,
  error,
  detailOpen,
  innerRef,
  onSelect,
  onToggleDetail,
  onAct,
}: {
  hunk: Hunk;
  path: string;
  selected: boolean;
  settle?: HunkAction;
  busy: boolean;
  error: string | null;
  detailOpen: boolean;
  innerRef: (el: HTMLDivElement | null) => void;
  onSelect: () => void;
  onToggleDetail: () => void;
  onAct: (id: HunkActionId, note?: string) => void;
}) {
  const decided = hunk.status !== "pending";
  const st = STATUS_LABEL[hunk.status];
  // Reject fades the card out; accept just settles (no glow, flat surface).
  const settleStyle =
    settle === "reject" ? { animation: "hunk-dissolve var(--dur-door) var(--ease) forwards" } : {};
  const selectedPending = selected && !decided && !settle;

  return (
    <div
      ref={innerRef}
      tabIndex={-1}
      onClick={onSelect}
      onContextMenu={(e) => {
        e.preventDefault();
        onToggleDetail();
      }}
      role="listitem"
      aria-current={selectedPending ? "true" : undefined}
      aria-label={`Hunk ${hunk.header}, ${busy ? "working" : error ? "failed" : st.label}`}
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
          fontSize: "var(--fs-small)",
          color: "var(--text-muted)",
          minWidth: 0,
        }}
      >
        <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{hunk.header}</span>
        <span style={{ marginLeft: "auto", color: st.color, flex: "0 0 auto" }}>{busy ? "working" : st.label}</span>
      </div>

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--fs-small)", lineHeight: 1.6, overflowX: "auto", minWidth: 0 }}>
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

      {error ? (
        <div role="alert" style={{ padding: "var(--ma-2) var(--ma-3)", color: "var(--git-del)", fontSize: "var(--fs-small)" }}>
          failed: {error}
        </div>
      ) : null}

      <div style={{ display: "flex", gap: "var(--ma-2)", alignItems: "center", padding: "var(--ma-3) var(--ma-4)", boxShadow: "inset 0 1px 0 0 var(--line)" }}>
        {!decided ? (
          <>
            <ActBtn label="Accept" hint="a" tone="accept" disabled={busy} onClick={(e) => { e.stopPropagation(); onAct("accept"); }} />
            <ActBtn label="Reject" hint="r" tone="reject" disabled={busy} onClick={(e) => { e.stopPropagation(); onAct("reject"); }} />
          </>
        ) : null}
        <button
          className="text-button"
          aria-expanded={detailOpen}
          title="Hunk detail: provenance, base hash and the rest of the actions (m)"
          onClick={(e) => {
            e.stopPropagation();
            onToggleDetail();
          }}
          style={{ marginLeft: decided ? 0 : "auto", fontSize: "var(--fs-small)", color: "var(--text-muted)", background: "none", border: "none" }}
        >
          {detailOpen ? "hide detail" : "detail"} <kbd style={kbd}>m</kbd>
        </button>
      </div>

      {detailOpen ? <HunkDetail hunk={hunk} path={path} busy={busy} onAct={onAct} /> : null}
    </div>
  );
}

/** Provenance, base hash, the invalidated verification, and every action the hunk's status allows.
 *  This is the contextual menu: it exists only while it is open, so the panel gains no new control.
 *  Exported so the review test renders the real menu rather than a copy of it. */
export function HunkDetail({
  hunk,
  path,
  busy,
  onAct,
}: {
  hunk: Hunk;
  path: string;
  busy: boolean;
  onAct: (id: HunkActionId, note?: string) => void;
}) {
  const [note, setNote] = useState("");
  const prov = hunkProvenance(hunk);
  const base = hunkBaseHash(hunk);
  const file = hunkFile(hunk, path);
  const ids = hunkActionsFor(hunk.status);
  return (
    <div
      role="menu"
      aria-label={`Actions for hunk ${hunk.header}`}
      style={{ padding: "var(--ma-3) var(--ma-4)", boxShadow: "inset 0 1px 0 0 var(--line)", display: "flex", flexDirection: "column", gap: "var(--ma-2)", fontSize: "var(--fs-small)", color: "var(--text-muted)" }}
      onClick={(e) => e.stopPropagation()}
    >
      <dl style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px var(--ma-3)", margin: 0 }}>
        <dt>file</dt>
        <dd className="t-code" style={{ margin: 0, color: "var(--text)" }}>{file}</dd>
        <dt>agent</dt>
        <dd style={{ margin: 0 }}>{prov?.agent ?? "not recorded"}</dd>
        <dt>turn</dt>
        <dd style={{ margin: 0 }}>{prov?.turn ?? "not recorded"}</dd>
        <dt>plan step</dt>
        <dd style={{ margin: 0 }}>{prov?.plan_step ?? "no originating plan step"}</dd>
        <dt>base hash</dt>
        <dd className="t-code" style={{ margin: 0, overflowWrap: "anywhere" }}>{base ?? "not recorded"}</dd>
      </dl>

      {hunk.status === "rejected" ? (
        <p style={{ margin: 0, color: "var(--git-mod)" }}>{invalidatedNote(file)}</p>
      ) : (
        <p style={{ margin: 0 }}>Rejecting reverts {file} through the verifying applier. {invalidatedNote(file)}</p>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-2)" }}>
        {ids
          .filter((id) => !hunkActionSpec(id).needsNote)
          .map((id) => {
            const spec = hunkActionSpec(id);
            return (
              <MenuBtn
                key={id}
                label={spec.label}
                shortcut={spec.shortcut}
                destructive={spec.destructive}
                disabled={busy}
                onClick={() => onAct(id)}
              />
            );
          })}
      </div>

      <div style={{ display: "flex", gap: "var(--ma-2)" }}>
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && note.trim()) {
              e.preventDefault();
              onAct("concern", note);
              setNote("");
            }
          }}
          placeholder="Concern about this hunk"
          aria-label={`Attach a concern about hunk ${hunk.header}`}
          spellCheck={false}
          style={{ flex: 1, minWidth: 0, background: "var(--input-bg)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "var(--radius-sm)", padding: "2px var(--ma-2)", fontFamily: "var(--font-mono)", fontSize: "var(--fs-small)" }}
        />
        <MenuBtn
          label="Attach a concern"
          disabled={busy || !note.trim()}
          onClick={() => {
            onAct("concern", note);
            setNote("");
          }}
        />
      </div>
    </div>
  );
}

function MenuBtn({
  label,
  shortcut,
  destructive,
  disabled,
  onClick,
}: {
  label: string;
  shortcut?: string;
  destructive?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      title={shortcut ? `${label} (${shortcut})` : label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "3px var(--ma-3)",
        borderRadius: "var(--radius-sm)",
        fontSize: "var(--fs-small)",
        color: destructive ? "var(--git-del)" : "var(--text)",
        background: "var(--input-bg)",
        border: "1px solid var(--border-strong)",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {label}
      {shortcut ? <kbd style={kbd}>{shortcut}</kbd> : null}
    </button>
  );
}

function ActBtn({
  label,
  hint,
  tone,
  disabled,
  onClick,
}: {
  label: string;
  hint: string;
  tone: "accept" | "reject";
  disabled?: boolean;
  onClick: (e: MouseEvent) => void;
}) {
  const isAccept = tone === "accept";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={isAccept ? "Accept this hunk (a)" : "Reject this hunk, reverting the file (r)"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "4px var(--ma-3)",
        borderRadius: "var(--radius-sm)",
        fontSize: "var(--fs-small)",
        color: isAccept ? "var(--accent-text)" : "var(--text)",
        background: isAccept ? "var(--accent)" : "var(--input-bg)",
        border: isAccept ? "none" : "1px solid var(--border-strong)",
        opacity: disabled ? 0.5 : 1,
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
  fontSize: "var(--fs-label)",
};

const kbd: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: "var(--fs-label)",
  padding: "1px 5px",
  borderRadius: 3,
  color: "var(--text-muted)",
  border: "1px solid var(--border-strong)",
  background: "var(--surface-2)",
};
