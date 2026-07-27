/*
  CodeActions.tsx: what a selection in the editor can do. A contextual menu anchored to the
  selection, opened by selecting code (or Shift+F10 / the context-menu key on the editor). No
  permanent editor control is added and nothing here invents a verb: every entry resolves to a
  catalog command id through runCommand, or to the ONE search engine in src/ui.tsx.

  A selection resolves to a stable SourceRef: path, line range and a content hash. The hash is
  re-read from the live buffer before every dispatch, so a selection whose text has moved or changed
  is STALE and refuses instead of citing lines that no longer say what they said.

  RETIRED with this stage:
    "fork and try 3"  -> a fleet_run action the header advertised and the menu never rendered. The
                         comment is gone with it.
    "refactor"        -> dispatched Intent::Custom{inline_edit}, a name reserved in
                         crates/hide-protocol/src/command.rs with NO host handler (log-only). The
                         same request now goes to the agent over submit_turn and says so.

  Deliberately ABSENT, with the reason (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md):
    add to context      -> `pin_span` is reserved on the wire with no host handler. Putting a
                           selection in front of the model IS the attach entry, so the two merged
                           rather than shipping a second control for one capability.
    verify SELECTION    -> the host's checker is file-scoped, so the menu runs it on the FILE and
                           says so. `run_static_analysis` is Custom-bound with a real host arm
                           (handle_static_analysis_intent, which publishes the diagnostics the
                           Problems counter reads), so the entry is a real check now, not a request
                           to the agent. `goal_evaluate` IS reachable too, but it grades the session
                           goal, not code, so it stays on the goal surface.
    mark protected scope-> no host capability exists (nothing in hide-backend, hide-core or
                           hide-protocol records a protected scope), so no control claims one.
*/
import { useEffect, useRef, useState } from "react";
import { runCommand, useStore } from "../../store";
import { ackState, heldNote, type IntentAck } from "../../wire";
import { HitRow, runHitAction, searchAll, type SearchHit } from "../../ui";

/* ---- The stable source reference --------------------------------------------------------------- */

export interface SourceRef {
  path: string;
  /** 1-based, inclusive. */
  startLine: number;
  endLine: number;
  /** Hash of the selected text, so a moved or edited selection is detectable. */
  hash: string;
  text: string;
}

/** FNV-1a, 32 bit, hex. Not a security hash: it only has to notice that a range changed.
 *  ponytail: the host seals with blake3; this side just needs cheap drift detection. */
export function hashText(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

export function sourceRef(path: string, startLine: number, endLine: number, text: string): SourceRef {
  return { path, startLine, endLine, hash: hashText(text), text };
}

/** How a selection is cited to the agent and in the receipt trail. */
export const citeRef = (ref: SourceRef): string =>
  `${ref.path}:${ref.startLine}${ref.endLine > ref.startLine ? `-${ref.endLine}` : ""}`;

/** True when the buffer no longer holds what the ref was taken from (moved, edited, or gone). */
export function isStale(ref: SourceRef, current: string | null): boolean {
  return current == null || hashText(current) !== ref.hash;
}

/** The quoted body a prompt carries. Long selections are cut so a turn is never a whole file. */
export function refBody(ref: SourceRef, limit = 600): string {
  return ref.text.length > limit ? ref.text.slice(0, limit) + "\n[cut]" : ref.text;
}

/** The symbol a reference lookup asks about: the first identifier in the selection. The code_index
 *  `references` leg takes a symbol, not a blob, so a whole-block selection still resolves to one. */
export function symbolOf(text: string): string {
  return /[A-Za-z_][A-Za-z0-9_]*/.exec(text)?.[0] ?? "";
}

/* ---- The action table --------------------------------------------------------------------------- */

export type SelActionId =
  | "explain"
  | "attach"
  | "references"
  | "trace_callers"
  | "side_chat"
  | "plan_step"
  | "test"
  | "verify"
  | "history";

export interface SelActionSpec {
  id: SelActionId;
  label: string;
  /** Catalog command id, or null when the route is a custom intent or the search engine. */
  command: string | null;
}

export const SEL_ACTIONS: SelActionSpec[] = [
  { id: "explain", label: "Explain this selection", command: "submit_turn" },
  // Merged: "add to context" is this. A live run is steered, otherwise the turn opens.
  { id: "attach", label: "Attach to the current turn", command: null },
  { id: "references", label: "Find references", command: null },
  { id: "trace_callers", label: "Ask the agent to trace the callers", command: "submit_turn" },
  { id: "side_chat", label: "Side chat about this selection", command: "create_side_chat" },
  { id: "plan_step", label: "Ask the agent to plan a change here", command: "submit_turn" },
  { id: "test", label: "Ask the agent to write a test for this", command: "submit_turn" },
  { id: "verify", label: "Run the checks on this file", command: "run_static_analysis" },
  { id: "history", label: "Inspect history for these lines", command: "run_command" },
];

export const selActionSpec = (id: SelActionId): SelActionSpec =>
  SEL_ACTIONS.find((a) => a.id === id) as SelActionSpec;

/** `references` answers with hits instead of an ack, so it is not part of the dispatch table. */
export const DISPATCHED: SelActionId[] = SEL_ACTIONS.filter((a) => a.id !== "references").map((a) => a.id);

export interface SelCtx {
  sessionId: string;
  runId: string;
}

/**
 * THE dispatch point for every selection gesture. Refuses a stale ref rather than citing lines that
 * have moved, and throws (never silently no-ops) so the caller shows the refusal.
 */
export async function runSelectionAction(id: SelActionId, ref: SourceRef, ctx: SelCtx, stale = false): Promise<IntentAck> {
  if (stale)
    throw new Error(`${citeRef(ref)} changed since it was selected. Select it again.`);
  const cite = citeRef(ref);
  const quoted = `${cite}\n\n${refBody(ref)}`;
  const ask = (text: string) => runCommand("submit_turn", { session_id: ctx.sessionId, text });

  switch (id) {
    case "explain":
      return ask(`Explain this code:\n\n${quoted}`);

    case "attach": {
      const text = `Referring to ${quoted}`;
      if (ctx.runId)
        return runCommand("steer", { run_id: ctx.runId, session_id: ctx.sessionId, text });
      return ask(text);
    }

    case "trace_callers":
      return ask(`Trace the callers of this code and report the call chain:\n\n${quoted}`);

    case "side_chat":
      return runCommand("create_side_chat", { session_id: ctx.sessionId, inherit: true });

    case "plan_step":
      return ask(`Add a plan step covering this code, then wait for approval:\n\n${quoted}`);

    case "test":
      return ask(`Write a test covering this code:\n\n${quoted}`);

    case "verify":
      // The real, model-free Tier1 oracle over the file this selection lives in. `paths` is the
      // form the host arm accepts from an app that has file paths rather than editor buffers.
      return runCommand("run_static_analysis", { session_id: ctx.sessionId, paths: [ref.path] });

    case "history":
      // A real terminal run through the catalog command, scoped to exactly these lines.
      return runCommand("run_command", {
        argv: ["git", "log", "-L", `${ref.startLine},${ref.endLine}:${ref.path}`, "--max-count=20"],
      });

    case "references":
      throw new Error("Find references answers with results, not an ack");
  }
}

/* ---- The surface --------------------------------------------------------------------------------- */

export function CodeActions({
  sel,
  top,
  left,
  readRange,
  onDone,
}: {
  sel: SourceRef;
  top: number;
  left: number;
  /** Re-reads the ref's line range from the live buffer, for stale detection. */
  readRange: (ref: SourceRef) => string | null;
  onDone: () => void;
}) {
  const sessionId = useStore((s) => s.sessionId);
  const runId = useStore((s) => s.activeRunId) ?? "";
  const ref = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [hitSel, setHitSel] = useState(0);
  const [stale, setStale] = useState(false);

  // Focus the first action when the menu opens (keyboard users land inside it).
  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>(".codeactions__btn")?.focus();
  }, []);

  // The buffer can change under an open menu, so the staleness is re-read, not assumed.
  useEffect(() => {
    const check = () => setStale(isStale(sel, readRange(sel)));
    check();
    const t = setInterval(check, 600);
    return () => clearInterval(t);
  }, [sel, readRange]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onDone();
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      // ONE ring: the action buttons, then the reference rows. The arrows used to cycle the actions
      // alone, so an open references list could be read but never walked or chosen from without a
      // mouse (its selection was set by hover and by nothing else).
      const btns = Array.from(ref.current?.querySelectorAll<HTMLButtonElement>(".codeactions__btn, .search-hit") ?? []);
      if (!btns.length) return;
      const i = btns.indexOf(document.activeElement as HTMLButtonElement);
      const next = e.key === "ArrowDown" ? (i + 1) % btns.length : (i - 1 + btns.length) % btns.length;
      btns[next]?.focus();
      // Keep the listbox selection on the focused row (Enter, Mod+Enter and Mod+Shift+Enter then act
      // on it through HitRow's own click handler, which carries the modifiers).
      if (next >= SEL_ACTIONS.length) setHitSel(next - SEL_ACTIONS.length);
    }
  };

  const act = (id: SelActionId) => {
    const spec = selActionSpec(id);
    const fresh = isStale(sel, readRange(sel));
    setStale(fresh);
    if (id === "references") {
      const symbol = symbolOf(sel.text);
      if (!symbol) {
        setStatus("No symbol in this selection to look up");
        return;
      }
      setStatus(`Searching references to ${symbol}`);
      void searchAll(symbol, ["references"], { sessionId, runId })
        .then((r) => {
          setHits(r.hits);
          setHitSel(0);
          setStatus(r.hits.length ? `${r.hits.length} reference hits` : r.errors[0] ?? "No references found");
        })
        .catch((e) => setStatus(e instanceof Error ? e.message : String(e)));
      return;
    }
    setStatus(`${spec.label}: working`);
    void runSelectionAction(id, sel, { sessionId, runId }, fresh)
      // `history` dispatches run_command, which the host parks at the security gate for a
      // destructive argv, so a two-state read here could print "sent" for something never run.
      .then((ack) => {
        const state = ackState(ack);
        setStatus(
          state === "held"
            ? heldNote(spec.label)
            : state === "accepted"
              ? `${spec.label}: sent`
              : ack.message ?? "The host refused that action",
        );
      })
      .catch((e) => setStatus(e instanceof Error ? e.message : String(e)));
  };

  return (
    <div
      ref={ref}
      className="codeactions glass"
      style={{ top, left, maxHeight: 320, overflowY: "auto" }}
      role="menu"
      aria-label={`Actions for ${citeRef(sel)}`}
      onKeyDown={onKeyDown}
    >
      <div className="t-micro" style={{ padding: "2px 8px", color: "var(--text-dim)", display: "flex", gap: 6, alignItems: "center" }}>
        <span className="t-code">{citeRef(sel)}</span>
        {stale ? (
          <span aria-label="Selection is stale; select it again" title="The buffer changed under this selection" style={{ color: "var(--git-mod)" }}>
            <span aria-hidden>△</span> stale
          </span>
        ) : null}
      </div>

      {SEL_ACTIONS.map((a) => (
        <button
          key={a.id}
          className="codeactions__btn"
          role="menuitem"
          aria-disabled={stale && a.id !== "references"}
          title={a.command ? `${a.label} (${a.command})` : a.label}
          onClick={() => act(a.id)}
        >
          {a.label}
        </button>
      ))}

      {hits?.length ? (
        <div role="listbox" aria-label="References" style={{ maxHeight: 140, overflowY: "auto", borderTop: "1px solid var(--border)" }}>
          {hits.map((h, i) => (
            <HitRow
              key={h.key}
              hit={h}
              selected={i === hitSel}
              onHover={() => setHitSel(i)}
              onActivate={(action, hit) =>
                void runHitAction(action, hit, { sessionId, runId })
                  .then(() => onDone())
                  .catch((e) => setStatus(e instanceof Error ? e.message : String(e)))
              }
            />
          ))}
        </div>
      ) : null}

      {status ? (
        <div role="status" aria-live="polite" className="t-micro" style={{ padding: "2px 8px", color: "var(--text-dim)" }}>
          {status}
        </div>
      ) : null}
    </div>
  );
}
