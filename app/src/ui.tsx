import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useFocusTrap } from "./shell/a11y";
import { callConnector, subscribeUi } from "./ipc";
import { runCommand, useStore } from "./store";
import { type IntentAck } from "./wire";
import { keyLabel } from "./surfaces/chat/actions";

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
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  title?: string;
  style?: CSSProperties;
  disabled?: boolean;
}) {
  return (
    <button className="gate" onClick={onClick} title={title} style={style} disabled={disabled}>
      {children}
    </button>
  );
}

export interface Command {
  id: string;
  label: string;
  /** The chord that really fires this command (store.boundShortcuts), or null when it has none. */
  shortcut?: string | null;
  run: () => void;
}

/* ---- THE ONE search experience ---------------------------------------------------------------
  Search is a capability, not a panel. This engine serves BOTH entry points (the palette overlay and
  the navigator field in Explorer.tsx), so there is no second search implementation to drift, and a
  hit means the same thing wherever it is shown.

  Real backends only:
    files / symbols          -> code_index.search      { query: { text, limit, include_* } }
    references               -> code_index.references  { symbol }
    transcript / threads / tools
                             -> the `run_search` catalog command (Custom intent), whose hits come
                                back as a `search_results` Custom UiEvent (hide-backend host.rs
                                handle_search_intent + publish_search_results).

  Semantic search is DEFERRED_MODEL_REQUIRED: `include_semantic` is pinned false and no scope offers
  it, so nothing here implies a capability that does not exist. Doctrine scopes with no honest index
  are ABSENT rather than faked: diff hunks and plan steps are not indexed anywhere, so those origins
  default to the scopes that really cover them (the changed files, and the `tool.result` items that
  carry test and verification output).
*/

/** Where the search was invoked from. Drives the DEFAULT scope set, nothing else. */
export type SearchOrigin = "editor" | "chat" | "diff" | "terminal" | "plan" | "global";

export type ScopeId = "files" | "symbols" | "references" | "transcript" | "threads" | "tools";

export interface ScopeSpec {
  id: ScopeId;
  label: string;
  /** The real backend that answers this scope. Shown as result provenance. */
  source: "code index" | "session log";
  hint: string;
}

export const SCOPES: ScopeSpec[] = [
  { id: "files", label: "Files", source: "code index", hint: "Literal matches in indexed workspace files" },
  { id: "symbols", label: "Symbols", source: "code index", hint: "Symbol definitions in the code index" },
  { id: "references", label: "References", source: "code index", hint: "Occurrences of the query as a symbol" },
  { id: "transcript", label: "Transcript", source: "session log", hint: "Items in the current session" },
  { id: "threads", label: "Threads", source: "session log", hint: "Items across every session" },
  { id: "tools", label: "Tool output", source: "session log", hint: "Tool and terminal results, logs, artifacts" },
];

export const scopeSpec = (id: ScopeId): ScopeSpec => SCOPES.find((s) => s.id === id) as ScopeSpec;

/** Context-sensitive default scope, by the surface the search was opened from. */
const DEFAULT_SCOPES: Record<SearchOrigin, ScopeId[]> = {
  editor: ["files", "symbols", "references"],
  chat: ["transcript", "threads", "tools"],
  // A diff is a set of changed files, and its tests report through tool.result items.
  diff: ["files", "tools"],
  terminal: ["tools"],
  // A plan step and its verification are both durable transcript items.
  plan: ["transcript", "tools"],
  global: ["files", "symbols", "references", "transcript", "threads", "tools"],
};

export const defaultScopes = (origin: SearchOrigin): ScopeId[] => DEFAULT_SCOPES[origin] ?? DEFAULT_SCOPES.global;

// The current origin, set by whichever surface owns the focus. Module-level (not store state)
// because it is a hint for ONE transient overlay, not part of the event-folded truth.
let ORIGIN: SearchOrigin = "global";
export const setSearchOrigin = (o: SearchOrigin): void => void (ORIGIN = o);
export const searchOrigin = (): SearchOrigin => ORIGIN;

/** One hit, whatever produced it. `path`/`line` or `event_id` is the provenance back to the source. */
export interface SearchHit {
  key: string;
  scope: ScopeId;
  title: string;
  preview: string;
  path?: string;
  line?: number;
  session_id?: string;
  event_id?: string;
  role?: string;
}

export interface SearchOutcome {
  hits: SearchHit[];
  /** Per-leg failures, surfaced in the surface rather than swallowed. */
  errors: string[];
}

export const SEARCH_LIMIT = 20;

const trimLine = (s: string): string => s.replace(/\s+/g, " ").trim();

// -- Wire normalizers. Pure, so every shape is tested without a backend. ------------------------

/** hawking-index SearchResult rows: `{ results: [{ span: { path, range }, title, snippet }] }`.
 *  Also accepts a bare array and the flat `{ path, line, preview }` row a local fallback produces. */
export function indexHits(raw: unknown, scope: ScopeId, limit: number): SearchHit[] {
  const box = raw as { results?: unknown } | null;
  const rows = Array.isArray(raw) ? raw : Array.isArray(box?.results) ? (box.results as unknown[]) : [];
  const out: SearchHit[] = [];
  for (const row of rows.slice(0, limit)) {
    const r = row as {
      span?: { path?: string; range?: { start_line?: number } | null };
      title?: string;
      snippet?: string;
      path?: string;
      line?: number;
      preview?: string;
    };
    const path = r.span?.path ?? r.path;
    if (!path) continue;
    const line = r.span?.range?.start_line ?? r.line ?? 0;
    out.push({
      key: `${scope}:${path}:${line}:${out.length}`,
      scope,
      title: path,
      preview: trimLine(r.snippet ?? r.preview ?? r.title ?? ""),
      path,
      line: line || undefined,
    });
  }
  return out;
}

/** hawking-index Occurrence rows: `{ occurrences: [{ symbol, file, range, role }] }`. */
export function referenceHits(raw: unknown, limit: number): SearchHit[] {
  const box = raw as { occurrences?: unknown } | null;
  const rows = Array.isArray(box?.occurrences) ? (box.occurrences as unknown[]) : [];
  const out: SearchHit[] = [];
  for (const row of rows.slice(0, limit)) {
    const r = row as { symbol?: string; file?: string; range?: { start_line?: number } | null; role?: string };
    if (!r.file) continue;
    out.push({
      key: `references:${r.file}:${r.range?.start_line ?? 0}:${out.length}`,
      scope: "references",
      title: r.file,
      preview: trimLine(`${r.symbol ?? ""} ${r.role ?? ""}`),
      path: r.file,
      line: r.range?.start_line || undefined,
      role: r.role,
    });
  }
  return out;
}

/** hide-backend TranscriptHit rows: `{ session_id, event_id, seq, kind, role, snippet, ts }`. */
export function transcriptHits(raw: unknown, scope: ScopeId, limit: number): SearchHit[] {
  const rows = Array.isArray(raw) ? raw : [];
  const out: SearchHit[] = [];
  for (const row of rows.slice(0, limit)) {
    const r = row as {
      session_id?: string;
      event_id?: string;
      kind?: string;
      role?: string;
      snippet?: string;
    };
    if (!r.event_id) continue;
    out.push({
      key: `${scope}:${r.event_id}`,
      scope,
      title: `${r.role ?? r.kind ?? "item"} item`,
      preview: trimLine(r.snippet ?? ""),
      session_id: r.session_id,
      event_id: r.event_id,
      role: r.role,
    });
  }
  return out;
}

/** Where a hit came from, in one line. Never color-only, and always resolvable to a real source. */
export function hitProvenance(hit: SearchHit): string {
  if (hit.path) return hit.line ? `${hit.path}:${hit.line}` : hit.path;
  if (hit.event_id) return hit.session_id ? `${hit.event_id} in ${hit.session_id}` : hit.event_id;
  return scopeSpec(hit.scope).source;
}

/** The full accessible name: what it is, where it came from, and the preview text. */
export const hitLabel = (hit: SearchHit): string =>
  `${hit.title}, ${hitProvenance(hit)}, from the ${scopeSpec(hit.scope).source}${hit.preview ? `, ${hit.preview}` : ""}`;

/** The reference text an attach or a side chat carries, so the agent gets the source, not a summary. */
export const citeHit = (hit: SearchHit): string =>
  `${hitProvenance(hit)}${hit.preview ? `\n${hit.preview}` : ""}`;

// -- The legs ----------------------------------------------------------------------------------

/** The exact `run_search` payload for a scope. Structured filters only; no semantic leg. */
export function searchPayload(
  query: string,
  scope: ScopeId,
  sessionId: string,
  limit: number,
): Record<string, unknown> {
  const payload: Record<string, unknown> = { query, limit };
  // `transcript` is this session; `threads` deliberately omits session_id so the host searches every
  // session; `tools` filters to the tool.result event kind (terminal output, logs, artifacts).
  if (scope === "transcript") payload.session_id = sessionId;
  if (scope === "tools") payload.kind = "tool.result";
  return payload;
}

/** Wait for the host's `search_results` Custom UiEvent for `query`.
 *  ponytail: correlation is by the echoed query text, which is all the host sends, so exactly one
 *  run_search leg is ever in flight (see searchAll). A request id on the wire would make it exact. */
function awaitSearchResults(query: string, timeoutMs = 4000): Promise<unknown[]> {
  return new Promise((resolve) => {
    let stop: (() => void) | null = null;
    let done = false;
    const finish = (hits: unknown[]) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      stop?.();
      resolve(hits);
    };
    const timer = setTimeout(() => finish([]), timeoutMs);
    stop = subscribeUi(
      (ev) => {
        if (ev.kind.type !== "custom") return;
        const d = ev.kind.data as { kind?: string; query?: string; hits?: unknown } | null;
        if (!d || d.kind !== "search_results" || d.query !== query) return;
        finish(Array.isArray(d.hits) ? d.hits : []);
      },
      () => finish([]),
    );
    if (done) stop();
  });
}

async function searchLeg(query: string, scope: ScopeId, ctx: SearchCtx): Promise<SearchHit[]> {
  const limit = ctx.limit ?? SEARCH_LIMIT;
  if (scope === "references") {
    return referenceHits(await callConnector<unknown>("code_index", "references", { symbol: query }), limit);
  }
  if (scope === "files" || scope === "symbols") {
    const raw = await callConnector<unknown>("code_index", "search", {
      query: {
        text: query,
        limit,
        include_symbols: scope === "symbols",
        include_lexical: scope === "files",
        include_semantic: false, // DEFERRED_MODEL_REQUIRED
      },
    });
    return indexHits(raw, scope, limit);
  }
  const waiting = awaitSearchResults(query);
  await runCommand("run_search", searchPayload(query, scope, ctx.sessionId, limit));
  return transcriptHits(await waiting, scope, limit);
}

export interface SearchCtx {
  sessionId: string;
  runId: string;
  limit?: number;
}

const LOG_SCOPES: ScopeId[] = ["transcript", "threads", "tools"];

/** THE search. Every entry point calls this; nobody else dials a search backend. */
export async function searchAll(query: string, scopes: ScopeId[], ctx: SearchCtx): Promise<SearchOutcome> {
  const q = query.trim();
  if (!q) return { hits: [], errors: [] };
  const hits: SearchHit[] = [];
  const errors: string[] = [];
  const record = (scope: ScopeId, e: unknown) =>
    errors.push(`${scopeSpec(scope).label} search failed: ${e instanceof Error ? e.message : String(e)}`);

  // The index legs are independent, so they run together.
  const indexScopes = scopes.filter((s) => !LOG_SCOPES.includes(s));
  const settled = await Promise.all(
    indexScopes.map((s) => searchLeg(q, s, ctx).catch((e) => (record(s, e), [] as SearchHit[]))),
  );
  for (const rows of settled) hits.push(...rows);

  // The log legs share one correlation key (the query text), so they go one at a time.
  for (const s of scopes.filter((x) => LOG_SCOPES.includes(x))) {
    try {
      hits.push(...(await searchLeg(q, s, ctx)));
    } catch (e) {
      record(s, e);
    }
  }
  return { hits, errors };
}

// -- Acting on a hit ---------------------------------------------------------------------------

export type HitAction = "open" | "attach" | "side_chat";

/** Modifier-key actions on the selected row, so nothing needs a mouse or a new button. */
export function hitActionFor(e: Pick<KeyboardEvent, "metaKey" | "ctrlKey" | "shiftKey">): HitAction {
  if (!(e.metaKey || e.ctrlKey)) return "open";
  return e.shiftKey ? "side_chat" : "attach";
}

export const HIT_ACTION_HINT = `Enter opens, ${keyLabel("Mod+Enter")} attaches to the turn, ${keyLabel("Mod+Shift+Enter")} starts a side chat`;

/** Resolve a hit back to its source and act on it, through the ONE command spine.
 *  Throws (never silently no-ops) so the caller can show the refusal. */
export async function runHitAction(action: HitAction, hit: SearchHit, ctx: SearchCtx): Promise<IntentAck> {
  const sessionId = hit.session_id ?? ctx.sessionId;
  if (action === "open") {
    if (hit.path) return runCommand("open_file", { path: hit.path, line: hit.line ?? null });
    // A transcript hit opens its SESSION (the host republishes that session's recorded transcript).
    // It used to fire `scrub_to_event`, which the host recorded and no arm acted on, so the row
    // reported success and nothing moved. Opening the session is the real host capability; it lands
    // on the session, not on the exact event, and the hit's provenance line still names the event.
    if (hit.session_id) return runCommand("open_session", { session_id: hit.session_id });
    throw new Error("This result carries no source to open");
  }
  if (action === "side_chat") {
    // A transcript hit forks the side chat AT its event; a file hit has no event to fork at.
    return runCommand("create_side_chat", {
      session_id: sessionId,
      inherit: true,
      ...(hit.event_id ? { at_event: hit.event_id } : {}),
    });
  }
  const text = `Referring to ${citeHit(hit)}`;
  // Attach means "put this in front of the model now": a live run is steered (the same real
  // InterruptHub route the composer uses), and with no run in flight it opens the turn.
  if (ctx.runId)
    return runCommand("steer", { run_id: ctx.runId, session_id: ctx.sessionId, text });
  return runCommand("submit_turn", { session_id: ctx.sessionId, text });
}

/** Arrow-key movement over a flat result list. Clamped, so focus never falls off either end. */
export function nextIndex(len: number, i: number, key: string): number {
  if (len === 0) return 0;
  if (key === "ArrowDown") return Math.min(i + 1, len - 1);
  if (key === "ArrowUp") return Math.max(i - 1, 0);
  if (key === "Home") return 0;
  if (key === "End") return len - 1;
  return i;
}

/** One result row, shared by both entry points so a hit reads identically wherever it appears. */
export function HitRow({
  hit,
  selected,
  id,
  onActivate,
  onHover,
}: {
  hit: SearchHit;
  selected: boolean;
  id?: string;
  onActivate: (action: HitAction, hit: SearchHit) => void;
  onHover?: () => void;
}) {
  return (
    <button
      id={id}
      role="option"
      className="ghost-button search-hit"
      aria-selected={selected}
      aria-label={hitLabel(hit)}
      title={`${hitProvenance(hit)} . ${scopeSpec(hit.scope).label} . ${HIT_ACTION_HINT}`}
      onMouseEnter={onHover}
      onClick={(e) => onActivate(hitActionFor(e), hit)}
    >
      <span className="search-hit__path">
        {hit.title}
        <span className="search-hit__line"> {hitProvenance(hit)}</span>
      </span>
      {hit.preview ? <span className="search-hit__preview t-code">{hit.preview}</span> : null}
    </button>
  );
}

/*
  Quick Open / command palette (Cmd+P), and THE global entry to the one search experience.

  The command list is still DERIVED from the catalog by the caller (App.tsx passes
  SHELL_COMMANDS + paletteCommands()); this component adds no second command list. Typing searches
  the HIDE object model in the same box: commands that match stay on top, hits from the current
  scopes follow, and one Enter activates whichever row is selected.

  RETIRED: the catalog's own `run_search` palette row ("Search"). It is bound to the very search this
  box runs, but from a bare palette gesture it carries no query, so choosing it fired an empty search
  that returned the head of the log. The input is that command now. That retirement lives in the
  derivation (store.REQUIRED_ARGS lists `query`), not in a special case here, so every other row that
  cannot carry its payload is filtered by the same rule.
*/
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
  const [scopes, setScopes] = useState<ScopeId[]>(() => defaultScopes(searchOrigin()));
  const [outcome, setOutcome] = useState<SearchOutcome>({ hits: [], errors: [] });
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const trapRef = useFocusTrap<HTMLDivElement>(open);
  const sessionId = useStore((s) => s.sessionId);
  const runId = useStore((s) => s.activeRunId);

  const query = q.trim();
  const filtered = useMemo(() => {
    const t = query.toLowerCase();
    return t ? commands.filter((c) => c.label.toLowerCase().includes(t)) : commands;
  }, [query, commands]);

  useEffect(() => {
    if (!open) return;
    setQ("");
    setSel(0);
    setStatus(null);
    setOutcome({ hits: [], errors: [] });
    // The default scope follows the surface the search was opened from (Explorer sets "editor").
    setScopes(defaultScopes(searchOrigin()));
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  // Live search, debounced. Cleared (never stale) as soon as the query or the scopes change.
  useEffect(() => {
    if (!open || query.length < 2 || scopes.length === 0) {
      setOutcome({ hits: [], errors: [] });
      setPending(false);
      return;
    }
    let live = true;
    setPending(true);
    const t = setTimeout(async () => {
      const r = await searchAll(query, scopes, { sessionId, runId: runId ?? "" });
      if (!live) return;
      setOutcome(r);
      setPending(false);
    }, 160);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [open, query, scopes, sessionId, runId]);

  const hits = outcome.hits;
  const rowCount = filtered.length + hits.length;

  const activate = useCallback(
    (index: number, action: HitAction) => {
      const cmd = filtered[index];
      if (cmd) {
        cmd.run();
        onClose();
        return;
      }
      const hit = hits[index - filtered.length];
      if (!hit) return;
      void runHitAction(action, hit, { sessionId, runId: runId ?? "" })
        .then((ack) => {
          if (action === "open") {
            onClose();
            return;
          }
          // Attach and side chat keep the palette (and the caret) exactly where they were, so a
          // second result can follow the first.
          setStatus(
            ack.accepted
              ? action === "attach"
                ? runId
                  ? "Attached to the running turn"
                  : "Sent as a new turn"
                : "Side chat started from this result"
              : ack.message ?? "The host refused that action",
          );
        })
        .catch((e) => setStatus(e instanceof Error ? e.message : String(e)));
    },
    [filtered, hits, onClose, runId, sessionId],
  );

  if (!open) return null;

  const toggleScope = (id: ScopeId) => {
    setSel(0);
    setScopes((cur) => (cur.includes(id) ? cur.filter((s) => s !== id) : [...cur, id]));
  };

  const state = pending
    ? `Searching ${scopes.map((s) => scopeSpec(s).label).join(", ")}`
    : query.length >= 2
      ? `${hits.length} result${hits.length === 1 ? "" : "s"}`
      : null;

  return (
    <div role="presentation" className="palette-overlay" onClick={onClose}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette and search" ref={trapRef} onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          value={q}
          role="combobox"
          aria-expanded
          aria-controls="palette-rows"
          aria-activedescendant={rowCount ? `palette-row-${Math.min(sel, rowCount - 1)}` : undefined}
          onChange={(e) => {
            setQ(e.target.value);
            setSel(0);
            setStatus(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") return onClose();
            if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Home" || e.key === "End") {
              e.preventDefault();
              return setSel((i) => nextIndex(rowCount, i, e.key));
            }
            if (e.key === "Enter") {
              e.preventDefault();
              activate(sel, hitActionFor(e));
            }
          }}
          placeholder="Type a command or search"
          aria-label="Type a command or search"
          className="t-body palette__input"
        />

        {query ? (
          <div
            role="group"
            aria-label="Search scopes"
            style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-1)", padding: "0 var(--ma-2) var(--ma-2)" }}
          >
            {SCOPES.map((s) => {
              const on = scopes.includes(s.id);
              return (
                <button
                  key={s.id}
                  className="ghost-button t-micro"
                  aria-pressed={on}
                  title={s.hint}
                  onClick={() => toggleScope(s.id)}
                  style={{
                    padding: "0 var(--ma-2)",
                    borderRadius: "var(--radius-sm)",
                    background: on ? "var(--list-active)" : "transparent",
                    color: on ? "var(--text-strong)" : "var(--text-dim)",
                  }}
                >
                  {on ? "on " : "off "}
                  {s.label}
                </button>
              );
            })}
          </div>
        ) : null}

        <ul className="palette__list" id="palette-rows" role="listbox" aria-label="Commands and results">
          {rowCount === 0 ? (
            <li className="t-body" style={{ padding: "var(--ma-3)", color: "var(--text-dim)" }}>
              {pending ? "Searching" : query ? "No commands or results" : "No commands"}
            </li>
          ) : (
            <>
              {filtered.map((c, i) => (
                <li key={c.id} role="presentation">
                  <button
                    id={`palette-row-${i}`}
                    role="option"
                    className="ghost-button palette__item t-body"
                    aria-selected={i === sel}
                    onMouseEnter={() => setSel(i)}
                    onClick={() => activate(i, "open")}
                    style={{ display: "flex", alignItems: "center", gap: "var(--ma-4)" }}
                  >
                    <span style={{ flex: 1, minWidth: 0, textAlign: "left" }}>{c.label}</span>
                    {/* The palette is the one place that aggregates every command, so it is the one
                        place the binding must be visible. */}
                    {c.shortcut ? (
                      <span style={{ color: "var(--text-2)", fontSize: "var(--fs-small)" }}>{keyLabel(c.shortcut)}</span>
                    ) : null}
                  </button>
                </li>
              ))}
              {hits.map((h, i) => {
                const index = filtered.length + i;
                return (
                  <li key={h.key} role="presentation">
                    <HitRow
                      id={`palette-row-${index}`}
                      hit={h}
                      selected={index === sel}
                      onHover={() => setSel(index)}
                      onActivate={(action) => activate(index, action)}
                    />
                  </li>
                );
              })}
            </>
          )}
        </ul>

        <div
          role="status"
          aria-live="polite"
          className="t-micro"
          style={{ padding: "0 var(--ma-3) var(--ma-2)", color: "var(--text-dim)" }}
        >
          {[status, ...outcome.errors, state, query ? HIT_ACTION_HINT : null].filter(Boolean).join(" . ")}
        </div>
      </div>
    </div>
  );
}
