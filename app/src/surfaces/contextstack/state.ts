/*
  state.ts: the Context Stack's local state plus the READ (receipt) and WRITE (command) logic the
  panel renders. Kept out of the .tsx so it is testable in the node vitest environment.

  Three parts:
  1. useSteer: the optimistic overlay for the one steer gesture that survives (evict a memory).
     The host owns truth; the overlay reconciles away when a fresh manifest arrives.
  2. The receipt selectors: pure reads over the REAL manifest the host serializes
     (hawking-context ContextManifest: retained / dropped / conflicts / budget / kv), with the
     older flat shape (retrieved / memory / tools) still honored so nothing regresses.
  3. The command plans: every backend action this panel can take, named by its catalog command id
     (src/generated/command_catalog.json). A plan is data, so a test can assert exactly which
     command a control means without a transport; runPlan is the one dispatch through the spine.
*/
import { useCallback, useState } from "react";
import { runCommand, type CommandArgs, type ContextManifest } from "../../store";
import { heldNote } from "../../wire";

export type SteerKind = "pin" | "mute" | "evict";

// A span's stable identity across manifests: source-kind + its natural key.
export function spanKey(kind: string, key: string): string {
  return `${kind}:${key}`;
}

export interface Steer {
  // is this span currently steered (locally), for the given kind?
  on(id: string, kind: SteerKind): boolean;
  // flip a steer and return the new value (caller emits the matching command).
  toggle(id: string, kind: SteerKind): boolean;
  // a free-text note injected against a span (or the turn at large when id === "turn").
  noteOn(id: string): string | undefined;
  setNote(id: string, text: string): void;
}

export function useSteer(): Steer {
  // two flat maps keep this dense: one boolean overlay, one note overlay.
  const [flags, setFlags] = useState<Record<string, boolean>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});

  const on = useCallback((id: string, kind: SteerKind) => flags[`${kind}|${id}`] === true, [flags]);

  const toggle = useCallback((id: string, kind: SteerKind) => {
    const k = `${kind}|${id}`;
    let next = false;
    setFlags((f) => {
      next = !f[k];
      return { ...f, [k]: next };
    });
    return next;
  }, []);

  const noteOn = useCallback((id: string) => notes[id], [notes]);
  const setNote = useCallback((id: string, text: string) => {
    setNotes((n) => {
      if (!text) {
        const { [id]: _drop, ...rest } = n;
        return rest;
      }
      return { ...n, [id]: text };
    });
  }, []);

  return { on, toggle, noteOn, setNote };
}

/* ---- 2. The receipt --------------------------------------------------------------------------
   The host serializes hawking_context::ContextManifest (crates/hawking-context/src/manifest.rs)
   over the context connector, and a flatter live patch over projection_patch{context_manifest}.
   Receipt is the superset the panel reads, so both arrive at the same rows and NOTHING here is
   invented: every field below exists in one of those two payloads.
*/

/** ContextSourceKind: unit variants serialize snake_case; Custom(String) as { custom: "..." }. */
export type SourceKind = string | { custom: string };

/** hide_core::types::Provenance. `labels` carries "instructions" / "claude-md-compat". */
export interface SpanProvenance {
  source?: string;
  trust?: string;
  confidence?: number;
  labels?: string[];
  derived_from?: string[];
}

/** A retained (included) span, in final window order. `id` is a blake3 content address. */
export interface RetainedSpan {
  id?: string;
  source?: SourceKind;
  title?: string;
  order_index?: number;
  token_count?: number;
  score?: number;
  signals?: { recency?: number; importance?: number; relevance?: number; redundancy?: number };
  pin?: string;
  banked?: boolean;
  provenance?: SpanProvenance;
  blob_ref?: { id?: string; hash?: string; size_bytes?: number; media_type?: string | null } | null;
  compacted_from?: { original_id?: string; method?: string; ratio?: number } | null;
}

/** A candidate that did not make it in. The older flat shape used title/would_be_tokens. */
export interface DroppedSpan {
  id?: string;
  source?: SourceKind;
  token_count?: number;
  score?: number;
  reason?: string;
  title?: string;
  would_be_tokens?: number;
}

export interface ManifestConflict {
  between?: string[];
  note?: string;
  resolved?: boolean;
}

export interface MemoryEntry {
  memory_id?: string;
  id?: string;
  claim?: string;
  fact?: string;
  confidence?: number;
  outcome_score?: number;
  status?: string;
  citations?: string[];
}

export interface Receipt {
  model?: {
    id?: string;
    arch?: string;
    ctx?: number;
    ctx_len_native?: number;
    ctx_len_effective?: number;
    tokenizer_sig?: string;
    profile?: string;
    sampling?: string;
  };
  retained?: RetainedSpan[];
  retrieved?: { path?: string; range?: string; relevance?: number }[];
  dropped?: DroppedSpan[];
  conflicts?: ManifestConflict[];
  memory?: MemoryEntry[];
  tools?: { name?: string; ok?: boolean }[];
  budget?: { total?: number; used?: number; free?: number };
  kv?: { prefix_reuse_tokens?: number; bank_hit?: boolean };
  used_tokens?: number;
  model_context_tokens?: number;
  // The live patch reports the model flat (host publish_turn_context_manifest), the compiled
  // manifest reports it as a block. Both are read.
  model_id?: string;
  arch?: string;
  ctx_len_native?: number;
  ctx_len_effective?: number;
  tq_multiplier?: number;
  recurrent_state_bytes?: number;
  live?: ContextManifest["live"];
}

/** Widen the store's manifest to the receipt shape. Both come off the same host payload. */
export const asReceipt = (m: ContextManifest | null | undefined): Receipt => (m ?? {}) as Receipt;

export interface ModelSummary {
  id: string;
  arch?: string;
  native?: number;
  effective?: number;
  tokenizerSig?: string;
  profile?: string;
}

/** The model the manifest was compiled against, from whichever of the two shapes arrived. */
export function modelSummary(r: Receipt): ModelSummary | undefined {
  const id = r.model?.id ?? r.model_id;
  if (!id) return undefined;
  return {
    id,
    arch: r.model?.arch ?? r.arch,
    native: r.model?.ctx_len_native ?? r.ctx_len_native,
    effective: r.model?.ctx_len_effective ?? r.ctx_len_effective,
    tokenizerSig: r.model?.tokenizer_sig,
    profile: r.model?.profile,
  };
}

export const sourceKindLabel = (k?: SourceKind): string =>
  typeof k === "string" ? k.replace(/_/g, " ") : k?.custom ?? "source";

const hasLabel = (s: RetainedSpan, label: string) => (s.provenance?.labels ?? []).includes(label);

/** A source that made it into the window, with the reason it did and what it cost. */
export interface IncludedRow {
  key: string;
  title: string;
  kind: string;
  /** Why this span is in the window: its origin, pin state, and the packer's score. */
  why: string;
  tokens: number;
  /** blake3 content address of the span (the receipt's integrity handle). */
  hash?: string;
  /** Files this span was derived from (for instruction spans, the loaded CLAUDE.md style files). */
  files: string[];
  /** True for a repo-instruction span, and separately for the Claude Code compat set. */
  instructions: boolean;
  compat: boolean;
  /** A real attachment rode along with this span (blob ref present). */
  attachment?: string;
  /** KV was reused rather than re-prefilled. */
  banked: boolean;
  /** A file path the row can open, when the span names one. */
  path?: string;
}

const looksLikePath = (s?: string) => !!s && /[/\\.]/.test(s) && !s.includes(" ");

export function includedSources(r: Receipt): IncludedRow[] {
  const rows: IncludedRow[] = (r.retained ?? []).map((s, i) => {
    const files = s.provenance?.derived_from ?? [];
    const why = [
      s.provenance?.source,
      s.pin === "user_pinned" ? "pinned by you" : s.pin === "never_evict" ? "never evicted" : null,
      typeof s.score === "number" ? `score ${s.score.toFixed(2)}` : null,
      s.compacted_from?.method ? `compacted (${s.compacted_from.method})` : null,
    ]
      .filter(Boolean)
      .join(" / ");
    return {
      key: s.id ?? `retained:${i}`,
      title: s.title ?? s.id ?? "span",
      kind: sourceKindLabel(s.source),
      why: why || "included by the packer",
      tokens: s.token_count ?? 0,
      hash: s.id,
      files,
      instructions: hasLabel(s, "instructions"),
      compat: hasLabel(s, "claude-md-compat"),
      attachment: s.blob_ref?.id ?? undefined,
      banked: s.banked === true,
      path: looksLikePath(files[0]) ? files[0] : looksLikePath(s.title) ? s.title : undefined,
    };
  });
  // The flatter retrieval shape (host retrieval patch): a path, a range, a relevance.
  for (const [i, x] of (r.retrieved ?? []).entries()) {
    const path = x.path ?? "";
    rows.push({
      key: `retrieved:${path}:${x.range ?? i}`,
      title: x.range ? `${path}:${x.range}` : path,
      kind: "code",
      why: typeof x.relevance === "number" ? `retrieval / relevance ${x.relevance.toFixed(2)}` : "retrieval",
      tokens: 0,
      files: path ? [path] : [],
      instructions: false,
      compat: false,
      banked: false,
      path: path || undefined,
    });
  }
  return rows;
}

export interface ExcludedRow {
  key: string;
  title: string;
  kind: string;
  reason: string;
  tokens: number;
  stale: boolean;
}

const REASON_WORD: Record<string, string> = {
  budget: "over budget",
  no_fit: "did not fit",
  duplicate: "duplicate",
  redundant: "redundant",
  stale: "stale",
  low_score: "low score",
  low_value: "low value",
  unsafe: "unsafe",
  source_unavailable: "source unavailable",
};

export function excludedSources(r: Receipt): ExcludedRow[] {
  return (r.dropped ?? []).map((d, i) => {
    const reason = String(d.reason ?? "");
    return {
      key: d.id ?? d.title ?? `dropped:${i}`,
      title: d.title ?? d.id ?? "span",
      kind: sourceKindLabel(d.source),
      reason: REASON_WORD[reason] ?? reason ?? "not included",
      tokens: d.token_count ?? d.would_be_tokens ?? 0,
      stale: reason === "stale" || reason === "source_unavailable",
    };
  });
}

export interface WarningRow {
  key: string;
  text: string;
  kind: "stale" | "conflict";
}

/** Everything the receipt says the user should distrust: stale exclusions and open conflicts. */
export function staleWarnings(r: Receipt): WarningRow[] {
  const out: WarningRow[] = excludedSources(r)
    .filter((x) => x.stale)
    .map((x) => ({ key: `stale:${x.key}`, text: `${x.title} is ${x.reason}`, kind: "stale" as const }));
  for (const [i, c] of (r.conflicts ?? []).entries()) {
    if (c.resolved) continue;
    out.push({
      key: `conflict:${i}`,
      text: c.note ?? (c.between ?? []).join(" vs "),
      kind: "conflict",
    });
  }
  return out;
}

export interface MemoryRow {
  key: string;
  /** The durable record id, when the host gave us one. Without it no outcome can be recorded. */
  id?: string;
  claim: string;
  /** Governed score if present, else the initial confidence. */
  score?: number;
  status?: string;
  citations: string[];
}

/** The memory the compiler drew from: the memory projection first, then Memory spans in the window. */
export function memoryRows(r: Receipt): MemoryRow[] {
  const rows: MemoryRow[] = (r.memory ?? []).map((m, i) => ({
    key: m.memory_id ?? m.id ?? m.claim ?? m.fact ?? `memory:${i}`,
    id: m.memory_id ?? m.id,
    claim: m.claim ?? m.fact ?? "",
    score: m.outcome_score ?? m.confidence,
    status: m.status,
    citations: m.citations ?? [],
  }));
  const seen = new Set(rows.map((x) => x.claim));
  for (const s of r.retained ?? []) {
    if (sourceKindLabel(s.source) !== "memory") continue;
    const claim = s.title ?? "";
    if (seen.has(claim)) continue;
    seen.add(claim);
    rows.push({
      key: s.id ?? `memory-span:${claim}`,
      id: undefined, // a window span carries its content hash, not a durable memory id
      claim,
      score: s.score,
      status: undefined,
      citations: s.provenance?.derived_from ?? [],
    });
  }
  return rows.filter((x) => x.claim);
}

export interface ReceiptTotals {
  usedTokens?: number;
  ceilingTokens?: number;
  stateBytes?: number;
  includedTokens: number;
  excludedTokens: number;
  reusedTokens?: number;
  /** Repo instruction files the backend loaded for this turn. */
  instructionFiles: string[];
  /** The Claude Code compatible subset of them (CLAUDE.md style config). */
  compatFiles: string[];
  attachments: number;
}

export function receiptTotals(r: Receipt): ReceiptTotals {
  const included = includedSources(r);
  const uniq = (xs: string[]) => [...new Set(xs)];
  return {
    usedTokens: r.used_tokens ?? r.budget?.used ?? r.live?.used_tokens_estimate,
    ceilingTokens: r.live?.effective_ceiling_tokens ?? r.ctx_len_effective ?? r.model_context_tokens,
    stateBytes: r.recurrent_state_bytes,
    includedTokens: included.reduce((n, x) => n + x.tokens, 0),
    excludedTokens: excludedSources(r).reduce((n, x) => n + x.tokens, 0),
    reusedTokens: r.kv?.prefix_reuse_tokens,
    instructionFiles: uniq(included.filter((x) => x.instructions).flatMap((x) => x.files)),
    compatFiles: uniq(included.filter((x) => x.compat).flatMap((x) => x.files)),
    attachments: included.filter((x) => x.attachment).length,
  };
}

/* ---- 3. Command plans -----------------------------------------------------------------------
   Each plan names a catalog command id, so the palette, a shortcut and this panel's control all
   resolve the SAME CommandSpec. Nothing here builds an Intent: runPlan hands the id to the one
   dispatch point in store.ts, which refuses honestly when a binding is not reachable yet.
*/

export interface CommandPlan {
  id: string;
  args: CommandArgs;
}

export type MemoryScope = { kind: "session" | "repo" | "user"; id: string };

export const sessionScope = (sessionId: string): MemoryScope => ({ kind: "session", id: sessionId });

/** A MemoryDraft exactly as hide-backend parse_memory_draft reads it (scope, claim, source, author). */
export const memoryDraft = (scope: MemoryScope, claim: string, citations: string[] = []) => ({
  scope,
  claim,
  source: "context_stack",
  author: "user",
  confidence: 1,
  citations,
});

export const plan = {
  /** Seal an integrity-verified restore point (blake3) instead of a local "saved" toast. */
  snapshot: (session_id: string, label: string): CommandPlan => ({
    id: "checkpoint_create",
    args: { session_id, label },
  }),
  /** Branch a new session whose history is this one folded up to `at_event`. */
  fork: (session_id: string, at_event: string): CommandPlan => ({
    id: "fork_session",
    args: { session_id, at_event },
  }),
  /** Store a durable, outcome-governed note. */
  remember: (scope: MemoryScope, claim: string, citations: string[] = []): CommandPlan => ({
    id: "memory_add",
    args: memoryDraft(scope, claim, citations),
  }),
  /** Replace a stale claim while keeping its history. */
  supersede: (
    old_id: string,
    scope: MemoryScope,
    claim: string,
    citations: string[] = [],
  ): CommandPlan => ({
    id: "memory_supersede",
    args: { old_id, replacement: memoryDraft(scope, claim, citations) },
  }),
  /** Report that a remembered claim held or failed, so a bad one self-quarantines. */
  outcome: (memory_id: string, success: boolean): CommandPlan => ({
    id: "memory_record_outcome",
    args: { memory_id, success },
  }),
  /** Re-check a scope's citations against the repo on disk. */
  revalidate: (scope: MemoryScope): CommandPlan => ({ id: "memory_revalidate", args: { scope } }),
  /** Open a cited file in the editor. */
  open: (path: string): CommandPlan => ({ id: "open_file", args: { path } }),
};

/**
 * Committing the note field: with a memory marked wrong it REPLACES that record (history kept),
 * otherwise it adds a new durable claim. One field, two honest writes, no extra control.
 */
export function notePlan(
  text: string,
  scope: MemoryScope,
  supersedeId?: string,
  citations: string[] = [],
): CommandPlan {
  return supersedeId
    ? plan.supersede(supersedeId, scope, text, citations)
    : plan.remember(scope, text, citations);
}

/** The ONE dispatch: a plan resolves its catalog command through the shared spine. */
export const runPlan = (p: CommandPlan) => runCommand(p.id, p.args);

/* ---- Action feedback -------------------------------------------------------------------------
   A control must show pending, done and failed, and must never claim a success the host did not
   give. Keyed by control so two rows never share one spinner; the key stays stable across the
   action so focus does not move.
*/
export type ActionState = "idle" | "pending" | "done" | "failed";

export interface Actions {
  stateOf(key: string): ActionState;
  messageOf(key: string): string | undefined;
  run(key: string, p: CommandPlan): Promise<boolean>;
}

export function useActions(onFail?: (message: string) => void): Actions {
  const [map, setMap] = useState<Record<string, { state: ActionState; message?: string }>>({});
  const mark = useCallback((key: string, state: ActionState, message?: string) => {
    setMap((m) => ({ ...m, [key]: { state, message } }));
  }, []);

  const run = useCallback(
    async (key: string, p: CommandPlan) => {
      mark(key, "pending");
      try {
        const ack = await runPlan(p);
        if (ack && ack.accepted === false) {
          const why = ack.message ?? `${p.id} was refused`;
          mark(key, "failed", why);
          onFail?.(why);
          return false;
        }
        // HELD at an approval gate: recorded, not run. Still pending (not a failure, so no error
        // notice), and false so no caller flips its own state as though the effect had landed.
        if (ack?.held) {
          mark(key, "pending", heldNote(p.id));
          return false;
        }
        mark(key, "done", ack?.message ?? undefined);
        return true;
      } catch (e) {
        const why = e instanceof Error ? e.message : String(e);
        mark(key, "failed", why);
        onFail?.(why);
        return false;
      }
    },
    [mark, onFail],
  );

  return {
    stateOf: (key) => map[key]?.state ?? "idle",
    messageOf: (key) => map[key]?.message,
    run,
  };
}
