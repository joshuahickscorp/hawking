# HIDE Context Operating System

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.1, §3.2, §3.5, §5), `hawking_ide_frontier_2026_07_19.md` (§3, §5.1, §5.12), Bible Book VII §43-44, §48.
Status: specification for a Hawking-native subsystem. Every load-bearing claim is tagged by evidence class (VERIFIED REPO / DOCUMENTED / MEASURED / INFERRED) and every mechanism by readiness (real-and-wired / real-but-unwired / partial / stub / missing).
Packed-crate file:line citations are pinned at the sealed commit `5a99d0e2` unless noted.
Siblings: `HIDE_MEMORY_SPEC.md` (durable memory records), `HIDE_STATE_CAPSULE_ABI.md` (reusable compute state), `HIDE_TWO_SURFACE_ARCHITECTURE.md` (the shared core this plane sits in), `HIDE_SPEED_FRONTIER.md` (prompt ABI, prefix reuse).

## 1. Why this is an operating system, not a prompt builder

The Context OS owns the scarce resource of every turn: the model's active attention. It decides what evidence occupies the window, at what fidelity, in what order, and how the leftover is checkpointed. Two flagship crates already implement most of it, and both are real-but-unwired.

| Flagship | Readiness | What it is | Evidence |
|---|---|---|---|
| `hawking-context` (~4.9k LOC) | real-but-unwired (REINTEGRATE) | reserve-then-fill compiler: carve system/response/scratchpad first, value-density knapsack, degrade ladder, head/tail anti-lost-in-the-middle order, replayable content-addressed manifest, SQLite FTS5 + cosine memory | archaeology §3.5, §5 |
| `hawking-index` (~4.7k LOC) | real-but-unwired (REINTEGRATE) | tree-sitter defs+refs, cAST chunking, BLAKE3 merkle change-gate, FTS5 + graph store, PageRank repo-map, hybrid RRF retriever, incremental MVCC daemon with crash recovery | archaeology §3.5, §5 |

The load-bearing defect is one sentence, code-verified: **the reserve-then-fill compiler runs only behind the `context` connector, and its compiled prompt is discarded relative to generation** (facade **S3**, archaeology §3.5). The live turn instead sends the raw prompt with empty history at `max_output_tokens=256` (facade **S2**, `host.rs:848-863` @5a99d0e2). So HIDE today has a token-true context compiler that nothing feeds to the model. This spec's first mandate (§11) is to close that wire; everything else specifies the object that flows through it.

This is a reconnection, not an invention (archaeology §6). The Context OS is `hawking-index` + `hawking-context` lifted out of `5a99d0e2`, wired to the flat kernel loop (`HIDE_AGENT_KERNEL_OPTIONS`), with the compiled `CompiledContext.prompt` actually handed to `hawking-serve`.

## 2. The honest "unbounded context" model (four layers, one truth format)

HIDE promises continuity, provenance, and reconstructability, not infinite active attention or lossless semantic compression (dossier §3.1, INFERENCE). Nominal context length does not guarantee reliable retrieval (NoLiMa, RULER; dossier §5.1, DOCUMENTED), and exploring more context is not using it (ContextBench; dossier §5.1). The system is therefore four cooperating layers, only one of which is the model window.

| Layer | Holds | Bound | HIDE substrate |
|---|---|---|---|
| L1 Durable project universe | repo snapshots, events, decisions, memory, artifacts, test results, traces, tool outputs | storage + retention policy | event log + `hawking-index` store + `HIDE_MEMORY_SPEC` |
| L2 Active model context | the exact evidence + instructions in the current inference call | selected model window | the ContextPack (§5), compiled by `hawking-context` |
| L3 Reusable compute state | transformer KV prefixes, recurrent state, or an execution-state capsule | memory + model identity + compatibility | `HIDE_STATE_CAPSULE_ABI` (RWKV lossless, transformer capsule missing) |
| L4 Working task state | plan, acceptance criteria, current patch, failures, hypotheses, next actions | compact, structured, reconstructable | the checkpoint (§8) |

**Weight-compression is not a context multiplier (G11, VERIFIED REPO).** The serve `/context` route reports `effective = native × tq_multiplier`, but real KV capacity is fixed at 4096 (`max_seq_len` hardcoded, `lib.rs:509`), and `HAWKING_QWEN_TQ_MULTIPLIER` (read at `http.rs:250`) is **never set anywhere in-repo (G9, VERIFIED REPO)**: the multiplier is an unset-env estimate, not a measured capacity. The compiler carries the same knob (`compiler.rs:320` `with_tq_multiplier` @5a99d0e2) and records `ctx_len_effective` on every manifest (`manifest.rs:200-206` @5a99d0e2). **Rule:** `ctx_len_effective` MUST equal `ctx_len_native` until a measured long-context KV result exists (`HIDE_SPEED_FRONTIER`); a `.tq` weight saving grows L1/L3 headroom (more resident model, more room for KV), never L2 attention span. Any product surface that prints an effective ceiling above native is a false claim.

Local and cloud share this format and differ only in policy (dossier §3.2): if a provider disappears, HIDE reconstructs L2 from L1; a cloud model receives the smallest necessary scoped context, never ownership of project history.

## 3. Context strata (the fillable pool, head to tail)

Every candidate that competes for L2 is one of ten strata (`ContextSourceKind`, `manifest.rs:296-307` @5a99d0e2), carried head-first to tail-last to defeat lost-in-the-middle (`compiler.rs:508-509`). Each stratum declares eight attributes so the allocator (§9) can rank it; the manifest records the realized values per span.

| Stratum (`ContextSourceKind`) | Identity | Source | Scope | Token-cost | State-cost (L3) | Value-estimate | Expiry / invalidation | Confidence |
|---|---|---|---|---|---|---|---|---|
| Pinned instructions (`System`) | blake3 span id | resolved CLAUDE.md tree + safety rails + tool namespace | user > project > local, per trust-domain | tokenizer-true | high (cache once as warm prefix) | fixed max (`PinState::NeverEvict`) | config edit / precedence change | authored |
| Task contract (`Plan` / `UserTurn`) | blake3 span id | user turn + acceptance criteria | session | tokenizer-true | low | high (drives the loop) | task redefinition | authored |
| Repo map (`Symbol` via index) | snapshot id + node ids | `hawking-index` PageRank repo-map | repo snapshot | budgeted (~1K default, §7) | none (recompiled) | high per token (deterministic baseline) | merkle root change | measured (graph rank) |
| Ranked source spans (`Code` / `Symbol`) | blake3 + file/commit/symbol | hybrid RRF retriever | repo snapshot | tokenizer-true | reusable if banked | relevance x importance | file edit / snapshot diverge | retrieval provenance |
| Current diff / transaction (`Code`) | patch txn id | working-tree patch transaction | session | tokenizer-true | none | high (authorizes edits) | patch apply / revert | exact bytes |
| Runtime evidence (`Diagnostics` / `ToolOutput`) | blake3 span id | test failures, logs, traces, LSP | session | tokenizer-true, maskable | none | high while unresolved | rerun / resolution | measured (execution) |
| Durable decisions (`Memory`) | record id + citations | `HIDE_MEMORY_SPEC` store | repo/user/org/trust-domain | tokenizer-true | foldable into capsule | importance x confidence | citation revalidation fails (stale -> audit-only) | provenance + confidence |
| Recent action window (`ToolOutput` / `UserTurn`) | blake3 span id | last N loop actions | session | tokenizer-true, maskable | none | recency-decayed | window slides | observed |
| Scratchpad (`Scratchpad`) | reservation, not a span | reserved output-adjacent working room | turn | reserved (§4) | none | reserved | turn end | n/a |
| Reusable compute state (L3 handle) | capsule id + `IdentityBinding` | `HIDE_STATE_CAPSULE_ABI` | session, identity-bound | zero L2 tokens (it IS state) | bytes of KV/recurrent | very high if warm | identity mismatch / boundary (`HIDE_STATE_CAPSULE_ABI` §9) | measured (byte-exact on RWKV) |

Notes on the honest columns:
- **Token-cost is tokenizer-true, never chars/4-by-default.** `TokenCounter::from_file` loads a real HuggingFace tokenizer (`budget.rs:82` @5a99d0e2); the `chars/4` heuristic is a labeled fallback only (`budget.rs:76`). This supersedes the historical character-count budgeting (archaeology §4, PARTLY SUPERSEDED).
- **State-cost is a distinct axis from token-cost.** A span that costs 800 tokens in L2 may cost zero L3 bytes (recomputed each turn) or may be banked as reused KV (`ContextSpan.banked`, `manifest.rs:271-275`). The allocator trades both.
- **Value-estimate is not directly observed** (§9). It is a blended prior from four signals `recency / importance / relevance / redundancy` (`SpanSignals`, `manifest.rs:234-240`), calibrated by experiment.
- **Expiry is a first-class invalidation condition** (dossier §3.3 item "freshness or invalidation condition"), not a TTL guess. A repo-map span expires when the merkle root changes (§7); a memory span expires when its citation fails revalidation and drops to audit-only (`HIDE_MEMORY_SPEC`, DOCUMENTED: Copilot Memory revalidates citations against the current branch).

## 4. Reserve-then-fill: the compile pipeline

The compiler carves fixed regions **before anything competes**, then greedily fills the remainder by value density, then orders the survivors head/tail. This is implemented and tested; the spec's job is to wire it and hold the invariants.

Pipeline (`compiler.rs:329-575` @5a99d0e2):

0. **Reserve** system / response / scratchpad out of the total before competition (`:332-345`). Default reservations: `system 0.08`, `response 0.20`, `scratchpad 0.06` of the window (`profiles.rs:189-193`), profile-selectable. Response + scratchpad are subtracted from the competition pool; system spans are real content and stay in the fillable pool but pinned.
1. **Gather candidates** from each registered `ContextSource`; lazy sources may declare `est_tokens` without materializing a body (`compiler.rs:127-158`, progressive disclosure).
2. **Admit pins first**, then run **value-density greedy** over the rest, re-ranked by `density = value / tokens` (`:422-491`).
3. **Degrade ladder** on over-budget spans: tool chatter is masked (head kept, body elided with a token-count placeholder, reasoning trace preserved, `:216-256`); everything else truncates on a whitespace boundary (`:173-214`). Compaction refuses past depth 2 and falls back to the original to stop compounding loss (`manifest.rs:243-252` `CompactedFrom.depth`).
4. **Backfill** a smaller candidate that now fits into leftover budget (`:491-508`).
5. **Order head/tail** to defeat lost-in-the-middle (`:508-509`, `order_head_tail`).

Hard invariant (asserted in code, `compiler.rs:571-572`): `used + reservation_response <= max_input_tokens`. The window plus the output reservation never exceeds the total. This spec adds one enforcement mandate: the compiler's `CompiledContext.prompt` is the ONLY string handed to generation (§11); no code path may reconstruct a raw prompt that bypasses this budget.

## 5. The token-true ContextPack contract

Each turn builds a versioned ContextPack whose ordered regions are (dossier §3.3):

1. stable policy and repository invariants;
2. stable, small tool namespace manifest (part of the prompt ABI, `HIDE_SPEED_FRONTIER`);
3. task contract and acceptance criteria;
4. repository map and current snapshot identity;
5. ranked exact source spans (file, symbol, commit, retrieval provenance);
6. current diff and transaction state;
7. test failures, logs, traces, runtime evidence;
8. compact durable decisions and unresolved questions;
9. recent action window;
10. immediate query last.

The realized ContextPack is the `ContextManifest` (`manifest.rs` @5a99d0e2), a **replayable, content-addressed** object: span ids are blake3 content addresses (`span_content_id`, `manifest.rs:282-289`), so two turns that include the same span share an id (dedup + replay key). Every retained span carries the per-item metadata the contract requires (dossier §3.3):

| Contract field | Manifest field | Location |
|---|---|---|
| content identity | `ContextSpan.id` (blake3) | `manifest.rs:255` |
| source and trust domain | `ContextSpan.source` + `provenance` | `manifest.rs:257,277` |
| token count (target tokenizer) | `ContextSpan.token_count` | `manifest.rs:262` |
| inclusion reason and score | `ContextSpan.score` + `SpanSignals` | `manifest.rs:264,266` |
| freshness / invalidation | `provenance` freshness + `banked` reuse flag | `manifest.rs:271-277` |
| explored/presented/cited/edited/test-relevant | `SpanSignals` (recency/importance/relevance/redundancy) | `manifest.rs:234-240` |

Dropped candidates are recorded with a typed `DropReason` (`Budget / NoFit / Duplicate / Redundant / Stale / LowScore / LowValue / Unsafe / SourceUnavailable`, `manifest.rs:317-329`), and contradictions between spans are **surfaced, not silently resolved** (`ManifestConflict`, `manifest.rs:334-341`): the compiler records the conflict for the user rather than picking a winner.

Doctrine, from the frontier finding (dossier §3.3, §5.1): **summaries orient, exact source spans authorize edits.** A model must never mutate code from a lossy summary when the authoritative bytes are local. The July-2026 "compressed exact source can beat summaries" result is promising but not yet a general rule (dossier §5.1); HIDE prefers exact ranked slices over whole-file stuffing and over summaries when acting on code.

The ContextPack is the UI's Context Stack, the loop's replay substrate, and a versioned public contract (`manifest.rs:1-5`). Cross-surface, clicking a context item reveals its source (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §4); that gesture reads `ContextSpan.provenance`.

## 6. Provenance and revalidation

Every span binds provenance (source kind + trust domain + retrieval reason + confidence). Two rules make it load-bearing:

- **Trust domain is a scope, not a label.** A span does not cross security domains (mirrors `HIDE_STATE_CAPSULE_ABI` §4 `security_domain`). Context assembled under one workspace/trust-domain is inadmissible in another.
- **Revalidated memory (DOCUMENTED, dossier §5.12).** Memory records carry evidence citations and a revalidation rule; a fact is checked against the current branch before it enters active context as truth. Corrections supersede rather than erase (`memory.rs` `supersedes`, archaeology §3.5). Stale or unverified facts remain in L1 for audit but do not enter L2 as truth. Full store semantics are in `HIDE_MEMORY_SPEC.md`.

## 7. The living index and the deterministic RepoMap baseline

L1 retrieval is `hawking-index` (real-but-unwired). Its parts, code-verified:

- **Change gate:** a BLAKE3 merkle-DAG over the workspace does an O(changed) diff; the file-system event is a hint, the merkle diff decides the real work, and a moved file is reported as a rename so the symbol graph remaps instead of delete+reinsert (`merkle.rs:1-11` @5a99d0e2).
- **Incremental daemon:** MVCC generations with crash recovery; a torn generation is truncated on restart before applying the diff (`daemon.rs:1-79` @5a99d0e2).
- **Repo-map:** edges load into petgraph, PageRank ranks definitions, and a token-budgeted elided signatures-only tree is rendered for the compiler (`graph.rs:6-7,59` @5a99d0e2).
- **Hybrid retriever:** lexical (FTS5) + symbol + semantic legs fused by RRF (`HybridRetriever`, `FusedHit`, `LegRanking`, `query.rs:16`), with query-shape routing (`classify_query_shape`, `:73`) and a precision tie-break so a symbol/definition hit outranks a same-score similar-code hit (`rerank_prefer_precise`, `:104`, `source_rank`, `:92`).

**The deterministic RepoMap baseline gate (DOCUMENTED, dossier §5.12).** Aider builds a graph-ranked repo map selecting signatures under an active token budget, defaulting near 1K tokens. HIDE ships that deterministic RepoMap first, and **every learned retriever, context agent, or extra orchestration layer must beat that ~1K-token control on held-out results to earn its latency and complexity.** mini-SWE-agent is the harness-complexity control. This is a kill criterion, not a preference: SWE-Explore and CORE-Bench (dossier §5.1) show general embeddings degrade on agentic code retrieval and that region ranking predicts repair, so the cheap graph baseline is genuinely competitive. Keep repository truth outside the prompt; use an agentic explorer under a hard line/token budget; measure retrieved-to-utilized overlap and tokens per solved task (dossier §5.1, INFERENCE). The `hawking-eval` harness (packed, cheap to reintegrate, archaeology §3.5) is the gate that scores this; only `hawking-bench` (perf) is live today.

## 8. Compaction is a checkpoint, not an essay

When the window fills, HIDE does not summarize into prose. It writes a structured checkpoint (dossier §3.4). This is the same object as a compaction boundary and as an L4 working-task snapshot.

A checkpoint carries: task + acceptance criteria; repository + worktree identity; hard constraints and invariants; decisions and rejected alternatives; touched files and symbols; current patch; tests executed and results; unresolved hypotheses; evidence/artifact references; next actions; **prompt/tool/model ABI version**; permissions and trust-domain state (dossier §3.4). The ABI-version field is what lets a resumed checkpoint verify it is still valid against the live engine (mirrors `HIDE_STATE_CAPSULE_ABI` §4 `IdentityBinding`).

Compaction is cache-safe: fork from the identical parent prefix and append the compaction request, never mutate earlier messages (dossier §5.2, DOCUMENTED for Anthropic/Codex; `HIDE_SPEED_FRONTIER`). In-code, a compacted span records `CompactedFrom { original_id, method, ratio, depth }` (`manifest.rs:243-252`) so compaction is reversible detail-hiding, not a lossy one-way summary, and refuses past depth 2.

**Continuity eval (required, dossier §3.4).** Reconstruct a task from the checkpoint ONLY and compare against an uncompacted control. Measure: forgotten constraints, repeated tool work, wrong-file edits, task-success delta. A compaction strategy ships only if its continuity delta clears a floor on this eval (`hawking-eval` lane). This is how "compaction as checkpoint" is proven, not asserted.

Current state, honest: the live `compact_context` is **logged, never performed** (facade **S4**, archaeology §3.5); its stated performer (the compiler watermark gate) is not on any live path. The FE `autocompact` policy is tested but never fires in dev (archaeology §3.4). So compaction is real-but-unwired: the object and the ladder exist, the trigger is not connected.

## 9. The context-value allocator experiment V(c)

The allocator ranks candidates by value density and fills under the reservation budget (§4). Its objective is a per-candidate value `V(c)`, blended from the four manifest signals:

```text
V(c) = w_rel * relevance(c)      // retrieval score, query-conditioned
     + w_imp * importance(c)     // salience / authorizes-edits weight
     + w_rec * recency(c)        // decay over action window
     - w_red * redundancy(c)     // overlap with already-admitted spans
density(c) = V(c) / token_count(c)
```

**V(c) is not directly observable.** relevance, importance, and salience are latent; the codebase already treats them as declared source priors (`ContextCandidate.score` is a "source-declared base value band in [0,1]", `compiler.rs:57-61` @5a99d0e2) plus optional per-`ContextSourceKind` band multipliers per profile (`SourceWeights::debug_bands`/`refactor_bands`, `profiles.rs:93-112`). The weights `w_*` and the bands are calibrated by experiment, not asserted:

- **Ablation on tokens-per-solved-task and retrieved-to-utilized overlap** (dossier §5.1): drop each stratum / vary each weight, measure downstream task success and wasted context. This is the only honest way to learn V(c); it is an experiment in `HIDE_EXPERIMENT_MENU`, gated by `hawking-eval`.
- **Baseline floor:** the deterministic RepoMap + FTS5 legs (§7) with fixed weights. A learned V(c) must beat that control (§7 gate).
- **Do not overfit orchestration:** each added ranking stage must earn its latency on held-out results (dossier §5.12).

The allocator is the one place a supremacy claim about "better context" must be measured, never designed.

## 10. The load-bearing wire fix (S3)

The single mandate that turns this spec from packed to shipping: **the compiled ContextPack MUST be the input to generation.**

Required change, grounded in the archaeology:

1. On the live turn, call `ContextCompiler::compile(input)` (`compiler.rs:329`) to produce `CompiledContext { prompt, manifest }`.
2. Send `CompiledContext.prompt` to `hawking-serve` as the generation prompt, replacing the raw-prompt / empty-history / `max_output_tokens=256` path (facade **S2**, `host.rs:848-863` @5a99d0e2).
3. Emit `manifest` as the Context Stack projection over the wire (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §3) so both surfaces render exactly what the model saw.
4. Feed the reservation-budgeted output room (`reservation_response`, §4) as the real `max_output_tokens`, not the hardcoded 256.

Until this lands, every downstream context claim is a claim about an object the model never receives. This is Phase 0/1 of the ladder (archaeology §6): lift `hawking-context` + `hawking-index` + the kernel loop out of `5a99d0e2` and make the compiled prompt the turn's actual input.

Related unwired seam: the compiler's `HttpKvStore` client already targets `hawking-serve`'s `/v1/hawking/kv/*` surface, but those endpoints are a deferred runtime-side seam and the client stores no KV bytes itself, only routes (`kv.rs:214-218` @5a99d0e2). L3 reuse (banked KV, warm capsules) is gated on that surface landing (`HIDE_STATE_CAPSULE_ABI` §8).

## 11. Parity vs supremacy ledger

**PARITY** (reproduce Claude Code behavior). Each maps to a behavioral-parity-spec entry.

| Parity target | Spec entry | HIDE status | Delivered by |
|---|---|---|---|
| Context-window map + optimization hints | `cost.usage_transparency` | partial (FE Digest/ContextStack real, mock-fed) | ContextPack manifest (§5) fed live (§10) |
| Focus-able compaction (`/compact [focus]`) | `cost.usage_transparency` | ui_only / packed_unwired | checkpoint compaction (§8) + focus profile |
| Read a project's CLAUDE.md tree + imports verbatim | `config.claude_md` | absent (no reader in active tree) | resolved-instruction stratum (§3 pinned) |
| Auto-memory tree recalled next session | `config.auto_memory` | absent | `HIDE_MEMORY_SPEC` + Memory stratum (§3) |
| Durable transcript / replay | `session.durable_transcript` | packed_unwired | manifest replay substrate (§5) + event log |

**SUPREMACY** (structurally better than a stateless-prompt cloud harness). Every claim is gated on a named build item; none ships today.

| Supremacy claim | Gated on | Basis / honesty |
|---|---|---|
| Resolved instructions compiled once into a warm prefix, free after the first fork (not re-tokenized per session) | L3 KV/capsule exposure (`kv.rs:214-218` seam) + `HIDE_STATE_CAPSULE_ABI` §8 | INFERRED; today the compiler carries the manifest but no L3 route exists (G1, missing) |
| Memory folded into the state fork: no 200-line/25KB truncation cliff, no re-tokenization | transformer/hybrid capsule (currently missing) + Memory stratum wired | gated; RWKV-only state is lossless today, transformer capsule unbuilt (`HIDE_STATE_CAPSULE_ABI` §6) |
| Compaction is reversible detail-hiding, not lossy summarization | `CompactedFrom` wired + continuity eval passing (§8) | supported design (`manifest.rs:243-252`), needs the eval floor (`hawking-eval` reintegration) |
| Context % is exact from local KV state (no estimate) | measured KV occupancy on `hawking-serve` | honest only after G9/G11 fixed; do not print `ctx_len_effective > native` (§2) |
| Skill/plugin corpus pre-indexed into a resident capsule, O(1) hydrate | capsule exposure + skill runtime (both absent) | INFERRED; `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI` owns the skill runtime |

## 12. Build items and gates (feed-forward)

1. **Wire S3 (§10):** compiled prompt -> generation, manifest -> wire. Phase 0/1. Blocks every context claim.
2. **Reintegrate `hawking-index`** with the MVCC daemon running, so the repo-map and RRF retriever feed real spans (not stubs).
3. **Ship the deterministic RepoMap baseline (§7) and the continuity eval (§8) under `hawking-eval` before any learned retriever.** Kill criterion, not optional.
4. **Fix the effective-context lie (§2):** clamp `ctx_len_effective = ctx_len_native` until a measured long-context KV result exists (`HIDE_SPEED_FRONTIER`); do not surface the unset-env multiplier (G9/G11).
5. **Wire compaction (§8):** connect the `autocompact` watermark trigger to the compiler's degrade+checkpoint path (fixes facade S4).
6. **Gate L3 reuse (§10)** on the `/v1/hawking/kv/*` surface landing (`HIDE_STATE_CAPSULE_ABI` §8); until then, banked-KV value estimates are projected, not measured.

The Context OS is proven parts waiting on wires. Its correctness discipline is the same as the rest of HIDE: exact bytes over summaries, measured over asserted, and no capability presented as shipping until the compiled prompt actually reaches the model.
