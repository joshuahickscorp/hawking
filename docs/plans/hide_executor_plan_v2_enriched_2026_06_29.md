# HIDE Executor — Enriched Master Plan (v2)

This supersedes the v1 backend plan by **deepening** it, not widening it. Same six phases; two new
**spines** woven through all of them; every existing surface enriched. The audit confirmed the engine
already carries most of the scaffolding — a per-turn `ContextManifest` (budget: total/used/free +
reservations; model: `ctx_len_native`/`ctx_len_effective`/`tokenizer_sig`), a `MemoryStore` /
`SqliteMemoryStore` in `hawking-context`, an append-only `JsonlEventLog` + projection/replay,
`AgentState.lessons`, a `CompactionEvent` struct, and `.tq` sidecar *detection*. A lot is **connecting
real pieces** — but an adversarial pass found the hardest parts are **genuinely new code**, not just
wiring. The honest ledger:

**Real today (wire it):** `ContextManifest` (model + budget), `MemoryStore`/`SqliteMemoryStore`,
`JsonlEventLog` + projection/replay, `AgentState.lessons` (as `Vec<String>`), `CompactionEvent` struct,
`RwkvState::size_bytes`, `.tq` sidecar detection (`supervisor.rs`), `UiEventBus` token coalescing.

**New code required (build it — do not pretend it exists):**
1. A **`.tq` header reader** → the effective-context multiplier. *Today `ctx_len_effective` is hardcoded
   `= total` (`compiler.rs:439`) and `supervisor.rs` only checks `.with_extension("tq").exists()`.* This
   is the crux of "dynamic context" and must be built, not assumed.
2. `GET /v1/hawking/context` (+ a read-only `/context/status`) and the `Engine`-trait context accessors.
3. The per-step `projection_patch{context_manifest}` streaming hook.
4. `RecallFidelityProbe` (the RWKV needle) + a `recall_fidelity_pct` manifest field.
5. `RecallOracle` + `CompactionRollback` event + a `depth` field on the compaction chain + a recovery path.
6. `AgentState.lessons: Vec<String>` → `Vec<Lesson{ text, phase, step_id, ts }>`; `verdict_history` + convergence.
7. `Interrupt::Steer(String)` (the enum has only Cancel/Pause/Resume today).
8. Word-level diff payload (the stream is token-by-token today; the diff struct is whole-hunk).

Framing, then: **wire the scaffolding, build those eight, and de-risk by shipping the read-only
live-context stream before the quality-gate rollback** (see Sequencing).

Two non-negotiables you set this turn: (A) the context window is **live-interpreted, never a hardcoded
number** — the platform reads what `.tq` actually yields on *that* model + hardware and stays aware
before it must compact; (B) compaction is **high-quality** — it remembers more, not half. Both become
spines, present in every phase.

---

## Spine A — Live context introspection (dynamic, never a constant)

The window is computed and streamed live; the UI reacts to the real ceiling, whatever it is.

**Engine (Hawking).** Add to the `Engine` trait (`hawking-core/src/lib.rs`):
`context_length_native()`, `context_length_effective()`, `recurrent_state_size_bytes()`,
`model_architecture()`. `RwkvState::size_bytes` already computes the constant state footprint; expose
`state_occupancy_bytes()`. Add `GET /v1/hawking/context` (`hawking-serve/src/http.rs`) →
`{ model_id, arch, ctx_len_native, ctx_len_effective, state_bytes_per_slot, active_slots, free_slots,
tokens_per_sec_estimate }`.

**The .tq multiplier is read, not assumed (NEW code).** Today `supervisor.rs` only checks
`weights.with_extension("tq").exists()` and `ctx_len_effective` is hardcoded `= total` — so this is a real
build, not a wiring task. Add a small `tq_metadata` reader that opens the sidecar, parses the strand-quant
header (magic + version + an effective-context multiplier field, or derives it deterministically from the
recorded bits-per-weight), and stores `SupervisorConfig.tq_multiplier: Option<f32>`. The endpoint surfaces
it as a **literal `effective_multiplier`**, and `ctx_len_effective = native × multiplier`. Roundtrip test:
a mock `.tq` with a known multiplier surfaces verbatim. Caveat the research flagged: the ratio is **not
constant across layers/families**, so it is a measured per-model number, never a literal in code.

**Two regimes, one signal.** Transformers: occupancy = `kv_seq_len / max_position_embeddings` (a real
fullness). RWKV-7 (SSM, constant state, **no token cap**): reframe to a **recall-fidelity horizon**.
Concretely (NEW): a `RecallFidelityProbe` trait `fn probe(&self, fact, age_steps) -> f32` with a
calibration pass at model load — inject a known fact, step N forward, measure recoverability into a spline
`f(state_age) -> fidelity_pct`; start with a conservative linear stub (`max(0.9 - age/100, 0.3)`) and
replace with measured values. Record `recall_fidelity_pct` in the manifest and surface it as **its own
Context Stack line** ("state age 2450 tok · recall 92%"), not the saturation cue. The UI shows "how
sharp," not "how full." (M1.)

**Per-step stream.** After each `kernel.step()`, `host.rs::generate_and_publish` emits
`projection_patch{context_manifest}` with `{ used_tokens, free_tokens, kv_occupied | recall_headroom,
watermark_level }`. FE `ContextManifest` type + `ContextStack` render it live.

**Watermarks drive compaction *before* the cliff**, computed against the live ceiling (not 8192-anything):
~60% soft ("layer ready to compact"), ~75% warn (recency decay begins), ~90% act (auto quality-compact).
Graceful degradation, never a hard 100% wall.

**Doctrine:** still **no budget meter**. Surface as an *ambient* cue — a quiet 2-3px saturation line that
warms only near the recall horizon, plus "whole project loaded · cached." Abundance, not a gauge.

## Spine B — Quality memory + compaction (remember more, not half)

Compaction is the feature, and its quality is **measured**, not assumed.

**Structured, not a blob.** Use the existing `hawking_context::memory::MemoryStore` (instantiate
`SqliteMemoryStore` at `.hide/memory/memory.db`, add as `BackendServices.memory_store`). Records are typed:
file-facts (path/hash/loc), **decisions (+rationale)**, test-results (pass/fail + assertion), entities
(files/symbols/tests + relations), **user-constraints**, **failed-approaches**. This is the Project Brain
as a knowledge graph, not a summary paragraph.

**Tiers (MemGPT/Letta).** Core (always in context: persona, hard constraints), Working (this run),
Archival (paged in on demand via the hybrid retrieval surface). Compaction moves cold spans down a tier,
it does not delete them.

**Lossless where it matters.** The compactor may summarize narration and tool chatter, but **never drops**:
touched file contents, decisions+rationale, test verdicts, user constraints, failed approaches. Salience
scoring (`value = (importance + relevance + recency − redundancy)/tokens`, query-aware) ranks the rest.
Each reduction writes a `CompactionEvent{ original_id, result_id, method: truncate|summarize|drop,
ratio }` so it is auditable and reversible.

**Measured quality (the anti-"half-memory" guarantee — NEW code).** The `CompactionEvent` struct exists
but carries only method/ratio; there is no recall test, threshold, or rollback path today. Build a
`RecallOracle`: pin 10-20 deterministic facts from the pre-compaction context, re-ask them on the
post-compaction context, compute recall@k. Extend `CompactionEvent` with `recall_score: f32`, `depth: u8`,
`rolled_back: bool`, and add a `CompactionRollback` event to `hide_core::event`. Wire into the compiler's
compact pass: if `recall < 0.85` **or** importance-weighted dropped tokens > 10% **or** test coverage
regresses, emit `CompactionRollback`, restore the prior manifest, and surface a minimal Timeline card
("compaction hurt recall, reverted"). Anti-compounding guard with teeth: a `CompactedFrom` chain tracks
`depth`; `compiler.rs::realize()` walks it and **at depth > 2 falls back to the original uncompacted text**
(auto-rollback, no question). The recall harness is also the Phase-verification oracle (see Verification).

**Hybrid recall (honest caveat).** Long context is for whole-program *reasoning*; keep
`code_index.search` as the exact-lookup path. Lost-in-the-middle is positional and not fully fixed by
better compaction — so recall-critical facts are pinned to Core tier (prompt start/end), not left mid-context.

**Wiring.** `EventLog::compact_before(seq)` archival hook (`hide-core`); `replay.rs::rebuild_with_summary()`
+ `BasicProjectionEngine::fold_from_memory()`; `AgentState.lessons: Vec<String>` →
`Vec<Lesson{ text, phase, step_id, ts }>` so learnings anchor to the decision that produced them.

---

## Deepening the six phases (enrichment, mapped to real code)

**Phase 0 — Live streaming.** + per-step `context_manifest` stream (Spine A); + salience score on each
`token_batch` (errors/control-flow weighted) feeding the Timeline; + **word-level** diff streaming so
hunks render incrementally as the kernel emits them (Phase-0 backend) instead of waiting for the whole diff.

**Phase 1 — Continuous loop, deepened.** Verify becomes an **oracle ladder** run cheapest-first with
short-circuit: build → typecheck → test → lint. Each Fail is distilled into a **minimal repair context**
(file/line/code/message) fed straight into the next generation — that is the reward-from-tests signal made
concrete. **Convergence detection**: track the last K verdict sets; identical across the window → stop
(prevents infinite churn). **Escalation ladder** by step difficulty: ReAct (1 try) → Best-of-N (oracle-
ranked) → Tree-of-plans (only on hard steps; **cap breadth** — ToT is exponential). Reflexion → structured
`Lesson`s, query-indexed; plateaus ~5 (research) so keep it bounded.

**Phase 2 — Fork-&-Try-N, deepened.** **Oracle-weighted branching**: fork count ∝ step difficulty +
oracle availability (don't fork uniformly). **Fork memory**: record which branch won and *why* (coverage,
quality, constraint-fit) into the MemoryStore so later forks start smarter. **Speculative pipeline
execution**: after a decision point, fork the RWKV state and pre-run the *next 2-3* likely tool calls
(read→typecheck→build) in the background — free state copy (M1), only the divergent generation costs
tokens (and tokens are local/free, M2). Convergence-preview cards on the board.

**Phase 3 — Long context, deepened.** **Permalayer** (stable: project graph + repo, cached via prefix
KV-reuse, content-addressed) + thin **volatile** action layer. **Project Brain = the knowledge graph**
above. **Trust scores** per code section: never-failed areas cached longer; flaky areas re-verified. A
**pre-turn `context.compile` refresh** (`handle_intent` before spawn) keeps the Permalayer warm.

**Phase 4 — Agency, deepened.** **Risk-tiered gates**: (1) inform-only Radiate dot, (2) suggest gate
(auto-dismiss if idle), (3) hard block (InlineGate) for destructive/network/push. **Provenance verdict
chain** on every action ("generated from task → phase → tool call → oracle verdict"). M4 grammar-guaranteed
tool calls = the call is always schema-valid JSON. Credentials via OS keychain / credential-request flow,
never typed by the agent.

**Phase 5 — Voice, deepened.** Local Whisper, **partial transcripts** streamed as you speak; voice can
**steer mid-run**. Concretely (NEW): add `Interrupt::Steer(String)` (the enum has only Cancel/Pause/Resume
today); `Governor::check()` appends it to `AgentState.steer` and **continues** (does not pause); the next
Verify/Repair phase prepends the steer text to the oracle input. The Composer emits a `Steer` per phrase as
transcription streams.

---

## Provenance & checkpoint spine (cross-cutting)
Every step snapshots at log seq N (`AgentCheckpoint`); restoring replays the fold to N (effects recorded,
not re-fired) — deterministic undo. Each action carries its verdict chain, surfaced as a light breadcrumb.
This is what makes a long autonomous run *trustable* and reversible.

## UX enrichments (deepen the surfaces we have)
- **Inline diff:** live per-hunk streaming + word-level highlight; on Tab-accept, pre-highlight the next
  hunk/card (predictive next-action).
- **StateTimeline:** continuous **drag-to-scrub** (not click); **diff-checkpoint markers** at each oracle
  pass; **manifest-ring** so scrubbing to step N shows the Context Stack *as it was* at that decision.
- **Fleet board:** convergence-preview micro-cards ("Branch 3 done · 2 edits from trunk · diff vs trunk").
- **Context Stack:** ambient saturation cue; a **minimal Decisions panel** ("persona pinned", "failed
  approach kept") that by default shows only *that* something compacted — **never ratios/methods/scores**
  (those would re-introduce a budget-meter feel); the full **Compiler Audit Trail** with per-span scores is
  power-user-only, collapsed behind Cmd+Shift+?. Plus the RWKV recall-fidelity line.
- **Radiate:** stratified — 8px dim dot at rest → 16px ring with phase labels on hover → full detail on click.
- **Composer:** peek/preview glass overlay on Cmd-click of a file/symbol reference.
- **Motion:** the existing `--ease` (cubic-bezier(.2,0,0,1)), 120-220ms, on hunk settle, dot select, scrub.

## Honesty / doctrine guardrails
No budget meter (ambient cue only). Never "infinite/perfect memory" → "always loaded, instantly resumed,
never billed, never truncated." Long context ≠ perfect recall (pin recall-critical to Core; keep exact
index). A surface shows real data or says it is a scaffold — never fake a stream.

## Known pitfalls (each with its countermeasure)
- `max_position_embeddings` ≠ usable capacity → measure effective live; treat native as a floor, not the number.
- Tokenizer counting is model-specific → use the model's own tokenizer; `chars/4` only as a labeled boot fallback.
- `.tq` ratio varies by layer/family → read/measure per model, surface as a literal; never a constant in code.
- Recursive summarization compounds loss → `depth ≤ 2` enforced in `realize()` with auto-rollback to original.
- Lost-in-the-middle is positional → **pin** recall-critical facts to Core (prompt start/end); don't trust mid-context.
- Optimistic UI needs <500ms backend → skeleton + reconcile; if a confirm lags, the optimistic state stays put, no flicker.
- ToT breadth is exponential → gate escalation by step `difficulty` and hard-cap branch count (≤5).
- Reflexion plateaus ~5 lessons → bound the list; evict lowest-salience rather than growing unboundedly.
- Deterministic oracles necessary-not-sufficient → a green build still goes through the critic/judge before Accept.

## Sequencing (de-risked: smallest honest ship first)
1. **MVP — Spine A read-only.** `GET /v1/hawking/context` + the `.tq` reader + per-step
   `projection_patch{context_manifest}` + the ambient Context-Stack cue. This delivers the **dynamic,
   live-interpreted window** you asked for first, with no dependency on the hard compaction work. Pure
   observation — low risk.
2. **Phase 0 streaming + Phase 1 loop** — token_batch persistence, the oracle ladder + convergence, the
   inline-diff streaming. The Executor becomes visibly continuous.
3. **Spine B — quality compaction.** Memory tiers + structured Project Brain first (additive), then the
   `RecallOracle`/`CompactionRollback` quality gate **once compaction is proven in the field** (the critics'
   point: don't block the first ship on the hardest proof).
4. **Spine C — the maximalist command system** (folded in from
   `docs/plans/hide_command_system_maximalist_2026_06_29.md`). The one trigger grammar (`/ @ # > : !` +
   Cmd+K) in the Executor composer is how the user *reaches* everything the spines built: `@`/`#` pull
   Spine A context + Spine B Project-Brain memory; `/` commands declare `flavor: single|generate-n|fleet|
   ambient` and ride Fork-&-Try-N; ambient commands are the tiered oracle lane. Sequenced MVP-first
   (`/` over the existing palette -> `@`/`>` -> `.hide/commands` files + `CommandDispatcher` -> the
   fleet/speculative/ambient multipliers -> pipelines/macros -> the frontier layer).
5. **Phases 2, 4, 5** + UX enrichments — interleave per surface.

> **Surface guardrail across all of the above (V16 engine, economy-car dashboard):** every spine adds
> CAPABILITY, never default chrome. The maximalism lives under the hood (fleets, ambient oracles,
> speculative previews, the command grammar); the surface stays minimalist concrete. New features are
> *invisible until summoned* — reached through the one trigger grammar, the command palette, and
> progressive disclosure, surfaced ambiently as quiet Context-Stack lines, never as new on-screen widgets.
> Run the clutter-creep check (see house rules) on every addition.

## Verification (each phase gated)
- Recall harness: needle-in-haystack across compression ratios — the compaction-quality oracle (must hold
  recall above threshold or rollback fires).
- Live-context probe: occupancy/recall stream tracks a scripted long turn; watermark fires before degradation.
- Fork cost: `try 5` measured as ~constant state-copy (no re-prefill).
- All behind the project's parity/CI discipline before merge to `main`.
