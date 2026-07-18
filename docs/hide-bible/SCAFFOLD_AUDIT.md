# HIDE Scaffold Audit — Gap Inventory

> **Audit date:** 2026-06-27
> **Subject:** The 11-crate HIDE (Hawking IDE) Rust scaffold (agent-generated) under `crates/`, measured against `docs/hide-bible/` chapters 01–11.
> **Method:** Every `.rs` file in every HIDE crate was read in full, plus each crate's `Cargo.toml`, plus the binding-contract sections of the relevant bible chapters. The workspace compiles clean (`cargo check --workspace` green) and contains **zero** `todo!()`/`unimplemented!()` markers — so gaps are **shallowness** (thin logic, surface-only types, deterministic placeholders, unwired seams), not explicit stubs. Classifications below reflect runtime depth, not whether code exists.

Legend for module classifications:

- **REAL** — substantive working logic, not a passthrough.
- **PARTIAL** — core happy path exists; missing branches / error-handling / edge cases.
- **SURFACE** — types & signatures defined, bodies trivial/placeholder (returns `Ok(Default::default())`, empty `Vec`, echoes input, sums a pre-filled field).
- **BOUNDARY-STUB** — deliberately stubbed where it will wire to the live Hawking runtime (`hawking-serve`/`hawking-core`) or an OS/external system; seam noted as *clean* (swappable trait/client) or *hardcoded/unwired*.
- **MISSING** — a bible contract with no corresponding code at all.

---

## 1. Executive summary

### Overall state

The scaffold is a **breadth-first skeleton with a genuinely solid foundation crate and a thin everything-else**. `hide-core` (the shared-contracts crate) is the standout: real chain-hashed event log (in-memory + JSONL with content-addressed integrity), a real permission engine (deny-beats-allow, lethal-trifecta gate), real file-backed persistence (blob CAS, projection store, KV store), and a real tool dispatcher that calls `simulate()` for dry-run and routes through the permission engine. `hide-backend` is the second-most-real: it actually composes all the sibling crates into a `BackendHost`, wires the intent→event→projection path, and folds events for replay — but it is **not** the Tauri host the bible describes (no `tauri` dep, no `#[tauri::command]`, no runtime-sidecar supervision).

Everything else is **type-faithful but logic-shallow**. The pattern is consistent across all nine remaining crates: the *vocabulary* the bible mandates is transcribed into well-formed serde types and clean trait seams, and a small number of pure algorithms are genuinely implemented (DAG ready-set + cycle detection, RRF kernel, permission/admission predicates, oracle-first tournament selection, a real event-fold projection). But **every piece the bible treats as the substance — the agent loop, the verification oracles, the context knapsack, real codebase parsing/embedding, the inference cascade, the merge engine, encryption, RLEF reward — is reduced to its most trivial form or left as an unwired seam.**

The single most important structural finding: **the scaffold's `Event` type diverges from the bible's cross-cutting event contract (ch.01 §4.6) in a way that the bible explicitly rejected.** This must be fixed first because chapters 02–06 all bind to it (details in §3 and §4).

### The 3–4 biggest gaps

1. **The agent kernel does not run an agent loop.** `hide-kernel` (the thinnest crate, highest priority) has the FSM *states* exactly right but the driver is a hardcoded straight-line march: it always synthesizes a one-step plan (bypassing the `Planner` trait), emits a fabricated `Verdict{status:Pass, score:1.0, oracle:"stub"}` inline (never calling the `VerificationGate` or any `Oracle`), and routes `Repair`/`Replan`/`Paused` into a single no-op arm. The three budgeted loops (step/repair/replan) and the approval gate — the chapter's entire thesis — **do not execute**. The deterministic oracle "suite" is one always-Pass stub, not the 8 mandated oracles. This guts tenet K1 ("no state advances on faith").

2. **The event contract diverges from the bible's normative envelope (ch.01 §4.6), and it's the wrong shape on purpose-rejected grounds.** The scaffold uses a **closed Rust enum** `EventPayload` — exactly the "giant `#[non_exhaustive]` enum" the bible says it rejected because "it makes the core the bottleneck for every new event kind and breaks WASM-plugin event emission." The bible mandates `payload: serde_json::Value` (`#[serde(flatten)]`) + an `ext` forward-compat capture + `cause`/`actor` fields (for OpenHands-style Action/Observation replay pairing) + ULID ids + blake3 chain hashing. The scaffold has none of those: enum payload, no `cause`/`actor`, hex-counter ids (not ULID), SHA-256 chain (not blake3). Because every other chapter binds to this envelope, the divergence propagates everywhere.

3. **"Codebase intelligence" has no parser, no hasher, no vectors, and no daemon.** `hawking-index` depends on **no** `tree-sitter`, **no** `blake3`, **no** vector store, **no** `rusqlite`/FTS5, **no** `notify`. Symbol "parsing" is a literal line-prefix scanner (`"pub fn "`), every occurrence is hardcoded `role:"definition"` (so `references()` always returns empty), `include_semantic` is accepted but never acted on, merkle is a trait with no impl and no hashing, and `InMemoryCodeIndex` dies with the process. The moat capability the bible calls "the thing the cloud literally cannot replicate" is a RAM map over a prefix scanner.

4. **The context compiler is naive concatenation, not a budgeted knapsack.** `hawking-context::compile()` is a single greedy `if used + token_count <= capacity` loop with `parts.join("\n\n")`; token counting is `chars/4` (no tokenizer dep); memory retrieval is `lexical_overlap + importance` (no recency, no embeddings, no SQLite/FTS5 — it's a RAM `BTreeMap`); the `KvStore` is a 2-method read-only trait with no implementation. Reservations, degrade ladder, head/tail ordering (anti-LITM), and the `PrefixKey` byte-compatibility with the in-tree `prefix_cache`/`prefill_disk` (the explicit interop requirement) are all absent.

**Honorable-mention gaps:** `hawking-orch` has a real registry/router and a real-but-primitive hand-rolled HTTP+SSE client, but the confidence-aware **escalation cascade** (ch.06's thesis) is entirely absent and 4 mandated modules don't exist; `hide-fleet` models scheduling decisions but has **no concurrency, no git worktrees, no merge engine** (it decides but never does); `hide-security` has a real audit hash-chain but **no crypto dep at all** (encryption-at-rest is a label) and redaction is a 2-prefix toy; `hawking-research` and `hide-personalize` are type-schemas with ~2 real modules each and **no runtime/model client**.

### Recommended completion order (foundational → leaf)

```
TIER 0 (foundation — everything binds to it):
  hide-core         ← fix Event envelope (§4.6 contract), blake3, ULID, cause/actor,
                       enrich ToolError/ToolResult. Already mostly REAL.

TIER 1 (the runtime seam + the loop's direct dependencies):
  hawking-orch      ← real HTTP client (all 3 endpoints), escalation cascade,
                       ModelRole.escalates_to, confidence signals.
  hawking-index     ← tree-sitter + blake3 merkle + SQLite/FTS5 + vectors + daemon.
  hawking-context   ← real knapsack + tokenizer + SQLite memory + KvStore client.
  hide-tools        ← EXEC_NONZERO fix, shell sandbox, edit family, real MCP.

TIER 2 (the brain — depends on Tier 0+1):
  hide-kernel       ← wire the real FSM loop, the oracle suite, the governor,
                       best-of-N search, replan/repair, runtime_client in Act.

TIER 3 (leaf features — depend on the kernel + tools):
  hide-fleet        ← git worktrees, async dispatch, merge engine, event projection.
  hide-security     ← crypto-at-rest, real redaction, sandbox exec, anchors.
  hawking-research  ← RuntimeClient, CAS, graph store, one real source adapter.
  hide-personalize  ← eval execution, RLEF reward derivation, meta-router. (moonshots)

TIER 4 (the host — composes everything):
  hide-backend      ← tauri deps + #[tauri::command], RuntimeSupervisor, push UiEvent channel.
```

The dependency logic: **hide-core's event/types must be normative before anything else** (every crate's event emission and ID scheme inherit it). **hawking-orch's `InferenceClient` and `hawking-index`/`hawking-context` must be real before hide-kernel's loop can do anything** (the kernel's `Act` step needs a runtime to generate and an index/context to ground it). The leaf feature crates and the Tauri host are last because they orchestrate the lower tiers.

---

## 2. Per-crate sections

### 2.1 `hide-core` → bible ch.01 (system architecture)

**The foundation, and it's genuinely real.** 2422 lines / 17 files. Real chain-hashed event log, real permission engine, real file-backed persistence. The gaps are *contract divergences* from ch.01 §4.6, not shallowness — which matters more because everything binds to this.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `event.rs` | **REAL** (divergent contract) | In-memory + JSONL `EventLog` with monotonic `seq`, real SHA-256 chain hash over canonical bytes, reopen-and-continue, parse-error context. Two real tests incl. reopen+chain. | **Diverges from §4.6 normative envelope** (the single most load-bearing finding): `payload: EventPayload` is a **closed Rust enum** — the bible explicitly *rejected* this ("makes the core the bottleneck for every new event kind, breaks WASM-plugin emission") in favor of `serde_json::Value` + `#[serde(flatten)]` + `ext` forward-compat capture. Missing `cause: Option<EventId>` (Action/Observation pairing for replay, T3) and `actor: Option<String>`. `ts` (micros) is absent. No Action/Observation **class** tagging. | **partial / N on envelope** |
| `ids.rs` | **PARTIAL** | 14 newtype ids via macro, `Display`/`From`/`Default`, atomic counter. | IDs are `{prefix}_{ms:013x}_{n:08x}` — **not ULID** (bible §4.6 mandates ULID: "sortable, timestamped, no coordination"). Not stable across machines/merges as the bible requires (`id` stable, `seq` local). `TimestampMs` is ms; bible uses micros for `ts`. | partial |
| `tool.rs` | **REAL** | Real `Tool` trait (BoxFuture-based), `ToolRegistry`, and a real `ToolDispatcher` that calls `simulate()` to derive the permission target + serve dry-run, routes through `PermissionEngine`, enforces deny. `ToolSpec`/`ToolResult`/`ToolStatus`/`ToolContent`/`ToolError`/`Purity` all present. | `ToolError` has `recoverable` but **no `retriable`/`fix_hint`/`schema_path`** (the self-correction triad, ch.03 §4.2.3) and error codes are ad-hoc strings, **not the mandated taxonomy** (`ARG_INVALID`/`EXEC_NONZERO`/…). `ToolResult` has **no `provenance="tool-output"`** field (TT8) and no `exit_code`. The trait is `BoxFuture` not the bible's `#[async_trait]` — fine, but downstream tools inherit the thin error shape. | partial |
| `permission.rs` | **REAL** | `StaticPermissionEngine`: deny-beats-allow rule matching, high-risk-network forced-Ask, `pattern_matches` glob (`/**` suffix), default-Ask, lethal-trifecta risk gate. Real test (deny beats allow). | `pattern_matches` is prefix-glob only (no full glob). No interactive grant-ledger / capability-negotiator (that's the host's job per §7.3). No per-grant exact-effect-hash enforcement path despite the field existing. | Y (mostly) |
| `persistence.rs` | **REAL** | `BlobStore` (in-mem + file CAS with sha256 content addressing + atomic rename), `ProjectionStore` (in-mem + file JSONL), `KeyValueStore` (in-mem + file). `EventLogIntegrity` trait + `IntegrityReport`. 3 real roundtrip tests. | CAS uses **sha256** not the bible's **blake3** (§4.7 "FastCDC + blake3 keys"); InMemoryBlobStore uses a `stub-` hash. No FastCDC chunking/dedup. No redb cache tier, no SQLite metadata DB (bible §4.7 names SQLite+sqlite-vec+redb explicitly) — these are file/JSON stand-ins. No segment format (`[u32 len][event][32B hash]`). | partial |
| `runtime.rs` | **REAL** (types) | `ModelDescriptor`, `ModelRole`, `RolePurpose`, `ProviderCaps`, `SamplerProfile`, `InferenceRequest`, `StreamChunk`, `GenerationStats`, `ModelProvider` trait, `TokenSink`. Good shapes consumed by hawking-orch. | `RolePurpose` diverges from bible `RoleKind` (HeroCoder≠Hero, Summarizer≠Compactor, **ToolPlanner invented**, **SsmLong + Custom missing** → no long-context SSM routing). `ModelRole` lacks `endpoint`, `cost`, **`escalates_to`** (so the cascade graph can't be expressed). `ProviderCaps` lacks `speculative`/`draft_for`/`adapters`/`max_batch`. | partial |
| `api.rs` | **REAL** (types) | `Intent` (11 variants incl. SubmitTurn/Cancel/Pause/Fork/ScrubToEvent), `IntentAck`, `UiEvent`, `UiEventKind` (ProjectionPatch/TokenBatch/RuntimeStatus/ToolProgress/SecurityGate/Error/Custom). Clean Wire-A/Wire-B shapes. | Pure data; the host that consumes it (hide-backend) doesn't yet stream `UiEvent` over a channel. No `cause`/causal threading. | Y |
| `security.rs` | **REAL** (types + logic) | `TaintedValue` w/ `is_untrusted()`, `LethalTrifectaAssessment::assess()` (real triple-AND logic), `SandboxProfile`, `SandboxTier`, `NetworkPolicy` (default-deny). | These are the data structures; OS enforcement lives in hide-security (which is itself shallow). `assess()` is correct but trivial. | Y (for its layer) |
| `config.rs` | **REAL** | `HideConfig` (runtime/persistence/security/context/index sub-configs) with `for_workspace` (HAWKING_HOME/HOME resolution), JSON load/save, real roundtrip test. Sensible secure defaults (network Deny, shell Ask). | No TOML, no 3-layer workspace/user/project layering (§4.10 mandates layered + `locked`); `ConfigLayer` type exists but no merge logic. | partial |
| `plugin.rs` | **SURFACE** | `ExtensionManifest`, `ExtensionRuntime` (TrustedRust/Wasm/Mcp/Skill), `ExtensionContribution` (Tool/Panel/ModelProvider/Indexer/…), `ExtensionRegistry` (register/get/len). | Pure registry of descriptors. No activation events, no capability negotiation, no WASM host. Matches §7.2 manifest *shape* loosely but no engine. | partial |
| `supervision.rs` | **SURFACE** | `ProcessSpec`, `ProcessStatus`, `BackoffPolicy` (with the bible's default backoff ladder). | Pure types; no supervisor (that's hide-backend's gap too). | partial |
| `observability.rs` | **SURFACE** | `LogRecord`/`MetricSample`/`HealthReport`/`HealthCheck`/`HealthStatus`. | Pure types; no tracing/metrics wiring despite `tracing` dep. | partial |
| `migration.rs` | **SURFACE** | `SchemaVersion`, `MigrationPlan`/`Step`/`Report`. | Types only; no migration runner. | partial |
| `project.rs` | **REAL** | `Workspace` + `WorkspaceLayout` — the full `.hide/` directory map (log/snapshots/projections/meta.sqlite/kv/vectors.sqlite/blobs/taint/cache/sandbox/tmp). Matches §4.8 on-disk layout closely. | Just paths; nothing creates/validates modes (the S12 0700 check is absent here and in hide-security). | Y |
| `types.rs` | **REAL** (types) | `TrustLevel`, `RiskLevel`, `Decision`, `Provenance` (w/ `trusted()`), `Effect`/`EffectSet`/`EffectKind`, `BlobRef`, `FileSpan`, ranges. Solid shared vocabulary. | `Provenance.confidence: f32` (bible A.2/F12) is absent — only `trust`. Downstream crates only ever set `trusted(...)`. | partial |

**To complete this crate:** This is the highest-priority and lowest-effort high-impact work. **(1)** Reshape `Event` to the §4.6 normative envelope: `payload: serde_json::Value` with `#[serde(flatten)]`, add `ext: BTreeMap` forward-compat capture, add `cause`/`actor`/`ts`(micros), and add an Action/Observation class tag — keep typed accessors for core kinds but stop forcing a core edit per kind. **(2)** Switch ids to real ULID (`ulid` crate) so `id` is machine-stable and `seq` stays local. **(3)** Switch the chain hash and blob CAS from sha256 to **blake3** (and coordinate the same change in hide-security's verifier). **(4)** Enrich `ToolError` (`retriable`/`fix_hint`/`schema_path` + stable code taxonomy) and `ToolResult` (`provenance`, `exit_code`). **(5)** Add `escalates_to`/`endpoint`/`cost` to `ModelRole`, `SsmLong`/`Custom` to the role enum, and `confidence` to `Provenance`. **(6)** Add config layering (TOML, workspace/user/project merge). Everything else (event log, permission, persistence, dispatcher) is production-shaped and only needs the contract corrections to propagate.

---

### 2.2 `hide-kernel` → bible ch.02 (agent kernel) — THINNEST, HIGHEST PRIORITY

**A structural skeleton, not a working kernel.** 947 lines / 25 files. Module topology is right; the FSM states match the bible *exactly*; three things do real work (projection fold, DAG ready-set + cycle detector, the runtime-client seam). Everything else is SURFACE, and critically **the loop does not execute**.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` (`AgentKernel`) | PARTIAL | Owns `DynEventLog`; `start_run` emits real intent event + builds `AgentState`; `step()` delegates to driver; passing integration test drives a run to `Done`. | Doesn't own sessions/runs/governor as the bible says — thin façade. No session registry, no run table, no interrupt plumbing. | partial |
| `machine/state.rs` (`Phase`,`AgentState`) | **REAL** (types) | `Phase` matches the bible's 12 states **verbatim** (Intake…Paused); `is_terminal()` correct; `AgentState` has phase/plan/cursor/budget/ledger/last_verdict/repair_count. | `AgentState` omits the bible's `stack: Vec<Frame>` (so bounded-depth search/subagent recursion is impossible), `replan_count`, `context_manifest`. `pending_approval: Option<String>` not a typed request. | partial |
| `machine/driver.rs` (FSM executor) | **SURFACE** | One-transition-per-call shape; emits phase + `plan.created` events; checks `budget_allows_step`; uses `PlanDag::ready_steps`. | **The core is faked.** `Plan` hardcodes `Plan::single_step(...)` (planner never called). `Act` emits `agent.action.stubbed` ("tool/model action boundary scaffolded") — no generation, no dispatch, no search. `Verify` **inlines** `Verdict{status:Pass, score:1.0, oracle:"stub"}` — the gate and oracles are never invoked. `Repair`/`Replan`/`Done`/`Aborted`/`Paused` collapse into one arm emitting "phase has no scaffold transition" — **all three loops + approval gate are unreachable**. No interrupt poll, no replay `Mode`. | **N** |
| `machine/guards.rs` | **SURFACE** | `budget_allows_step` + `plan_has_ready_step`. | The bible's guards (`deps satisfied ∧ pending`, `dag.acyclic()`, `repair_count<max`, `autonomy==suggest-only ∧ effectful`) are absent except the trivial two. `has_cycle` exists in dag.rs but is never used as a guard. | partial |
| `machine/effects.rs` | **PARTIAL** | Two real event constructors (`agent.phase`, custom). | K5's defining property — effects never run during replay — is absent: no `Mode::{Live,Replay}`, no short-circuit, no recorded-outcome fold. | partial |
| `plan/schema.rs` | **PARTIAL** | Solid serde `Plan`/`PlanStep`/`StepKind`/statuses; `single_step` ctor. | **Diverges from normative A.1.** `PlanStep` has **no `acceptance`** (the bible's "most important field" — every step declares its oracle up front, K1), no `parent`/`rationale`/`produced`/`search_hint`/`attempts`. `Plan` carries **no `budget`**. `StepKind` variants differ (Implementation/ToolCall/Summary vs investigate/edit/command/verify/synthesize/decompose/delegate). | partial |
| `plan/planner.rs` | **SURFACE / BOUNDARY-STUB** | Clean `Planner` trait + `StubPlanner` (single-step). Seam swappable. | No decomposition / constrained-decode synthesis. **The driver bypasses it** — even the seam is unwired. | partial |
| `plan/replan.rs` | **SURFACE** | `ReplanRequest`/`ReplanResult` structs; `supersede(plan)` flips one status to `Superseded`. | The archetype shallow body: **no localized-vs-full logic, no diff, no lesson carry-forward, no `revise()`**. The entire §4.5.3 replanning machine is missing. | **N** |
| `plan/dag.rs` (`PlanDag`) | **REAL** | `ready_steps` (pending ∧ all deps completed) and `has_cycle` (real DFS w/ visiting set) — both algorithmically sound. | `has_cycle` is dead (never gates Plan/Replan as §4.5.2 requires). No topo order, no parallel-branch detection for subagent fan-out. | partial |
| `verify/oracle.rs` (`Oracle`,`Verdict`) | **PARTIAL** | `Oracle` trait + `Verdict{status,score,oracle,detail}` + `VerdictStatus`. | **Diverges from A.2.** Trait lacks `class()->OracleClass{Deterministic\|Probabilistic}` — the *entire ranking mechanism* (deterministic outranks probabilistic) is unexpressable. Missing `cost_hint()`. `Verdict` lacks `failures: Vec<Failure>{file,line,code,category,message}` (so minimal-repair context, §4.7, is impossible), `artifacts`, `duration_ms`. | partial |
| `verify/gate.rs` (`VerificationGate`) | **PARTIAL** | Real `decide(&[Verdict])->GateDecision`: any Fail→Repair, any Pass≥min→Accept, else Replan. | No oracle ranking (can't — `class()` absent), no tie-break (§4.8.4), no Inconclusive→consistency/judge path. **The driver never calls `decide()`** — orphaned. | partial (orphaned) |
| `verify/deterministic.rs` | **SURFACE** | `StubDeterministicOracle` — `verify` always returns Pass/1.0. | Bible mandates **8 real oracles** (`patch_apply, typecheck, build, test, lint, grep_ast, schema, runtime_smoke`) in a directory. Here: **one always-Pass stub**. Biggest single content gap vs the chapter's reliability core. No `llm_judge.rs`/`consistency.rs`. | **N** (1 of 8+, and a no-op) |
| `search/strategy.rs` | **SURFACE** | `SearchStrategy` trait; `SearchTier{React,BestOfN,TreeOfThoughts,Lats,Debate}` (names match); `Candidate`; `EscalationLadder`. | **Zero strategy implementations.** No `best_of_n()` (the chapter's self-described "centerpiece"), no parallel worktree fan-out, no oracle-gated selection, no scoring, no `pick_tier`. Names only. | **N** |
| `skills/mod.rs` | **SURFACE** | `SkillRecord`/`SkillQuery`/`RankedSkill` DTOs. | No store/retrieve/curate; no capture-on-success, no promote/decay; `SkillRecord` diverges from A.6 (flat `body:String`, no `kind/trigger/validation/importance/embedding_ref`). | partial |
| `subagent/mod.rs` | **SURFACE** | `SubagentSpec`/`Handle`/`Return`/`IsolationMode` DTOs. | No `spawn`/`join`/fold, no isolation forking, no nested-`AgentState` recursion, no budget rollup. Diverges from A.4 (no `return_contract`/`context_seed`/`deadline`). | partial |
| `tools/mod.rs` | **PARTIAL** | `lint_tool_call` (2 real checks: non-empty name, args-is-object); idempotency/dispatch record structs; uses real `hide_core::tool` types. | No dispatcher, no idempotency dedup/replay logic (structs only), no ACI lint depth (unknown-tool / hallucinated-file / broken-edit). | partial |
| `govern.rs` (`Budget`,`Ledger`,`Interrupt`) | **PARTIAL** | `Budget` + defaults; `BudgetLedger.within()` (steps + token cap); `consume_step()`; `Interrupt{Abort,Pause,Steer}`. | Missing most of A.5 (`max_subagents/stack_depth/tool_calls/edits_per_file/search_*/self_consistency_k/escalation`). **No Governor object, no `check()` with structured Abort, no telemetry, no autonomy levels, no interrupt handling** (the enum is never polled). The "single chokepoint" (K8) is a 2-condition boolean. | partial |
| `cooperate.rs` | **SURFACE / BOUNDARY-STUB** | DTOs naming the §4.12 hooks (logprob/entropy confidence, schema constraint, draft control). | No behavior; explicitly `[RUNTIME-SIDE — LATER]`, so DTO-only is defensible, but there's no seam (trait/client) to the runtime. | partial (LATER) |
| `checkpoint.rs` | **SURFACE** | `AgentCheckpoint`/`ReplayRequest` DTOs. | No snapshot/restore/resume/fork, no seeds (so K2 deterministic replay is unprovable). §4.13 "three operations, one mechanism" unimplemented. | partial |
| `projection.rs` (`BasicProjectionEngine`) | **REAL** | A genuine event-fold reducer: iterates events, maps `agent.phase`→`Phase`, folds intent/plan/error payloads into transcript/plan/errors. Most substantive non-trivial module besides dag. | **Latent bug:** phase parsing round-trips `{:?}`-formatted (PascalCase) names but the parser expects snake_case → phase mapping silently no-ops in practice. Doesn't fold most payload kinds. | partial+ |
| `runtime_client.rs` (`KernelRuntimeClient`) | **BOUNDARY-STUB (clean seam)** | Real wiring: holds `Arc<dyn Router>` + `Arc<dyn InferenceClient>` from hawking-orch; delegates `route()`/`generate()`. Cross-crate types verified to exist. The cleanest part of the crate. | **Not called by the driver** (the FSM never generates). Seam itself is correct. | **Y** (seam) |
| `session.rs` (`SessionProjection`) | PARTIAL | View-model consumed by projection.rs. | No behavior of its own (correct — it's the fold target). | partial |

**Cargo deps:** real & used — `hide-core` (heavy), `hawking-orch` (router+inference in runtime_client), `futures`, `serde`/`serde_json`, `tokio` (test). **Declared but UNUSED:** `hawking-context` and `hawking-index` (zero refs — the context/memory binding the bible requires is absent), `parking_lot`, `thiserror`, `tracing` (no instrumentation despite K11).

**To complete this crate:** (1) Make schemas normative — `acceptance` on `PlanStep`, `budget` on `Plan`, `OracleClass`+`cost_hint` on `Oracle`, `failures`/`artifacts` on `Verdict`, full A.5 `Budget`, `stack`/`replan_count`/`context_manifest` on `AgentState`. (2) Wire the real loop in `driver.rs` — call `Planner::synthesize`, gate `Plan` on `dag.acyclic()`, in `Verify` actually run `step.acceptance.oracles` and call `VerificationGate::decide`, implement Repair/Replan/Paused arms with budgets, add `Mode::{Live,Replay}`. (3) Build the Governor (`check()` over all caps, ledger tracking replans/tool_calls/edits/wallclock, autonomy→Paused, interrupt polling). (4) Implement the deterministic oracle suite (even shelling to `cargo`/`git apply` makes K1 real) + consistency vote + gated llm_judge. (5) Implement `best_of_n` (fork worktrees, run oracles per candidate, select) + `pick_tier`. (6) Give replan/subagent/skills/checkpoint real behavior. (7) Call `KernelRuntimeClient` from `Act`; import the unused context/index crates. (8) Fix the projection phase-string bug; drop/use dead deps.

---

### 2.3 `hide-tools` → bible ch.03 (tool system)

**A thin, faithful vertical slice — ~6 of ~80 catalog tools.** 981 lines / 6 files. FS + shell-run are genuinely working (real I/O, real process spawn, real `simulate()`). The moat features — tiered edit applier, shell **sandbox**, real MCP — are absent.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | REAL | Module surface + `register_builtin_tools`. | — | Y |
| `fs.rs` | **REAL** (3 of 5) | `fs.read`/`fs.list`/`fs.write` — real `std::fs`, UTF-8, cap check, correct `simulate()` → Read/Write `EffectSet`, `Purity::PureFs`, 2 dispatcher tests. | No `fs.stat`/`fs.glob`/`fs.watch`. `fs.read` has no `range`/`encoding` args; over-cap **hard-errors** instead of spilling to `bytes_ref` with a head preview (TT5/§4.5). No `.gitignore` respect. | partial |
| `git.rs` | **PARTIAL/SURFACE** (1 of ~10) | `git.status` only — shells out to `git status --short --branch`, cap truncation, `stdin(null)`. | **9 of 10 missing** incl. the **`git.worktree.*` trio** (the agent-isolation primitive, §4.6.6/§4.9.3). **Non-zero exit → `ToolStatus::ToolError`** — violates the `EXEC_NONZERO`-is-data discipline (§4.2.3, F15). | partial |
| `shell.rs` | **PARTIAL** (run) + **SURFACE** (plan) | `shell.run` — real spawn, **argv-form** (injection-safe ✓), cwd/env, stdin(null), cap truncation, `Purity::Impure`. `shell.plan` validates+describes. | **NO SANDBOX** (the bible's #1 default, §4.8): unconfined host `Command`, no Seatbelt/bubblewrap, no network-deny, **no timeout enforcement** (`timeout_ms` ignored, no watchdog). `shell.plan` self-admits it ("shell.run must be sandbox-wired separately"). Non-zero exit → ToolError (same F15 violation). | partial → **N on sandbox** |
| `mcp.rs` | **SURFACE** | Two pure functions: `mcp_tool_to_hide_spec` + `hide_result_to_mcp` (gets the critical `isError = status != Ok` projection right). | **No MCP protocol** — no JSON-RPC, no `initialize`/`tools/list`/`tools/call`, no stdio/HTTP transport, no `2025-11-25` handshake. `McpTransport` enum is dead data. Not a clean boundary-stub — a type sketch with no engine. Not registered. | **N** |
| `registry.rs` | REAL | Registers all 6 tools into hide-core's registry. | No MCP tool registered; no discovery/hot-reload. | Y |

**Cargo deps:** `hide-core`, `futures`, `serde`/`serde_json`, `thiserror`; `tokio` is **dev-only** — so all tools spawn **blocking** `std::process::Command` inside async fns (will block the executor). **Missing:** `git2`, runtime `tokio`/`tokio::process`, any MCP/JSON-RPC crate, `reqwest`, sandbox crate (`nix`/`libc`), `globset`/`walkdir`/`ignore`, `similar`/`diffy`.

**To complete this crate:** (1) Fix `EXEC_NONZERO` in `shell.run`+`git.status` (non-zero exit = `Ok` + `exit_code`, or the agent misreads diagnostics). (2) Wire the shell sandbox (move to `tokio::process`, add `tokio::time::timeout` + SIGTERM→SIGKILL ladder, Seatbelt/bubblewrap + network-deny). (3) Build the edit family (`search_replace`/`apply_patch` + AST tier, §4.7). (4) Implement real MCP (JSON-RPC over stdio + HTTP, carry annotations as untrusted). (5) Round out the catalog (`fs.stat/glob/watch`, `git.diff/log/commit/worktree.*`, `search.*`, `test.run`/`build.run`). (6) `bytes_ref` spill for over-cap output. Several need the hide-core `ToolError`/`ToolResult` enrichment — coordinate upstream.

---

### 2.4 `hawking-context` → bible ch.04 (context + memory)

**Thin but architecturally honest.** 621 lines / 8 files. Right type vocabulary, clean seams, but every algorithm the bible treats as substance is reduced to its most trivial form.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | REAL (trivial) | Module decls + re-exports. | Doesn't re-export memory/kv/profiles/budget at root. | Y |
| `budget.rs` | **SURFACE** | `TokenBudget` w/ `available_input()`; `RegionBudget`; `estimate_tokens`. | **No real allocation** — `RegionBudget` never read by compiler (no per-region reservations). `estimate_tokens = (chars+3)/4` — **not tokenizer-accurate** despite §4.2 mandate. No reserve-then-fill. | partial |
| `compiler.rs` | **SURFACE** | `ContextCompiler` (add_source/compile); `ContextSource` trait (name+gather); sorts by score desc, tie-break on id; greedy fill; builds manifest with retained+dropped. 2 tests. | **Naive concatenation, not a knapsack:** `if used + token_count <= capacity { parts.push(text) }` then `join("\n\n")`. No value-density sort, no reservations, no pinning, no redundancy penalty, no degrade ladder, no head/tail ordering (LITM), no `realize()`/`degrade()` split (eager bodies). `ContextSource` is 2 methods vs the bible's `kind/candidates/realize/degrade`. No scorer. | partial |
| `manifest.rs` | **PARTIAL** | `ContextManifest`/`ContextSpan`/`ContextSourceKind`/`DropReason`; serde + versioned. | Missing A.1: `turn_id/session_id`, `profile`/`model`/`budget` blocks, per-span `order_index/value/signals/pin/banked/compacted_from`, **`conflicts[]`**, **`kv{}`**, **`compaction_events[]`**. Span ids are `"name:idx"` — **not blake3 content-addressed**. | partial |
| `memory.rs` | **SURFACE/PARTIAL** | `MemoryRecord`/`MemoryKind`/`MemoryQuery`/`RankedMemory`/`MemoryStore` trait; `InMemoryMemoryStore` (BTreeMap) with real filter+score+sort. | **Not the bible's store** — bible mandates SQLite+FTS5+sqlite-vec at `.hide/memory/memory.db`; this is RAM-only. Scoring is `lexical_overlap + importance` (no recency, no embeddings/cosine). API is `put/query` not `retrieve/upsert/supersede/pin` (A.2). Record missing `embedding_ref/decay/links/supersedes/pinned/version`. | partial |
| `profiles.rs` | **SURFACE** | `ContextProfile` + `coding_default()` (3 region budgets). | Missing most of A.3: `position_policy/working_set_mode/eviction/kv_precision/source_weights/recency_half_life/compaction/retrieval_k`. No Tight/Standard/Long/Unbounded presets. The pin/order bools exist but **nothing reads them**. | partial |
| `sources.rs` | **PARTIAL/REAL seam** | `StaticContextSource` + `CodeIndexContextSource` — a **genuine cross-crate seam** calling `hawking_index::CodeIndex::search` and mapping results with `path:line` provenance. Test wires index→compiler. | Hardcodes `include_semantic: false`. No Memory/Plan/ToolOutput/Scratchpad/Diagnostics/System sources (8 mandated, 2 exist). Provenance is blanket `trusted("code-index")` — doesn't propagate trust/confidence (F12). | partial |
| `kv.rs` | **BOUNDARY-STUB (clean, minimal)** | `KvHandle`/`KvTier`/`KvCheckpoint`/`KvStoreClient` trait (`lookup_prefix`,`checkpoint`). Doesn't fake a KV map. | **No implementation.** Named `KvStoreClient` not `KvStore` (A.4); 2 read-only methods vs A.4's 8 (`warm_into_slot/demote/set_policy/restore/list_checkpoints/stats`). `key: String` opaque — **no `PrefixKey` byte-compatible with in-tree `prefix_cache::PrefixKey`/`prefill_disk::PrefillKey`** (the bible's explicit interop requirement). | partial |

**Cargo deps:** `hide-core`, `hawking-index` (real seam), `futures`, `parking_lot`, `serde`/`serde_json`. **Missing — and the absences are the diagnosis:** no tokenizer (`tokenizers`/`tiktoken`), no embeddings/vector dep, no `rusqlite`/FTS5, no `blake3`, no HTTP client (`reqwest`), no `ulid`.

**To complete this crate:** Add a tokenizer; rewrite `compile()` into the §4.2.3 pipeline (reservations first, `candidates()`/`realize()`/`degrade()` split, real scorer with recency/importance/relevance via `/v1/embeddings` minus redundancy + pins, value-density greedy + local-improvement, head/tail ordering). Promote manifest to full A.1 (blake3 CAS, signals/pin/banked, kv/conflicts/compaction blocks). Replace `InMemoryMemoryStore` with SQLite(FTS5+sqlite-vec) at `.hide/memory/memory.db` with `retrieve/upsert/supersede/pin`. Expand `ContextProfile` to A.3 + 4 presets. Rename `KvStoreClient`→`KvStore`, give it the 8-method surface + a `PrefixKey` asserted byte-compatible with `hawking-core`, back it with a `reqwest` client to `hawking-serve`. Add the missing source modules + real trust/confidence.

---

### 2.5 `hawking-index` → bible ch.05 (codebase intelligence)

**A contracts-and-reference-stub scaffold — no parser, no hasher, no vectors, no daemon.** 503 lines / 7 files. Its own lib.rs admits "in-memory reference implementation… backends can slot in behind the same traits." Only `InMemoryCodeIndex` works (naive lexical + a prefix-scanner "parser").

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | REAL (façade) | Module decls + re-export of `CodeIndex`/`InMemoryCodeIndex`/`SearchQuery`/`SearchResult`. Honest doc. | Surface is `query::*` only — none of the §4.11 `Index` trait exported (none exists). | partial |
| `query.rs` (330 ln) | **PARTIAL** | The only working code. `InMemoryCodeIndex` (RwLock BTreeMaps), **real lexical search** (lowercased substring `.find()` + hand-rolled `lexical_score`), `definition`/`references` filter by role, `index_path` reads from disk, 2 tests. | **Symbol "parsing" is `simple_definition` — a line-prefix scanner** matching `"pub fn "`/`"struct "`/etc.; no tree-sitter, no scopes/qualified names. **Every occurrence is hardcoded `role:"definition"`** → `references()` **always returns empty** (the reverse-reference/blast-radius moat is unreachable). `include_semantic` is **never read** (Leg C no-ops). No RRF, no graph leg, no rerank, no `min_generation`/`precise`/provenance. `score` hardcoded `1.0`. `health()` hardcodes `stale_files:0`. O(files×lines) per query. | partial |
| `graph.rs` | **SURFACE** | serde `Symbol`/`Occurrence`/`GraphEdge`/`EdgeKind` (Defines/References/Calls/Imports/Implements/Tests/Dataflow/Performance) + `RepoMap` types. Reasonable shape. | **Pure types — zero behavior.** No graph ever built, no `petgraph`, no PageRank, no `render_elided`. No SCIP string-IDs (uses `path::name`). | partial (types) |
| `merkle.rs` | **SURFACE/BOUNDARY-STUB** | `MerkleNode`/`ChangeSet` + `MerkleScanner` trait (`scan_workspace`/`diff`). Clean seam shape. | **No implementation** (`impl MerkleScanner` → none). **Zero hashing** — no `blake3`/`sha2`; `hash: String` never computed. No O(changed) diff, no rename-by-identical-hash (§4.8). The biggest incremental-indexing win is a trait with no body. | partial (trait) |
| `semantic.rs` | **SURFACE** | `EmbeddingRecord`/`HybridRetrievalWeights` + **one real fn**: `reciprocal_rank_fusion(ranks,k)` = Σ1/(k+rank). | **No embeddings, no ANN** — no HTTP client to `/v1/embeddings`, no Lance/usearch/sqlite-vec, no chunking. `vector` never populated. RRF + weights are **dead** (search() never fuses). Generates nothing — not even placeholder vectors. | partial (one unwired helper) |
| `daemon.rs` | **SURFACE** | `IndexDaemonConfig`/`IndexDaemonState` (debounce/idle/concurrency knobs). | **No daemon** — no `notify`/watcher, no scheduler, no idle/GPU detection, no incremental pipeline, no atomic manifest swap, no crash recovery, no `hawking-indexd` binary. Inert config no loop reads. | partial (config) |
| `store.rs` | **SURFACE** | `IndexStoreConfig`/`StoreGeneration` (path strings). | **No store** — no SQLite/WAL/FTS5, no schema, no Lance, no CAS, no segments/MANIFEST, no MVCC. `manifest_hash` never computed. Nothing persists. | partial (config) |

**Cargo deps:** `hide-core`, `futures`, `parking_lot`, `serde`/`serde_json` — plumbing only. **Absent (confirmed absent workspace-wide):** `tree-sitter` (bible: "the bedrock parse layer… Non-negotiable"), any vector store, `blake3`, `rusqlite`/`tantivy`/FTS5, any LSP crate, `notify`/`ignore`, `petgraph`, `reqwest`. **The crate fakes codebase intelligence with no real parsing/embedding deps.**

**To complete this crate** (greenfield behind clean seams): (1) `blake3` + real `MerkleScanner` (leaf+dir hash, O(changed) diff, rename detection) — the gate for all incremental work. (2) `tree-sitter` + grammars, replace `simple_definition` with real tag extraction (both def and ref roles, SCIP string IDs). (3) `rusqlite` (WAL + FTS5 + §4.10 schema with materialized reverse edges) for durable, index-backed queries. (4) `petgraph` + call/import/type graphs + PageRank repo-map. (5) wire `semantic.rs` (reqwest→`/v1/embeddings`, cAST chunking, a vector store, actually call RRF + rerank in `search()`). (6) build the daemon into `hawking-indexd` (notify+debouncer, scheduler, generation/manifest MVCC, crash recovery). (7) expand `CodeIndex` to the full §4.11 `Index` trait.

---

### 2.6 `hawking-orch` → bible ch.06 (model inference orchestration)

**Narrow but mostly real — ~40% of ch.06.** 740 lines / 9 files. Real registry + router; a real-but-primitive hand-rolled HTTP+SSE client (clean `InferenceClient` seam). Missing: the escalation cascade (the chapter's thesis) and 4 mandated modules.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `registry.rs` | **REAL** | `RoleRegistry` (RwLock map); register/get/by_purpose/all; `default_hawking_local_roles()` builds 4 concrete roles (fast-draft/hero-coder/embedder/tool-planner) w/ real caps+samplers; test. | No `.hide/roles.toml` load; missing `endpoint`/`cost`/`escalates_to`/`quant`/`draft_for`/`adapters` fields; no SsmLong/Reranker defaults. | partial |
| `router.rs` | **REAL** | `SimpleRouter` truly routes: embed→Embedder, grammar→ToolPlanner, `difficulty>0.65`→HeroCoder else FastDraft w/ fallback; emits `RouteDecision`; 2 tests prove selection. | **No cascade/escalation execution** (§4.4 step 5 — the confidence-gated retry-up loop, the chapter's spine, absent); no architecture routing (SSM/LongWatch); no budget admission; no spec-decode. `provider` hardcoded. | partial |
| `difficulty.rs` | **PARTIAL** (real heuristic) | Genuinely estimates: `min(chars/12000, 0.45)` + 0.12/keyword over a 5-word set, capped, w/ breakdown. Not constant. | Crude (char-len + keyword bag). No first-token entropy (the "unfair local angle" — needs logprobs, absent). | partial |
| `http_client.rs` (221 ln) | **BOUNDARY-STUB — real primitive client, clean seam** | **Not canned:** opens a real `TcpStream`, writes a hand-built `POST /v1/hawking/generate HTTP/1.1` w/ `Accept: text/event-stream`, parses **real SSE** (`data:`/`[DONE]`/`{"text","tok_index"}`→Token, stats→`GenerationStats`). Implements the swappable `InferenceClient` trait. Tests cover parser+body. | Hand-rolled HTTP over **blocking** `TcpStream` in an `async` fn (blocks executor); `read_to_string` buffers the **entire** stream before parsing (**defeats streaming**); http-only; no chunked-encoding. **Targets only `/v1/hawking/generate`** — the mandated `/v1/chat/completions` and `/v1/embeddings` (embedder role!) are unimplemented. (Note: the target endpoint *does* exist in `hawking-serve`, so the seam is real.) | partial |
| `inference.rs` | **REAL** (trait) + BOUNDARY-STUB (impl) | Clean `InferenceClient` seam; `StubInferenceClient` (canned token, legit test double). | No `embed()` on the trait (embedder role can't be driven through it). | Y (trait) |
| `sampler.rs` | **SURFACE** | `SamplerCatalog{edit,planning,brainstorm}` presets. | Pure data, **never referenced** by router. Bible §4.6 superset fields (`min_p/typical_p/dry/logit_bias/stop`) absent from `SamplerProfile`. | partial |
| `grammar.rs` | **SURFACE/fake** | `GrammarCompiler` trait + `StubGrammarCompiler`. | The "compiler" **fabricates** hashes (`format!("stub:{}:{}", name, len)`); no schema parsing, no FSM/matcher, no `mask_logits`. No `GrammarSpec` enum, no `GrammarMatcher`. Router attaches `grammar: Some(...)` but nothing enforces it. | partial (types) |
| `supervisor.rs` | **SURFACE** | `RuntimeLock`/`RuntimeSupervisorStatus` + `down()`. | **Zero supervision logic** — no spawn/kill/health-poll/restart/backoff. A status DTO, not a supervisor. | partial (types) |
| `lib.rs` | REAL (wiring) | Declares 8 modules; re-exports registry+router. Honest "HTTP/interface only." | Doesn't surface escalation/confidence/adapters/scheduler (don't exist). | n/a |

**MISSING modules (bible-mandated, no file):** `escalation.rs` (§4.4 cascade), `confidence.rs` (§4.7 entropy/self-consistency), `adapters.rs` (§4.9 LoRA), `scheduler.rs` (§4.11 energy/thermal/RAM admission). No spec-decode planning (§4.8).

**Cargo deps:** `hide-core`, `futures`, `parking_lot`, `serde`/`serde_json`; `tokio` dev-only. **MISSING (headline):** **no HTTP client crate at all** — no `reqwest`/`hyper`/`ureq`, **no SSE parser** (`eventsource-stream`). Confirmed absent crate-wide and workspace-wide. The runtime seam is `std::net::TcpStream` + hand-rolled HTTP/1.1.

**Divergences worth flagging:** `RolePurpose` (hide-core) vs bible `RoleKind` — `ToolPlanner` invented, `SsmLong`+`Custom` missing (no RWKV-7/Mamba-2 long-ctx routing). `ModelRole` lacks `escalates_to` so the cascade graph can't be expressed. **Nothing wires router→client** (no endpoint resolution → generate call).

**To complete this crate:** Add `reqwest`+`reqwest-eventsource`; rewrite `http_client.rs` to stream incrementally + speak `/v1/chat/completions` and `/v1/embeddings` (add `embed()` to the seam). Extend hide-core contracts (A.1): `endpoint`/`cost`/`escalates_to` on `ModelRole`, `SsmLong`/`Custom` on the enum, `speculative`/`draft_for`/`adapters` on `ProviderCaps`. Build the 4 missing modules — especially `escalation.rs` (the §4.4 step-5 cascade executor, the crate's reason to exist) and `confidence.rs` (start with [SHELL-TODAY] self-consistency voting, no runtime hook needed). Replace `StubGrammarCompiler` with the validate-and-retry fallback + real `GrammarSpec`/`GrammarMatcher`. Wire an executor that resolves `RouteDecision.role_id`→endpoint→`InferenceClient`. Load `.hide/roles.toml`.

---

### 2.7 `hawking-research` → bible ch.08 (research & knowledge lab)

**Type-complete, logic-shallow — ~15% of the chapter; no model/runtime integration anywhere.** 856 lines / 9 files. The FSM wires `search→fetch→ingest→verify` with a passing test, but every step requiring intelligence or I/O is faked.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | REAL (trivial) | Module decls + re-exports. Honest doc. | — | Y |
| `ingest.rs` | **SURFACE/BOUNDARY-STUB** | `SourceAdapter` trait (search/fetch); rich `SourceType` enum (Arxiv/OpenAlex/PdfLocal/…); `StructuredDoc`/`DocSection`. `InMemorySourceAdapter` does substring search + quality sort. | **No real adapter** — zero arXiv/OpenAlex/PDF/HTML code, no `reqwest`, no PDF parser. Trait misses bible's `kind/can_handle/resolve/parse/metadata` + `IngestCtx`. `content_hash` **never populated** → no content-addressing (Tenet 7). §4.5 (PDF/equation/table/OCR) 100% absent. | partial |
| `kg.rs` | **SURFACE** | Full §4.2 node/edge kind enums + `ConfidenceTier`/`ProvenanceSpan`; `KnowledgeGraph` trait + `InMemoryKnowledgeGraph` (add_node/edge, nodes_by_kind, edges_from); `ingest_doc_shell` mints Paper + 1 Claim/section. | **Not a knowledge graph** — query API is 2 filters; **no GraphRAG Local/Global/Path/Hybrid/Cypher** (§4.8), no traversal, no community detection, no entity resolution. "Extraction" = `section.text.take(160)`. Every claim hardcoded `Extracted`. No content-addressed IDs, no KùzuDB, no persistence (RAM, lost on exit). | partial (schema Y; behavior N) |
| `verify.rs` | **SURFACE** | `ClaimVerification`/`ClaimStatus`; `verify` counts "supporting" via `lexical_overlap>0.4`, "refuting" via `contains("increase")&&contains("decrease")`. | **Not adversarial verification** — no independence test, no `MIN_CORROBORATION`, no targeted re-search, **no citation re-verification against CAS content-hash** (the #1 anti-hallucination guard, §4.7.3 — impossible since no CAS/hashes exist). Refutation = 2 hardcoded antonym pairs. | **N** |
| `pipeline.rs` | **PARTIAL** | Best file. 8-state FSM (PlanScope→FanOut→Fetch→Read→Verify→Synthesize→Complete); `run_once` drives to completion; `Fetch` really iterates adapters+ingests, `Verify` runs the verifier; passing `#[tokio::test]`. | **Most states are no-op `state = next`**: PlanScope/Read/Synthesize do nothing. No query decomposition, no fan-out parallelism, no Triage/dedup, no Reflect loop, no budget, no checkpoint/resume. **`Synthesize` produces no report** (the chapter's headline output). No `RuntimeClient`. | partial |
| `run_ledger.rs` | **REAL** (narrow) | `ResearchLedger` trait + `InMemory`/`Jsonl` (real file append, sync, line-parse w/ context, roundtrip test). | Records the run *summary* as one JSON line — **not the per-event resumable checkpoint journal** (§4.6 "resume from last checkpoint without re-fetching"). No `env_hash`/`code_rev`/`seed`. | partial |
| `experiments.rs` | **SURFACE** | `Hypothesis`/`ExperimentRun` + ctor. | Pure data, **zero logic** — no runner, no pre-registration, no `code_rev`/`env_hash`/`seed` (§4.11), no metrics→Claim feedback. Not wired into pipeline/KG. | partial |
| `litmap.rs` | **SURFACE** | `LiteratureMap`/`Cluster`/`Gap` structs. | **No functions** — none of the 6 §4.9 workflows (build/compare/timeline/gaps/influence/consensus). | **N** (types) |
| `bridge.rs` | **SURFACE** | `claim_to_issue` + `node_to_memory` (real cross-crate types); `CodeResearchLink`. | Below §4.10/§4.13: `FindingIssue` lacks `IssueDraft` (acceptance criteria, `linked_symbols`, effort, `ExperimentSpec`); no `EquationBridge`, no `IssueSink`, no `MINTED_ISSUE`/`IMPLEMENTS` edges, no memory dedup/back-link. `CodeResearchLink` never used. | partial |

**Cargo deps:** `hide-core`, `hawking-context`, `hawking-index`, `futures`, `parking_lot`, `serde`/`serde_json`. **All external capability faked — absent:** PDF parser, HTTP client, graph DB (`kuzu`/`petgraph`), CAS/hashing in *this* crate, **any model/runtime client** (the `RuntimeClient` seam, §4.1 "the only way the lab talks to a model," is not a dep/trait/usage), `strsim`.

**To complete this crate:** Land the **`RuntimeClient` trait first** (embed/chat/chat_stream) — every intelligent step depends on it. Add CAS/content-addressing (`sha2` + wire `hide-core::BlobStore`) so node IDs are content-addressed and `content_hash` is populated (unlocks idempotent ingest + citation re-verification). Swap the `BTreeMap` KG for an embedded graph store (KùzuDB) + §4.8 query modes + entity resolution. Build one real `SourceAdapter` (arXiv/OpenAlex + a PDFium text-layer parse). Flesh the empty FSM states (planner, Triage, a real cited-report `Synthesize`, Persist/Reflect + per-event checkpoint ledger). Give `verify.rs` independence/corroboration logic; turn litmap/experiments/bridge into the §4.9–4.11 workflows.

---

### 2.8 `hide-fleet` → bible ch.09 (parallel agents & workstation)

**Models the fabric's decisions but cannot execute the fabric — no concurrency, no git worktrees, no merge engine.** 532 lines / 8 files. Three pure algorithms are real (DAG ready-set, admission predicate, port allocator, tournament selector); everything that touches the world is absent. **Not referenced by any other crate.**

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | SURFACE | Module decls + 2 re-exports. | No `FleetManager` (§4.1's mandated entry orchestrator); no `fleetview`. Nothing composes the parts. | partial |
| `queue.rs` (150 ln) | **REAL** (in-mem) / PARTIAL | `JobGraph` (RwLock map); enqueue/get/set_status; `ready_jobs()` does real dependency-gating + priority-then-FIFO; `has_cycle()` real DFS. | A **sync container, not a dispatcher** — nothing dequeues/launches/runs concurrently. No durability / event-log projection (§4.5 "the queue is a projection of the event log", P8). `AgentJob` omits ~10 A.1 fields (`kind/parent_job/base_ref/isolation/concurrency_class/attempts/result_ref/schedule/schema_version`). `JobStatus` swaps in non-spec `Ready`, drops `Admitted/Preempted/Merging`. | partial |
| `scheduler.rs` (118 ln) | **REAL** (logic) / SURFACE (as "scheduler") | `FleetGovernor::admit()` real: thermal + free-mem-headroom + slot-saturation → structured `AdmissionDecision`; test. `Scheduler::next()` returns first admissible ready job. | `next()` only **picks** — never spawns/runs/preempts. **No `schedule_tick` loop, no preemption/`checkpoint_and_yield`** (§4.6.3, the moat), no EWMA spawn-rate **circuit-breaker** (§4.6.4), no two-pool Model-vs-CpuOnly split (§4.5.1, "load-bearing"). `ResourceSnapshot` is caller-fed — no `ResourceProbe` reads real RAM/`/metrics`/`dec_tps`; thermal proxy nobody populates. | partial |
| `isolate.rs` (63 ln) | **REAL** (PortAllocator) / **BOUNDARY-STUB, unwired** (worktree) | `PortAllocator::lease/release` — real disjoint-range allocator. `WorktreeLease` references real `hide_core::security::SandboxProfile`. | **`WorktreeLease` is a PathBuf-holding struct — NO code creates a worktree.** No `git worktree add`, no `std::process::Command`, no remove/prune, no env namespacing. The `isolate_run`/`release_run` lifecycle (§4.3) is unimplemented. **The single biggest gap vs the chapter's load-bearing primitive** (worktree-per-run). | partial (types) / **N** (behavior) |
| `merge.rs` (57 ln) | **SURFACE** | `CandidatePatch`/`MergeStrategy`; `TournamentSelector::select()` does real oracle-first filter+sort → winner or RejectAll. | **No merge happens** — no 3-way/AST merge, no integration-branch funnel (§4.4.1), no conflict ladder (§4.4.3). `conflicts` always empty; `diff_hash`/`changed_files` caller-fed, never computed. `ThreeWay`/`Structured`/`ManualReview` variants are dead. | partial (selector) / **N** (engine) |
| `patterns.rs` (42 ln) | **REAL** (decision) / SURFACE (patterns) | All 7 `OrchestrationPattern` variants; `choose_pattern(...)` correctly transcribes the §4.2.2 selection rule. | The **rule** is implemented; the **patterns are not** — no `Pattern` trait, no fan_out→runs→reduce execution. Not wired to any executor. | partial |
| `remote.rs` (61 ln) | **SURFACE** (protocol types) | `RemoteRequest`/`RemoteUpdate` serde-tagged enums w/ real hide-core types; `RemoteAuthPolicy` (deny-first). | **No server, no transport, no session, no auth handshake** — no WebSocket/JSON-RPC framing (§4.9.2 mandates JSON-RPC 2.0), no `from_seq` replay, no pairing/token. `RemoteAuthPolicy` is inert config. | partial / **N** (wire) |
| `batch.rs` (25 ln) | **SURFACE** | `BatchJob`/`BatchSchedule`/`WakeReport` types. | **Pure data, zero behavior** — no schedule-gate eval (idle/ac_power/cron), no DAG drain, no checkpoint/resume across reboot (§4.7, P8), no wake-report assembly. Thinner than A.3. | partial |
| `cluster/`, `fleetview.rs` | **MISSING** | — | §4.1 mandates `cluster/pool.rs` (TIER-4, optional/defensible) + `fleetview.rs` (live projection, P12 "observability is mandatory" — **not** optional). Neither exists. | **N** |
| event kinds (A.5) | **MISSING** | — | No code emits any `job.*`/`merge.*`/`governor.*`/`batch.report` event. | **N** |

**Cargo deps:** used — `hide-core`, `parking_lot`, `serde`/`serde_json`. **Declared but unused:** `futures` (no async), **`hide-kernel`** (the whole point per §4.1 — schedule kernel runs — declared but never imported), `thiserror` (no error enum). `tokio` dev-only. **Missing — and their absence IS the finding:** **no `git2`/`gix`** (→ no worktrees), **no async runtime in deps + no channels** (→ the fleet is structurally sequential, runs nothing in parallel), **no diff/merge crate** (→ no merge), **no WebSocket/JSON-RPC** (→ no remote wire), **no `sysinfo`** (→ resources hand-fed).

**To complete this crate:** (1) Add a git backend (`gix` or `Command` git) + implement the worktree lifecycle + env namespacing in `isolate.rs` — highest leverage. (2) Add `tokio` to deps + build `schedule_tick` with `spawn_run` actually launching `hide-kernel` runs, bounded-admission/unbounded-completion channels, preemption, spawn-rate breaker. (3) Add `similar`(+`tree-sitter`) + the integration-branch funnel + conflict ladder. (4) Make the queue a projection of the event log; emit A.5 events. (5) Reconcile `AgentJob`/`JobStatus` with A.1 (esp. `concurrency_class` + two-pool split). (6) Build `remote.rs` into a JSON-RPC-over-WS server with session persistence + `from_seq` replay. (7) Add `FleetManager` + `fleetview`. Wire the crate into a consumer.

---

### 2.9 `hide-security` → bible ch.10 (local-first infra & security)

**Security as data structures, not enforcement — one real module (audit), no crypto dep at all.** 377 lines / 5 files. The audit hash-chain is real (but uses the wrong hash); redaction is a toy; sandbox is a clean boundary-stub; storage is cosmetic. **Not referenced by any other crate.**

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `audit.rs` (189 ln) | **REAL** (algo-divergent) | `compute_event_chain`/`verify_event_chain` over `&[Event]`: canonical re-serialize (null chain_hash), rolling `H(prev‖bytes)`, embedded-hash verification w/ mismatch+missing branches, `chain_root`=tip, `IntegrityReport` via the `EventLogIntegrity` trait. 2 tests incl. tamper-rejection. | Uses **`Sha256`** — bible §4.2.1/§4.11 mandate **`blake3`**. No genesis salt (uses zero prev). **No signed `ANCHORS`** (§4.11). No segment-format awareness, no compaction re-link. **Duplicates** `hide_core::event::compute_chain_hash` (also SHA-256). | partial |
| `redaction.rs` (85 ln) | **SURFACE** | `Redactor` w/ `SecretPattern{prefix,min_len}`; `redact()` splits whitespace, masks tokens by prefix+len → `<REDACTED>`, counts. | **No regex, no entropy detector** — only TWO prefixes (`sk-`, `ghp_`). Bible §4.8 mandates AWS keys/PATs/PEM/JWTs + Keychain-fingerprint. Marker is `<REDACTED>` not `«redacted:<detector>»`. Misses anything without these prefixes or containing whitespace. No `Event.redactions` wiring. | partial (toy) |
| `sandbox.rs` (58 ln) | **BOUNDARY-STUB (clean)** | `render_macos_seatbelt(&SandboxProfile)` emits real SBPL (`(version 1)(deny default)`, per-root file-read*/write*, network), `escape()`. `default_workspace_profile()`. Honestly records gaps in `warnings`. | **Deliberately** omits `process-exec` allowlist (warned) — §4.5.2 mandates it. No proxy-port egress, no `.hide/log` write-deny, no per-grant `.sb` emission, no `sandbox-exec` spawn. Clean stub (surfaces gaps rather than faking). | partial (seam honest) |
| `storage.rs` (33 ln) | **N** (cosmetic) | `AtRestPolicy` (flags, all-off default) + `LayoutValidation` structs. Only `fn` is derived `Default`. | **Zero behavior.** **No crypto crate** (AES-GCM/SQLCipher/age/ring) — encryption-at-rest is structurally impossible. No Keychain. `LayoutValidation` has no producer — the S12 0700/`.hide/log`-unwritable check never runs. Default posture is plaintext. Not even re-exported. | **N** |
| `lib.rs` | REAL (honest) | Module decls + re-exports audit+redaction. Doc admits OS enforcement is host-specific. | Doesn't re-export sandbox/storage. | n/a |

**Cargo deps:** `hide-core`, `serde`/`serde_json`, `sha2`, `thiserror` (unused). **Missing — the headline:** **no crypto crate** (→ at-rest impossible), **no `regex`/entropy** (→ redaction is a 2-prefix list), **no `keyring`/`security-framework`**, **no `blake3`** (→ audit uses SHA-256, diverging). The only crypto present (SHA-256) powers the one working module.

**Divergences:** hash algorithm (SHA-256 vs mandated blake3) — a real, load-bearing contradiction shared with `hide-core/event.rs`. No tamper-evidence anchors (chain exists but nothing signs/anchors `chain_root`). Encryption-at-rest is a no-op (policy flags only). Layout/perm validation never runs (S12 unmet).

**To complete this crate:** (1) audit: switch both this and `hide-core/event.rs` to **blake3**, add genesis salt + the ANCHORS story (Keychain-signed `(seq, chain_root, sig)` + `security.anchor`/`integrity_alarm` events); resolve the core/security duplication into one owner. (2) redaction: add `regex` + a real detector suite (AWS/PAT/PEM/JWT/entropy), switch marker to `«redacted:<detector>»`, emit JSON-pointer paths into `Event.redactions`. (3) storage: pull in `aes-gcm`(or sqlcipher/age) + `keyring`, implement WDK + Keychain-wrapped per-segment AEAD + the `LayoutValidation` producer (enforce 0700 + `.hide/log` non-writability, fail-closed). (4) sandbox: render `process-exec` allowlist + `.hide/log` deny + proxy-port egress + per-grant `.sb` + a `sandbox-exec` spawn. Add the crate to a downstream consumer (nothing depends on it).

---

### 2.10 `hide-personalize` → bible ch.11 (bleeding-edge & moonshots)

**Type-schema scaffold — 2 real modules (store, curate), 5 dead deps, 4 of 7 moonshots are pure type sketches.** 567 lines / 10 files. The seams are clean (moonshots are correctly type-only, staged post-shell), but most are *sketches with no executable seam*, not stubs that call an unimplemented backend.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `store.rs` (151 ln) | **REAL** | `PersonalizationStore` trait (append/load_all + default load_recent/by_task); `InMemory` + `Jsonl` (real `OpenOptions` append, `sync_data`, line-parse w/ context). Roundtrip test. | No curated `dataset/vNNN/` layout, no secrets-scrubbing on write (§11.1.1 mandate), no date/ulid path scheme, no egress guard. No SQLite. | partial |
| `curate.rs` (126 ln) | **REAL** | `curate()` implements §11.1.2 rules 1,2,5,6: keep Accepted, gate Modified by rewrite-ratio, dedup on diff, cap, optional latency-drop, DPO pair-building, ~10% held-out split. `CurationPolicy` + test. | Rule 3 (best-of-N) degenerate (pairs any same-`prompt_hash` rejected). Rule 4 (p95×3 outlier) **not** implemented. No recency-weighting. Split is positional. Input is raw `target_diff`, not the reconstructed context manifest the bible requires. | partial |
| `records.rs` (68 ln) | **SURFACE** | `PersonalizationRecord`/`TaskClass`(8)/`Outcome`(4); `accepted()` ctor. | Schema drift: `prompt_hash`/`context_fingerprint` are `String` not `[u8;32]` blake3; `observed_at_ms` not `_us`; `tok_s` optional (bible non-optional). Only the happy-path `accepted()` ctor (no Rejected/Modified/Abandoned). No scrub hook. | partial |
| `rlef.rs` (36 ln) | **SURFACE** | `ExecutionFeedback`/`FeedbackSignal`(7)/`RlefTrajectory`. | **No reward computed from execution.** Only logic is `recompute_reward()` = `feedback.iter().map(\|f\| f.reward).sum()` — the reward is a caller-supplied field; nothing maps a `FeedbackSignal`/oracle result→number. No `RewardConfig` shaping (§11.7.2), no GRPO/PPO, no advantage, no KL, no PPL gate, no daemon. A data envelope + a sum, not RL. | **N** |
| `eval.rs` (40 ln) | **SURFACE** | `EvalCase`/`EvalOracle`(Command/GoldenDiff/Regex/Human)/`EvalResult`; one real predicate `AdapterGateReport::passes()` (the §11.1.4 accept-rate gate). | **No eval ever runs** — nothing executes a Command/Regex/GoldenDiff. No `EvalMiner` daemon, no §11.3 mining types, no SWE-bench, no `--live`. | partial |
| `prompts.rs` (60 ln) | **SURFACE** | `PromptModule`/`PromptVersion`/`OptimizationMetric`; `promote()` enforces `eval_n>=min` + `score>current`. | No DSPy/ADAS/GEPA loop — no candidate gen, no mini-eval, no `prompt.promoted` event. `promote()` is a guarded setter. No `LoopVariant`/ADAS topology types. | partial |
| `retrieval.rs` (30 ln) | **SURFACE** | `RetrievalFeedback`/`LearnedRetrievalWeights` (hand-tuned default). | **No learning, no routing** — no `MetaRouter` trait, no online SGD, no §11.6.2 schema, no ε-greedy. Weights are static; nothing reads/updates them. `hawking-index` declared but unused. | **N** |
| `kv_handoff.rs` (20 ln) | **SURFACE/BOUNDARY-STUB (unwired)** | `AgentHandoff`/`KvHandoff` (opaque `handle`/`provider` strings). | **Does not represent real KV state** — no `KvKey`/`fork_seq`/`KvShareGroup`, no checkpoint/restore, no link to in-tree `copy_kv_prefix_to_slot`, no `kv_seed`. The richer §11.4.2 `AgentHandoff` (typed payload, `seq`, `kv_ref`, `thought_vec`) is downgraded to a session-summary envelope. | partial |
| `world.rs` (15 ln) | **SURFACE** | `SimulationRequest`/`SimulationResult` (flat string vecs). | No `StaticSimulator` trait, no `predict_edit`, no tree-sitter/LSP, no `PredictedOutcome`/`SimulatedIssue`. Diverges from §11.8.2 shape. Tier 1 ("mostly built" per bible) here is just a result struct. | **N** (Tier 1) |
| `lib.rs` (21 ln) | REAL (trivial) | Module decls + re-exports. Honest "scaffold." | — | Y |

**Cargo deps:** real & used — `hide-core`, `serde`/`serde_json`, `parking_lot`. **5 of 9 dead:** `hide-kernel`, `hawking-context`, `hawking-index`, `futures`, `thiserror` (zero imports). The manifest overstates integration.

**To complete this crate:** (store/curate need only finishing: scrub-on-write, dataset layout, curate's p95-outlier + recency rules.) Highest-leverage per §11.10: (1) make `eval.rs` actually run its oracles + add `EvalMiner` (gates DSPy + RLEF). (2) give `rlef.rs` a `RewardConfig` + `FeedbackSignal→reward` mapping so reward is *derived*, then an `RlefDaemon`/PPL-gate seam. (3) define `MetaRouter` in `retrieval.rs`, wire the declared `hawking-index`. (4) reshape `kv_handoff.rs` into the §11.5 `KvShareGroup` protocol with a clean seam to `copy_kv_prefix_to_slot`. (5) reconcile `records.rs` types (`[u8;32]`, `_us`) + all-four-outcome ctors. Use or remove the 4 dead deps.

---

### 2.11 `hide-backend` → bible ch.01 (host) + ch.07 (IDE surface)

**A real in-process composition facade — but NOT the Tauri host.** 1897 lines / 8 files. Genuinely composes all sibling crates, wires intent→event→projection, folds for replay, runs 5 live connectors with tests. But no `tauri` dep, no `#[tauri::command]`, no runtime-sidecar supervision.

| module | classification | what exists | what's missing / shallow | bible match |
|---|---|---|---|---|
| `lib.rs` | REAL (thin) | Module tree + re-exports. Honest doc ("a future Tauri command layer… can instantiate these services"). | Pure glue. | Y (honest) |
| `commands.rs` (166 ln) | **REAL** | `CommandRouter::handle(Intent)` matches **all 11 `Intent` variants**, appends `user.intent.*` event, returns `IntentAck{accepted:true, event_seq}`. Exactly the §4.4 Wire-A shape minus Tauri. | **Not a Tauri command** (plain async fn, no `#[tauri::command]`/`invoke`). `accepted` **always true** — no validation/rejection (§4.4 "each handler validates"). Control intents (Cancel/Pause) are *logged* but **nothing consumes them** to actually cancel/pause. | partial |
| `host.rs` (442 ln) | **PARTIAL** | The composition root. `from_services` wires registry+dispatcher+connectors+kernel+replay. `dispatch_tool` (strongest path): dispatch→append tool.call+tool.result→write projection. `run_command`, `run_agent_to_terminal` (drives kernel.step to terminal), substantive `health()`. | **No process supervision** — never boots/supervises `hawking serve`, no `RuntimeSupervisor`, no `/healthz` poll, no `runtime.lock`, no restart/backoff (§4.3 absent). No runtime HTTP client — `run_agent_to_terminal` drives the FSM but **no model inference backs it**. No Tauri `App`/window/`invoke_handler`. | partial / **MISSING** for §4.3 supervisor + §4.12 tree |
| `connectors.rs` (672 ln) | **REAL** | Biggest, most real. `Connector` trait + registry + **5 working connectors** over real stores: Personalization, Research, Runtime (roles+`route` via SimpleRouter), CodeIndex (add/index/search/def/refs), Context (`compile`→builds compiler over the index). Good error taxonomy. | These are **internal crate facades, NOT external/MCP/network connectors** — no MCP server, no socket, no HTTP; each wraps an in-process sibling. The §7 plugin spine (WASM host, capability negotiator, grant ledger) is absent. `contributions` is **always empty**. | partial (real intra-process; plugin/MCP host MISSING) |
| `replay.rs` (192 ln) | **REAL** | A genuine event fold: `rebuild_session` scans + calls `BasicProjectionEngine::fold` + persists at last seq. `ui_events` maps via a near-exhaustive `EventPayload→UiEventKind` match. | The fold logic lives in hide-kernel; this orchestrates it. **No time-travel** (`ScrubToEvent`/`ForkSession` are logged in commands.rs but never replayed-to-a-point). `ui_events` is poll/scan — **no live `Channel<UiEvent>` push** (§4.4 Wire-B is a pull API, not the ordered push channel). | Y for fold; partial for time-travel/streaming |
| `services.rs` (262 ln) | **REAL** | `BackendServices` (11 wired subsystems); `open()` creates the full `.hide` layout + **durable** stores (Jsonl log, File blob/projection/kv, Jsonl personalize/research, EventChainAuditor). Reopen test. | `session()` returns a **fresh `SessionId` every call** (no registry). `BackendCapabilities::default()` hardcodes everything `true` incl. `fleet`/`remote_protocol` — but `hide-fleet` is never imported and no remote exists → **caps overstate reality**. The durable path doesn't persist the code index. | Y (topology); caps partial |
| `security.rs` (76 ln) | **REAL** | Real bridge: `policy_for_config` builds a concrete `PermissionPolicy` (workspace read/write, git.status, shell.exec + risk + lethal-trifecta gate); `permission_engine`; `render_workspace_sandbox`. Holds Redactor + AtRestPolicy. | The §7.3 capability **negotiator/grant ledger** (interactive per-call grants) is absent — static policy only. `redactor`/`at_rest` not wired into dispatch within this crate. | partial |
| `tools.rs` (69 ln) | REAL (thin) | `build_default_tool_registry`/`build_default_tool_dispatcher` bridges; test confirms registration + policy gating. | Trivially thin (correct — a 2-fn bridge). | Y |

**Cargo deps:** used — `hide-core`(7 files), `hide-kernel`(2), `hide-personalize`(2), `hide-security`(2), `hide-tools`(1), `hawking-context`(1), `hawking-index`(2), `hawking-research`(3); `hawking-orch` (only `RoleRegistry`+`SimpleRouter`). **Dead:** **`hide-fleet`** (used in 0 source files — only the `fleet:true` cap flag). **MISSING:** **`tauri`** (the single most important omission — §4.1 names this the "Tauri 2 host"; confirmed absent crate-wide and workspace-wide), **`tokio` is dev-only** (every method is async; the host doesn't own a runtime — §4.9 mandates a multi-threaded runtime w/ single-writer event task + worker pool), **HTTP client** (no `reqwest`/`hyper` → no runtime client at all). Correctly does NOT depend on `hawking-serve`/`hawking-core` (T5).

**To complete this crate:** (1) Add `tauri`(+shell/pty plugins) + `tokio`/`reqwest` as real deps; wrap `CommandRouter::handle` in `#[tauri::command]` behind `invoke('hide_intent')`; add the validate→reject branch. (2) Replace pull `ui_events` with a push `ipc::Channel<UiEvent>` driven off the broadcast bus, with render-coalescing + bounded backpressure (§4.4). (3) Build the missing `RuntimeSupervisor` (spawn `hawking serve`, poll `/healthz`, Down→Booting→Ready→Degraded→Failed + backoff + `runtime.lock`, + an HTTP `ModelProvider` so the kernel actually generates). (4) Wire `hide-fleet` (or drop the dead dep + set caps false). (5) Stand up the §7 plugin host (wasmtime + negotiator/ledger) and populate connector `contributions`, or rename "connector"→"service facade". (6) Add scrub-to-event/fork-at-event in `replay.rs`. The good news: commands/services/replay/connectors are real and bible-shaped — additive host work, not a rewrite.

---

## 3. Cross-crate dependency order (what to flesh out first)

```
                         ┌─────────────────────────────────────────┐
                         │  hide-core  (Event §4.6, ids ULID,       │   ← TIER 0
                         │  blake3, ToolError/Result, ModelRole)    │      do FIRST
                         └───────────────┬─────────────────────────┘
            ┌───────────────┬────────────┼─────────────┬──────────────┐
            ▼               ▼            ▼             ▼              ▼
     hawking-orch     hawking-index  hawking-context  hide-tools    hide-security  ← TIER 1
   (InferenceClient,  (tree-sitter,  (knapsack,       (sandbox,     (blake3 audit,
    escalation,        merkle/blake3,  tokenizer,       EXEC_NONZERO, crypto, redact)
    /v1 endpoints)     SQLite, vec,    SQLite mem,      edit family,
                       daemon)         KvStore client)  MCP)
            └───────────────┴────────────┬─────────────┴──────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  hide-kernel  (loop,  │                  ← TIER 2 (the brain)
                              │  oracles, governor,   │
                              │  best-of-N, runtime_  │
                              │  client in Act)       │
                              └──────────┬────────────┘
                  ┌──────────────┬───────┼────────┬──────────────┐
                  ▼              ▼       ▼        ▼              ▼
            hide-fleet    hide-security  hawking-research  hide-personalize  ← TIER 3 (leaves)
          (worktrees,    (consumer)     (RuntimeClient,    (eval exec,
           async, merge)                 CAS, graph)        RLEF, router)
                  └──────────────┴───────┬────────┴──────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  hide-backend (tauri, │                  ← TIER 4 (host)
                              │  RuntimeSupervisor,    │
                              │  UiEvent channel)      │
                              └──────────────────────┘
```

**The hard ordering constraints:**

1. **hide-core's `Event`/ids/`ModelRole` must be normative before anything emits events or routes models.** Every crate inherits the envelope; fixing it later means touching every crate's event emission. Fix the closed-enum→flatten-Value, ULID, blake3, and `escalates_to` *first*.
2. **hawking-orch's `InferenceClient` (real HTTP, all 3 endpoints) must exist before hide-kernel's `Act` can generate.** The kernel's `runtime_client.rs` is a clean seam pointed at orch; orch's client is primitive (buffers the whole stream, only 1 endpoint). The kernel cannot do real work until orch can stream.
3. **hawking-index + hawking-context must be real before the kernel can ground a step.** The kernel's per-step context (§4.10) comes from these; today the kernel doesn't even import them. A real loop needs a real index (search/refs) and a real context compiler.
4. **hide-tools' sandbox + edit family + EXEC_NONZERO fix are prerequisites for the kernel's deterministic oracles** (the `build`/`test`/`patch_apply` oracles shell out via tools; if non-zero exit reads as a broken tool, the verifier is poisoned).
5. **hide-kernel is the integration point** — it's Tier 2 because it consumes orch + index + context + tools. Don't flesh it out before they're real or you'll wire against stubs.
6. **The leaf crates (fleet/research/personalize) and the host (backend) are last** — they orchestrate the kernel and tools. `hide-backend`'s `RuntimeSupervisor` is what finally boots `hawking serve`, so the host is the capstone.

---

## 4. Runtime boundary map (where the scaffold wires to live hawking-serve/hawking-core)

The live runtime target **exists and is real**: `crates/hawking-serve/src/http.rs` exposes (via axum, with SSE) `POST /v1/chat/completions`, `POST /v1/embeddings`, and `POST /v1/hawking/generate` (+ `/v1/hawking/tokens`), driving a `hawking_core::Engine` through a continuous-batch scheduler. `hawking-core` carries the `Engine` trait, `GenerateRequest{abort,json_mode,max_stall_ms}`, `SamplingParams{seed}`, `StreamEvent`, `GenStats{dec_tps,draft_*}`, `SpeculateMode`, and `stateful/system_kv_bank.rs` (the KV-cache substrate ch.04 consumes). The scaffold correctly does **not** link these engine crates (T5 — the host talks HTTP only).

| # | Boundary | Where in scaffold | Seam quality | Status |
|---|---|---|---|---|
| B1 | **Inference (generate/stream)** | `hawking-orch/http_client.rs` `HawkingHttpClient` → `/v1/hawking/generate` | **Clean trait** (`InferenceClient`), real SSE, **but** primitive (blocking `TcpStream`, buffers whole stream, http-only, 1 of 3 endpoints) | BOUNDARY-STUB, clean but underbuilt — **rewrite with reqwest, stream, add the other 2 endpoints** |
| B2 | **Embeddings** | (none — `InferenceClient` has no `embed()`) | seam **missing** the method; `/v1/embeddings` exists in serve but no caller | MISSING — add `embed()` to the seam + a client |
| B3 | **OpenAI-compat chat** | (none) | `/v1/chat/completions` exists in serve; orch only speaks the native endpoint | MISSING — add a client path |
| B4 | **Kernel → runtime** | `hide-kernel/runtime_client.rs` `KernelRuntimeClient` (holds `Arc<dyn Router>` + `Arc<dyn InferenceClient>`) | **Cleanest seam in the kernel** — proper swappable traits to orch | Clean, **but the driver never calls it** (Act is stubbed) |
| B5 | **Model routing** | `hawking-orch/router.rs` `SimpleRouter` | Real role selection; **but nothing maps `RouteDecision.role_id`→endpoint→client** (no `escalates_to`/`endpoint`) | Half-wired — router produces decisions nothing executes |
| B6 | **KV-cache handoff / prefix cache** | `hawking-context/kv.rs` `KvStoreClient` (+ `hide-personalize/kv_handoff.rs`) | **Clean** (doesn't fake a KV map) but **unimplemented**; named `KvStoreClient` not `KvStore`; **no `PrefixKey` byte-compatible with in-tree `prefix_cache::PrefixKey`/`prefill_disk::PrefillKey`** (the bible's explicit interop requirement) | BOUNDARY-STUB — sketch of an 8-method contract; **needs a reqwest client to serve + PrefixKey interop** |
| B7 | **Embeddings for index** | `hawking-index/semantic.rs` | RRF helper real but **dead**; **no HTTP client to `/v1/embeddings`** | MISSING — wire reqwest→serve |
| B8 | **Embeddings for memory/context** | `hawking-context/memory.rs` + scorer | **no embeddings dep/client** — lexical-only | MISSING |
| B9 | **Research → model** | `hawking-research` (none) | The mandated `RuntimeClient` (§4.1) is **not a dep/trait/usage** | MISSING — the lab can't talk to a model at all |
| B10 | **Runtime process supervision** | `hide-backend/host.rs` (+ `hawking-orch/supervisor.rs` types) | **No spawn/supervise** of `hawking serve` anywhere; supervisor.rs is a status DTO | MISSING — the host never boots the runtime |
| B11 | **Grammar / constrained decode** | `hawking-orch/grammar.rs` `StubGrammarCompiler` | **Fabricates hashes** — no `GrammarMatcher`/`mask_logits`; serve/core has `json_constrain.rs` to bind to | BOUNDARY-STUB (fake) — needs the validate-and-retry shell fallback at minimum |
| B12 | **Model-cooperation hooks (logprobs/entropy/draft)** | `hide-kernel/cooperate.rs` | Inert DTOs; explicitly `[RUNTIME-SIDE — LATER]`; **no seam (trait/client)** to the runtime | Placeholder — acceptable for now, but add a seam |
| B13 | **OS sandbox (Seatbelt/bubblewrap)** | `hide-security/sandbox.rs` + `hide-tools/shell.rs` | sandbox.rs renders real SBPL (clean stub, gaps warned); **shell.rs never invokes it** | BOUNDARY-STUB — render exists, **no `sandbox-exec` spawn, shell unconfined** |
| B14 | **Git (worktrees)** | `hide-fleet/isolate.rs` + `hide-tools/git.rs` | `WorktreeLease` is a PathBuf; **no `git worktree add`**; git.rs only does `status` | MISSING — the agent-isolation primitive is unbuilt |

**Net seam assessment:** the *kernel-side* seams (B4, B1, B6) are clean trait-based designs — the architecture is right. The problem is (a) the orch HTTP transport is a toy that doesn't actually stream and covers 1 of 3 endpoints, (b) most embedding/KV/grammar seams are unimplemented sketches, (c) the runtime is **never booted** (B10), and (d) the router→client execution path (B5) and the kernel→runtime call (B4) are both unwired so nothing actually flows end-to-end.

---

## 5. Recommended agent wave (parallelizable work packages)

Each package = one crate or a tight cluster. Dependencies noted. Packages within the same wave can run fully in parallel.

### WAVE A — foundation (must land before everything; mostly one package)

- **WP-1 · hide-core contract normalization** *(no deps; BLOCKS all)*
  Reshape `Event` to the §4.6 normative envelope (`payload: serde_json::Value` + `#[serde(flatten)]` + `ext` forward-compat + `cause`/`actor`/`ts`-micros + Action/Observation class), switch ids to ULID, switch chain-hash + blob CAS to **blake3**, enrich `ToolError` (`retriable`/`fix_hint`/`schema_path` + code taxonomy) and `ToolResult` (`provenance`/`exit_code`), add `escalates_to`/`endpoint`/`cost` to `ModelRole` + `SsmLong`/`Custom` to the role enum + `confidence` to `Provenance`, add config layering. *Smallest LOC, highest leverage — do alone, merge before Wave B.*

### WAVE B — runtime seam + the loop's dependencies (parallel; depend only on WP-1)

- **WP-2 · hawking-orch real inference + cascade** *(deps: WP-1)*
  Add `reqwest`+`reqwest-eventsource`; rewrite `http_client.rs` to stream incrementally + speak `/v1/chat/completions` + `/v1/embeddings` (add `embed()` to `InferenceClient`); build `escalation.rs` (the §4.4 step-5 cascade — the crate's reason to exist), `confidence.rs` (self-consistency voting, shell-only), `scheduler.rs`, `adapters.rs`; replace `StubGrammarCompiler` with the validate-and-retry fallback + real `GrammarSpec`/`GrammarMatcher`; wire `RouteDecision.role_id`→endpoint→client.

- **WP-3 · hawking-index real intelligence** *(deps: WP-1)*
  Add `blake3` (real `MerkleScanner`: leaf/dir hash, O(changed) diff, rename detection), `tree-sitter`+grammars (replace `simple_definition`, emit def+ref roles + SCIP IDs), `rusqlite`+FTS5 (durable, index-backed `search`), `petgraph` (call/import graphs + PageRank repo-map), wire `semantic.rs` (reqwest→`/v1/embeddings` + a vector store + actually call RRF/rerank), build `hawking-indexd` daemon (`notify`+scheduler+MVCC). *Largest greenfield package — consider splitting parse+merkle from vectors+daemon into two sub-agents.*

- **WP-4 · hawking-context real compiler + memory + KV** *(deps: WP-1; soft-dep WP-3 for the index seam, already present)*
  Add a tokenizer; rewrite `compile()` into the §4.2.3 knapsack (reservations, candidates/realize/degrade split, real scorer with recency/importance/relevance, value-density greedy + local-improvement, head/tail ordering); promote manifest to full A.1 (blake3 CAS, signals/conflicts/kv blocks); replace `InMemoryMemoryStore` with SQLite(FTS5+sqlite-vec) + `retrieve/upsert/supersede/pin`; expand `ContextProfile` + presets; rename `KvStoreClient`→`KvStore` (8 methods) + a `PrefixKey` byte-compatible with `hawking-core` + a reqwest client.

- **WP-5 · hide-tools sandbox + edit + MCP + EXEC_NONZERO** *(deps: WP-1 for the enriched ToolError/Result)*
  Fix `EXEC_NONZERO` (non-zero exit = `Ok`+`exit_code`) in shell+git; wire the shell sandbox (`tokio::process` + timeout + SIGTERM→SIGKILL + Seatbelt/bubblewrap + network-deny); build the edit family (`search_replace`/`apply_patch` + AST); implement real MCP (JSON-RPC over stdio+HTTP); round out the catalog (`fs.stat/glob/watch`, `git.diff/commit/worktree.*`, `search.*`, `test.run`/`build.run`); `bytes_ref` spill.

- **WP-6 · hide-security crypto + redaction + sandbox-exec** *(deps: WP-1 for blake3 alignment)*
  Switch audit to **blake3** + genesis salt + signed ANCHORS (resolve the core/security duplication); add `regex`+entropy redaction (`«redacted:<detector>»` + `Event.redactions`); add `aes-gcm`/`keyring` for at-rest AEAD + the `LayoutValidation` producer (0700 + `.hide/log` fail-closed); render `process-exec` allowlist + a `sandbox-exec` spawn. *Smaller; can pair with WP-5 (both touch sandboxing).*

### WAVE C — the brain (depends on Wave B being real)

- **WP-7 · hide-kernel real agent loop** *(deps: WP-1, WP-2, WP-3, WP-4, WP-5)*
  Make schemas normative (`acceptance` on step, `budget` on plan, `OracleClass`/`failures` on the verifier, full A.5 budget, `stack`/`replan_count` on state); wire the real FSM in `driver.rs` (call `Planner`, gate on `dag.acyclic()`, run `acceptance.oracles` + `VerificationGate::decide`, implement Repair/Replan/Paused with budgets, add `Mode::{Live,Replay}`); build the Governor (`check()` over all caps + telemetry + autonomy + interrupts); implement the deterministic oracle suite (shell to cargo/git via WP-5 tools) + consistency + gated judge; implement `best_of_n` + `pick_tier`; give replan/subagent/skills/checkpoint behavior; **call `KernelRuntimeClient` from `Act`** + import the (currently unused) context/index crates; fix the projection phase-string bug. *The biggest integration package — do not start before Wave B merges.*

### WAVE D — leaf features (parallel; depend on Wave C)

- **WP-8 · hide-fleet real fleet** *(deps: WP-7 for kernel runs; WP-5 for git/worktree tools)*
  Add a git backend (`gix`/`Command`) + the worktree lifecycle in `isolate.rs`; add `tokio` to deps + build `schedule_tick` with `spawn_run` launching `hide-kernel` runs + bounded-admission/unbounded-completion channels + preemption + spawn-rate breaker; add `similar`(+tree-sitter) + the integration-branch funnel + conflict ladder; make the queue an event-log projection + emit A.5 events; reconcile `AgentJob`/`JobStatus` with A.1 (`concurrency_class` + two-pool split); build `remote.rs` (JSON-RPC-over-WS + session resume); add `FleetManager`+`fleetview`.

- **WP-9 · hawking-research real lab** *(deps: WP-1 for CAS; WP-2 for the model client; WP-3 graph patterns optional)*
  Land the `RuntimeClient` trait (embed/chat/stream) + wire it; add CAS/content-addressing (`sha2`/blake3 + `hide-core::BlobStore`) so node IDs are content-addressed + `content_hash` populated (unlocks idempotent ingest + citation re-verification); swap the `BTreeMap` KG for KùzuDB + §4.8 query modes + entity resolution; build one real `SourceAdapter` (arXiv/OpenAlex + PDFium parse); flesh the empty FSM states (planner, Triage, cited-report `Synthesize`, Persist/Reflect + per-event checkpoint ledger); real adversarial `verify.rs`; litmap/experiments/bridge workflows.

- **WP-10 · hide-personalize moonshots** *(deps: WP-7 for execution feedback; WP-3 for the index in retrieval)* *(lowest priority — moonshots)*
  Make `eval.rs` run its oracles + add `EvalMiner`; give `rlef.rs` a `RewardConfig` + `FeedbackSignal→reward` mapping + an `RlefDaemon` seam; define `MetaRouter` in `retrieval.rs` + wire `hawking-index`; reshape `kv_handoff.rs` into the §11.5 `KvShareGroup` protocol; reconcile `records.rs` types; finish store/curate (scrub-on-write, dataset layout, p95+recency rules); use/remove the 4 dead deps.

### WAVE E — the host (capstone; depends on everything)

- **WP-11 · hide-backend Tauri host** *(deps: all — esp. WP-2 for the runtime client, WP-7 for the kernel)*
  Add `tauri`(+shell/pty) + `tokio`/`reqwest` real deps; wrap `CommandRouter::handle` in `#[tauri::command]` + add the validate→reject branch; replace pull `ui_events` with a push `ipc::Channel<UiEvent>` + render-coalescing + backpressure; build the `RuntimeSupervisor` (spawn `hawking serve`, `/healthz` poll, state machine + backoff + `runtime.lock`, + an HTTP `ModelProvider`); wire `hide-fleet` (or drop the dead dep + set caps honestly); stand up the §7 plugin host (wasmtime + negotiator/ledger) + populate connector `contributions`; add scrub-to-event/fork-at-event.

**Parallelism summary:** Wave A = 1 package (serialize). Wave B = 5 packages fully parallel. Wave C = 1 package (integration gate). Wave D = 3 packages parallel. Wave E = 1 package. WP-3 (index) is the largest and could be split into two parallel sub-agents (parse+merkle | vectors+daemon). The critical path is **WP-1 → WP-2/WP-3/WP-4/WP-5 → WP-7 → WP-11**.
