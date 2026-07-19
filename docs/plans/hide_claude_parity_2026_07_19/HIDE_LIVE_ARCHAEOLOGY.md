# HIDE Live Archaeology - verified map of the live implementation

Run date: 2026-07-19
Repository truth pinned at: `4fbca8bc` (branch `main`)
Method: 6 read-only audit agents, every claim reconciled against current code / git history with file:line evidence (fleet `wf_3a62e82a-5c8`, 710k tokens, 189 tool calls, 0 errors).
Evidence class for this document: VERIFIED REPO unless labeled otherwise.

## 0. Executive verdict

HIDE today is **a real, polished frontend and a real, fast inference runtime with the agent spine sealed out between them.** The React/Tauri app and the Hawking serving runtime are both genuinely built and individually strong. The ~50k-line agent backend that connects them (context compiler, code index, planner/verifier kernel, typed tools, fleet) was extracted to a sealed pack on 2026-07-18 and is recoverable **only from git commit `5a99d0e2`** - the offline archive named in the handoff is **gone from disk**. As shipped from the active source tree, the production vertical slice is **broken**: the frontend speaks `/v1/hide/*` on port 8744 to a `hide-serve` binary that no longer builds, while the live `hawking-serve` runtime serves a disjoint OpenAI-compatible surface on 8080.

The single most important structural fact - confirmed at both the sealed backend and the live runtime - is that **the "pass state, not text" moat is real but dead-ended below the product boundary.** RWKV-7 recurrent state is serializable, forkable by memcpy, and byte-exact-verified in unit tests; the engine carries a `save_checkpoint / load_checkpoint / fork_state` seam; and yet **no HTTP route, no CLI subcommand, and no live turn ever calls them.** Every Hawking-native advantage the supremacy thesis will rest on is present in the codebase as a real-but-unwired primitive, not a shipping capability.

This is good news for the campaign: parity and supremacy are **wiring and integration problems on top of proven parts**, not greenfield invention. The build ladder's job is to reconnect a vertical slice and expose the state moat over the wire.

## 1. Live crate inventory (the active workspace)

Seven crate directories exist under `crates/` at `4fbca8bc`; six are workspace members (`hawking-seed` is an empty non-member; the workspace also has three `tools/` members). The 13-crate HIDE backend is not among them.

| Crate | LOC (rs) | Role | Status |
|---|---|---|---|
| `hawking-core` | ~78k | Engine, model archs, Metal kernels, RWKV state, caches, tokenizer, `.tq` | ACTIVE |
| `hawking-serve` | ~5.7k | OpenAI-compatible + native HTTP server, continuous batching | ACTIVE |
| `hawking` | ~4.9k | CLI (`generate`, `serve`, `bench`, …) | ACTIVE |
| `hawking-speculate` | ~6.3k | Spec-decode toolbox (verifier, governor, EAGLE5, ngram/suffix drafts) | ACTIVE (CLI-only, default-off) |
| `hawking-seed-c` | ~6.2k | Event-Horizon research CLI + absorbed **Pack ABI + provider registry** | ACTIVE (registry ACTIVE_UNEXPOSED) |
| `hawking-bench` | ~1.1k | Performance/throughput harness (decode/prefill/bandwidth/competitive) | ACTIVE |
| `hawking-seed` | 0 | Empty stray directory (only an empty `tests/`), not a workspace member | MISSING |

## 2. The two-packs clarification (critical, resolves handoff ambiguity)

There are **two unrelated "packs" concepts.** Conflating them is the single biggest risk to this campaign, so it is stated first.

**(a) `hawking-packs` - ABSORBED and RETIRED.** The external Event-Horizon packs nucleus was absorbed into `crates/hawking-seed-c/src/providers/` at `0adcab57` as in-crate modules: one Pack ABI (`pack.rs` `PackManifest`, additively extended and proven byte-identical for legacy manifests) and one capability registry (`providers/registry.rs`). No external crate, path dependency, or `Cargo.lock` entry was recreated (verified: `grep hawking-packs Cargo.lock` → none). The local folder was retired/deleted at `4fbca8bc`, archive bundle `756cee3e` recovery-proven. **Do not recreate it** (Bible §7). This is what the git log and the auto-memory refer to.

**(b) `hawking-hide-desktop` - STILL SEALED.** A *separate* sealed manifest (`packs/hawking-hide-desktop.json`, sealed 2026-07-18) covering the 13-crate HIDE product backend (~50,351 content LOC, 164 files). Recoverable only from git `5a99d0e2` (`git checkout 5a99d0e2 -- crates Cargo.toml`). **Correction to the handoff:** its `offline_cache` path (`/Users/scammermike/Downloads/hawking-packs/...`) no longer exists on disk; git history is the only lifeline. The manifest and source commit are both verified present.

## 3. Subsystem live truth

### 3.1 Core engine, models, and state (the moat substrate)

- **RWKV-7 recurrent state is a real, byte-exact, forkable atom** - and it is unexposed. `RwkvState` serializes to a self-describing `DSSSMV1` blob (`rwkv7.rs:292-370`, unit-tested bit-identical `:611-621`); `fork` is a memcpy clone with no re-prefill (`:376-378`); an int8 packaging codec `DSSSMI8` exists (`:395-483`, lossy on wkv, never labeled lossless). The engine trait exposes `save_checkpoint / load_checkpoint / fork_state` (`engine.rs:339-358`); RWKV implements them for real (`rwkv7.rs:1724-1734`); a model-gated end-to-end parity test asserts a restored fork reproduces bit-identical next-token logits (`tests/rwkv7_state_checkpoint_parity.rs:31-73`). **No serve route and no CLI subcommand calls any of them** - only `recurrent_state_size_bytes` is surfaced, as a metric in `GET /v1/hawking/context`.
- **There is NO complete execution-state capsule.** Transformer `KvCache` is not serializable (`cache/mod.rs`), and `qwen_dense` / `deepseek_v2` / `llama` never override the checkpoint seam, so transformer sessions cannot be snapshot or forked at all. The `DSSSMV1` header carries only shape + a `fresh` flag - no model id, tokenizer, or position/boundary metadata. **Only the RWKV recurrent atom exists; the capsule (KV + recurrent + metadata at a committed boundary) is MISSING.**
- **GPU→CPU recurrent readback is unimplemented.** `save_checkpoint` snapshots the CPU oracle `self.state`, which on the shipping macOS GPU decode path is stale relative to the live GPU arena (`rwkv7.rs:1720-1723` admits this). Only CPU-only runs or checkpoints taken at a fresh-prefill boundary are byte-exact. GPU capture cost is entirely unmeasured. **This is the first hard gate on the state moat.**
- **Warm-state store exists but fires nowhere.** `SstateDiskCache` (content-addressed `.sstate` store, atomic write, unit-tested, `cache/sstate_disk.rs`) has no caller anywhere in generate/serve.
- **`.tq` native serving: the historical gap is CORRECTED but gated off.** Both `qwen_dense` (`tq_ffn`/`ensure_tq_cache`) and `rwkv7` (`load_tq_artifact` → STR2 parse) now read `.tq` - but only under the non-default `tq` cargo feature plus `HAWKING_*_TQ` env flags, so `.tq` serving is **absent from the shipping default build**, and the GPU bitslice GEMV is staged (CPU RHT matvec is the parity oracle).
- **Models:** dense (`qwen_dense`, `llama`) ACTIVE; MoE **ships via `deepseek_v2`** (routed + shared experts, MLA); the generic `qwen_moe` loader is a STUB (`Unimplemented`, "lands in Phase 3"); `rwkv7` ACTIVE and is the only engine carrying serializable state; `gemma2/phi3/mamba2/mixtral/olmoe` are PACKED (extracted to `hawking-adapters-extra`).
- **Prefix reuse is the one state lever genuinely wired:** serve KV-prefix sharing (`lib.rs:605-918`) over HTTP, and an in-RAM `InMemoryPrefixCache` default-on in CLI `generate` (`qwen_dense.rs:1496-1531`). Note: serve and CLI use *different* prefix mechanisms.

### 3.2 Serve HTTP, batching, cache, security

A real, working OpenAI-compatible axum server with a single continuous-batching loop (parallel prefill → decode step → stream), wired end-to-end for QwenDense on Metal. Nine routes; **no state routes.** All eleven first-pass gap claims reconcile **CONFIRMED**:

| Gap | Verdict | Evidence |
|---|---|---|
| G1 no HTTP state save/load/fork/rollback | CONFIRMED | `http.rs:160-172` route table has none; primitives at `engine.rs:339-358` unrouted |
| G2 no session→slot affinity | CONFIRMED | slots keyed by anonymous `u32`; `wait_queue` carries no session id (`http.rs:124,127`) |
| G3 `max_seq_len` hardcoded 4096, no Serve override | CONFIRMED | sole literal `lib.rs:509`; Serve subcommand has no flag; `--auto` is "advisory" |
| G4 direct-admit / batch-one misses prefix reuse | CONFIRMED | `copy_kv_prefix_to_slot` sole call site is the queue-drain branch (`lib.rs:908`); default `max_batch_size=1` cold-prefills the first/only request |
| G5 system KV bank is a routing hint | CONFIRMED | stores **zero KV bytes** (`system_kv_bank.rs:12-18`); detached store deferred; a stale source slot degrades to cold prefill |
| G9 `/context` reports an unproven multiplier | CONFIRMED | `HAWKING_QWEN_TQ_MULTIPLIER` read at `http.rs:250` is **never set in-repo** |
| G10 LAN bind, no auth | CONFIRMED | CLI default `0.0.0.0:8080` (`main.rs:133`); no auth middleware anywhere |
| G11 weight-compression ≠ context multiplier | CONFIRMED | `effective = native × tq_multiplier` while real KV capacity is fixed at 4096 |

### 3.3 Tools, structured output, speculative decoding

- **Tool calling is thin API-shaping, not an agent loop.** `tool_calls.rs` renders a Hermes/Qwen `<tools>` preamble into the prompt and parses the completion back to OpenAI `tool_calls`. There is **no tool-execution/dispatch loop** in the serve runtime; the client must execute tools and resubmit via `role:tool` (T6 CONFIRMED). Tool-bearing SSE **buffers to completion before parsing** - a tools request is effectively non-streaming (T7 CONFIRMED).
- **Stop strings, JSON constraint masking, and spec-decode all live only in single-sequence `generate()`, which serve never calls** (T8 CONFIRMED). The batched decode path (`driver.rs:52-110`) forces `json_mode:false`, terminates only on EOS/max_tokens, and does no draft/verify. So over the HTTP boundary, `response_format:json_object` does not constrain output, custom stop strings are ignored, and no spec-decode speedup is available.
- **`hawking-speculate` is a standalone leaf crate, consumed only by CLI `generate()` (env-gated, default-off)** - REFUTED that it is wired into serve. `spec_gov.rs` inside serve is a **dead duplicate governor with zero callers**. The real governor and the **exact-match lossless verify gate** (accept iff `argmax == draft` → bit-identical to greedy, `verifier.rs:77-133`) exist but run only single-seq.
- **First-try-valid tool decode is unbuilt in the active tree.** `JsonConstraint` (structural well-formedness mask) is ACTIVE_UNEXPOSED (single-seq only); `GrammarConstraint` (schema `required_keys`/`Choices`) is a **post-hoc validator with no per-token masking and zero callers**; jump-forward / prompt-lookup of a tool-schema prefix is MISSING here - **but the packed `hawking-orch` already contains a training-free, lossless tool-spec-decode primitive** (see §3.5), which is the right thing to reintegrate.

### 3.4 Frontend (app/) - real shell, mock-fed, backend-deferred differentiators

A genuinely built React 19 + Zustand + Monaco + xterm + Tauri v2 application, styled to the Tadao Ando / Geist Mono doctrine, with every two-surface product surface present. F1 - F5 all CONFIRMED.

- `wire.ts` is a real typed contract: **11 intents, 7 event kinds, 30 custom names, 26 projections** - but its cited Rust source of truth (`crates/hide-core/src/api.rs`) is PACKED, so **the contract is now unanchored and can drift** (REFUTED that Rust keeps it in lockstep).
- `ipc.ts` selects Mock (dev default) vs Live (PROD). Live targets `/v1/hide/{intent,events,connector}` on 8744 - a backend the active tree does not build. In dev, the whole run is scripted by `MockTransport`.
- **Wired and real:** Chat transcript, SteerBar (redirect/pause/resume/cancel), security gate, Home courtyard, Digest (doctrine-clean, hides token meter), IDE Editor (Monaco), per-hunk DiffReview (Tab/Esc accept/reject), Explorer, CodeActions, command palette, Settings, **no-metering doctrine enforced across the FE**.
- **UI-real / backend-deferred ("plan 2"):** PlanCard, DiffChipRow (no mock emits them), **Fleet board / Fork-and-Try-N** (optimistic seeds, "backend forks real state in plan 2"), **StateTimeline scrub/fork** ("backend snapshots state in plan 2"), Context Stack **snapshot / save-skill** (notice-only), `autocompact` (tested policy, never fires in dev).
- **Not functional:** integrated **Terminal has no PTY** (commands only "queued" as intents); **voice records but never transcribes**; attachments never upload; first-run Onboarding surface referenced but missing; StatusBar branch/problems hardcoded.
- Tauri sidecar targets the packed `hide-serve`; a single stale prebuilt binary (Jun 29) exists but cannot be rebuilt from source.

### 3.5 The sealed HIDE backend (13 crates @ `5a99d0e2`, ~49.6k LOC)

All five "facade" claims are CONFIRMED with file:line evidence into `5a99d0e2`:

- **S1** default kernel is `AgentKernel::new` with `StubPlanner`, empty oracle suite, `runtime/dispatcher/grounding = None` (a fully-wired `KernelBuilder` exists but the host never uses it).
- **S2** the live `SubmitTurn` generation sends the **raw prompt, empty message history, `max_output_tokens = 256`** (`host.rs:848-863`).
- **S3** the reserve-then-fill **context compiler runs only behind the `context` connector; its compiled prompt is discarded** relative to generation.
- **S4** `compact_context` is **logged, never performed**; its stated performer (compiler watermark gate) is not on any live path.
- **S5** the planned `/v1/hawking/kv/*` state routes **exist only as a shell-side client seam marked `[RUNTIME-SIDE - LATER]`** with zero server endpoints.

The central defect is one sentence: **every high-value asset is real but unwired from the live turn.** The triage:

| Crate | LOC | Verdict | Why |
|---|---|---|---|
| `hawking-context` | ~4.9k | **REINTEGRATE** (flagship) | reserve-then-fill compiler: carve system/response/scratchpad first, value-density knapsack, degrade ladder, head/tail anti-lost-in-middle order, replayable content-addressed manifest; SQLite/FTS5+cosine memory |
| `hawking-index` | ~4.7k | **REINTEGRATE** (flagship) | tree-sitter defs+refs, cAST chunking, SCIP ids, BLAKE3 merkle change-gate, FTS5+graph, PageRank repo-map, hybrid RRF retriever, incremental MVCC daemon with crash recovery |
| `hide-kernel` | ~5.7k | **REINTEGRATE** (highest-value idea) | plan-as-data DAG where each step declares its acceptance oracle up front; deterministic-first verify gate; governor/interrupts; checkpoint/subagent/skills |
| `hide-tools` | ~5.7k | **REINTEGRATE** | tiered verifying edit applier (search_replace/apply_patch/write_file + base_hash optimistic concurrency), sandboxed `shell.run` w/ watchdog, proc (exec-nonzero-as-data), ignore-walker search, git worktree trio, JSON-RPC MCP client (stdio + Streamable HTTP) |
| `hawking-orch` | ~4.0k | **REINTEGRATE** | role router, confidence-gated escalation (self-consistency vote), grammar validate-and-retry, energy/thermal/RAM admission, **training-free lossless tool-spec-decode (schema jump-forward + prompt-lookup)** |
| `hide-core` | ~2.9k | **REINTEGRATE** (foundation) | pure shared contracts: api/event/ids/permission/persistence/runtime/tool/config/migration/observability/security/supervision traits |
| `hide-serve` | ~0.5k | **REINTEGRATE** (the product boundary) | thin axum over `BackendHost`; the real `/v1/hide/*` shipping seam on loopback:8744 |
| `hawking-eval` | ~0.3k | **REINTEGRATE** (cheap, high-leverage) | pass@1 + Wilson CI over the serve chat path; the "build eval first" gate |
| `hide-security` | ~2.5k | **REINTEGRATE** (logic) / REDESIGN (OS) | blake3 hash-chain audit, AES-256-GCM at-rest, secret redaction, macOS Seatbelt rendering (pure logic real+tested; egress proxy / microVM / ES monitor are seams) |
| `hide-backend` | ~6.6k | **REDESIGN** | good supervisor/event-bus/command-router; the turn loop bypasses the kernel - keep the scaffolding, rebuild the loop |
| `hide-fleet` | ~5.1k | **REDESIGN** | parallel-agent fabric (job DAG, resource admission, isolation leases, merge); heavy relative to single-box need; not HTTP-reachable |
| `hide-personalize` | ~3.0k | **REDESIGN** | RLEF loop real at the edges but stub at the load-bearing core (LoRA grad, PPL pass, KV block copy are seams) |
| `hawking-research` | ~3.6k | **ARCHIVE** | knowledge graph / arXiv ingest; a scope trap for the coding-IDE slice |

### 3.6 Packs / provider registry (absorbed nucleus)

The absorbed Pack ABI and provider registry are **real, integration-tested, and ACTIVE_UNEXPOSED.** `pack.rs` (content-addressed verify + tamper refusal + rollback) is reachable from the `hawking-seed-c` CLI; `providers/registry.rs` (one registry answering *which pack/impl provides a capability, why, compat, loc/bytes, source, tests, rollback*) plus the provider trait and verifier are proven under real Seed authority in `tests/providers_harness.rs`, but **no CLI verb or HTTP route activates them** - "inspect provider capabilities directly in Hawking" is currently a library/test capability. The arch adapters are **declarative descriptors, not a runtime engine** (real execution still uses `crate::adapter::build_plan`), so the registry cannot yet drive live model-role routing. **CORRECTED prior memory:** the absorption is committed (not "designed/uncommitted"); deletion reached 20/20; packs retired at `4fbca8bc`.

## 4. Master reconciliation (Bible §9 historical claims)

| Historical claim | Verdict | Correction |
|---|---|---|
| Real backend host | CONFIRMED (packed) | Real at `5a99d0e2`; absent from active tree |
| Planner - Executor - Verifier loop | CONFIRMED as built, REFUTED as wired | Real `KernelBuilder`; the live turn uses `StubPlanner` and never calls `step()` |
| Worktree fleets / governor / merge funnel | CONFIRMED (packed, unwired) | `hide-fleet` real; not HTTP-reachable; launches stub kernel |
| Repository index | CONFIRMED (packed) | `hawking-index` real+rich; unwired from live turn |
| Context Stack | CONFIRMED | Compiler packed (unwired); FE surface real, mock-fed |
| Session registry / fork / time travel | REFUTED as shipping | Intents wired in FE; backend snapshot is "plan 2"; no serve route |
| Tool parser / runner / parallel read-only tools | MIXED | Parser ACTIVE (API-shaping); runner+parallelism packed in `hide-tools`, unwired |
| MCP scaffolding | CONFIRMED (packed) | JSON-RPC MCP client in `hide-tools` (stdio + Streamable HTTP), unwired |
| Tool-call grammar / jump-forward primitives | CONFIRMED (packed) | `hawking-orch::tool_spec_decode` real+tested; runtime consumption is a seam |
| Model serving seam unwired | CONFIRMED | Serve works; state/KV/HIDE routes absent |
| Tool loop not connected to live turns | CONFIRMED | Live turn is single-shot 256-tok generate |
| Context compiler output discarded | CONFIRMED | Runs only in connector; compiled prompt never fed to generation |
| Character-count token budgeting | PARTLY SUPERSEDED | Compiler uses true tokenizer budgets; but serve `/context` still surfaces an unproven multiplier |
| No persistent live turn memory | CONFIRMED | No history assembled on the live path |
| Prefix/state reuse not exposed | CONFIRMED | Prefix reuse partial (queue-path only); state reuse unexposed entirely |
| Front end incomplete / specced | CORRECTED | FE is substantially built and polished; it is *backend-deferred*, not unbuilt |
| Rigid FSM possibly wrong for open-ended coding | SUPPORTED | `hide-kernel` machine/driver FSM exists; the frontier direction is a flatter loop |
| apply_patch correctness defects | NEEDS PROBE | `hide-tools` uses base_hash optimistic concurrency; correctness untested here (flag for a probe) |

## 5. Hawking-native lever ledger (feeds the supremacy thesis)

Readiness key: **wired** (reachable on a shipping path), **unwired** (real+tested, no caller), **partial**, **stub**, **missing**.

| Lever | Readiness | Location |
|---|---|---|
| Continuous batching (parallel prefill + decode) | wired | `hawking-serve/lib.rs:598-962`, `driver.rs` |
| Prefix reuse (skip re-prefill on shared prefixes) | wired (queue-path) / partial (direct-admit) | serve `lib.rs:605-918`; CLI `InMemoryPrefixCache` |
| Greedy token-only decode lane | wired | `driver.rs:71-94` |
| Exact-match lossless spec-decode verifier | unwired | `hawking-speculate/verifier.rs:77-133` |
| RWKV-7 recurrent state serialize | unwired | `rwkv7.rs:292-370` (byte-exact, tested) |
| RWKV-7 state **fork** (memcpy, no re-prefill) | unwired | `rwkv7.rs:376-378`; `StateShareGroup:840-` |
| Warm-state `.sstate` disk store | unwired | `cache/sstate_disk.rs` |
| Transformer KV capsule | missing | `KvCache` not serializable |
| GPU→CPU recurrent readback (exact live capture) | missing | no reverse readback fn; `rwkv7.rs:1720-1723` |
| HTTP state save/load/fork routes | missing | absent from `http.rs:160-172` |
| Session→slot affinity | missing | anonymous slots |
| Serve long-context (`max_seq_len` override) | missing | hardcoded 4096 |
| Detached reusable KV-block store | missing | deferred (`system_kv_bank.rs:20-23`) |
| Reserve-then-fill context compiler | unwired (packed) | `hawking-context` @ `5a99d0e2` |
| Living code index (tree-sitter/merkle/FTS5/PageRank/RRF) | unwired (packed) | `hawking-index` @ `5a99d0e2` |
| Plan-as-data kernel + oracle-gated verify | unwired (packed) | `hide-kernel` @ `5a99d0e2` |
| Typed tools + verifying edit applier + MCP client | unwired (packed) | `hide-tools` @ `5a99d0e2` |
| Training-free lossless tool-spec-decode | partial (packed) | `hawking-orch/tool_spec_decode.rs` @ `5a99d0e2` |
| `.tq` sub-4-bit native serving | partial (feature+env gated) | `qwen_dense`/`rwkv7`, `tq` feature off by default |
| One Pack ABI + provider/capability registry | unwired | `hawking-seed-c/providers/` |
| Local capability eval (pass@1 + Wilson CI) | unwired (packed) | `hawking-eval` @ `5a99d0e2` |
| Perf eval harness | wired | `hawking-bench` via `hawking bench` |
| Local-first no-metering doctrine | wired (FE) | across `app/src` |

## 6. What this means for the ladder (feed-forward)

1. **The vertical slice is a reconnection, not a build.** Lift `hide-core` + `hide-serve` + `hawking-context` + `hawking-index` + `hide-kernel` (RuntimePlanner, not StubPlanner) + `hide-tools` + `hawking-eval` out of `5a99d0e2`, and replace the 256-token single-shot turn with a flat compile-context → kernel-loop-with-tools → deterministic-verify path. This is Phase 0/1 and it makes the polished FE real.
2. **The state moat needs two things before it is load-bearing:** (a) GPU→CPU recurrent readback so live-GPU capture is byte-exact, and (b) `/v1/hawking/state/{save,load,fork}` + session→slot affinity on `hawking-serve`. Until both land, "fork one warm state into N agents" is a demo of unwired primitives.
3. **First-try-valid tool calls are a reintegration, not research:** wire `hawking-orch::tool_spec_decode` (jump-forward + prompt-lookup, lossless) and the single-seq JSON mask into the batched serve path.
4. **Security must precede autonomy:** `hide-security` gives real audit/redaction/at-rest/Seatbelt logic; the OS enforcement (egress, microVM) is still a seam. Trust-before-config and loopback+auth are prerequisites (serve currently binds `0.0.0.0` with no auth).
5. **Capability claims currently have no active harness:** `hawking-eval` is sealed; only `hawking-bench` (perf) is live. Reintegrating the eval harness is cheap and unblocks every "capability-dense" claim.
