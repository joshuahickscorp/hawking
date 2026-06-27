# 01 · System Architecture & The Extensibility Spine

> **Purpose:** Define the process/component model, the deterministic replayable event backbone, and the plugin spine for the Hawking IDE (HIDE) — engineered so deeply that no further *architectural* design is needed for any subsystem chapter that follows. Every capability is an extension; every state change is an event; every session is replayable.

**Status:** DESIGN. This is the load-bearing chapter — every other chapter binds to the contracts defined in §4.6 (Event Schema) and §7.2 (Extension Manifest). HIDE is the second product in the Hawking family after *Hawking Condense*. The model/runtime layer is treated as a **stable localhost OpenAI-compatible HTTP surface** (see §4.3); runtime-completion items (32B `.tq`, native serving) are marked *later / not shell-gating*.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Domain design tenets](#2-domain-design-tenets)
3. [State of the art + competitor limits](#3-state-of-the-art--competitor-limits)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [Process & component model](#41-process--component-model)
   - 4.2 [ASCII architecture diagram](#42-ascii-architecture-diagram)
   - 4.3 [The runtime as a managed sidecar](#43-the-runtime-as-a-managed-sidecar-stable-http-surface)
   - 4.4 [IPC & the typed event bus](#44-ipc--the-typed-event-bus)
   - 4.5 [Everything-is-an-event backbone](#45-everything-is-an-event-deterministic-replayable-backbone)
   - 4.6 [The event schema (cross-cutting contract)](#46-the-event-schema--cross-cutting-contract)
   - 4.7 [Data-store topology & ownership](#47-data-store-topology--ownership)
   - 4.8 [On-disk layout](#48-on-disk-layout)
   - 4.9 [Async / threading model & backpressure](#49-async--threading-model--backpressure)
   - 4.10 [Config & profiles](#410-config--profiles-workspace--user--project-layering)
   - 4.11 [Versioning, upgrade & migration](#411-versioning-upgrade--migration)
   - 4.12 [Error taxonomy & supervision](#412-error-taxonomy--supervision)
5. [How we EXCEED](#5-how-we-exceed-local-superpowers)
6. [Failure modes / edge cases / mitigations](#6-failure-modes--edge-cases--mitigations)
7. [Extensibility / plugin points](#7-extensibility--the-plugin-spine)
   - 7.2 [The extension manifest (cross-cutting contract)](#72-the-extension-manifest--cross-cutting-contract)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions / dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)

---

## 1. Purpose & scope

This chapter specifies the **skeleton** of HIDE: the processes, the wires between them, the durable backbone that records and replays everything, and the registry that makes every feature a hot-swappable extension. It does **not** specify the agent reasoning loop (Chapter 02), the editor surface (Chapter 03), tools (Chapter 04), context/memory (Chapter 05), or model-runtime internals — but it defines the **contracts** all of those plug into.

**In scope**

- The four-tier process/component model (Tauri host, agent kernel, runtime sidecar, web front-end).
- The IPC transport and the typed, structured event taxonomy.
- The append-only event log → deterministic scrub / replay / resume.
- The plugin/extensibility spine: manifest, capability negotiation, registry, host functions.
- Data-store topology (event log, SQLite metadata, vector store, content-addressed blobs, KV/context) and ownership.
- Async/threading, backpressure, config layering, versioning/migration, supervision/crash-recovery.

**Out of scope (delegated)** — agent planning policy (Ch.02), diff/merge UX (Ch.03), individual tool semantics (Ch.04), retrieval ranking (Ch.05), sampler/grammar/runtime kernels (Ch.06+), distribution/packaging of *Hawking-HF* models (deferred until 32B ready).

**The over-engineering mandate.** The brief is explicit: build the spine so that *new capabilities slot in without core changes*. Concretely, that means the core ships **zero hard-coded feature lists**. Tools, panels, agents, model providers, indexers, memory stores, commands, and even event *kinds* are all entries in a registry, declared by a manifest, and negotiated against host capabilities. The litmus test, applied throughout: *"To add capability X, does anyone touch a file under `core/`?"* If yes, the design has failed and is revised.

---

## 2. Domain design tenets

These ten tenets are the constitution. Every later decision cites one.

| # | Tenet | Consequence |
|---|-------|-------------|
| **T1** | **Everything is an event.** Every observable state transition is an append-only, immutable, ordered record. | The event log is the system of record; all stores are *derived projections* (§4.5). |
| **T2** | **State is derived; the log is truth.** No projection holds authoritative state the log cannot rebuild. | Crash recovery = replay. Corruption of a projection is non-fatal: rebuild it. |
| **T3** | **Replay edits, never re-fire effects.** Replaying the log rebuilds buffers/UI but must not re-run side-effecting tool calls or file writes. | Effects are *recorded as their observed outcome* and replayed as data, never re-executed (the Elm/OpenHands rule, §3). |
| **T4** | **Capability, not ambient authority.** A plugin gets *nothing* by default; every power is an explicitly granted, scoped capability. | Manifest declares needs; host grants per-scope; deny beats allow (§7). |
| **T5** | **The runtime is a replaceable sidecar behind a stable HTTP contract.** | The IDE never links the model engine; it speaks OpenAI + native HTTP. Runtime can crash/restart/upgrade independently (§4.3). |
| **T6** | **Determinism is a feature.** Same log + same seeds ⇒ same derived state, byte-for-byte where the runtime offers it. | `seed` plumbed end-to-end; greedy paths are bit-identical (the runtime already guarantees this — see §3). |
| **T7** | **Local superpowers are first-class, not bolt-ons.** Raw logits, custom samplers, grammar decode, KV surgery, LoRA hot-swap, extended context, draft+spec decode are exposed *through* the architecture, not hidden behind it. | Native endpoints + capability kinds reserve these (§5). |
| **T8** | **Backpressure is mandatory, never best-effort.** Every producer→consumer hop is a bounded channel with a defined overflow policy. | No unbounded queues anywhere (§4.9). |
| **T9** | **Spend lavishly, locally.** No per-token cost, no rate limits → exhaustive logging, many parallel agents, overnight compute, full verification are *defaults*, not luxuries. | The log records *everything*; multi-agent fan-out is a core primitive (§5). |
| **T10** | **Forward-compatible on disk.** Schemas are versioned; unknown fields survive round-trips; migrations are explicit and replayable. | `schema_version` on every persisted artifact; additive-by-default (§4.11). |

---

## 3. State of the art + competitor limits (cited)

**Tauri 2 IPC & process model.** Tauri 2's `invoke()` is a JSON-RPC-like call over a custom `ipc://localhost` protocol (`http://ipc.localhost/` on Windows/Android) that does not hit the network ([Tauri IPC concept](https://v2.tauri.app/concept/inter-process-communication/), [Tauri calling-rust](https://v2.tauri.app/develop/calling-rust/)). The **event** system (`emit`/`listen`) carries **JSON-string payloads only** and warns of **out-of-order delivery** for rapid async listeners — explicitly "not suitable for bigger messages" ([Tauri calling-frontend](https://v2.tauri.app/develop/calling-frontend/)). The **`tauri::ipc::Channel<T>`** is "the recommended mechanism for streaming data": it embeds a per-channel `AtomicUsize` index so the frontend reassembles in order even if evals arrive out of order, and switches transport by size — raw bytes ≤ `1024 B` and JSON ≤ `8192 B` go by direct eval, larger payloads are pulled via an internal fetch command ([Tauri channel.rs](https://docs.rs/tauri/latest/x86_64-apple-ios/src/tauri/ipc/channel.rs.html)). A maintainer benchmark put a 10 MB transfer at ~5 ms on macOS vs ~200 ms on Windows ([Tauri discussion #11915](https://github.com/orgs/tauri-apps/discussions/11915)). **Sidecars** are declared in `bundle.externalBin`, spawned via `tauri-plugin-shell`'s `app.shell().sidecar(...)` returning `(Receiver<CommandEvent>, CommandChild)` with `Stdout/Stderr/Terminated/Error` events; auto-restart/health-check is **not** in core (community plugins only) ([Tauri sidecar docs](https://v2.tauri.app/develop/sidecar/), [plugins-workspace #3062](https://github.com/tauri-apps/plugins-workspace/issues/3062)). **Capabilities v2** replace Tauri 1's global allowlist with per-window/per-platform capability files (`identifier`, `permissions`, `windows`, `platforms`, `remote.urls`), where **deny precedes allow** and scopes pass arbitrary serde data into commands ([Tauri capabilities](https://v2.tauri.app/security/capabilities/), [Tauri permissions](https://v2.tauri.app/security/permissions/)).

> **HIDE takeaway:** stream tokens over `ipc::Channel<T>` (ordered, fast) — never `emit/listen` — and treat the runtime as a sidecar behind HTTP, *not* over stdin/stdout, so it can be restarted and even run as a shared daemon. We adopt the capability/ACL pattern wholesale and extend it to *our own* plugin spine.

**Event-sourcing & replayable agent sessions.** The canonical pattern: events are immutable past-tense facts, state is rebuilt by replaying the log, snapshots are an optimization not a replacement ([Fowler, Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html); [Azure Event Sourcing pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing)). **OpenHands** is the reference agent design: an `Event`/`Action`/`Observation` model where `EventStream` is the append-only spine (one JSON file per event under `sessions/<sid>/events/N.json`), rejects any event that already has an id (anti-replay-loop guard), and rebuilds `state.history` from the stream; its **replay manager replays only Actions and regenerates Observations** rather than pasting a transcript ([OpenHands stream.py](https://github.com/All-Hands-AI/OpenHands/blob/0.40.0/openhands/events/stream.py), [replay.py](https://github.com/All-Hands-AI/OpenHands/blob/0.40.0/openhands/controller/replay.py)). **aider** uses *git itself* as the event log — auto-committing every AI edit, `/undo` reverting the last aider commit ([aider git docs](https://aider.chat/docs/git.html)). **Cursor** uses **local file checkpoints** stored separately from git, restoring file state only — and explicitly **does not revert terminal commands** ([Cursor checkpoints](https://cursor.com/docs/agent/chat/checkpoints)). **Continue.dev** stores one JSON per session under `~/.continue/sessions/` ([Continue paths.ts](https://raw.githubusercontent.com/continuedev/continue/main/core/util/paths.ts)). The **replay-side-effect rule** is independently rediscovered everywhere: Elm's time-travel debugger "replays messages but deliberately does not re-run commands" to avoid double effects ([Elm time travel](https://elm-lang.org/news/time-travel-made-easy)). CRDT op-logs (Automerge `Change` hash-DAG; Zed `Operation` with Lamport+global clocks persisted as `buffer_operation` rows replayed on join) are *the same durable replayable artifact* ([Zed CRDTs](https://zed.dev/blog/crdts), [Zed buffers.rs](https://raw.githubusercontent.com/zed-industries/zed/main/crates/collab/src/db/queries/buffers.rs)).

> **Competitor limits we beat:** Cursor checkpoints are *file-only* and *git-separate* — a partial, lossy timeline that forgets terminal effects. aider's git-as-log conflates the user's VCS with the agent's history and can only undo whole commits. OpenHands has the right spine but is cloud-deployed Python with no editor and no extensibility spine for *panels/providers/indexers*. **HIDE unifies a full event log (T1) with deterministic replay (T2/T3) AND an editor AND a plugin spine, on-device.**

**Plugin architectures.** VS Code runs extensions in a separate **extension host** process for stability, with a manifest of `contributes` points (commands, views, languages, …) and lazy `activationEvents` — but an activated extension has **full Node/OS access by default**; isolation is process-level for stability, not a capability sandbox ([VS Code extension host](https://code.visualstudio.com/api/advanced-topics/extension-host), [contribution points](https://code.visualstudio.com/api/references/contribution-points)). The modern alternative is the **WebAssembly Component Model + WASI 0.2** (ratified Jan 2024) with **WIT** worlds declaring imports (capabilities required) and exports (functionality provided), where a guest has **no ambient authority** — the host hands each capability in via the `wasmtime::component::Linker`, and sandboxing is enforced with **fuel** (deterministic metering), **epoch interruption** (cheaper deadline traps), and a **`ResourceLimiter`** memory cap ([Component Model WIT](https://component-model.bytecodealliance.org/design/wit.html), [WASI 0.2](https://bytecodealliance.org/articles/WASI-0.2), [wasmtime interrupting](https://docs.wasmtime.dev/examples-interrupting-wasm.html)). WASI **0.3.0** (ratified Jun 11 2026) adds `stream<T>`/`future<T>` async ([WASI 0.3](https://bytecodealliance.org/articles/WASI-0.3)). **Zed** is the proof this works for an IDE: extensions are Rust→WASM (`wasm32-wasip2`) against a versioned `zed:extension` WIT world, with an `extension.toml` manifest declaring grammars/language-servers/slash-commands/context-servers and an explicit `[[capabilities]]` allow-list (`process:exec`, `download_file` host-scoped) ([Zed extensions](https://zed.dev/docs/extensions/developing-extensions), [Zed WIT](https://github.com/zed-industries/zed/blob/main/crates/extension_api/wit/since_v0.6.0/extension.wit), [Zed capabilities](https://zed.dev/docs/extensions/capabilities)).

> **HIDE takeaway:** adopt the **two-tier** plugin model — *trusted in-process Rust* extensions for first-party speed-critical capabilities, *sandboxed WASM Component* extensions (wasmtime, WIT, fuel+epoch+ResourceLimiter) for third-party code — both described by **one manifest schema** and gated by **one capability negotiator**. This is strictly more capable than Zed's "language/theme/slash-command only" surface (HIDE plugins *can* contribute panels, agents, providers, indexers) while being strictly safer than VS Code's "full Node by default."

**Embedded data stores.** `sqlite-vec` (Alex Garcia) is brute-force-only as of v0.1 — fine to "low millions" of vectors but no ANN yet (tracking issue open) ([sqlite-vec stable](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html), [ANN issue #25](https://github.com/asg017/sqlite-vec/issues/25)). Pure-Rust embedded KV: **redb** is a COW B+tree with ACID transactions, per-transaction durability, and XXH3-128 Merkle checksums that detect/rollback partial commits after a crash ([redb design](https://github.com/cberner/redb/blob/master/docs/design.md)); **fjall** is an LSM engine (RocksDB-like) with configurable `PersistMode` ([fjall](https://github.com/fjall-rs/fjall)). **FastCDC** content-defined chunking (pure-Rust `fastcdc` crate) gives deterministic, dedup-friendly chunk boundaries for a content-addressed blob store ([fastcdc-rs](https://github.com/nlfiedler/fastcdc-rs)). **SCIP** (Sourcegraph) is a Protobuf code-intelligence format with human-readable symbol IDs that *unblocks incremental indexing* (only re-index changed files), unlike LSIF's opaque global IDs; **tree-sitter** re-parses modified code in O(log n) over 40+ languages ([SCIP announce](https://sourcegraph.com/blog/announcing-scip), [scip repo](https://github.com/sourcegraph/scip/)).

**Runtime ground truth (this repo).** HIDE's runtime already exposes everything the shell needs over HTTP: `hawking-serve` is an axum OpenAI server with `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/models`, `/v1/embeddings`, and **native** `/v1/hawking/generate` + `/v1/hawking/tokens` (raw token-id SSE, lowest overhead); a continuous-batching loop with a per-slot scheduler, prefix-cache KV reuse, and a system-prompt KV bank; `/healthz` + `/metrics`. The `Engine` trait streams via a `sink: &mut dyn FnMut(StreamEvent)` where `StreamEvent::{Token{id,text}, Done{reason,stats}}`, `GenStats::dec_tps()` reports throughput, `GenerateRequest` already carries an `abort: Option<Arc<AtomicBool>>` cancellation flag and a `json_mode` constrained-decode flag, and `JsonConstraint::mask_logits` implements grammar-masked sampling. (Files: `crates/hawking-core/src/engine.rs`, `crates/hawking-serve/src/{lib,http}.rs`, `crates/hawking-core/src/json_constrain.rs`.) These are the seams HIDE binds to in §4.3.

---

## 4. The Hawking design (concrete)

### 4.1 Process & component model

HIDE is **four processes** (the runtime can optionally be a shared daemon → "3+N"):

1. **Tauri 2 host (Rust)** — the trust root and process supervisor. Owns the WebView, the OS-level capabilities (filesystem, shell, network), the event log, all data stores, and the plugin host. Single binary; the only process the user launches.
2. **Agent kernel (in-process Rust crate, `hide-kernel`)** — *not* a separate process. It is a library crate the Tauri host hosts on a dedicated runtime. It contains the session manager, the event-log writer/reader, the projection engine, the plugin registry + capability negotiator, the tool dispatcher, and the runtime HTTP client. Co-locating it with the host (vs a 5th process) avoids a second IPC hop on the hottest path (event emission) while keeping it a *clean crate boundary* so it is unit-testable headless and reusable by a future CLI.
3. **Runtime sidecar (`hawking serve`)** — the model engine, managed via `tauri-plugin-shell` but **addressed over localhost HTTP**, not stdin/stdout (§4.3). Crash-isolated, restartable, upgradable, shareable.
4. **Front-end (React + Monaco, in the WebView)** — pure view + interaction. Holds **no authoritative state**; it renders projections pushed from the kernel and sends intents back. A crashed/reloaded WebView loses nothing (T2).

**Plus N WASM plugin instances** — each sandboxed extension runs in a wasmtime `Store` *inside* the host process but isolated by the WASM sandbox + fuel/epoch/memory limits (§7.4). Trusted Rust plugins are linked in-process.

**Why these boundaries.** The cut between (1/2) host+kernel and (3) runtime follows T5: the runtime is the only component that holds GPU/Metal state, is the most likely to OOM or hang, and is on its own release cadence (it is *Hawking Condense*'s output target). The cut between (1/2) and (4) front-end follows T2: the view is disposable. The kernel is *inside* the host (not a 5th process) because the event bus is the hottest wire and a process hop there would tax every token; a crate boundary gives the testability benefit without the IPC tax.

**Threading homes** (detail in §4.9): the host runs a multi-threaded tokio runtime; the kernel owns a **single-writer** event-log task (serialization point for ordering, T1), a pool of **session worker** tasks (one per active agent run; many run in parallel, T9), a **projection** task per live session, and a **plugin executor** pool. GPU work lives entirely in the runtime sidecar.

### 4.2 ASCII architecture diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TAURI 2 HOST PROCESS (Rust, trust root, supervisor)                           │
│                                                                                │
│  ┌────────────────────────── WebView (React + Monaco) ──────────────────────┐ │
│  │  Editor │ Diff │ Agent panel │ Plan view │ Test panel │ Terminal │ …      │ │
│  │  (renders PROJECTIONS; holds NO authoritative state — T2)                 │ │
│  └───────▲───────────────────────────────────────────────────┬──────────────┘ │
│          │  ipc::Channel<UiEvent>  (ordered, fast, §4.4)      │ invoke(Intent) │
│   PROJECTION STREAM (host → webview)                          │ COMMANDS       │
│          │                                                    ▼                │
│  ┌───────┴────────────────────── hide-kernel (in-process crate) ────────────┐ │
│  │                                                                           │ │
│  │   ┌─────────────┐   ┌──────────────────┐   ┌───────────────────────────┐ │ │
│  │   │  Session    │   │  EVENT BUS        │   │  Plugin Registry +        │ │ │
│  │   │  Manager    │──▶│  (broadcast       │◀──│  Capability Negotiator    │ │ │
│  │   │  (worker    │   │   fan-out, T1)    │   │  (manifests, grants, T4)  │ │ │
│  │   │  per run)   │   └───┬──────────┬────┘   └────────────┬──────────────┘ │ │
│  │   └──────┬──────┘       │          │                     │                │ │
│  │          │      ┌───────▼───┐  ┌───▼─────────┐    ┌──────▼─────────────┐  │ │
│  │          │      │ Event-Log │  │ Projection  │    │  Plugin Executors  │  │ │
│  │          │      │ WRITER    │  │ Engine      │    │  ┌──────┐ ┌──────┐  │  │ │
│  │          │      │ (single   │  │ (derives    │    │  │WASM  │ │Rust  │  │  │ │
│  │          │      │  writer = │  │  buffers/UI │    │  │sbox  │ │trust │  │  │ │
│  │          │      │  order)   │  │  state, T2) │    │  │fuel/ │ │in-   │  │  │ │
│  │          │      └─────┬─────┘  └──────┬──────┘    │  │epoch │ │proc  │  │  │ │
│  │          │            │               │           │  └──────┘ └──────┘  │  │ │
│  │   ┌──────▼────────────▼───────────────▼───────────────────┐            │  │ │
│  │   │  Tool Dispatcher (capability-checked) + Runtime Client │            │  │ │
│  │   └──────┬───────────────────────────────────┬────────────┘            │  │ │
│  └──────────┼───────────────────────────────────┼─────────────────────────┘ │ │
│             │ OS capabilities (fs/shell/net,     │ HTTP (localhost, §4.3)     │ │
│             │ ACL-gated — Tauri caps v2)         │                            │ │
│  ┌──────────▼─────────────── DATA STORES (host-owned, §4.7) ────────────────┐ │ │
│  │  EVENT LOG (append-only, fjall/segmented)  │  SQLite (metadata, WAL)     │ │ │
│  │  Vector store (sqlite-vec/usearch)         │  Blob store (CAS, FastCDC)  │ │ │
│  │  KV/Context cache (redb)                                                 │ │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┼────────────────────────────────────────────┘
                                     │ spawn + supervise (tauri-plugin-shell)
                                     ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  RUNTIME SIDECAR  `hawking serve`  (separate process, GPU)    │
        │  axum OpenAI server + native endpoints:                       │
        │    /v1/chat/completions (SSE)   /v1/hawking/generate (SSE)    │
        │    /v1/completions  /v1/models  /v1/embeddings                │
        │    /v1/hawking/tokens (raw token-id SSE)  /healthz  /metrics  │
        │  continuous-batch loop · slot scheduler · prefix-cache KV     │
        │  system-prompt KV bank · json_mode grammar mask · spec-decode │
        │  Engine trait: generate(req, sink: FnMut(StreamEvent))        │
        │  [later/not-shell-gating: 32B .tq native serving]            │
        └──────────────────────────────────────────────────────────────┘
```

### 4.3 The runtime as a managed sidecar (stable HTTP surface)

**Contract (T5).** The kernel never links `hawking-core`. It speaks HTTP to `hawking serve` on `127.0.0.1:<port>`. This is the single most important decoupling in the system: the runtime is *Hawking Condense*'s deliverable, evolving on its own cadence, and the most crash-prone component (GPU/Metal). Treating it as a network service means the IDE survives runtime crashes, can hot-upgrade the runtime, can point at a *remote* runtime on another box, and can run the runtime as a **shared daemon** across multiple IDE windows.

**Why HTTP, not stdin/stdout.** Tauri sidecars can be driven over stdio (`CommandChild::write`, `CommandEvent::Stdout`), but stdio is a single byte-stream with no multiplexing — every concurrent agent run would contend on one pipe, and a hung read blocks all. The runtime *already is* an HTTP server with SSE streaming, per-slot scheduling, and continuous batching (`crates/hawking-serve`). HTTP gives us: (a) N concurrent streams (one per agent worker), (b) backpressure via TCP + per-slot SSE channels, (c) standard tooling (curl, the `openai` SDK) for debugging, (d) trivial relocation to a remote host. The sidecar mechanism is used **only for lifecycle** (spawn, health, restart), not data.

**Lifecycle state machine** (owned by the kernel's `RuntimeSupervisor`):

```
        spawn()              GET /healthz == ok            (crash / exit)
 Down ──────────▶ Booting ───────────────────▶ Ready ───────────────────▶ Down
   ▲                │                            │   (restart, backoff)      │
   │                │ boot timeout (60s)         │ N consecutive 5xx /       │
   │                ▼                            │ /healthz fail (3×2s)      │
   └──────────── Failed ◀────────────────────── Degraded ◀──────────────────┘
       give up after K restarts within window     (drain in-flight, restart)
```

- **Booting → Ready:** poll `/healthz` (the runtime returns `ok` when alive); model load is 10–60 s. The supervisor surfaces a `runtime_status` UI event at each transition (§4.6).
- **Ready → Degraded:** 3 consecutive `/healthz` failures at 2 s interval, OR N=5 consecutive HTTP 5xx within 30 s. In-flight requests get a `runtime.unavailable` error event; their sessions pause (not fail).
- **Degraded → Booting:** restart with exponential backoff (1 s, 2 s, 4 s, … capped 30 s).
- **Failed:** K=5 restarts inside a 5-minute window ⇒ stop, surface a blocking banner, offer "retry" / "open logs" / "switch model provider" (model providers are extensions — §7, so the user can fall back to a different local model or even a cloud provider plugin).

**Port & discovery.** The supervisor binds the runtime to an ephemeral port written to `<workspace>/.hide/runtime.lock` (`{pid, port, model_id, started_at, pid_namespace}`). On startup the kernel checks for a live lock (shared-daemon reuse) before spawning. The lock is `flock`-guarded to avoid double-spawn races between IDE windows.

**The model-provider abstraction.** The runtime sidecar is the *default* provider, but "model provider" is an **extension kind** (§7.2). A provider extension implements a small trait:

```rust
// host-side trait; WASM providers implement the mirror WIT world
pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &str;                      // "hawking-local", "openai", "anthropic", …
    fn capabilities(&self) -> ProviderCaps;    // streaming? logits? grammar? loras? embeddings?
    fn generate(&self, req: GenRequest, sink: TokenSink) -> Result<GenStats>;  // SSE under the hood
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    // local-only superpowers (None on cloud providers — see ProviderCaps):
    fn raw_logits(&self, req: GenRequest) -> Option<LogitStream> { None }
    fn load_lora(&self, path: &Path) -> Option<LoraHandle> { None }
    fn kv_handle(&self, session: SessionId) -> Option<KvHandle> { None }
}
```

`ProviderCaps` is how the agent kernel *negotiates*: a planner that wants grammar-constrained tool-call output checks `caps.grammar` and degrades gracefully if absent. The Hawking-local provider sets every cap true and routes the superpowers to native endpoints (raw logits, LoRA hot-swap, KV surgery — §5). Cloud-provider plugins set those `None`. **This is the seam that lets HIDE both ship local-first and still interoperate.**

**Request mapping.** The kernel maps its internal `GenRequest` to either `/v1/chat/completions` (role/message envelope, for chat-shaped turns) or `/v1/hawking/generate` (lean, no envelope — for raw completion / fill-in-the-middle / agent-scaffold prompts) or `/v1/hawking/tokens` (raw token-id SSE — for spec-decode experiments and minimum-overhead loops). `seed`, `temperature`, `top_p`, `stop`, and `json_mode` flow through; the SSE token stream is re-emitted as `token` events on the bus (§4.6).

### 4.4 IPC & the typed event bus

There are **three wires**, each with a defined direction and discipline:

**Wire A — Intents (WebView → kernel), via Tauri commands.** User actions become typed `invoke('hide_intent', { intent })` calls. Intents are *requests to do something*, never state mutations themselves. Example intents: `SubmitTurn`, `AcceptDiff`, `RejectDiff`, `CancelRun`, `ScrubToEvent`, `ForkSession`, `OpenFile`, `RunCommand`. Each handler validates, then **appends one or more events** to the log (the only way state changes, T1). Commands return immediately with an `ack` (`{accepted, event_seq}`) — the *result* arrives later as events on Wire C. This keeps the UI responsive and the model uniform: *intent in, events out*.

**Wire B — Projection stream (kernel → WebView), via `ipc::Channel<UiEvent>`.** The kernel pushes a single ordered channel of `UiEvent`s per window. We use **Channel, not emit/listen**, precisely because Channel guarantees order (atomic index) and handles large payloads via fetch ([Tauri channel.rs](https://docs.rs/tauri/latest/x86_64-apple-ios/src/tauri/ipc/channel.rs.html)) — token streams at 40–120 tok/s and large diff payloads both ride this wire safely. `emit/listen` is reserved for *rare, fire-and-forget* UI signals (e.g. "settings changed") where order doesn't matter.

**Wire C — The internal event bus (kernel-internal fan-out).** This is the heart. Every event appended to the log is *also* fanned out to in-process subscribers: the projection engine, plugins with an `events:subscribe` capability, the indexer, the memory store, and observability. Implementation: a `tokio::sync::broadcast` per session for live fan-out, backed by the durable log for replay and late joiners. `broadcast` lag/overflow is handled explicitly (§4.9).

**Event envelope on the wire.** All three wires carry the same canonical `Event` (§4.6) — UI events are a *projection-flavored subset* (the kernel never ships internal-only events like raw KV pointers to the WebView). One schema, multiple consumers.

**Backpressure across the IPC boundary.** The WebView is the slowest consumer (rendering). The projection channel is bounded (capacity 4096 UiEvents). On a slow WebView, the kernel does **not** drop authoritative events (they are already durable in the log); instead the projection engine **coalesces** — e.g. 50 buffered `token` events for one stream collapse into one `token_batch` UiEvent before the channel send. This is the "render-coalescing" pattern: the log keeps every token, the UI gets batched updates. (Detail §4.9.)

### 4.5 Everything-is-an-event: deterministic, replayable backbone

**The core loop.** Every state transition in HIDE is:

```
intent / runtime-output / tool-result / file-change
        │
        ▼
   [validate] ──▶ append immutable Event to LOG (single writer assigns seq)
        │                              │
        │                              ├──▶ broadcast to bus (Wire C)
        │                              │       ├──▶ Projection engine ──▶ UiEvent (Wire B)
        │                              │       ├──▶ Indexer / Memory store
        │                              │       └──▶ Subscribed plugins
        │                              └──▶ fsync per durability policy (§4.9)
```

**Sources of events** (`EventSource`): `User` (intents), `Agent` (planner/model decisions), `Tool` (tool invocations + their observed results), `Runtime` (token stream, stats, status), `System` (lifecycle, migrations, errors), `Plugin` (extension-emitted, namespaced). Mirrors OpenHands' `AGENT|USER|ENVIRONMENT` but finer-grained for an IDE.

**Causality.** Every event carries `parent: Option<EventId>` (the event that directly caused it) and `run_id`. An `Observation`-class event (a tool result) carries `cause: EventId` pointing at the `Action`-class event (the tool call) that produced it — exactly OpenHands' `cause` link. This builds a causal DAG, not just a flat stream, so the UI can show "this diff resulted from that plan step which resulted from that user turn."

**Replay semantics (T3 — the load-bearing rule).** To replay/scrub/resume:

1. **Pick a target seq** S (scrub slider, or "resume from crash" = last durable seq).
2. **Find the nearest snapshot** ≤ S (§4.7 — projections are snapshotted periodically).
3. **Fold events (snapshot, S]** through the projection engine to rebuild derived state (buffers, plan tree, panel state).
4. **Crucially: do NOT re-execute effects.** Tool calls, file writes, shell commands, and model generations are **recorded as their observed outcome** and replayed *as data*. A `tool.result` event already contains the bytes the tool returned; replay applies those bytes to the projection, it does not re-run the tool. A `file.write_applied` event contains the post-image (or a blob ref); replay restores the file content, it does not re-issue the write to disk unless the user explicitly *resumes forward* past S into new live execution.

This distinction is what makes replay safe and fast: scrubbing the timeline is a pure fold over recorded data. Only when the user chooses **"resume execution from here"** does the kernel transition from *replay mode* to *live mode*, re-attaching the runtime and tool dispatcher and appending *new* events from S onward. (Forking — "resume on a branch" — is identical but writes new events to a child session, leaving the original intact, like LangGraph's checkpoint-fork or git-branch.)

**Determinism (T6).** Given the same log prefix and the same seeds, the derived state is identical. Model generations are recorded *with their seed and the exact request*, and the Hawking-local runtime guarantees greedy bit-identity (the repo's runtime already pins this — `seed` plumbed through `SamplingParams`, greedy paths bit-identical). So a recorded run can be *re-derived* deterministically; and a *forked* run that changes one upstream event re-executes downstream with reproducible results. Non-determinism is quarantined to (a) sampled (temperature>0) generations and (b) genuinely external tool results (network) — both of which are *recorded*, so replay is still deterministic even when live execution was not.

**Append-only, immutable, never edited.** Events are never mutated or deleted (T1). "Undo" is a *new* event (`diff.reverted`, a compensating event in Fowler's sense), not a deletion. This makes the entire history auditable and the "forever-editable memory" promise (T9) literal: memory is a projection over an immutable log the user owns.

**Compaction without losing truth.** Unbounded logs are the obvious risk. Mitigation (standard event-sourcing): **snapshot + tail + segment-archival**. The log is segmented (§4.7); cold segments are compacted by *key-compaction* for high-churn event kinds (e.g. only the latest `cursor.moved` per file survives compaction — cursors have no historical value) while *semantically important* kinds (turns, plans, tool calls, diffs) are never compacted. Compaction is itself recorded (`system.segment_compacted` with the pre/post hashes) so the log's integrity chain is unbroken.

### 4.6 The event schema — CROSS-CUTTING CONTRACT

> **This is the contract every other chapter binds to.** Chapters 02–06 emit and consume these events. The schema is **additive-by-default** (T10): new `kind`s and new fields may be added without a major version bump; renames/removals require a migration (§4.11). Unknown fields **must** survive round-trips (serde `flatten` capture).

**Canonical envelope (Rust + serialized form):**

```rust
/// The single immutable record. Serialized as newline-delimited JSON in the hot
/// log and as a typed row for indexed query. NEVER mutated after append.
#[derive(Clone, Serialize, Deserialize)]
pub struct Event {
    pub schema_version: u16,        // event-schema version (currently 1)
    pub seq: u64,                   // monotonic, assigned by the single writer; the total order
    pub id: EventId,                // ULID — globally unique, sortable, embeds timestamp
    pub session_id: SessionId,      // which session/conversation
    pub run_id: Option<RunId>,      // which agent run within the session (None for ambient events)
    pub parent: Option<EventId>,    // direct causal parent (builds the causal DAG)
    pub cause: Option<EventId>,     // for Observations: the Action that produced this (OpenHands-style)
    pub ts: u64,                    // wall-clock micros since epoch (informational; `seq` is the order)
    pub source: EventSource,        // User | Agent | Tool | Runtime | System | Plugin{ns}
    pub actor: Option<String>,      // sub-actor id (which agent/plugin/tool instance)
    pub kind: String,               // dotted kind, e.g. "tool.call" — NAMESPACED, registry-validated
    #[serde(flatten)]
    pub payload: serde_json::Value, // kind-specific body; schema registered per-kind (§7.2)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redactions: Option<Vec<String>>, // JSON-pointer paths scrubbed for privacy/secrets
    #[serde(flatten)]
    pub ext: BTreeMap<String, serde_json::Value>, // forward-compat capture of unknown fields (T10)
}

pub enum EventSource { User, Agent, Tool, Runtime, System, Plugin(String) }
```

`id` is a **ULID** (sortable, timestamped, no coordination) for the public identity; `seq` is the **authoritative total order** assigned by the single-writer task. Two fields, two jobs: `id` is stable across machines/merges, `seq` is the local replay order.

**The kind taxonomy (initial registered set).** Kinds are dotted and namespaced. Core kinds live in the `hide.*`-implicit namespace; plugin kinds are `pluginname.*`. Every kind has a registered payload schema (§7.2); the table below is the *initial* set — extensions add more without core changes.

| Family | `kind` | Payload (key fields) | Class | Emitted by |
|---|---|---|---|---|
| **Session** | `session.created` | `{title, workspace_root, model_id}` | — | System |
| | `session.forked` | `{from_session, at_seq}` | — | User |
| | `session.resumed` | `{from_seq}` | — | User |
| **Turn** | `turn.user` | `{text, attachments[], mentions[]}` | Action | User |
| | `turn.assistant_started` | `{run_id}` | — | Agent |
| | `turn.assistant_ended` | `{run_id, stop_reason}` | — | Agent |
| **Plan** | `plan.step` | `{step_id, parent_step, title, status, rationale}` | — | Agent |
| | `plan.step_updated` | `{step_id, status: pending\|active\|done\|failed\|skipped}` | — | Agent |
| **Token** | `token` | `{run_id, text, token_id, logprob?}` | — | Runtime |
| | `token_batch` | `{run_id, text, count}` (coalesced — §4.4) | — | Runtime |
| **Tool** | `tool.call` | `{call_id, tool, args, capability_grant_id}` | Action | Agent |
| | `tool.progress` | `{call_id, message, fraction?}` | — | Tool |
| | `tool.result` | `{call_id, ok, output, bytes_ref?, exit_code?}` | Observation | Tool |
| **Diff/File** | `diff.proposed` | `{diff_id, path, hunks[], pre_blob, post_blob}` | Action | Agent |
| | `diff.applied` | `{diff_id, path, post_blob, on_disk_hash}` | Observation | Tool |
| | `diff.reverted` | `{diff_id, reason}` (compensating) | — | User/System |
| | `file.changed_external` | `{path, new_blob, watcher}` | Observation | System |
| **Test/Build** | `test.status` | `{suite, passed, failed, skipped, duration_ms}` | Observation | Tool |
| | `build.status` | `{ok, errors[], warnings}` | Observation | Tool |
| **Context** | `context.update` | `{added[], dropped[], token_budget, reason}` | — | Agent |
| | `context.retrieval` | `{query, hits[], store}` | — | Agent |
| **Memory** | `memory.written` | `{key, scope, blob_ref}` | — | Agent/User |
| **Runtime** | `runtime.status` | `{state, model_id, port}` | — | System |
| | `runtime.stats` | `{dec_tps, prompt_tokens, completion_tokens, ...}` | — | Runtime |
| | `runtime.unavailable` | `{reason, retry_in_ms}` | — | System |
| **Error** | `error` | `{taxonomy_code, message, fatal, context}` | — | any |
| **System** | `system.migration` | `{from_v, to_v, applied[]}` | — | System |
| | `system.segment_compacted` | `{segment, pre_hash, post_hash}` | — | System |
| **Plugin** | `<plugin>.<kind>` | plugin-defined (schema registered) | — | Plugin |

**Action / Observation / neither.** Following OpenHands, events are tagged by *class* (`Action`, `Observation`, or neither). This matters for replay (T3): only `Action`-class events represent effects; `Observation`-class events carry the *recorded outcome* that replay applies. The replay engine asserts: *for every `Action` in the log there is at most one `Observation` with `cause == action.id`*; a dangling `Action` (no observation) means the run was interrupted mid-effect (recovery handles it — §4.12).

**Why this exact shape (decisions other chapters inherit):**

- **`seq` is the order, `ts` is decoration.** Never sort by `ts` (clock skew, batching). Replay folds in `seq` order.
- **`payload` is `flatten`ed `serde_json::Value`, not an enum.** A closed Rust enum would force a core edit for every new kind, violating the mandate. Instead, kinds are *registered* with a JSON Schema (§7.2) validated at append time. Typed accessors are generated for core kinds; plugins validate against their registered schema. *(Alternative considered: a giant `#[non_exhaustive]` enum. Rejected: it makes the core the bottleneck for every new event kind and breaks WASM-plugin event emission, which cannot extend a Rust enum. Open-string-kind + registered-schema is the extensibility-preserving choice — see §9 Q1 for the residual type-safety trade.)*
- **`bytes_ref` indirection.** Large tool outputs / file post-images are not inlined; they are content-addressed blob refs (`blake3:…`) into the blob store (§4.7). The event stays small (Tauri/JSON-friendly) and the blob is deduped.
- **`redactions`.** Secrets (API keys in shell output, tokens) are scrubbed *before* the event is durable, with the scrubbed paths recorded so the redaction itself is auditable (privacy, T-privacy / §5).

### 4.7 Data-store topology & ownership

Five stores, **all owned by the Tauri host** (single trust + lifecycle root), each a *projection or sink* of the event log except the log itself.

| Store | Tech | Owns | Authoritative? | Rebuildable from log? |
|---|---|---|---|---|
| **Event log** | Segmented append-only files (length-prefixed NDJSON + index), or `fjall` LSM for the indexed mirror | The total ordered history (T1) | **YES — system of record** | n/a (it *is* the source) |
| **Metadata DB** | **SQLite** (WAL mode) | Sessions, runs, file/symbol catalog, settings, plugin registry, indexer bookkeeping | No (projection) | Yes (replay) |
| **Vector store** | **sqlite-vec** (brute-force, low-millions) → **usearch/hnsw_rs** sidecar when scale demands ANN | Code/chunk/memory embeddings for retrieval | No (projection) | Yes (re-embed from blobs) |
| **Blob store** | Content-addressed (CAS), **FastCDC** chunking + blake3 keys | File post-images, large tool outputs, attachments, model artifacts | No (deduped sink) | Partially (re-derivable where the producing event is replayable; cold blobs may be GC'd) |
| **KV/context cache** | **redb** (COW B+tree, XXH3-checksummed, per-txn durability) | Prompt-assembly cache, retrieval cache, projection snapshots | No (cache) | Yes (recompute) |

**Ownership rules.**

- **Only the single-writer event-log task appends to the log.** This is the serialization point that gives every event its `seq` (T1). Nothing else writes the log.
- **Projections subscribe to the bus and update their store; they never write the log.** The metadata DB, vector store, and indexer are *downstream*. If a projection store is deleted/corrupted, the kernel rebuilds it by replaying (T2). This is tested in CI: "delete `meta.sqlite`, restart, assert identical projection."
- **The blob store is a *sink*, write-once-by-hash.** A blob keyed by its blake3 hash is immutable; writes are idempotent (same content → same key → no-op). GC is mark-and-sweep over *reachable* refs from non-compacted events (§4.12).
- **`seq` is the cross-store join key.** Every projection records `last_applied_seq`. On restart, each projection replays `(last_applied_seq, head]` to catch up — incremental, not full. A projection that has *never* been built replays from seq 0 (or nearest snapshot).

**Why SQLite for metadata (not redb).** Metadata needs *ad-hoc relational query* (list sessions by recency, find symbol → definition, join runs to events) — SQLite's SQL + indexes + sqlite-vec colocation win. WAL mode gives concurrent readers + single writer with crash safety. redb is reserved for the *cache* tier where we want a pure-Rust, checksummed, COW KV with snapshot isolation and no SQL overhead (projection snapshots are big opaque blobs keyed by `(session, seq)` — a perfect redb table).

**Why a separate blob CAS.** Inlining file contents and tool outputs into events would bloat the log and the IPC channel and lose dedup. FastCDC content-defined chunking means two near-identical file versions share most chunks; blake3 keys make the store self-verifying and merge-friendly. This is the same git/jj insight applied to agent artifacts.

**Vector store scale path.** sqlite-vec (brute-force) is correct and simple to the "low millions" of vectors — enough for a single large repo's chunks ([sqlite-vec ANN issue](https://github.com/asg017/sqlite-vec/issues/25)). The retrieval store is an **indexer extension** (§7.2), so when a workspace exceeds brute-force comfort, the user installs an ANN-backed indexer (usearch/hnsw_rs) with *no core change* — it registers as a `memory-store`/`indexer` capability and the kernel routes `context.retrieval` to it. Chapter 05 owns the ranking; this chapter owns the *pluggable seam*.

### 4.8 On-disk layout

Per-workspace state lives under `<workspace>/.hide/`; user-global state under the OS app-data dir (`~/Library/Application Support/com.hawking.hide/` on macOS). **Two scopes, deliberately separate** (privacy + portability: a workspace folder is self-contained and shareable; user-global is machine-local).

```
<workspace>/.hide/                         ← per-workspace, self-contained, git-ignorable
├── hide.toml                              ← workspace config (layer 3, §4.10)
├── runtime.lock                           ← {pid, port, model_id, started_at} (flock-guarded)
├── log/                                   ← THE EVENT LOG (system of record, T1)
│   ├── MANIFEST                           ← {schema_version, segment list, head_seq, integrity_root}
│   ├── 000000.seg                         ← segment: length-prefixed NDJSON events
│   ├── 000001.seg
│   ├── …
│   └── 000042.seg.active                  ← current append target (fsync per policy)
├── snapshots/                             ← projection snapshots for fast replay (redb)
│   └── projections.redb                   ← keyed (session_id, seq) → serialized projection state
├── meta.sqlite  (+ -wal, -shm)            ← metadata projection (WAL mode)
├── vectors.sqlite                         ← sqlite-vec embeddings (or ann/ dir for usearch index)
├── blobs/                                 ← content-addressed store
│   ├── ab/cd/abcd…ef                      ← blake3-sharded (first 2 bytes = dirs); FastCDC chunks
│   └── index.redb                         ← chunk → {refcount, size}; GC bookkeeping
├── cache/                                 ← derivable; safe to delete (redb)
│   └── prompt_kv.redb
└── tmp/                                   ← scratch; cleared on boot

~/Library/Application Support/com.hawking.hide/   ← user-global
├── config.toml                            ← user config (layer 2)
├── plugins/                               ← installed extensions
│   ├── <plugin-id>/
│   │   ├── manifest.toml                  ← the extension manifest (§7.2)
│   │   ├── plugin.wasm                    ← WASM component (sandboxed kind) OR
│   │   ├── lib.dylib                      ← native cdylib (trusted kind, signed)
│   │   └── assets/
│   └── registry.sqlite                    ← installed-plugin catalog + grant ledger
├── models/                                ← model registry (runtime weights / .tq, when distributed — LATER)
└── logs/                                  ← host/runtime process logs (rotated)
```

**Segment format** (`*.seg`): a sequence of `[u32 len][len bytes of JSON event]` records; the trailing `.seg.active` is the only file appended to. On `fsync`, the writer also updates an in-memory CRC chain; `MANIFEST.integrity_root` is the blake3 of the segment hashes (Merkle-ish), so corruption is detectable per-segment and recovery rolls back to the last intact record (the redb-style "detect partial commit and roll back" guarantee, applied to the log).

**Why segmented files, not one DB table, for the log.** Append-only segmented files are the simplest thing that gives O(1) append, sequential replay (mmap + scan), trivial archival (move cold segments to cold storage or compress), and no write-amplification. The *indexed* view (query "all `tool.call` in run R") is the SQLite/fjall mirror, rebuilt from segments. This mirrors OpenHands' "one file per event" and Zed's "operation rows," choosing segmented files over per-event files to avoid inode blowup on million-event sessions.

### 4.9 Async / threading model & backpressure

**Runtime topology.**

- **Host:** multi-threaded tokio runtime (worker threads = physical cores − 1, leaving headroom for the WebView and the runtime sidecar's CPU-side work).
- **Single-writer event-log task:** owns the log. Receives `AppendEvent` messages on a **bounded** mpsc channel (capacity 8192). Assigns `seq`, serializes, writes to the active segment, applies the fsync policy, then broadcasts. *This is the only writer* — it is the global ordering oracle (T1). It is CPU-light (serialize + write); if it ever becomes a bottleneck, batching multiple appends per fsync is the lever (see fsync policy).
- **Session workers:** one tokio task per active agent run. They drive the agent loop, call the runtime over HTTP (each its own SSE connection — concurrency without pipe contention, §4.3), dispatch tools, and *produce events* (send to the writer). Many run in parallel (T9 — "many parallel agents" is a core primitive, e.g. fan-out a refactor across 8 files = 8 workers). Bounded by a semaphore (default = runtime `max_batch_size`, since the runtime batches; configurable).
- **Projection task (per live session):** subscribes to the bus, folds events into derived state, emits `UiEvent`s on the Channel. CPU-bound bursts (large diffs) run on `spawn_blocking` to avoid stalling the reactor.
- **Plugin executor pool:** WASM plugins run on a dedicated `spawn_blocking` pool (wasmtime calls are sync); each call is fuel+epoch-bounded (§7.4). Trusted Rust plugins run inline if cheap, on the pool if not.
- **Runtime sidecar:** its own process, its own threads, its own GPU queue — opaque to the host except over HTTP.

**The bus.** Live fan-out is `tokio::sync::broadcast` per session. `broadcast` has bounded capacity and **lagging-consumer semantics**: a slow receiver that falls behind gets a `RecvError::Lagged(n)` telling it how many it missed ([tokio broadcast docs]). HIDE's policy:

- **The projection engine must not lag-drop authoritative state.** If it gets `Lagged(n)`, it does **not** try to consume the missed messages from the broadcast (they're gone); instead it **falls back to the durable log**: re-reads `(last_applied_seq, head]` from segments and resumes. The broadcast is a *fast path*; the log is the *correctness path*. This is the key insight that makes lag safe: *every consumer can always recover from the log* (T2), so the live channel is allowed to drop.
- **Indexer / memory / observability subscribers** are *allowed* to lag and catch up lazily from the log (they record `last_applied_seq` in the metadata DB). They are eventually consistent by design.

**Backpressure ladder** (T8 — every hop bounded):

```
WebView intent ──(Tauri command, naturally backpressured: ack-then-events)──▶ kernel
kernel append  ──(bounded mpsc 8192 → AWAIT on full = producer slows)──────▶ log writer
log writer     ──(broadcast cap 1024 → lag = consumer recovers from log)───▶ bus subscribers
projection     ──(Channel cap 4096 → COALESCE on pressure)─────────────────▶ WebView
session worker ──(HTTP/TCP flow control + per-slot SSE)────────────────────▶ runtime sidecar
runtime SSE    ──(token events → bounded → coalesced as token_batch)───────▶ projection
```

- **Producer-slows (the default):** appends to the log block when the writer's mpsc is full. This naturally throttles a runaway agent — it cannot produce events faster than they can be made durable. Correct and simple.
- **Coalesce (the UI path):** the projection→WebView Channel coalesces high-frequency low-value events under pressure: N `token`s → one `token_batch`; rapid `cursor.moved` → latest only; many `tool.progress` → latest fraction. The *log* keeps every individual event; only the *render stream* is coalesced. This decouples render rate from generation rate — a 120 tok/s stream never floods a 60 fps UI.
- **Never unbounded:** there is no `Vec`-as-queue or unbounded channel anywhere in the path. CI lints for `unbounded_channel`.

**fsync / durability policy** (a dial, §4.10 / §9). Three modes:

| Mode | Behavior | Use |
|---|---|---|
| `strict` | fsync after every event append | max durability; default for `tool.call`/`diff.applied` (effect-bearing events) regardless of mode |
| `batched` (default) | fsync every 16 events or 50 ms, whichever first; effect-bearing events force a flush | balanced; loses ≤ a few cosmetic events on power-loss, never an effect |
| `lazy` | fsync every 500 ms | max throughput for non-critical sessions |

The single-writer + batched-fsync design means the writer is never the bottleneck: it amortizes fsync across a burst while *still* flushing before any effect is acknowledged. (The runtime already proves the pattern works at speed — its continuous-batch loop is a single blocking thread that serializes all GPU dispatch.)

### 4.10 Config & profiles (workspace / user / project layering)

**Layered config**, highest-precedence last (deny/override semantics like Tauri caps):

```
Layer 0  built-in defaults            (compiled in; the safe baseline)
Layer 1  enterprise/policy (optional) (admin-pinned; can FORCE values, e.g. "no cloud providers")
Layer 2  user config                  (~/…/config.toml — personal prefs across all workspaces)
Layer 3  workspace config             (<workspace>/.hide/hide.toml — committed-or-not, per-project)
Layer 4  session/runtime overrides    (set via intents at runtime; transient, recorded as events)
```

Resolution is **deep-merge with precedence**, except Layer-1 policy keys marked `locked` which *cannot* be overridden by higher layers (enterprise control). Every effective config value carries provenance (`which layer set this`) for a "why is this on?" inspector — the runtime already does exactly this with its profile `contract()` strings ("profile=fast: … quality: mild trade …").

**Profiles** are named bundles, mirroring the runtime's `RuntimeProfile` pattern (`default`/`fast`/`race`/`efficient`/`exact`) and `WorkloadPack` pattern (`code-completion`/`chat-shared-prompt`/`local-agent-loop`/…). A HIDE **agent profile** bundles: model provider + model, sampling defaults, tool-grant set, context budget, autonomy level (suggest-only ↔ auto-apply), and a runtime workload pack. Profiles are *data*, defined in config or contributed by extensions:

```toml
[profiles.fast-edit]
provider       = "hawking-local"
runtime_pack   = "code-completion"   # → runtime Race profile, greedy, low latency
autonomy       = "auto-apply-with-tests"
context_budget = 16000
tools          = ["fs.read", "fs.write", "shell.run", "test.run"]

[profiles.careful-refactor]
provider       = "hawking-local"
runtime_pack   = "local-agent-loop"
autonomy       = "suggest-only"
context_budget = 64000               # spend lavishly — local, no per-token cost (T9)
tools          = ["fs.read", "fs.write", "search.*"]
```

Switching profile mid-session is an event (`session.profile_changed`), so it is replayable and the timeline shows when behavior changed.

### 4.11 Versioning, upgrade & migration

**Three independently-versioned surfaces:**

1. **Event schema version** (`Event.schema_version`, currently `1`). Additive changes (new kind, new optional field) do **not** bump it — old readers ignore unknown kinds/fields (the `ext` capture, T10). Breaking changes (rename/remove/retype a field) bump it and ship an **event upcaster**.
2. **On-disk layout version** (`MANIFEST.schema_version`). Governs segment format, store layouts.
3. **Plugin API version** (per host-function WIT world / Rust trait, SemVer). Negotiated per-plugin (§7.4) — exactly Zed's `since_vX.Y.Z` WIT-versioning approach.

**Event upcasting (the migration engine).** Because state is *derived by replaying events*, a schema change cannot retroactively rewrite history (events are immutable, T1). Instead, migration is **read-time upcasting**: a registered chain of `upcast(event_at_vN) -> event_at_vN+1` functions transforms old events into the current shape *as they are read* for replay. The on-disk events stay original (auditable); the replay always sees current-shape events. This is the CQRS/event-sourcing-canonical migration strategy and avoids ever rewriting the log.

```rust
pub trait Upcaster: Send + Sync {
    fn from_version(&self) -> u16;
    fn applies_to(&self, kind: &str) -> bool;
    fn upcast(&self, e: Event) -> Result<Event>;   // pure; bumps schema_version by 1
}
// The registry chains upcasters: an v1 event read under v3 runs through v1→v2→v2→v3.
```

A `system.migration` event is appended once when the host first runs a new layout version, recording which upcasters/store-migrations were applied (auditable, T10).

**Store migrations** (SQLite/redb schema): standard versioned forward migrations, run once at boot, each wrapped in a transaction; on failure the boot aborts and the prior version's binary still reads the prior layout (so a bad upgrade is recoverable by reverting the app). Because every projection store is *rebuildable from the log*, the nuclear option for a botched projection migration is "drop and replay" — a guarantee no cloud agent IDE has.

**Backward/forward compat tests** (CI gate): (a) a corpus of v1 logs replays cleanly under the current upcasters; (b) a current-version event with extra unknown fields round-trips losslessly; (c) "delete every projection store, replay, assert byte-identical derived state."

### 4.12 Error taxonomy & supervision

**Error taxonomy** (every error is an `error` event with a stable `taxonomy_code`, and the HTTP edge mirrors the runtime's structured `{error:{message,type,code}}` shape):

| Domain | Code prefix | Examples | Fatal? | Recovery |
|---|---|---|---|---|
| **User input** | `input.*` | `input.invalid_intent`, `input.empty_prompt` | no | reject, surface inline |
| **Runtime** | `runtime.*` | `runtime.unavailable`, `runtime.timeout`, `runtime.oom`, `runtime.5xx` | no (session pauses) | supervisor restart (§4.3); pause runs |
| **Tool** | `tool.*` | `tool.denied` (capability), `tool.failed`, `tool.timeout`, `tool.not_found` | no | record `tool.result{ok:false}`; agent decides |
| **Plugin** | `plugin.*` | `plugin.trap` (WASM fault), `plugin.fuel_exhausted`, `plugin.epoch_timeout`, `plugin.cap_violation` | no (plugin only) | kill instance, surface, disable on repeat |
| **Storage** | `storage.*` | `storage.log_corrupt`, `storage.projection_corrupt`, `storage.disk_full`, `storage.migration_failed` | varies | log-corrupt = roll back to last intact record; projection-corrupt = rebuild from log |
| **IPC** | `ipc.*` | `ipc.channel_closed`, `ipc.webview_gone` | no | WebView reload re-derives from log (T2) |
| **Config** | `config.*` | `config.parse`, `config.locked_override` | no | fall back to lower layer + warn |
| **Internal** | `internal.*` | `internal.invariant` (a replay assertion failed) | yes-ish | safe-mode boot, offer log export |

**Supervision tree** (Erlang-flavored; the host is the root supervisor):

```
Host (root supervisor)
├── RuntimeSupervisor ──── manages the sidecar (restart w/ backoff, K-in-window give-up, §4.3)
├── EventLogWriter ─────── CRITICAL: a crash here is fatal (it's the ordering oracle). On panic,
│                          the host flushes, exports a recovery bundle, and refuses new writes
│                          until restarted (fail-stop, never lose ordering).
├── SessionSupervisor ──── one child per session; a session worker panic is isolated:
│   ├── Session A worker    record `error{fatal:false}`, mark the run failed, OTHER sessions live
│   └── Session B worker
├── ProjectionSupervisor ─ a projection crash → rebuild that projection from the log (T2)
└── PluginHost ──────────── a plugin trap/fuel/epoch fault kills ONLY that instance; repeated
                            faults (3 in 60 s) auto-disable the plugin + surface a banner
```

**Crash recovery (cold start after a hard kill):**

1. Open the log; **scan the active segment to the last intact `[len][bytes]` record** (CRC chain). A torn final record (partial write, the redb-detected case) is truncated; `head_seq` = last intact.
2. Detect **dangling Actions**: any `Action`-class event with no matching `Observation` (`cause`) means a tool/effect was in-flight at crash. Policy: append a synthetic `tool.result{ok:false, reason:"interrupted_by_crash"}` so the causal DAG closes and replay is well-formed; the run is marked `interrupted` and the user is offered "retry this step."
3. **Catch up projections:** each projection replays `(last_applied_seq, head]`. Fast because snapshots bound the tail (§4.7).
4. **Re-attach the runtime** (RuntimeSupervisor boots the sidecar; Booting→Ready).
5. Surface a non-blocking "recovered session at turn N" toast. The user lost *nothing* durable (T2/T9) — at most the few cosmetic events between the last fsync and the crash (mode-dependent, §4.9), never an applied effect.

This is the determinism+persistence dividend (T6): a power-loss mid-refactor resumes exactly where it was, with the full plan, diffs, and tool history intact — *because the log is the truth and effects are recorded, not lost.*

---

## 5. How we EXCEED (local superpowers)

The architecture is the *substrate* for the unfair advantages. Each is wired through a specific seam above.

| Superpower | Architectural seam | Why cloud literally cannot |
|---|---|---|
| **Total recall / forever-editable memory** | Append-only event log (T1) the user owns on disk; memory is a projection over it (§4.5). | Cloud agents persist truncated transcripts on someone else's server with retention limits; they cannot give you an immutable, *local*, fully-auditable, replayable history you own forever. |
| **Deterministic replay / time-travel / resume-across-days** | Snapshot + tail-replay; effects recorded not re-fired (T3); seeds plumbed (T6) (§4.5, §4.12). | Cloud sessions are non-deterministic and non-resumable at the token level — no raw seed control, no checkpoint of *their* KV, no "scrub to event 4,210 and fork." |
| **Many parallel agents, overnight compute, exhaustive logging** | Session-worker-per-run with a semaphore = runtime batch width; the log records *everything* (T9) (§4.9). | Per-token billing + rate limits make "spawn 8 agents and log every token for a week" economically and operationally impossible in the cloud. Local = free + unlimited. |
| **Raw logits / custom samplers / constrained-grammar decode** | `ModelProvider::raw_logits` + `ProviderCaps.grammar`; runtime's `json_mode`/`JsonConstraint::mask_logits` and native `/v1/hawking/tokens` (raw token-id SSE) (§4.3). | Cloud APIs expose at most top-k logprobs and a fixed JSON mode; they never hand you the raw logit vector to run your own sampler or arbitrary grammar/automaton mask. |
| **KV-cache surgery + extended/custom context length** | `ModelProvider::kv_handle`; the runtime's prefix-cache + system-prompt KV bank are first-class (copy/seed/reuse KV across slots) (§3, §4.3). | Cloud context is a fixed black box; you cannot reach in to splice a KV prefix, persist it across days, or extend the window past the served limit. |
| **LoRA hot-swap + fine-tune-at-Condense** | `ModelProvider::load_lora`; the runtime can hot-load adapters; *Hawking Condense* co-designs the weights (T7). | Cloud cannot let you hot-swap a personal LoRA per-workspace mid-session, let alone co-design the quantized weights to your harness. |
| **Draft + speculative decode tuned to the harness** | Runtime already has spec-decode + a proposal market; exposed via provider caps (§3). | Cloud spec-decode is internal and invisible; you cannot supply your own draft model or n-gram proposer. |
| **Full local access (FS/OS/processes/GPU), no sandbox jail** | Tool dispatcher with Tauri caps v2 over real OS capabilities; persistent daemons; real env (§4.1, §4.3). | Cloud agents run in ephemeral sandboxes with no persistent local daemon, no real GPU access, no durable local FS that *is* your project. |
| **Privacy / nothing leaves the machine** | Everything on-device; `redactions` scrub secrets before durability; cloud providers are *opt-in plugins* a policy layer can forbid (§4.6, §4.10). | A cloud-first IDE *is* the exfiltration path; HIDE's default is air-gappable. |

**The explicit "cloud literally cannot do this" list:** (1) hand you the raw logit vector for a custom sampler; (2) let you splice/persist/extend the KV cache; (3) give you a byte-immutable, locally-owned, replayable event log of every token and effect; (4) deterministically resume a token-level session across days from a seed; (5) hot-swap a per-workspace LoRA mid-session; (6) co-design the quantized model weights to your IDE's harness; (7) run unlimited parallel agents + overnight compute at zero marginal cost; (8) operate fully air-gapped with no data egress. Every one of these is a *seam in §4*, not an afterthought.

---

## 6. Failure modes / edge cases / mitigations

| # | Failure / edge case | Mitigation |
|---|---|---|
| F1 | **Log grows unbounded** (long-lived session, millions of events). | Segment + snapshot + key-compaction of high-churn cosmetic kinds; cold-segment archival; semantically-important kinds never compacted (§4.5). Compaction is itself recorded (integrity preserved). |
| F2 | **Torn write / partial event on power-loss.** | CRC chain per record; on boot, truncate to last intact record (redb-style detect-and-rollback) (§4.8, §4.12). Effect-bearing events fsync *before* ack, so no applied effect is ever lost (§4.9). |
| F3 | **Runtime hang (GPU stall) — no token for N seconds.** | Per-request watchdog (`max_stall_ms` already in `GenerateRequest`) + supervisor `/healthz` probe → Degraded → restart; in-flight runs pause with `runtime.unavailable`, not fail (§4.3, §4.12). |
| F4 | **Runtime OOM on a too-big model** (the 32B-on-18GB tier). | Runtime aborts pre-allocation when working set exceeds `memory_limit_mb` (already in `EngineConfig`); supervisor surfaces `runtime.oom`; the model-provider plugin can fall back to a smaller model or cloud provider (§4.3). *(32B `.tq` is runtime-testing, not shell-gating — §1.)* |
| F5 | **WebView reload / crash mid-stream.** | View holds no authoritative state (T2); on reconnect the kernel replays the projection from the log → identical UI; the Channel re-subscribes (§4.4, §4.5). |
| F6 | **Slow UI consumer floods (120 tok/s into 60 fps).** | Render-coalescing: `token`→`token_batch`, latest-only for cursor/progress; the log keeps every event (§4.9). |
| F7 | **Two IDE windows spawn two runtimes / port clash.** | `runtime.lock` (flock-guarded) → shared-daemon reuse or distinct ephemeral ports; lock records `{pid,port,model_id}` (§4.3, §4.8). |
| F8 | **Plugin infinite-loops or allocates wildly (WASM).** | Fuel metering (deterministic) + epoch deadline (wall-clock trap) + `ResourceLimiter` memory cap; 3 faults/60 s auto-disables (§4.12, §7.4). |
| F9 | **Malicious/over-reaching plugin** (reads `~/.ssh`, exfiltrates). | Capability negotiation: no ambient authority (T4); fs/net/shell are host-granted, path/host-scoped; deny beats allow; sandboxed WASM cannot syscall except via granted host fns (§7). |
| F10 | **Dangling Action at crash** (tool was mid-write). | Synthetic `tool.result{interrupted}` closes the causal DAG; run marked `interrupted`; user offered retry (§4.12). |
| F11 | **Replay re-fires a destructive effect** (the classic event-sourcing footgun). | T3 is enforced in code: the replay engine has *no path* to the tool dispatcher; it folds recorded `Observation` bytes only. Crossing into live mode is an explicit, user-initiated transition (§4.5). A CI test asserts replay performs zero filesystem/shell/network syscalls. |
| F12 | **Projection store corrupted/schema-drifted.** | Drop and replay from the log (T2); store migrations are forward-only + transactional; nuclear rebuild always available (§4.7, §4.11). |
| F13 | **Clock skew makes `ts` non-monotonic.** | `seq` (single-writer-assigned) is the *only* ordering authority; `ts` is decoration. Replay never sorts by `ts` (§4.6). |
| F14 | **Event-kind schema mismatch between core and a plugin that emits it.** | Kinds register a JSON Schema at install; append-time validation rejects malformed plugin events with `plugin.cap_violation`; unknown-but-well-formed kinds are stored and ignored by non-subscribers (T10) (§4.6, §7.2). |
| F15 | **Disk full mid-append.** | `storage.disk_full` error; writer enters read-only mode (replay/scrub still works); UI banner; effect-bearing intents rejected until space freed. |
| F16 | **Secret leaks into an event** (API key in shell output). | Redaction pass before durability (`redactions` records scrubbed paths); secret-scanner runs on `tool.result` payloads (§4.6). |
| F17 | **Two stores disagree (metadata vs log).** | The log wins, always (T2). `last_applied_seq` mismatch triggers a catch-up replay; a divergence assertion (`internal.invariant`) triggers safe-mode + rebuild (§4.7). |
| F18 | **Migration upcaster bug corrupts replay.** | Upcasting is read-time and pure; on-disk events are never rewritten (§4.11). A bad upcaster is a code fix + re-run, never data loss; the original log is intact. |

---

## 7. Extensibility / the plugin spine

This is the over-engineering mandate made concrete: **every capability is a registered extension.** The core ships a registry, a manifest format, a capability negotiator, and a host-function surface — and *no hard-coded feature lists*.

### 7.1 Extension kinds (the open registry)

The core defines *extension-point kinds* but never enumerates the *extensions*. Each kind is a contribution surface other chapters consume:

| Kind | Contributes | Consumed by (chapter) |
|---|---|---|
| `tool` | An agent tool (name, JSON-schema args, handler) | Agent loop (Ch.02), Tools (Ch.04) |
| `panel` | A UI panel/view (id, mount point, web bundle) | Editor surface (Ch.03) |
| `agent` | An agent persona/strategy (planner policy) | Agent loop (Ch.02) |
| `model-provider` | A `ModelProvider` impl (§4.3) | Runtime client (this ch.), Ch.06 |
| `indexer` | A code/asset indexer (parse + chunk + symbol graph) | Context/retrieval (Ch.05) |
| `memory-store` | A retrieval/embedding backend (vector or other) | Context/retrieval (Ch.05) |
| `command` | A command-palette command + keybinding | Editor surface (Ch.03) |
| `event-kind` | A new event `kind` + its payload schema | Event bus (this ch., §4.6) |
| `formatter`/`linter`/`lsp` | Language tooling adapters | Editor surface (Ch.03) |
| `context-server` | An MCP server connection (tools/resources) | Tools (Ch.04) |

Adding any of these touches **zero** `core/` files — the litmus test (§2) holds. New kinds *themselves* can be registered by meta-extensions (a kind is just a registry entry with a schema), so even the kind list is open.

### 7.2 The extension manifest — CROSS-CUTTING CONTRACT

> **Every extension is described by one `manifest.toml`.** This is the second contract other chapters bind to (alongside the event schema). It is modeled on Zed's `extension.toml` + VS Code's `contributes` + WASI/Component-Model capability declarations, unified.

```toml
# ── Identity (immutable once published) ─────────────────────────────────────
id              = "acme.rust-tools"      # globally unique, reverse-DNS-ish
name            = "Acme Rust Tools"
version         = "1.4.2"                # SemVer
schema_version  = 1                      # manifest-schema version (this doc = 1)
authors         = ["Acme <dev@acme.io>"]
description      = "Rust refactors, clippy panel, and a cargo tool."
repository       = "https://github.com/acme/rust-tools"
license          = "Apache-2.0"

# ── Runtime kind & entry point ──────────────────────────────────────────────
[runtime]
kind    = "wasm"                         # "wasm" (sandboxed) | "native" (trusted, signed) | "external" (subprocess)
entry   = "plugin.wasm"                  # cdylib path for native; binary for external
api     = ">=0.6, <0.8"                  # host API (WIT world / trait) version range — NEGOTIATED (§7.4)
abi     = "component"                    # "component" (WASI 0.2/0.3) for wasm
trust   = "community"                    # "first-party" | "verified" | "community" — gates default grants

# ── Contributions (what this extension PROVIDES) ────────────────────────────
[[contributes.tools]]
name        = "cargo_check"
description  = "Run cargo check and return diagnostics."
args_schema = "schemas/cargo_check.json" # JSON Schema for the tool args (validated)
export      = "tool_cargo_check"          # the WIT/trait export implementing it

[[contributes.panels]]
id          = "acme.clippy"
title        = "Clippy"
mount        = "right-dock"
bundle       = "ui/clippy.js"

[[contributes.commands]]
id          = "acme.fix-all"
title        = "Acme: Fix All Clippy"
keybinding   = "cmd+alt+l"

[[contributes.event_kinds]]
kind        = "acme.clippy.run"
payload_schema = "schemas/clippy_run.json"  # registered → append-time validation (§4.6, F14)

[[contributes.model_providers]]      # optional; a provider plugin
id          = "acme.local-mlx"
caps        = ["streaming", "embeddings"]    # declares ProviderCaps it satisfies

[[contributes.indexers]]
id          = "acme.rust-scip"
languages    = ["rust"]
emits        = ["symbol-graph", "chunks"]

# ── Activation (lazy, like VS Code activationEvents) ─────────────────────────
[activation]
events = ["onLanguage:rust", "onCommand:acme.fix-all", "onStartupFinished"]

# ── Capabilities REQUESTED (what this extension NEEDS — no ambient authority) ─
# Mirrors WASI/Zed: the host grants each per-scope; DENY beats ALLOW; the user
# sees and approves these at install (the grant ledger, §7.3).
[[capabilities]]
kind  = "fs:read"
paths = ["$WORKSPACE/**"]            # scoped; $WORKSPACE/$HOME placeholders resolved by host

[[capabilities]]
kind  = "fs:write"
paths = ["$WORKSPACE/**/*.rs"]       # narrower than read — least privilege

[[capabilities]]
kind    = "process:exec"
command = "cargo"                    # only cargo, not arbitrary shell
args    = ["check", "clippy", "--message-format=json"]   # arg allow-list (regex allowed)

[[capabilities]]
kind  = "net:fetch"
hosts = ["crates.io", "*.crates.io"]   # host-scoped egress (default: NONE)

[[capabilities]]
kind  = "events:subscribe"
kinds = ["diff.applied", "file.changed_external"]   # which event kinds it may observe

[[capabilities]]
kind  = "events:emit"
kinds = ["acme.clippy.run"]          # may only emit kinds it registered

[[capabilities]]
kind  = "model:infer"                # may call the model provider (agent-style plugins)
providers = ["hawking-local"]

# ── Resource limits (enforced for wasm; advisory for native) ────────────────
[limits]
fuel_per_call_M   = 500              # wasmtime fuel budget (millions) per host call
epoch_deadline_ms = 2000             # wall-clock trap
max_memory_mb     = 256              # ResourceLimiter cap
```

**Contract guarantees** (binding on all chapters):

1. **Capabilities are declarative and scoped.** No extension gets fs/net/shell/model access it didn't declare; the host enforces the scope (paths/hosts/commands/args) at the call boundary; **deny precedes allow** (Tauri-caps semantics). A WASM extension *physically cannot* do more — it has no syscalls, only granted host functions (§7.4).
2. **Contributions register; they don't patch.** A tool/panel/command/event-kind is *added to a registry*, never injected into core code. The registry is queryable (`registry.list("tool")`) and the agent kernel resolves tools/providers/indexers through it at runtime.
3. **Event kinds a plugin emits must be registered with a schema**, validated at append (§4.6, F14). A plugin cannot poison the log with malformed events.
4. **API version is negotiated**, not assumed (§7.4). A plugin built against host API `0.6` runs on host `0.7` if `0.7` is back-compatible; otherwise it is quarantined with a clear `plugin.cap_violation`/version error.
5. **`trust` gates defaults, not capabilities.** `first-party` plugins may get broader default grants; `community` plugins start with *nothing* until the user approves the requested capabilities. Trust never *bypasses* the negotiator — it only changes the *default* answer.

### 7.3 The capability negotiator & grant ledger

At install/first-activation, the host:

1. Parses the manifest's `[[capabilities]]`.
2. Computes the **effective grant** = (requested) ∩ (policy layer allows) ∩ (user approves), minus (any deny). Policy layer (§4.10, Layer 1) can hard-forbid kinds (e.g. enterprise: `net:fetch` denied globally).
3. Records the decision in the **grant ledger** (`registry.sqlite`): `{plugin_id, capability, scope, granted_by, granted_at, grant_id}`. Every `tool.call` event references its `capability_grant_id` (§4.6) — so the log proves *which grant authorized which effect* (audit, T9/privacy).
4. Re-prompts on *capability escalation* (a plugin update that requests more) — never silently widens.

This is strictly stronger than VS Code (full Node by default) and at least as strong as Zed (allow-listed host fns), with the addition of an **auditable per-effect grant trail** in the event log that neither has.

### 7.4 Host functions & the WASM sandbox

**Trusted (native Rust) plugins** link a host-provided trait object surface and run in-process (first-party, signed). **Sandboxed (WASM Component) plugins** are the default for third-party code:

- Run on **wasmtime** with the **Component Model**; the host defines a versioned **WIT world** (`hawking:hide@0.7`) whose **imports** are the granted host functions (fs/net/process/model/events/log-query/ui) and whose **exports** are the contribution handlers (the tool/indexer/command impls). This is exactly Zed's `zed:extension` pattern, widened to HIDE's kinds.
- **Capability grants map to which imports the `Linker` satisfies.** A plugin without `net:fetch` simply has no `net.fetch` import linked — calling it traps. No ambient authority (T4).
- **Resource bounds:** `Config::consume_fuel` (deterministic metering) + `Config::epoch_interruption` with a host timer thread (`Engine::increment_epoch`) for wall-clock deadlines + `StoreLimits` memory cap. A runaway/hostile plugin is *bounded by construction* (F8).
- **API versioning:** the host records the WIT world version a plugin was compiled against and instantiates only against a compatible host world (Zed's `since_vX.Y.Z` approach). WASI 0.2 today; 0.3 `stream<T>`/`future<T>` adopted when the toolchain settles (§8).

```
   manifest [[capabilities]]            wasmtime Store (per plugin instance)
            │                            ┌───────────────────────────────────┐
   negotiator computes grants ──────────▶│  fuel budget + epoch deadline +   │
            │                            │  ResourceLimiter(max_memory)      │
   Linker.add_to_linker(ONLY granted) ──▶│  imports: fs.read?(scoped),       │
                                          │           net.fetch?(host-scoped),│
   plugin.wasm exports ─────────────────▶│           process.exec?(allow-list)│
   (tool/indexer/command handlers)        │           model.infer?, events.*  │
                                          └───────────────────────────────────┘
```

### 7.5 The explicit extension points (summary the rest of the bible binds to)

- **Tools** (Ch.04): register `tool` kind; args JSON-schema-validated; effects recorded as `tool.call`/`tool.result` with grant id.
- **Panels/Commands** (Ch.03): register `panel`/`command` kind; mount points + keybindings; UI bundles loaded into the WebView.
- **Agents** (Ch.02): register `agent` kind (planner strategy); drives the session-worker loop.
- **Model providers** (Ch.06): register `model-provider` kind implementing the `ModelProvider` trait + `ProviderCaps`; the Hawking-local provider is the default, cloud/other are plugins.
- **Indexers + Memory stores** (Ch.05): register `indexer`/`memory-store` kinds; the kernel routes `context.retrieval` through them; the vector-store backend is swappable (sqlite-vec → ANN) with no core change.
- **Event kinds** (any chapter): register `event-kind` with a payload schema; emit/subscribe gated by capability.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by (impact × feasibility); each tagged PROVEN-substrate vs SPECULATIVE with difficulty/impact.

1. **WASM Component plugins on wasmtime + WIT (PROVEN substrate; difficulty M, impact HIGH).** Zed ships this for a real IDE today ([Zed extensions](https://zed.dev/docs/extensions/developing-extensions)); wasmtime fuel/epoch/ResourceLimiter are documented and stable ([wasmtime](https://docs.wasmtime.dev/examples-interrupting-wasm.html)). The only "moonshot" is the *breadth* of HIDE's WIT world (panels+providers+indexers, not just languages). **Do it.**

2. **Deterministic full-session replay with effect-recording (PROVEN substrate; difficulty M, impact HIGH).** Event-sourcing + the replay-side-effect rule are battle-tested (OpenHands, Elm, Fowler) ([OpenHands replay](https://github.com/All-Hands-AI/OpenHands/blob/0.40.0/openhands/controller/replay.py)). HIDE's edge: pairing it with the *bit-identical greedy runtime* (T6) for *re-derivable* (not just re-playable) runs — a guarantee cloud cannot match. **Do it.**

3. **Shared runtime daemon across windows + remote runtime (PROVEN substrate; difficulty M, impact MED-HIGH).** Because the runtime is HTTP (§4.3), one `hawking serve` can back many IDE windows, or live on a beefier box on the LAN. The lock-file + provider abstraction already accommodate it.

4. **CRDT-backed multiplayer sessions (PROVEN substrate; difficulty H, impact MED).** The event log *is* an op-log; layering Automerge/Yjs-style CRDT merge (Zed proves it for buffers) enables two engineers (or a human + N agents) on one live session ([Zed CRDTs](https://zed.dev/blog/crdts), [Automerge 3.0](https://automerge.org/blog/automerge-3/)). Reserve `id`-as-ULID + per-event `parent` now so the schema is merge-ready; defer the merge engine.

5. **WASI 0.3 async streams for plugins (SPECULATIVE→shipping; difficulty M, impact MED).** WASI 0.3.0 ratified Jun 2026 with `stream<T>`/`future<T>` ([WASI 0.3](https://bytecodealliance.org/articles/WASI-0.3)); adopting it lets plugins stream (e.g. an indexer streaming chunks) natively. Gate on toolchain maturity; design the WIT world to add async exports later.

6. **Speculative UI / optimistic projection (SPECULATIVE; difficulty M, impact MED).** Render the *predicted* projection from a draft model's tokens while the target verifies, reconciling on the authoritative event — the UI analog of the runtime's spec-decode. Local-only (needs raw draft tokens, T7). Risky (flicker); prototype behind a flag.

7. **Content-defined-chunked, dedup-everything blob store with VectorCDC (SPECULATIVE; difficulty M, impact MED).** FastCDC is proven ([fastcdc-rs](https://github.com/nlfiedler/fastcdc-rs)); VectorCDC's 8–26× SIMD speedup ([arXiv 2508.05797](https://arxiv.org/pdf/2508.05797)) would make "snapshot every file version forever" cheap enough to be the default. Adopt FastCDC now, VectorCDC when a crate lands.

8. **Activation-steering / persona vectors as a `model-provider` capability (SPECULATIVE; difficulty H, impact MED).** Expose the runtime's activation hooks (T7) as a provider cap so an `agent` extension can steer style/behavior without prompt engineering. Research-grade; reserve the cap name, defer.

9. **Local SCIP-grade incremental symbol graph as the default indexer (PROVEN substrate; difficulty M-H, impact HIGH for Ch.05).** SCIP unblocks file-incremental indexing ([SCIP](https://sourcegraph.com/blog/announcing-scip)); tree-sitter re-parses in O(log n). This is Chapter 05's core, but the *seam* (an `indexer` extension emitting a symbol graph + chunks) is reserved here. **Plan it.**

---

## 9. Open questions / dials

| # | Question / dial | Default | Trade-off |
|---|---|---|---|
| Q1 | **Open-string event `kind` + registered schema vs a closed Rust enum.** | Open-string + schema (§4.6) | Extensibility (WASM plugins can emit kinds; no core edit per kind) **vs** compile-time exhaustiveness. Mitigation: codegen typed accessors for core kinds; schema-validate the rest. |
| Q2 | **Event-log storage: segmented files vs an embedded DB (fjall) as primary.** | Segmented files primary, fjall as the *indexed mirror* (§4.8) | Simplicity/archival/O(1)-append **vs** built-in indexed query. Revisit if segment management proves fiddly at 10M+ events. |
| Q3 | **fsync policy default.** | `batched` (16 events / 50 ms; effects force-flush) (§4.9) | Durability **vs** throughput. `strict` for paranoid users; `lazy` for ephemeral sessions. |
| Q4 | **Agent kernel in-process vs 5th process.** | In-process crate (§4.1) | Hot-path latency (no IPC hop for event emission) **vs** crash isolation of the kernel. Chosen: in-process, because the *runtime* (the crash-prone part) is already isolated; the kernel is mostly orchestration. |
| Q5 | **Vector store: sqlite-vec (brute-force) vs ANN from day one.** | sqlite-vec; ANN is a pluggable indexer (§4.7) | Simplicity + "good to low-millions" **vs** large-monorepo scale. The plugin seam means we don't have to choose globally. |
| Q6 | **WASM-only third-party plugins vs allowing native `cdylib`.** | WASM default; native only for `first-party`/signed `verified` (§7.2) | Safety (sandbox) **vs** raw speed / FFI reach. Native escape hatch exists but is gated by trust + signature. |
| Q7 | **Coalescing granularity for the UI stream.** | token→token_batch, latest-only cosmetics (§4.9) | Render smoothness **vs** per-token fidelity in the UI (the *log* always has every token). Tunable per panel. |
| Q8 | **Snapshot cadence for projections.** | every 512 events or 30 s per session | Replay speed after crash **vs** snapshot write cost / disk. Dial per durability mode. |
| Q9 | **Compaction policy: which kinds are key-compacted vs immortal.** | Cosmetic kinds (cursor/progress) compactable; turns/plans/tools/diffs immortal (§4.5) | Log size **vs** historical completeness. Per-kind flag in the kind registry. |
| Q10 | **Where redaction/secret-scanning runs** (pre-durability vs at-read). | Pre-durability (§4.6, F16) | Privacy (secret never hits disk) **vs** append latency. Pre-durability chosen; scanner is a pluggable hook. |
| Q11 | **Cross-session shared memory store.** | Per-workspace by default; opt-in user-global memory | Privacy/portability (workspace self-contained) **vs** convenience (recall across projects). Mirrors Continue.dev's "sessions are global" pain point — we default *local* and let users opt up. |

---

## 10. Cross-references

- **Chapter 02 — Agent Kernel & Reasoning Loop:** binds to the event schema (`turn.*`, `plan.*`, `tool.*`), the session-worker model (§4.9), the `agent` extension kind (§7.1), and the replay/fork semantics (§4.5). The runtime's spec-decode/proposal-market internals (`docs/plans/hawking_event_horizon_*.md`) are *runtime-side* and reached only via the `ModelProvider` HTTP contract (§4.3).
- **Chapter 03 — Editor Surface (Monaco) & Diff/Apply:** binds to `panel`/`command` kinds (§7.1), the projection stream (Wire B, §4.4), and `diff.*`/`file.*` events (§4.6).
- **Chapter 04 — Tools & MCP:** binds to the `tool`/`context-server` kinds, the capability negotiator + grant ledger (§7.3), and `tool.call`/`tool.result` recording (effects, T3).
- **Chapter 05 — Context, Retrieval & Memory:** binds to `indexer`/`memory-store` kinds, the vector/blob/metadata stores (§4.7), and `context.*`/`memory.*` events. Owns the SCIP/tree-sitter indexer (§8 #9) behind this chapter's `indexer` seam.
- **Chapter 06 — Model Runtime & Providers:** owns the runtime internals; binds to `ModelProvider`/`ProviderCaps` (§4.3) and the local superpowers (raw logits, KV surgery, LoRA, grammar, spec-decode — §5). The HTTP surface (`hawking-serve`) is the boundary.
- **Repo runtime sources (ground truth):** `crates/hawking-core/src/engine.rs` (`Engine`, `StreamEvent`, `GenStats`, `GenerateRequest`, `SamplingParams`), `crates/hawking-serve/src/{lib,http}.rs` (OpenAI + native endpoints, continuous batch loop, profiles, KV bank), `crates/hawking-core/src/json_constrain.rs` (grammar mask), `crates/hawking-core/src/sidecar.rs` (the `.hawking` weight-sidecar format — note: distinct from HIDE's *runtime process* sidecar).

---

### Cross-cutting decisions this chapter fixes for the whole bible

1. **The `Event` envelope (§4.6) is the universal currency.** `{schema_version, seq, id(ULID), session_id, run_id, parent, cause, ts, source, actor, kind(dotted, namespaced), payload(flatten), redactions, ext(forward-compat)}`. `seq` is the only ordering authority; `id` is the stable cross-machine identity. Action/Observation classing governs replay.
2. **State is derived; the append-only log is the system of record** (T1/T2). Every store is a rebuildable projection. Replay folds recorded outcomes and **never re-fires effects** (T3) — enforced by giving the replay path no access to the tool dispatcher.
3. **The runtime is a stable localhost HTTP surface** (§4.3), reached via the `ModelProvider`/`ProviderCaps` abstraction. The IDE never links the engine. Runtime-completion items (32B `.tq`, native serving) are *later / not shell-gating*.
4. **The extension manifest (§7.2) + capability negotiator (§7.3) are the universal plugin contract.** Every capability (tool/panel/agent/provider/indexer/memory-store/command/event-kind) is a registered extension with declared, scoped capabilities; **deny beats allow**; no ambient authority; WASM-sandboxed by default with fuel/epoch/memory bounds; per-effect grant ids recorded in the log. Adding any capability touches zero `core/` files.
5. **Token streams ride `ipc::Channel<T>` (ordered), never `emit/listen`;** every producer→consumer hop is a bounded channel with a defined overflow policy (producer-slows for correctness paths, coalesce for the UI path); the single-writer log task is the global ordering oracle.
