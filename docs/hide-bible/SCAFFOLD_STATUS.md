# HIDE Scaffold — Completion Status

> **Date:** 2026-06-27
> **State:** the 11-crate HIDE backend is complete as a runnable, headless **vestigial structure** — every bible contract has a real, tested implementation or an explicitly documented seam. Nothing here is wired to a live model yet (that waits on Hawking-proper: Condense + the 32B `.tq`), and there is no GUI yet (the Tauri frontend is deferred). But the agent loop and all capabilities are built; "no further development on the agent loop" is satisfied.
>
> This supersedes the *before* picture in [SCAFFOLD_AUDIT.md](SCAFFOLD_AUDIT.md), which captured the original Codex skeleton.

## How it was built

Dependency-ordered waves of parallel agents, each crate **adversarially depth-reviewed** against its bible chapter and then **repaired** where the reviewer found shallowness. Every wave was committed + pushed to `main` only after `cargo check --workspace` was green and the crate tests passed. Commit range: `06fefd6 … b68b156`.

| Wave | Commits | Crates |
|---|---|---|
| Foundation (WP-1) | `06fefd6` | hide-core Event contract → bible §4.6 |
| B + repair | `3e564a5`, `904824d` | orch, index, context, tools, security |
| C + repair (the kernel) | `107c3d4`, `b4397df` | hide-kernel |
| D + repair | `3e7557b`, `d0ba6e8` | fleet, research, personalize |
| E + repair (the host) | `abdca78`, `b68b156` | hide-backend |

**~410 tests passing** across the HIDE crates. Workspace compiles clean; the live runtime crates (`hawking-core`/`hawking-serve`) were never touched (T5 — HIDE talks HTTP only).

## Per-crate real state

| Crate | Bible | What's real now | Review |
|---|---|---|---|
| `hide-core` | ch.01 | Open `Event` envelope (`payload: Value` + `payload_as::<T>()`, `cause`/`actor`, ULID, blake3 chain), canonical `ToolCall/Result/Error`, permission engine, CAS persistence | foundation |
| `hawking-orch` | ch.06 | reqwest streaming `InferenceClient` (all 3 endpoints + embed), confidence-gated escalation cascade, energy/thermal admission scheduler, LoRA adapters, real grammar matcher | solid 5/5 |
| `hawking-index` | ch.05 | blake3 merkle (O(changed)+rename), tree-sitter Rust/Python/TS def+ref, rusqlite WAL+FTS5, petgraph PageRank repo-map, embeddings+RRF+rerank, notify daemon | acceptable 4/5 |
| `hawking-context` | ch.04 | Budgeted knapsack compiler (realize/degrade dispatch, head/tail ordering), tokenizer counts, full manifest, rusqlite memory (FTS5+vectors), KvStore + PrefixKey interop | acceptable 4/5 |
| `hide-tools` | ch.03 | EXEC_NONZERO-is-data, tokio sandboxed shell (SBPL+sandbox-exec, fail-closed off-macOS), tiered edit (search-replace + unified-diff apply), 22-tool catalog incl. git worktree trio, real MCP (stdio + HTTP) | solid 5/5 |
| `hide-security` | ch.10 | blake3 audit chain + signed anchors, regex+entropy redaction (guillemet markers → `Event.redactions`), aes-gcm at-rest w/ keychain-wrapped key, sandbox allowlist + layout fail-closed | solid 5/5 |
| `hide-kernel` | ch.02 | **The real agent loop**: Planner→Executor→Verifier FSM, declared-acceptance oracles, 8-oracle deterministic suite shelling to real cargo/git, VerificationGate (deterministic outranks probabilistic), Governor over all A.5 caps, Repair/Replan/Paused, Replay-skips-effects. Flagship test drives a real cargo-check failure → Repair | **audited GENUINE** |
| `hide-fleet` | ch.09 | git worktree lifecycle, FleetGovernor (two-pool + thermal + spawn-breaker), schedule_tick launching real kernel runs, 3-way merge funnel, queue-as-event-projection, JSON-RPC/WS remote, fleetview | solid 5/5 |
| `hawking-research` | ch.08 | RuntimeClient over orch, blake3 CAS evidence + sound citation re-verify, petgraph KG (Local/Global/Path + entity resolution), 8-state pipeline FSM + checkpoint ledger, arXiv adapter, corroboration verify | acceptable 4/5 |
| `hide-personalize` | ch.11 | eval runs real oracles + EvalMiner, RLEF reward derivation, MetaRouter retrieval routing, Tier-1 `StaticProjectSimulator`, KvShareGroup handoff seam, scrub-on-write dataset | solid 4/5 |
| `hide-backend` | ch.01/07 | **Runnable host**: RuntimeSupervisor (boots/supervises `hawking serve`, /healthz poll, backoff, fail-closed PID lock), HTTP ModelProvider, push UiEvent broadcast bus, intent validation+rejection+interrupt signalling, session registry, time-travel scrub/fork, fleet wired | solid 5/5 |

## Deliberately deferred seams (documented, not faked)

These are intentionally left as clean, documented trait/seam boundaries — they need either the live runtime, the GUI, or are post-shell moonshots per the bible:

- **Live model wiring** — every model call goes through `InferenceClient`/`ModelProvider`/`KernelRuntimeClient`; tests use stubs. Wire to a real `hawking serve` once the 32B `.tq` exists.
- **Tauri frontend** — `CommandRouter::handle` is transport-agnostic; a `#[tauri::command]` layer wraps it later. No `tauri` dep pulled in.
- **WASM plugin host** — `ExtensionRegistry` holds descriptors; the wasmtime component host is post-shell. No `wasmtime` dep.
- **KV-cache handoff** (`hide-personalize::kv_handoff`, `hawking-context::kv`) — `KvShareGroup`/`KvStore` contracts + `PrefixKey` byte-interop are defined; the live bridge to in-tree `copy_kv_prefix_to_slot` is a seam.
- **Moonshots** (ch.11) — DSPy/ADAS prompt optimization and the Tier-2 *learned* world-model are labeled post-shell seams; RLEF reward derivation + dataset assembly are real, on-device training is deferred.
- **PDF full-text ingest** (research) — the JSON-API source adapter (arXiv) is real; PDF parsing is a seam.
- **Cross-machine cluster pool** (fleet TIER-4) — single-machine fabric is real; the remote WS protocol exists; the cluster pool is omitted per the bible.

## When you return to HIDE

1. **Wire the live runtime**: point `RuntimeSupervisor` at the real `hawking` binary + a Condense'd `.tq`; the kernel will generate against it through the existing `ModelProvider`. Run the thesis gate (`hawking-eval`, to be built per ch.13) to get the GO/KILL verdict.
2. **Build the Tauri shell**: wrap `CommandRouter` in `#[tauri::command]`, subscribe the frontend to the `UiEvent` broadcast bus, render the Context Stack from `ContextManifest`.
3. **Open the deferred seams** as needed (plugin host, KV handoff, moonshots).

Residual minor items the reviewers flagged (non-blocking, logged in commit messages): index TS/JS tags coverage breadth, a couple of advisory fleet occupancy estimates, the `generate_and_publish` router-execution (B5) gap. None affect the agent loop's correctness.
