# Chapter 13 — Roadmap and Build Sequencing

> **Purpose (one line).** Translate the entire HIDE bible into a concrete, shell-first build sequence: three parallel tracks, four milestones, one load-bearing thesis gate, and an explicit kill protocol — so that every engineer knows what to build next, in what order, and how to prove it done.

**Status:** OPERATIONAL BLUEPRINT — the final chapter. This is the plan that makes all twelve preceding chapters actionable. Every item is scoped, sequenced, and verifiable. References to chapter sections are anchors into the rest of the bible; references to source files are anchors into the existing `hawking` codebase. The estimated durations assume a small focused team (1–3 engineers) with the repo already building.

---

## Table of contents

1. [Build philosophy](#1-build-philosophy)
2. [Track structure](#2-track-structure)
3. [Milestone 0 — Foundations (parallel kickoff)](#3-milestone-0--foundations-parallel-kickoff)
4. [Milestone 1 — Thesis gate + walking skeleton](#4-milestone-1--thesis-gate--walking-skeleton)
5. [Milestone 2 — Model lanes (signature feature)](#5-milestone-2--model-lanes-signature-feature)
6. [Milestone 3+ — Labs (post-shell, sketch only)](#6-milestone-3--labs-post-shell-sketch-only)
7. [Critical path and dependencies](#7-critical-path-and-dependencies)
8. [Kill protocol](#8-kill-protocol)
9. [Scope contract](#9-scope-contract)
10. [Verification checklists (per milestone)](#10-verification-checklists-per-milestone)
11. [HF distribution lane (when it is ready)](#11-hf-distribution-lane-when-it-is-ready)
12. [Licensing and third-party notices](#12-licensing-and-third-party-notices)
13. [Eval task pool — structure and curation criteria](#13-eval-task-pool--structure-and-curation-criteria)

---

## 1. Build philosophy

Three principles govern every sequencing decision in this chapter. They are not preferences — they are constraints. Any proposed change to the build order must be reconciled against all three.

### 1.1 Thesis-gate-first

The eval harness (`hawking-eval`) runs before any UI chrome beyond the minimal chat shell. The GO/CONDITIONAL/KILL verdict is the most valuable output of Milestone 1 — more valuable than any feature — because it determines whether every subsequent engineering dollar is well spent. If the local model cannot drive a multi-step agent loop to completion at ≥70% rate, within 20 points of a cloud control, at ≥15 tok/s, in ≤5 minutes wallclock, then the shell-first build plan is wrong and the kill protocol (§8) fires. Finding this out after three months of Tauri polish is expensive. Finding it at week six — before Model Lab, before the parallel agent workstation, before the live HF catalog — is cheap.

Concretely: A1.4 (run the eval, emit the verdict) is a **hard prerequisite** for M2. Nothing in M2 is built until A1.4 is green and the verdict is GO or CONDITIONAL-with-triage. This is not a soft gate that gets waived when schedule pressure arrives; it is encoded in the milestone definition.

### 1.2 Shell-first, labs second

Every feature that surfaces in the Tauri UI must work headlessly first. The agent loop runs correctly from a CLI harness before the Timeline panel renders it. The diff-apply tool is unit-tested before Monaco renders its output. The constrained-JSON tool-call path is integration-tested via `curl` before any React component consumes it. This discipline has two effects: (a) the headless path is always the truth — the UI is a projection, never the source of correctness; (b) new UI surfaces ship fast because the underlying capability already exists and is already tested.

The labs features defined in §6 — Research Tab, parallel agent workstation, Remote Mac-Studio, RLEF — are all post-shell. They are designed in the bible but not scheduled in the initial build. The shell ships first, the metrics are observed, and the labs are prioritized by measured user demand.

### 1.3 Parity-gated hardware

No `.tq` GPU serving path ships without a token-for-token parity test against the CPU oracle. This is the runtime invariant established by the existing `hawking-serve` codebase (see `tq_bake` parity tests in `crates/hawking-core`) and it extends to every new GPU dispatch path added during HIDE development. The parity gate is not a one-time checkpoint: it is a CI step that runs on every commit that touches `qwen_dense.rs`, `tq_gpu.rs`, or the Metal shaders. A new runtime path that passes perplexity but fails token-for-token parity against CPU is not shippable — it introduces non-determinism into the agent loop and breaks the replay guarantee (ch.01 §4.5).

The parity gate has a specific structure: three fixture sequences (short/medium/long context: 32/128/512 tokens), each producing 32 decode tokens, compared token-for-token between the CPU oracle and the GPU path. The reason for the 32-token decode window (not a single token) is that divergence is often not at token 1 but accumulates through the KV cache after a few steps. A path that matches token 1 but diverges at token 15 is not deterministic; the 32-token window catches these cascade failures. The CPU oracle is defined as: float32 forward pass with no quantization, `temp=0, top_p=1.0` (greedy). The GPU path under test must produce the same greedy argmax at every position.

---

## 2. Track structure

Three tracks proceed in parallel, with explicit sync points at each milestone boundary. A track can sprint ahead internally, but the milestone gate requires all three to be green before the next milestone begins. The tracks are named for their role:

**Track A — Agent + Eval (the brain).** Everything that makes the agent reason, plan, use tools, and be measured. `hawking-agent`, `hawking-eval`. No UI dependencies — this track is headless-first by construction.

**Track B — Shell (the face).** The Tauri 2 desktop app: React/TS/Monaco/xterm, the Chat tab, the Diff Review panel, the Context Stack, the terminal, the file tree. Track B consumes Track A's headless interfaces and Track C's HTTP surface. It cannot sprint ahead of Track A or Track C for integration milestones but can scaffold and stub freely.

**Track C — Runtime (the engine).** The existing `hawking-serve` extended with tool-calling, TQ GPU serving, the first-class `.tq`+`meta.gguf` loader, and the `hawking-router` multi-model supervisor. Track C has the most direct dependency on hardware (Metal GPU, RAM) and should be started in parallel with Track A from day one.

**Sync points** are defined at the end of each milestone section. A sync point is not a code freeze — it is a list of concrete, binary-verifiable conditions. All conditions must be true before the team proceeds to the next milestone.

**Track dependencies across milestones.** The tracks are parallel within a milestone but develop inter-track dependencies at the sync points. Track B cannot fully integrate with Track A until A0.1 produces a working `AgentLoop` event stream; until then, Track B works against a stub event emitter. Track A cannot integrate with Track C's embedding endpoint until C1.2 is live; until then, Track A uses lexical-only retrieval. These stub-then-integrate patterns are intentional: they decouple tracks within a sprint so that progress is not blocked by another track's schedule variance. The integration is resolved at the sync point, not mid-sprint. The result is that each track's internal progress is visible at any time — a Track B engineer can always show the chat tab working, even if Track A is not yet complete — which makes the milestone status observable without a formal status meeting.

**What "green" means at a sync point.** Green does not mean polished. It means: the condition is machine-verifiable (a test exits 0, a file exists, a manual check is described precisely enough that any engineer can reproduce it), and it passed on the current HEAD of the main branch, not on a feature branch. Sync points are gates on `main`. Work that passes locally but has not been merged to `main` does not count as green. This prevents the pattern where a milestone is "done" according to a branch but integration reveals failures at merge time.

---

## 3. Milestone 0 — Foundations (parallel kickoff)

**Goal:** Standing up the three tracks so that they can sprint independently. By the end of M0, tool calls flow end-to-end from the agent to the HTTP server; the Tauri shell is visible with a live chat tab; and the `.tq` loader is scaffolded. M0 is not a feature milestone — it is the parallel kickoff that eliminates the longest lead times.

**What M0 is not.** M0 does not produce an agent that completes real tasks. It does not produce a GPU TQ serving path that passes the parity gate. It does not produce a router that handles concurrent model requests under memory pressure. These are all M1 targets. M0 produces the interfaces and stubs that M1 builds on. The test for whether M0 is complete is the sync point, not "is the thing working end-to-end" — that comes at M1.

**Estimated duration:** ~2 weeks of focused parallel work.

---

### Track C items

Track C in M0 has four items but they are not all equally urgent. C0.1 (tool-calling) is the critical dependency for Track A; it should be first and fast. C0.2 (LM-head TQ) and C0.3 (`.tq` loader) are independent of each other and of C0.1; they can start immediately and proceed in parallel on separate branches. C0.4 (router scaffold) can also start immediately and proceed independently. The recommended order: start C0.1 first (1.5 days, blocks Track A), then start C0.2, C0.3, and C0.4 in parallel as soon as C0.1 is merged. Track C has the most parallelizable work in M0; if there are two C-track engineers, one takes C0.1 → C0.2 and the other takes C0.3 → C0.4.

**C0.1 — Tool-calling endpoint in `http.rs`**

*Description.* Add `tools`, `tool_choice` to `ChatReq` (currently at line 245 of `crates/hawking-serve/src/http.rs`) and `tool_calls` to the response formatter and SSE streamer. The constrained-decode path already exists via `json_mode` (~line 319); tool-calling reuses that path with a grammar compiled from the tool schema union. The grammar ensures the model emits a valid `tool_calls` array or a `content` turn, never a hybrid. See ch.06 §4.3 (constrained grammar service) for the schema-to-grammar compilation design.

*Input.* `http.rs` builds and passes existing tests. `json_mode` path working.

*Output.* `/v1/chat/completions` accepts a `tools` array and a `tool_choice` field. When the model chooses a tool, the response (or SSE delta) carries a well-formed `tool_calls` array with `id`, `type: "function"`, `function.name`, `function.arguments` (valid JSON). The non-tool path is byte-identical to today.

*Verification.* Unit test: POST `{"model":"…","messages":[…],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"tool_choice":"auto"}` → response contains `tool_calls[0].function.name == "read_file"` AND `tool_calls[0].function.arguments` parses as valid JSON against the schema. Parallel test: same request with `tool_choice: "none"` → response contains `content` field, no `tool_calls`.

*Scope.* ~1.5 days. LOW risk — the constrained-decode infrastructure already exists; this is plumbing.

---

**C0.2 — LM-head TQ serve**

*Description.* The `ensure_tq_cache` function in `crates/hawking-core/src/model/qwen_dense.rs` (line 4306) currently bakes 7 FFN projections into the `.tq` cache (the `projs` array at line 4452). Extend the `projs` list to include the LM-head linear as an 8th entry, and extend `DenseDecodeArena` dispatch to route LM-head through the TQ GPU path. The LM-head is the largest matrix in the forward pass for small vocabularies and the most memory-bandwidth-intensive at decode time; TQ serving it eliminates one Q4_K GEMV from the critical path.

*Input.* C0.1 merged. Existing `tq` feature gate compiling.

*Output.* When `HAWKING_TQ=1` and a `.tq` file is present, the LM-head decode goes through the TQ GPU path. PPL regression ≤ +0.05 vs Q4_K_M LM-head baseline (measured on wikitext-103 20K tokens).

*Verification.* Existing `tq_bake` parity test suite extended with an LM-head fixture: 3 sequences × 32 tokens, CPU oracle vs GPU TQ, token-for-token identical. PPL measured and logged.

*Scope.* ~2 days. MEDIUM risk — alignment of the LM-head weight layout against the TQ bake format needs a careful audit; the FFN projs are column-major for the GEMV but LM-head may be transposed. Audit first, bake second.

---

**C0.3 — First-class `.tq` + `meta.gguf` loader**

*Description.* Today `load_engine` in `crates/hawking-core/src/model/mod.rs` loads all weights from a single GGUF file. The 32B `.tq` serving strategy requires a split: linears (FFN projections, attention projections, LM-head) come from `.tq`; norms and embeddings (which are tiny and precision-sensitive) come from a companion `meta.gguf`. Extend `load_engine` to detect when a `.tq` file is provided as the model path, load the companion `meta.gguf` from the same directory (or a `--meta-gguf` flag), and populate `QwenDense` fields accordingly. This is the enabler for the 32B-on-18GB RAM win: the `.tq` linears are 3–4 bit, the meta is fp16 norms only, total RSS fits under the Metal shared memory budget.

*Input.* `tq_bake` pipeline producing `.tq` + `meta.gguf` for a 7B model (the existing condense toolchain produces this).

*Output.* `hawking-serve --model /path/to/model.tq` loads, warms up, and serves correctly. The existing `generate` endpoint works. RSS is ≤70% of the equivalent Q4_K_M single-GGUF load for the same model.

*Verification.* Load a 7B `.tq` + `meta.gguf`. Run the standard smoke suite (3 prompts, 64 tokens each). Token-for-token parity against Q4_K_M CPU oracle. RSS reported by `/metrics` endpoint is within gate.

*Scope.* ~3 days. MEDIUM risk — the loader touches the initialization critical path; a mis-mapped tensor offset produces silent garbage output, not a crash. Careful field-by-field logging on first load.

---

**C0.4 — `hawking-router` scaffold**

*Description.* A new crate `crates/hawking-router` that acts as a subprocess-per-model supervisor. It starts and monitors child `hawking-serve` processes (one per model role: hero, draft, embed), routes incoming `/v1/chat/completions` and `/v1/embeddings` requests by the `model` field, exposes a unified `/v1/models` list aggregated from all live children, and enforces a simple LRU memory governor: when total RSS of all children exceeds the configured budget, evict the least-recently-used model by sending it SIGTERM and waiting for it to drain. The router is an axum server itself; it speaks the same OpenAI-compatible HTTP surface upstream as the agent and downstream to each serve child.

*Input.* A working `hawking-serve` binary. No agent dependency.

*Output.* `hawking-router --config router.toml` starts, health-polls children, routes requests, and returns 503 with a retry-after header when a model is being evicted and reloaded.

*Verification.* Start router with two model configs. POST to `/v1/chat/completions` with `model: "hero"` → routed to hero child, response arrives. Kill the hero child externally → router restarts it within 5s. Trigger LRU eviction by loading a third model → least-recently-used child receives SIGTERM, drains, exits; memory governor logs the event.

*Scope.* ~3 days. LOW risk — the router is a thin process supervisor with no novel kernel code; the hard parts (GPU serving, constrained decode) live in the children.

---

### Track A items

Track A is the only track with hard sequential internal dependencies: A0.1 must exist before A0.3, and A0.3 must exist before A1.1. Within M0, the A-track work is therefore a short chain, not a parallel fan-out. The right pattern is: A0.1 first (2 days), then A0.2 and A0.3 in parallel (since they are independent once the scaffold exists). A0.2 (eval skeleton) is best started immediately after A0.1 compiles, even if A0.1 is not fully feature-complete — the eval harness exercises the agent interface, and writing the harness early surfaces ambiguities in the `AgentLoop` API that are cheaper to resolve in the scaffold than after A1.1 is implemented.

**A0.1 — `hawking-agent` crate scaffold**

*Description.* Create `crates/hawking-agent/` with three primary types: `AgentLoop` (the state machine defined in ch.02 §4.2 — Idle → Planning → Executing → Verifying → Done/Failed/RepairNeeded), `ModelClient` (a trait over an async HTTP client pointing at an OpenAI-compatible base URL, making the agent swappable between local serve and cloud endpoints for eval), and `ToolRegistry` (a map from tool name to async handler, populated at startup). The crate has no dependency on Tauri — it runs headlessly from a CLI binary. The CLI binary is the primary test surface for this track.

*Input.* C0.1 merged (tool-calling endpoint available). Rust workspace builds.

*Output.* A CLI binary `hawking-agent-cli` that: reads a task from stdin (or a JSON file), connects to a local or remote `base_url`, calls the model with a `tools` array, receives tool calls, dispatches them through `ToolRegistry`, and prints the result. The state machine logs each transition to stderr as structured JSON.

*Verification.* Integration test: point `hawking-agent-cli` at a local `hawking-serve` instance with C0.1. Provide a trivial task ("read the file /tmp/hello.txt and return its contents"). Agent calls `read_file`, receives the content, calls `done` with the answer. State machine transitions logged correctly.

*Scope.* ~2 days. LOW risk — the state machine is well-specified in ch.02; this is transcription and plumbing.

---

**A0.2 — `hawking-eval` skeleton + task curation**

*Description.* Create `crates/hawking-eval/` with the eval harness defined in the thesis gate specification: a JSONL task file (each line: `{id, repo_url, setup_cmds, task_description, oracle_type, oracle_args}`), a runner that clones the repo into a temp worktree, runs setup, invokes `AgentLoop`, and evaluates the oracle. The oracle types for Tier-A tasks are machine-checkable: `pytest` (run tests, expect all green), `exact_file` (file at path must equal expected content), `regex_match` (output matches regex), `exit_code` (command exits 0). Curate 20–30 Tier-A tasks: small repos (under 5k LoC), well-scoped single-file changes, deterministic oracles. The task pool is the primary artifact of this work item.

*Input.* A0.1 scaffold done (so the harness can invoke `AgentLoop`).

*Output.* `hawking-eval run --tasks tasks.jsonl --model local --base-url http://localhost:8080` runs all tasks and emits a `results.jsonl` with `{task_id, completion, verdict, wall_secs, tok_total}`. The CLI binary runs without error even when individual tasks fail (failure is a verdict, not a crash).

*Verification.* Run 3 Tier-A tasks manually end-to-end (with a stub agent that always calls `done` with a wrong answer). Verify the oracle correctly returns `FAIL` for wrong answers and `PASS` for correct ones. JSONL schema validates.

*Scope.* ~2 days for the harness; task curation is ongoing but 20 tasks is achievable in parallel with other work.

---

**A0.3 — Core tool implementations**

*Description.* Implement the 6 core tools defined in ch.03 §4: `read_file` (read a file path, return content as string, truncated at a configurable token budget), `edit_file` (accept a unified diff or a patch-apply block; validate the patch applies cleanly to current file state before writing; return success/conflict), `run_command` (spawn a subprocess in the repo root, capture stdout/stderr, enforce a timeout, return exit code + combined output), `search_codebase` (lexical grep over the working tree via `ripgrep` subprocess, return file/line/snippet matches up to a limit), `run_tests` (a specialization of `run_command` that detects the test runner from the repo root and formats results), `done` (terminal tool that records the agent's answer and transitions the state machine to `Done`). Each tool is registered in `ToolRegistry` with its JSON Schema for constrained-decode schema compilation.

*Input.* A0.1 scaffold (ToolRegistry exists). `hawking-agent-cli` CLI binary.

*Output.* All 6 tools pass their unit test suite. `edit_file` correctly rejects a patch that does not apply to current file state. `run_command` times out and returns non-zero exit when the subprocess exceeds the budget.

*Verification.* Unit tests for each tool. Integration test: end-to-end agent run using `hawking-agent-cli` that exercises at minimum `read_file → edit_file → run_tests → done` in sequence on a real small repo.

*Scope.* ~3 days. LOW risk — the tools are straightforward subprocess wrappers; the only tricky part is `edit_file` patch validation, which uses the `patch` crate or `git apply --check`.

---

### Track B items

Track B items in M0 are pure scaffolding: the goal is to eliminate the build-system risks (Tauri 2 + Vite + Rust workspace co-existing cleanly in CI) and establish the IPC patterns that B1.x items will build on. No feature work happens on Track B in M0 beyond a working chat tab. Track B engineers should expect to spend a non-trivial fraction of M0 on build tooling — the Tauri 2 + Vite + pnpm workspace configuration is non-trivial on its first setup, especially getting the Rust sidecar bundle paths right for CI.

**B0.1 — Tauri 2 scaffold**

*Description.* Initialize `app/src-tauri/` (Rust host with Tauri 2 dependency) and `app/src/` (React + TypeScript + Vite). The scaffold has no feature logic — it boots, shows a blank window, and exposes a single `ping` Tauri command that returns `"pong"`. Configure the Tauri capability manifest (ch.01 §7 pattern) with minimum permissions: filesystem read (for file tree), shell (for PTY), and window management. Set up CI to build the Tauri app on `macos-14` (matching the existing CI runner).

*Input.* Node.js and Rust toolchains. Tauri CLI v2.

*Output.* `pnpm tauri build` succeeds. `.app` bundle runs. `ping` command returns `"pong"` from the Rust host.

*Verification.* CI step: `pnpm tauri build` on `macos-14` exits 0. App launches, window is visible, DevTools console shows `ping → pong`.

*Scope.* ~1 day. LOW risk — scaffolding is well-documented in Tauri 2 docs.

---

**B0.2 — Chat tab: SSE streaming, Monaco, file tree**

*Description.* The primary view of the shell is the Chat tab. Wire it to the local `hawking-serve` (or `hawking-router`) HTTP endpoint: POST `/v1/chat/completions` with `stream: true`, consume the SSE stream, render tokens as they arrive into a Monaco `editor.createDiffEditor`-style read-only text area (or a plain `monaco.editor.create` in read-only mode for assistant turns). Add a file tree panel using the Tauri FS APIs (`tauri-plugin-fs`) to browse the current workspace directory. The file tree is read-only at this stage — clicking a file opens it in Monaco in a read-only view. No agent integration yet; this is purely the chat UI talking to the serve HTTP surface.

*Input.* B0.1 scaffold. C0.1 or a stub server that responds with a streaming completion.

*Output.* Tauri app shows a Chat tab. User types a message, it streams back tokens from the local server. File tree shows workspace files. Clicking a file opens it in Monaco.

*Verification.* Manual: start `hawking-serve` locally, launch the Tauri app, type a prompt, observe streaming tokens. File tree renders the repo root. No JS errors in DevTools.

*Scope.* ~3 days. LOW risk — SSE streaming from a local server is straightforward; Monaco is well-documented.

---

**B0.3 — xterm terminal wired to PTY**

*Description.* Add a terminal tab backed by a PTY spawned in the Tauri Rust host using the `portable-pty` crate (or `tauri-plugin-shell` PTY mode). The xterm.js frontend connects to the PTY via a Tauri `ipc::Channel<T>` (ch.01 §4.4 pattern — ordered, streamed). User keystrokes from xterm are sent to the PTY's stdin; PTY output streams back to xterm. The terminal opens in the workspace root.

*Input.* B0.1 scaffold.

*Output.* Terminal tab in the Tauri app. User can type shell commands and see output. `ls`, `cargo build`, `git status` all work.

*Verification.* Manual: open terminal tab, run `echo hello`, see `hello`. Run `cargo build` in the repo root, observe real-time output.

*Scope.* ~2 days. LOW risk — PTY-over-channel is a well-worn pattern in Tauri apps; several reference implementations exist.

---

### Milestone 0 sync point

All three conditions must be true before M1 work begins:

1. **C0.1 green:** `curl` test for tool-calling endpoint passes (tool call returns well-formed `tool_calls` JSON).
2. **A0.1 green:** `hawking-agent-cli` completes a trivial tool-call round-trip against a local serve.
3. **B0.2 green:** Tauri app is visible on macOS with a streaming chat tab.

**M0 integration note.** The M0 sync point deliberately does not require C0.2 (LM-head TQ), C0.3 (`.tq` loader), or C0.4 (router) to be complete — only C0.1 (tool-calling) is required. This is intentional: the agent track (A) and shell track (B) can validate their foundations against a plain `hawking-serve` instance without TQ or router. C0.2–C0.4 are M0 items but they feed M1's C1.1/C1.2, not M0's sync point. If C0.2–C0.4 slip into the first week of M1, that is acceptable provided C0.1 is done. The sync point is a minimum-viable gate, not an exhaustive completion certificate.

---

## 4. Milestone 1 — Thesis gate + walking skeleton

**Goal:** The GO/CONDITIONAL/KILL verdict is the primary deliverable. The walking skeleton (agent runs a real task end-to-end, UI shows it live) is secondary but must be present by sync point. This milestone is the decision point for the entire project.

**What makes M1 the hardest milestone.** M1 is harder than M2 not because M2 has less work, but because M1 is where the three tracks converge for the first time into an integrated system. The agent calls tools, the tools run against a real repo, the model serves at production quality via the TQ path, the router manages model memory, and the UI shows it all live. Each of these subsystems was tested in isolation in M0. M1 is the first integration. Integration always surfaces assumptions that were implicit in the isolation tests. Reserve the last 2 days of M1 for integration debugging, not new feature work.

**Estimated duration:** ~4 weeks from M0 completion.

---

### Track C items

Track C in M1 is primarily about closing the last gaps in the GPU TQ serving path and hardening the router. C1.1 depends on C0.2 and C0.3 from M0; if those items are not fully complete at the M0 sync point, they finish at the start of M1 before C1.1 begins. The Track C M1 items are not on the A-track critical path (the agent can run its eval against a non-TQ serve); however, the thesis gate (A1.4) requires the local hero model to be served via TQ at ≥15 tok/s — so C1.1 must be complete and green before A1.4 fires. If C1.1 is delayed, A1.4 is delayed proportionally.

**C1.1 — Complete GPU `.tq` serving + parity gate**

*Description.* Integrate C0.2 (LM-head TQ) and C0.3 (first-class loader) into a full GPU TQ serving pipeline for a real 7B `.tq` model. Add the parity CI gate: for each commit touching `qwen_dense.rs`, `tq_gpu.rs`, or the Metal shader bundle, run 3 fixture sequences × 32 tokens each, compare CPU oracle output token-for-token, fail the build if any token diverges. Add a PPL measurement step: run wikitext-103 20K token eval, assert PPL ≤ Q4_K_M + 0.3 (absolute). Add an RSS measurement step: load the `.tq` model, measure peak RSS from `/metrics`, assert ≤ 70% of the Q4_K_M baseline RSS for the same parameter count.

*Input.* C0.2 and C0.3 complete. A `.tq` + `meta.gguf` for a 7B model (produced by the existing condense toolchain).

*Output.* CI step `tq-parity` green on every commit. PPL gate passes. RSS gate passes. `hawking-serve --model 7b.tq` runs at ≥15 tok/s decode on the development machine.

*Verification.* CI log showing parity pass for all 3 fixtures. PPL reported in CI summary. RSS reported in CI summary. Decode benchmark (single-stream, 128-token context, 64 decode tokens) shows ≥15 tok/s.

*Scope.* ~1 week. MEDIUM risk — GPU `.tq` serving is mostly built (see MEMORY.md: `HAWKING_RWKV7_TQ` proven, Qwen CPU FFN-only TQ exists; the gap is all-linear GPU dispatch). The alignment audit from C0.2 is the highest-risk sub-step.

---

**C1.2 — `hawking-router` v1: 3-model subprocess routing**

*Description.* Extend C0.4 scaffold to handle the 3-model production configuration: hero (32B or best available `.tq`), draft (7B `.tq`), embed (a small embedding model). The router starts all three children, polls their `/health` endpoints, routes requests by the `model` field, enforces the LRU eviction budget, and supports concurrent requests across models (different requests going to different children simultaneously). Add a `/v1/models` aggregation endpoint. Add a router config schema validated at startup.

*Input.* C0.4 scaffold. C1.1 GPU serving green (so the hero child can actually serve at speed).

*Output.* `hawking-router --config router.toml` starts 3 children, routes correctly, handles concurrent requests, evicts under memory pressure, reports aggregate model list.

*Verification.* Integration test: 3 concurrent requests (one to each model role) complete without deadlock. Kill the hero child → router restarts it within 10s. Memory pressure test: configure budget below the total RSS of 3 children → eviction fires, one child is stopped, memory returns below budget.

*Scope.* ~3 days. LOW risk — extends C0.4 scaffold; the hardest part (serving) is in the children.

---

### Track A items

Track A in M1 has the highest complexity per-item of any milestone in the build sequence. A1.1 (the full loop), A1.2 (permission + sandbox), and A1.3 (codebase-intel) are all medium-to-large items with meaningful design surface. The recommended sequencing within M1 is: A1.1 first and in full (the loop must be solid before the permission model is layered on), then A1.2 (short, well-specified), then A1.3 (can start before A1.2 is done since they are independent), then A1.4 once the preceding three are green on `main`. The total A-track M1 effort is approximately 3 weeks; if the team is a single engineer on this track, A1.3 and A1.2 should be started as partial work-in-progress during A1.1's second week to avoid a pure serial bottleneck.

**A1.1 — Complete Planner→Executor→Verifier loop**

*Description.* Flesh out the `AgentLoop` state machine from A0.1 into the full Planner→Executor→Verifier design (ch.02 §4.2–§4.7): HTN plan generation (the model generates a `Plan` object as a constrained-JSON tool call with a `steps` array and `depends_on` edges), step execution via `ToolRegistry`, oracle verification (deterministic: does the step's output satisfy the declared postcondition?), and repair/replan on failure (up to `max_repair_depth = 3` before the loop emits `Failed`). The `Plan` object is stored as an event in the session log (ch.01 §4.5) so it can be replayed.

*Input.* A0.1 scaffold. A0.3 core tools. C0.1 tool-calling endpoint. An HTN plan schema (defined inline in this work item, validated by the constrained-decode grammar).

*Output.* `hawking-agent-cli` can complete a multi-step task (e.g., "add a function to src/lib.rs, run the tests, fix any failures") by generating a plan, executing steps in dependency order, verifying each, and repairing on failure.

*Verification.* Integration test using a synthetic small repo: task = "add a pub function `double(x: i32) -> i32` to src/lib.rs that returns `x * 2`, and make `cargo test` pass." Agent generates a plan with ≥2 steps (`edit_file` + `run_tests`), executes, verifies, completes with `done`. The oracle fires `PASS`.

*Scope.* ~1 week. MEDIUM risk — the repair/replan path has subtle state management (the plan is partially executed; repair must not re-execute completed steps). Careful event log checkpointing here.

---

**A1.2 — Permission model + worktree sandbox**

*Description.* Implement the permission model defined in ch.10 §4: each tool call is classified as `auto` (run without asking), `ask` (prompt the user via a Tauri dialog or stdin for the CLI), or `deny` (reject immediately). The default policy is `auto` for read-only tools (`read_file`, `search_codebase`), `ask` for write tools (`edit_file`), and `deny` for shell commands outside the workspace root. Add git-worktree sandboxing: before a task starts, `git worktree add /tmp/hawking-<uuid> HEAD`; the agent operates in this worktree; on `done`, the diff is presented to the user for review; on `undo`, the worktree is deleted.

*Input.* A1.1 loop complete.

*Output.* CLI binary prompts for confirmation on `edit_file` calls (when not in `--auto` mode). Worktree is created at task start and cleaned up at task end. `undo` command deletes the worktree and reverts the session.

*Verification.* Test: run a task in `ask` mode; manually deny an `edit_file` call; agent receives a `PermissionDenied` tool result; agent reports `Failed` with reason. Test: run a task end-to-end; verify the worktree is deleted after `done`.

*Scope.* ~3 days. LOW risk — the worktree mechanics are well-understood; the permission model is a straightforward enum dispatch.

---

**A1.3 — Codebase-intel-lite**

*Description.* Implement the lightweight codebase intelligence defined in ch.05 §4: a repo map generator (tree-sitter parse of all source files → symbol table → PageRank-like importance score based on reference count), lexical-first retrieval (ripgrep over the working tree, ranked by BM25 score against the query), and embedding re-rank (call the embed model via `hawking-router`'s `/v1/embeddings` endpoint; re-rank lexical candidates by cosine similarity). The repo map is rebuilt on `git checkout` and invalidated incrementally on `edit_file` calls. The retrieval result is passed to `AgentLoop` as a `ContextManifest` (ch.04 §4.4) and included in the model's context window.

*Input.* C1.2 router v1 (for the embed model endpoint). A0.3 core tools. tree-sitter Rust bindings.

*Output.* `hawking-agent-cli` on a medium-sized repo (e.g., the hawking codebase itself, ~50k LoC) returns a ranked list of relevant files and symbols for a given query in ≤2s. The agent's context window includes the repo map summary and the top-k retrieved chunks.

*Verification.* Benchmark on the hawking codebase: query "where is the TQ bake pipeline defined?" → top-3 results include `qwen_dense.rs::ensure_tq_cache` and `tq_bake`. Latency ≤2s on the development machine.

*Scope.* ~1 week. MEDIUM risk — the PageRank scoring is the trickiest part; a simpler reference-count heuristic is the fallback.

---

**A1.4 — Run the eval: thesis gate verdict**

*Description.* Run `hawking-eval` on the full 20–30 Tier-A task pool with two configurations: (1) cloud control (GPT-4o or equivalent via the `ModelClient` trait pointing at a cloud base-url), (2) local `.tq` hero model via `hawking-router`. Emit a `verdict.json` with:

```json
{
  "run_date": "…",
  "local_model": "qwen-32b.tq",
  "cloud_model": "gpt-4o",
  "local_completion_pct": 74,
  "cloud_completion_pct": 91,
  "gap_pts": 17,
  "local_median_tps": 18.4,
  "local_median_wall_secs": 187,
  "verdict": "GO",
  "notes": "…"
}
```

The verdict is GO if `local_completion_pct ≥ 70 AND gap_pts ≤ 20 AND local_median_tps ≥ 15 AND local_median_wall_secs ≤ 300`. CONDITIONAL if one metric is within 10% of the gate (e.g., 65% completion or 22pt gap). KILL otherwise.

*Input.* A1.1 full loop. A1.2 permission model. A1.3 codebase-intel. C1.1 GPU serving green. C1.2 router v1.

*Output.* `verdict.json` committed to the repo. The verdict governs all subsequent milestone decisions.

*Verification.* The `verdict.json` file is present and parseable. The task runner log is archived alongside it (one JSONL row per task with all intermediate tool calls, plan steps, and oracle results).

*Scope.* ~2 days of eval running time (the eval run itself; the implementation preconditions are A1.1–A1.3). LOW risk procedurally — the harness is built; this is pressing the button and reading the output.

---

### Track B items

Track B items in M1 shift from scaffolding to real feature work. The three panels (B1.1 Timeline, B1.2 Diff Review, B1.3 Context Stack) are all projections of the Track A event stream. They should be developed against a `stub_agent` binary that emits pre-recorded session events at 200ms intervals — this allows Track B to develop and test the UI without waiting for Track A's full loop to be stable. The stub is removed once A1.1 is integrated; the real event stream is a drop-in replacement because the schema is fixed.

**B1.1 — Agent-Run timeline panel**

*Description.* A panel in the Tauri app that renders the agent's execution in real time: plan steps as a vertical timeline (Idle → Planning → Step 1 → Step 2 → … → Done), tool calls as expandable rows showing the tool name, arguments, and result, and a repair indicator when the loop enters a repair cycle. The panel subscribes to the session event log via a Tauri `ipc::Channel<T>` stream from the Rust host. The event schema is the one defined in ch.01 §4.6. The design is informed by the OpenHands event model (MIT licensed) but the implementation is original.

*Input.* A1.1 loop emitting structured events to the channel. B0.1 Tauri scaffold.

*Output.* While an agent run is in progress, the Timeline panel updates in real time. Plan steps turn green on success, red on failure, yellow on repair. Tool call rows are expandable.

*Verification.* Manual: run a 3-step agent task; watch the Timeline panel update live for each step.

*Scope.* ~3 days. LOW risk — the UI is a projection of the event stream; the hard part (the event stream) is in Track A.

---

**B1.2 — Diff Review panel**

*Description.* When the agent calls `edit_file`, the proposed change is staged in the worktree (A1.2). The Diff Review panel shows the unified diff in a Monaco `createDiffEditor` view with accept/reject buttons per hunk. Accepting a hunk applies it to the real working tree; rejecting it reverts the hunk in the worktree and sends a `PermissionDenied` result back to the agent. The panel is informed by the Cline and Void diff views (Apache-2.0 licensed) but uses Monaco's native diff editor, which is already in the dependency tree. The accept/reject hunk logic is in the Rust host (using the `patch` crate), not in the frontend.

*Input.* A1.2 worktree sandboxing. B0.2 chat tab (Monaco already present).

*Output.* After an agent `edit_file` call, the Diff Review panel appears. User accepts or rejects individual hunks. The file on disk reflects accepted hunks only.

*Verification.* Manual: trigger an `edit_file` in a task; verify the diff panel appears with the correct hunk view; accept one hunk and reject another; verify the file state on disk.

*Scope.* ~3 days. LOW risk — Monaco's `createDiffEditor` is the hard part and it is well-documented.

---

**B1.3 — Context Stack right-rail**

*Description.* A collapsible right-rail panel that shows, in real time during an agent run: the retrieved files and symbols (from A1.3), the active tools in the registry, the current memory state (key-value pairs from the session KV store), the most recent test results, and the current budget consumption (tokens used, steps taken, wall time). The panel is a pure read-only projection of the session event stream — it never initiates actions. It updates on every event from the `ipc::Channel<T>`.

*Input.* A1.3 codebase-intel (so there is retrieval context to show). B1.1 timeline panel (channel subscriber pattern already established).

*Output.* Right-rail visible in the app, updating live during an agent run. Collapsible to save screen space.

*Verification.* Manual: run a task; verify that retrieved files appear in the right-rail as the agent calls `search_codebase`; verify budget counter increments.

*Scope.* ~4 days. LOW risk — another event-stream projection; the challenging UX is collapsibility and scroll-lock behavior.

---

### Milestone 1 sync point

All conditions must be true before M2 begins:

1. **A1.4 verdict emitted:** `verdict.json` is present in the repo with `verdict: "GO"` or `verdict: "CONDITIONAL"`. If KILL, §8 fires.
2. **C1.1 parity green:** GPU TQ parity CI gate passes for all 3 fixtures. PPL gate passes. RSS gate passes.
3. **B1.3 visible:** Context Stack right-rail renders live during an agent run in the Tauri app.

**M1 integration note.** The M1 sync point is the hardest gate in the build sequence because it requires convergence from all three tracks simultaneously. The practical pattern is a two-day integration sprint at the end of M1: stop all new feature work, merge all branches to `main`, run the full checklist (§10, Milestone 1 checklist), fix any failures, re-run. Reserve these two days explicitly in the schedule — do not assume M1 items will integrate cleanly on first merge. The most common integration failures are: (a) the agent's event schema has drifted from what the Timeline panel expects (fix: tighten the schema contract before diverging); (b) the GPU TQ parity gate was passing locally but fails on CI's M-series runner due to a different Metal driver version (fix: pin the CI runner to a specific macOS version that matches development); (c) the eval harness is flaky on CI because it spawns a real `hawking-serve` process that sometimes fails to start within the timeout (fix: add a health-check retry loop with a 30s total timeout before the first eval request).

---

## 5. Milestone 2 — Model lanes (signature feature)

**This milestone is gated on M1 = GO or CONDITIONAL-with-triage.**

If M1 = KILL, stop here and read §8. If M1 = CONDITIONAL, convene a triage session: identify the dominant failure mode from the task runner log, apply a targeted fix (see §8 options), re-run the eval on the failing subset, and proceed only if the re-run clears the CONDITIONAL threshold.

**Goal:** Close the end-to-end loop from "paste a HuggingFace URL" to "the agent uses the model." This is the signature feature of HIDE — the thing that no cloud IDE can offer. A user pastes a HF link, HIDE downloads, condenses, and serves the model, and the agent immediately benefits from it.

**The M2 narrative for users.** Without M2, HIDE requires a pre-baked `.tq` model provided by the user or the build team. With M2, HIDE becomes self-provisioning: any public HuggingFace model ID becomes a source of a condensed, serving-ready model with a single paste. The 32B case is the flagship story: a 32B model that would require 20GB in Q4_K_M GGUF format condenses to ~14GB via `.tq`, and HIDE ships it to the user's `~/.hawking/models/` directory with a single click. No other local IDE tool offers this workflow. The M2 build sequence must be tight enough to deliver this story before it is announced — do not announce the live catalog until C2.2 passes the download gate and the 32B `.tq` is uploaded to the HF org.

**Estimated duration:** ~3 weeks from M1 GO.

---

### Track C items

Track C in M2 has the highest implementation risk of any track in any milestone. C2.1 (the condense pipeline) touches network I/O, subprocess management, and the bake pipeline — three categories where failure modes are hard to enumerate ahead of time. The right approach is to build it in layers: start with the bake-only path (user provides a local GGUF, condense produces a `.tq`), prove that round-trip works end-to-end, then add the download layer on top. This lets the Track B Condense Wizard (B2.2) be tested against a real condense pipeline earlier, before the network layer is stable. C2.2 (hawking-hub) is deliberately gated behind the catalog URL being live; do not block C2.1 or C2.2's download logic on that URL — use a local fixture and a feature flag to keep the code shippable regardless of when the external URL is ready.

**C2.1 — `hawking-condense` crate**

*Description.* A new crate `crates/hawking-condense/` that implements the pipeline defined in ch.06 §4.7: given a HuggingFace model ID or direct URL, (1) download the model files to a staging directory using `hf_hub` or direct HTTPS with `reqwest`, with resume-on-interrupt (content-addressed chunk storage so a half-downloaded model can resume); (2) if necessary, convert from safetensors to GGUF using the existing `llama.cpp` convert scripts (invoked as a subprocess); (3) invoke `tq_bake` to produce the `.tq` + `meta.gguf` pair; (4) register the resulting model in `~/.hawking/models.json`. Progress is streamed as structured events so the UI can display a progress bar. The crate is designed around the condense toolchain already in `tools/condense/`.

*Input.* The existing `tools/condense/` pipeline (ladder.py, sweep.py). C0.3 first-class loader (so the resulting `.tq` can be loaded). M1 GO verdict.

*Output.* `hawking-condense --hf-model Qwen/Qwen2.5-7B-Instruct --bits 3` downloads, condenses, and registers the model. A progress event stream is emitted throughout. The model appears in `~/.hawking/models.json` and is immediately loadable by `hawking-serve`.

*Verification.* End-to-end test: condense a small model (0.5B or 1.5B for speed), verify `.tq` + `meta.gguf` are present, load with `hawking-serve`, run smoke suite, verify PPL within gate.

*Scope.* ~2 weeks. HIGH risk — the download+convert+bake pipeline has many failure modes (network interruption, GGUF format variations, out-of-disk during bake). Each stage must be resumable and report its failure reason clearly.

---

**C2.2 — `hawking-hub` — live catalog download**

*Description.* A new crate `crates/hawking-hub/` that fetches the Hawking HF org index JSON (at a stable URL to be published when the 32B model is ready), presents a list of pre-condensed `.tq` models with metadata (parameter count, bpw, PPL, tok/s measured on Apple M-series), and orchestrates resumable, content-addressed downloads via BLAKE3 chunk hashing. The hub is dormant until the conditions in §11 are met. At M2, it ships in a gated state: the crate exists and its download logic is tested, but the UI surface (Model Store tab) is behind a feature flag and the live catalog URL is a placeholder.

*Input.* C2.1 condense crate (for the download and verify infrastructure it shares). The Hawking HF org must have a stable index URL (owner action, not code).

*Output.* `hawking-hub list` fetches the catalog (or a local fixture JSON in tests) and prints available models. `hawking-hub download <model-id>` downloads, verifies BLAKE3, and registers in `~/.hawking/models.json`. Partial downloads resume correctly.

*Verification.* Unit test with a local HTTP fixture: download a 3-chunk model, interrupt after chunk 2, resume, verify BLAKE3 of final file matches manifest. Integration test (when the HF org URL is live): `hawking-hub list` returns ≥1 model.

*Scope.* ~1 week. MEDIUM risk — the download/verify logic is straightforward; the dependency on the HF org URL being live is an external constraint, hence the gated approach.

---

### Track B items

The Track B items in M2 are the first HIDE UI surfaces that have no analog in any existing cloud IDE. The Model Lab tab and Condense Wizard are unique to a local-first tool. Track B engineers should resist the temptation to over-build these panels: the M2 goal is a working round-trip (paste URL → condense → serve → agent uses it), not a polished store UI. Polish follows once the round-trip is verified. The Status Bar chip (B2.3) is the cheapest item in M2 and should be done first — it gives constant feedback on whether the serve stack is healthy, which benefits all other M2 development.

**B2.1 — Model Lab / Store tab**

*Description.* A new tab in the Tauri app that shows the list of models available from `~/.hawking/models.json` (locally condensed or downloaded) and, when the hub catalog URL is live, the list of models available from the Hawking HF org. Each model row shows: name, parameter count, bpw, PPL, estimated tok/s, download state (local / downloading / available). Users can click "Download" to start a download (which invokes `hawking-hub download` as a subprocess and streams progress events back to the UI) or "Use" to switch the active hero model in `hawking-router`. Behind a feature flag until C2.2 catalog URL is live.

*Input.* C2.2 hub crate. B0.1 Tauri scaffold.

*Output.* Model Lab tab shows local models. When a model is selected, `hawking-router` is sent a config update to use it as the hero.

*Verification.* Manual: add a local model to `~/.hawking/models.json`; open Model Lab; verify it appears; click "Use"; verify `hawking-router` config updates and the model serves the next request.

*Scope.* ~4 days. LOW risk — the UI is a list view with download progress; the logic is in the crates.

---

**B2.2 — Condense Wizard**

*Description.* A modal dialog (accessible from the Model Lab tab and from a "+" button in the model selector) that walks the user through condensing a model from a HuggingFace URL. Step 1: paste the HF model ID or URL. Step 2: select target bpw (3-bit, 4-bit, or auto). Step 3: confirm disk space and estimated time. Step 4: progress view showing download → convert → bake with a per-stage progress bar and ETA. Step 5: success screen with "Use this model" button. Error states are informative: disk full, network error, GGUF conversion failed.

*Input.* C2.1 condense crate. B2.1 Model Lab tab.

*Output.* User can condense a new model entirely in-app from a HF URL with no command-line interaction.

*Verification.* Manual end-to-end: paste `Qwen/Qwen2.5-1.5B-Instruct`, select 3-bit, complete the wizard, verify the model appears in Model Lab and can be selected.

*Scope.* ~3 days. LOW risk — primarily a progress-display UI over an existing CLI pipeline.

---

**B2.3 — Status bar chip**

*Description.* A persistent chip in the bottom status bar showing: active hero model name, serve health (green dot = healthy, red dot = unhealthy/loading), current decode throughput (tok/s, updated every 5s from the `/metrics` endpoint), and a click target that opens the Model Lab tab. This is the always-visible signal that the model is running and healthy.

*Input.* C1.2 router v1 `/metrics` endpoint. B0.1 Tauri scaffold.

*Output.* Status bar chip visible in all tabs. Tok/s updates every 5s. Clicking opens Model Lab.

*Verification.* Manual: stop the serve process; verify chip turns red within 10s. Restart; verify it returns green.

*Scope.* ~1 day. LOW risk.

---

### Milestone 2 sync point

All conditions must be true before M3+ work begins:

1. **Round-trip verify:** condense a small HF model → load → serve → agent uses it for a complete task → oracle PASS.
2. **`~/.hawking/models.json` registration:** condensed model appears in the file with correct metadata.
3. **Download resume:** interrupted download resumes correctly (BLAKE3 verified on completion).
4. **Catalog gate:** `hawking-hub list` returns results (from fixture in CI; live URL gated separately).

---

## 6. Milestone 3+ — Labs (post-shell, sketch only)

These capabilities are designed in the bible but not scheduled in the initial build. The gate for each is: M2 shipped, user base established, thesis metrics confirmed in production, and the specific preconditions below met. They are listed here for completeness and to anchor each lab feature to its bible chapter and its prerequisite track work.

| Lab feature | Defined in | Requires to build | Tier |
|---|---|---|---|
| **Research Tab** — multi-source research pipeline, knowledge graph construction, citation tracking | ch.08 | hawking-agent at M1, A1.3 codebase-intel, web-search tool addition, vector store for knowledge graph | POST-SHELL TIER-2 |
| **Memory Browser** — editable long-term project memory, episodic/semantic recall UI | ch.04 §4.6 | Session event log fully implemented, vector store wired, semantic search over memory | POST-SHELL TIER-2 |
| **Parallel Agent Workstation** — fan-out swarms, merge funnel, worktree per agent, multi-agent timeline UI | ch.09 | hawking-agent subagent spawn (ch.02 §4.10), hawking-router concurrent model routing, per-agent worktree isolation | POST-SHELL TIER-2 |
| **Remote Mac-Studio** — WSS + JSON-RPC 2.0 bridge, server-authoritative reconnect, remote model lane | ch.09 §4.4 | Parallel agent workstation shipped, stable network protocol, remote auth | POST-SHELL TIER-4 |
| **RLEF on-device** — PPO/GRPO fine-tune loop, reject sampling from agent runs, PPL gate before activation | ch.11 | RLEF data flywheel (accepted edits logged), hawking-condense fine-tune extension, ≥32B model on device | POST-SHELL TIER-3 moonshot |
| **DSPy-style prompt optimization** — self-improving workflows, few-shot bootstrap from eval runs | ch.11 §4.7 | hawking-eval extended with prompt variation support, A1.4 eval infrastructure | POST-SHELL TIER-3 |
| **Full JSON Schema grammar compiler** — schema → LALR grammar → constrained decode, handles $ref, oneOf, anyOf | ch.06 §4.3 | C0.1 tool-calling shipped, M1 eval results as quality signal for grammar tightness | POST-SHELL TIER-2 (fast-follow after M1) |
| **Agentic fine-tune at Condense time** — personalization flywheel, user's accepted edits as fine-tune signal | ch.06/ch.11 | hawking-condense pipeline, RLEF infrastructure, ≥100 accepted edits in logs | POST-SHELL TIER-3 |
| **Parallel draft racing** — side-by-side Monaco split view for A/B draft comparison, winner-select UI | ch.09/ch.11 | Parallel agent workstation, Monaco split editor wired to two independent agent runs | POST-SHELL TIER-2 |

The tiers are not arbitrary. TIER-2 means "clear path, build after M2 ships." TIER-3 means "requires M2 shipped + specific infrastructure preconditions." TIER-4 means "requires TIER-2 shipped + external dependency (remote hardware, network protocol)." Moonshot means the capability is valuable if it works but the success probability is genuinely uncertain.

---

## 7. Critical path and dependencies

The single longest path from today to the M1 thesis gate verdict runs through:

```
C0.1  tool-calling endpoint
  │
  └─► A0.1  agent scaffold (ModelClient + ToolRegistry)
        │
        ├─► A0.2  eval skeleton + task curation
        │
        ├─► A0.3  core tools (read_file, edit_file, run_command, ...)
        │         │
        │         └─► A1.1  Planner→Executor→Verifier loop
        │                   │
        │                   ├─► A1.2  permission model + worktree sandbox
        │                   │
        │                   └─► A1.3  codebase-intel-lite ──────────────────────┐
        │                             (requires C1.2 router for embed endpoint)  │
        │                                                                         │
        └────────────────────────────────────────────────────────────────────────►A1.4
                                                                                  THESIS
C0.3  .tq+meta.gguf loader                                                        GATE
  │                                                                               VERDICT
  └─► C1.1  GPU TQ serving + parity gate ──────────────────────────────────────►│
        │
C0.4  hawking-router scaffold
  │
  └─► C1.2  hawking-router v1 (3-model routing) ──────────────────► (feeds A1.3)
```

**Items that can run in parallel from day one:**
- C0.1 and C0.3 and C0.4 (no mutual dependencies)
- A0.1 starts as soon as C0.1 is available (or against a stub server)
- B0.1 and B0.3 start immediately, no dependencies
- B0.2 needs C0.1 or a stub; can start against a stub

**Sequential bottlenecks:**
- A1.4 cannot start until A1.1 + A1.2 + A1.3 + C1.1 + C1.2 are all complete. This is the convergence point.
- A1.3 cannot start until C1.2 router v1 is available (it calls the embed endpoint). C1.2 depends on C0.4. C0.4 can start immediately. So A1.3 is gated approximately 3 days into M1 (after C0.4 finishes and C1.2 starts).
- C1.1 parity gate depends on C0.2 (LM-head TQ) and C0.3 (first-class loader) both being complete. Both can start immediately in M0.

**Single longest path (estimated calendar days, sequential):**
C0.1 (1.5d) → A0.1 (2d) → A0.3 (3d) → A1.1 (5d) → A1.3 (5d, partly overlaps with C1.2) → A1.4 (2d) = **~18.5 days** of sequential dependency, placing the earliest possible verdict at week 4 from kickoff assuming parallel track execution of C and B.

The second-longest path: C0.3 (3d) → C1.1 (5d) → A1.4 (**8d** sequential, runs in parallel with the A-track bottleneck). This is not the critical path; it finishes before A1.4's A-track prerequisites.

**Where schedule risk lives.** The estimates above are "focused work" days, meaning one engineer on the item without context-switching. In a team of 2–3, context-switching is unavoidable, and some items will land later than estimated. The three highest-risk items by schedule variance are: A1.1 (the full Planner→Executor→Verifier loop — the repair/replan path is underspecified until you hit the first real failure in testing), A1.3 (codebase-intel — tree-sitter parse latency on large repos is unpredictable until measured), and C1.1 (GPU TQ serving — the LM-head alignment audit from C0.2 is a hard prerequisite and its duration is uncertain). If any of these items lands late, the critical path extends proportionally.

**Risk mitigations.** For A1.1: start with a flat `max_repair_depth=1` limit and a hard `RepairNotAttempted` fallback during M0; implement the full 3-depth repair in M1 only when the simpler path is proven. For A1.3: start with lexical-only retrieval (ripgrep, no embedding re-rank) in M0; add embedding re-rank in M1 only if lexical alone fails the quality bar in the eval. For C1.1: run the LM-head alignment audit in the first two days of M0 alongside C0.2, before any GPU dispatch code is written; if the layout is incompatible, fall back to Q4_K LM-head for the parity gate and defer TQ LM-head to M2.

**The B track is not on the critical path for A1.4.** Track B items (B0.1–B0.3, B1.1–B1.3) are decoupled from the thesis gate verdict. If the B track slips, A1.4 still fires. The M1 sync point requires only B1.3 (Context Stack right-rail) to be visible — not that it is polished. A stub that shows any live event from the agent session satisfies the gate.

---

## 8. Kill protocol

**Definition.** M1 = KILL fires when `verdict.json` contains `verdict: "KILL"`: local completion < 50% even with schema-constrained decode and a functioning repo-map, OR local_median_tps < 10 tok/s, OR local_median_wall_secs > 600 (10 minutes) on the Tier-A task pool.

**What the KILL verdict means.** The local model floor is too low for general-purpose multi-step coding tasks at the Tier-A difficulty level. This is not a code quality failure — the agent loop, the tool implementations, the codebase intel, and the GPU serving may all be working correctly. The failure is that the available model cannot reason reliably enough over multi-step plans in the task domain. Finding this at week 6 is the plan working.

**Options, ranked by recovery cost:**

| Option | Description | Cost | Preserves |
|---|---|---|---|
| **1. Raise model floor** | Push the Condense quality workstream: improve PPL recovery, wait for the 32B `.tq`, run a calibrated fine-tune on coding tasks using the eval task pool as training signal. Re-run A1.4 when model quality improves. | Medium — timeline delay of 4–12 weeks depending on Condense progress | Everything. The shell and eval infrastructure are fully reusable. |
| **2. Narrow task scope** | Restrict HIDE to a constrained domain: single-file edits, test fixing, docstring generation. Re-define Tier-A tasks to match this scope. Re-run A1.4 on the narrowed pool. | Low — the agent and UI are unchanged; only the task definition changes | Everything except the "general coding agent" positioning. The value proposition is narrowed, not eliminated. |
| **3. Hybrid mode** | Declare local serve as the default for low-risk tasks (read-only, small edits) and offer an optional cloud model (user provides API key) for hard tasks. The `ModelClient` trait already supports remote base-url; this is a routing policy change, not a new implementation. | Very low — 1–2 days of configuration work | All infrastructure. This is a product positioning change: HIDE becomes a local-first-but-not-local-only tool. |

**What ships regardless of verdict.** The runtime improvements — C0.1 (tool-calling endpoint), C0.2 (LM-head TQ), C0.3 (first-class loader), C0.4 (router scaffold), C1.1 (GPU TQ parity gate), C1.2 (router v1) — are shipped unconditionally. They benefit Hawking Condense benchmarking, `hawking-serve` users, and the 32B `.tq` development workstream regardless of the IDE verdict. The `hawking-eval` infrastructure is reused for model quality benchmarking regardless of the IDE verdict. The `hawking-agent` loop and core tools are reused in any of the three recovery options above.

**Distinguishing KILL from CONDITIONAL.** A CONDITIONAL verdict — where one metric is within 10% of a gate — is significantly different from KILL. CONDITIONAL means the model is close enough that a targeted intervention (a better calibration set, an improved constrained-decode grammar, a tighter repo-map summary) might clear the threshold. The appropriate response is a focused two-week sprint on the dominant failure mode, followed by a re-run of the eval on the failing task subset (not the full pool — that would take too long). KILL means the gap is structural: the model is not capable of multi-step reasoning over the task domain at any latency, and the only path forward is Option 1 (raise model floor) or Option 2 (narrow task scope). The eval task runner logs contain enough diagnostic information to tell them apart: a CONDITIONAL failure tends to cluster on a specific task type (e.g., tasks requiring multi-file coordination, or tasks requiring precise patch application), while a KILL failure is spread uniformly across all task types.

**Failure mode taxonomy from the task logs.** After any verdict, triage the task logs by failure category: (a) tool-call format error (the model emits malformed JSON despite constrained decode — grammar is too loose); (b) wrong plan (the model's HTN plan is correct in structure but wrong in content — too few context tokens, repo map too sparse); (c) execution error cascade (the first step fails, repair fires, and the model loops — repair depth too shallow or repair prompt too generic); (d) timeout (the model is simply too slow for the task budget — TPS or plan depth is wrong). Each failure mode has a different fix, and the task log uniquely identifies which mode dominated.

**The KILL verdict is not a failure.** It is the plan working: we find out cheaply, before the full shell is built, whether the local model can carry the load. We commit to this measurement because we are confident in the runtime and the agent loop; the unknown is the model quality at task-relevant capability dimensions. The eval task pool was designed to surface this unknown as early as possible.

---

## 9. Scope contract

The following table is the binding scope definition for the initial shell. Anything in the OUT OF SCOPE column is explicitly deferred. If a stakeholder asks "is X in scope?", the answer is this table, not a verbal agreement.

Two rules govern this table. First, scope creep is a synchronization failure: when a new item is added to IN SCOPE mid-milestone, it either delays the milestone sync point or reduces the depth of another in-scope item. Both outcomes are worse than deferring. The table should be changed only at sync points, with deliberate decision by the team, not by individual engineers adding "just one more thing" during implementation. Second, the OUT OF SCOPE items are not forgotten — they are deferred. Each out-of-scope item has a bible chapter that specifies it in full, and the code architecture is designed to accommodate it (via the extension manifest, ch.01 §7.2). The OUT OF SCOPE column is the queue for after M2.

| IN SCOPE (ship in M0–M2 if GO) | OUT OF SCOPE (explicitly deferred) |
|---|---|
| `hawking-agent`: loop, 6 core tools, permission model, worktree checkpoints, codebase-intel-lite | Hawking HF org live catalog (deferred until 32B `.tq` ready and §11 conditions met) |
| `hawking-eval`: thesis gate harness, 20–30 Tier-A tasks, `verdict.json` emission | Research Tab (POST-SHELL TIER-2) |
| `hawking-router`: subprocess-per-model routing, LRU governor, `/v1/models` aggregation | Parallel agent workstation UI (POST-SHELL TIER-2) |
| `.tq` GPU serving: LM-head TQ, first-class `.tq`+`meta.gguf` loader, parity CI gate, PPL gate, RSS gate | Remote Mac-Studio (POST-SHELL TIER-4) |
| Constrained-JSON tool calls via `json_mode` extension in `http.rs` | RLEF / DSPy / personalization flywheel (POST-SHELL TIER-3 moonshots) |
| Tauri 2 app shell: Chat tab (SSE streaming, Monaco, file tree), Diff Review panel, Context Stack right-rail, Agent-Run timeline, xterm terminal | Full JSON-Schema grammar compiler with `$ref` / `oneOf` / `anyOf` (fast-follow after M1 if GO) |
| `hawking-condense` (M2, gated on GO): HF download, convert, bake, register | YaRN/RoPE override, trained spec head, ANE routing (runtime fast-follows, not shell-gating) |
| `hawking-hub` (M2, gated on GO + §11 conditions): download, BLAKE3 verify, content-addressed store | Model Lab / Store live catalog UI (M2 only when §11 conditions met; ships gated) |
| Model Lab tab: local model list, "Use" switch, Condense Wizard (M2) | Agentic fine-tune at Condense time (POST-SHELL TIER-3) |
| Status bar chip: active model, health, tok/s | Parallel draft racing in Monaco split view (POST-SHELL TIER-2) |
| `THIRD_PARTY_NOTICES.md` + CI license gate (Apache-2.0, MIT, MPL-2.0 scan) | Memory Browser with semantic search UI (POST-SHELL TIER-2) |

**Rationale for key deferrals.** The full JSON-Schema grammar compiler (`$ref`, `oneOf`, `anyOf`) is deferred because the current `json_mode` path (flat object schema) is sufficient for the 6 core tools, and the thesis gate is more informative about model capability than grammar tightness. The grammar compiler is valuable but it is a quality-of-completions improvement, not a capability gate. The Research Tab is deferred because it requires a web-search tool (network I/O from the agent, not in the 6 core tools), a vector store for the knowledge graph, and a fundamentally different retrieval pattern from codebase-intel-lite; none of these are needed for the thesis gate. YaRN/RoPE override and the trained spec head are runtime improvements that benefit `hawking-serve` users independently of HIDE; they proceed on the runtime workstream timeline, not the HIDE build timeline.

---

## 10. Verification checklists (per milestone)

### Milestone 0 checklist

These items are binary: each must be true, checked by the engineer completing the work item, before the M0 sync point is called.

| # | Item | How to verify |
|---|---|---|
| M0-1 | Tool-calling endpoint unit test passes | `cargo test -p hawking-serve tool_call` exits 0 |
| M0-2 | `curl` test: POST `/v1/chat/completions` with `tools` array → well-formed `tool_calls` JSON | Manual curl; `jq .choices[0].message.tool_calls[0].function.name` returns expected tool name |
| M0-3 | `curl` test: `tool_choice: "none"` → `content` field present, no `tool_calls` | Manual curl; `jq .choices[0].message.content` returns non-null string |
| M0-4 | `hawking-agent-cli` completes trivial tool-call round-trip against local serve | CLI binary exits 0; tool call logged in state machine output |
| M0-5 | Tauri app builds on `macos-14` | CI step `pnpm tauri build` exits 0 |
| M0-6 | Chat SSE visible in app | Manual: type prompt, see streaming tokens appear in Chat tab |
| M0-7 | xterm terminal responds to keystrokes | Manual: open terminal tab, type `echo hello`, see output |
| M0-8 | File tree renders workspace root | Manual: file tree panel shows top-level files of the hawking repo |
| M0-9 | `hawking-router` starts 2 model children and routes | Integration test: 2 requests to different model names, both succeed |
| M0-10 | LM-head TQ parity: 3 fixtures × 32 tokens, token-for-token identical | `cargo test -p hawking-core lm_head_tq_parity` exits 0 |
| M0-11 | CI `macos-14` build green on latest push to `main` | GitHub Actions shows green for `build` job on the most recent commit to `main` |
| M0-12 | `hawking-agent-cli` exits 0 on trivial `read_file → done` task | `cargo run -p hawking-agent-cli -- --task trivial.json --base-url http://localhost:8080` exits 0; tool call logged |

### Milestone 1 checklist

| # | Item | How to verify |
|---|---|---|
| M1-1 | GPU TQ parity CI gate green | CI step `tq-parity` exits 0; log shows 3/3 fixtures match |
| M1-2 | PPL within +0.3 of Q4_K_M baseline | CI step `tq-ppl` reports delta ≤ 0.3; value logged in CI summary |
| M1-3 | RSS ≤ 70% of Q4_K_M baseline for same parameter count | CI step `tq-rss` reports ratio ≤ 0.70; value logged in CI summary |
| M1-4 | Agent completes ≥1 Tier-A task end-to-end | `hawking-eval run --tasks tier_a.jsonl` produces ≥1 row with `verdict: "PASS"` |
| M1-5 | Thesis gate verdict JSON emitted | `verdict.json` present in repo root, parseable, `verdict` field is one of GO/CONDITIONAL/KILL |
| M1-6 | Diff Review renders and accept/reject work | Manual: trigger an `edit_file` call; accept one hunk; verify file on disk reflects accepted hunk only; reject another; verify file unchanged |
| M1-7 | Context Stack renders live during agent run | Manual: run a 3-step task; verify right-rail updates after each `search_codebase` call |
| M1-8 | Timeline panel updates in real time | Manual: run a 3-step task; each step turns green on success; repair indicator appears on failure |
| M1-9 | `hawking-router` v1: 3 children, concurrent requests | Integration test: 3 concurrent requests to 3 different model roles all complete; no deadlock |
| M1-10 | Worktree created and cleaned up | Integration test: run a task; verify `/tmp/hawking-<uuid>` exists during run and is deleted after `done` |
| M1-11 | Codebase-intel query latency ≤ 2s on hawking repo | Benchmark: `hawking-agent-cli --query "where is ensure_tq_cache defined"` returns in ≤ 2s; result includes qwen_dense.rs |
| M1-12 | Repair loop fires and recovers on a synthetic failure | Integration test: inject a deliberate `edit_file` failure (target file does not exist); verify agent enters repair state; verify it recovers or reports `Failed` with reason (not a panic) |
| M1-13 | Agent run task log archived | `hawking-eval run` produces `task_log.jsonl` alongside `verdict.json`; each row is parseable and contains `task_id`, `verdict`, `wall_secs`, `tok_total` |

### Milestone 2 checklist

| # | Item | How to verify |
|---|---|---|
| M2-1 | Condense end-to-end: HF model → `.tq` + `meta.gguf` | `hawking-condense --hf-model Qwen/Qwen2.5-1.5B-Instruct --bits 3` completes; output files present |
| M2-2 | Condensed model registered in `~/.hawking/models.json` | `cat ~/.hawking/models.json` shows new entry with correct metadata |
| M2-3 | Condensed model loads and serves | `hawking-serve --model ~/.hawking/models/<name>.tq` starts; smoke suite passes |
| M2-4 | Round-trip verify events OK | `hawking-eval run` on a Tier-A task using the freshly condensed model produces `verdict: "PASS"` on ≥1 task |
| M2-5 | Download resume works | Unit test: interrupt download after chunk 2 of 3; resume; BLAKE3 of final file matches manifest hash |
| M2-6 | Catalog fixture test passes | CI test with local HTTP fixture: `hawking-hub list` returns ≥1 model entry |
| M2-7 | Model Lab tab shows local models | Manual: add entry to `~/.hawking/models.json`; open Model Lab; entry appears |
| M2-8 | Condense Wizard completes in-app | Manual: paste `Qwen/Qwen2.5-1.5B-Instruct` into wizard; complete all steps; model appears in Model Lab |
| M2-9 | Status bar chip reflects health and tok/s | Manual: stop serve; chip turns red within 10s. Restart; chip returns green; tok/s displayed |
| M2-10 | `THIRD_PARTY_NOTICES.md` present and CI license gate passes | `cargo deny check licenses` exits 0; `THIRD_PARTY_NOTICES.md` present in repo root |
| M2-11 | Router config hot-update: switch active hero model without restart | Manual: "Use" button in Model Lab; verify `hawking-router` logs a config update; next request uses new hero model |
| M2-12 | Condense progress events visible in wizard during long bake | Manual: start a condense of a 3B model; verify the wizard shows per-stage progress (download %, convert %, bake %) updating in real time |

---

## 11. HF distribution lane (when it is ready)

The Hawking HuggingFace org live catalog is not wired until four conditions are all simultaneously true. These are external dependencies, not code milestones — they require coordination between the condense workstream, the infra team, and the app shipping timeline.

**Condition 1: The 32B `.tq` model is production-ready.** The 32B Qwen `.tq` is quantized, benchmarked (PPL ≤ gate, RSS ≤ 14GB to fit within the 18–19GB Metal shared memory budget that is out of reach for Q4_K_M), and validated on the Tier-A eval task pool. "Production-ready" means: (a) the GPU TQ serving gate passes token-for-token parity; (b) decode throughput ≥ 15 tok/s on an M-series Mac with 24GB RAM; (c) the model card on HuggingFace is published with honest benchmark numbers.

**Condition 2: `hawking-hub` passes the download+verify gate.** The `hawking-hub` crate's download, chunk-resume, and BLAKE3 verify pipeline passes the full integration test suite — including a real network download of a multi-GB model, interrupted and resumed. This is measured, not estimated.

**Condition 3: The Hawking HF org index JSON is published at a stable URL.** The index is a JSON file at a stable HTTPS URL (e.g., `https://huggingface.co/datasets/hawking-ai/model-index/resolve/main/index.json`) that lists all pre-condensed models with their metadata (model ID, parameter count, bpw, PPL, tok/s, file size, BLAKE3 hash of `.tq` + `meta.gguf`). The URL must be stable because it is hard-coded in `hawking-hub` as the catalog root. Publishing this URL is an owner action, not a code action.

**Condition 4: App shell is stable (M2 shipped).** The Model Lab tab, Condense Wizard, and Status Bar chip are all shipped and stable. The live catalog is gated behind the Model Lab tab; if the tab has bugs, users encounter catalog errors on top of UI bugs, which is a worse experience than no catalog at all.

**The four conditions are independent.** Condition 1 (32B model ready) is on the condense workstream timeline. Condition 2 (hub gate passes) is a code milestone within this build plan. Condition 3 (catalog URL published) is an owner action. Condition 4 (M2 shipped) is a build milestone. They can be completed in any order, but the live catalog only activates when all four are simultaneously true. This means the catalog can be tested in staging (using a non-public HF dataset URL) while waiting for the 32B model to finish.

**Until all four conditions are met**, HIDE ships with:

- A local model directory at `~/.hawking/models/` scanned on startup.
- A manual "Add model" flow that accepts: (a) a local filesystem path to a `.tq` + `meta.gguf` pair, or (b) a direct HuggingFace model URL (which invokes the Condense Wizard). The "Add model" flow does not require the HF org catalog to be live — it works against any public HF model repo.
- The `hawking-hub list` command returns an empty list (with an informative message: "Hawking model catalog not yet available; add models manually with `hawking-condense`") until the catalog URL is published.

This ensures users who build from source or receive early access have a working model management workflow before the catalog exists. It also avoids the worst outcome: shipping the catalog URL before the 32B model is ready, causing users to see an empty or incomplete model store on their first launch.

---

---

## 12. Licensing and third-party notices

HIDE incorporates several open-source components whose licenses must be tracked from the first commit, not retroactively at ship time. The licensing posture is: MIT and Apache-2.0 components can be used freely; GPL components cannot be linked into the main binary (they can be invoked as subprocesses); LGPL components require careful link mode (dynamic link only); MPL-2.0 components (like some Mozilla crates) are file-level copyleft and require those specific files to remain open.

**Components with known license implications:**

| Component | License | Usage | Obligation |
|---|---|---|---|
| Monaco Editor | MIT | Tauri frontend (bundled) | Include MIT notice in `THIRD_PARTY_NOTICES.md` |
| xterm.js | MIT | Tauri frontend (bundled) | Include MIT notice |
| tree-sitter (Rust bindings) | MIT | `hawking-agent` codebase-intel | Include MIT notice |
| tree-sitter grammars (per language) | MIT / Apache-2.0 | Per-language parsing | Include per-grammar notice |
| ripgrep (invoked as subprocess) | MIT / UNLICENSE | `search_codebase` tool | No linking; subprocess invocation OK |
| portable-pty | MIT | Tauri PTY host | Include MIT notice |
| tokio | MIT | All Rust async | Include MIT notice |
| axum | MIT | hawking-router HTTP | Include MIT notice |
| hf_hub | Apache-2.0 | hawking-condense download | Include Apache-2.0 notice |
| patch (Rust crate) | MIT / Apache-2.0 | `edit_file` patch apply | Include notice |
| BLAKE3 | CC0 / Apache-2.0 | hawking-hub content addressing | CC0 = no obligation; Apache-2.0 portion = notice |
| OpenHands event model (reference only) | MIT | B1.1 timeline design inspiration | No code copied; no obligation |
| Cline diff view (reference only) | Apache-2.0 | B1.2 diff design inspiration | No code copied; no obligation |

**CI license gate.** A `cargo deny check licenses` step runs on every commit. The deny configuration (`deny.toml`) specifies the allowed license list and rejects any new dependency that introduces GPL, AGPL, or BUSL terms. The `THIRD_PARTY_NOTICES.md` file is regenerated by a CI script (`tools/gen_notices.sh`) that extracts license text from all bundled dependencies and formats them for inclusion in the app bundle. This file must be present before any public release.

**Third-party code that IS copied (not just referenced).** If any MIT-licensed UI component is copied directly into the frontend source (rather than installed as an npm dependency), the file must include the original copyright header. The CI license gate includes a `license-header-check` step that scans for files in `app/src/` that lack a copyright header and fails if any copied-code file is missing one.

---

## 13. Eval task pool — structure and curation criteria

The Tier-A task pool is the primary artifact of A0.2 and the load-bearing input to A1.4. Its quality determines whether the thesis gate produces a signal or noise. This section defines what makes a good Tier-A task and gives examples of each oracle type.

**Curation criteria for a valid Tier-A task.** A task is Tier-A if all five conditions hold: (1) the repo is publicly available and < 5k LoC (so the full repo map fits in the model's context window at 32B token budget); (2) the task is completable by a human with the codebase in < 5 minutes (this is the ceiling for a 5-minute wallclock gate); (3) the success criterion is machine-checkable without human judgment; (4) the task requires at minimum 2 tool calls (`read_file` + `edit_file` at minimum, or `search_codebase` + `edit_file`); (5) the task has a single correct output class (there is not an infinite family of correct implementations — the oracle distinguishes correct from incorrect).

**Oracle type examples:**

| Oracle type | Example task | Verification |
|---|---|---|
| `pytest` | "Fix the failing test in `tests/test_parser.py::test_empty_input`." | `pytest tests/test_parser.py::test_empty_input` exits 0 |
| `exact_file` | "Add a `VERSION` file at the repo root containing only `1.0.0`." | `diff VERSION expected/VERSION` exits 0 |
| `regex_match` | "Add a `greet(name)` function to `src/utils.py` that returns `Hello, {name}!`." | `python -c "from src.utils import greet; print(greet('World'))"` output matches `^Hello, World!$` |
| `exit_code` | "Fix the `cargo build` failure in this repo." | `cargo build` exits 0 |

**Task difficulty calibration.** The 20–30 task pool should span three difficulty bands: (a) trivial (5–7 tasks) — single-file single-function addition, oracle is `exact_file` or `regex_match`; these establish the floor for the local model; (b) medium (10–15 tasks) — multi-step tasks requiring `search_codebase` + `edit_file` + `run_tests`; these are the bulk of the eval and where real capability differences surface; (c) hard (3–5 tasks) — tasks requiring understanding of cross-file interfaces, fixing a test that requires changes in two files, or interpreting a failing test message to locate a root cause. The hard band is where CONDITIONAL verdicts cluster; the medium band is where GO vs KILL is decided.

**The cloud control run.** A1.4 runs the same task pool against a cloud model (via the `ModelClient` trait pointing at an external base-url). The cloud model serves as the calibration anchor: if the cloud model achieves < 80% completion on the Tier-A pool, the tasks are too hard and must be re-calibrated. The cloud run is expected to achieve > 85%; if it does not, the task pool is revised before the local run is used as a verdict. This prevents a situation where the local model is penalized for tasks that are genuinely beyond any model's capability.

**Task pool maintenance.** The task pool is a living artifact. After A1.4 fires, tasks that produced ambiguous verdicts (e.g., tasks where the oracle passed but the agent's solution was clearly wrong in ways the oracle did not catch) are retired or upgraded to use a stronger oracle. Tasks that every model — cloud and local — fails are examined for oracle correctness: a task where the oracle itself has a bug is worse than no task at all. The pool grows over time as the HIDE eval workstream identifies new failure modes worth testing. By M2, the pool should have ≥ 40 tasks across the three difficulty bands to provide a more stable measurement of the completion rate.

---

*This chapter is the operational blueprint. Every item has a defined input, output, and verification. The thesis gate at M1 is the load-bearing measurement. The kill protocol ensures we find out cheaply if the local model cannot carry the load. Everything else follows from the verdict.*
