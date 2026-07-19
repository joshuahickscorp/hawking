# HIDE Tool + Skill + Plugin + MCP ABI

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.3, §3.5, §5), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`, `hawking_ide_frontier_2026_07_19.md` (§5.6, §5.7, §5.10). Code confirmed read-only against `main@4fbca8bc` and the sealed backend `@5a99d0e2`.
Status: specification for the HIDE action plane (Bible Book IX). Every load-bearing claim carries a file:line citation or a parity-spec id, and every Hawking mechanism carries a readiness label: real-and-wired / real-but-unwired / partial / stub / missing.
Covers parity ids: `skills.progressive_disclosure`, `plugins.bundles`, `hooks.lifecycle`, `mcp.client_host_server` (with `palette.unified`, `perm.rule_engine`, `trust.workspace_gate`, `subagents.file_defined` as adjacent gates).

## 1. The one-sentence defect

HIDE has a rich, tested action plane that is unwired from the live turn. The active runtime holds only thin API-shaping tool parsing; the dispatch loop, verifying edit applier, MCP client, and lossless tool-spec-decode all exist as real code that no live path calls. This ABI defines the contract that reconnects them and states honestly which parts are shipping capabilities versus packed primitives.

Verified surface (readiness map):

| Piece | Readiness | Location |
|---|---|---|
| Tool preamble render + completion parse (`render_tools_preamble` / `extract_tool_calls`) | real-and-wired (API-shaping only) | `hawking-serve/tool_calls.rs:5-16` |
| Tool-execution / dispatch loop in the runtime | missing | T6 CONFIRMED, archaeology §3.3 |
| Tool-bearing SSE (streams incrementally) | missing (buffers to completion, non-streaming) | T7 CONFIRMED |
| Per-token schema mask on the batched path | missing (`driver.rs:52-110` forces `json_mode:false`) | T8 CONFIRMED |
| `JsonConstraint` structural well-formedness mask | real-but-unwired (single-seq `generate()` only) | `hawking-core/json_constrain.rs:63` |
| `GrammarConstraint` (`required_keys` / `Choices`) | stub (post-hoc `validate()` only, zero callers) | `json_constrain.rs:325-350` |
| Verifying edit applier (base_hash optimistic concurrency) | real-but-unwired (packed) | `hide-tools` @ `5a99d0e2` |
| Sandboxed `shell.run` (watchdog), `proc` (exec-nonzero-as-data), ignore-walker search, git worktree trio | real-but-unwired (packed) | `hide-tools` @ `5a99d0e2` |
| JSON-RPC MCP client (stdio + Streamable HTTP) | real-but-unwired (packed) | `hide-tools` @ `5a99d0e2` |
| Training-free lossless tool-spec-decode (schema jump-forward + prompt-lookup) | partial (packed) | `hawking-orch/tool_spec_decode.rs` @ `5a99d0e2` |
| Exact-match lossless verifier (accept iff `argmax == draft`) | real-but-unwired (single-seq) | `hawking-speculate/verifier.rs:77-133` |
| MCP host + MCP server (expose HIDE's own tools) | missing | archaeology §3.5, parity `mcp.client_host_server` |
| Skills runtime (SKILL.md) | missing (FE `ContextStack` SKILLS is a hardcoded const) | parity `skills.progressive_disclosure` |
| Plugins / marketplaces | missing | parity `plugins.bundles` |
| Hooks engine | packed_unwired (CommandRouter / GateBook in `hide-backend`) | parity `hooks.lifecycle` |

The reintegration is Phase 1/2 of the ladder (`HIDE_LIVE_ARCHAEOLOGY.md` §6): lift `hide-tools` + `hawking-orch` out of `5a99d0e2` and wire them into the flat kernel loop and the batched serve path. No invention is required for the tool core; skills, plugins, MCP host/server, and hooks are new build on top of proven scaffolding.

## 2. Tool ABI: the typed contract

A HIDE tool is a typed, effect-declaring unit. The contract is what lets the runtime schedule reads in parallel, commit results deterministically, gate effects at the permission boundary, and roll back on failure. Every field is load-bearing for a downstream mechanism named in the last column.

| Field | Meaning | Consumed by |
|---|---|---|
| `name` | namespaced id (`fs.read`, `mcp__server__tool`, `plugin:skill`), stable across a session | deferred registry (§3), cache-stable prompt prefix |
| `description` | one-line intent for the model + palette; the only thing loaded before discovery | progressive disclosure (§3, §7) |
| `input_schema` | typed, validated params (JSON Schema); the grammar target for first-try-valid decode | schema mask + jump-forward (§6) |
| `output_schema` | typed result shape; large payloads returned as an artifact handle, not inline text | artifact ledger, context budget |
| `effects` | declared effect set: `{reads, writes, network, spawns, deletes, external_send}` | permission gate, parallel scheduler (§5) |
| `permissions` | required capability tokens matched against the rule engine before execution | `perm.rule_engine`, `HIDE_PERMISSION_AND_EFFECT_SYSTEM` |
| `idempotency` | `pure_read` / `idempotent_write` / `effectful`; only `pure_read` is speculation-eligible | parallel reads (§5), speculation gate (§6) |
| `timeout` | wall-clock budget; a watchdog kills the process (the `shell.run` watchdog already does this) | `hide-tools` sandboxed exec |
| `streaming` | may the tool emit incremental output before completion | tool feed rendering, `loop.collapsed_tools` |
| `pagination` | cursor contract for bounded output; warn at 10k tokens, hard cap 25k with spill-to-disk | MCP output limits (parity `mcp.client_host_server`) |
| `rollback` | inverse or transaction handle; writes are transactions, not fire-and-forget | edit applier, `session.checkpoint_rewind` |
| `sandbox` | isolation profile (fs mounts, egress) the tool runs under | `security.sandbox`, `hide-security` Seatbelt |

Two enforcement rules that are non-negotiable:

- Effects are declared, not inferred. A tool that does not declare `writes`/`network`/`external_send` may not perform them; the scheduler and permission gate trust the declaration and the sandbox enforces it (defense in depth, per `hawking_ide_frontier_2026_07_19.md` §5.9: probabilistic model defenses cannot replace deterministic environment boundaries).
- Writes are transactions. The verifying edit applier (`search_replace` / `apply_patch` / `write_file`) uses `base_hash` optimistic concurrency (packed in `hide-tools` @ `5a99d0e2`): a stale base hash fails the edit as data rather than clobbering the file. `apply_patch` correctness is flagged NEEDS PROBE in the archaeology (§4) and must be re-tested on reintegration. This is the mechanical basis for `session.checkpoint_rewind`.

Non-zero process exit is data, not an exception (the packed `proc` tool already models this): the model sees the exit code and stderr as a tool result and can react, rather than the turn aborting.

## 3. Registry: deferred discovery + cache-stable stable order

Tool definitions are expensive (Anthropic reports single-request tool definitions of 55K to 134K tokens; `hawking_ide_frontier_2026_07_19.md` §5.6). The registry is therefore part of the PromptABI (`HIDE_SPEED_FRONTIER`, dossier §5.2), and its ordering is a cache-correctness invariant, not a cosmetic detail.

- Two-tier disclosure. Only 3 to 5 always-present core tools ship full schemas up front. Everything else is a stable deferred stub (name + one-line description); the full `input_schema` is appended to context only after the model requests that tool by name. This matches the parity requirement in `mcp.client_host_server` ("deferred tool loading default-on") and feeds `palette.unified` (the `/` palette lists stubs; selection hydrates the schema).
- Stable order is mandatory. Cache hits require an exact shared prefix (dossier §5.2); reordering or redefining tools mid-session breaks reuse. The registry emits tools in a deterministic canonical order and appends new tools rather than reordering existing ones. Reuse the stable-order registry discipline already proven in `hawking-seed-c/providers/registry.rs` (one registry answering which impl provides a capability, with loc/bytes/source/tests/rollback), which is real-but-unwired and integration-tested.
- Namespaces are small and stable. `mcp__<server>__<tool>`, `plugin:<skill>`, and built-in namespaces are fixed; a plugin or MCP server that changes its tool set changes the registry version (and thus the `tool_registry_id` bound into the state capsule, `HIDE_STATE_CAPSULE_ABI` §4).

## 4. Programmatic / code-mode orchestration

Round trips, not generation, dominate tool latency (dossier §5.6: Programmatic Tool Calling reports 37% average token reduction and elimination of 19+ inference passes in a 20-call example). HIDE exposes a code-mode control program so the model can express deterministic control flow once instead of returning to the model after every result.

- The model emits a small program (loops, joins, filters, fan-out/reduce over tool calls) that runs in a sandboxed interpreter with the typed tool registry as its callable surface.
- Intermediate results stay outside model context. Only the final reduced value (or an artifact handle) re-enters the transcript. A 20-file grep-then-filter-then-summarize becomes one control program, not 20 observe-then-decide turns.
- The interpreter honors the same `effects` / `permissions` / `sandbox` declarations as direct calls: a program may only call tools whose effects are permitted, and writes inside a program are still transactions.
- Bounds: every program runs under a time/turn/resource budget (dossier §5.8) and is itself a reviewable artifact (its source and its call log are in the effect ledger, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §2).

Readiness: missing in the active tree; this is new build, but it composes directly on the packed typed-tool surface and the sandboxed `shell.run` in `hide-tools`.

## 5. Safe parallel read-only tools + deterministic commit order

Concurrency is gated on the `effects`/`idempotency` fields, and correctness is preserved by committing in a fixed order rather than completion order.

- Only `pure_read` tools (declared `reads`-only, no `writes`/`network`/`external_send`, idempotent) may run concurrently. This is the same envelope the frontier requires for tool speculation (dossier §5.7: "authorized, local/private, side-effect-free, idempotent reads").
- Writes and effectful tools are serialized through the single-writer effect ledger (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §2; workspace single-writer lock, dossier §5.9). A read that races a pending write is ordered after it or refused, never interleaved.
- Deterministic commit order. Parallel read results are committed to context in a stable, declared order (call order), not the order they finish. This is required twice over: for prompt-cache stability (a fixed prefix, §3) and for byte-reproducible runs (`sdk.headless` supremacy: fixed seed + resident weights + deterministic commit order = replayable). Completion-order commit would make the same inputs produce different prefixes.
- External sends are never speculative or parallel-by-default (dossier §5.7, "Ghost Tool Calls": even discarded external requests can leak predicted intent). `external_send` effects always route through the permission gate (`HIDE_PERMISSION_AND_EFFECT_SYSTEM`).

## 6. First-try-valid tool calls (the Hawking speed lever)

This is the sharpest supremacy claim in the action plane, and it is a reintegration, not research. Claude Code cannot mask its own decode at the token level from the client side; HIDE runs the decoder, so it can constrain and speculate on tool syntax at zero network cost.

The mechanism, assembled from real-but-unwired parts:

1. Per-token schema mask. `JsonConstraint` (real-but-unwired, `json_constrain.rs:63`, single-seq only) enforces JSON well-formedness during decode. Schema-level enforcement of `required_keys` / `Choices` is currently only the post-hoc `GrammarConstraint::validate()` (stub, `json_constrain.rs:332-350`). The per-token mask FSM is the deferred item the code itself names: threading a `grammar` field through the 44 `GenerateRequest` construction sites and the runtime FSM marked "RUNTIME-SIDE - LATER" in `hawking-orch/grammar.rs` (`json_constrain.rs:320-323`).
2. Jump-forward + prompt-lookup. The fixed prefix of a tool call (the schema-mandated key names, braces, quotes) is not sampled token by token; it is emitted directly, and the model only decodes the free-choice spans. `hawking-orch/tool_spec_decode.rs` (partial, packed) already implements this training-free and lossless (schema as the structural draft, retrieved historical calls as the value draft).
3. Lossless verification. Drafted tokens are accepted only when they match the target's greedy argmax, using the exact-match verifier (`verifier.rs:77-133`, real-but-unwired): accept iff `argmax == draft`, which is bit-identical to unspeculated greedy. No quality is traded for the speedup.
4. Tag dispatch, not naive masking. A single mask over prose-versus-tool would suppress legitimate tool selection (dossier §5.6). Use two-stage / tag-dispatch semantics (XGrammar-2 style) and measure tool-selection recall, not merely JSON validity.

Build item (gates every claim below): reintegrate `hawking-orch::tool_spec_decode` and the single-seq `JsonConstraint`/verifier into the batched serve decode path (`driver.rs:52-110`, which today forces `json_mode:false` and does no draft/verify). Until that lands, tool constraint and spec-decode exist but are unreachable over HTTP (T8 CONFIRMED). See `HIDE_SPEED_FRONTIER` for the decode-path integration and the acceptance/wasted-work measurement contract.

## 7. MCP: client + host + server

Parity `mcp.client_host_server` requires HIDE to be all three. Today it is one, packed.

- Client (real-but-unwired, packed). `hide-tools` @ `5a99d0e2` carries a JSON-RPC MCP client over stdio and Streamable HTTP. Reintegration must add the remaining live transports (SSE is deprecated but still consumed; WebSocket) and OAuth 2.0 (DCR / CIMD / pinned scopes) per the parity contract.
- Host (missing). HIDE hosts external MCP servers behind its own security/tool gateway (dossier §5.10): a hosted server's tools enter the registry as `mcp__server__tool` with declared effects, are subject to the same permission gate and sandbox, and never bypass the effect ledger. Output is bounded (warn 10k / hard 25k tokens, spill-to-disk), and `@server:proto://path` resources plus `/mcp__server__prompt` prompts surface in the `@` picker and `/` palette (`palette.unified`).
- Server (missing). HIDE exposes its own tools and agents as an MCP server so external agents (and the ACP boundary, dossier §5.10) can drive the HIDE surface. This is what keeps the IDE from becoming a closed harness.
- Manifest layering. `.mcp.json` (project) under `~/.claude.json` (user/local) under managed, with the same precedence and merge rules as the config system (`config.settings_precedence`), and inert until the workspace trust gate accepts (`trust.workspace_gate`; no server is launched from project config before trust).

Supremacy (gated): keep long-lived local stdio servers warm across sessions, and snapshot the resolved + authenticated server set into a state capsule so a fork inherits connected servers with no re-handshake (`HIDE_STATE_CAPSULE_ABI`; gated on capsule exposure and session-slot affinity). Transient provider continuation ids belong in provider state and must never replace the canonical local tool/artifact ledger (dossier §5.6).

## 8. Skills (`skills.progressive_disclosure`)

Readiness: missing. The FE `ContextStack` SKILLS entry is a hardcoded const with no runtime behind it (archaeology §3.3). The full mechanism is new build.

Contract (Claude-compatible so an existing corpus imports verbatim):

- SKILL.md = YAML frontmatter + body. A two-tier registry loads only `name` + `description` into context; the body loads lazily on trigger (auto description-match, or explicit `/skill-name`); bundled scripts load only when referenced. This is the same two-tier shape as the tool registry (§3).
- Turn-scoped `allowed-tools`. A skill pre-approves a specific tool subset only for the turn that invokes it, then the approval lapses. This binds skills to the permission system rather than granting standing capability (`HIDE_PERMISSION_AND_EFFECT_SYSTEM`).
- `context:fork` + agent. A skill may run in a forked context so its scratch work does not pollute the parent transcript. On HIDE this is the native state-fork primitive, not a re-prefill.
- Scope precedence enterprise > personal > project; `plugin:skill` namespacing; `!` dynamic command injection in the body; `model` / `effort` overrides. Custom commands are merged into the skill surface (one palette, `palette.unified`).
- Claude-compatible import: read an existing `.claude/skills/**/SKILL.md` tree with the documented frontmatter semantics, no rewrite required (mirrors the `config.claude_md` migration discipline).

Supremacy (gated on skill runtime + state-capsule exposure): pre-index the skill corpus into a resident capsule and hydrate a body by forking it, giving O(1) load with no per-turn token accounting and no eviction cliff; bundled scripts can touch `.tq` artifacts, the GPU, and private data a cloud sandbox cannot reach.

## 9. Plugins (`plugins.bundles`)

Readiness: missing. New build; the install-inventory and cost accounting reuse the packed provider-registry fields.

- `.claude-plugin/plugin.json` manifest bundles skills / agents / hooks / `.mcp.json` / `.lsp.json` / monitors / themes / output-styles / settings.json, plus `userConfig` for sensitive per-install values. Convention dirs per component type; namespaced names.
- Lifecycle: install / enable / disable / uninstall via `/plugin` from marketplaces (`marketplace.json`: git / owner-repo / URL / local). SHA-pinned for supply-chain integrity. Auto-update opt-in.
- Pre-install honesty. A "Will install" inventory enumerates exactly what the bundle grants (tools, hooks, MCP servers, permissions) plus a per-turn context-cost estimate, and unused-plugin detection flags dead weight. This is the same "which capability, why, loc/bytes, source, tests, rollback" accounting already proven in `hawking-seed-c/providers/registry.rs` (real-but-unwired): reuse it as the plugin inventory backend rather than inventing a second accounting path.
- Trust ordering: a plugin's hooks/MCP/skills are parsed as data and stay inert until the workspace trust gate accepts (`trust.workspace_gate`), and admin allowlists constrain marketplaces.

Supremacy: a HIDE bundle can ship local runtime assets a cloud plugin structurally cannot: quantized model shards (`.tq`), Metal kernels, and state-capsule presets, so installing a plugin can install an entire specialized local model + workflow. Gated on the plugin runtime plus `.tq` native serving (partial, feature+env-gated, archaeology §3.1).

## 10. Hooks (`hooks.lifecycle`)

Readiness: packed_unwired. `hide-backend` carries CommandRouter / GateBook; no hook engine is wired (archaeology, parity `hooks.lifecycle`).

- Event taxonomy: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SessionStart`, `PreCompact`, `Notification`, `SubagentStop`, `PermissionRequest`, `ConfigChange`, `InstructionsLoaded`, each with matchers.
- Wire contract: JSON on stdin, exit-code semantics, and a JSON decision object (`allow` / `deny` / `ask`, `block` + `reason`, `updatedInput`, `additionalContext`, `updatedToolOutput`).
- `PreToolUse` is a hard pre-execution gate. It runs before the tool and can deny / ask / allow / rewrite input; it is not advisory. It composes with the rule engine (`perm.rule_engine`) as the programmable extension of the effect gate.
- Trust ordering (hard). No project-defined hook runs before the workspace trust gate accepts (`trust.workspace_gate`; the CVE-2026-33068 class is exactly project config resolving before trust). Hooks are data until trust, executable after.

Supremacy: local hooks are latency-free and can read and modify true runtime state (live KV, resident model, real tool output) rather than a serialized transcript, and the gate they enforce can be pushed to the syscall / sandbox layer (`hide-security` Seatbelt rendering, real-but-tested; egress proxy / microVM still seams). See `HIDE_PERMISSION_AND_EFFECT_SYSTEM` for how hook decisions and rule-engine decisions compose.

## 11. Exposure: the missing wire (ordered build items)

Nothing here is greenfield for the tool core; the order matters because later items depend on earlier ones.

1. Reintegrate `hide-tools` (verifying edit applier, sandboxed `shell.run`, `proc`, ignore-walker search, git worktree trio, MCP client) behind the typed Tool ABI (§2), and re-probe `apply_patch` correctness (archaeology §4).
2. Wire a dispatch loop into the flat kernel turn so tool calls execute and results re-enter context (closes T6); stream tool output incrementally (closes T7).
3. Reintegrate `hawking-orch::tool_spec_decode` + `JsonConstraint` + the exact-match verifier into the batched decode path `driver.rs:52-110` (closes T8); thread `grammar` through the 44 `GenerateRequest` sites. See `HIDE_SPEED_FRONTIER`.
4. Add MCP host + server on top of the reintegrated client; layer `.mcp.json` under the trust gate.
5. Build the skills runtime (two-tier registry, turn-scoped `allowed-tools`, `context:fork`), then the plugin/marketplace layer on top of it, then the hook engine wired to the permission gate.
6. Persist a resolved + authed tool/MCP/skill set into the state capsule (`HIDE_STATE_CAPSULE_ABI`) so a fork inherits the warm action plane. Gated on capsule exposure + session-slot affinity.

## 12. Parity vs supremacy ledger

| Claim | Kind | Status | Gate |
|---|---|---|---|
| Verifying transactional edits with optimistic concurrency | parity | supported once reintegrated | `hide-tools` lift + `apply_patch` re-probe (§11.1) |
| Live tool dispatch loop, results re-entering context | parity | build item | §11.2 (closes T6/T7) |
| MCP client over live transports + OAuth | parity | supported once reintegrated | §11.4 |
| MCP host + server (drive HIDE from external agents) | parity | build item | §11.4 |
| Skills / plugins / hooks with Claude-compatible import | parity | build item | §11.5 |
| Deferred discovery + cache-stable stable-order registry | parity | build item | §3 (PromptABI, `HIDE_SPEED_FRONTIER`) |
| First-try-valid tool calls, bit-identical to greedy | supremacy | supported once wired | §6 build item (spec-decode into batched path) |
| Zero-network per-token constraint + speculation | supremacy | structural (local decode) | §6 wiring |
| Programmatic orchestration eliminating round trips | supremacy | build item | §4 |
| Warm long-lived MCP/skill set inherited by a fork | supremacy | gated | §11.6 + capsule exposure (`HIDE_STATE_CAPSULE_ABI`) |
| Bundles shipping local model shards / kernels / capsules | supremacy | gated | plugin runtime + `.tq` serving (partial) |
| Latency-free hooks over true runtime state, syscall-enforced | supremacy | gated | hook engine + OS enforcement (`hide-security` seams) |

Every parity row is a wiring problem on proven parts. Every supremacy row is gated on a specific build item and, where the mechanism is not yet reachable, is labeled a build item or structural claim, never presented as a shipping capability. The single unlock that turns the largest number of these green is step §11.3: putting the constraint + spec-decode primitives onto the batched serve path.
