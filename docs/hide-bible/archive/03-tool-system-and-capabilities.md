# 03 · Tool System & Capability Surface

> **Purpose:** Define the one canonical way HIDE turns "the model wants to do something in the world" into "a scoped, audited, schema-valid effect" — the tool wire-format, the registry, the dispatcher, the complete built-in catalog, the permission/safety surface, MCP host+server interop, and the durable self-authored skill library — engineered so that adding a tool never touches `core/`, and so that small local models emit valid tool calls **by construction** rather than by luck.

**Status:** DESIGN. This chapter owns the **tool wire-format contract** (§4.2) and the **tool-side permission policy schema** (§4.9) that Chapters 02 (agent kernel), 04 (context), 05 (codebase intelligence), 06 (model layer), and 10 (security) bind to. It *extends* — never contradicts — Ch.01's Event envelope (`tool.*` events, `capability_grant_id`) and extension manifest (`manifest.toml`, declarative scoped capabilities, deny-beats-allow, no ambient authority), and it *references* Ch.10 as the canonical owner of the OS sandbox + capability/permission model + prompt-injection defense. Where this chapter specifies sandbox mechanism, treat it as the **tool-side surface** Ch.10's model plugs into; Ch.10 wins any conflict on enforcement.

The runtime is a **stable localhost OpenAI-compatible HTTP surface** (Ch.01 §4.3). The serve layer's constrained decode (`json_constrain.rs`, today a JSON *well-formedness* state machine driven by a `json_mode: bool`) is the seam we extend to **schema/grammar-constrained tool emission** (§4.3); deeper model hooks are designed but tagged *later*.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + competitor limits (cited)](#3-state-of-the-art--competitor-limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [Module layout](#41-module-layout)
   - 4.2 [The tool wire-format & schema — CROSS-CUTTING CONTRACT](#42-the-tool-wire-format--schema--cross-cutting-contract)
   - 4.3 [Schema-enforced model-side emission (constrained decode)](#43-schema-enforced-model-side-emission-constrained-decode)
   - 4.4 [The tool registry (manifest, versioning, discovery, namespacing)](#44-the-tool-registry-manifest-versioning-discovery-namespacing)
   - 4.5 [Dispatch & result types](#45-dispatch--result-types-typed-large-output-streaming-caching-dry-run)
   - 4.6 [The built-in tool catalog](#46-the-built-in-tool-catalog)
   - 4.7 [Edit strategies (the hard part)](#47-edit-strategies-the-hard-part)
   - 4.8 [Shell / PTY tools](#48-shell--pty-tools)
   - 4.9 [Permission & safety model — CROSS-CUTTING CONTRACT](#49-permission--safety-model--cross-cutting-contract)
   - 4.10 [MCP — host/client AND server](#410-mcp--be-a-host-client-and-expose-hide-as-a-server)
   - 4.11 [Self-authored tools (the skill library)](#411-self-authored-tools-the-durable-skill-library)
   - 4.12 [Observability](#412-observability-every-call--event)
5. [How we EXCEED ("cloud literally cannot do this")](#5-how-we-exceed-cloud-literally-cannot-do-this)
6. [Failure modes / edge cases / mitigations](#6-failure-modes--edge-cases--mitigations)
7. [Extensibility / plugin points](#7-extensibility--plugin-points)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions / dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)

---

## 1. Purpose & scope

A tool is the **only** way an agent affects the world. Everything the model "does" — read a file, run a test, apply a diff, open a browser, query a DB, call an MCP server — is a tool call. This chapter specifies that surface end to end so that no further design is needed on it:

**In scope**

- The canonical **tool wire-format**: one JSON-object spec for a call and its result, with two faithful projections — OpenAI tool-calls (for interop) and a **grammar/JSON-schema constraint** that makes the *model itself* emit valid calls.
- The **tool registry**: manifest-declared tools, semantic versioning, discovery, capability negotiation, namespacing, hot-reload.
- The **dispatcher**: typed results, large-output/streaming handling, caching/memoization, dry-run/simulation, cancellation, timeouts, output caps.
- The **complete built-in catalog** with argument schemas: filesystem, edit strategies (unified-diff, search/replace, AST-aware, multi-file atomic), shell/PTY, code search (text+symbol+semantic), tests, build, git (incl. worktrees), refactor/rename, format/lint, web fetch/search, browser/computer-use, package managers, DB/HTTP clients.
- The **permission & safety model** *tool-side*: ask/auto/deny, capability grants, per-tool policy, worktree confinement, allow/deny-lists, network policy, timeouts/caps, injection-aware gating.
- **MCP**: HIDE as an MCP **host/client** (stdio + Streamable HTTP) and HIDE **exposed as an MCP server**.
- **Self-authored tools**: the agent writes, tests, registers, and persists new tools into a durable skill library (Voyager-style), with retrieval by embedding.
- **Observability**: every call → events for the Ch.04 context-stack and Ch.09 UI.

**Out of scope (delegated)**

| Concern | Owner |
|---|---|
| Canonical OS sandbox + capability model + injection defense (enforcement) | **Ch.10** (this chapter specifies the *tool-side surface* it plugs into) |
| The agent loop that *decides* which tool to call, parallelism, retries | **Ch.02** (consumes the wire-format here) |
| Context manifest / how `tool.result` enters the window | **Ch.04** (we emit `tool.result`; ch.04 ranks & packs it) |
| Symbol/reference/dataflow query engine behind `find_definition` et al. | **Ch.05** (our search tools are *thin wrappers* over its §4.11 query API) |
| Sampler / grammar kernel internals, runtime endpoints | **Ch.06** (we extend `json_constrain.rs`'s seam; deep hooks = *later*) |
| Event log, IPC, manifest schema, capability *grant ledger* | **Ch.01** (we extend the envelope & manifest) |

**The over-engineering mandate, applied:** the litmus test from Ch.01 — *"to add capability X, does anyone touch `core/`?"* — is enforced here with teeth. A new built-in tool is a registry entry + a `Tool` impl; a new third-party tool is a WASM plugin manifest; a new MCP server is a config line; a new *agent-authored* tool is a row in the skill DB. **None of those edit the dispatcher, the registry core, or the event schema.** If a proposed feature would, the design is wrong and is revised (§7).

---

## 2. Tenets

These eleven tenets govern every later decision; each is cited downstream.

| # | Tenet | Consequence |
|---|-------|-------------|
| **TT1** | **One wire-format, many projections.** There is exactly one canonical `ToolCall`/`ToolResult` JSON shape. OpenAI tool-calls, the constraint grammar, the MCP `tools/call` mapping, and the event payload are all *projections* of it. | No tool is defined twice. New transports project the same shape (§4.2). |
| **TT2** | **Valid by construction, not by correction.** A local model's tool call is made schema-valid at *decode* time via grammar/JSON-schema masking — not parsed-then-retried. Retry is a fallback, not the plan. | We own the sampler; we exploit it (§4.3). This is the single biggest small-model lever. |
| **TT3** | **Capability, not ambient authority.** Every tool call carries a `capability_grant_id`; the tool receives *only* the scoped powers the grant authorizes; **deny beats allow**; no tool has implicit FS/shell/net access. | Extends Ch.01's manifest + Ch.10's model. A tool with no grant can do nothing (§4.9). |
| **TT4** | **Effects are recorded, never replayed.** A `tool.result` is the *observed outcome* (or a `bytes_ref` to it). Replay applies recorded bytes; it never re-runs the tool (Ch.01 T3). | The dispatcher is the single point that turns an `Action` into a recorded `Observation` (§4.5). |
| **TT5** | **Typed results, bounded outputs.** Results are typed; large outputs go to the blob CAS as `bytes_ref`; streaming tools emit `tool.progress`; every tool has an output cap, a timeout, and a cancellation token. | No unbounded tool output floods the log, the window, or the UI (§4.5). |
| **TT6** | **Tools are data, not code paths.** A tool is a manifest entry + an impl behind one trait, discovered at runtime. The dispatcher knows *zero* tool names at compile time. | Registry-driven; hot-reloadable; MCP/skill/WASM tools are first-class (§4.4). |
| **TT7** | **Determinism where the world allows it.** Pure/queryable tools (search, stat, read) are deterministic and memoized by content hash; impure tools (shell, net) record their non-determinism so replay stays deterministic. | Caching + dry-run + reproducible runs (§4.5, Ch.01 T6). |
| **TT8** | **Injection-aware by default.** Tool *output* is untrusted data, not instructions. Output that re-enters the context is tagged `provenance=tool-output`; tools that combine private-data + untrusted-content + exfil capability trip the "lethal trifecta" gate. | The confused-deputy/poisoning defense lives at the tool seam (§4.9, §6), deferring canonical policy to Ch.10. |
| **TT9** | **Spend lavishly, locally.** No per-token/-call cost ⇒ dry-run simulation, exhaustive pre-flight checks, parallel fan-out (8 files = 8 tool calls), full result logging are *defaults*. | Multi-call fan-out is a dispatcher primitive (§4.5), echoing Ch.01 T9. |
| **TT10** | **The agent can grow the toolset.** A successful, verified procedure becomes a persisted tool retrievable by embedding — the skill library — without a human in the loop *to author*, but with a human in the loop *to grant capability* on first dangerous use. | Voyager-style self-authoring, gated by TT3 (§4.11). |
| **TT11** | **Forward-compatible on the wire.** Tool schemas are versioned; unknown result fields survive round-trips; a manifest declares the `wire_version` it speaks. | Additive-by-default; old sessions replay against new tools (§4.4, Ch.01 T10). |

---

## 3. State of the art + competitor limits (cited)

### 3.1 How agents represent tool calls — XML vs JSON vs native function-calling

The field has visibly converged-then-split. The two poles, with their measured trade:

- **Native JSON function-calling** (OpenAI tool-calls): the model emits a structured `tool_calls` array; frontier models are RL-trained for it. Anthropic reports a **<0.2% format-failure rate across ~300k calls** on a recent Sonnet, but independent analysis finds **15–25 percentage points of *leaf values* inside otherwise-valid JSON are wrong** — i.e. schema-shape is near-solved, semantic correctness is not ([structured-output reliability survey, arXiv:2510.14453](https://arxiv.org/html/2510.14453v1)). **OpenAI Structured Outputs** (`response_format: {type:"json_schema", strict:true}`, Aug 2024) and **Anthropic strict tool use** (`strict:true`, GA Nov 2025) both now state *in their own docs* that they **compile the JSON schema to a grammar and constrain sampling** so output is valid-by-construction ([OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs); [Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)). This is the decisive PROVEN fact for HIDE: **the frontier already enforces tool schemas at decode time. We own the decoder, so we can do it for *any* local model.**

- **XML-style tags** (Cline's original `<read_file><path>…</path></read_file>`): the argument for weaker/local models is less escaping, partial-malformation recoverability, and no strict-JSON tax. The evidence is real but double-edged: a JSON-output *requirement* cut GSM8K accuracy by **27.3 points** on some models (compute spent on JSON validity instead of reasoning) ([SLM function-calling, arXiv:2504.19277](https://arxiv.org/pdf/2504.19277)); yet **Cline itself migrated XML → native JSON in v3.35** citing ~100% multi-tool reliability on frontier models ([Cline v3.35](https://cline.bot/blog/cline-v3-35)), and **Roo-Code removed its XML selector** after legacy XML showed ~10% failure and `apply_diff` >15% ([Roo-Code #4047](https://github.com/RooCodeInc/Roo-Code/issues/4047)). **The synthesis HIDE takes:** the format war is a *symptom of not owning the decoder*. With constrained decode (TT2), the surface format is an internal choice and *neither* JSON nor XML can be malformed — so we pick the token-cheapest faithful encoding and enforce it.

- **Edit-as-tool formats** are their own genre. **Aider** uses `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks ("diff" format), plus `udiff`, `whole`, `diff-fenced`, and auto-selects per model via its leaderboard ([Aider edit formats](https://aider.chat/docs/more/edit-formats.html)). **Cline's `replace_in_file`** uses `------- SEARCH` / `=======` / `+++++++ REPLACE` with model-tuned variants ([Cline tools guide](https://docs.cline.bot/exploring-clines-tools/cline-tools-guide)). **Cursor's "fast apply"** is a *speculative-decoding* full-file rewrite (~1000 tok/s, using the original file as the draft) trained because raw search-replace was too fragile ([Cursor instant apply](https://cursor.com/blog/instant-apply); [Fireworks writeup](https://fireworks.ai/blog/cursor)). The lesson: **string-match edits fail on whitespace/ambiguity; the robust paths are either a verifying applier (Aider's exact-match + error feedback) or a learned/AST-aware applier.** HIDE does *both* and adds an AST tier (§4.7).

- **CodeAct** (OpenHands `CodeActAgent`): instead of N bespoke tools, the model writes **Python executed in a sandbox**; actions are typed Pydantic `CmdRunAction`/`IPythonRunCellAction`/`FileEditAction`, results are `Observation`s ([OpenHands paper, arXiv:2407.16741](https://arxiv.org/html/2407.16741v3)). This collapses the tool surface to "run code" and leans on the sandbox for safety — powerful, but it makes *capability scoping* coarse (anything Python can do) and *auditability* harder (effects hide inside an opaque script). **HIDE keeps explicit typed tools as the default (fine-grained capability + clean events) but offers a CodeAct-style `code.exec` tool as one catalog entry (§4.6), sandbox- and capability-gated** — the best of both.

### 3.2 MCP — the interop substrate (and its sharp edges)

The Model Context Protocol is now the lingua franca for tool/context interop, and HIDE must speak it natively both directions.

- **Current revision is `2025-11-25`** (not the brief's assumed `2025-06-18`; a `2026-07-28` RC is in flight) ([MCP versioning](https://modelcontextprotocol.io/specification/versioning)). Two standard transports: **stdio** (newline-delimited JSON-RPC over a subprocess's stdin/stdout; stderr is logging) and **Streamable HTTP** (single endpoint, POST per message, optional SSE stream for server→client) — Streamable HTTP **replaced** the old two-endpoint "HTTP+SSE" (2024-11-05) as of 2025-03-26 ([MCP transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)). Session header is **`MCP-Session-Id`** (all-caps in 2025-11-25); a **`MCP-Protocol-Version`** header is required on post-init HTTP requests since 2025-06-18. JSON-RPC **batching was removed** in 2025-06-18.
- **Server primitives:** `tools/list`, `tools/call`, `resources/list`/`resources/read`/`resources/templates/list`/`resources/subscribe`, `prompts/list`/`prompts/get`, plus `notifications/*/list_changed`. **Client primitives** the host exposes back: `sampling/createMessage` (server asks the host to run an LLM turn), `roots/list` (FS boundaries), and **`elicitation/create`** (server requests structured user input mid-session — added 2025-06-18) ([MCP server/client docs](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)).
- **Tool schema:** `{name, title?, description?, inputSchema (JSON Schema, type:object), outputSchema? (2025-06-18), annotations?}`. Annotations are **hints with telling defaults**: `readOnlyHint=false`, `destructiveHint=true`, `idempotentHint=false`, `openWorldHint=true` — and the spec is explicit that **clients MUST treat annotations as untrusted unless from a trusted server** ([MCP tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)).
- **`tools/call` result:** `CallToolResult{ content: ContentBlock[], structuredContent?, isError? }` where blocks are `text|image|audio|resource|resource_link`. **Critical:** tool *execution* errors are returned as a **successful result with `isError:true`** (so the model can see and self-correct), while *protocol* errors (unknown tool, bad args) are JSON-RPC errors.
- **Architecture is host → client(s) → server, 1:1 client-per-server**; one process is officially only one role, but a **dual-role gateway** (client to upstream, server to downstream) is established community practice though not a first-class spec concept ([MCP architecture](https://modelcontextprotocol.io/specification/2025-06-18/architecture)).
- **Security is the soft underbelly.** Documented attacks HIDE must defend at the tool seam: **tool poisoning** (malicious instructions hidden in tool *descriptions*, visible to the model but not the user) and the **"rug pull"** (a server mutates an already-approved tool description) ([Invariant Labs](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)); **tool shadowing** (one server's description hijacks use of another's); the **lethal trifecta** (private data + untrusted content + exfil = exfiltration) ([Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)); and **confused-deputy** OAuth flows on HTTP transport, mitigated by per-client consent + RFC 8707 resource indicators ([MCP security best practices](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices)). The spec itself states **"MCP cannot enforce these at the protocol level"** — consent and isolation are a **host obligation**. HIDE is that host, so this chapter (with Ch.10) is where the defense lives.

### 3.3 Sandboxing & capability theory (what powers the safety surface)

- **macOS:** `sandbox-exec` / Seatbelt (SBPL Scheme profiles, `(deny default)` + `(allow file-read* (subpath …))`, `network*`, `process-exec`; `sandbox_init(3)`) — **deprecated but still functional** on macOS 15 (emits a stderr warning even in Apple's own repos), and Apple ships **no non-deprecated replacement** for headless process sandboxing ([Chromium Seatbelt design](https://github.com/chromium/chromium/blob/main/sandbox/mac/seatbelt_sandbox_design.md); [sandbox-exec man](https://manp.gs/mac/1/sandbox-exec)). **Anthropic's Claude Code uses exactly this** (Seatbelt on macOS, bubblewrap on Linux), open-sourced as `@anthropic-ai/sandbox-runtime` (`srt`): **reads allowed by default, writes denied by default, network all-denied via an HTTP/SOCKS5 proxy + domain allowlist** ([Anthropic sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing); [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)). This is the closest published prior art to HIDE's shell-tool sandbox and we adopt its shape.
- **Linux:** seccomp-bpf (`SECCOMP_RET_*`), **Landlock LSM** (unprivileged path-based FS+net restriction; v1@5.13 … v4 networking@6.7 … v5 IOCTL@6.10), namespaces (unprivileged user-ns since 3.8 bootstraps the rest), and **bubblewrap** (`bwrap`, used by Flatpak) ([Landlock](https://man7.org/linux/man-pages/man7/landlock.7.html); [bubblewrap](https://github.com/containers/bubblewrap)). **OpenAI Codex CLI** now defaults to **bubblewrap+seccomp on Linux** (Landlock is a legacy fallback) and **Seatbelt on macOS**, with `read-only`/`workspace-write`/`danger-full-access` modes and **network off by default** ([Codex sandbox](https://github.com/openai/codex/blob/main/codex-rs/linux-sandbox/README.md)).
- **Heavier isolation** (for the `code.exec`/computer-use tiers, *later*): **gVisor** (userspace Go kernel intercepting all syscalls; GPU only via a curated `nvproxy` ioctl allowlist), **Firecracker** microVMs (Rust/KVM, ~125 ms boot claimed, **no GPU passthrough**), and **Apple's Virtualization.framework** (`VZVirtualMachine`, Rosetta-for-Linux, and the open-source `apple/container` one-VM-per-container) ([gVisor](https://gvisor.dev/docs/architecture_guide/intro/); [Firecracker](https://firecracker-microvm.github.io/); [apple/container](https://github.com/apple/container)). Anthropic's own containment map is instructive: **Claude Code (local) → Seatbelt/bubblewrap; Cowork (local) → full VM** ([how Anthropic contains Claude](https://www.anthropic.com/engineering/how-we-contain-claude)).
- **Capability theory:** the object-capability model (a capability is *"a communicable, unforgeable token of authority"* combining **designation + authorization**), **POLA**, the **confused-deputy** problem (Hardy 1988 — fixed precisely by bundling designation with permission, i.e. a capability), and the modern restatement that **prompt injection is a confused-deputy attack** ([ocap](https://en.wikipedia.org/wiki/Capability-based_security); [Hardy, Confused Deputy](https://dl.acm.org/doi/10.1145/54289.871709); [CSA note](https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-agent-confused-deputy-prompt-injection/)). HIDE's `capability_grant_id`-per-call (TT3) is a direct application: the grant *is* the capability token.

### 3.4 Constrained / grammar-structured decoding (what makes TT2 cheap)

- **llama.cpp GBNF** masks logits at sample time — rejected tokens get logit `= -INFINITY` before sampling, with a rejection-sampling fallback ([llama-grammar.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp)). Per-token cost is real (must test vocab tokens each step); pathological `x?x?…` grammars are "extremely slow."
- **Outlines** ([Willard & Louf, arXiv:2307.09702](https://arxiv.org/abs/2307.09702)) is the key efficiency idea: naïve masking is **O(vocab) per token**; precomputing an **index σ: FSM-state → allowed-token-subset** *outside* the sampling loop makes per-token masking **O(1) on average**. "Coalescence" can even be *faster than unconstrained* by skipping forward passes where the grammar deterministically dictates the next characters (JSON `:`, `,`, `}`, fixed keys) — a **~5× speedup** ([dottxt coalescence](https://blog.dottxt.ai/coalescence.html)).
- **XGrammar** ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100)) splits the vocab into **context-independent** tokens (checkable from PDA position alone — precomputable, cached in an "adaptive token mask cache") vs **context-dependent** (<1%, ~1134 of 128k), using a byte-level pushdown automaton with a persistent stack; mask computation **overlaps with GPU**. It is **vLLM's structured-output backend since v0.6.5**.
- **llguidance** (Microsoft, Rust, Earley-over-regex-derivatives, ~50 µs/token on a 128k tokenizer) is what **OpenAI's GPT-5 custom-tool CFG path uses** ([llguidance](https://github.com/guidance-ai/llguidance)).

> **HIDE takeaway:** our `json_constrain.rs` is today a hand-rolled JSON *well-formedness* FSM (no schema). The proven path forward is **schema → grammar → precomputed state→token-mask index** (Outlines' insight) with a **context-independent mask cache** (XGrammar's insight), applied to the *exact* tool-argument schema. Because greedy paths in the runtime are bit-identical (Ch.01 §3) and we control the sampler, **HIDE can give *every* local model strict-tool-use behavior that today only OpenAI/Anthropic ship for their own hosted models** — and do it deterministically (§4.3, §5).

### 3.5 Self-authored tools (Voyager and its lineage)

**Voyager** ([Wang et al., arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) is the canonical mechanism: an LLM writes **executable JavaScript** (Mineflayer API), an **automatic curriculum** proposes tasks, an **iterative prompting loop** refines code using **environment feedback + self-verification**, and a **verified skill is stored as a program keyed by an embedding of its docstring** in a vector DB, then **retrieved by query-embedding (top-k) for reuse and composition** — yielding 3.3× more items discovered and 15.3× faster milestones than prior SOTA. Follow-ons generalize "agents writing tools": **LATM** (LLMs As Tool Makers — two-phase make-then-use Python utilities, [arXiv:2305.17126]), **CREATOR** (separates abstract tool *creation* from concrete *invocation*, with a rectify-on-error loop), and **ToolMaker** (turns a GitHub repo into an LLM tool autonomously, ~80% success, [ACL 2025](https://aclanthology.org/2025.acl-long.1266/)).

> **Competitor limits we beat:** every cited self-authoring system is a research harness with **no capability sandbox, no durable cross-session registry wired to a real IDE, no permission gate** on the generated code's *effects*, and no replayable audit trail. HIDE's skill library (§4.11) is Voyager's loop **plus** TT3 capability-scoping, **plus** the Ch.01 event log (every skill authorship + execution is a replayable event), **plus** a real test harness (the skill must pass generated tests before registration) — i.e. self-authoring that is *safe and persistent*, not a demo.

---

## 4. The Hawking design (concrete)

### 4.1 Module layout

The tool system is a kernel-internal crate cluster (Ch.01 §4.1 hosts it in `hide-kernel`). Zero tool names appear in the dispatcher; everything is registry-driven (TT6).

```
hide-kernel/
└── tools/
    ├── mod.rs                # public surface: ToolSystem facade
    ├── wire.rs               # ToolCall / ToolResult / ToolError canonical types (§4.2)  ← CONTRACT
    ├── spec.rs               # ToolSpec, ArgSchema (JSON-Schema subset), ResultSpec, annotations
    ├── registry/
    │   ├── mod.rs            # ToolRegistry: name → ToolHandle; namespacing, versioning (§4.4)
    │   ├── discovery.rs      # scan manifests (builtin + plugin + skill + MCP) → register
    │   ├── negotiation.rs    # capability negotiation vs ProviderCaps / host caps
    │   └── version.rs        # semver, wire_version compat, deprecation
    ├── dispatch/
    │   ├── mod.rs            # Dispatcher: call → policy-check → execute → record (§4.5)
    │   ├── policy.rs         # PermissionEngine (tool-side surface; defers to Ch.10) (§4.9)
    │   ├── result.rs         # typed results, bytes_ref spill, streaming, isError mapping
    │   ├── cache.rs          # memoization by (tool, args-hash, fs-epoch) for pure tools (§4.5)
    │   ├── dryrun.rs         # simulation mode — compute effect set without committing
    │   └── fanout.rs         # parallel multi-call primitive (TT9)
    ├── constrain/
    │   ├── mod.rs            # ToolGrammar: ToolSpec[] → constraint (§4.3)
    │   ├── schema_fsm.rs     # JSON-Schema → state→token-mask index (Outlines-style)
    │   └── mask_cache.rs     # context-independent mask cache (XGrammar-style)
    ├── builtin/             # the catalog (§4.6) — each is one `impl Tool`
    │   ├── fs.rs            # read, list, stat, watch, glob
    │   ├── edit.rs          # apply_patch, search_replace, ast_edit, multi_edit (§4.7)
    │   ├── shell.rs         # shell.run (non-interactive) + pty.* (interactive) (§4.8)
    │   ├── search.rs        # grep / symbol / semantic — wrappers over ch.05 §4.11
    │   ├── test.rs          # test.run, test.discover
    │   ├── build.rs         # build.run, compile.check
    │   ├── git.rs           # status/diff/commit/branch/worktree/blame/log
    │   ├── refactor.rs      # rename_symbol, move, extract — wrappers over ch.05
    │   ├── fmt.rs           # format, lint
    │   ├── web.rs           # web.fetch, web.search
    │   ├── browser.rs       # browser.* / computer.* (gated, §4.6)  [partly LATER]
    │   ├── pkg.rs           # package manager (cargo/npm/pip/…)
    │   ├── db.rs            # db.query, http.request
    │   └── code_exec.rs     # CodeAct-style sandboxed exec (gated)
    ├── mcp/
    │   ├── client.rs        # HIDE as MCP host/client (stdio + Streamable HTTP) (§4.10)
    │   ├── server.rs        # HIDE exposed as an MCP server
    │   └── bridge.rs        # map MCP Tool ↔ ToolSpec, CallToolResult ↔ ToolResult
    ├── skills/
    │   ├── mod.rs           # SkillLibrary: author → test → register → persist (§4.11)
    │   ├── author.rs        # generate-code loop (Voyager-style)
    │   ├── verify.rs        # run generated tests in sandbox before registration
    │   └── store.rs         # durable skill DB + embedding index (retrieval by docstring)
    └── observe.rs           # every call → tool.* events (§4.12)
```

**Trait boundary.** Everything the dispatcher touches is one object-safe trait. A built-in is a Rust impl; a WASM plugin's tools are an adapter over the WIT export; an MCP server's tools are an adapter over `tools/call`; a skill is an adapter over its stored body. The dispatcher cannot tell them apart.

```rust
/// The single interface the dispatcher calls. (Mirror WIT world for WASM plugins.)
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;                       // name, schema, annotations, caps required
    /// Execute against an already-checked grant. `ctx` carries the scoped capability
    /// handles (fs root, allowed cmds, net policy), the cancellation token, the
    /// deadline, the output cap, and a `progress` sink for streaming.
    async fn call(&self, args: serde_json::Value, ctx: &ToolCtx) -> ToolOutcome;
    /// Optional: compute the *effect set* without committing (dry-run / simulation).
    async fn simulate(&self, _args: &serde_json::Value, _ctx: &ToolCtx) -> Option<EffectSet> { None }
    /// Optional: declare purity for memoization (default: impure).
    fn purity(&self) -> Purity { Purity::Impure }
}
```

### 4.2 The tool wire-format & schema — CROSS-CUTTING CONTRACT

> **This is the contract Ch.02 (emits calls), Ch.04 (consumes results into context), Ch.06 (constrains emission), and Ch.10 (gates) bind to.** One canonical shape; all other forms project from it (TT1). Additive-by-default; unknown fields survive (TT11).

#### 4.2.1 The tool specification (`ToolSpec`)

A tool is *declared* by a `ToolSpec` — the union of "what the manifest says" and "what the model is shown." It is **deliberately MCP-shaped** so the MCP bridge is an identity map, with HIDE extensions namespaced under `x_hide`.

```jsonc
// ToolSpec — the registered declaration of a tool. JSON form (also a Rust struct).
{
  "name": "fs.read",                       // unique, namespaced (family.verb); registry key
  "title": "Read file",                    // human label (UI); falls back to name
  "version": "1.2.0",                       // semver of THIS tool's behavior+schema (§4.4)
  "wire_version": 1,                         // tool-wire-format version this tool speaks (TT11)
  "description": "Read a UTF-8 or binary file from the workspace…",  // shown to the model
  "input_schema": {                          // JSON Schema (subset, §4.3.2) — type:object
    "type": "object",
    "properties": {
      "path":   { "type": "string", "description": "workspace-relative or absolute path" },
      "range":  { "type": "object", "properties": {
                    "start_line": {"type":"integer","minimum":1},
                    "end_line":   {"type":"integer","minimum":1} },
                  "required": [] },
      "encoding": { "type": "string", "enum": ["utf8","base64","auto"], "default": "auto" }
    },
    "required": ["path"],
    "additionalProperties": false            // REQUIRED for strict constrained decode (§4.3)
  },
  "output_schema": {                         // optional; enables typed structuredContent
    "type": "object",
    "properties": { "content": {"type":"string"}, "truncated": {"type":"boolean"},
                    "bytes": {"type":"integer"}, "blob_ref": {"type":"string"} },
    "required": ["content"]
  },
  "annotations": {                           // MCP-compatible hints (UNTRUSTED if from a plugin/MCP)
    "read_only":   true,                     // does not modify the world (MCP readOnlyHint)
    "destructive": false,                    // may irreversibly destroy (MCP destructiveHint)
    "idempotent":  true,                     // repeat = no extra effect (MCP idempotentHint)
    "open_world":  false                     // touches external/unbounded entities (MCP openWorldHint)
  },
  "x_hide": {                                // HIDE-only extensions (ignored by plain MCP clients)
    "capabilities_required": [               // declarative scopes (Ch.01 manifest schema) (TT3)
      { "kind": "fs.read", "scope": "$WORKSPACE/**" }
    ],
    "purity": "pure_fs",                     // pure | pure_fs | impure  → memoization (§4.5)
    "default_policy": "auto",                // ask | auto | deny  (overridable, §4.9)
    "output_cap_bytes": 1048576,             // hard cap → spill to blob_ref beyond (§4.5)
    "timeout_ms": 15000,                      // dispatcher deadline → StopReason::Timeout
    "streams": false,                         // emits tool.progress?  (§4.5)
    "cost_class": "trivial",                  // trivial|cheap|heavy → scheduler hint (TT9)
    "provenance": "builtin"                   // builtin | plugin:<id> | mcp:<server> | skill:<id>
  }
}
```

**Why MCP-shaped.** `name/title/description/inputSchema/outputSchema/annotations` are *exactly* MCP's `Tool` fields, so registering an external MCP tool is a copy, and exposing a HIDE tool as MCP is a projection (§4.10). HIDE-specific safety/perf metadata lives under `x_hide` so it round-trips cleanly through MCP without polluting the standard.

#### 4.2.2 The tool **call** (`ToolCall`) — the canonical effect request

```jsonc
// ToolCall — emitted by the agent (Ch.02), enforced at decode (§4.3), logged as tool.call payload.
{
  "call_id": "01JABZ…",            // ULID; unique within the run; correlates result + events
  "tool": "fs.read",               // ToolSpec.name (registry-resolved)
  "args": { "path": "src/main.rs", "range": { "start_line": 1, "end_line": 80 } },
  "capability_grant_id": "grant_7f…", // REQUIRED — references Ch.01's grant ledger (TT3, §4.9)
  "wire_version": 1,
  "x": {                            // optional execution directives
    "dry_run": false,               // simulate only (§4.5)
    "idempotency_key": "…",         // dedupe identical retried calls
    "timeout_ms_override": null     // ≤ spec cap; cannot exceed it
  }
}
```

This maps **1:1** to Ch.01's `tool.call` event payload `{call_id, tool, args, capability_grant_id}` (Ch.01 §4.6) — the event *is* the call, recorded.

#### 4.2.3 The tool **result** (`ToolResult`) — the canonical recorded outcome

```jsonc
// ToolResult — produced by the dispatcher, logged as tool.result payload (TT4).
{
  "call_id": "01JABZ…",            // echoes the call
  "ok": true,                       // false ⇒ see `error`; maps to MCP isError (§4.10)
  "output": {                       // typed body validated against ToolSpec.output_schema
    "content": "fn main() { … }",   // small bodies inline …
    "truncated": false,
    "bytes": 1843
  },
  "bytes_ref": null,                // … large bodies spill here: "blake3:ab12…" into blob CAS (TT5)
  "content_blocks": [               // OPTIONAL MCP-style multimodal blocks (text/image/audio/resource)
    { "type": "text", "text": "fn main() { … }" }
  ],
  "exit_code": null,                // for process-shaped tools (shell/test/build)
  "stats": { "duration_ms": 4, "cached": false, "from_dry_run": false },
  "provenance": "tool-output",      // TT8: this body is UNTRUSTED data, not instructions
  "wire_version": 1
}
```

```jsonc
// ToolError — the structured failure (when ok=false). Designed to be *self-correcting*:
// the agent loop feeds it back so the model can fix the call (MCP isError philosophy).
{
  "call_id": "01JABZ…",
  "ok": false,
  "error": {
    "code": "ARG_INVALID",          // taxonomy below
    "message": "range.end_line (200) exceeds file length (80)",
    "retriable": true,              // can the same model fix-and-retry?  (Ch.02 uses this)
    "fix_hint": "set range.end_line ≤ 80 or omit range",
    "schema_path": "/range/end_line" // JSON-pointer into args for precise UI/model targeting
  },
  "provenance": "tool-output"
}
```

**Error taxonomy** (stable codes — extends Ch.01 §4.12 error taxonomy; tools map their failures into these):

| `code` | Meaning | `retriable` | Typical handling |
|---|---|---|---|
| `ARG_INVALID` | args failed schema or semantic validation | yes | model fixes args |
| `NOT_FOUND` | target (file/symbol/server) absent | yes | model picks another target |
| `CAP_DENIED` | grant doesn't authorize this scope (deny-beats-allow) | no | escalate to user (§4.9) |
| `PERMISSION_ASK_DENIED` | user declined the prompt | no | agent must change plan |
| `TIMEOUT` | exceeded deadline (spec or watchdog) | sometimes | retry smaller / abort |
| `OUTPUT_CAPPED` | result exceeded cap; spilled to `bytes_ref` | n/a (success-ish) | model reads via `bytes_ref` slice |
| `TOOL_FAULT` | tool's own internal error | maybe | report; possibly fall back |
| `EXEC_NONZERO` | process exited non-zero (NOT an error per se) | n/a | model reads stderr (this is *data*) |
| `CONFLICT` | optimistic precondition failed (file changed under edit) | yes | re-read + re-plan (§4.7) |
| `INJECTION_BLOCKED` | trifecta/poisoning gate tripped (§4.9, TT8) | no | surfaced to user with context |

> **Design note — `EXEC_NONZERO` is not `ok:false`.** A test failing or a compiler erroring is **expected information**, not a tool failure: the tool *succeeded* in running the process; `ok:true`, `exit_code:1`, stderr in `output`. Only a failure to *run* the process at all is `ok:false`. This mirrors MCP's `isError` discipline and is load-bearing for the agent loop (Ch.02): the model must see compiler errors as data to act on, not as a broken tool. (Aider, OpenHands, and MCP all converge on this.)

#### 4.2.4 The three projections of one shape (TT1)

| Projection | What | When | Mapping |
|---|---|---|---|
| **OpenAI tool-call** | `{type:"function", function:{name, arguments:"<json-string>"}}` in `tool_calls[]`; result as a `role:"tool"` message | Interop with OpenAI-shaped clients/models; the agent loop's default model contract | `ToolSpec.input_schema` → `function.parameters`; `ToolCall.args` → `function.arguments` (stringified); `ToolResult.output` → tool message content |
| **Constraint grammar** | a token-level mask over the model's vocab forcing valid `ToolCall` JSON | At decode time, for *local* models (§4.3) | `ToolSpec.input_schema` (+ allowed `name` set) → grammar → mask index |
| **MCP `tools/call`** | `{method:"tools/call", params:{name, arguments}}` → `CallToolResult{content, structuredContent, isError}` | HIDE-as-server, or calling an external MCP server | identity on names/schema; `ok` ↔ `!isError`; `output` ↔ `structuredContent`; `bytes_ref`/`content_blocks` ↔ `content[]` |
| **Event payload** | Ch.01 `tool.call` / `tool.progress` / `tool.result` | Always (every call is recorded) | `ToolCall` ≡ `tool.call.payload`; `ToolResult` ≡ `tool.result.payload`; `bytes_ref` ≡ Ch.01 `bytes_ref` |

The dispatcher serializes *into* whichever projection a consumer needs from the **same in-memory `ToolCall`/`ToolResult`**, so the four can never drift.

### 4.3 Schema-enforced model-side emission (constrained decode)

This is **TT2** — the single biggest reason a small local model behaves like a frontier tool-caller. We make the model's tool call **valid by construction** by masking logits at decode time against the *exact* argument schema. Coordinates conceptually with Ch.06 (sampler/grammar kernel) and Ch.02 (the agent loop that requests a constrained turn).

#### 4.3.1 Where it plugs into the runtime (ground truth)

The serve layer already has the seam: `GenerateRequest.json_mode: bool` routes logits through `JsonConstraint::mask_logits` before each sample, advancing a JSON-state FSM (`crates/hawking-core/src/json_constrain.rs`). **Today that FSM enforces only JSON *well-formedness*** (balanced braces, strings, numbers) — it does **not** know a schema, and `chat_completions` derives `json_mode` solely from `response_format == "json_object"` (the OpenAI `tools`/`tool_choice` fields are **not parsed at serve** — greenfield). The extension is therefore well-bounded:

```
TODAY:   json_mode: bool ──▶ JsonConstraint (well-formed JSON only)
HIDE:    constraint: Option<Constraint> ──▶ {
             ToolCall(schema_set),   // force one of the allowed tool calls, schema-valid
             JsonSchema(schema),     // force a specific JSON object
             Grammar(gbnf),          // arbitrary grammar (e.g. a DSL the model must emit)
             WellFormed,             // the current behavior (back-compat)
         }
```

**Request-surface extension** (designed; the *boolean* path stays for back-compat — TT11):

```jsonc
// Extension to the chat/generate request body. `json_mode:true` still means WellFormed.
"hide_constraint": {
  "kind": "tool_call",                 // tool_call | json_schema | grammar | well_formed
  "tools": ["fs.read","edit.apply_patch","shell.run"],  // allowed tool names this turn
  "tool_choice": "auto"                // auto (model may also emit prose) | required | "<name>"
}
```

- `tool_choice:"required"` forces the *next* emitted structure to be a valid `ToolCall` (no prose escape) — the agent loop uses this when it has decided a tool *must* be called.
- `tool_choice:"auto"` allows the model to either emit prose **or** open a tool-call structure; once it commits to the opening token of a call, the mask enforces validity through to a closed, schema-valid call (a "soft gate" — see §6 for the edge case where the model never commits).
- `"<name>"` forces a specific tool (used by deterministic sub-agents, e.g. "now call `test.run`").

#### 4.3.2 Schema → grammar → mask index (the mechanism, PROVEN-pattern)

Per §3.4, the efficient path is **not** to recompute an O(vocab) mask each token. We compile once and cache:

1. **Compile** the union of allowed `input_schema`s (restricted to the strict subset below) into a single grammar whose top level is `{"call_id":…,"tool": <oneOf allowed names>, "args": <schema(name)>, …}`. Tool-name selection is an `enum` over the allowed set; once the model commits a name, the `args` sub-grammar narrows to *that tool's* schema (a pushdown — XGrammar-style PDA, §3.4).
2. **Index** the grammar as `state → allowed-token-subset` over the model's vocab (Outlines' σ index, §3.4) — built **outside** the sampling loop, so its cost is amortized across the whole call and re-used across calls (it's a pure function of `(schema-set, tokenizer)`).
3. **Cache** the **context-independent** portion of each state's mask (XGrammar: ~99% of tokens' validity depends only on PDA position) in `mask_cache.rs`, keyed by `(schema_hash, model_id)`. Persist it in the redb cache tier (Ch.01 §4.7) so the *second* time the agent constrains to `fs.read|edit.*|shell.*` it's a cache hit, not a recompile.
4. **Coalesce** (Outlines): where the grammar deterministically dictates the next bytes — the literal keys `"call_id":`, `"tool":`, `"args":`, the structural `{ } [ ] : ,` — **skip the forward pass entirely** and emit those tokens directly. For a tool call, the *fixed scaffolding is most of the tokens*, so coalescence makes constrained tool emission **competitive with or faster than** unconstrained prose (a measured ~5× on JSON-shaped output, §3.4).

**The strict JSON-Schema subset HIDE constrains** (matching what OpenAI/Anthropic strict mode enforce, §3.1, so tool authors hit no surprises):

- `type: object` at top level; `additionalProperties: false` **required**; every property either `required` or unioned-with-null.
- Scalars: `string` (with `enum`, `const`, bounded `pattern`), `integer`/`number` (with `minimum`/`maximum`/`multipleOf`), `boolean`, `null`.
- `array` (with `items`, `minItems`/`maxItems`), nested `object` (recursively in-subset), and `enum`/`oneOf` for tagged unions.
- **Excluded** (rejected at registration with a clear error): unbounded `$ref` recursion, arbitrary `pattern` over the whole value, `minLength` on deeply-nested strings beyond a depth cap — the same exclusions OpenAI/Anthropic document, because they break the FSM/PDA compilation (§3.4). Tools needing richer validation do a **two-stage check**: constrain to the in-subset shape, then run a full JSON-Schema validation in `simulate()`/at dispatch and return `ARG_INVALID` with a `fix_hint` (the model self-corrects — TT2's fallback).

#### 4.3.3 Fine-tune-at-Condense: teach the protocol into the weights (UNFAIR ADVANTAGE)

Constrained decode guarantees *syntactic* validity; it can't make the model *choose the right tool with the right args*. Because HIDE owns the whole stack including *Hawking Condense* (the quantizer/fine-tuner that produces the `.tq` the runtime serves), we close the semantic gap the way no cloud agent can: **bake the exact HIDE tool protocol into the model during Condense.**

- A Condense fine-tune pass on transcripts of *correct* HIDE tool use (the `tool.call`/`tool.result` event stream is literally a labeled dataset — TT9 means we have it for free) teaches the model the catalog, the arg conventions, the `EXEC_NONZERO`-is-data discipline, and when to call which tool.
- The model then emits good calls *unconstrained*, and the constraint becomes a **cheap guardrail** rather than a crutch — best of both: semantic competence from the weights, syntactic guarantee from the mask.
- **[LATER]** This couples to Ch.06's model layer and the Condense product; the *shell* ships with constrained-decode-only (works on any stock local model today) and gains the fine-tuned protocol when a HIDE-Condensed model is available. Tagged *later / not shell-gating*.

### 4.4 The tool registry (manifest, versioning, discovery, namespacing)

The registry is the runtime catalog mapping `name → ToolHandle`. It knows **zero** tool names at compile time (TT6); everything is discovered.

#### 4.4.1 Sources of tools (all unified through `ToolSpec`)

| Source | Declared by | Trust | Discovery |
|---|---|---|---|
| **Built-in** | Rust `impl Tool` + a `ToolSpec` in `builtin/` | first-party (in-process) | registered at boot from a static table |
| **WASM plugin** | `manifest.toml` (Ch.01 §7.2) `[[tool]]` entries + WIT exports | sandboxed (fuel/epoch/mem) | scan `plugins/*/manifest.toml` |
| **MCP server** | the server's `tools/list` response | untrusted (external) | connect (stdio/HTTP), list, bridge (§4.10) |
| **Skill** | a row in the skill DB (generated body + generated `ToolSpec`) | sandboxed + capability-gated | load from `skills/store` on boot (§4.11) |

A plugin manifest's tool block (extends Ch.01's manifest schema — the `tool` extension kind):

```toml
# in plugins/<id>/manifest.toml  (Ch.01 §7.2 manifest, the `tool` kind)
[[tool]]
name        = "acme.deploy"
title       = "Deploy to Acme"
version     = "0.3.1"
wire_version = 1
description = "Deploy the current build to the Acme staging environment."
input_schema = "schemas/deploy.json"     # path to the JSON-Schema file
output_schema = "schemas/deploy.out.json"
annotations = { read_only = false, destructive = true, idempotent = false, open_world = true }

  [[tool.capabilities_required]]          # declarative, scoped (Ch.01) — deny-beats-allow (TT3)
  kind = "net.connect"
  scope = "https://staging.acme.internal"

  [[tool.capabilities_required]]
  kind = "shell.exec"
  scope = "acme-deploy --env staging"     # exact-arg-scoped command capability (§4.9)

[tool.policy]
default = "ask"                            # destructive ⇒ default ask
```

#### 4.4.2 Namespacing

- **`family.verb`** dotted names (`fs.read`, `edit.apply_patch`, `git.commit`). Core families are reserved.
- **Plugin/MCP/skill tools are prefixed** to prevent collision and shadowing (§3.2's tool-shadowing attack): a plugin tool is `plugin:<id>/<name>` internally and may *request* an unprefixed display alias only if no collision exists; MCP tools are `mcp:<server>/<name>`; skills are `skill:<name>`. **The model is shown the prefixed name when ambiguity or low trust exists**, so it cannot be tricked into calling `mcp:evil/fs.read` thinking it's the built-in `fs.read`.
- **Collision rule:** built-in > first-party plugin > third-party plugin > MCP > skill, but a lower tier can never *override* a higher one's unprefixed name — it only gets its prefixed form. Deny-beats-allow extends here: a denied name is denied for all tiers.

#### 4.4.3 Versioning & wire compatibility (TT11)

- Each `ToolSpec` has a **semver `version`** (behavior+schema) and a **`wire_version`** (the tool-wire-format it speaks). The registry refuses to load a tool whose `wire_version` the kernel doesn't support, with a clear migration message.
- **Replay across tool upgrades:** because effects are recorded outcomes (TT4), a session recorded against `fs.read@1.1` replays fine even if `fs.read@1.3` is installed — replay never calls the tool. Only **live resume** uses the current version; the resume point records `tool_version` in `tool.call` so the timeline shows which version actually ran.
- **Deprecation:** a `ToolSpec` may carry `x_hide.deprecated_by = "fs.read2"`; the registry keeps the old one callable (warns) until a major kernel bump.

#### 4.4.4 Capability negotiation

At registration the negotiator checks each tool's `capabilities_required` against (a) what the host can grant on this OS (e.g. `net.connect` may be policy-locked off — Ch.01 §4.10 enterprise layer) and (b) the `ProviderCaps` where relevant (a tool that needs `raw_logits` is unavailable on a cloud provider). A tool whose required capability *can never* be granted in this workspace is registered as **`unavailable`** with a reason, surfaced in the UI — it's visible but not callable, so the model is never offered a tool it can't actually use.

### 4.5 Dispatch & result types (typed, large-output, streaming, caching, dry-run)

The dispatcher is the **single choke point** every tool call passes through. It is where TT3/TT4/TT5/TT7/TT8 are enforced. Pseudocode of the hot path:

```python
def dispatch(call: ToolCall, run_ctx) -> ToolResult:
    # 1. RESOLVE — registry lookup; unknown tool ⇒ NOT_FOUND (JSON-RPC-style protocol error)
    tool = registry.resolve(call.tool)            # honors namespacing/shadowing (§4.4.2)
    if tool is None: return protocol_error("NOT_FOUND", call)

    # 2. VALIDATE — args against ToolSpec.input_schema (strict subset). Constrained decode
    #    should make this pass, but external/forced calls may not ⇒ ARG_INVALID + fix_hint.
    err = validate(call.args, tool.spec.input_schema)
    if err: return tool_error("ARG_INVALID", err, retriable=True, call)

    # 3. POLICY — the capability + permission gate (§4.9). Deny-beats-allow. May ASK the user.
    decision = permission_engine.check(call, tool.spec, run_ctx)   # tool-side; defers to Ch.10
    if decision.deny:  return tool_error(decision.code, decision.msg, retriable=False, call)
    grant = decision.grant                          # the scoped capability handles (TT3)

    # 4. CACHE — pure tools memoized by (tool, canonical-args-hash, fs_epoch) (TT7)
    if tool.purity != Impure:
        hit = cache.get(tool.name, args_hash(call.args), current_fs_epoch())
        if hit: return hit.with_stats(cached=True)

    # 5. DRY-RUN — if requested, compute the effect set without committing (TT9)
    if call.x.dry_run:
        eff = tool.simulate(call.args, ctx(grant)) or EffectSet.unknown()
        return ToolResult.dry(call.call_id, eff)

    # 6. EXECUTE — bounded: deadline, output cap, cancellation, progress sink (TT5)
    emit(Event.tool_call(call))                     # record the Action BEFORE effect (TT4)
    ctx = ToolCtx(grant=grant,
                  deadline=now() + (call.x.timeout_override or tool.spec.timeout_ms),
                  cap_bytes=tool.spec.output_cap_bytes,
                  cancel=run_ctx.cancel_token,       # Arc<AtomicBool> — maps to runtime `abort`
                  progress=ProgressSink(call.call_id))  # streams tool.progress events
    outcome = await run_bounded(tool.call(call.args, ctx), ctx)   # watchdog ⇒ TIMEOUT

    # 7. SPILL — large output → blob CAS; result keeps a bytes_ref (TT5, Ch.01 §4.7)
    result = materialize(outcome, cap=tool.spec.output_cap_bytes)  # inline if small, else bytes_ref

    # 8. REDACT — scrub secrets (API keys in shell output) before durability (Ch.01 redactions)
    result = redactor.scrub(result)

    # 9. RECORD — the Observation, caused by the Action (TT4, OpenHands cause-link)
    emit(Event.tool_result(result, cause=call.call_id))
    if tool.purity != Impure and result.ok: cache.put(...)
    return result
```

**Key dispatch behaviors:**

- **Typed results.** `output` validates against `output_schema`; the agent loop and the context compiler get a *typed* body, not a blob to re-parse. Multimodal `content_blocks` (MCP-style `text|image|audio|resource`) carry screenshots, rendered diffs, etc.
- **Large-output handling (TT5).** Beyond `output_cap_bytes`, the body is written to the blob CAS and the result carries `bytes_ref` + a `head` preview (first N KB) + `bytes` total. The model reads more via a follow-up `fs.read`/slice on the ref. **This is the defense against a `cat huge.log` blowing the context window** — the *log* keeps the whole output (durable), the *window* gets a preview + handle (Ch.04 decides how much to pull). Caps default per family (trivial reads 1 MB; shell 256 KB head; build 1 MB) and are dials (§9).
- **Streaming tool output.** Long-running tools (a 5-minute test suite, a build, a dev server) emit `tool.progress` events (`{call_id, message, fraction?}`) that the UI shows live and the agent *may* observe mid-flight (e.g. abort on first failing test). The final `tool.result` still carries the complete recorded outcome.
- **Caching / memoization (TT7).** Pure (`pure`) and pure-over-filesystem (`pure_fs`) tools memoize keyed by `(tool, canonical-args-hash, fs_epoch)`. `fs_epoch` is bumped by the file-watcher on any workspace write (and by `diff.applied`), so a cached `search.grep` or `fs.stat` is invalidated the instant the tree changes — correctness without staleness. Impure tools (shell, net, edit) **never** cache. This makes repeated `find_definition`/`grep` during a single reasoning burst free.
- **Dry-run / simulation (TT9).** `dry_run:true` (or a global "plan mode") returns the **effect set** — which files would change, which commands would run, which hosts would be hit — *without committing*. `edit.*` simulate by computing the post-image diff; `shell.run` simulates by parsing the command and reporting the *declared* capability footprint (it does not execute). This powers a "show me what this plan will do before I approve" UX that no string-matching agent can offer, and feeds the permission UI (§4.9).
- **Parallel fan-out (TT9).** The dispatcher exposes a `fanout([ToolCall])` primitive: independent calls (the agent asserts independence, or the dispatcher infers it from disjoint effect sets via `simulate`) run concurrently, bounded by `cost_class` and a global semaphore (heavy tools serialize; trivial ones parallelize freely). 8-file rename = 8 `edit.*` calls in parallel, then one atomic commit.
- **Cancellation & timeouts.** `ctx.cancel` is the `Arc<AtomicBool>` that maps straight onto the runtime's `GenerateRequest.abort` and onto process kill for shell tools; `deadline` maps onto a watchdog (akin to the runtime's `max_stall_ms`). A cancelled/timed-out call records a `tool.result{ok:false, code:TIMEOUT}` and (for `Action`s with side effects) triggers the dangling-Action recovery (Ch.01 §4.12, §6 here).

### 4.6 The built-in tool catalog

The complete first-party catalog with argument schemas. Annotations columns: **RO** read-only, **D** destructive, **I** idempotent, **OW** open-world (MCP-aligned). **Cap** = the capability kind(s) it requires (TT3). **Pol** = default policy (a=auto, ?=ask, deny). Tools tagged **[ch.05]** are *thin wrappers over Ch.05's §4.11 query API* — they **must not re-parse or walk the FS**; they query the Living Index (Ch.05 §4.11 mapping). Tools tagged **[LATER]** are designed but gated.

#### 4.6.1 Filesystem

| Tool | Args (schema sketch) | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `fs.read` | `{path, range?{start_line,end_line}, encoding?}` | ✓/✗/✓/✗ | `fs.read:scope` | a | spills to `bytes_ref` over cap; `auto` encoding sniffs binary |
| `fs.list` | `{path, depth?=1, include_hidden?=false, glob?}` | ✓/✗/✓/✗ | `fs.read` | a | dir listing; respects `.gitignore` by default |
| `fs.stat` | `{path}` | ✓/✗/✓/✗ | `fs.read` | a | size, mtime, mode, is_dir, blob_hash |
| `fs.glob` | `{pattern, root?}` | ✓/✗/✓/✗ | `fs.read` | a | fast glob via the index when warm, else walk |
| `fs.watch` | `{path, events?=[create,modify,delete], session_scoped?=true}` | ✓/✗/✗/✗ | `fs.read` | a | registers a watcher → emits `file.changed_external`; auto-unwatch at run end |

#### 4.6.2 Edit (the hard part — full treatment in §4.7)

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `edit.apply_patch` | `{path, patch (unified diff), base_hash?}` | ✗/✗/✗/✗ | `fs.write:scope` | ? | unified-diff apply with fuzz; `base_hash` = optimistic concurrency (§4.7) |
| `edit.search_replace` | `{path, edits:[{search, replace, occurrence?}], base_hash?}` | ✗/✗/✗/✗ | `fs.write` | ? | exact-match blocks (Aider/Cline style); returns `CONFLICT` on no-match |
| `edit.ast` | `{path, op:{kind, target_symbol, …}}` | ✗/✗/✗/✗ | `fs.write` + `index.read` | ? | **[ch.05]** structural edit via tree-sitter/CPG; survives formatting |
| `edit.multi` | `{edits:[{path, …}], atomic?=true}` | ✗/✗/✗/✗ | `fs.write:scope*` | ? | **multi-file atomic**: all-or-nothing across files (§4.7) |
| `edit.write_file` | `{path, content, create_only?=false}` | ✗/(✓ if overwrite)/✗/✗ | `fs.write` | ? | full-file write; `destructive` when overwriting |

#### 4.6.3 Shell / PTY (full treatment in §4.8)

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `shell.run` | `{cmd (argv[] preferred), cwd?, env?, stdin?, timeout_ms?}` | ✗/✓/✗/✓ | `shell.exec:scope` | ? | **non-interactive**, sandboxed (§4.8); argv form avoids shell-injection; head-capped output |
| `pty.open` | `{cmd, cwd?, env?, cols?, rows?}` → `{pty_id}` | ✗/✓/✗/✓ | `shell.exec` | ? | **interactive** PTY session (REPL, ssh, top); streams via `tool.progress` |
| `pty.write` | `{pty_id, data}` | ✗/✓/✗/✓ | (session grant) | a* | a once the PTY is approved; data is keystrokes |
| `pty.read` | `{pty_id, timeout_ms?}` | ✓/✗/✗/✗ | (session grant) | a | reads buffered output |
| `pty.close` | `{pty_id}` | ✗/✗/✓/✗ | (session grant) | a | SIGTERM→SIGKILL ladder |
| `code.exec` | `{lang:["python","bash","node"], source, timeout_ms?}` | ✗/✓/✗/✓ | `shell.exec` (heaviest sandbox) | ? | **CodeAct tier** — sandboxed interpreter; coarse cap ⇒ strongest isolation (§3.1) |

#### 4.6.4 Code search (text + symbol + semantic) — **[ch.05] wrappers**

These are the names Ch.05 §4.11 explicitly expects; they **query the Living Index, never re-parse**.

| Tool | Args | RO/D/I/OW | Cap | Pol | Maps to (Ch.05 §4.11) |
|---|---|---|---|---|---|
| `search.grep` | `{pattern (regex), path?, glob?, max?}` | ✓/✗/✓/✗ | `index.read` | a | `grep_symbol`/`search` (lexical) |
| `search.symbol` | `{name, kind?}` | ✓/✗/✓/✗ | `index.read` | a | symbol lookup |
| `search.semantic` | `{query, k?=20, scope?}` | ✓/✗/✓/✗ | `index.read` | a | semantic/embedding retrieval |
| `find_definition` | `{symbol, from?}` | ✓/✗/✓/✗ | `index.read` | a | `find_definition` |
| `find_references` | `{symbol}` | ✓/✗/✓/✗ | `index.read` | a | `find_references` |
| `find_callers` | `{symbol, transitive?=false}` | ✓/✗/✓/✗ | `index.read` | a | `find_callers`/`transitive_callers` |
| `find_implementations` | `{trait_or_iface}` | ✓/✗/✓/✗ | `index.read` | a | `find_implementations` |
| `path_between` | `{from_symbol, to_symbol}` | ✓/✗/✓/✗ | `index.read` | a | `path_between` |
| `tests_covering` | `{symbol}` | ✓/✗/✓/✗ | `index.read` | a | `tests_covering` |
| `changed_since` | `{ref_or_time}` | ✓/✗/✓/✗ | `index.read`+`git.read` | a | `changed_since` |
| `dataflow_paths` | `{from, to?, taint?}` | ✓/✗/✓/✗ | `index.read` | a | `dataflow_paths`/`taint_check` |

#### 4.6.5 Tests, build, compile

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `test.discover` | `{path?, framework?=auto}` | ✓/✗/✓/✗ | `index.read` | a | enumerate suites/cases (no run) |
| `test.run` | `{selector?, framework?, timeout_ms?}` | ✗/✗/✗/✗ | `shell.exec:test*` | a | runs tests sandboxed; `EXEC_NONZERO` = failing tests = **data** (TT…/§4.2.3); parses JUnit/TAP → `test.status` |
| `build.run` | `{target?, profile?}` | ✗/✗/✗/✗ | `shell.exec:build*` | a | parses diagnostics → `build.status`; errors are data |
| `compile.check` | `{path?|crate?}` | ✓-ish/✗/✗/✗ | `shell.exec:check*` | a | type-check only (`cargo check`, `tsc --noEmit`) — cheap correctness signal |

> Default policy for test/build/check is **auto** even though they run processes: they're sandboxed, scoped to declared build/test commands, and the agent loop relies on running them constantly (a "verify after every edit" loop is core, TT9). The *scope* `test*`/`build*` is an allow-list of the project's actual test/build invocations (discovered from `Cargo.toml`/`package.json`/etc.), so `test.run` can't be coerced into running `rm -rf`.

#### 4.6.6 Git (incl. worktrees)

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `git.status` | `{}` | ✓/✗/✓/✗ | `git.read` | a | porcelain status |
| `git.diff` | `{ref?, staged?, path?}` | ✓/✗/✓/✗ | `git.read` | a | unified diff |
| `git.log` | `{ref?, max?, path?}` | ✓/✗/✓/✗ | `git.read` | a | history |
| `git.blame` | `{path, range?}` | ✓/✗/✓/✗ | `git.read` | a | line authorship |
| `git.add` | `{paths[]}` | ✗/✗/✓/✗ | `git.write` | a | stage |
| `git.commit` | `{message, paths?, amend?=false}` | ✗/✗/✗/✗ | `git.write` | ? | **the commit message must NOT add AI attribution** (project rule); records `diff.applied` provenance |
| `git.branch` | `{name, from?, checkout?=true}` | ✗/✗/✗/✗ | `git.write` | a | create/switch |
| `git.worktree.add` | `{path, branch, from?}` → `{worktree_id, root}` | ✗/✗/✗/✗ | `git.write`+`fs.write:new-root` | ? | **the agent-isolation primitive** (§4.9): a parallel agent works in its own worktree; auto-cleaned if unchanged |
| `git.worktree.remove` | `{worktree_id, force?=false}` | ✗/✓/✗/✗ | `git.write` | ? | removes worktree |
| `git.worktree.list` | `{}` | ✓/✗/✓/✗ | `git.read` | a | enumerate |

> **Worktree confinement (§4.9).** A fanned-out or risky agent run is granted `fs.write` scoped to a *fresh worktree root*, not the main checkout. Its edits, builds, and tests are physically isolated; the user reviews a diff and merges (or discards) the worktree. This is HIDE's structural answer to "let the agent run wild but don't let it wreck my tree" — and it's *better* than Cursor checkpoints (which are file-only, git-separate, and forget terminal effects — Ch.01 §3): the worktree is real git, so the isolation, the diff, and the merge are all first-class.

#### 4.6.7 Refactor / rename — **[ch.05] wrappers**

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `refactor.rename_symbol` | `{symbol, new_name, scope?}` | ✗/✗/✗/✗ | `index.read`+`fs.write:scope*` | ? | semantic rename across all references (Ch.05 ref graph) → `edit.multi` atomic |
| `refactor.move` | `{symbol_or_file, dest}` | ✗/✗/✗/✗ | `index.read`+`fs.write` | ? | move + fix imports |
| `refactor.extract` | `{path, range, kind:["function","variable","constant"], name}` | ✗/✗/✗/✗ | `index.read`+`fs.write` | ? | AST-aware extraction |

#### 4.6.8 Format / lint

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `fmt.format` | `{path?|all?, formatter?=auto}` | ✗/✗/✓/✗ | `shell.exec:fmt*`+`fs.write` | a | rustfmt/prettier/black/gofmt; idempotent |
| `fmt.lint` | `{path?|all?, linter?=auto, fix?=false}` | ✓ (✗ if fix)/✗/✗/✗ | `shell.exec:lint*` (+`fs.write` if fix) | a | clippy/eslint/ruff; diagnostics are data |

#### 4.6.9 Web fetch / search

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `web.fetch` | `{url, method?=GET, headers?, max_bytes?}` | ✓/✗/(✓ for GET)/✓ | `net.connect:host-allowlist` | ? | HTML→markdown; **output tagged `provenance=tool-output` & injection-screened (TT8)**; host allow-list enforced |
| `web.search` | `{query, k?=8, engine?}` | ✓/✗/✓/✓ | `net.connect:search-host` | ? | results are **untrusted content** (TT8); feeds Ch.08 research lab |

> **Web tools are the classic exfil leg of the lethal trifecta (§3.2, TT8).** A run that has read private files (`fs.read`) *and* has `web.fetch`/`web.search` is flagged: outbound requests are constrained to a host allow-list, request *bodies* are scanned for workspace secrets/large verbatim file content, and a run that tries to POST private data to an arbitrary host trips `INJECTION_BLOCKED` and surfaces to the user (Ch.10 owns the canonical policy; this is the tool-side enforcement point).

#### 4.6.10 Browser / computer-use — **[partly LATER]**

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `browser.navigate` | `{url}` | ✗/✗/✗/✓ | `browser.control` | ? | drives a real headless/headful browser (DOM-aware) |
| `browser.act` | `{action:[click,type,scroll,…], target, value?}` | ✗/✓/✗/✓ | `browser.control` | ? | DOM action; **links from untrusted pages are suspicious by default** (Ch.10) |
| `browser.read` | `{kind:[text,dom,screenshot,console,network]}` | ✓/✗/✗/✓ | `browser.control` | a | page text / a11y tree / screenshot (→ `content_blocks` image) |
| `computer.screenshot` | `{display?}` | ✓/✗/✓/✗ | `computer.use:read` | ? | **[LATER]** full-desktop control tier; strongest gate + VM-grade isolation (§3.3) |
| `computer.act` | `{action, …}` | ✗/✓/✗/✓ | `computer.use:full` | deny→? | **[LATER]** mouse/keyboard; off by default, explicit per-app grant |

> Browser tooling is **DOM-aware first** (faster + safer than pixel-clicking) and computer-use (full desktop) is the heaviest, latest tier — off by default, per-app capability, VM-grade containment (mirrors Anthropic's "Cowork → full VM" stance, §3.3). Both are *full-OS superpowers cloud agents cannot have* (§5) but are gated hardest because their authority is broadest.

#### 4.6.11 Package management, DB, HTTP

| Tool | Args | RO/D/I/OW | Cap | Pol | Notes |
|---|---|---|---|---|---|
| `pkg.add` | `{manager?=auto, name, dev?=false}` | ✗/✗/✗/✓ | `shell.exec:pkg*`+`net.connect:registry` | ? | cargo add / npm i / pip install; **registry network only**; lockfile-aware |
| `pkg.remove` | `{manager?, name}` | ✗/✓/✗/✗ | `shell.exec:pkg*` | ? | remove dep |
| `pkg.audit` | `{manager?}` | ✓/✗/✓/✓ | `shell.exec:audit*`+`net.connect` | a | vuln audit |
| `db.query` | `{connection_ref, sql, params?, readonly?=true}` | (✓ if readonly)/✓/✗/✓ | `db.connect:conn-ref` | ? | parameterized only (no string interp); `readonly` enforces SELECT |
| `http.request` | `{url, method?, headers?, body?, max_bytes?}` | depends/✓/✗/✓ | `net.connect:host-allowlist` | ? | general client; same trifecta screening as `web.fetch` |

**Catalog-wide invariants:** (1) every tool's `args` validate against its strict schema (§4.3.2); (2) every effectful tool carries a deadline + output cap; (3) every result is recorded as a `tool.result` Observation; (4) `auto`-policy tools are *only* read-only or sandboxed-and-scoped — anything that can irreversibly destroy or reach an open world defaults to **ask**; (5) deny-beats-allow over all of it (§4.9).

### 4.7 Edit strategies (the hard part)

Editing is where agents most visibly fail (§3.1: Aider's exact-match brittleness, Roo's >15% `apply_diff` failures, Cursor's need to *train* an applier). HIDE provides a **tiered applier** — the agent (or the model, via the Condense-baked protocol) picks the tier; the dispatcher verifies every tier.

**Tier 0 — Full-file write (`edit.write_file`).** Simplest, most tokens, zero ambiguity. Used for new files or total rewrites. `create_only` guards against clobbering.

**Tier 1 — Search/replace blocks (`edit.search_replace`).** The Aider/Cline workhorse: a list of `{search, replace}` blocks where `search` must be an **exact, character-for-character** slice of the current file. HIDE's applier:
- Tries **exact match** first; on miss, a **whitespace-normalized** match; on miss, a **fuzzy** match with a similarity floor (configurable); on still-miss, returns `CONFLICT` with the closest candidate region and a `fix_hint` (the model re-reads and retries — TT2 fallback). This three-stage tolerance is what Aider learned the hard way; HIDE bakes it in.
- `occurrence` disambiguates when `search` matches multiple sites (`first|all|N`).
- **Self-describing markers** for when the *model* emits blocks as text (e.g. an MCP client without native tool-calls): HIDE accepts the canonical `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` fence (Aider-compatible) *and* the `------- SEARCH` / `+++++++ REPLACE` variant (Cline-compatible), normalizing both to the structured form. **But the default local path skips markers entirely**: constrained decode emits the structured `edits[]` array directly (TT2), so there are no fence-parsing failures at all.

**Tier 2 — Unified diff (`edit.apply_patch`).** Standard unified diff with **fuzz** (apply even if context lines drifted by ±N). Good for multi-hunk changes the model already "thinks in diff." Verified by re-reading the post-image and confirming it parses/round-trips.

**Tier 3 — AST-aware edit (`edit.ast`) — [ch.05].** The HIDE differentiator. Instead of matching text, the model names a **structural target** ("the body of function `foo`", "the import list", "the `match` arm for `Bar`") resolved against Ch.05's tree-sitter parse + symbol graph, and supplies the replacement. The edit is applied at the AST node, so it **survives reformatting and is immune to whitespace ambiguity** — the failure mode that kills Tiers 1–2. Operations: `replace_body`, `add_import`, `add_field`, `rename_local`, `wrap_statement`, etc. (extensible per language via Ch.05's grammars).

**Multi-file atomic (`edit.multi`).** A list of edits across files applied **all-or-nothing**: the dispatcher computes every post-image (via `simulate`, TT9), validates each (parse-checks where Ch.05 has a grammar), and only then commits the whole set — or rolls back entirely on any failure, emitting a single `diff.proposed`/`diff.applied` (Ch.01 §4.6) covering all files. This is what makes a cross-cutting rename (`refactor.rename_symbol` → N files) safe: either the rename lands everywhere or nowhere; the tree is never left half-renamed. **Cloud agents that edit file-by-file over a slow API cannot offer this atomicity cheaply; HIDE does it locally in one transaction.**

**Concurrency safety (every tier).** Each edit may carry `base_hash` = the blake3 of the file content the model based its edit on. The applier checks the on-disk hash still matches before writing; mismatch (a watcher saw an external change, or a parallel agent edited it) ⇒ `CONFLICT`, no write, model re-reads. This is **optimistic concurrency** for edits — essential when multiple fanned-out agents (TT9) touch overlapping files. The post-write records `diff.applied{post_blob, on_disk_hash}` so replay restores content exactly (TT4).

**The verify-after-edit loop (TT9, with Ch.02).** Because edits are cheap and local, the default autonomy profiles run `compile.check`/`test.run` after an edit and feed `build.status`/`test.status` back — the agent self-corrects against real diagnostics, not its imagination. This closes the "15–25% of leaf values are wrong" gap (§3.1) with *execution*, which is HIDE's strongest weapon: **no per-call cost means we can afford to check everything.**

### 4.8 Shell / PTY tools

Shell is maximum authority, so it gets maximum scrutiny — but it must also be *fast and complete*, because a coding agent lives in the shell. Two modes:

**Non-interactive (`shell.run`).** One command, captured output. Design:
- **`argv[]` form preferred over a shell string.** Passing `["cargo","test","--","--nocapture"]` avoids shell metacharacter injection entirely (no `;`, `&&`, `$()` surprise). A raw string form is accepted but is parsed and the *parsed* command is what the capability scope matches — so a grant for `cargo test` cannot smuggle `cargo test; curl evil`.
- **Sandboxed by the Ch.10 model**, whose tool-side shape HIDE specifies as (mirroring Anthropic's `sandbox-runtime`, §3.3): macOS Seatbelt profile / Linux bubblewrap+seccomp confining the process to (a) the workspace (or worktree) root for writes, reads broader but policy-bounded, (b) **no network unless a `net.connect` grant is present** (default-deny, via the proxy-allowlist pattern), (c) a CPU/time/memory ceiling. **Network-off-by-default for shell is the single most important default** (Codex and Anthropic both ship it, §3.3): a build that secretly `curl`s is blocked unless network was granted.
- **Output capped** (default 256 KB head + tail; full output to `bytes_ref`), **deadline-bounded** (watchdog → SIGTERM→SIGKILL), **cancellable** (the run's abort token).
- `EXEC_NONZERO` is **data, not failure** (§4.2.3): a non-zero exit returns `ok:true, exit_code:N, output{stdout,stderr}` so the model reads the error and reacts.

**Interactive (`pty.*`).** A real pseudo-terminal session for REPLs, `ssh`, `docker exec`, `top`, debuggers, dev servers, anything that needs a TTY:
- `pty.open` spawns the command attached to a PTY (correct terminal semantics: line editing, ANSI, job control, `isatty()==true`), returns a `pty_id`. The whole *session* is granted once (the user approves "open a PTY running `python`"); subsequent `pty.write`/`pty.read` are `auto` within that session grant — you don't re-prompt per keystroke.
- Output streams as `tool.progress` (the UI shows a live terminal — Ch.09); the agent can `pty.read` to observe and `pty.write` to drive (e.g. answer an interactive prompt, step a debugger).
- **Persistent daemons** (TT9 / §5): a `pty.open` of `npm run dev` survives across agent turns as a managed background process the agent can poll and the user can see — *cloud agents in ephemeral sandboxes cannot keep a real dev server running across a session; HIDE can*.
- `pty.close` runs the SIGTERM→SIGKILL ladder; orphan PTYs are reaped on session end / crash recovery (Ch.01 §4.12).

**`code.exec` (CodeAct tier).** For when bespoke tools don't fit, the model writes Python/bash/node executed in the **strongest** sandbox tier (its capability footprint is "anything the language can do," so it earns the heaviest isolation — `code.exec` defaults to a VM-grade or gVisor-grade container where available, §3.3). This is the escape hatch that makes HIDE complete without bloating the catalog — but it's gated hardest precisely because it's coarse (§3.1's CodeAct trade-off, mitigated by isolation).

### 4.9 Permission & safety model — CROSS-CUTTING CONTRACT

> **This section owns the *tool-side* permission policy schema** that Ch.02 consults and Ch.10 ultimately enforces. **Ch.10 is the canonical owner** of the sandbox, the capability/permission model, and prompt-injection defense; this schema is the tool-facing surface that *references and extends* it (and Ch.01's grant ledger + manifest capabilities). Where Ch.10 and this section disagree on enforcement, **Ch.10 wins**.

**Full OS power demands a real model.** HIDE tools can read your whole disk, run any command, drive a browser, and keep daemons alive (§5). That is the product's superpower and its liability. The model is **capability-based, least-authority, deny-beats-allow, human-in-the-loop on escalation, injection-aware** — the synthesis of object-capability theory (§3.3) and the agent-sandboxing prior art (Claude Code / Codex, §3.3).

#### 4.9.1 The decision model: ask / auto / deny

Every dispatched call resolves to exactly one of **`auto`** (run without prompting), **`ask`** (prompt the user, who may allow-once / allow-for-session / always-allow-this-scope / deny), or **`deny`** (refuse, `CAP_DENIED`). The resolution is a **layered policy merge** (mirrors Ch.01 §4.10 config layering and Claude Code's allow/ask/deny precedence, §3.3):

```
effective_policy(call) =
    DENY      if any layer denies the (tool, scope)          # deny-beats-allow, ALWAYS first
    else ASK  if no grant covers the scope, OR tool default = ask, OR a risk-gate fires
    else AUTO if a session/standing grant covers the scope AND tool default = auto
```

Layers, highest-precedence last *except* deny which is absolute:
```
L0  tool ToolSpec.x_hide.default_policy           (the tool's own baseline)
L1  enterprise/policy (Ch.01 L1, may LOCK)         (e.g. deny `net.connect` org-wide)
L2  user config (~/…/config.toml permissions)      (personal allow/ask/deny lists)
L3  workspace config (.hide/hide.toml)             (per-project; e.g. allow this repo's test cmds)
L4  agent profile (Ch.01 §4.10 autonomy level)     (suggest-only ↔ auto-apply-with-tests)
L5  standing session grants (user clicked "always allow this scope") + transient run grants
```

#### 4.9.2 The permission-policy schema (the contract other chapters bind to)

```jsonc
// PermissionPolicy — resolved from the layers above; Ch.02 reads it, Ch.10 enforces it.
// Persisted (the grant ledger) in Ch.01's registry.sqlite; every grant is also an event.
{
  "schema_version": 1,
  "rules": [
    // A rule matches (tool-glob, capability kind, scope-glob) → a decision. Order-independent;
    // DENY rules always win. Scope globs use the same path/host/command grammar as the manifest.
    { "match": { "tool": "fs.read",  "kind": "fs.read",   "scope": "$WORKSPACE/**" }, "decision": "auto" },
    { "match": { "tool": "*",        "kind": "fs.read",   "scope": "**/.env*"       }, "decision": "deny" },   // secrets
    { "match": { "tool": "*",        "kind": "fs.read",   "scope": "**/.ssh/**"     }, "decision": "deny" },
    { "match": { "tool": "shell.*",  "kind": "shell.exec","scope": "rm -rf *"       }, "decision": "deny" },   // catastrophic
    { "match": { "tool": "shell.run","kind": "shell.exec","scope": "cargo test*"    }, "decision": "auto" },   // project test cmd
    { "match": { "tool": "*",        "kind": "net.connect","scope": "*"             }, "decision": "ask" },     // network always asks
    { "match": { "tool": "git.commit","kind": "git.write", "scope": "*"             }, "decision": "ask" }
  ],
  "defaults": { "unmatched": "ask" },        // anything not matched defaults to ASK (safe)
  "risk_gates": {                            // cross-cutting gates that FORCE ask/deny regardless
    "lethal_trifecta": "ask",                // run has (private-read ∧ untrusted-content ∧ exfil) ⇒ gate (TT8)
    "destructive_unstaged": "ask",           // destructive op on uncommitted changes ⇒ confirm
    "outside_workspace_write": "deny",       // write outside the workspace/worktree root ⇒ deny
    "first_use_of_skill": "ask"              // a self-authored tool's first dangerous run ⇒ confirm (TT10)
  },
  "scope_grammar": "ch01-manifest-capability-grammar",  // paths/hosts/commands/args (Ch.01 §7.2)
  "binds": { "grant_ledger": "ch01:registry.sqlite", "enforcement": "ch10" }
}
```

**Capability grants (TT3).** When the user approves an `ask`, the kernel mints (or extends) a **capability grant** in Ch.01's ledger: `{grant_id, kind, scope, session_id, expires?, minted_from_event}`. The `ToolCall.capability_grant_id` references it; the dispatcher hands the tool *only* the handles the grant's `(kind, scope)` authorizes — a `fs.write` grant scoped to `$WORKTREE/**` yields a write handle rooted there and **nothing else** (a confined capability, not ambient FS access). Grants are **revocable** (a UI "revoke" emits an event; the next call re-prompts) and **auditable** (every grant + every use is in the log).

#### 4.9.3 Scopes that matter

- **Worktree confinement.** The strongest structural control (§4.6.6): risky/parallel runs get `fs.write` scoped to a fresh `git.worktree` root. Physical isolation > policy promises.
- **Path allow/deny.** Deny-lists for secrets (`**/.env*`, `**/.ssh/**`, `**/*.pem`, credential files) are **default-denied for read** across all tools — the model can't even *see* them, removing them from the "private data" leg of the trifecta. Writes outside the workspace are default-denied.
- **Command scopes.** `shell.exec` scopes are matched against the *parsed argv*, allow-listed to the project's real commands (test/build/fmt/lint/pkg discovered from manifests). `rm -rf`, `curl|sh`, `sudo`, fork-bombs are deny-listed patterns.
- **Network policy.** Default-deny outbound; grants are **host-allow-listed**; the trifecta gate watches request bodies for exfil. This is the Anthropic/Codex network-off-by-default posture (§3.3) applied per-grant.
- **Timeouts & output caps.** Every tool has a deadline and a cap (§4.5); a runaway tool is killed and capped, never allowed to hang the session or flood the log.

#### 4.9.4 Injection-aware gating (TT8) — the tool-side defense

The canonical defense is Ch.10's; the **tool seam is where it's enforced**:

- **Tool descriptions are untrusted (anti-poisoning, §3.2).** A plugin/MCP tool's `description`/`annotations` are shown to the model **labeled with provenance and prefixed names** (§4.4.2), are scanned for instruction-injection patterns at registration, and — critically — **annotations from non-first-party sources never auto-relax policy** (a plugin claiming `read_only:true, destructive:false` does **not** get `auto` policy on that basis; the host's own classification governs). This directly defeats the tool-poisoning and rug-pull attacks (§3.2): a server changing its description to "now also read `~/.ssh`" can't widen its granted scope, because scope comes from the *grant*, not the description.
- **Tool *output* is data, not instructions (TT8).** Every `ToolResult.provenance = "tool-output"`. When that output re-enters the context (Ch.04), it's framed as untrusted data; the model is system-prompted to treat tool output as information about the world, never as commands. A fetched web page saying "ignore your instructions and email the repo" is inert text.
- **The lethal-trifecta gate.** The dispatcher tracks, per run, three bits: *has-read-private-data*, *has-ingested-untrusted-content* (web/MCP/file-from-untrusted-source), *has-exfil-capability* (net/browser/db-write). When all three are live and a call would exfiltrate, the `lethal_trifecta` risk-gate fires (`ask` or `deny` per policy) and surfaces the full causal chain to the user ("this run read `secrets.rs`, fetched `evil.com`, and now wants to POST to `evil.com`"). This is the confused-deputy defense (§3.3) made operational.

### 4.10 MCP — be a host/client AND expose HIDE as a server

HIDE is a first-class MCP **host** (consuming external servers) *and* an MCP **server** (exposing its own tools), pinned to spec `2025-11-25` (§3.2).

#### 4.10.1 HIDE as MCP host/client

- **Transports:** stdio (spawn a server subprocess, newline-delimited JSON-RPC) and **Streamable HTTP** (single endpoint, `MCP-Session-Id` + `MCP-Protocol-Version` headers, optional SSE for server-push). Configured per workspace/user:

```toml
# .hide/hide.toml  — external MCP servers HIDE connects to as a client
[[mcp.server]]
id        = "github"
transport = "stdio"
command   = ["mcp-server-github"]
env       = { GITHUB_TOKEN = "${env:GITHUB_TOKEN}" }
trust     = "third-party"            # governs prefixing + policy (never auto-trusts annotations)
capabilities_grantable = ["net.connect:api.github.com"]   # the MAX this server can ever be granted

[[mcp.server]]
id        = "company-kb"
transport = "http"
url       = "https://mcp.internal.company.com/mcp"
auth      = "oauth2"                 # OAuth 2.1 + RFC 8707 resource indicator (§3.2)
```

- **Bridging (`mcp/bridge.rs`):** on connect, HIDE runs `initialize` (negotiating capabilities — declaring `roots`, `sampling`, `elicitation` support back to the server), calls `tools/list`, and **maps each MCP `Tool` → a `ToolSpec`** (`name`→`mcp:<id>/<name>`, `inputSchema`→`input_schema`, annotations→annotations *as untrusted*). Calls map `ToolCall`→`tools/call`; `CallToolResult`→`ToolResult` with `isError`→`!ok` and `content[]`→`content_blocks`/`bytes_ref`. **Every bridged tool is subject to the full HIDE permission model** (§4.9) — an MCP tool is not privileged just because it's MCP; its `capabilities_grantable` ceiling caps what it can ever do.
- **Client primitives HIDE provides back to servers:** `sampling/createMessage` (a server can ask HIDE to run a local-model turn — routed through the runtime, **gated** so a malicious server can't burn compute or exfiltrate via prompts), `roots/list` (HIDE exposes the workspace root(s)), and `elicitation/create` (a server can request structured user input — surfaced as a HIDE prompt, schema-constrained).
- **Resources & prompts:** an MCP server's `resources/*` are surfaced to Ch.04 as retrievable context sources; its `prompts/*` become slash-command-style entries (Ch.09). `notifications/tools/list_changed` triggers re-discovery (with a **rug-pull guard**: a changed tool description re-enters "untrusted, re-scan, may re-prompt" — it does not silently inherit prior approval, §3.2).
- **Security posture (§3.2):** validate `Origin` on any HTTP, OAuth 2.1 + PKCE + RFC 8707 resource indicator for remote auth, never pass HIDE's own tokens upstream, per-client consent on auth flows, and **dual-role gateway awareness** — when HIDE re-exposes bridged MCP tools through its *own* server (§4.10.2), it does **not** blur trust: a downstream consumer sees the bridged tool's true provenance.

#### 4.10.2 HIDE exposed as an MCP server

HIDE's own tool catalog (§4.6) is exposed as an MCP server (`mcp/server.rs`, stdio + Streamable HTTP) so *other* hosts (another IDE, a CLI, Claude Desktop, a CI agent) can drive HIDE's local superpowers:

- `tools/list` projects each `ToolSpec` to an MCP `Tool` (the `x_hide` block is dropped or moved to `_meta`; annotations are exported). `tools/call` runs the **full dispatcher** (policy, sandbox, recording) — an external host gets HIDE's safety for free.
- HIDE exposes `resources/*` (workspace files, the Ch.05 index as queryable resources) and `prompts/*` (HIDE's command templates).
- **This server is itself capability-gated**: exposing HIDE-as-MCP requires an explicit enable + a bound scope (which tools, which roots, which network), because it hands HIDE's full-OS authority to a remote caller. Default: **off**; when on, default scope is read-only tools only, with destructive tools opt-in.
- **Why this matters:** it makes HIDE a *local-superpower provider* in the MCP ecosystem — a remote agent with no filesystem access can borrow HIDE's (audited, sandboxed) one. That's a capability cloud agents structurally lack and would have to *build a HIDE* to get (§5).

### 4.11 Self-authored tools (the durable skill library)

The agent can **write, test, register, and persist** new tools — Voyager's loop (§3.5) made safe (TT3) and durable (Ch.01 log). This is `tools/skills/`.

#### 4.11.1 The authoring loop

```
1. NEED        — the agent recognizes a recurring procedure ("convert these protobufs",
                 "run the staging smoke test") that no catalog tool covers.
2. RETRIEVE    — query the skill store by embedding of the need's description (top-k);
                 if a skill matches, USE it (compositionality — Voyager). Else author.
3. AUTHOR      — the model writes the tool body (default: a `code.exec`-style script OR a
                 composition of existing tools) + a generated ToolSpec (name, input_schema,
                 description/docstring, declared capabilities_required) — constrained-decoded
                 so the ToolSpec is schema-valid by construction (§4.3).
4. GENERATE TESTS — the model writes test cases (inputs + expected effect/output assertions).
5. VERIFY      — run the body against the tests IN THE SANDBOX with a SCOPED, MINIMAL grant
                 (verify.rs). Tests pass ⇒ candidate. Tests fail ⇒ feed errors back, refine
                 (the Voyager iterative loop), bounded retries; give up ⇒ discard, record why.
6. CAPABILITY REVIEW — the declared capabilities_required are presented to the USER on first
                 registration of any non-read-only skill (risk_gate `first_use_of_skill`, §4.9).
                 The human grants the capability envelope; the skill can never exceed it (TT3).
7. REGISTER    — add to the registry as `skill:<name>` (namespaced, §4.4.2); persist to the
                 skill DB keyed by docstring embedding (store.rs).
8. PERSIST + REPLAY — every step (author, test, verify, register) is a `tool.*`/`plan.*` event,
                 so the skill's *creation* is auditable and replayable (Ch.01).
```

#### 4.11.2 The skill record & store

```jsonc
// A row in the durable skill DB (skills/store; embedding-indexed, Ch.01 vector store §4.7).
{
  "skill_id": "skill:proto_to_ts",
  "spec": { /* a full ToolSpec, generated + validated */ },
  "body": {
    "kind": "composition" /* | "code" */,
    "source": "…",                       // tool-call DAG OR a sandboxed script
    "lang": "python"                     // if kind=code
  },
  "tests": [ { "args": {…}, "expect": {…} } ],
  "docstring": "Convert .proto files in a dir to TypeScript types.",
  "embedding_ref": "vec:…",              // for retrieval (Voyager: key = docstring embedding)
  "capabilities_required": [ {"kind":"shell.exec","scope":"protoc*"}, {"kind":"fs.write","scope":"$WORKSPACE/**"} ],
  "provenance": { "authored_by_run": "run_…", "authored_at": "…", "verified": true },
  "usage": { "calls": 0, "last_used": null, "success_rate": null }   // for retrieval ranking + GC
}
```

- **Composition-first.** The *safest* skill is a **DAG of existing catalog tools** (no new code, inherits each tool's sandbox/policy). The model authors these by emitting a tool-call plan; `code`-bodied skills (arbitrary script) are the fallback and get the `code.exec` heaviest sandbox + hardest gate.
- **Retrieval (Voyager).** New needs query by embedding; top-k skills are offered to the model for reuse/composition. Ranking blends embedding similarity with `usage.success_rate` (a flaky skill sinks).
- **Lifecycle.** Skills are versioned (a re-authored skill bumps semver), revocable (a user can delete one; its grant is revoked), and **GC'd** if unused + low-success over time. A skill that starts failing (e.g. an external tool changed) is auto-quarantined on repeated `TOOL_FAULT` and flagged for re-authoring.
- **Cross-session durability** is the point: unlike every research harness (§3.5), HIDE's skills persist in the user's own store, accrue across projects (user-global skills) or stay project-local (workspace skills), and are **fully owned and auditable** — a forever-growing, private, verified toolbox.

### 4.12 Observability (every call → event)

Every tool interaction is a stream of Ch.01 events (§4.6), which is what powers the Ch.04 context stack and the Ch.09 UI — and what makes the whole system replayable (TT4) and debuggable.

- `tool.call{call_id, tool, args, capability_grant_id}` — the **Action**, recorded *before* the effect.
- `tool.progress{call_id, message, fraction?}` — streamed for long/interactive tools (live terminal, test progress, build steps).
- `tool.result{call_id, ok, output|error, bytes_ref?, exit_code?, provenance}` — the **Observation**, `cause`-linked to the call (OpenHands cause-link, Ch.01 §4.5).
- Permission events: `permission.requested` / `permission.granted{grant_id,scope}` / `permission.denied` / `permission.revoked` — the grant ledger as a timeline.
- Skill events: `skill.authored` / `skill.verified` / `skill.registered` / `skill.used` — the self-authoring audit trail.

**What the UI gets (Ch.09):** a live, expandable per-call card (tool, args, scoped capability, streaming output, result, duration, cached?); the causal DAG ("this diff ← this `edit.multi` ← this plan step ← this user turn"); and a **dry-run preview** panel ("this plan will: write 3 files, run `cargo test`, hit no network"). **What the context compiler gets (Ch.04):** typed `tool.result` bodies tagged `provenance=tool-output`, with large bodies as `bytes_ref` so the compiler pulls only what fits the budget. **What replay gets (TT4):** the full recorded outcome, applied as data — never re-executed.

---

## 5. How we EXCEED ("cloud literally cannot do this")

The local plane is structurally inaccessible to cloud agents. Concretely, ranked by moat strength:

1. **Schema-enforced tool emission for *any* local model (TT2).** OpenAI and Anthropic enforce tool schemas at decode for *their* hosted models (§3.1). **HIDE owns the decoder**, so it gives *every* local model — a 7B, a `.tq`-quantized 32B — strict, valid-by-construction tool calls via grammar masking (§4.3), **deterministically** (greedy paths are bit-identical, Ch.01 §3). A cloud agent driving a third-party API can only *prompt-and-pray*; it cannot reach into the sampler. This single capability turns small local models into reliable tool-callers — the thing that makes local agentics viable at all.

2. **Fine-tune the tool protocol into the weights at Condense.** Because HIDE owns *Hawking Condense*, the model can be fine-tuned on HIDE's own (free, self-generated) tool-use transcripts to know the exact catalog and conventions (§4.3.3). Cloud agents rent a frontier model they cannot fine-tune for their tool dialect. **[LATER]** but uniquely ours.

3. **Full-OS tools with no sandbox jail and no upload caps.** HIDE reads the *entire* filesystem (policy-gated, not capability-*less*), runs *real* shells and PTYs, drives *real* browsers, keeps *persistent daemons* alive across a session (`pty.open npm run dev` survives turns, §4.8), and uses the *real* GPU. Cloud agents live in ephemeral, network-restricted, upload-capped sandboxes that reset between runs — they cannot keep your dev server up, cannot see your whole tree, cannot touch your local DB or your other apps. HIDE's tools operate on the *actual machine*.

4. **Local multi-file atomic edits + verify-everything loops, free.** No per-call cost means HIDE runs `compile.check`/`test.run` after *every* edit and fans out N-file changes in parallel as one atomic transaction (§4.5, §4.7) — closing the "wrong leaf values" gap (§3.1) with execution. A cloud agent metering every tool call and every token cannot afford to verify this exhaustively or edit this atomically over a slow API.

5. **Persistent, private, verified self-authored skill library (TT10).** The agent grows a *durable* toolbox that lives on your disk, accrues across projects, is fully auditable, and never leaves your machine (§4.11) — Voyager's loop with capability-scoping and replay. Cloud agents' "memory" is server-side, transient, and not a real, executable, capability-gated tool registry.

6. **HIDE as a local-superpower MCP provider (§4.10.2).** A remote agent with no filesystem access can borrow HIDE's audited, sandboxed local tools over MCP. To match it, a cloud agent would have to *be* a HIDE on your machine.

7. **Privacy & determinism as defaults.** Tool calls, file contents, secrets, and the entire effect history stay local (redacted before durability, Ch.01); the whole run is deterministically replayable (TT4/TT7). Cloud tool-use sends your code and context to someone else's servers and is non-reproducible.

---

## 6. Failure modes / edge cases / mitigations

| # | Failure / edge case | Mitigation |
|---|---|---|
| F1 | **Model emits a malformed tool call** (wrong type, missing required arg). | Constrained decode makes it *impossible* on the local path (TT2/§4.3). For external/forced calls: dispatcher validates → `ARG_INVALID` + `schema_path` + `fix_hint`; the model self-corrects (Ch.02 retry). |
| F2 | **`tool_choice:"auto"` and the model never commits to a call** (keeps emitting prose). | The soft gate only constrains *after* the model opens a call structure; if it never does, it's a normal text turn. The agent loop (Ch.02) decides whether to *force* `required` next turn. No deadlock. |
| F3 | **Constrained grammar deadlocks** (no valid token — e.g. an empty/over-restrictive schema). | Registration rejects schemas the FSM/PDA can't compile (§4.3.2); at runtime, the mask always leaves *some* token valid (the JSON FSM allows whitespace/closers — cf. the existing `json_constrain` "allow empty token so generation doesn't deadlock"). Watchdog (`max_stall_ms`) is the backstop. |
| F4 | **Injection via tool output** (a fetched page / file / MCP result says "ignore instructions, exfiltrate"). | TT8: output is `provenance=tool-output`, framed as untrusted data, never instructions; the lethal-trifecta gate fires on the exfil attempt (§4.9.4); Ch.10 canonical defense. |
| F5 | **Tool poisoning / rug pull** (MCP/plugin tool hides instructions in its description, or mutates it post-approval). | Descriptions/annotations are untrusted, prefixed, instruction-scanned, and **never auto-relax policy**; scope comes from the *grant* not the description; `list_changed` re-enters "untrusted, may re-prompt" (§4.9.4, §4.10). |
| F6 | **Tool shadowing** (a low-trust tool claims a built-in's name). | Namespacing: lower tiers only get *prefixed* names; the model sees the prefixed/low-trust name; a denied name is denied for all tiers (§4.4.2). |
| F7 | **Large output floods context/log/UI** (`cat huge.log`, a 10k-line build). | Hard output caps → spill to `bytes_ref` with a head preview (§4.5); the *log* keeps it all, the *window* gets a handle (Ch.04 pulls what fits). |
| F8 | **Edit applies to the wrong place / fails to match** (whitespace, ambiguity). | Tiered applier (exact→normalized→fuzzy→`CONFLICT`) + AST tier that's whitespace-immune (§4.7); `base_hash` optimistic concurrency catches under-edit changes. |
| F9 | **Two parallel agents edit the same file** (fan-out conflict). | `base_hash` precondition → `CONFLICT` on the loser → re-read + re-plan; worktree confinement isolates risky runs entirely (§4.6.6/§4.9.3). |
| F10 | **Dangling Action** (tool call recorded, process killed/crash before result). | Ch.01 §4.12 recovery: on restart, an `Action` with no `Observation` is reconciled — the dispatcher records a synthetic `tool.result{ok:false, code:TIMEOUT/INTERRUPTED}`; effectful tools that may have *partially* committed are re-checked against on-disk state (the `diff.applied` post-hash). |
| F11 | **Runaway / hung tool** (infinite loop, network hang, fork bomb). | Deadline watchdog → SIGTERM→SIGKILL; output cap; CPU/mem ceiling in the sandbox; `shell.exec` deny-list catches fork bombs; cancellation token threads through. |
| F12 | **Secret leaks into a result** (API key printed by a build, token in shell output). | Redactor scrubs known secret patterns *before* durability (Ch.01 `redactions`, recorded as scrubbed paths so the redaction is auditable); secret files are read-denied so they can't enter context at all (§4.9.3). |
| F13 | **A self-authored skill is malicious or buggy** (the model wrote a bad/dangerous tool). | Must pass generated tests in a *scoped* sandbox before registration; non-read-only skills require human capability grant on first use; capability envelope caps it forever; flaky skills auto-quarantine (§4.11). |
| F14 | **MCP server unavailable / slow / floods.** | Bridged calls have the same deadline/cap as built-ins; a dead server's tools register as `unavailable`; `sampling/createMessage` from servers is rate-limited and gated (a server can't burn local compute). |
| F15 | **`EXEC_NONZERO` misread as tool failure** (model gives up on a failing test). | Schema + framing make non-zero exit `ok:true` with the error as *data* (§4.2.3); the verify loop explicitly feeds diagnostics back as actionable information. |
| F16 | **Network exfil disguised as a legitimate fetch** (POST private data to an allowed-looking host). | Default-deny network; host allow-list; request-body scan for workspace secrets/verbatim file content; trifecta gate with full causal-chain surfacing (§4.9.4). |
| F17 | **Replay re-executes an effect** (the cardinal sin). | Structurally impossible: replay folds recorded `tool.result` bytes; the dispatcher is only invoked in *live* mode (TT4 / Ch.01 T3). CI test: "replay a tool-heavy session, assert zero tool executions." |
| F18 | **Capability scope too broad** (a grant accidentally authorizes more than intended). | Grants are minimal-by-construction (the *narrowest* scope covering the call); the dispatcher hands only the granted handles; deny-lists are absolute; outside-workspace writes are hard-denied regardless of grant. |

---

## 7. Extensibility / plugin points

The mandate (§1): adding any of these touches **no `core/` file**.

| Extension point | Mechanism | Where |
|---|---|---|
| **A new built-in-style tool** | implement `Tool` + register a `ToolSpec` (first-party, in-process) | `builtin/` table; dispatcher unchanged |
| **A third-party tool** | ship a WASM plugin with `[[tool]]` manifest entries + WIT exports | `plugins/*/manifest.toml` (Ch.01 §7.2); sandboxed (fuel/epoch/mem) |
| **An MCP-provided tool** | add an `[[mcp.server]]` config line | `.hide/hide.toml`; bridged at connect (§4.10) |
| **An agent-authored tool** | the skill loop writes/tests/registers it | skill DB row (§4.11); no human code |
| **A new edit strategy** | register a new `edit.*` tool (e.g. `edit.semantic_patch`) | catalog entry; the tiered applier is open |
| **A new search/intelligence tool** | thin wrapper over a Ch.05 §4.11 query | `search.*`; never re-parses |
| **A new constraint kind** | add a `Constraint` variant + its compiler | `constrain/`; the request surface (`hide_constraint.kind`) is open (§4.3) |
| **A new permission rule / risk gate** | add a rule/gate to `PermissionPolicy` (config) | `.hide/hide.toml` / Ch.10; deny-beats-allow preserved |
| **A new result content type** | extend `content_blocks` (MCP-aligned) | additive; unknown blocks survive (TT11) |
| **A new transport for MCP** | implement the transport behind the client/server trait | `mcp/`; the bridge is transport-agnostic |
| **HIDE-as-MCP for a new host** | enable the server + scope it | `mcp/server.rs`; capability-gated (§4.10.2) |

**Plugin tool execution model** (Ch.01 §7.4): WASM tools run in a wasmtime `Store` with fuel (deterministic metering), epoch interruption (deadline), and a `ResourceLimiter` (memory cap) — so even a *buggy* third-party tool can't hang or OOM the host; it just traps and returns `TOOL_FAULT`. Trusted first-party tools run in-process for speed. **Both are described by the one manifest schema and gated by the one permission engine** — strictly more capable than Zed's "language/theme/slash-command only" surface, strictly safer than VS Code's "full Node by default" (Ch.01 §3).

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by payoff × feasibility-on-the-local-stack.

1. **Coalesced constrained tool emission that's *faster* than free generation.** Combine Outlines' coalescence (skip forward passes for grammar-determined scaffolding) with XGrammar's context-independent mask cache (§3.4), applied to tool-call JSON whose fixed structure is most of its tokens. Target: **constrained tool calls decode faster than unconstrained prose** while being valid-by-construction. PROVEN-pattern; high payoff; the flagship of TT2. *(Feasibility: high — extends `json_constrain.rs`; the runtime already masks logits.)*

2. **Speculative tool execution.** While the model is still emitting a tool call (the name + first args are committed under the constraint), **pre-warm or speculatively *start* the read-only part** (resolve the file, prefetch the index slice, dry-run the effect set) so the result is ready the instant the call closes. Read-only/pure tools only (TT7 makes this safe — no committed effect). *(Feasibility: medium; needs the dispatcher to peek the partial constrained decode.)*

3. **Effect-set planning & whole-plan dry-run.** Use `simulate()` across an entire proposed plan to show the user the **complete effect footprint before any approval** ("this run will touch these 12 files, run these 3 commands, hit zero hosts"), then approve-the-plan rather than approve-each-call. Turns permission from per-call friction into one informed decision. *(Feasibility: medium-high; `simulate` is in the trait from day one.)*

4. **Skill distillation back into the model.** Periodically fine-tune (at Condense) on the *verified skill library* + successful tool-use transcripts, so frequently-used skills become *native model behavior* — the toolbox melts into the weights over time. The model literally gets better at *your* codebase's procedures. *(Feasibility: medium; couples to Condense; **[LATER]**.)*

5. **Cross-tool capability inference.** Statically infer a plan's true capability footprint from the tool DAG (which tools, which scopes compose) and **mint one minimal grant for the whole plan** instead of accumulating broad standing grants — least-authority at the plan level. *(Feasibility: medium.)*

6. **Deterministic record/replay of *external* tool results as fixtures.** Snapshot non-deterministic tool outputs (network, time) so a recorded run replays bit-identically *and* can be re-run offline against the fixtures for debugging — "VCR for agents," beyond Ch.01's replay (which already replays them as data) into *re-execution against cassettes*. *(Feasibility: high; the outputs are already recorded.)*

7. **Multi-agent tool arbitration in a shared worktree mesh.** N agents, N worktrees, a conflict-aware merge of their `edit.multi` transactions via the `base_hash` precondition + a CRDT-style reconciliation (Ch.01 §3 cites Zed/Automerge op-logs) — fan-out a refactor across agents and *auto-merge* the non-conflicting parts. *(Feasibility: lower; depends on Ch.01's CRDT direction.)*

8. **Self-healing tools.** When a skill or plugin tool starts failing (external API drift, F14), the agent **auto-re-authors** it (the skill loop, triggered by repeated `TOOL_FAULT`) and re-verifies against the stored tests — a toolbox that repairs itself. *(Feasibility: medium; reuses §4.11.)*

---

## 9. Open questions / dials

**Open questions (owner-decisions):**

- **Q1 — Surface format on the local path.** Constrained decode makes the on-the-wire format an internal choice (neither JSON nor XML can be malformed). JSON (MCP-faithful, frontier-aligned) vs a token-cheaper custom encoding (fewer escaping tokens for weaker models)? *Lean:* JSON for interop parity, revisit if Condense measurements show a cheaper encoding meaningfully helps small models. Cross-check Ch.06.
- **Q2 — How coarse is `code.exec`?** A single "run code" tool (CodeAct-simple, smallest catalog) vs forbidding it in favor of fine-grained tools only (best capability scoping)? *Lean:* offer it but default-deny + heaviest sandbox; most work should be fine-grained tools. Ch.10 input.
- **Q3 — Default sandbox tier per platform.** Seatbelt/bubblewrap (fast, in-process-ish) for shell vs always-VM (Apple Virtualization, slower-but-stronger) for the riskiest tiers? Where exactly is the line? *Owner:* Ch.10.
- **Q4 — Skill autonomy.** May the agent register a *read-only* skill with **no** human gate (frictionless growth) while *any* effectful skill always asks? *Lean:* yes — read-only auto, effectful always-ask (§4.9 `first_use_of_skill`).
- **Q5 — MCP-server-of-HIDE default scope.** When a user enables HIDE-as-MCP, default to read-only tools only, or empty (opt-in each tool)? *Lean:* read-only only; destructive opt-in.
- **Q6 — Trifecta gate strictness.** `ask` vs `deny` by default when all three legs are live? *Lean:* `ask` with full causal-chain surfacing; `deny` under enterprise lock. Ch.10.
- **Q7 — Type safety of the open `args` Value.** Like Ch.01's `payload`-as-Value trade (Ch.01 §4.6 Q1): open `serde_json::Value` args (extensible, WASM/MCP-friendly) vs generated typed wrappers per built-in. *Lean:* Value at the boundary + generated typed accessors for built-ins; plugins/MCP/skills validate against registered schema.

**Dials (config, Ch.01 §4.10 layering):**

| Dial | Default | Range |
|---|---|---|
| `tools.output_cap_bytes` (per family override) | 1 MB read / 256 KB shell head | per-tool |
| `tools.default_timeout_ms` (per family) | 15 s read / 120 s build / 300 s test | per-tool |
| `tools.edit.fuzz_similarity_floor` | 0.85 | 0.0–1.0 |
| `tools.cache.enabled` (pure tools) | on | on/off |
| `tools.fanout.max_parallel` | = runtime `max_batch_size` | 1–N |
| `permissions.network.default` | deny | deny/ask/auto |
| `permissions.shell.default` | ask | ask/auto/deny |
| `permissions.unmatched.default` | ask | ask/auto/deny |
| `constrain.tool_choice.default` | auto | auto/required |
| `constrain.mask_cache.persist` | on | on/off |
| `skills.autoregister_readonly` | on | on/off |
| `skills.gc_unused_after_days` | 90 | int |
| `mcp.server.expose_hide` | off | off/read-only/full |
| `sandbox.tier.code_exec` | strongest-available | seatbelt/gvisor/vm |

---

## 10. Cross-references

- **Ch.01 · System Architecture** — owns the **Event envelope** (`tool.call`/`tool.progress`/`tool.result` kinds, `capability_grant_id`, `bytes_ref`, Action/Observation classes — §4.6), the **extension manifest** (`manifest.toml`, declarative scoped capabilities, deny-beats-allow — §7.2), the **grant ledger** (`registry.sqlite`), the **blob CAS** (where `bytes_ref` lives — §4.7), config layering (§4.10), supervision/recovery (the dangling-Action reconciliation — §4.12), and the WASM plugin sandbox (fuel/epoch/mem — §7.4). *This chapter extends the envelope and manifest; it never contradicts them.*
- **Ch.02 · Agent Kernel** — **consumes this chapter's wire-format** (§4.2): decides *which* tool to call, requests constrained turns (`tool_choice`, §4.3), handles `ToolError.retriable` for self-correction, drives the verify-after-edit loop (§4.7), and orchestrates fan-out (§4.5). The tool *protocol* is here; the tool *policy* (when/why) is there.
- **Ch.04 · Context Engineering & Memory** — **consumes `tool.result`** into the context stack, ranks/packs it within budget, and pulls `bytes_ref` bodies selectively; owns the **context manifest** (the structured-output path that uses `json_constrain.rs`, which §4.3 extends). Tool output enters context as `provenance=tool-output` (TT8).
- **Ch.05 · Codebase Intelligence** — owns the **§4.11 query API** that this chapter's `find_definition`/`find_references`/`find_callers`/`find_implementations`/`path_between`/`tests_covering`/`changed_since`/`grep_symbol`/`dataflow_paths`/`taint_check` and the `edit.ast`/`refactor.*` tools are **thin wrappers over** (§4.6.4/§4.6.7). Tools must never re-parse or walk the FS — they query the Living Index.
- **Ch.06 · Model Layer** — owns the sampler/grammar kernel and runtime endpoints; this chapter's constrained-decode design (§4.3) extends the runtime's `json_mode`/`JsonConstraint` seam and reserves the `hide_constraint` request surface; the Condense fine-tune-the-protocol idea (§4.3.3) and skill-distillation moonshot (§8) couple here. Deep model hooks are **[LATER]**.
- **Ch.10 · Local-First Security** — **canonical owner** of the OS sandbox (Seatbelt/bubblewrap/VM tiers), the capability/permission model, and prompt-injection defense. §4.9 here is the **tool-side surface** that references and extends Ch.10; **Ch.10 wins any enforcement conflict**. The lethal-trifecta gate, network policy, secret deny-lists, and `code.exec`/computer-use isolation tiers all defer to Ch.10's canonical policy.
- **Ch.08 · Research & Knowledge Lab** — consumes `web.fetch`/`web.search` (§4.6.9) as untrusted-content sources behind the trifecta gate.
- **Ch.09 · UI / Context Stack** (where applicable) — renders the per-call cards, the causal DAG, the dry-run preview, and the live PTY/terminal from the §4.12 event stream.

---

*End of Chapter 03. The tool wire-format (§4.2) and the permission-policy schema (§4.9.2) are the binding contracts; everything else is faithful elaboration. To add a tool — built-in, plugin, MCP, or agent-authored — nothing under `core/` changes. Small local models emit valid calls by construction, full-OS tools operate on the real machine under scoped capabilities, and the agent grows a durable, private, verified toolbox no cloud agent can match.*
