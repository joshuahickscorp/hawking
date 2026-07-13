# Agentic Tool System — Research + Build Plan (2026-07-11)

**Goal.** Make Hawking/HIDE the best *and* fastest local agentic coding runtime: the
most extensive, best-designed tool surface (ripping the strongest ideas from Claude
Code and OpenAI Codex), delivered on a serving path engineered so that tool calls are
(a) always schema-valid and (b) mostly *free* — via a small structured-speculation /
jump-forward decode layer that exploits how predictable tool-call syntax is.

**One-line thesis.** You are not missing an agent framework — you already have a
deterministic agent FSM, a permission-gated dispatcher, ~22 real tools, a full MCP
client, per-token JSON masking, and mature spec-decode (EAGLE5/n-gram/suffix). The
five things that are actually missing are all *connections*, not new inventions:
1. a **model-output → tool-call parser** (there is none today),
2. **native tool support on the serve API** (no `tools` field, no `tool_calls` out),
3. **schema-aware constrained decoding on the batched serve lanes** (masking is wired
   only into the single-sequence `forward()` path),
4. a **tool-aware speculative/jump-forward layer** (spec-decode is dormant on serve),
5. **wiring the already-built MCP client, ACI lint, idempotency ledger, and parallel
   dispatch** that sit in the tree uncalled.

Do those five and Hawking is at frontier parity on capability and *ahead* on latency,
because the "spec-decode layer for the tools" the user asked about is a real,
published, ~4× lossless technique (ToolSpec, arXiv 2604.13519) that maps almost 1:1
onto primitives you already have.

---

## Part I — The frontier (research synthesis)

Sources are cited inline. Anthropic-reported metrics are flagged as vendor numbers.

### I.1 Claude Code's tool architecture — what to rip

**The built-in surface** (canonical list: Claude Code [Tools reference](https://code.claude.com/docs/en/tools-reference)):

- **File I/O + search:** `Read` (numbered output, paginates over the token cap with a
  `PARTIAL view` notice + `offset`/`limit`, reads images/PDF/ipynb), `Write` (whole-file,
  overwrite requires prior Read), `Edit` (exact string replace with **three gates**:
  read-before-edit, exact match, uniqueness-or-`replace_all`), `Glob` (mtime-sorted,
  **100-file cap** + truncation flag), `Grep` (ripgrep-backed, output modes
  files/content/count, respects `.gitignore`), `LSP` (post-edit diagnostics + jump/refs),
  `NotebookEdit`.
- **Execution:** `Bash` (cwd persists within project; **env does not persist**, shell
  rc aliases do; 30k-char output cap → spill-to-file + preview; `run_in_background`),
  `Monitor` (stream a long-running command's lines back mid-conversation).
- **Web:** `WebFetch` (HTML→Markdown→**summarized by a small fast model**, lossy by
  design, 15-min cache), `WebSearch` (titles+URLs only, up to 8 backend searches/call,
  domain allow/block lists).
- **Orchestration:** `Agent` (subagent, fresh context, **only final text returns**),
  `Workflow` (script orchestrating dozens–hundreds of agents), `Skill`, the `Task*`
  task-list tools (TodoWrite disabled-by-default now), `ToolSearch` (deferred-tool
  loading), MCP resource tools.

**The design doctrine** — Anthropic's [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
and [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents):
- A tool is *"a contract between deterministic systems and non-deterministic agents"* —
  design for the model's ergonomics, not a programmer's.
- **Consolidate around workflows**, don't wrap every endpoint (`schedule_event`, not
  `list_users`+`list_events`+`create_event`).
- **Namespace** tools (`asana_search`, `asana_projects_search`) — naming measurably
  moves eval scores. (Your `fs.*`/`edit.*`/`git.*` scheme already does this.)
- **Return high-signal fields**, resolve opaque UUIDs to names.
- Give a **token-efficiency dial** (`response_format: concise|detailed`), default to
  hard caps + pagination + spill (Claude Code caps tool results at ~25k tokens).
- **Error messages are a steering surface** — say what's wrong and how to fix it so the
  agent self-corrects. (Your `ToolError { retriable, fix_hint, schema_path }` is exactly
  this — you're ahead here.)
- **Poka-yoke** the argument shapes; **eval-drive** the tools, then let the model
  refactor them.

**Parallel + streaming** ([parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use),
[fine-grained streaming](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/fine-grained-tool-streaming)):
multiple `tool_use` blocks in one assistant turn, executed concurrently, all
`tool_result`s returned together; `tool_choice` = auto/any/tool/none with
`disable_parallel_tool_use`. Fine-grained streaming emits `input_json_delta`
fragments (unvalidated) so large tool args start executing sooner.

**Subagents as context firewalls** ([SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)):
fresh window, the **only** parent→child channel is the Agent-tool prompt string, only
the final message returns, per-agent tool allowlist + model override, resumable
transcripts stored separately (survive compaction). *(You already have
`subagent::spawn_and_join` returning summary-only — same shape.)*

**MCP** ([announcement](https://www.anthropic.com/news/model-context-protocol), spec rev
2025-11-25): host runs one client per server over JSON-RPC 2.0; servers expose
**tools/resources/prompts**; **stdio** transport for local, **Streamable HTTP** for
remote. Supporting the client = inherit the entire server ecosystem for free.

**2025–26 frontier features** (the genuinely new levers, from
[Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) and
[Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)):
- **Tool Search / deferred loading** (`defer_loading: true`): only a ~500-token search
  tool loads upfront; the model searches the catalog and expands the few tools it needs.
  Vendor-reported ~85% tool-token reduction, and it's **prompt-cache-friendly** (deferred
  defs are stripped before the cache key). This is exactly how *this* session's tools work.
- **Programmatic Tool Calling (PTC) / code-execution-over-MCP**: the model writes code
  that calls tools as functions in a sandbox; intermediate results stay in the execution
  env, not the context. Vendor-reported 98.7% token cut on a Drive→Salesforce pipeline.
- **Tool Use Examples** (`input_examples`): realistic call samples in the tool def;
  vendor-reported 72%→90% on complex params.
- **Memory tool** (`memory_20250818`, GA): client-side file tool over `/memories`
  (view/create/str_replace/insert/delete/rename) with mandatory path-traversal guards.
- **Agent Skills** (`SKILL.md` + 3-level progressive disclosure), context-management
  primitives (compaction, structured note-taking, just-in-time retrieval).

### I.2 OpenAI Codex, function calling, and other agents — cross-checks

*(Verified against primary repos/docs/papers; benchmark multipliers are author/vendor
numbers on NVIDIA/AMD hardware — evidence the overhead is removable, not figures to
reproduce on Metal.)*

- **Codex CLI** ([openai/codex](https://github.com/openai/codex), Rust): a **small, sharp**
  tool surface — `shell_command` / `exec_command` (PTY, returns a `session_id` for ongoing
  interaction) / `write_stdin`, plus **`apply_patch`**, `update_plan`, `view_image`,
  `web_search`, `request_permissions`, and MCP tools. Two things to steal:
  - **The `apply_patch` format has no line numbers** (`*** Begin Patch` / `*** Add|Update|
    Delete File:` / optional `*** Move to:` / `@@ func-header` hunks with ` `/`-`/`+` lines).
    Context is located by a **4-pass fuzzy matcher** (`seek_sequence`): exact → ignore
    trailing whitespace → ignore leading+trailing → Unicode-normalize (curly quotes/dashes →
    ASCII). It's exposed to strong models as a **freeform "custom" tool constrained by a Lark
    grammar** (not JSON) and streamed as `custom_tool_call_input.delta` to drive **live patch
    previews**. This is the single most important edit-tool design decision — your
    `edit.apply_patch` should adopt the line-number-free format + fuzzy applier.
  - **Two orthogonal axes:** `approval_policy` × `sandbox_mode` (read-only / workspace-write /
    danger-full-access), with **per-command escalation** + justification, enforced by OS
    sandboxes (macOS **Seatbelt** `sandbox-exec`, deny-by-default). Your permission engine
    already models this shape; add the per-call escalation directive.
- **OpenAI function calling / Responses API** ([guide](https://developers.openai.com/api/docs/guides/function-calling)):
  `tools` with JSON-Schema functions, **parallel function calling**, and **Structured
  Outputs** (`strict:true`). Strict mode's mechanism *is* Part I.3: the schema is compiled to
  a **CFG, cached, and used to mask invalid next-tokens to probability 0** → reported **100%
  conformance** (vs ~93% from training alone). Subset rules: **root object,
  `additionalProperties:false` everywhere, every property in `required`** (optional = union
  with `null`). **Gotcha:** strict schema is **not applied during parallel tool calls** — gate
  parallelism per call if you need both. GPT-5 also has **freeform custom tools with a
  grammar** (`format:{type:"grammar", syntax:"lark"|"regex"}`) — the most transferable
  primitive for emitting constrained code/DSL locally.
- **Aider** ([repo-map](https://aider.chat/2023/10/22/repomap.html),
  [diffs](https://aider.chat/docs/unified-diffs.html)): a **repo-map** (tree-sitter +
  **personalized PageRank** boosting in-chat files/symbols, binary-searched to fit a token
  budget) and an **edit-format ladder** (`whole` / `diff` SEARCH-REPLACE / `diff-fenced` /
  `udiff`). Findings: unified diffs make GPT-4 Turbo **3× less lazy** (20%→61%); **disabling
  fuzzy patching = 9× more edit errors**; asking for whole functions cuts errors 30–50%. Give
  the model a *map*, not the repo; match the format to model strength.
- **SWE-agent's ACI** ([paper](https://arxiv.org/abs/2405.15793)): *interface design
  dominates model choice*. A **windowed ≤100-line viewer** (18.0% vs 12.7% full-file), a
  **lint-on-every-edit that rejects broken writes** (18.0% vs 10.3%), and **summarized
  search** (a *bad* search tool scored 12.0%, **worse than no search at 15.7%**). Your unwired
  `lint_tool_call` is exactly this guardrail.
- **OpenHands CodeAct** ([paper](https://arxiv.org/abs/2402.01030)): actions as **executable
  code** beat a JSON-tool menu by ~20% success / ~30% fewer steps — the empirical case for
  Programmatic Tool Calling (Part I.1).
- **Cline/Roo:** XML tool calls, **one tool per message**, explicit **Plan vs Act** modes (a
  read-only gate before mutation). **Cursor's instant-apply** is the standout latency idea
  (see I.4). Net across all of these: the differentiators are **edit reliability,
  repo-mapping, and latency — not tool count.**

**Takeaway:** the capability frontier is a **small, sharp, well-described, guard-railed**
tool set + **MCP** for breadth + **strict schema-valid args**. You are ~80% there. The
*differentiator* you can own is **latency**, via Part I.3–I.4.

### I.3 Constrained / grammar-constrained decoding — guaranteed-valid tool calls

Mechanism: at each decode step, compile the target grammar/schema into a mask over the
vocabulary and set logits of tokens that can't continue a valid string to `-inf` before
sampling. Nothing invalid can ever be emitted.

- **GBNF (llama.cpp)**, **Outlines** (regex/JSON-Schema → FSM, O(1) per-state token-set
  lookup), **XGrammar**, **LM Format Enforcer** (pure logit filter, batch-friendly),
  **guidance** are the references. The design to copy is **XGrammar**
  ([paper](https://arxiv.org/html/2411.15100v1)): split the vocabulary into
  **context-independent** tokens (validity depends only on local automaton position —
  >99%, precomputed) vs **context-dependent** (**<1%; only ~1134/128k for Llama-3.1 JSON**,
  checked live), and — critically — **run mask generation on a CPU thread overlapped with
  the GPU/Metal forward pass** so masking is ~free (<40µs/mask). For a Rust engine,
  **llguidance** ([repo](https://github.com/guidance-ai/llguidance)) is a candidate
  drop-in constraint backend rather than hand-rolling the pushdown automaton.
- **Quality guard** (["Let Me Speak Freely?"](https://arxiv.org/html/2408.02442v1)): strict
  formats can degrade *reasoning*. Mitigation: **put a free-text reasoning field before the
  constrained tool-call fields** so chain-of-thought is never masked away.
- Your runtime already has the primitive: `json_constrain::JsonConstraint` +
  `JsonVocabIndex::mask_logits` sets invalid tokens to `-inf`. It's just (a) *binary*
  (generic JSON, not schema-specific) and (b) wired only into single-sequence `forward()`,
  not the batched serve lanes.

The upgrade: compile each tool's `input_schema` (already on every `ToolSpec`) into a
per-call grammar, and — the outer envelope — constrain the *tool-selection* step to the
**finite set of registered tool names**. That alone eliminates hallucinated tool names and
malformed args, killing the validate-and-retry round-trips.

### I.4 Speculative decoding for tools — the "small spec-decode layer" (core)

This is the piece the user specifically flagged, and it's real and published.

**Spec-decode basics** ([Leviathan et al. 2211.17192](https://arxiv.org/abs/2211.17192),
[Chen et al. 2302.01318](https://arxiv.org/abs/2302.01318)): a cheap **drafter** proposes
k tokens, the target model **verifies them in one batched forward pass**, and a lossless
accept/reject rule (accept token i with prob `min(1, p_i/q_i)`, else resample from
`max(0, p−q)`) guarantees the output distribution is **identical** to plain decoding.
2–3× on general text.

**Why tool calls are the ideal spec-decode target** — a tool call is
~60–80% *deterministic given the schema*:
- **Structural scaffolding is forced**: after choosing a tool, `{"` + first key name +
  `":"` etc. are often the *only* legal continuation under the grammar → emit them with
  **zero forward passes** ("jump-forward" / "fast-forward" decoding, as in SGLang's
  compressed FSM and XGrammar).
- **Tool names are a finite set** → trivially drafted / constrained.
- **Argument *values* are frequently copied** from context (a file path just read, a
  symbol from the diff) → **prompt-lookup / n-gram speculation** nails these (2–4× on
  input-grounded tasks; the standard win for code editing where the model copies existing
  lines).
- **The whole edited file is drafted by the original.** Cursor's instant-apply
  ([writeup](https://cursor.com/blog/instant-apply)) feeds the **original file as the
  speculative draft** so only the *divergences* from the original cost real generation —
  a fine-tuned model hit **~1000 tok/s (~13×)** on whole-file rewrites. This "original-as-
  draft" trick is directly reproducible on your existing spec-decode runtime and is the
  highest-leverage latency win for an `apply_patch`/whole-file edit tool.

**The direct hit — ToolSpec** ([arXiv 2604.13519](https://arxiv.org/html/2604.13519v2)):
combines a **schema-FSM** (jump over forced scaffolding) with **retrieval/prompt-lookup
speculation** for the free-text values, verified losslessly by the target. Reports
**~3.5–4.2×** speedup on tool-calling workloads. This is, almost exactly, the "small spec
decode layer for the tools" the user described — and it composes cleanly with your
existing EAGLE5/exact-shared/suffix-automaton drafters.

**Speculative tool *execution* / prefetch** ([2512.15834](https://arxiv.org/abs/2512.15834),
PASTE [2603.18897](https://arxiv.org/abs/2603.18897)): once the tool name + args are
predicted with high confidence (or forced by grammar), begin running the tool *before* the
model has finished emitting — hiding tool latency behind decode. **Hard safety rule: only
speculatively execute pure / read-only / idempotent tools** (never a `shell.run` that
mutates, never a `git.commit`). Your `Tool::purity` (`Pure`/`PureFs`/`Impure`),
`Tool::simulate` (predicts `EffectSet` with no side effects), `ToolCall.x.dry_run`, and the
unwired `IdempotencyLedger` are *precisely* the safety primitives this needs.

**The synthesis (cheapest-first layering), all training-free and lossless:**
1. **Grammar constraint** on the tool envelope + args (Part I.3) — guarantees validity.
2. **Jump-forward** over grammar-forced tokens — free scaffolding, no forward pass.
3. **Prompt-lookup / n-gram** drafting for argument values copied from context.
4. *(optional, later)* schema-FSM retrieval speculation (ToolSpec-style) and your existing
   EAGLE5 drafter for the free-text spans.
5. *(optional, later)* speculative **execution** of read-only tools behind the decode.

Layers 1–3 are the minimal high-ROI build and reuse code you already have.

### I.5 Fast multi-tool loops — the surrounding latency wins

- **KV-cache reuse across agent turns** (prefix caching — you have prior art in
  `docs/plans/research/` and shipped prefix-cache work): the system prompt + tool defs +
  early conversation are stable; cache them so each agent turn only prefills the new suffix.
- **Prompt-cache-friendly tool layout** + **deferred tool loading** so a big catalog/MCP
  set doesn't blow the prefill or invalidate the cache.
- **Parallel tool execution** of independent read-only calls (join, don't serialize).
- **Tool-result truncation/spill** with "N more, refine your query" hints (context
  defense — Claude Code's 25k cap pattern).

---

## Part II — Where Hawking/HIDE stands today (grounded)

*(File:line references verified against the tree.)*

**Strong, already-built:**
- **Tool core** — `hide-core/src/tool.rs`: `ToolSpec` (MCP-shaped, carries `input_schema`),
  `ToolCall` (with `x.dry_run`/`idempotency_key`/`timeout_override`), self-correcting
  `ToolError { retriable, fix_hint, schema_path }`, `Tool` trait with `simulate()` +
  `purity()`, permission-gated `ToolDispatcher` (`tool.rs:271-349`).
- **~22 built-in tools** — `hide-tools/src/registry.rs`: `fs.*`, `edit.{search_replace,
  apply_patch,write_file}` (tiered verifying applier w/ `base_hash` optimistic
  concurrency), `shell.{run,plan}`, `test/build/compile`, `search.text`, `git.*` +
  worktree trio.
- **A real agent FSM** — `hide-kernel/src/machine/driver.rs`: Intake→Plan→SelectStep→Act→
  Observe→Verify→{Repair|Replan|Paused}→Finalize, oracle-gated verify, governor on every
  transition, stall/convergence detection; **subagents** `subagent::spawn_and_join`
  (summary-only, depth-capped).
- **Complete MCP client** — `hide-tools/src/mcp.rs` (rev 2025-11-25, stdio + Streamable
  HTTP, initialize/tools/list/tools/call, proxied as untrusted-provenance tools).
- **Per-token JSON masking** — `hawking-core/src/json_constrain.rs` (`mask_logits` → `-inf`).
- **Mature spec-decode** — `hawking-core/src/speculate/*` (EAGLE5, exact-shared, n-gram/
  user-ngram, suffix automaton/array, retrieval, tree, verifier, governor) + a serve-side
  `SpecGovernor` (`hawking-serve/src/spec_gov.rs`).
- **OpenAI-compatible serve** — `hawking-serve/src/http.rs`: `/v1/chat/completions`,
  `/v1/completions`, native `/v1/hawking/generate`, SSE streaming, continuous batching.

**The five real gaps (all connections):**
1. **No tool-call parser.** Nothing turns model output into a `ToolCall`; the agent only
   acts on pre-authored `PlanStep.tool_hint`. (Grep: no `tool_calls`/`<tool_call>`/
   `parse_tool_call` anywhere.)
2. **Chat path is single-shot text.** `SubmitTurn` → `generate_submit_turn` does one
   `runtime.generate` and streams tokens (`host.rs:801-972`) — the FSM/tools loop exists
   but isn't on the chat turn. Serve `ChatReq` has **no `tools` field** and emits no
   `tool_calls`.
3. **Constrained decode is off the serve path.** `json_mode` masking runs only in
   single-seq `forward()`; the batched lanes `forward_multiseq_*` hardcode
   `json_mode:false` (`batch/driver.rs:209`, `batch/scheduler.rs:520`). `InferenceRequest.
   grammar` collapses to a boolean en route and is dropped by the provider.
4. **Spec-decode is off the serve path.** Live in single-seq `forward()`
   (deepseek_v2/qwen_dense), dormant on `forward_multiseq_*`; `SpecGovernor` ships but the
   batch lanes never invoke spec.
5. **Built-but-unwired safety/extension seams:** `lint_tool_call` + `IdempotencyLedger`
   (`hide-kernel/src/tools/mod.rs`) have no callers; the MCP client is registered nowhere;
   parallel dispatch is unused (`do_act` takes only `ready_steps().next()`).

---

## Part III — Strategy

Two tracks, run in parallel; they meet at the serve path.

- **Track A — Capability (the "most extensive tool list"):** give the local model *native*
  tool calling (parse loop + serve `tools` support), close the tool-catalog gaps vs
  Claude/Codex (web, subagent-as-tool, plan/todo, memory, notebook, batch), and wire the
  MCP client so the catalog is open-ended. Wire the ACI lint + idempotency guardrails.
- **Track B — Speed (the "fastest"):** move constrained + speculative decoding onto the
  batched serve lanes, upgrade constraint from binary-JSON to **schema/tool-grammar-aware**,
  and add the **jump-forward + prompt-lookup tool-spec layer** so most tool-call tokens are
  free. Add prefix-cache reuse across agent turns and parallel read-only tool execution.

**Design laws (adopt as house rules for tools):** consolidate around workflows; namespace;
high-signal results with hard caps + spill; instructive errors (already have the type);
poka-yoke args; eval-drive everything. Keep the HIDE voice/telemetry rules (memory
`feedback-hide-house-rules`).

---

## Part IV — The build plan (phased, concrete)

Each phase lists the seam, the work, and how to verify. Phases 0–2 are the keystone;
3–4 are the latency moat; 5–6 are polish + defensibility.

### Phase 0 — The tool-call protocol: parse → lint → dispatch → feed-result  *(keystone)*
- **Seam:** `hide-kernel/src/machine/driver.rs` `do_act` / `act_tool`; the unwired
  `lint_tool_call` + `IdempotencyLedger` in `hide-kernel/src/tools/mod.rs`.
- **Build:** a `tool_call` parser that extracts calls from model output. Pick a format the
  local models emit reliably (recommend **OpenAI-style `tool_calls` JSON**, plus a
  tolerant `<tool_call>{...}</tool_call>` fallback — decide by measuring your target
  models). Loop: parse → `lint_tool_call` (reject empty/unknown-tool/args-not-object/
  hallucinated-path with a `fix_hint`) → `IdempotencyLedger.lookup` (dedup) → dispatch →
  append `tool.result` observation → continue. Reuse the existing `ToolError` fields for
  self-correction re-prompts.
- **Verify:** unit tests on malformed/valid/duplicate calls; an integration test where a
  local model completes a 3-tool task (read → edit → test) end-to-end.

### Phase 1 — Native tool calling on the serve API + catalog expansion
- **Seam:** `hawking-serve/src/http.rs` `ChatReq`/`chat_completions`; `render_chat`;
  `hide-tools/src/registry.rs`.
- **Build:**
  - Add `tools`, `tool_choice`, and `tool_calls`/`tool` message roles to `ChatReq`;
    render tool defs into the prompt per arch template; emit `tool_calls` in the response
    (and as SSE deltas). This makes Hawking a drop-in OpenAI-tools server.
  - **Close the catalog gaps** (new `Tool` impls in `hide-tools`): `web.fetch` (fetch →
    HTML→md → summarize with the local model, lossy like WebFetch) and `web.search`;
    `agent.spawn` (subagent-as-tool over the existing `spawn_and_join`); `plan.todo`
    (task-list tool); `memory.*` (client-side file tool over a `/memories` dir with
    **path-traversal guards** — non-negotiable); `notebook.edit`; a batch/multi-edit
    convenience. Give each `input_examples` (Anthropic's Tool-Use-Examples pattern).
  - **Wire the MCP client:** load `McpServerDescriptor`s from HIDE config, register
    discovered `McpProxyTool`s into the registry at startup. Instant open ecosystem.
  - **Upgrade `edit.apply_patch` to Codex-grade:** line-number-free envelope + a **4-pass
    fuzzy `seek_sequence` applier** (exact → ignore trailing ws → ignore leading+trailing →
    Unicode-normalize), a **whole-file-rewrite fallback** for weaker models (Aider's ladder),
    and **reject-on-lint** (wire `lint_tool_call`; a broken write is refused with a `fix_hint`,
    not applied — the SWE-agent guardrail worth ~+3–8 pts). Keep the existing `base_hash`
    optimistic-concurrency check.
- **Verify:** `curl` the serve `/v1/chat/completions` with a `tools` payload and confirm a
  well-formed `tool_calls` response; a test MCP stdio server's tool shows up and dispatches;
  an intentionally-broken patch is rejected with a fix hint, not written.

### Phase 2 — Schema-constrained decoding on the batched serve lanes  *(guaranteed-valid)*
- **Seam:** `hawking-core/src/json_constrain.rs`; `forward_multiseq_greedy_tokens` /
  `forward_multiseq_batched` in `batch/driver.rs`; `InferenceRequest.grammar`;
  `hawking-orch/src/grammar.rs` (the "[RUNTIME-SIDE — LATER]" note).
- **Build:**
  - Replace the boolean `json_mode` wire with a real `grammar: Option<GrammarSpec>` carried
    end-to-end (stop collapsing it to a bool in `http_client.rs` / dropping it in
    `model_provider.rs`).
  - Apply `mask_logits` inside the `forward_multiseq_*` lanes (per-sequence constraint
    state), so constrained decode works under continuous batching — not just single-seq.
  - Compile a tool's `input_schema` → grammar; constrain the **tool-name** step to the
    registered finite set. Keep the shell-side `validate`+`RetryHint` as a backstop only.
- **Verify:** property test — under constraint, a fuzzed decode can *never* emit an invalid
  tool name or malformed args JSON; measure that retry-round-trips drop to ~0.

### Phase 3 — The tool spec-decode / jump-forward layer  *(the latency moat)*
- **Seam:** `hawking-core/src/speculate/*` (n-gram/prompt-lookup, suffix automaton,
  verifier, governor); the Phase-2 grammar FSM; `SpecGovernor`.
- **Build (cheapest-first, all lossless):**
  1. **Jump-forward:** when the grammar FSM has a single legal continuation (forced
     scaffolding), emit those tokens with **no forward pass**. Free 40–60% of tool-call
     tokens.
  2. **Prompt-lookup drafting** for argument *values* copied from context (paths, symbols,
     lines just read) — reuse your n-gram/suffix drafter, verified by the target's batched
     pass with the standard `min(1,p/q)` rule. 2–4× on the copied spans.
  3. **Original-file-as-draft for edits (biggest single win):** for `apply_patch`/whole-file
     rewrites, feed the **original file as the speculative draft** so only the divergences
     cost real generation (Cursor's instant-apply, ~13×). Reuses the same verify loop; the
     draft source is just "the file already in context" instead of a draft model.
  4. Route all of the above through the existing `verifier` + `SpecGovernor` (auto-disable
     when accept rate drops). *(Later: ToolSpec-style schema-FSM retrieval speculation +
     EAGLE5 for free-text spans.)*
- **Verify:** measure tokens-emitted-per-forward-pass on a tool-heavy trace; target ≥2×
  effective decode throughput on tool-call spans vs unconstrained; assert **bit-identical**
  output to plain constrained decode (losslessness gate — you have prior discipline here,
  memory `eh-verify-kernel-not-lossless`: verify at near-ties).

### Phase 4 — Parallel + speculative tool execution
- **Seam:** `do_act`/`ready_steps` in `driver.rs`; `Tool::purity`, `Tool::simulate`,
  `dry_run`, `IdempotencyLedger`.
- **Build:**
  - **Parallel dispatch** of independent ready steps via `FuturesUnordered` over
    `Arc<ToolDispatcher>`, gated by `purity` (only `Pure`/`PureFs`/read-only run
    concurrently; `Impure` serialize).
  - **Speculative execution / prefetch:** once a tool call is grammar-forced or
    high-confidence, start **read-only** tools before generation finishes; discard if the
    committed call differs. **Never** prefetch a destructive/side-effecting tool — enforce
    via `annotations.destructive` + `purity`; the `IdempotencyLedger` backstops re-runs.
- **Verify:** a task with 3 independent reads runs them concurrently (wall-clock ≈ slowest,
  not sum); a fuzz test proves no `Impure` tool is ever speculatively executed.

### Phase 5 — Context/token defenses + agent-loop-on-chat
- **Build:** deferred tool loading (`ToolSearch`-style) so large MCP catalogs stay
  prompt-cache-friendly; result caps + spill-to-CAS with "N more" hints (you have the blob
  store); prefix-cache reuse across agent turns; and route the interactive chat turn
  through the `hide-kernel` FSM (with tools) instead of single-shot generate — behind a
  flag, measured.
- **Verify:** cache-hit rate across turns; tool-token budget stays flat as catalog grows.

### Phase 6 — Evals + the moat
- **Build:** an agentic tool-use eval harness (multi-call tasks with verifiable outcomes,
  à la Anthropic's method) capturing accuracy / tool-call-count / tokens / latency; wire
  into `hawking-eval`. Then let the model refactor its own tool descriptions against it.
- **Moat:** the combination that's hard to copy = **local + native-tool + schema-guaranteed
  + jump-forward-fast**, i.e., a private coding agent whose tool calls are *both* always
  valid *and* mostly free to emit. That's the "dominate the field" position — ground it in
  measured latency, not prose (memory `feedback-reach-for-more`, `mop-gold-prompt`).

---

## Part V — Sequencing, risks, thesis

**Critical path:** Phase 0 → 1 → 2 unlocks *native, guaranteed-valid tool calling on the
serve API* — that is the minimum to be "agentic" and correct. Phase 3 is where "fastest"
is won and is the user's specifically-requested spec-decode-for-tools layer; it depends on
Phase 2's grammar being real. Phases 4–6 compound.

**Do-first (highest ROI, mostly wiring):** Phase 0 parse loop + `lint`/`idempotency`;
Phase 1 MCP-client registration + serve `tools` field; Phase 2 move masking to
`forward_multiseq_*` + finite tool-name constraint; Phase 3 jump-forward + prompt-lookup.

**Risks / watch-items:**
- **Losslessness:** every spec/jump layer must be bit-identical to plain constrained decode
  — gate with a parity test (your standing discipline; the near-tie verify hazard is real).
- **Batched constraint state:** per-sequence FSM state under continuous batching is the
  fiddly part of Phase 2 — budget for it.
- **Tool-call format choice** must be validated against *your* target models empirically,
  not assumed.
- **Speculative execution safety** is a correctness/security boundary — enforce
  purity/destructive gates in code, fuzz it, and keep it opt-in.
- **Vendor metrics** (Anthropic's %s, the ~4× ToolSpec number) are *reported*, not
  reproduced here — treat as direction, measure your own.

**Why this wins.** Everyone can bolt more tools onto a chat model. Almost no one ships a
*local* agent where the tool calls are simultaneously (1) schema-guaranteed valid and
(2) emitted at 2×+ effective throughput because the deterministic 60–80% of every tool call
is filled for free. You already own every hard primitive — the FSM, the dispatcher, the
mask, the drafters, the MCP client. The work is connection and measurement, not invention.

---

### Appendix — key file seams (quick index)
- Tool core / dispatcher: `crates/hide-core/src/tool.rs`
- Builtin catalog: `crates/hide-tools/src/registry.rs`, impls in `hide-tools/src/{fs,edit,shell,proc,search,git}.rs`
- MCP client (unwired): `crates/hide-tools/src/mcp.rs`
- Agent FSM + subagents: `crates/hide-kernel/src/machine/driver.rs`, `hide-kernel/src/subagent/mod.rs`
- ACI lint + idempotency (unwired): `crates/hide-kernel/src/tools/mod.rs`
- Serve API + batching: `crates/hawking-serve/src/http.rs`, `hawking-serve/src/batch/{driver,scheduler}.rs`
- Constrained decode: `crates/hawking-core/src/json_constrain.rs`, `hawking-orch/src/grammar.rs`
- Spec-decode: `crates/hawking-core/src/speculate/*`, `hawking-serve/src/spec_gov.rs`
- Prior local research: `docs/plans/research/c_loop_speculation.md`, `f4_grammar_cache_deep.md`

### Appendix — primary sources
- Claude Code tools: https://code.claude.com/docs/en/tools-reference
- Writing tools for agents: https://www.anthropic.com/engineering/writing-tools-for-agents
- Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- Advanced tool use (search/PTC/examples): https://www.anthropic.com/engineering/advanced-tool-use
- Code execution with MCP: https://www.anthropic.com/engineering/code-execution-with-mcp
- Subagents (SDK): https://code.claude.com/docs/en/agent-sdk/subagents
- MCP: https://www.anthropic.com/news/model-context-protocol
- Parallel tool use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
- Spec-decode: https://arxiv.org/abs/2211.17192 , https://arxiv.org/abs/2302.01318
- ToolSpec (tool spec-decode, ~4×): https://arxiv.org/html/2604.13519v2
- Speculative tool execution / PASTE: https://arxiv.org/abs/2512.15834 , https://arxiv.org/abs/2603.18897
- Prompt-lookup / n-gram decoding: https://github.com/apoorvumang/prompt-lookup-decoding
- Medusa / EAGLE: https://arxiv.org/abs/2401.10774 , https://arxiv.org/abs/2401.15077 , https://arxiv.org/abs/2406.16858 , https://arxiv.org/abs/2503.01840
- SGLang compressed-FSM jump-forward: https://www.lmsys.org/blog/2024-02-05-compressed-fsm/
- XGrammar (CPU/GPU-overlap masking): https://arxiv.org/html/2411.15100v1
- llguidance (Rust constraint backend): https://github.com/guidance-ai/llguidance
- OpenAI function calling / Structured Outputs: https://developers.openai.com/api/docs/guides/function-calling , https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI Codex (repo): https://github.com/openai/codex
- Aider (repo-map + diffs): https://aider.chat/2023/10/22/repomap.html , https://aider.chat/docs/unified-diffs.html
- SWE-agent ACI: https://arxiv.org/abs/2405.15793
- OpenHands CodeAct: https://arxiv.org/abs/2402.01030
- Cursor instant-apply (original-file-as-draft, ~13×): https://cursor.com/blog/instant-apply
- Prefix/KV caching: https://docs.vllm.ai/en/stable/design/prefix_caching/ , https://arxiv.org/abs/2312.07104 (SGLang RadixAttention)
- Prompt caching (ordering + numbers): https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Reasoning-vs-format caveat: https://arxiv.org/html/2408.02442v1
