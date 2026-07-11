# Agentic Tool System: scaffold audit + rating (2026-07-11)

Companion to `docs/plans/agentic_tool_system_2026_07_11.md` (the research + build plan) and
the run report `M1ULTRA_RUN_REPORT.md`. This grades what the scaffold wave actually landed,
out of 10, under the program's proof discipline: a unit test on this box is MEASURED-LAB
(author-rerunnable, below R3), never a public WIN; anything needing a live model to validate
is GATED until we have our own `.tq`-served model to run it against. House style: no em or en
dashes. Nothing here is claimed above the evidence.

## 0. What "scaffold all" produced this wave

Scaffolded as real, compiling, unit-tested library code, highest-leverage-first (the
keystone, then the differentiator). All are model-independent, so they are testable now even
though we do not yet have our own served model. Deep runtime integration (serve wiring,
live-model validation) is deliberately left for the drive-to-10 step, per the operator's
scaffold -> test -> audit -> 10/10 sequence.

| Component | File | Tests | Grade (this box) |
|---|---|--:|---|
| Phase 0: tool-call parser | `hide-kernel/src/tools/parse.rs` | 14 | MEASURED-LAB |
| Phase 0: parse->lint->dedup->dispatch loop + parallel | `hide-kernel/src/tools/runner.rs` | 10 | MEASURED-LAB |
| Phase 0: lint + idempotency (shared) | `hide-kernel/src/tools/mod.rs` | 2 | MEASURED-LAB |
| Phase 2/3: tool-call grammar (jump-forward) + prompt-lookup | `hawking-orch/src/tool_spec_decode.rs` | 13 | MEASURED-LAB |

Owned scaffold tests green: 47 (parse 14 + runner 10 + shared mod.rs 2 + tool_spec_decode 13 +
memory 8). Catalog expanded: the `memory` tool (durable cross-session scratchpad, 23rd builtin)
lands with a path-traversal guard tested against absolute / `..` / percent-encoded escapes.
Full workspace: 392 passed, 7 failed; the 7 are pre-existing `q8_kv_parity` tests in
`hawking-core` failing on Metal kernels (`kv_append_q8_0_f32`, `mla_decode_kernel_q8kv`) that
are absent from every `.metal` source in the repo. That crate was never touched this session,
so those failures are pre-existing on this branch, not caused by the scaffold. Zero non-q8kv
failures.

WITHDRAWN: Phase 1d (the Codex-grade fuzzy `apply_patch` / `seek_sequence`) was applied and
tested green, then reverted intentionally (the working tree returned `edit.rs` to its original
`locate`-based applier). It is therefore not part of the delivered scaffold and is not graded
here. The adversarial review of the reverted-back original `apply_patch` did surface two
pre-existing bugs in it (a wrap-around slice panic and an unverified `-` line drop that can
silently corrupt a file); see section 3.1. Those live in `edit.rs`, which is under separate
management, so they are reported not fixed.

## 0.1 Adversarial verification (this wave)

A five-dimension review workflow (finder + independent refuting verifier per dimension, 11
agents) hunted the scaffold for correctness, losslessness, safety, and overclaims. It
confirmed six real defects, each reproduced end to end. Four were in the delivered scaffold and
are now FIXED with regression tests; two are in the reverted-back `edit.rs` original and are
reported not fixed.

| # | Dimension | Defect | Status |
|--:|---|---|---|
| 1 | parser | a leading `[...]` (markdown link / citation) shadowed a bare-JSON call, dropping it | FIXED (all-spans scan; 3 tests) |
| 2 | runner | untrusted tool output could forge a `<tool_call>` in feedback (TT8 violation) | FIXED (envelope escaping; 2 tests) |
| 3 | apply_patch | `locate` wrap-around can return anchor < cursor -> slice panic on a backward multi-hunk patch | pre-existing in reverted `edit.rs`; not fixed |
| 4 | apply_patch | empty-context desync + unverified `-` drop silently corrupts a file, reports ok | pre-existing in reverted `edit.rs`; not fixed |
| 5 | specdecode | `scaffold_for` forced a single key even with optional props present (losslessness violation) | FIXED (gate on closed single-prop schema; 2 tests) |
| 6 | audit doc | per-file test counts were off by one | FIXED (this revision) |

## 1. Per-component rating, with the wall to 10

Scoring: the proven-today score for the scaffolded logic, and the gap that separates it from
10 (which for every row is the same shape: live-model validation + runtime integration, i.e.
a TIME-wall, not a structural one).

### Phase 0: the tool-call protocol  -  proven 6.5 / 10
The single biggest gap in the whole system (there was no model-output tool-call parser at
all) is now closed as a library. The parser accepts the four community formats (Hermes
`<tool_call>`, OpenAI `tool_calls`, fenced JSON, bare JSON), tolerates prose around the call,
recovers a valid call when a sibling block is malformed, and never drops arguments (a
non-JSON string is wrapped, not lost). The loop wires the previously-uncalled `lint_tool_call`
(hallucinated tool/file rejected before any effect, the SWE-agent ACI guardrail) and
`IdempotencyLedger` (keyed calls dedup), and formats every outcome as Hermes feedback that
round-trips back into the parser. What holds it under 10: it is not yet wired into the live
FSM driver (`hide-kernel/src/machine/driver.rs` still acts only on pre-authored `tool_hint`s),
and the format choice is not yet validated against a real local model's output. Both are
TIME-walls for the 10/10 step.

### Phase 1d: Codex-grade apply_patch  -  WITHDRAWN
Applied and tested green, then reverted intentionally in the working tree (`edit.rs` returned
to its original `locate`-based applier). Not part of the delivered scaffold; not graded. The
original `apply_patch` it reverted to carries two pre-existing bugs the review surfaced (see
section 3.1).

### Phase 2/3: the tool spec-decode layer  -  proven 6.0 / 10
This is the differentiator the operator specifically asked for ("a small spec-decode layer
with the tools"). Built as pure, lossless primitives: `ToolCallGrammar` reports the forced
opening scaffolding, the shared prefix of the still-consistent tool names (jump-forward within
the name), the full skeleton once the tool is fixed, a validity gate for constrained decode,
and a `forced_fraction` measure; `PromptLookup` drafts copied argument values from context;
`accepted_prefix_len` is the lossless-verify accounting. Every piece is unit-tested. What holds
it under 10, and it is the load-bearing gap for the whole "fastest" claim: none of this is
wired into the batched serve lanes yet (`forward_multiseq_*` still ignore constraints and
spec), and the actual tok/s win is UNPROVEN by rule until measured on a served model. That is
a TIME-wall (the wiring) plus a measurement (the receipt), exactly the shape the audit calls
convertible.

### Phase 4: parallel tool execution  -  proven 5.5 / 10
`dispatch_parallel` and `dispatch_purity_gated` run independent read-only calls concurrently
while keeping mutating calls strictly sequential and order-preserving, gated by the caller's
`Tool::purity` decision. Tested for order preservation across a mixed batch. What holds it
under 10: not wired into the driver's single-step `do_act`, and the read-only speculative
prefetch (running read-only tools before the model commits) is designed but not built; the
safety boundary (never speculate a mutating tool) is enforced at the call site by construction
but has no live exercise yet.

## 2. Map to the scoreboard

This wave moves the HIDE tool-calling rows off their floors as MEASURED-LAB scaffolds (not yet
R3, so no WIN cell flips):

- HIDE M4 (grammar-guaranteed tool calls), proven 3.0: the grammar validity gate + jump-forward
  primitives are now real library code with tests. Honest new state: the compiler exists AND a
  schema-aware tool-call grammar + lossless jump-forward primitives exist; still GATED on the
  decode-loop wiring and a first-try-valid measurement on a real schema set. The audit's honest
  ceiling for this row is "first-try-valid ~95% + repair," never a literal guarantee; nothing
  here overclaims that.
- Adjacent: the Phase 0 parser is the missing half of "pass state not text" / the agentic loop;
  the apply_patch hardening lifts edit reliability (the ACI lesson) for HIDE ship-readiness.

## 3.1 Pre-existing `edit.rs` bugs the review surfaced (reported, not fixed)

The adversarial review of the reverted-back original `apply_patch` (`crates/hide-tools/src/edit.rs`)
confirmed two real bugs that predate this session and live in code under separate management, so
they are reported here rather than patched:
- SAFETY (panic): `locate` scans forward then wraps to the file top, so a later hunk whose context
  matches only earlier returns `anchor < cursor`, and `apply_unified` then evaluates
  `orig_lines[cursor..anchor]` with start > end, which panics. Repro: original `"target\na\nb\n"`,
  patch `"@@\n-b\n+B\n@@\n-target\n+TARGET\n"`. Minimal fix: guard `if anchor < cursor` -> return a
  CONFLICT instead of slicing.
- LOSSLESSNESS (silent corruption): the `-` branch drops `orig_lines[cursor]` by count without
  checking it equals the removed text, and the empty-context branch can desync the cursor, so a
  patch can delete the wrong line and commit it with `ok:true`. Repro: original `"a\nb\n"`, patch
  `"@@\n\n-a\n+A\n"` yields `"a\nA\n"` (drops `b`, keeps `a`). Minimal fix: verify
  `orig_lines[cursor] == removed_text` before dropping and return CONFLICT on mismatch.

## 3. The path to 10/10 (all TIME-walls)

Every gap above is a build or a run, not a structural wall. In leverage order:
1. Wire `ToolLoop` into the live chat/FSM path and add one end-to-end integration test (Phase 0
   to a real dispatch), behind a flag so the working single-shot path stays green.
2. Add the serve `tools`/`tool_calls` field (Phase 1a) so the native API speaks OpenAI tools;
   render tool defs per arch template; parse tool_calls out. Testable via HTTP with the engine.
3. Thread a real grammar field end-to-end and apply the mask + jump-forward inside
   `forward_multiseq_*` (Phase 2/3 runtime), then MEASURE first-try-valid and tokens-per-forward
   -pass on a served model. This is the receipt that converts the "fastest" claim; it needs our
   own `.tq`-served model (the thesis-gate dependency), so it is gated on that, honestly.
4. Wire parallel + read-only speculative prefetch into the driver (Phase 4) with a fuzz test
   proving no mutating tool is ever speculated.
5. Fill the remaining catalog gaps (memory with path-traversal guards, plan.todo, notebook,
   batch, web, agent.spawn) and register the built-but-dormant MCP client.

## 4. Specialization: pushing past what we pulled (the operator's step 5)

Everything above is parity engineering (ripping Claude Code and Codex). Where hawking's own
codebase can go past the frontier, because it owns the runtime that hosted agents rent:

- State-native tool handoff, not text. The `hide-personalize::kv_handoff` + RWKV
  `to_bytes/from_bytes/fork` primitives let a subagent inherit the parent's KV/SSM state
  directly, so a spawned tool-agent starts warm instead of re-reading a text prompt. No hosted
  agent can do this; it is the "pass state not text" moat (scoreboard HIDE M1).
- Native-`.tq` jump-forward. Because we own the sampler, the grammar mask + jump-forward can
  fold into the fused-arena `.tq` decode path, so the forced-scaffolding tokens cost literally
  zero dispatches, not just a skipped forward pass. A GGUF-based agent cannot reach the sampler
  to do this.
- Original-file-as-draft on our own spec-decode. Cursor's ~13x instant-apply trick (feed the
  original file as the speculative draft) drops straight onto our existing EAGLE5 / n-gram /
  suffix verifier, giving whole-file rewrites at draft speed with a lossless gate we already
  enforce.
- Tool-call as a first-class KV citizen. Prefix-cache the tool-def block and the stable system
  prompt as a pinned slot so a large / MCP-expanded catalog costs one prefill ever, and every
  agent turn reprocesses only the divergent suffix. Combined with the flat-cost SSM long-context
  serve, an agent can hold a huge tool set and history without the per-turn re-prefill tax that
  bounds hosted agents.
- Speculative tool execution under a real effect ledger. We already have `Tool::simulate` +
  `dry_run` (predicted `EffectSet`, no side effects) and the idempotency ledger, so read-only
  prefetch can be proven safe mechanically rather than heuristically, which is the exact thing
  the published speculative-tool-execution work flags as the hard part.

These are the bleeding-edge bets that are only available to a team that owns the engine, not
just the harness. They are the honest "how do we dominate" answer, and each is gated on the
same live-model measurement the rest of the program is.
