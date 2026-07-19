# HIDE Capability-Density and Workflow Eval System (Bible Book XV)

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.2, §3.3, §3.5, §5), `hawking_ide_frontier_2026_07_19.md` §2, §4.8, §8, `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`.
Status: specification for the measurement plane. Every readiness label is the readiness of the primitive the metric depends on, not an aspiration.

## 0. The one hard fact this document exists to fix

**There is no active capability or quality eval harness in the live tree.** Verified:

- `hawking-eval` (pass@1 + Wilson CI over the serve chat path) is **PACKED, unwired** at git `5a99d0e2` (archaeology §5 lever ledger; §3.5 triage marks it REINTEGRATE, "cheap, high-leverage"). It is ~0.3k LOC across two files, drives a model through the existing OpenAI-compatible serve path, computes pass@1 with a Wilson interval, and makes **no engine change** (`5a99d0e2:crates/hawking-eval/src/lib.rs:3-13,61-113,148`, VERIFIED REPO). At N=100 the Wilson 95% half-width is about ten points (`:244-248`).
- `hawking-bench` (decode / prefill / bandwidth / throughput / competitive-vs-mlx/llamacpp) is the **only wired eval harness**, and it measures **performance only** (archaeology §1 crate table, §5 lever ledger: "Perf eval harness | wired").

Consequence: today HIDE can report tokens/s but **cannot report whether a patch is correct**, whether a tool call was valid, or whether a resumed session behaved. Every "fastest / most-capable / densest" claim is therefore currently unfalsifiable from the tree. This document specifies the harness that must exist before any such claim, and the evidence bar (§10) each claim must clear. It is a measurement spec, not a leaderboard.

Non-negotiable ordering (dossier decisions 11-12; archaeology §6.5): reintegrate `hawking-eval` **first** (§9). It is the cheapest unlock in the whole ladder and it gates every capability claim downstream.

## 1. Objective function (restated, not reinvented)

From dossier §2.1: capability density is a **Pareto surface**, and release is a **lexicographic** decision, not one vanity scalar.

```text
                verified task utility
  density ~=  ------------------------------------------------
              resident bytes x model-visible tokens x
              critical-path seconds x (energy or dollar)
```

The fraction is for intuition. The release decision is ordered:

1. meet correctness, security, and task-success floors (a gate, not a score);
2. maximize the number and difficulty of **verified** workflows within the device envelope;
3. minimize p95 end-to-end critical-path latency;
4. minimize resident memory, model-visible tokens, energy, remote cost;
5. minimize human attention and intervention.

A tiny unreliable model is not dense (fails floor 1). A huge model that solves the task while consuming the machine for minutes is not dense either (fails floors 3-4). "Verified" is load-bearing: a task counts only when an oracle passed (§5, §7), never on self-judgment (dossier decision 8).

## 2. Capability-density metrics

Density is measured against real denominators, not nameplate specs. MoE is why the per-parameter axis matters: HIDE ships MoE via `deepseek_v2` (routed + shared experts, MLA; archaeology §3.1), where active parameters are much smaller than total, so density-per-active-parameter is a genuine architectural lever, not accounting.

| Metric | Definition | Denominator source | Readiness of the denominator |
|---|---|---|---|
| Verified tasks / active param | pass-gated tasks per active parameter (MoE routed+shared active set, not total) | model card + router trace (`tool_rounds` unaffected) | INFERRED; active-param count from `deepseek_v2` config, not yet emitted as a trace field |
| Verified tasks / installed GB | pass-gated tasks per on-disk footprint (weights + `.tq` artifacts + index) | filesystem; `.tq` STR2 artifact bytes | partial: `.tq` native serving is feature+env gated off by default (archaeology §3.1), so quantized-footprint runs need the `tq` feature loaded |
| Verified tasks / resident GB | pass-gated tasks per working-set RAM (weights + KV/recurrent + arena) | live KV/recurrent state size; `recurrent_state_size_bytes` is already surfaced in `/v1/hawking/context` | wired for the metric readout; the eval that consumes it is packed |
| Verified tasks / joule | pass-gated tasks per energy to reach green | powermetrics-class sampling on the critical path | MISSING harness; no energy trace field today |
| Verified tasks / wall-clock hour | throughput of **verified** work, not tokens | end-to-end trace `total` time (§8) | MISSING (needs the eval + trace plane) |
| Verified tasks / human intervention | pass-gated tasks per operator touch to green | intervention/approval trace fields (§8) | MISSING (needs the eval + trace plane) |
| Quantization delta | pass@1(quant) minus pass@1(reference) at matched config | two eval runs, same task revision | MISSING until `hawking-eval` is wired **and** `.tq` runs are measurable |

Rules:
- Density numbers are **paired**: quant vs reference-quality, cold vs warm cache, active vs total params. A single number without its pair is inadmissible (dossier §8.1 pinning; §10 here).
- No dollar meter in-product (doctrine; parity `cost.usage_transparency` "STRONGEST INVERSION"). **Energy per verified task is the honest cost axis** and replaces the budget bar HIDE refuses to draw.
- The `/context` "effective tokens" multiplier is **not** a density denominator: `HAWKING_QWEN_TQ_MULTIPLIER` is read at `http.rs:250` but is never set in-repo (archaeology G9/G11), and real KV capacity is fixed at 4096 (G3). Model-visible tokens are counted from the true tokenizer budget of what was actually sent, never from the weight-compression multiplier.

## 3. Workflow metrics

These measure the loop, not the model. Each is an instrumented field on the run trace (§8). Parity metrics reproduce Claude Code behavior; supremacy metrics are gated on the specific build item named.

| Metric | Definition | Class | Readiness / gate |
|---|---|---|---|
| Time-to-first-useful-action (TTFUA) | wall time from submit to the first action a user keeps (first correct tool call or first accepted edit token) | parity | needs the flat kernel loop wired; live turn today is a single-shot 256-tok generate (archaeology S2) so TTFUA is not yet observable |
| Time-to-verified-patch (edit-to-green) | submit to an oracle-passing patch | parity (primary) | MISSING; needs eval + verifier plane |
| Tool-call validity | fraction of tool calls that parse and satisfy the schema first try | parity | **currently measures the degraded path**: JSON constraint mask and spec-decode live only in single-seq `generate()`; batched serve forces `json_mode:false` (archaeology T8). Wiring `hawking-orch::tool_spec_decode` (lossless, packed) is the gate for the supremacy number |
| Turn count | model turns to completion | parity | needs kernel loop |
| Tool rounds | discrete tool round-trips to completion | parity | needs typed tool runtime (`hide-tools` packed) |
| Context tokens | model-visible tokens per turn (true tokenizer budget) | parity | compiler emits true budgets when reintegrated (`hawking-context` packed); serve `/context` multiplier is not this number (§2) |
| Prefill avoided | reused prefix tokens not re-prefilled | supremacy | wired on the queue-drain path only; direct-admit / batch-one still cold-prefills (archaeology G4). Gate: direct-admit prefix reuse |
| State-fork latency | wall time to fork a warm session (should be O(state bytes), no re-prefill) | supremacy | **RWKV lane only**; `fork` is a tested memcpy (`rwkv7.rs:376-378`) but unwired, and transformer/Hybrid capsules are unbuilt. Gate: `HIDE_STATE_CAPSULE_ABI.md` exposure items (serve state routes, session->slot affinity) |
| Resume fidelity | byte-exact next-token logits after restore vs a fresh run | supremacy | RWKV restore is byte-exact at a committed boundary (`tests/rwkv7_state_checkpoint_parity.rs:31-73`); GPU mid-stream capture is not exact until GPU->CPU readback lands (G-CAP-1). Transformer resume is fidelity-via-re-prefill only until the KV capsule is built |
| Merge success | fraction of fleet branches that merge without conflict/regression | parity | `hide-fleet` merge funnel packed, not HTTP-reachable (archaeology §3.5) |
| Permission prompts | approvals surfaced per task | parity metric; supremacy target | fewer is the supremacy goal: sandbox is MEASURED at ~84% prompt reduction (parity `security.sandbox`), egress-default-off removes the exfiltration class (parity `perm.rule_engine`). Enforcement is packed/unwired today |
| Agent-loop failures | taxonomy count: invalid-tool, stall/timeout, non-terminating loop, cancel-not-honored, context-overflow | parity | needs the kernel + governor (`hide-kernel` packed) to even generate these events |

Interpretation guard: until Phase 0/1 wiring (archaeology §6.1) reconnects the vertical slice, most workflow metrics are **not yet observable** because there is no live agent turn to trace. Reporting a workflow number before the slice is wired is inadmissible.

## 4. Product metrics

These are the "usefulness" scoreboard (dossier §2.2, floor 5): does HIDE reduce human effort without creating cleanup? They are longitudinal and are measured **locally, opt-in, with no telemetry egress** (no-metering doctrine, archaeology §3.4). Blind head-to-head preference is a separate instrument; see `HIDE_BLIND_PREFERENCE_STUDY.md`.

| Metric | Definition | Signal source |
|---|---|---|
| Adoption | sessions started per active repo per week | local session registry |
| Repeat use | return rate (session N+1 given session N), day-over-day retention | session timestamps |
| Feature discovery | fraction of shipped affordances a user has invoked (palette, plan mode, fleet, rewind) | local intent ledger |
| Session completion | fraction of sessions ending in an accepted patch or explicit "done" vs abandoned | terminal session state |
| Abandonment | sessions closed mid-turn with no kept artifact; correlate with the failure taxonomy (§3) | session state + loop-failure trace |
| Undo / rollback rate | accepted edits later reverted; `/rewind` and checkpoint restores per session | patch ledger + checkpoint events (parity `session.checkpoint_rewind`) |
| Trust rating | user-reported confidence to let HIDE act unattended (auto/plan mode dwell) | opt-in prompt + permission-mode telemetry |
| Perceived control | steer/interrupt usage vs regret (interrupt-then-redo rate); did the user feel in the driver's seat | SteerBar events + undo correlation |

Doctrine constraint: these are diagnostics for the builder, not a dashboard sold back to the user as a countdown. Undo rate and abandonment are the two that most directly falsify a "capability-dense" claim: high-density work that is routinely reverted is not useful work.

## 5. The Claude Code behavioral parity suite

This is the **parity** gate (reproduce the loved behavior), kept distinct from capability density (§2, raw capability) and product love (§4). Each domain maps to concrete eval tasks, an objective oracle, and the behavioral-parity-spec entries it must satisfy. `hide_status` is copied from the parity spec so the suite and the contract cannot drift.

| Domain | What the eval task exercises | Oracle | Covers (parity-spec ids) | hide_status today |
|---|---|---|---|---|
| Repo understanding | locate a symbol/def/ref across a multi-file repo; answer a "where is X" scoped query | expected file:line set (retrieval precision/recall) | (index plane; `hawking-index` packed) | packed_unwired |
| Planning | produce an editable plan, block edits until approval, honor graded approval mode | plan artifact exists + no source write pre-approval | `perm.plan_mode`, `loop.todo_list` | ui_only |
| Editing | multi-file edit via search_replace / apply_patch / write_file with base-hash concurrency | patch applies + compiles | (tools plane; `hide-tools` packed; apply_patch correctness is a flagged probe, archaeology §4) | packed_unwired |
| Tool use | first-try-valid tool call, count-coalesced rendering, resubmit on `role:tool` | schema-valid parse + expected effect | `loop.collapsed_tools`, `mcp.client_host_server` | partial (parser active, runner packed) |
| Verification | run tests/build, treat exit code as data, gate "done" on green | oracle exit status | (verification plane) | packed_unwired |
| Git | staged/branch state in status line, worktree isolation, no force-push to main | git state assertion | `loop.status_line`, `session.background_supervisor` | partial (status hardcoded) |
| Resume | `--continue`/`--resume` restores a session; byte-exact where the lane supports it | resume fidelity (§3) | `session.durable_transcript`, `session.resume_picker`, `session.checkpoint_rewind` | packed_unwired / ui_only |
| Subagents | file-defined specialist, isolated context, summary-only return, bounded nesting | correct delegation + isolation | `subagents.file_defined`, `subagents.fork_worker`, `teams.coordinated` | packed_unwired |
| MCP | client + host + server over stdio/HTTP, `@resource`, prompts-as-commands, deferred loading | round-trip a real MCP tool | `mcp.client_host_server` | packed_unwired (client only) |
| Skills | two-tier progressive disclosure, description auto-trigger, turn-scoped tool grant | correct hydrate + tool scoping | `skills.progressive_disclosure`, `plugins.bundles` | absent |
| Hooks | PreToolUse gate can deny/ask/allow/rewrite; taxonomy fires; none run before trust | decision object honored | `hooks.lifecycle` | packed_unwired |
| Permissions | deny->ask->allow precedence, compound-command split, protected-path pre-gate, circuit breaker | rule evaluation matches expected verdict | `perm.rule_engine`, `perm.persistence_tiers`, `perm.mode_cycle`, `perm.auto_mode`, `trust.workspace_gate` | packed_unwired / absent |
| Background | detached session survives close/sleep/reboot, dashboard, conditional auto-PR | supervisor round-trip + recovery | `session.background_supervisor`, `goal.evaluator_loop` | packed_unwired / absent |
| IDE bridge | one session across Chat + IDE, loopback token-authed selection/diagnostics inject, native diff review | cross-surface state identity | `ide.two_surface_bridge` (see `HIDE_TWO_SURFACE_ARCHITECTURE.md`) | partial |

Every row is a **regression suite**, not a demo: the parity number for a domain is the fraction of its tasks whose oracle passes at a pinned config (§10). A domain at `packed_unwired` scores zero until reintegrated, which is the honest reading and the point of the archaeology.

## 6. Public benchmarks cannot be the objective

PRIMARY SOURCE (dossier §8.1, source ledger):

- OpenAI's 2026 audit estimates roughly **30% of SWE-Bench Pro tasks are broken** and retracts its earlier recommendation; its earlier audit found SWE-bench Verified increasingly contaminated and misaligned. [DOCUMENTED]
- Anthropic reports a **six-percentage-point Terminal-Bench swing** from infrastructure setup alone, and warns against reading small differences without matched resources. [DOCUMENTED, MEASURED]

Therefore:

- public benchmarks are **regression signals only** (a number that dropped means something changed), never evidence that HIDE is fast, correct, or useful;
- **private, rotating, contamination-audited real-work tasks are the primary release gate** (dossier decision 11; §7 release lane);
- a benchmark result is **inadmissible** unless it pins, in the record, all ten fields:

```text
task-revision | harness | model | quantization | context-policy
| tools | compute-envelope | cache-state | trial-count | confidence-interval
```

A headline score without the full pin is marketing, and doctrine refuses it (dossier decisions 11-12; archaeology §6.5). This is the same discipline the state ABI applies to lossless-vs-lossy claims (`HIDE_STATE_CAPSULE_ABI.md`): the label must carry its receipt.

## 7. The three eval lanes

| Lane | Trigger | Latency budget | Gate it enforces | Contents (dossier §8.2) |
|---|---|---|---|---|
| Per-change fast | every commit / PR | seconds to low minutes | blocks merge on correctness regressions in the plumbing | wire-format + tool round-trip tests; tokenizer/chat/FIM golden; context-compiler determinism + budget; prefix/state parity; patch transaction + rollback; stop/structured-output correctness; cache invalidation; sandbox fs/network escape; cancel/replay/crash repair; targeted small coding tasks |
| Nightly | scheduled | tens of minutes to hours | trend + capability-drift detector | private Rust + TS issue suite; multi-file edit-format tasks; ContextBench-style retrieval recall/precision/utilization/efficiency; BFCL-style single/multi-turn tool use + hallucination + latency; Terminal-Bench 2.x as a **noisy** indicator with infra receipts; Aider-Polyglot-style edit compliance; long-horizon resume + compaction continuity; prompt/tool/memory injection suite; quantized-vs-reference deltas |
| Release | pre-ship | hours to days | the authorizing gate for any product/superiority claim | rotating contamination-audited private repo tasks; repeated trials with CIs; cold + warm cache; interactive + background latency envelopes; human acceptance/undo/intervention/time-saved study; security red team; crash/power-loss recovery; model/provider outage + offline fallback |

The fast lane is the only lane whose pieces partly exist today (unit tests around the live crates); the nightly and release lanes both depend on `hawking-eval` being wired and a private corpus existing (§9, §11). Terminal-Bench is deliberately downgraded to a noisy indicator per the six-point infra swing (§6).

## 8. Required trace fields

Every run emits one trace spanning queue -> routing -> cache -> retrieval/packing -> prefill/TTFT -> decode -> model-to-tool gap -> tool exec -> verification -> checkpoint/compaction -> patch (dossier §4.8, §8.3). Minimum non-sensitive record:

- task, snapshot, run, trace, parent IDs;
- model, provider, quantization, tokenizer, template, engine, **prompt ABI**, **tool ABI** versions (the ABI fields tie to `HIDE_SPEED_FRONTIER.md` and `HIDE_STATE_CAPSULE_ABI.md` IdentityBinding);
- context item identities, token counts, sources, trust domains, retrieval scores;
- cache hit/write/eviction and reused-token counts (feeds "prefill avoided", §3);
- queue, retrieval, prefill, TTFT, decode, tool-gap, tool, verification, total times;
- state checkpoint/fork/restore sizes and times (feeds "state-fork latency" + "resume fidelity", §3);
- tool names, effect classes, status, retries, artifact handles;
- tests/oracles and outcomes;
- user interventions and approvals (feeds §2 per-intervention density + §4 product metrics);
- final patch identity and accept/undo outcome.

Rules: OpenTelemetry-compatible GenAI concepts (source ledger); **do not capture prompts, source, tool arguments, or results by default**; sensitive-content capture is opt-in, local, redacted, sampled explicitly, retention-limited (dossier §4.8). A metric without its backing trace field is not measurable, and several §2/§3 rows are MISSING precisely because their field does not yet exist.

## 9. Reintegration order (the build gate for this plane)

1. **Reintegrate `hawking-eval` first.** ~0.3k LOC, two files, consumes the existing serve path, **zero engine change** (`5a99d0e2:crates/hawking-eval`). It gives pass@1 + honest Wilson CI immediately and unblocks every capability claim. This is the single highest-leverage, lowest-cost item in the ladder (archaeology §3.5, §6.5).
2. **Add the trace plane (§8)** behind a feature flag, non-sensitive by default. Without it, workflow and density metrics are unbacked.
3. **Stand up the private rotating corpus (§11 open decision).** Until it exists, the release lane cannot run and no superiority claim is admissible.
4. **Wire the fast lane (§7)** onto the existing crate unit tests plus the new eval, so every change is correctness-gated, not just perf-gated.
5. **Gate each supremacy metric on its build item, explicitly:** prefill-avoided -> direct-admit prefix reuse (G4); state-fork latency + resume fidelity -> capsule exposure (`HIDE_STATE_CAPSULE_ABI.md`, G-CAP-1); tool-call validity -> `hawking-orch::tool_spec_decode` wired into the batched path (T8); permission-prompt reduction -> sandbox enforcement wired (`security.sandbox`). No metric is reported off an unwired primitive.

## 10. The evidence bar before a superlative

A claim of the form "fastest / most-capable / densest / better than Claude Code on X" is admissible **only** when all of the following hold. This is the doctrine gate for `HIDE_SUPREMACY_THESIS.md` and the acceptance criterion for `HIDE_PRIORITIZED_BUILD_LADDER.md` items.

1. The **private rotating real-work gate passed** for the relevant domain (not a public leaderboard, §6).
2. **N trials with a stated confidence interval** (Wilson for pass rates; the packed harness already does this at N=100 ~ +/-10 pts, so N is chosen for the width you need).
3. The **full ten-field pin** is in the record (§6): task-revision, harness, model, quant, context-policy, tools, compute, cache, trials, CI.
4. A **matched-resource baseline** (same hardware envelope, same task revision), per the Terminal-Bench infra-swing lesson (§6).
5. **Cold and warm cache** both reported (a warm-only number is a best case, not a claim).
6. **Receipts from the real app path**, not a microbench (dossier decision 12; the live serve path, the real kernel loop, not `generate()` in isolation).
7. For any quality claim, the **quantization delta is within the reference-quality floor** (§2), measured, not assumed.
8. The claim names the **lane** that authorized it (§7); only the release lane authorizes a product-facing superiority claim.

Anything short of this is a lead, not a result, and is labeled INFERRED or UNKNOWN rather than asserted. The current honest state of the tree is that **no superlative is yet admissible**, because item 1 (no corpus), item 2 (no wired eval), and item 6 (broken vertical slice) all fail. Fixing that is §9, and it is cheap.
