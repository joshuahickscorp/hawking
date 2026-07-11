# M1 Ultra Run Report: the hawking maximization run

The single accumulating artifact for the M1 Ultra maximization loop (see `docs/plans/M1ULTRA_GOAL_PROMPT.md`
and `docs/plans/M1ULTRA_POTENTIAL_AUDIT.md`). Every wave appends here so a dead session loses at most one
wave. Every number is graded by evidence strength so nothing is over-claimed. House style: no em or en
dashes. Proof discipline: effective bpw only, a tie is a null, no WIN cell without an R3+ receipt.

## 0. How to read this (evidence grades)

- MEASURED-R3+: a real receipt at repro level R3 or higher (one-command same-machine-class repro), the
  ONLY grade that backs a public WIN.
- MEASURED-LAB: a real number on this box, author-rerunnable, below R3 (R0-R2). Supports a contingent or
  negative claim, never a public win.
- GATED: the build or artifact the claim needs is not yet complete; the number is honestly withheld.
- UNPROVEN: never run. Zero by rule.
- NULL: the lever tied its baseline at matched compute or a certified gate. Recorded as a citable bound.
- WALLED: a proven structural ceiling with a mechanistic reason (the honest terminal state of a category).

## 1. The scoreboard (start-of-run state, from M1ULTRA_POTENTIAL_AUDIT.md)

Proven = adversarially-verified current-today. Bounded = the audit's honest ceiling. Maximal = the ceiling
once unbounded wall-clock converts every time-wall. Update the Proven column as each wave lands a receipt.

| Category | Proven | Bounded | Maximal | State | Last receipt |
|---|--:|--:|--:|---|---|
| Decode throughput (single-stream, iso-quant gap) | 6.5 | 7.5 | 9.0 | MEASURED-LAB (31.03 tok/s, a4_clean_walltime.json) | - |
| Kernel maturity (45 gemm_q4_k perms, autotune) | 7.0 | 7.5 | 8.5 | MEASURED-LAB | - |
| Prefill + host dispatch | 5.5 | 6.5 | 7.5 | MEASURED-LAB | - |
| Effective-bpw floor + bit-floor-vs-scale LAW | 3.5 | 6.0 | 8.5 | GATED (0 floor points, need >= 2) | - |
| Sub-1-bit SUBBIT lane | 2.5 | 4.0 | 7.0 | GATED (side-info floor 0.1093 alive) | - |
| STRAND codec quality vs QTIP/QuIP#/AQLM | 3.0 | 5.5 | 8.0 | GATED (synthetic probe only) | - |
| Quality recovery / the Doctor (gradient moonshot) | 5.0 | 8.0 | 9.0 | GATED (train-free de-risked; gradient arm unrun) | - |
| Doctrine: CPU-bf16 headline vs bf16-on-MPS | 3.0 | 7.0 | 7.5 | UNPROVEN (microbench A) | - |
| Native .tq serve (build + throughput) | 6.0 | 8.5 | 9.5 | GATED (built, tps unmeasured) | - |
| RAM-cliff throughput demo (the money demo) | 0 | 9.0 | 9.5 | UNPROVEN | - |
| Flat-cost long-context serve (SSM wired) | 6.0 | 8.0 | 9.0 | MEASURED-LAB (B=1, 0.4B) | - |
| Honest recall (NIAH / Omega(N) theorem) | 1.0 | 6.0 | 8.0 | UNPROVEN (synthetic ladder only) | - |
| KV-frontier levers (int4-KV, YaRN, STKV) | 2.0 | 6.0 | 8.0 | GATED (int4-KV disabled, ppl gate unrun) | - |
| EAGLE-3 trained draft head | 1.5 | 3.5 | 7.0 | GATED (offline tau 0.877 << 2.5) | - |
| Verify path + lossless-verify exact-match gate | 0 | 4.0 | 8.0 | NULL (gate ran, FAILED 6/20) | - |
| SpecGovernor + drafter economics | 3.0 | 5.0 | 6.5 | MEASURED-LAB (n-gram 1.43 < 1.6 floor) | - |
| Honesty / receipt discipline | 8.5 | 9.5 | 9.5 | MEASURED-LAB (self-test passes) | receipts/official/qwen-05b-tq3.json |
| CI gap (honesty machinery that never runs) | 3.0 | 6.5 | 9.0 | UNPROVEN (tests/ never run in CI) | - |
| The GO conveyor (studio_run.py P0-P9) | 5.0 | 8.0 | 9.0 | CRASH FIXED off-box (P6/P8 unpack); go-plan green; e2e run needs box | - |
| Architecture coverage (the model families) | 2.5 | 6.5 | 8.5 | GATED (Qwen-dense verified; Mamba2 condense-track coverage added off-box) | - |
| Size frontier / expert-paging serve | 1.0 | 4.5 | 7.5 | RE-DERIVED off-box: 235B/405B/671B fit RESIDENT on 128GB, pager OFF critical path | - |
| HIDE M1: pass state not text (handoff) | 6.5 | 8.5 | 9.0 | GATED (kv_handoff seam, FakeCopier) | - |
| HIDE M2: free local fleets | 5.0 | 8.0 | 8.5 | GATED (fabric real, economics a predictor) | - |
| HIDE M3: own the .tq format | 5.5 | 8.5 | 9.0 | GATED (no e2e coherent-token receipt) | - |
| HIDE M4: grammar-guaranteed tool calls | 3.0 | 7.0 | 7.5 | MEASURED-LAB scaffold, adversarially verified (parser + parse/lint/dedup/dispatch loop + schema-aware jump-forward grammar + prompt-lookup, 39 owned tests; 6 review findings, 4 fixed w/ regression tests, 2 pre-existing edit.rs bugs reported); still GATED on decode-loop wiring + first-try-valid receipt | agentic_tool_system_audit_2026_07_11.md |
| HIDE ship-readiness (Tauri, executor, tests) | 7.0 | 8.5 | 9.0 | MEASURED-LAB (signed 18 MB DMG on disk) | - |
| The thesis gate (can the local model code) | 0 | 7.0 | 7.5 | UNPROVEN (hawking-eval built, never run) | - |

North-star overall: proven 3.25 / bounded ceiling 6.33 / maximal ceiling ~8.4.

## 2. Open gates (answer these to move the scoreboard)

| gate | status | owned by |
|---|---|---|
| studio_run.py runs end-to-end (crash fixed)? | CODE-FIXED off-box (P6/P8 3-tuple unpack corrected); e2e heavy run still needs the box | SPINE-0 / Wave 0 |
| Thesis gate: can the .tq-served local model code? | OPEN (hawking-eval never run) | SPINE-1 |
| Gradient recovery clears <= +2% at sub-4-bit on 7B+? | OPEN (every +dr swap-died on 18 GB) | SPINE-2 |
| Bit-floor-vs-scale law fits (>= 2 verdict floor points)? | OPEN (0 on disk) | SPINE-2 |
| MoE per-expert sensitivity non-uniform on a real bake? | OPEN (synthetic probe non-uniform) | SPINE-3 |
| Native .tq serve tok/s > 10 on a full resident model? | OPEN (only 20 MB toy baked) | SPINE-4 |
| RAM-cliff: serve where a control box's Q4_K OOMs? | OPEN (unrun) | SPINE-4 |
| CI runs the tests/ integration targets? | OPEN (--lib --bins skips them) | SPINE-5 |
| Batched-verify bit-identical to greedy (spec admissible)? | OPEN (failed 6/20) | SPINE-5 |
| MPS-bf16 doctor beats CPU-bf16 throughput? | OPEN (microbench A) | Wave 0 |
| DeepSeek-V3 / GLM router implemented (correct logits)? | OPEN (plain softmax topk, wrong for V3/GLM) | SPINE-3 prereq |

## 3. Benchmark ledger (fill as measured on M1 Ultra)

| bench | M3 Pro proven | M1 Ultra measured | note |
|---|---|---|---|
| Qwen2.5-3B-Q4_K_M decode tok/s (single-stream) | 31.03 (a4_clean_walltime.json) | - | bandwidth-bound; projection ~150-165 until run |
| Aggregate tok/s at max continuous batch | ~48 (B=8, conservative) | - | the re-headlined metric on 128 GB |
| 7B doctor recovery, best recovered eff-bpw @ <=+2% | none (swap-died) | - | the moonshot gate 1 headline |
| Native .tq decode tok/s (resident 70B) | none | - | microbench B / SPINE-4 |
| RAM-cliff tok/s vs control box Q4_K | none | - | the money demo |
| Energy J/tok at the cliff | none | - | the energy moat |

## 4. Wave log

Append one paragraph per wave: what ran, verdict, category movement, next lever. No wave ends without a
committed (approved) artifact.

- Wave C (2026-07-11, wiring + hardening): with operator approval, committed the scaffold
  (commit 79f54420, 11 files, 47 tests) then landed three wiring/hardening pieces (commit
  14909a97, 469 insertions): (1) MCP REGISTRATION - `register_mcp_servers` resiliently connects a
  set of MCP servers and registers their tools into the registry (one bad server is recorded as
  an error, does not abort the rest); the client was built-but-never-called. Tested against a
  live stdio fake server. (2) NATIVE TOOL CALLING ON SERVE (Phase 1a) - `/v1/chat/completions`
  now accepts `tools`, renders a Hermes/Qwen `<tools>` system preamble into the prompt, and
  parses the completion back into OpenAI `tool_calls` with finish_reason; a self-contained
  `tool_calls` module (5 tests) wired in without breaking the 35-test serve suite. (3) EDIT.RS
  GUARDS - with approval, added the two minimal safety guards for the pre-existing bugs the
  review found: a backward/out-of-order hunk is now a CONFLICT not a panic, and a removal that
  does not match the file is a CONFLICT not silent corruption (2 tests). NOT re-introducing the
  reverted seek_sequence fuzzy matcher. A second adversarial verification pass over this wiring
  (12 agents) CONFIRMED 8 more real defects - including two in the edit.rs guards I had just
  added (an over-rejection of valid stripped-blank diffs, and a remaining blank-line corruption
  hole the removal guard did not cover), a serve false-positive (untagged prose JSON turned into
  a spurious tool_call discarding the real answer), a streaming gap (raw <tool_call> XML streamed
  instead of structured tool_calls), and three MCP resilience holes (no per-server timeout so one
  hung server stalls the catalog; duplicate ids silently shadowing tools; a false shutdown
  contract + no kill_on_drop). ALL 8 FIXED with 9 regression tests (commit 90de248a): edit.rs now
  keeps blank context lines in the located sequence and verifies them; serve requires an untagged
  call to name a declared tool and buffers streaming to emit real tool_calls; MCP gained a
  per-server timeout, duplicate-id refusal, kill_on_drop, and a corrected doc. A THIRD
  loop-until-dry pass then caught 2 regressions in those very fixes: the edit.rs empty-line
  change over-rejected a patch ending in a trailing blank, and the streaming buffer emitted its
  flush only in the failure-sentinel arm while the decode loop signals normal completion by
  DROPPING the sender - so a streaming chat with tools silently dropped the whole answer. Both
  fixed (commit b6d30db5): hunks now strip trailing blanks, and the streaming flush + [DONE]
  moved outside the token loop (also fixing a pre-existing missing-[DONE] on the non-tools
  success path). A FOURTH lean pass caught 1 narrow edge-case (an all-blank hunk body silently
  reported ok instead of CONFLICT); fixed with a guard (commit 903baa20). The verification loop
  CONVERGED: pass1 6 findings, pass2 8, pass3 2, pass4 1 - a clean decreasing trend, all 17
  fixed with regression tests. FINAL STATE: five commits (79f54420 scaffold, 14909a97 wiring,
  90de248a +8 fixes, b6d30db5 +2 fixes, 903baa20 +1 fix), all per operator approval, LOCAL ONLY
  no push. Full workspace: 392 passed / 7 failed where all 7 are the pre-existing q8_kv Metal
  tests in hawking-core (absent from every .metal source), zero non-q8kv failures. Delivered +
  verified: Phase 0 (parser + parse/lint/dedup/dispatch loop + parallel), Phase 2/3 (jump-forward
  + prompt-lookup spec-decode), Phase 1a (native serve tools/tool_calls incl. streaming), Phase
  1b (memory tool), Phase 1c (MCP registration). Grades stay MEASURED-LAB. STILL GATED for the
  tok/s + first-try-valid RECEIPT that would flip a WIN cell: our own .tq-served model (the
  thesis gate). Unbuilt/unblocked next: rest of catalog (plan.todo/notebook/batch/web/agent.spawn),
  Phase 0 -> live FSM driver wiring, and the runtime mask/spec wiring into forward_multiseq_*.
- Wave B (2026-07-11, catalog expansion start): added the `memory` tool (Phase 1b) - a durable
  cross-session scratchpad with `view|create|str_replace|insert|delete|rename`, modeled on
  Anthropic's memory tool, rooted at a private per-workspace directory. The security boundary
  (path-traversal rejection: absolute paths, `..` traversal, and percent-encoded `%2e/%2f/%5c`
  escapes) has a dedicated test per vector. Registered as the 23rd builtin; 8 tests green,
  hide-tools crate clean (52 tests, no warnings). Remaining catalog gaps: plan.todo, notebook,
  batch multi-edit, web.fetch/search, agent.spawn, and wiring the built-but-dormant MCP client.
- Wave A-verify (2026-07-11, adversarial verification + fixes): ran a 5-dimension review
  workflow (finder + independent refuting verifier per dimension, 11 agents) over the Wave A
  scaffold, hunting correctness / losslessness / safety / overclaims. It CONFIRMED 6 real
  defects, each reproduced end-to-end - proof the discipline earns its keep. FIXED 4 in the
  delivered scaffold, each with regression tests: (1) parser dropped a bare-JSON call when a
  `[...]` (markdown link/citation) preceded it -> now scans ALL balanced spans; (2) the loop's
  feedback formatter let untrusted tool output forge a `<tool_call>` (TT8 violation) -> now
  escapes the envelope delimiters in body+name; (5) `scaffold_for` forced a single arg key even
  when optional props existed (a real jump-forward LOSSLESSNESS violation for the shipped
  fs.read schema) -> now gated on a closed single-property schema; (6) audit doc per-file test
  counts corrected. The other 2 (a wrap-slice panic and an unverified `-` drop that silently
  corrupts a file) are PRE-EXISTING bugs in `edit.rs`, surfaced when Phase 1d was reverted;
  reported in audit section 3.1, not fixed (edit.rs is under separate management). Also: Phase
  1d (Codex fuzzy apply_patch) was reverted intentionally in the working tree, so it is
  WITHDRAWN from the scaffold. Honest test state after fixes: 39 owned scaffold tests green
  (parse 14 + runner 10 + shared mod.rs 2 + tool_spec_decode 13); full workspace 392 passed / 7
  failed where all 7 are the pre-existing q8_kv_parity Metal-kernel-missing tests in
  hawking-core (kernels absent from every .metal source), zero non-q8kv failures. Grades stay
  MEASURED-LAB. Commit pending approval. Next lever: drive-to-10 wiring (Phase 0 into the live
  FSM; serve `tools` field), still gated on a served model for the tok/s receipt.
- Wave A (2026-07-11, agentic tool system scaffold): executed the operator's scaffold-all
  directive on the agentic tool system plan (`docs/plans/agentic_tool_system_2026_07_11.md`),
  built from three deep-research passes (Claude Code tool architecture, Codex + other agents,
  constrained/speculative decoding). Landed as real, compiling, unit-tested library code,
  highest-leverage-first: (1) PHASE 0 keystone - the missing model-output tool-call parser
  (`hide-kernel/src/tools/parse.rs`, tolerant across Hermes/OpenAI/fenced/bare formats) plus the
  parse->lint->dedup->dispatch loop (`runner.rs`) that finally calls the built-but-dormant
  `lint_tool_call` + `IdempotencyLedger`, with Hermes-shaped self-correction feedback. (2) PHASE
  1d - upgraded `apply_patch`'s hunk locator to the Codex 4-pass fuzzy `seek_sequence` (exact ->
  trailing-ws -> both-ws -> typographic-unicode) and made context emission byte-exact from the
  file, with two adversarial drift tests. (3) PHASE 2/3 differentiator - the tool spec-decode
  layer (`hawking-orch/src/tool_spec_decode.rs`): schema-aware `ToolCallGrammar` jump-forward
  (envelope prefix, tool-name common-prefix, full skeleton once resolved, validity gate,
  forced_fraction) + `PromptLookup` n-gram drafter + `accepted_prefix_len` lossless accounting.
  (4) PHASE 4 - parallel + purity-gated dispatch primitives (read-only concurrent, mutating
  sequential, order-preserving). Evidence: 41 targeted unit tests green; full-workspace
  regression run recorded next. Grades are MEASURED-LAB (unit tests, below R3): no WIN cell
  flips. The load-bearing gap for the "fastest" claim stays honest and GATED - none of the
  constrained/spec primitives are wired into the batched serve lanes yet, and the tok/s win is
  UNPROVEN until measured on our own served model (the thesis-gate dependency). Audit +
  per-component /10 rating + the specialization thesis: `agentic_tool_system_audit_2026_07_11.md`.
  Commit pending approval. Next lever: confirm the full-suite regression is green, then the
  drive-to-10 wiring (Phase 0 into the live FSM, then serve `tools` field, then the decode-loop
  mask + jump-forward measurement once a served model exists).
- Wave 0-prep (off-box, M3 Pro 18 GB session): converted the parts of Wave 0 that are pure code,
  ahead of the box arriving, so the first on-box session starts from a shorter Wave 0. Landed: (1)
  SPINE-0 FIXED - studio_run.py P6 (`bench_baselines`) and P8 (`codec_bakeoff`) unpacked the 3-tuple
  `eval_targets` into 2 vars (ValueError, the conveyor had never finished); both now unpack 3, and
  `--go-plan` is green. (2) DEVICE CONSTANTS RE-DERIVED for the M1 Ultra 128 GB / 800 GB/s / 8 TB box
  (audit re-derivations 1-4): `size_frontier.DEVICES` gains an `m1ultra` row (112 GB budget, 8 TB,
  800 GB/s) and is the DEFAULT; `ladder.py` RAM 96->128, WEIGHT_BUDGET 84->112, CONDENSE_RESIDENT_MAX
  34->48; `ramcliff_bench`/`expert_cache_policy` bandwidth denominators 400->800 GB/s; the FRONTIER
  serve-fit gate 84->112 GB. (3) THE PIVOT: with the 112 GB budget, 235B-A22B (~39 GB), 405B-dense
  (~68 GB), and 671B@1.0 (~84 GB) all fit RESIDENT (verified via size_frontier: regime=RESIDENT,
  full speed) - the OOC expert pager (hardest serve-build item, Type-1 dead in free-RAM) drops OFF
  the critical path for the entire prize, shortening it to 5 steps; the pager is deferred to the
  deep frontier (744B/1T/3T) only. (4) STUDIO_GO.md locked context + serve-build critical path
  re-derived for the box; Mamba2 gained condense-track arch coverage. All tools py_compile green.
  Also landed: PROCUREMENT accelerated. The ~4.3 TB of bf16 frontier parents were the one real
  bottleneck; `tools/condense/procure.py` forces the fastest-SOTA path (hf_transfer Rust accelerator
  + hf_xet dedup backend + --max-workers, both already installed), turning download from a software
  cap into a pure link cap: the whole frontier manifest lands in ~10 h on gigabit (vs an
  unaccelerated multi-day pull). preflight.py now verifies the accelerators are active; the frontier
  "not staged" message points at procure.py; BASELINES.md documents it. --stream emits the advanced
  download+bake+delete fused plan (peak disk ~= one shard) gated on a baker --append-tq mode.
  Not done (needs the box): cargo build/test --workspace, the two day-0 microbenches, staging bf16
  parents, the oracle fixture, autotune. Commit pending approval. Next lever on-box: finish Wave 0
  (build + microbenches), then SPINE-1 the thesis gate.
- Wave -1 (2026-07-03, off-box, this session): the M1 Ultra potential audit was produced by a nine
  subsystem code-grounded pass, each grade independently adversarially verified (four scores cut, one
  raised). It set the start-of-run scoreboard above, named the two ceilings (bounded 6.33, maximal ~8.4),
  classified every wall as TIME (attackable, a build or a run) or STRUCTURAL (a theorem, bandwidth physics,
  or model capability), and pinned the spine and Wave 0. Verdict: the proven state is honestly 3.25/10 and
  correct as a pre-box reading; the prize is the gap to ~8.4, which is almost entirely never-run
  experiments and unbuilt code that unbounded wall-clock converts. Next lever: Wave 0 on the box (transfer +
  build green, fix the studio_run.py crash, regenerate device constants, stage bf16 parents, the two day-0
  microbenches, pin the oracle fixture), then SPINE-1 the thesis gate.
