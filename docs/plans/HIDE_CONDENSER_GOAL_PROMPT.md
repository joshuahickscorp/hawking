# HIDE condenser goal prompt

Standing prompt for the next session. Paste this as the opening message (or run the goal loop against it). Written 2026-07-16 after a 82-agent deep audit and an 81-agent SOTA frontier sweep. House rules apply: no em/en dashes, no middots, no ellipsis, no git self-attribution, verify before claiming, no commit/push without approval.

## Mission

Turn HIDE from good-parts-unwired into a working local coding agent by ripping the mid-2026 frontier and improving on it where our RWKV-state / .tq / kernel-control position lets us. This is the condenser mindset: replicate what the frontier proved, then push past it on our home turf. Judge only what is on main; the dense refactor is a separate track.

## Read first (source of truth)

1. docs/plans/hide_deep_audit_2026_07_16.md - the Zed-anchor audit: 12 verified blockers, the one root cause (the agent kernel is unwired; the live turn is a single-shot 256-token raw-text completion).
2. docs/plans/hide_sota_frontier_and_regrade_2026_07_16.md - the frontier read + regraded ladder (10 = SOTA) + the 16-build ripped ladder + the 6 strategic forks.
3. Memory: [[hide_deep_audit_2026_07_16]] and this session's frontier notes.

Do not re-derive these. Trust the file:line evidence in the audit; spot-check only where you are about to change code.

## The one root cause to fix

The live user turn (`generate_submit_turn`, crates/hide-backend/src/host.rs:801) is a single-shot `runtime.generate` of the raw last message: no tools, no history, no context, no FSM, 256-token cap. The whole agent stack (kernel FSM, ToolLoop, context compiler, RWKV state fork, oracle suite) exists and is tested but has zero production callers. Everything the product looks like it does is placebo wired to dropped intents. Fixing this is not wiring the 12-phase FSM; the frontier proved coding is the open-ended agent case, so collapse it into ONE flat execution-grounded loop.

## Non-negotiable sequencing

The eval gate comes first because every capability claim is unfalsifiable without a number, and the four Tier-0 unblockers are all S-effort with no dependencies. Do them before anything ambitious.

**Phase A (unblock, all S, no deps):**
- B1: hawking-serve completes the standard OpenAI tool round-trip (assistant tool_calls + content:null, then role=tool continues; today turn 2 400s); wire the unit-tested parser+ToolLoop into the batched lane; honor stop strings on the batch path; stop double-tokenizing under the engine lock.
- B3: wire the real cargo/git/clippy oracle (live only inside the dead FSM today) into the live accept path as one outcome-label API.
- B6: stop discarding the context compiler output (only a blake3 hash is kept); feed it as the actual prompt; replace chars/4 with the real .tq tokenizer.
- B2: expose RWKV-7 state save/fork/restore as HTTP routes behind a prefix-hash session store (FlashRT capsule model); move the transformer prefix-reuse lookup into admit so batch=1 stops re-prefilling from 0; lift the hardcoded max_seq_len 4096.
- B4: FIM tab completion via the Qwen2.5-Coder native FIM tokens (151659/151661/151660), bounded middle-only decode, BASE weights.

**Phase B (the working agent + first number, M):**
- B5: flat ReAct loop (mini-swe-agent shape) inside ToolLoop, sandboxed exec with per-command approval, stop on sentinel + oracle + iteration cap. Expect ~20% SWE-bench Verified at 7B and MEASURE it; do not inherit the frontier 74%.
- B7: OpenAI-compatible /v1 + the tiered eval gate (EvalPlus pass@1 week one, Aider Polyglot edit-format compliance, BFCL offline, then SWE-bench). Add a per-capability .tq quant-delta column. This produces our first-ever coding number. Nothing about product differentiation moves until this runs.
- B8: constrained decode + jump-forward on the batch path (llguidance Rust MIT, or FFI xgrammar), TagDispatch to dodge the Tool-Suppression trap, mask on a background CPU thread overlapping the Metal forward.
- B9: deterministic read-only pre-dispatch gates + BRT fail-to-pass cogeneration for repair.

**Phase C (speed + moat conversion, M/L):** B10 bit-exact Metal chain-verify kernel (the trust gate), then B11 file-as-draft fast-apply (3-13x on localized edits, larger relative win for us because decode is bandwidth-bound at batch=1), B13 SuffixDecoding drafter, B12 hybrid best-of-N over RWKV state-forks (the moat-to-accuracy conversion, +7-8pt).

**Phase D (context depth + memory, M/L):** B14 Rust RepoMap (tree-sitter -> PageRank -> ~1k-token repo view), B15 tool-output pruner (SWE-Pruner/Squeez), B16 Mem0-style semantic memory with hybrid retrieval.

## Rule on these before they block you

1. Security of local exec: the moment B5 ships, the agent runs model-emitted bash on the user's machine. Decide the sandbox model (macOS seatbelt; hide-tools already confines shell.run but not the terminal RunCommand path) and per-command approval UX BEFORE B5 lands.
2. Model track: pure RWKV-7 vs hybrid (1-in-4 full-attention or a bounded exact cache) given the linear-attention multi-key recall cliff. Gates the moat.
3. Latency budget: state the TTFT and total-turn envelope; FIM needs sub-100ms; best-of-N is Nx wall-clock at batch=1.

## Operating doctrine (from the user's standing feedback)

- Build custom Rust solutions on silicon work, then audit-to-improve; do not just bench.
- On perf, never stop at +1; stack levers and attack the next. Carry the full spread, label proxy vs measured.
- Verify parity/bit-identity YOURSELF in main before merging any source/decode lane; gate merges on CI exit code, not a watch command.
- Receipts outrank prose. One heavy job at a time on the box.
- Every build above ends in a real number from B7's gate or a bit-identity proof, not a claim.

## Definition of done for this arc

A stranger who opens the DMG can point HIDE at a Rust repo, type a task, watch a flat agent loop run real tools against real cargo/git oracles with a live number behind it, get a file-as-draft fast edit, and see the RWKV state-fork give near-free branching. Every facet in the regraded ladder has moved because a measured capability landed, not because a placebo got prettier.
