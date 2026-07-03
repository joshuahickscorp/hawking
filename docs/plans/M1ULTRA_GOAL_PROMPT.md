# M1 ULTRA GOAL PROMPT: the iterative maximization loop for the delivered box

This is the cold-start prompt for the first session ON the delivered Mac Studio (M1 Ultra, 20-core CPU,
48 to 64-core GPU, 128 GB unified memory, ~800 GB/s, 8 TB SSD). It is deliberately a GOAL prompt, not a
task list: it sets a bar above the honest expected value (the maximal ceiling from
`M1ULTRA_POTENTIAL_AUDIT.md`) and wraps the work in an explicit wave loop so every session converges on
either a converted receipt or a proven wall, never on drift.

On the Studio the whole thing is one command: type `/goal` in a session at the repo root (the repo ships
`.claude/commands/goal.md`, which reads the full handoff stack, this file included, then executes the
fenced block below). Pasting the fenced block verbatim as the opening message works identically where
slash commands are unavailable. House style everywhere: no em or en dashes.

Design notes (why each clause exists):
- The goal is set at the MAXIMAL ceiling (~8.4 overall) knowing the expected value is nearer the bounded
  6.33. The unrealistic bar is the mechanism; the proof discipline (effective bpw, R3 receipts, the fake-win
  ban, the adversarial reproduction) is the brake. Aim at the ceiling, report honestly against it.
- The operator's standing directive is UNBOUNDED WALL-CLOCK: no time limit, the box is plugged in 24/7,
  optimize for maximum proof and maximum coverage, EXPAND not reduce. The slower M1-generation CPU makes
  every doctor pass, bake, and floor-search take LONGER, and that is expected and accepted, not a reason to
  cut scope. The stop-check is purely about knowledge convergence, never about running out of time.
- The audit's key analytic move is baked in: most of the deflationary audit's "walls" are TIME-walls
  (unbuilt code or unrun experiments), which unbounded wall-clock converts; only a handful are STRUCTURAL
  (the bandwidth-invariant iso-quant gap, the dense PTQ entropy floor, the Omega(N) recall theorem, the
  spec low-batch regime, model intelligence at fixed size, R5). The loop attacks every time-wall and PROVES
  or CIRCUMVENTS every structural wall.
- SPINE-0 is pinned because the GO conveyor has a latent crash (`studio_run.py:279/:286` unpacks a 3-tuple
  into 2 vars, ValueError, crashes P6/P8). The conveyor has never run end-to-end; nothing downstream is
  real until that two-line fix lands.
- The two moonshot gates (does gradient recovery heal 7B-32B; is MoE per-expert sensitivity non-uniform)
  are the reason to go, both still UNMEASURED, both runnable only on 128 GB. They own the spine after the
  thesis gate.
- Wave 0 is boring on purpose: transfer/build green, fix the crash, regenerate the stale device constants
  (no m1ultra row exists in `size_frontier.py`; the m3pro18 profile is stale), and two day-0 microbenches
  that decide open doctrine questions by MEASUREMENT (MPS-vs-CPU-bf16 doctor throughput; resident `.tq`
  serve tok/s) instead of assumption.
- The adversarial reproduction (`frontier_verifier` re-derives every ship-candidate from scratch) and
  preregistration in code before compute are the two clauses that keep the fake-win ban real at scale. They
  are non-negotiable.
- `M1ULTRA_RUN_REPORT.md` is the single accumulating artifact so a dead session loses at most one wave.

```text
You are on the Mac Studio (M1 Ultra, 20-core CPU, 48-64 core GPU, 128 GB unified memory, 800 GB/s, 8 TB
SSD), repo at ~/hawking. Read in order before any work: docs/plans/M1ULTRA_POTENTIAL_AUDIT.md (the
scoreboard, the two ceilings, the spine, Wave 0), docs/plans/STUDIO_GO.md (the P0-P9 conveyor),
docs/plans/quintessential_engine_2026_06_29.md (the serve-build critical path), docs/plans/
hawking_handoff_2026_06_28.md (the codebase map), M1ULTRA_RUN_REPORT.md (the scoreboard). Hard rules: no em
or en dashes anywhere; never attribute Claude in git; NO commit or push without explicit approval;
vendor/strand-quant and tools/strand are audit-only (branch + PR, never direct main); Apple Silicon only
(Metal/MPS, no CUDA). Proof discipline, non-negotiable: EFFECTIVE bpw only (baker AGGREGATE incl RHT +
outlier + side-info), never nominal; quality = output-space ppl vs the f16 parent with MULTIWINDOW >= 4 AND
the multi_eval capability tripwire (a floor is void if ppl passes but a capability collapses); judge low-bit
on 7B+ never 0.5B; FAKE-WIN BAN (a served tensor rehydrated to f16 counts ZERO; spec counts ONLY under the
exact-match bit-lossless gate); no public WIN below repro level R3; a tie or near-tie is a NULL; every
candidate WIN gets an INDEPENDENT adversarial reproduction (frontier_verifier re-derives the recipe from
scratch at MULTIWINDOW=5) before it enters any doc; do NOT re-litigate the killed dead-levers (uniform
STE-QAT through the trellis, low-rank LoRA as the main lever, AWQ x residual, diverse > domain calib,
subbit_admm / NanoQuant, the madvise expert-pager in the free-RAM regime); see docs/dead_levers.md.

STANDING DIRECTIVE, unbounded wall-clock: there is no time limit; the box runs 24/7; optimize for MAXIMUM
proof and coverage, EXPAND not reduce. The slower M1 CPU lengthening every doctor pass, bake, and
floor-search is expected and accepted, never a reason to cut scope. Run the FULL model ladder, the WHOLE
L0-L6 recovery registry composed and ledgered per model, 30-seed protocols as the DEFAULT gate, and the
exhaustive codec bakeoff. The stop-check is knowledge convergence only, never a clock.

THE GOAL, set at the maximal ceiling on purpose (aim at it, report honestly against it): drive every
evaluated category from its proven-today score to its M1 Ultra maximal ceiling (see the scoreboard),
overall from ~3.25 toward ~8.4, by CONVERTING every time-wall and PROVING or CIRCUMVENTING every structural
wall. Concretely: (1) ANSWER THE TWO MOONSHOT GATES with R3+ receipts: does gradient recovery (the Doctor
L0-L6 stack) heal to <= +2% ppl at sub-4-bit on 7B-32B (18 GB never could, FAIL-001), and is MoE per-expert
sensitivity non-uniform on a REAL bake. (2) LAND the six-step serve-build critical path so P4/P7 flip GATED
-> MEASURED. (3) PROVE the RAM-cliff money demo: serve a model where llama.cpp Q4_K OOMs, native .tq folded
into the fused arena, > 10x tok/s and lower J/tok, on a resident 70B AND a resident 235B/405B (no pager
needed under 128 GB). (4) FIT the bit-floor-vs-scale law on many floor points from the full ladder. (5)
SCORECARD.md: every WIN cell backed by an R3+ receipt, zero unbacked GO. This box is a NEW instrument, not
the M2-Max-96GB plan's executor: the locked context (96 GB, 2 TB, 400 GB/s, 405B-is-cloud,
CPU-bf16-is-the-headline-device) was DERIVED from a box that never arrived. Re-derive it wave by wave per
the audit's re-derivation list. You will likely land short of every ceiling; the mandate is that every
category ends CONVERTED (R3+ receipt) or WALLED (a named mechanistic reason), so whatever number remains is
proven.

WAVE 0, once, before science: transfer repo + models onto the 8 TB SSD; cargo build + test --workspace
green INCLUDING the tests/ integration targets CI skips; FIX the studio_run.py:279/:286 unpack crash
(SPINE-0, the conveyor has never run end-to-end); run studio_run.py --go-plan (dry) to confirm all nine
phases wire; run hawking autotune on M1 Ultra and add the m1ultra row to size_frontier.py DEVICES (every
frontier number is currently computed for the wrong box); stage bf16 7B + 14B parents (the 32B on disk is
Q4_K GGUF, not bf16); DAY-0 MICROBENCH A: doctor throughput MPS/Metal-bf16 vs CPU-bf16 on a 7B step
(decides the headline-device doctrine); DAY-0 MICROBENCH B: native .tq decode tok/s on a full resident .tq
(not the 20 MB ffn_down toy), first streamed token + coherence, confirm M=1 predec GEMV hits >= 50% of 800
GB/s; PIN a small GGUF fixture so llama_cpp_oracle asserts instead of self-skipping; create
M1ULTRA_RUN_REPORT.md (scoreboard: category scores, open gates, tok/s + eff-bpw + J/tok, wave log). Commit
only with approval.

THEN LOOP until the goal is met or every category is walled. studio_run.py go is the conveyor (P0-P9);
resumable, RAM-packed, completed lanes skip. Each wave: (1) ORIENT: reread the scoreboard; list open
time-walls with expected quality-points or tok/s per unit of proof. (2) SELECT the highest-leverage lever.
SPINE ORDER: thesis gate (hawking-eval on the .tq-served model, gates every HIDE moat) -> moonshot gate 1
(P1 CONDENSE + the doctor L4-L6 resident, the full 27-model ladder, emit floor points, fit the law) ->
moonshot gate 2 (P4 FRONTIER resident 235B, needs a correct V3/GLM router first) -> serve/cliff receipts
that ride the artifacts -> in parallel and independent, SPINE-5: close the CI gap + make batched-verify
bit-identical. Never refine an owned number while an unbuilt instrument blocks a WIN cell; the deep frontier
(671B/744B, SSD-bound + router), the EAGLE-3 head, the RWKV-X recall hybrid, and the codec-quality push are
LAST. (3) PREREGISTER in code: the gate threshold (<= +2% ppl, effective bpw, MULTIWINDOW), the baseline
neutrality set (BASELINES.md, same memory pressure), the receipt level. (4) BUILD + RUN under the RAM
scheduler: disk kill-switch, resumable per-lane floor files, nohup + caffeinate; the 18 GB swap-death must
not recur. (5) VERIFY: frontier_verifier reproduces every ship-candidate from scratch; unverified = null;
the greedy-kernel losslessness property gate guards every spec claim. (6) LEDGER: emit the receipt, update
M1ULTRA_RUN_REPORT + SCORECARD, append nulls to FAILURES.md, gates green, commit plain (with approval). (7)
STOP-CHECK: two waves with no movement and no wall proof -> escalate (recovery ties on a loss-needing model
-> density-only ~3.3-3.8 bpw is the verdict; expert sensitivity uniform -> fall back to 405B dense) or write
the wall proof and close the category. DECISION GATES: recovery clears <= +2% at sub-4-bit on 7B+ or the
Doctor is bounded; expert sensitivity non-uniform or the sub-1-bit MoE lane is dead (671B@1.0=84GB is the
prize, else 405B@1.34=68GB dense resident); serve GEMV hits fused-arena parity + throughput or the RAM-cliff
stays GATED; batched-verify == greedy at near-ties or spec counts ZERO. Report each wave in one paragraph:
what ran, verdict, category movement, next lever. No wave ends without a committed (approved) artifact.
```

Usage: `/goal` on the Studio, or paste only the fenced block. The wrapper you are reading stays in the repo
as the record of what the prompt is and why. If the session dies mid-wave, the next session re-runs `/goal`
(or re-pastes the block); Wave 0 detects its own completion (build green + the crash fixed + the microbenches
recorded + `M1ULTRA_RUN_REPORT.md` exists) and the loop resumes from the scoreboard.
