# Overnight build queue — 2026-05-31 (LOCAL ONLY, unattended)

Autonomous overnight haul. No Colab, no cloud, no user in the loop. Driven by
the main session: launch ONE agent for the next step → agent runs the gate →
main session evaluates + commits-on-pass / reverts+logs-on-fail → launch next.
**Executed CONSECUTIVELY — one agent at a time.** Serial is what was asked for
and it's the safest unattended: no concurrent edits to the shared decode path
(`qwen_dense.rs`), no git-index races, no GPU contention. The A/B/C grouping
below is logical, not concurrent; default run order is A1→A10, then B1→B2,
then C1 (with the clean baseline captured by A1's before/after bench).

## Execution protocol (unattended-safe — respects CLAUDE.md)
- **Gate every step.** Parity (bit-identical for exact levers, atol 1e-3 for
  fp16-lossy, relative-L2 for quality-trade) + paired bench under the §1 gate.
- **Commit-on-pass only.** Single-purpose commit, inline Joshua Hicks identity,
  **no attribution, NO PUSH** (user reviews + pushes in the morning).
- **Halt-don't-thrash.** ≤2 attempts per item; on failure the main session
  saves the agent's diff to `reports/<step>.patch` + findings, **reverts the
  working tree to the last good commit** (so the next step builds clean), and
  moves to the next item. Apply the Kill Protocol (Type-1/2 + named oracle) to
  any NO-GO.
- **Agents do not commit** (avoids index races); the main session commits each
  step on pass, staging that step's files only. Never touch the user's
  uncommitted `throughput_bible`/`roadmap` edits.
- **Default-off for risky paths.** Anything touching the live decode path
  (LM-head reroute, prefix-cache) lands behind an env flag / bit-identical gate,
  enabled only if it validates exactly.
- **One agent at a time.** No concurrent source edits; the GPU is never contended.

---

## Lane A — kernel / tps (serial, GPU)

- [x] **A1. Clean baseline + LM-head→predec.** DONE — PASS. Bit-identical
  32-tok; paired dec_tps **30.94 → 31.55 (+2.0%)**; clean baseline anchor ≈31
  (the 36.9 was contaminated). Flag `DISMANTLE_QWEN_LMHEAD_PREDEC` default-on.
  Committed.
- [x] **A2. Hoist audit.** DONE — NO-CHANGE. Path already tight: zero per-token
  weight allocs/dequants/predecode in `forward_token_greedy_tcb`; all
  token-invariant work hoisted to load (predec tables, requant, PSOs). Only
  leftover = RoPE `pow` (not bit-identical to hoist + sub-noise 0.2–2.4%).
  Note for **A9:** `KernelArgBuffer::new` allocs per dispatch — real host churn,
  not token-invariant (carries `pos`) → buffer-pool refactor belongs in A9.
- [x] **A3. Wire f16s scales into decode + paired bench.** DONE — NO-GO
  (marginal, Type-2). Quality perfect (**0/96 token drift**, 73/73 tests) but
  dec_tps only **+2.5% median (noise floor)** — the dominant fused gate+up pairs
  stay f32 (no f16s *pair* kernel), so too little scale traffic is cut. Validated
  wiring saved → `reports/overnight_patches/a3_f16s_decode_wiring.patch`; tree
  reverted to A1. **Revisit only after an f16s _pair_ kernel exists**, then
  re-apply the patch.
- [x] **A4. Per-kernel decode profile.** DONE (committed report). MST couldn't
  attribute (Claude.app compositor owned the GPU intervals) → §1-clean gpu_prod
  fallback. **predec GEMVs = 89.4% of decode**; dominant = `predec_pair` (fused
  gate+up) **46.6% @ ~56% peak BW (~1.6-1.8× headroom)**; `predec_2r` 42.8%.
  → A5/A6 target `_pair` first. Also explains A3 (f16s never touched `_pair`).
- [x] **A5. Vectorized nibble unpack (uint4).** DONE — NO-CHANGE, **Type-1
  kill**. The byte loads are already coalesced at *simdgroup* granularity (32
  lanes → 32 contiguous bytes = one transaction); per-thread they're stride-32,
  so a `uint4` can't apply without reordering the bit-identical FMA chain. Death
  is an Apple-GPU memory-model fact (do not re-test). Redirect: the 56%-peak
  stall is occupancy / scale-read / x-traffic, not load width → **A6 + A10**.
- [x] **A6. Threadgroup / occupancy tuning.** DONE — NO-CHANGE, **Type-1
  (BW-bound, not occupancy-bound)**. `_pair` is oversubscribed (76 TGs/core vs
  ~24-32 ceiling), no idle cores, no shmem/barrier lever. tg128 +0.9% (noise),
  tg384 −0.2% — below gate. Reverted clean. → the BW gap is layout (A10) +
  scale-byte volume (A6.5 below), not geometry.
- [ ] **A6.5 (INSERTED, profile-driven). f16s _pair_ kernel + full f16s wiring.**
  A4 says `_pair` (46.6%) dominates; A5/A6 say it's BW-bound; A3 showed f16s is
  quality-clean (0/96 drift) but marginal because `_pair` stayed f32. Build
  `gemm_q4_k_v4_predec_pair_f16s` (clone `_pair`, f16 scales — mirror how
  `_2r_f16s` cloned `_2r`), wire BOTH pair + non-pair predec through f16s under
  `DISMANTLE_QWEN_PREDEC_F16SCALES` (re-use the A3 patch for the non-pair half),
  so f16s now covers ~89% of decode. Gate: rel-L2 quality + token-drift small +
  dec_tps win ≥ +3%. (Stage-2 BW, the real f16s test, M-H)
  → **DONE — PASS (committed). +6.1%/+8.9% (two runs), 0/32 drift on code AND
  prose, rel-L2 2.5e-4, 4 parity + 73 lib tests green. DEFAULT-OFF opt-in
  (`DISMANTLE_QWEN_PREDEC_F16SCALES=1`). Flip default-on only after a broad
  N-prompt drift sweep. The night's headline win.**
- [x] **A7. MLX-class simdgroup-matrix decode GEMV.** DONE — NO-GO / **Type-1
  (decode)**. `dismantle-q4k-mma` is M>1/prefill (8×8 MMA tile, its VERDICT.md:
  "dead for decode GEMV N=1"). Decode is BW-bound (MMA cuts compute, moves no
  bytes off the bus) + M=1 underfills the units 7/8. No decode oracle → stays
  dead. **Prefill MMA (silicon #8) is a SEPARATE live lever** (Stage-5 TTFT),
  not decode. No change. (assessed in 66s, no wasted build)
- [ ] **A8. Q3_K kernel microbench (byte-cut speed premise).** RESCOPED for
  unattended safety — a full Q3_K dense-path is too big to wire overnight.
  Instead microbench the Q3_K GEMV options on Qwen shapes: **Q3_K fused (110
  B/block)** vs **Q3_K predec (160 B/block — predec ADDS f32 scale bytes)** vs
  **Q4_K predec (192 B/block)**. Answers: (a) which Q3_K kernel is fastest
  (predec's compute-saving vs fused's fewer bytes), (b) does Q3 (fewer bytes)
  beat Q4 in GEMV time = the byte-cut speed premise. No model serving / no
  dense-path wiring. (byte-cut characterization, M)
  → **DONE (oracle committed).** Q3_K FUSED is the right byte-cut kernel (fewer
  bytes; the committed Q3_K *predec* adds 64 B/block for no gain). **Byte-cut
  SPEED premise FAILS today: Type-2** — fastest Q3_K is 22-43% SLOWER than
  Q4_K-predec because the Q3_K kernels run at 6-33 GB/s (NOT BW-bound; lack the
  2r/pinned fast path). Named oracle = re-run this bench after a Q3_K GEMV is
  rewritten to the Q4_K-predec standard. Byte-cut value = footprint (−11%), not
  speed, until then.
- [~] **A9. §7.5 host loop = GPU-busy.** SKIPPED (Kill Protocol — recorded
  Type-1). Host CPU-encode overhead is a recorded kill (`cpu_gpu_pipelining`,
  `icb`: ≤0.5-0.9% of wall, ceiling +0.14 tps); the A2-flagged `KernelArgBuffer`
  per-dispatch alloc is a subset of that already-measured envelope. Don't
  re-test a recorded Type-1 kill. Revisit only at 100+ tps where host overhead
  becomes a real fraction (bible §7.5).
- [x] **A10. §7.1 access-order weight layout / coalesced repack.** DONE —
  HALT-WITH-DESIGN, **Type-1 (built+measured, not inferred).** Bit-identical
  per-thread-contiguous repack = **−16.8%** (de-coalesces the simdgroup;
  stride-32 is already the HW optimum); vectorized variant not bit-identical +
  no gain. Confirms A5 empirically. Design note + kill recorded; tree reverted.
  **→ kernel-microopt BW track EXHAUSTED** (A5/A6/A7/A10 Type-1; A6.5 the only
  win). Decode GEMV is at the memory-model optimum; more tps needs fewer bytes
  (Q3, A8) or spec/stateful axes.

## Lane B — runtime / capability (parallel)

- [x] **B1. Prefix-cache BUILD (§8 L1.2, the moat).** DONE — **PASS
  (committed).** `InMemoryPrefixCache` (longest-strict-prefix, exact KV
  snapshot/restore, LRU) behind `DISMANTLE_QWEN_PREFIX_CACHE` (default-off).
  **Bit-identical reuse on real Qwen-3B (TCB path), re-verified green (58s);
  prefill 5551→892 ms (~84% cut)**, 81 lib tests pass. Fixed a TCB-arena stale-KV
  subtlety (note: the disk cache has the same latent bug). Opt-in until a
  PrefixCacheBudget byte cap is wired (then default-on is plausible — exact
  reuse). The differentiated capability, landed.
- [x] **B2. KV-working-set (§8 L1.1).** DONE — NO-GO mid-ctx, **Type-2
  (regime-limited).** Built the attention-capture instrument (committed,
  default-off, bit-identical) — the design's missing prereq. Finding (586-tok
  code): attention DIFFUSE — 99% mass needs 78-92% of positions; sinks+recent
  covers only 18-73%. StreamingLLM/H2O/SnapKV all die here. **Type-2 reframe =
  LONG context (>16K)** where eviction literature's sparsity sharpens — unrun
  (the built instrument is the named oracle; re-run at 16-32K / prose / larger
  Qwen). Eviction NOT built (halt-with-design).

## Lane C — measurement (cheap, anytime)

- [x] **C1. Energy / joules-per-token (§8 L4.2).** DONE — HALT-WITH-FINDING
  (tooling, not a kill). No sudo-free power source unattended (`macmon` not
  installed; `powermetrics` needs sudo; IOReport not loadable; battery = whole-
  machine AC). Built turnkey `tools/bench/measure_joules.sh` (committed,
  sudo-refusing, `--f16s` compares A6.5). **Morning 1-liner:**
  `brew install macmon && tools/bench/measure_joules.sh --tokens 256 --f16s`.

---

## Progress log (main session updates as steps land)
- _run started 2026-05-31, consecutive (one agent at a time)._
- **A1 PASS** — LM-head→predec, bit-identical, 30.94→31.55 dec_tps (+2.0%); clean baseline ≈31. Committed. → launching A2.
- **A2 NO-CHANGE** — decode path already tight (all token-invariant work hoisted); RoPE-pow leftover not worth parity risk. No commit. → launching A3.
- **A3 NO-GO** (marginal) — f16s decode-wiring +2.5% median (noise floor), 0/96 token drift; dominated by f32 fused pairs. Patch saved, tree reverted to A1. → launching A4.
- **A4 PASS** (profile) — predec GEMVs 89.4% of decode; `predec_pair` 46.6% @ 56% peak BW = the stall. Report committed. → launching A5 (uint4 unpack on `_pair`).
- **A5 NO-CHANGE** (Type-1) — loads already simdgroup-coalesced; uint4 inapplicable + would break bit-identical. Stall is occupancy/scale/x, not load width. No commit. → launching A6 (occupancy).
- **A6 NO-CHANGE** (Type-1, BW-bound) — `_pair` oversubscribed, no occupancy lever; geometry sweeps noise. → BW gap is layout/scale-volume. **Inserted A6.5** (f16s _pair_ kernel) as the profile-indicated next build. → launching A6.5.
- **A6.5 PASS (committed)** — f16s pair kernel, **+6-9% decode** opt-in, 0/32 drift code+prose, default-off. THE win of the night (profile arc A3→A4→A5→A6→A6.5). → launching A7.
- **A7 NO-GO** (Type-1, decode) — MMA is a compute lever; decode is BW-bound + M=1 underfills (silicon's own VERDICT.md agrees). Prefill MMA = separate lever. No change, 66s. → launching A8 (Q3_K kernel microbench, rescoped).
- **A8 DONE** (oracle committed) — byte-cut SPEED Type-2-dead (Q3_K 22-43% slower than Q4_K-predec; Q3_K kernels not BW-bound, lack 2r/pinned). FUSED is the right Q3 kernel. Footprint lever (−11%) alive. → A9 skipped, launching A10.
- **A9 SKIPPED** (Kill Protocol) — host-loop = recorded Type-1 kill (CPU-encode ≤0.5% wall); don't re-test; revisit at 100+ tps.
- **A10 HALT-WITH-DESIGN** (Type-1, built+measured) — bit-identical repack −16.8% (de-coalesces); kernel at HW optimum. Kill + design committed. **Kernel-BW track exhausted (A6.5 the lone win).** → launching B1 (prefix-cache moat).
- **B1 PASS (committed)** — prefix-cache moat: **bit-identical KV reuse (real Qwen-3B, re-verified), ~84% prefill cut**, default-off opt-in. The differentiated capability. → launching B2 (KV-working-set oracle).
- **B2 NO-GO mid-ctx (Type-2)** — built+committed the attn-capture instrument (default-off, bit-identical); attention diffuse at 586 tok (99% mass needs 78-92% of positions). Long-context (>16K) reframe unrun; instrument is the named oracle. → launching C1 (energy).
- **C1 HALT-WITH-FINDING** — no sudo-free power source unattended; built turnkey measure_joules.sh (committed). One-command attended recipe recorded. **QUEUE COMPLETE.**

---

## CLOSEOUT (2026-05-31, queue complete)

**3 real wins committed** (all gated, all LOCAL/unpushed):
- **A1** LM-head→predec: +2.0%, **bit-identical**, default-on. Clean baseline ≈31 dec_tps (the 36.9 was contaminated).
- **A6.5** f16s-scales covering the FFN gate+up pair: **+6-9%**, 0/32 drift (code+prose), opt-in `DISMANTLE_QWEN_PREDEC_F16SCALES` (default-off; default-on after a corpus drift sweep).
- **B1** in-RAM prefix cache (the moat): **bit-identical KV reuse, ~84% prefill cut**, opt-in `DISMANTLE_QWEN_PREFIX_CACHE` (default-off; default-on after a byte-budget cap).

**Tooling/oracles committed:** A4 per-kernel profile; A8 Q3_K byte-cut microbench; B2 attention-capture instrument; C1 joules harness.

**Honest kills (the discipline working):** kernel-microopt BW track EXHAUSTED — A5 uint4 (Type-1, coalesced), A6 occupancy (Type-1, BW-bound), A7 MMA (Type-1, compute lever vs BW-bound+M=1), A10 layout (Type-1, built+measured −16.8%). A6.5 was the lone BW win → **the Q4_K predec decode GEMV is at the Apple-GPU memory-model optimum.** A2 no-change (path already tight). A9 skipped (recorded Type-1 host kill). A3 f16s-non-pair NO-GO (superseded by A6.5 — its patch `reports/overnight_patches/a3_f16s_decode_wiring.patch` is now OBSOLETE, discard).

**Type-2 reframes (alive, named oracle each):** byte-cut SPEED (A8 — needs a 2r/pinned Q3_K kernel; re-run q3k_bytecut_bench); KV-working-set (B2 — re-run the attn-capture instrument at 16-32K / prose / larger Qwen).

**MORNING TO-DO:**
1. Review + **push** ~15 local commits (`git log origin/codex/maximal-spec-colab..HEAD`).
2. Decide default-on for A6.5 (drift sweep) + B1 (budget cap) — both exact/clean, just need the safety wiring.
3. Energy baseline: `brew install macmon && tools/bench/measure_joules.sh --tokens 256 --f16s`.
4. Pre-existing issues (NOT from this haul): `tests/v1_1_phase5A_batched_forward_parity.rs` stale `SpeculateMode::NGram` compile error (commit 822e779); the on-disk prefill cache has the same TCB-arena stale-KV latent bug B1 fixed for the RAM tier.
5. Discard the obsolete A3 patch (A6.5 supersedes it).

---

## Phase 2 — next-lever queue (post the parallel lanes; the cron/chain pulls from here)

Grounded in the finding: **decode kernel-microopt is exhausted** (GEMV at the HW optimum), so the live frontiers are **fewer bytes**, **prefill/TTFT**, **stateful**, and the **one spec survivor**. Prioritized; each is a local agent-task with a gate. One source-agent at a time (worktrees for parallel); GPU work serializes.

- **P1. Prefill MMA → TTFT (silicon #8 port).** A7 confirmed simdgroup-MMA is dead for decode (M=1) but LIVE for **prefill (M>1)**. Port `silicon-builds/dismantle-q4k-mma` into the prefill path. Gate: bit-identical prefill + TTFT win on a COLD prompt (complements B1's cache-HIT prefill cut). GPU. **[High — clear regime fit, A7-flagged]**
- **P2. n-gram batched-verify-with-logits.** The n-gram lookahead is green but perf-parity (serial verify, τ=1.43 — the only spec survivor). Unlock = verify K draft tokens in ONE batched forward with logits. Gate: bit-identical greedy + dec_tps win on copy-heavy code. GPU. **[Medium — sub-gate τ, but batched-verify could bank a small code-workload win]**
- **P3. Long-context fused quantized-KV attention.** Read 4/8-bit KV inline (no FP16 buffer, mlx-qsdpa pattern) → cut KV bandwidth + memory at long ctx. **Gated on Lane C's B2 long-ctx result** (if attention concentrates at >16K, this + eviction unlock 200K+ ctx). Gate: parity atol-1e-3 + KV-byte cut at >16K. GPU. **[Medium — long-file-context capability]**
- **P4. Sub-3-bit byte-cut (QTIP-class) — CHAINED on Lane 2.** If Lane 2's 2r fused Q3_K makes the byte-cut speed-viable, the next step is sub-3-bit (QTIP lookup-free trellis — the surviving codec): more footprint cut, now with speed. Local quant (llama.cpp) + the Q3-class kernel. Gate: PPL quality + dec_tps. **[Medium — depends on Lane 2 PASS]**
- **P5. §8 L3.1 online vocab/draft specialization.** Prune the output head to the vocab actually in use + tune the draft on accept/reject history — compounding user-specific, exact (certifiable vocab screen). Local runtime; needs accumulated usage. **[Lower — the moat++, data-dependent]**

Deferred/attended (not auto-pullable): §7.6 distillation (different model), §8 L3.3 on-device LoRA (heavy training), the A10 layout (Type-1 dead), §7.5 host loop (recorded Type-1, revisit at 100+ tps).

**P4 UPDATE (Lane 2 result):** the Q3 byte-cut SPEED reframe is **FALSIFIED**. Lane 2 built `gemm_q3_k_fused_2r` (committed `c1f5275`, recorded-dead/unwired) and measured: 2r helps only the square shape (+8%), regresses wide FFN (−5 to −30%), and stays −32 to −55% slower than Q4-predec. Root cause: **Q3_K GEMV is COMPUTE-bound on the inline 6-bit scale decode** (7-21 GB/s on a 150 GB/s machine), NOT bandwidth-bound — so row-ILP (hides DRAM latency) targets the wrong bottleneck. ⇒ P4 (sub-3-bit speed) is blocked by the SAME root cause. A real Q3/sub-3-bit speed win needs a **cheaper-decode layout** (not row-ILP / not predec which adds scale bytes) — research-y, out of scope. Byte-cut value stays **footprint-only (−11%)**.

---

## Parallel-lanes run (post-closeout) + lessons

- **Lane 2 (Q3 2r kernel): HALT** — byte-cut not speed-viable (see P4 update). `c1f5275` on main (its worktree isolation FELL BACK — see lesson).
- **Lane 1+3 (prefix-cache cap/default-on) + benches:** in flight.
- **LESSON — worktree race:** launching TWO `isolation:worktree` agents off the SAME branch simultaneously is racy — git can't put two worktrees on one branch, so one (Lane 2) fell back to the MAIN tree and committed to `codex/maximal-spec-colab` directly. **Launch worktree agents one at a time** (or off distinct bases). The cron's "one source-agent at a time" rule already avoids this.

---

## Handoff #1 — branch review findings (2026-05-31) + revised plan

A parallel read-only session reviewed the haul commits (`plans/handoff_branch_review_2026_05_31.md`). **Load-bearing guarantees HOLD** (f16s flag-off-unchanged; prefix-cache bit-identical reuse, verified line-by-line). **Verdict: f16s NO-GO for default-on; prefix-cache GO-WITH-FIXES.**

- **[HIGH] Disk-tier prefill cache persists STALE KV on the TCB path.** `qwen_dense.rs:1519` `cache.store(key, &self.kv)` runs with NO `mirror_arena_kv_into_self` and BEFORE the RAM mirror (:1535). On TCB, prefill K/V live only in the GPU arena → the disk entry stores stale/zeroed KV → **silent corruption on a later-process disk hit** (reachable: `DISMANTLE_PREFIX_CACHE_DIR` + TCB). Pre-existing (not B1).
- **[HIGH, blocks default-on] RAM cache UNBOUNDED.** `PrefixCacheBudget::default` = None/None → no eviction → OOM risk. `evict_to` is correct, just unarmed. (= exactly the Lane B re-do's job.)
- **[MED] f16s routes the LM HEAD → argmax-sensitive.** Kernels sound (flag-off byte-identical; FMA order == f32 pair; production `narrow` == parity-test `predecode_q4_k_scale_table_f16`; ragged guards OK). But not bit-identical (rel-L2 2.5e-4), and f16 rounding on the LM-head logits can flip a near-tie argmax — a 2-prompt drift check wouldn't catch it.
- **[LOW]** RAM collision guard is length-only (rests on SHA-256; comment overstates). Add a `debug_assert_eq!` on (n_layers,n_kv_heads,head_dim,max_seq) vs arena in `mirror_arena_kv_into_self`.
- **P2 CLEAN** (e0fdf80 was the lone struct/set_bytes mismatch; all other wrappers correct). **P3 security CLEAN** (RAM cache in-process; new `unsafe` bounds-correct).

**Revised action items (queued; GPU/tree busy with Lane C):**
1. **Lane B re-do** = cap the RAM cache (ship a non-None default budget) + LOW hardening (debug_assert, fix the collision comment) → RAM `DISMANTLE_QWEN_PREFIX_CACHE` **default-on** (review's GO, post-cap).
2. **NEW [HIGH] fix:** hoist `mirror_arena_kv_into_self` above the `:1519` disk store (or gate the disk store off on TCB). Gate: a disk hit reproduces no-cache output bit-identical.
3. **f16s default-on REFINEMENT:** keep the **LM head on the f32 predec path** (argmax-exact) and f16s ONLY the FFN pair (where the +6-9% lives) → broad drift sweep → re-eval default-on. Until then f16s stays opt-in.
- **⚠ pre-existing (flag for morning):** `tests/v1_1_phase5A_batched_forward_parity.rs` fails to compile at HEAD (stale `SpeculateMode::NGram`, from old commit 822e779 — NOT from this haul). Isolated to that one test binary; lib + all haul tests compile fine.
