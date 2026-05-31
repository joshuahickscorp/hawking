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

- [ ] **A1. Clean baseline + LM-head→predec.** Route the LM head through the
  predec GEMV; hoist nothing yet. Gate: bit-identical greedy; paired bench.
  Build-off: a *clean* absolute dec_tps anchor + the first +tps. (exact, High)
- [ ] **A2. Hoist audit.** Eliminate redundant per-token recompute (scale
  decodes, repeated norms) the §6 fusion left behind. Gate: bit-identical;
  bench. Build-off: cheaper per-token before the big kernel work. (exact, M-H)
- [ ] **A3. Wire f16s scales into decode + paired bench.** `_2r_f16s` is parity-
  validated (rel-L2 2.6e-4) + microbenched +5–12% (dispatch-bound). Wire the
  load-time f16 table behind `DISMANTLE_QWEN_PREDEC_F16SCALES`, paired-bench
  steady-state. Gate: rel-L2 quality + net dec_tps win. (Stage-2 BW #1, M)
- [ ] **A4. MST/xctrace per-kernel profile.** `mst_export.sh`→`mst_analyze.py`
  on a real `.trace`; name the dominant decode stall. No code change — this
  *targets* A5–A7 so they don't guess (the `_4r` lesson). Build-off: gates the
  next three. (profiling, High value)
- [ ] **A5. Vectorized nibble unpack (uint4 loads)** in the predec GEMV, aimed
  at A4's stall. Gate: bit-identical; bench. (exact, M)
- [ ] **A6. Threadgroup / occupancy tuning** of the predec GEMV (rows/TG, simd
  width), bench-driven from A4. Gate: bit-identical; bench. (exact, M)
- [ ] **A7. MLX-class simdgroup-matrix decode GEMV.** Prototype from
  `silicon-builds/dismantle-q4k-mma`; the big ceiling (~41%→60–80% peak BW).
  Gate: parity (atol 1e-3) + paired bench. Halt-and-log if it doesn't clear —
  hard/multi-session. (Stage-2 ceiling, hard)
- [ ] **A8. Q3_K predec wired into the dense path + Q3_K_M bench.** Kernel is
  validated (max_abs 1.5e-5). Requant Qwen-3B → Q3_K_M locally (llama.cpp;
  pessimistic-from-Q4 — flag it), run it on the predec path, measure dec_tps +
  local PPL. The byte-cut *speed* test the kernel unblocks. (byte-cut, M)
- [ ] **A9. §7.5 host loop = GPU-busy.** Zero-alloc persistent decode loop +
  GPU-side sampling + tightest CB reuse; drive `host_wall − Σgpu_us → 0`. Gate:
  bit-identical; bench. (exact floor, M-H)
- [ ] **A10. §7.1 access-order weight layout / coalesced repack.** Reorder
  weight bytes to decode access order; validate via busy-time BW. Gate:
  bit-identical; bench. (exact BW, M)

## Lane B — runtime / capability (parallel)

- [ ] **B1. Prefix-cache BUILD (§8 L1.2, the moat).** Implement the
  `PrefixCache` stub bodies (KV-block retention keyed by prefix hash), wire into
  the decode entry **behind a default-off flag**, validate bit-identical reuse
  on a multi-turn session, measure prefill-skip. Gate: bit-identical reuse +
  prefill-time saved. (build/moat, High)
- [ ] **B2. KV-working-set (§8 L1.1).** First a LOCAL attention-mass oracle
  (capture attention on a long-context prompt → fraction of tokens holding 99%
  mass per layer). If concentrated → build the eviction policy (StreamingLLM/
  H2O/SnapKV via the stub trait), bounded working set, lossless escape hatch.
  Gate: oracle GO, then quality-vs-context-length curve. (build, M; oracle-first)

## Lane C — measurement (cheap, anytime)

- [ ] **C1. Energy / joules-per-token (§8 L4.2).** Instrument with
  `powermetrics`/`macmon`; establish a joules-per-token baseline + report it
  alongside dec_tps. The branded axis nobody flies. (measurement, High)

---

## Progress log (main session updates as steps land)
- _run started 2026-05-31; kicked off A1 (GPU) + B1 (runtime) + C1 (cheap)._
