# Session closeout — 2026-05-31 (review + prefix-cache ship + Phase-2 design/handoff)

**Branch:** `codex/maximal-spec-colab` — **12 commits ahead of origin, all LOCAL (unpushed).**
**Final state:** GPU free, no agent active, tree clean except 3 untracked junk files.

## What landed (this session's commits)
- **`fc93ea0` — prefix-cache default-ON + disk-tier stale-KV fix + 3 GiB cap.** The §8 L1.2
  moat lever flipped default-on (opt-out `env_opt_out`), with the two review-blocking fixes:
  (1) disk-tier stale-KV corruption — `mirror_arena_kv_into_self` hoisted ahead of *both* cache
  stores (flag-off decode path byte-unchanged); (2) unbounded RAM cache → bounded
  `DEFAULT_MAX_BYTES = 3 GiB` LRU. **Gate verified myself (release, real Qwen-3B):** RAM e2e
  2/2 bit-identical (hit+miss), disk e2e 3/3 (hit-matches-no-cache + miss + baseline). HIT
  prefill: RAM 16.1% of off, disk 7.4–11.3% of no-cache (absolute ms contaminated; ratios valid).
- **`290a15e` — bible §3.0 + §8 framework + roadmap forward path** (the user's 369-line edit:
  decode-kernel-microopt CLOSED at ~31; ~39-vs-~31 anchor UNRECONCILED; contamination first-class;
  the 15-lever 5-layer moat framework + Kill Protocol §8.3.1 + Phases A–E).
- **`e3ca9ab` — two Phase-2 design docs:** `cheaper_decode_q3_design` (f16-predec-Q3 on a repacked
  128-B block, −27%/−38% bytes; free pre-build oracle = re-run `q3k_bytecut_bench`) +
  `stateful_moat_continuation_design` (semantic cache + L3.1 vocab/draft specialization).
- **`819631e` — handoff: stateful-moat L3.1 first move** (3 NumPy oracles then `usage_capture`).

## Decisions / kills recorded
- **f16s predec GEMV → KEEP OPT-IN, NO-GO for default-on** (8.2% category-correlated drift: clean
  on code/SQL/factual/lists, drifts 18–30% on math/dialogue/prose). Already recorded (Lane C, `2aabcaf`).
- **Prefix-cache → GO-WITH-FIXES → shipped** (fixes applied + verified in `fc93ea0`).
- **Branch correctness/security review (handoff #1):** completed; both default-on guarantees hold,
  one reachable HIGH disk-corruption bug found + fixed.

## In flight / preserved (NOT lost)
- **P1 prefill-MMA** — attempted, **stopped mid-build by the user**, preserved in **`stash@{0}`**:
  both MMA kernels (`gemm_q4_k_m_batched_v3w_mma` + predec twin, `simdgroup_multiply_accumulate`,
  one-simdgroup/TG) drafted in `quant.metal`; **no wrappers, no wiring, no parity yet.** Resumable.
- Older sibling-session stashes `stash@{1}`/`stash@{2}` — untouched, unrelated.

## What attended work unblocks (2 open decisions)
1. **L3.1 oracle lane** — launch as a CPU background agent *here* (parallel-safe, not GPU-
   contaminated) vs. a separate session. Spec ready: `plans/handoff_l31_moat_oracles_2026_05_31.md`.
2. **P1** — resume the MMA port (`stash@{0}`) or leave stashed.

## Followups (housekeeping, none urgent)
- **Clean-room benches** (need Claude quit — absolute-metric, contamination-sensitive): the
  ~39-vs-~31 anchor reconciliation (bible §3.0 open item), the Q3 free oracle (GB/s threshold),
  the energy baseline (`tools/bench/measure_joules.sh` + macmon 0.7.2, ready).
- **Push decision:** 12 commits local-only; user pushes when ready.
- **Junk:** `on_smoke.err`, `server_smoke.err` (tiny stderr), `traces/` (~250 MB MST profiling).

**Next session opener:** `plans/handoff_next_session_2026_05_31.md`.
