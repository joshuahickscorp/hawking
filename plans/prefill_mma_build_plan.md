# Prefill simdgroup-MMA (silicon #8) — build plan from the stashed kernels

**Date:** 2026-05-31 · **Scope:** PREFILL only (the batched Q4_K GEMM at B>1).
Decode-kernel micro-opt is a **closed Type-1** front (bandwidth-bound at the
HW memory-model optimum — `overnight_haul_2026_05_31`, `dead_levers.md`), so
this lever does NOT touch the M=1 decode GEMV. It augments the prefill /
batched-verify path that already ships behind `DISMANTLE_QWEN_BATCH_PREFILL=1`.

This is a **plan only** — no stash pop, no code edits, no GPU runs were
performed. Every perf number below is tagged `(measured)`, `(proxy)`, or
`(estimate)`; the single decisive gate is named at the end.

---

## 0. TL;DR verdict

- **What exists:** `stash@{0}` holds **2 Metal kernels** in
  `crates/dismantle-core/shaders/quant.metal` (+205 lines, applied not popped):
  `gemm_q4_k_m_batched_v3w_mma` and its predec twin
  `gemm_q4_k_m_batched_v3w_mma_predec`. **No wrappers, no dispatch wiring, no
  parity test, no prewarm entry, no env flag.** They are inert source text.
- **What the kernels are:** the plain **8-row × 8-N single-simdgroup MMA tile**
  (one simdgroup / TG, `ceil(rows/8)` TGs). NOT the N-wide W-reuse variant
  (`gemm_q4k_mma_nwide`) that the silicon #8 VERDICT and the tier-1 handoff §1.1
  flagged as the large-N prefill winner.
- **The verdict already on record** (`plans/p1_prefill_mma_integration_handoff_2026_05_31.md`,
  `dead_levers.md` §"Q4_K batched MMA"): the plain MMA is a **+22–24% (measured,
  in-tree microbench, ±1%)** WIN on the **tall ffn gate/up shape (11008×2048,
  rows>cols)** and a **LOSS on square/wide** (ffn_down 2048×11008 −8.8%; attn
  q/o 2048×2048 −10–16%) — a Type-1 occupancy reality. So MMA is **shape-gated
  to rows>cols**.
- **The blocker that makes this attended, not a force-merge:** the shipped
  batched path is **predec-default-ON**, and the predec cache **covers ffn_gate
  + ffn_up** (`qwen_dense.rs:2897–2898`). So the v3w-MMA, wired only into the
  v3w `else` branch, is **DORMANT on the exact shape it wins.** The shipped TTFT
  win requires wiring the **predec-MMA twin** into the predec branch. Option A
  (v3w-MMA only) lands a parity-safe but **dormant** kernel; **Option B** (predec
  twin in the predec branch) is the only path that moves shipped prefill.

---

## 1. The stash, read read-only (`git stash show -p stash@{0}`)

Stash label: *"P1 partial: 2 MMA batched-prefill kernels (quant.metal only — no
wrappers/wiring/parity yet)"*. Diff is **one file**:
`crates/dismantle-core/shaders/quant.metal`, +205 lines, inserted right after
`gemm_q4_k_m_batched_v3w` (~line 1773) and before the predec kernel block.

### Kernel 1 — `gemm_q4_k_m_batched_v3w_mma`
Buffers: `(0)` w_q4 `uchar*`, `(1)` x_batch `float*`, `(2)` y_batch `float*`,
`(3)` `ArgbufBatchedRowsCols{rows,cols,batch}`, `threadgroup(0)` shmem `float*`.
- **Geometry (differs from v3w):** grid `(ceil(rows/8)*32, 1, 1)`, threadgroup
  `(32,1,1)` = **one simdgroup, 8 rows/TG**. (v3w is 256 threads / 8 simdgroups
  / TG.) N = batch ∈ 1..=8; tile columns N..8 are zero-padded.
- **Shmem:** fixed **576 f32 = 2.25 KB** regardless of batch
  (`Ws[256]` = 8 rows×32 K, `Xs[256]` = 32 K×8 N, `Os[64]` = 8×8 result).
  Note this is **independent of batch** — unlike v3w/predec whose wrapper sets
  `batch*256*4`. The wrapper for this kernel must set `576*4 = 2304` bytes.
- **Math:** dequant the 8×32 Q4_K weight tile into `Ws` (decoding the 144-B
  block header per element — fp16 d/dmin + 6-bit s/m), stage `Xs`, then
  `simdgroup_load`/`simdgroup_multiply_accumulate` in **4 depth-8 MMA steps per
  32-wide K sub-block, 8 steps per 256-block**. Output `C[m][n]` →
  `y_batch[n*rows + (row0+m)]` (matches v3w's transposed-batch output layout).
- **Activation dtype is `float`** (not `half` as in the standalone silicon #8
  bench) — i.e. the stash already adapted the kernel to the in-tree f32 staging
  contract. Good: parity vs the existing f32 v3w is apples-to-apples.

### Kernel 2 — `gemm_q4_k_m_batched_v3w_mma_predec`
Identical geometry + MMA loop; buffers shift by one to insert
`(1) scales float*` (16 f32/block predec table: `ds, dm` per sub-block). The
per-element header decode is replaced by `ds*nib - dm` (mirrors
`gemm_q4_k_m_batched_v3w_predec`). Weight nibbles still read from `w_q4`.

### What is MISSING (the entire integration surface)
1. **Dispatch wrappers** in `src/kernels/mod.rs` — none. Need
   `gemm_q4_k_m_batched_v3w_mma_pinned_tcb` and
   `..._mma_predec_pinned_tcb` (clone of the v3w / v3w_predec wrappers at
   `mod.rs:595` / `:671`, with the corrected grid + shmem; see §4).
2. **Parity test** — none. Need `q4k_batched_mma_parity.rs` (clone of
   `tests/q4k_batched_predec_parity.rs`; see §5).
3. **Call-site wiring** in `qwen_dense.rs` — none. The shape-gated swap inside
   `batched_proj!` + the ffn_down branch (§3, §6).
4. **Env flag** — none. `DISMANTLE_QWEN_Q4K_MMA` reserved but unread.
5. **Prewarm entry** — `gemm_q4_k_m_batched_v3w_mma` not in `QWEN_TCB_KERNELS`
   (`qwen_dense.rs:1126`), so first prefill would eat a JIT compile.

> The handoff doc also references a **preserved branch**
> `worktree-agent-a08c1cb44eb3d4e47 @ c9b1c07` (base `22dd6f4`) that *did* build
> the wrappers + wiring + parity test. **Do NOT 3-way merge it** — base
> `22dd6f4` predates the `stateful/` module + the batched predec kernel, and
> cherry-picking produces a ~1384-line misaligned `qwen_dense.rs` conflict
> (handoff §"Why deferred" #1). The stash kernels + this plan are the **clean
> re-derivation** path; treat the branch only as a reference for the wrapper
> bodies if useful, then discard it.

---

## 2. The current prefill GEMM path the kernel augments

Entry: **`forward_tokens_batch_tcb`** (`qwen_dense.rs:4702`), gated on
`DISMANTLE_QWEN_BATCH_PREFILL` (probe at `:1461`). Per-layer it calls the
**`batched_proj!` macro** (`:4865`) for q/k/v/o and ffn_gate/ffn_up, and a
dedicated ffn_down branch (`:5063`).

The macro's Q4_K arm (`:4869–4890`) is exactly:
```rust
if let Some(scales) = predec_cache.and_then(|c| c.get(&$tref.offset)) {
    kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(...);   // predec branch
} else {
    kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(...);          // v3w branch
}
```
`predec_cache = self.q4k_predec_cache.as_ref()` (`:4832`).

**The two MMA-winning shapes (rows>cols) and how they route today:**

| site | line | rows×cols | which branch (predec ON) |
|------|------|-----------|--------------------------|
| ffn_gate | `:5037` | intermediate×h = **11008×2048** | **predec** (cache covers ffn_gate, `:2897`) |
| ffn_up   | `:5043` | **11008×2048** | **predec** (cache covers ffn_up, `:2898`) |
| ffn_down (requant Q4_K) | `:5067` | h×intermediate = 2048×11008 (rows<cols → MMA LOSES) | predec |
| q_proj | `:4926` | q_dim×h = 2048×2048 (square → MMA LOSES) | predec |

**This is the crux.** The MMA wins *only* on ffn_gate/ffn_up (rows>cols), and
those go through the **predec** branch — so a v3w-`else`-branch MMA never fires
in the shipped (predec-on) config. Confirmed against
`ensure_q4k_predec_cache` (`:2865–2901`), which inserts q/k/v/o/gate/up/down.

> **DISMANTLE_QWEN_BATCH_PREFILL is itself default-OFF** (`env_on` at `:1461`).
> So the prefill MMA only matters when batched prefill is enabled. If batched
> prefill is not on the shipped default, the realized TTFT win is conditional on
> that flag too — name both flags in the bench (§7).

---

## 3. Where the new kernel lands (decision: Option B, the shipped win)

Per the handoff, **Option A** (wire only the v3w-MMA into the `else` branch) is
parity-safe but **dormant in shipped** — it fires only with
`DISMANTLE_QWEN_Q4K_PREDEC=0`, which nobody runs. **Do Option B:** wire the
**predec-MMA twin** into the **predec** branch, shape-gated to `rows > cols`,
behind `DISMANTLE_QWEN_Q4K_MMA`. Land Option A's v3w-MMA in the same patch (it
costs ~10 lines and gives a predec-off parity anchor), but the value is in B.

Target the swap at the **macro level** so all `batched_proj!` sites inherit it,
but the shape gate (`rows > cols`) means it only actually swaps on ffn_gate /
ffn_up (q/k/v/o are square or short; ffn_down is wide). Concretely the predec
arm of `batched_proj!` becomes:
```rust
if let Some(scales) = predec_cache.and_then(|c| c.get(&$tref.offset)) {
    if mma_on && $rows > $cols {
        kernels::gemm_q4_k_m_batched_v3w_mma_predec_pinned_tcb(
            &mut tcb, mmap_buf, $tref.offset, $tref.byte_size,
            scales, 0, $rows, $cols, b, $x_batch, $out_batch)?;
    } else {
        kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(/* …unchanged… */)?;
    }
} else if mma_on && $rows > $cols {
    kernels::gemm_q4_k_m_batched_v3w_mma_pinned_tcb(/* …Option A… */)?;
} else {
    kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(/* …unchanged… */)?;
}
```
where `mma_on` is read once near the top of `forward_tokens_batch_tcb`
(env idiom below). The dedicated ffn_down branch (`:5063`) is rows<cols → leave
it on predec/v3w (MMA loses there; do NOT gate it on).

---

## 4. Files to touch (Option B)

### 4a. `crates/dismantle-core/shaders/quant.metal` — already staged in stash
The 2 kernels are in `stash@{0}`. **Re-derive by hand** (do not pop the stash
blindly — it was *applied* per the handoff, so confirm `git diff` shows the +205
lines present in the working tree before building; if a clean tree is wanted,
copy the two kernel bodies from `git stash show -p stash@{0}`). The
`#include <metal_simdgroup_matrix>` is **already present** at `quant.metal:161`
(used by the existing decode simdmat kernel), so no new include is needed. The
`ArgbufBatchedRowsCols` struct (`:1495`) is reused unchanged.

### 4b. `crates/dismantle-core/src/kernels/mod.rs` — 2 new wrappers
Clone `gemm_q4_k_m_batched_v3w_pinned_tcb` (`:595`) → `..._mma_pinned_tcb`, and
`gemm_q4_k_m_batched_v3w_predec_pinned_tcb` (`:671`) → `..._mma_predec_pinned_tcb`.
**Three things change vs the v3w wrappers:**
1. `const KERNEL` = the new name.
2. **Grid + threadgroup:** the MMA kernel is **one simdgroup / 8 rows / TG**:
   ```rust
   const ROWS_PER_TG: u32 = 8;
   const TG_THREADS: u32 = 32;             // v3w uses 256
   let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
   // dispatch_threads grid = (n_tg * TG_THREADS, 1, 1), tg = (TG_THREADS,1,1)
   ```
3. **Threadgroup memory:** fixed **576 f32**, not `batch*256`:
   ```rust
   let shmem_bytes = (576 * std::mem::size_of::<f32>()) as u64;  // 2304 B
   ```
   Keep all the existing arg-validation (cols%256==0, batch∈1..=8, byte-size,
   x/y buffer-length checks) verbatim — the I/O contract is identical to v3w.
   The predec wrapper additionally keeps the `scales_buf` length check
   (`rows*blocks_per_row*16` f32) and binds scales at buffer slot 1 (shifting
   x/y/args to 2/3/4), matching kernel 2's signature.

### 4c. `crates/dismantle-core/src/model/qwen_dense.rs` — flag + prewarm + swap
1. **Env flag**, read once at the top of `forward_tokens_batch_tcb` (idiom from
   `tier1_scaffold_handoffs` §shared):
   ```rust
   let mma_on = { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
       *E.get_or_init(|| std::env::var_os("DISMANTLE_QWEN_Q4K_MMA")
           .map(|v| v != "0").unwrap_or(false)) };
   ```
2. **Prewarm:** add `"gemm_q4_k_m_batched_v3w_mma"` and
   `"gemm_q4_k_m_batched_v3w_mma_predec"` to `QWEN_TCB_KERNELS` (`:1126`).
3. **Swap:** edit the `batched_proj!` macro Q4_K arm (`:4869`) per §3. The macro
   already has `$rows`/`$cols` in scope, so the `rows > cols` gate is local.
   `mma_on` must be visible inside the macro — it is, since the macro expands
   inside `forward_tokens_batch_tcb` where `mma_on` is bound.

> **Do NOT** also wire the MMA into `forward_tokens_verify` (`:5119`, the
> batched-verify-with-logits workhorse) in this haul unless the verify GEMM also
> routes through `batched_proj!`. If it has its own GEMM dispatch, that is a
> second (smaller, N≤8) site — note it as a followup, don't expand scope.

### 4d. `crates/dismantle-core/tests/q4k_batched_mma_parity.rs` — new
See §5.

---

## 5. The parity test (the gate that must pass before any commit)

**Clone `tests/q4k_batched_predec_parity.rs`** — it is the ideal template: it
already builds a random Q4_K weight + f32 activations, runs B=1..=8, and asserts
the predec kernel matches v3w. The MMA test does the same but with a **looser,
per-contract tolerance**:

- **Reference:** `gemm_q4_k_m_batched_v3w_pinned_tcb` (the scalar staged path).
- **Under test:** `gemm_q4_k_m_batched_v3w_mma_pinned_tcb` AND
  `gemm_q4_k_m_batched_v3w_mma_predec_pinned_tcb` (the latter vs
  `gemm_q4_k_m_batched_v3w_predec_pinned_tcb`).
- **Tolerance:** **`atol = 1e-3` fp16, NOT bit-identical.** Rationale: the MMA
  reorders the K reduction (depth-8 hardware tiles + a different accumulation
  tree) vs the scalar FMA chain, so FMA-recontraction makes them numerically
  close but **not** `to_bits()`-equal. This matches the project's verification
  rule (atol 1e-3 fp16 is the parity regime; the handoff reported the standalone
  MMA at **8.0e-5 → 1.26e-4 (agent-measured)**, ≈8× under 1e-3). Use
  `assert!((a-b).abs() < 1e-3)` element-wise, like `q4k_predec_f16s_parity.rs`
  does for its non-bit-identical f16-scale case.
- **Shapes:** test the **winning shape** explicitly — `rows=11008, cols=2048`
  (ffn gate/up) at B∈{1,2,4,8} — plus a small `rows=512, cols=512` sanity tile.
  Do NOT only test square; the whole point is rows>cols.
- **Independent re-run / token-identity:** the project's CLAUDE.md evidence rule
  wants a clean re-run. For this lever the decisive *correctness* check beyond
  the unit parity is a **token-identity generate**: with
  `DISMANTLE_QWEN_BATCH_PREFILL=1 DISMANTLE_QWEN_Q4K_MMA=1` vs `…_MMA=0`, the
  first 3 greedy token IDs on a long prompt must match the locked baseline
  (`tests/golden/_phase1_token_baseline.hashes`). Because prefill only changes
  the KV written before decode, a 1e-3 GEMM perturbation **could** flip a token
  on a knife-edge logit — if tokens diverge, that is the real gate, not the unit
  atol. **Run this in `main` yourself; do not trust an agent's "parity passed"**
  (per `feedback_worktree_parity_verify`).

Build/run sequence (CLAUDE.md build-hygiene rule):
```
cargo build --release --workspace
cargo test --workspace --lib                         # 15 pre-existing must stay green
cargo test -p dismantle-core --test q4k_batched_mma_parity
```

---

## 6. Exact anchors (current main, branch codex/maximal-spec-colab)

| what | file:line |
|------|-----------|
| stash kernels | `stash@{0}` → `shaders/quant.metal` +205 (after `:1773`) |
| `metal_simdgroup_matrix` include (present) | `shaders/quant.metal:161` |
| `ArgbufBatchedRowsCols` struct | `shaders/quant.metal:1495` |
| v3w wrapper (clone for MMA) | `kernels/mod.rs:595` |
| v3w_predec wrapper (clone for MMA-predec) | `kernels/mod.rs:671` |
| `predecode_q4_k_scale_table` (16 f32/block) | `kernels/mod.rs:1042` |
| prewarm list `QWEN_TCB_KERNELS` | `qwen_dense.rs:1126` (v3w at `:1129`) |
| `BATCH_PREFILL` probe | `qwen_dense.rs:1461` |
| `ensure_q4k_predec_cache` (covers gate/up/down) | `qwen_dense.rs:2865`, inserts gate/up at `:2897–2898` |
| `forward_tokens_batch_tcb` | `qwen_dense.rs:4702` |
| `batched_proj!` macro (Q4_K arm to edit) | `qwen_dense.rs:4865`, predec-vs-v3w at `:4877` |
| ffn_gate batched_proj (rows>cols WIN) | `qwen_dense.rs:5037` |
| ffn_up batched_proj (rows>cols WIN) | `qwen_dense.rs:5043` |
| ffn_down branch (rows<cols, leave alone) | `qwen_dense.rs:5063–5078` |
| parity-test template to clone | `tests/q4k_batched_predec_parity.rs` |
| f16/non-bit-identical atol template | `tests/q4k_predec_f16s_parity.rs` |
| existing kill record | `reports/dead_levers.md` §"Q4_K batched MMA … on rows ≤ cols" |

---

## 7. Paired-bench protocol (GPU lane — do NOT run here; GPU-free wave)

The unit parity is GEMM-level; the *lever decision* is a **prefill-TTFT paired
bench** in the shipped (predec-ON) config. Use `tools/bench/paired_lever.sh`.

```sh
# Correctness gate (bit-tolerant parity already covered by the cargo test).
# Perf gate — paired prefill TTFT on a LONG prompt (so B>1 prefill dominates):
tools/bench/paired_lever.sh --label prefill_mma --no-parity \
  --env-a "DISMANTLE_QWEN_BATCH_PREFILL=1 DISMANTLE_QWEN_Q4K_MMA=0" \
  --env-b "DISMANTLE_QWEN_BATCH_PREFILL=1 DISMANTLE_QWEN_Q4K_MMA=1"
```

Protocol requirements:
- **Measure prefill ms / TTFT, not decode_tps.** The MMA only touches the
  batched prefill GEMM; decode is untouched (and is the closed Type-1 front).
  Reuse whatever long-prompt TTFT measurement the bench exposes; if it only
  reports decode tps, this lever needs a TTFT harness first (followup).
- **Long prompt.** The win scales with prefill token count (N per GEMM). A
  10-token prompt won't separate from noise. Use ≥256-token prompt so the
  per-layer ffn_gate/up GEMMs run at meaningful N.
- **Contamination cancels in the paired delta** (`feedback_bench_with_claude_open`):
  the A/B share the same Claude-app GPU load, so the *relative* TTFT delta is the
  signal even with Claude open. Still, take **3 runs, report the full spread**,
  not a single mean (`feedback_report_spread_and_label_estimates`).
- **Confirm the swap actually fired.** Because of the predec-default-ON trap, a
  silent failure mode is "MMA on but predec branch still ran scalar." Add a one-
  shot eprintln (or check via a counter) that `…_mma_predec_pinned_tcb` was the
  dispatched wrapper for ffn_gate/up. A "+0% flat" result almost certainly means
  the swap didn't engage, not that MMA is worthless.

---

## 8. Honesty ledger — what is measured vs estimated, and the contradiction

| claim | tag | source |
|-------|-----|--------|
| plain 8-wide MMA = +22–24% on 11008×2048 (rows>cols), in-tree N=8 GEMM µbench, ±1% | **(measured)** | handoff §Verdict + `dead_levers.md`; agent-reported, not re-run by me |
| plain MMA = −8.8% ffn_down (2048×11008), −10–16% attn (2048×2048) | **(measured)** | same; the Type-1 occupancy loss |
| MMA parity atol 8.0e-5 → 1.26e-4 fp16 vs v3w/GEMV | **(measured, agent-reported)** | handoff §Verdict — **re-run in main before trusting** |
| standalone silicon #8 µbench: +15% (N=8), +11→20.5% (N=512 w/ nwide), square M=K=2048 | **(measured)** | `silicon-builds/dismantle-q4k-mma/VERDICT.md` — vs the crate's OWN scalar kernel, not tuned v3w |
| shipped TTFT win on the real prefill path | **(estimate / UNMEASURED)** | the handoff says E2E TTFT was noise-dominated under contamination; **§7 is the gate** |
| predec covers ffn_gate/ffn_up → v3w-MMA dormant in shipped | **(verified, code-read)** | `qwen_dense.rs:2897–2898` + `:4877` this session |

**The contradiction to resolve at bench time (flag, don't hand-wave):** the
standalone VERDICT measured the plain MMA at **+15% on square 2048×2048 (N=8)**,
but the **in-tree** lane measured the *same* plain MMA at **−10–16% on square
attn 2048×2048**. Both are "measured." The reconciliation is the **baseline**:
standalone MMA was vs the crate's *own* shmem-scalar kernel; in-tree MMA was vs
the *tuned* `gemm_q4_k_m_batched_v3w` (256-thread / 8-simdgroup, which fills the
GPU far better than a 32-thread single-simdgroup TG on square shapes). So the
in-tree square loss is **real and is the number to trust** for shipping — the
standalone square win is an artifact of a weak baseline. This is why the lever
is shape-gated to rows>cols (where even the tuned v3w underfills less than MMA).

---

## 9. The N-wide W-reuse fork (a known better kernel the stash does NOT contain)

The silicon #8 VERDICT's headline catch (audit-to-improve #3) is that the plain
8-wide MMA win **shrinks as N grows** (dequant W re-decoded once per 8-col
N-tile → at N=512 each W row is dequant'd 64×). The fix —
`gemm_q4k_mma_nwide` (`silicon-builds/dismantle-q4k-mma/src/bin/bench.rs:132–177`):
dequant the 8×32 W tile **once per K-step**, reuse across **4 N-tiles** (BN2=32,
4 accumulator matrices) — recovered **N=512 from +11% → +20.5% (measured,
standalone)**.

**Why it matters for prefill but not verify:** the **verify** path is N≤8 (one
8-wide tile, no redundancy) — the stashed plain MMA is *already optimal* there.
The **prefill** path at a long prompt runs N=64–512 per GEMM, where the stashed
plain MMA leaves the N-wide win on the table.

**Plan stance:** ship the stashed plain MMA first (it is the measured rows>cols
winner and the parity surface is small). Treat `gemm_q4k_mma_nwide` as a
**Phase-2 followup**, NOT this haul:
- It is **not in the stash** — porting it is net-new kernel work (a third kernel
  + a 3rd wrapper, 4 simdgroup accumulators, grid `gx = N/32`), out of scope for
  "land the stashed kernels."
- Its `+20.5%` is `(measured)` only vs the standalone weak baseline; vs the
  tuned in-tree v3w on the real ffn gate/up shape it is **(estimate)** until
  benched — same caveat as §8.
- The tier-1 handoff §1.1 explicitly recommended porting `nwide` for N≥32; that
  recommendation stands as the next lever once the stashed plain MMA is landed
  and its TTFT delta is measured.

---

## 10. Halt / scope notes (CLAUDE.md compliance for the GPU lane)

- **One commit**, subject e.g. `prefill: G-MMA Q4_K simdgroup-matrix (rows>cols,
  predec twin) behind DISMANTLE_QWEN_Q4K_MMA`. Files: the 2 wrappers, the macro
  edit + flag + prewarm, the new parity test, the (already-staged) shader. No
  sweep.
- **Parity-first.** Run `q4k_batched_mma_parity` (atol 1e-3) + token-identity
  generate **in main** before committing; if tokens diverge, **halt** and record
  whether it is a real bug or a knife-edge logit flip (write the blocked doc).
- **Kill is already recorded** for rows≤cols (Type-1 occupancy) — do **not**
  wire MMA into attn/ffn_down. The Type-2 reframe (multi-simdgroup-per-TG tile to
  fill small-rows shapes) stays dead until its named occupancy oracle (TG count
  vs M3 Pro core count at those dims) is built — that is a separate lever.
- **Do not** "peel-onion" the predec-coverage check: the cache demonstrably
  covers gate/up (`:2897–2898`), so Option B will engage. If at bench time the
  swap shows +0% flat, debug the dispatch (did `…_mma_predec` actually run?)
  before concluding the lever is dead.
