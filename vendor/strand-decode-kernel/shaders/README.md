# `strand_trellis_gemv.metal` — the M3 bandwidth-vs-compute gate kernel

This is the **one real GPU experiment** for the STRAND → dismantle fusion's *speed* claim. Everything
upstream (`decode_lean`, `.strand` v2 writer) is scaffolding; this kernel + an M3 Instruments trace is
the ridge proof. It is **authored source + a measurement plan**, not yet measured — the M3 GPU is busy
with the live quantization sweep, so this README is the exact recipe to run when the GPU frees.

Companions: `docs/STRAND-metal-decode-gate.md` (ridge analysis), `docs/STRAND-metal-kernel-impl.md`
(the build sheet §B/§C this file implements), `docs/STRAND-density-roadmap.md` (strategy). The
template is dismantle's `crates/dismantle-core/shaders/quant.metal:85 gemm_q4_k_m_fused`.

---

## What's in the file

Two kernels, byte-identical decode, different occupancy strategies (impl spec §B.6):

| kernel | strategy | use when | extra TG mem |
|---|---|---|---|
| `strand_trellis_gemv`        | **B.6a fused** — one thread walks one 256-block, decode+MAC inline | wide tensors (`bpr = cols/256` large, ~43–74 for 7B `down_proj`/`up`) | none beyond LUT+reduce (~1.5 KB) |
| `strand_trellis_gemv_predec` | **B.6b predecode** — pass A state-walks blocks → Q12 row tile in shmem; pass B does full-width MAC across all 256 threads | narrow tensors (`bpr` small, e.g. `cols=2048` → 8 blocks → 248 idle lanes in the fused path) | + `cols·4 B` Q12 tile (≤32 KB ⇒ `cols ≤ ~7000`) |

Both move the **same weight bytes**, so they cannot land on different sides of the gate — only the
clock differs. §C measures both; ship whichever is faster per shape.

---

## Buffer bindings (dispatch-compatible with `gemm_q4_k_m_fused`)

Dispatch shape: **one threadgroup per output row**, **256 threads/threadgroup**, grid =
`ceil(rows)` threadgroups. Mirror the Q4_K template exactly so dismantle's backend seam swaps it in
with no host change.

| idx | name | type | contents |
|---|---|---|---|
| `buffer(0)` | `w_bits`  | `device const uchar*` | tensor's contiguous `k`-bit symbol stream, **LSB-first**, **4-byte aligned per tensor** (writer invariant — see below) |
| `buffer(1)` | `x_rht`   | `device const float*` | **`RHT(x)`** — the host applies the per-tensor RHT to the activation ONCE per GEMV (see RHT section); length = `cols` |
| `buffer(2)` | `y`       | `device float*`       | output, length = `rows` |
| `buffer(3)` | `rows`    | `constant uint&`      | `out_features` |
| `buffer(4)` | `cols`    | `constant uint&`      | `in_features`, **multiple of 256** |
| `buffer(5)` | `tbl`     | `device const BlockEntry*` | v2 `(row,block)` table, **row-major**, stride = `bpr = cols/256` entries/row |
| `buffer(6)` | `k_bits`  | `constant uint&`      | index bits/symbol — **3** at deploy |
| `buffer(7)` | `l_bits`  | `constant uint&`      | register width — **7** ⇒ 128-entry LUT |
| `buffer(8)` | `lut_q12` | `device const int*`   | frozen Q12 codebook, `2^L` entries, **state-indexed** (`= strand_quant::codebook::codebook_lut(l_bits)`) |
| `threadgroup(0)` | `sh_lut`  | `threadgroup int*`   | `2^L` ints (128 → 512 B), staged once/TG |
| `threadgroup(1)` | `sh_red`  | `threadgroup float*` | 256 floats (1 KB) reduction scratch |
| `threadgroup(2)` | `sh_wq12` | `threadgroup int*`   | **predec kernel only**: `cols` ints (decoded Q12 row tile) |

**On-disk vs. GPU struct — the v2 schema resolution.** The authoritative *on-disk*
per-block record is `strand_quant::format::BlockOffsetRecord`: a lean **16 B**
`{ bit_offset: u64 (tensor-relative), init_state: u32, scale_q: i32 }`, parsed by the
canonical `strand_quant::format::read_strand_v2_header` (the single owner of the wire
format). The `BlockEntry` below is the **GPU-side, in-memory** struct the *loader builds
at upload time* — it expands `eff[8]` from `scale_q` + the side-info sub-scales and bakes
the per-tensor-relative `bit_offset` into an absolute position in the row blob. It is NOT
the on-disk layout. The file stays dense (16 B/block) while the kernel still gets the
pre-expanded record it wants.

`BlockEntry` (`#[repr(C)]`, **52 B** in-memory — the loader PROBES the GPU `sizeof(BlockEntry)` against the host `size_of` at init; no hardcoded-size assert):

```
struct BlockEntry {
    uint  bit_offset;  // ABSOLUTE bit pos of block's first symbol in w_bits (buffer 0 is one flat per-tensor blob)
    uint  init_state;  // baked start state (low l_bits used); baked UNCONDITIONALLY (even under tail-biting)
    int   scale_q;     // per-block super-scale Q16 (debug / future un-folded path)
    int   eff[8];      // PRE-EXPANDED effective sub-scales (Q16) = eff_scale_q(scale_q, sub_code[s]); zero hot-path unpack
    ushort n;          // weights in block (last block of a row may be < 256)
    ushort d;          // vector dim (1 at deploy; reserved for B.7 vector trellis)
    uint  _pad;        // trailing pad → 52 B total (4-byte aligned)
};
```

### v2 writer invariants the kernel relies on (do these in the `.strand` v2 baker, not the kernel)
1. **`w_bits` 4-byte aligned per tensor** — so `load_u32_le` reads whole words (and a future
   `device const uint*` reinterpret is legal). The kernel assembles from bytes to stay correct if
   alignment ever slips, but the aligned-`uint*` path is a measured §C lever.
2. **`init_state` baked into every entry** — even for tail-bitten blocks. `BlockMeta.init_state` is
   always recorded on the CPU side (`encode.rs:533`/`650`/`815`), so the baker copies it
   unconditionally; this removes the kernel's sequential trailing-symbol pre-scan (a whole pass/block).
3. **`eff[8]` pre-expanded at bake** — `eff[s] = eff_scale_q(scale_q, sub_scale_code[s])`
   (`decode.rs:59`), so the kernel does no 6-bit sub-scale unpack. (GPU analog of the CPU fold; always
   worth it host-side because it amortizes over every token.)
4. **3-bit deploy point ⇒ `has_affine_min == false`** — the offset add is compiled out of the hot
   loop. If a 4-bit config is ever run here, the writer must add a pre-expanded `int off[8]`
   (`eff_min_q`, `decode.rs:80`) to `BlockEntry` and the kernel must add `e->off[j>>5]` after
   reconstruct. Flagged for hardening.

---

## RHT on the activation (the QTIP/QuIP# trick — host does it, not the kernel)

The encoder quantizes each weight **row in RHT space** (`rht_forward_rows_inplace`, per-input-channel
Rademacher sign-flip + normalized FWHT over 256-wide segments restarted at each row, seed =
FNV-1a(tensor name) | 1, `rht.rs` + `quantize-model.rs:439`). Because the Hadamard is orthogonal:

```
y[o] = <W_weightspace[o], x> = <W_rhtspace[o], RHT(x)>
```

So the host computes `x_rht = rht_forward(x, RhtConfig::from_seed(seed))` **once per GEMV** and hands
it in as buffer 1; the kernel dots decoded RHT-space weights straight against it — **no per-row
inverse-RHT**. The per-tensor `seed` travels in the `.strand` v2 header (`rht_seed`, written at
`quantize-model.rs:1028`); the host reads it and builds the matching `RhtConfig` (block = 256).

**Determinism caveat (flagged for hardening):** this per-token FWHT is **float** and is *not* covered
by the integer-decode guarantee — it's ordinary float GEMV preprocessing (same status as the
`q·(1/4096)·x` MAC). The **weights stay bit-identical** cross-device; the activation rotation is float.
Pin the host RHT to the encoder's segmentation exactly — **256-wide block, row-restart, same seed** —
or `y` silently corrupts. `in_features % 256 == 0` for every Qwen2.5-7B tensor (assert it; flat-path
fallback only when it isn't). A mismatched block size is the single most likely integration bug.

---

## Inner-loop op estimate (the gate x-axis)

Per weight, the fused inner loop (`strand_trellis_gemv`, count from the emitted MSL):

| step | code | ~int/fp ops |
|---|---|---|
| refill test + (amortized) word load | `if (have<k) { load_u32_le; OR; have+=32 }` | ~1 amortized (1 LE load every `⌈32/k⌉≈11` weights at k=3; the byte-assemble `load_u32_le` is ~7 ops but only ~1/11 of the time ⇒ ~0.6/weight; an aligned `uint*` reinterpret drops this further — a §C lever) |
| symbol pop | `sym = acc & imask; acc>>=k; have-=k` | ~3 |
| state advance | `state = ((state<<k)|sym) & mask` | ~3 |
| LUT load | `q = sh_lut[state]` | 1 (shmem) |
| eff-scale load | `es = e->eff[j>>5]` | 1 |
| reconstruct | `(long)es*(long)q >> 16`, native 32×32→64 | ~2–3 (`mul.lo`+`mul.hi`+shift) |
| Q12→real + MAC | `partial += w * (1/4096) * x_rht[col0+j]` | ~2 fp |

**≈ 9–11 ops/weight.** Bytes/weight at 3-bit ≈ `k/8 + side-info ≈ 0.375 + ~0.05 ≈ 0.4–0.45 B/weight`.
⇒ **arithmetic intensity `I ≈ 9/0.45 ≈ 20` to `11/0.4 ≈ 28` ops/byte** — at/under the modeled M3 Pro
ridge (~30 ops/byte = ~7 TFLOP/s ÷ ~150 GB/s, int ≤ FP32 on the unified ALUs). **Borderline by
design** — that's exactly why it must be *measured*, not argued. (paradigmshift.md:280 budgets QTIP
decode at `3INST` ≈ 3 ALU ops/weight for the *decode core*; our extra ops are the bitstream pop +
state walk + the MAC the GGUF path also pays.)

Key authoring decisions baked into the source:
- **64-bit `acc`** (not the spec's single `uint`): a refill fires when `have < k`, so `nxt << have`
  would reach bit ~36 and overflow a 32-bit word. A `ulong` is unconditionally correct for any
  `k ≤ MAX_K(4)` at no measurable cost (refill is ~1/11 iters). Resolves the spec's "acc width" risk.
- **Native 32×32→64 reconstruct, NOT i32** — `scale_q(Q16)·quantile_q(Q12) = Q28`; a magnitude-~4
  weight hits `2^32` and overflows i32. The 64-bit product is required *and* cheap on Apple.
- **Runtime `lut_q12`/`l_bits` buffers** (not a compile-time-folded LUT) so LEVER B3 (`--dist`, custom
  codebook) ships without a recompile.

---

## EXACT M3 measurement steps (run when the GPU frees — `cargo`-free until then)

> Do **not** run any of this from a wave agent while the sweep is live (it contends the GPU and the
> 12-core CPU). This is the post-sweep recipe.

### 0. Correctness gate FIRST (a fast wrong kernel is worse than no kernel)
- Build a tiny `metal`-crate (or ObjC/Swift shim beside dismantle's kernel-bench) host that bakes
  **one** tensor to v2, uploads buffers 0/5/8, dispatches `strand_trellis_gemv` for a **batch-1** GEMV.
- Add a debug variant that dumps decoded `w` (the Q12 int, before the MAC) for one block; assert it
  **equals `decode_tensor_fixed` / `decode_lean` bit-for-bit** for that block. Only trust timing after
  this passes. Also assert the kernel `y` matches the CPU `matvec` (`strand-decode-kernel/src/lib.rs`)
  to within float-reduction tolerance.

### 1. Peak microbenches on THIS M3 (datasheet numbers are not admissible)
- **Streaming-load kernel** (pure `memcpy`/sum-reduce over a big buffer) → measured **STREAM
  bandwidth** `BW_peak` (~100–150 GB/s on M3 Pro; use the *measured* number).
- **Fused-multiply loop kernel** (FMA-bound, no memory) → measured **ALU peak** `OPS_peak` (int and
  FP32; int ≤ FP32 on unified ALUs).
- **Recompute the ridge** `I* = OPS_peak / BW_peak` from these — **do not ship the modeled 30.**

### 2. Time the kernel with GPU counters, not wall-clock
- Use `MTLCounterSampleBuffer` with `.timestamp` sampled at encoder boundaries (excludes
  dispatch/CPU). Warm up, then median of N runs.
- Shapes — the real Qwen2.5-7B GEMV rows: at least one **wide** (`down_proj`, `cols≈18944`; `up`/`gate`
  `cols≈18944`) and one **narrow** (`q`/`k`/`v`/`o`, `cols≈3584`/`4864`-class, and synthetic
  `cols=2048` to stress B.6). Run both `strand_trellis_gemv` and `strand_trellis_gemv_predec` per shape.
- Batch sizes `{1, 4, 16}`: **batch 1 is the gate** (token-decode regime); 4/16 just confirm the
  expected, fine cross-over to compute-bound.
- Head-to-head: run `gemm_q4_k_m_fused` on the **same shapes, same harness, same warm-up, same
  timestamp method** for an apples-to-apples tps comparison.

### 3. Instrument (the numbers that decide it)
Per run capture:
1. **Achieved BW** = `bytes_read / time`; `bytes_read ≈ rows·cols·(k/8) + table + cols·4 (x_rht) +
   rows·4 (y)` (weights dominate). Report as **% of `BW_peak`**.
2. **Achieved OPS** = `(ops/weight from §"inner-loop op estimate") · rows·cols / time`. Report as **%
   of `OPS_peak`**.
3. **Arithmetic intensity** `I = work_ops / bytes_read` (the roofline x-axis).
4. One **Xcode Instruments "Metal System Trace"** frame capture on the representative wide tensor →
   occupancy / threadgroup residency, and whether we're barrier-bound or memory-latency-bound (tells
   us which B.6 variant wins and why).

### 4. Verdict (commit these BEFORE running — no goalpost-moving)
- **PASS (build the dismantle shim):** at batch 1, achieved BW **≥ 60% of `BW_peak`** AND measured `I <
  I*` (bandwidth side) AND tps beats `gemm_q4_k_m_fused` on the same shapes by **≥ 25%** (the
  `4.5/3.34 − 1` byte-ratio Q4_K→STRAND-3bit). Bandwidth-bound + byte win realized ⇒ go.
- **MARGINAL (one optimization pass, then ship):** bandwidth-bound (`I < I*`, ≥60% BW) but tps win
  **< 25%**. Apply the lean levers harder — aligned `uint*` reads (drop `load_u32_le` byte-assemble),
  pop 2 symbols/iter, the predec variant, fold the LUT into constant memory — and re-measure.
- **FAIL (the Q3_K trap — do NOT build the shim):** at batch 1, `I ≥ I*` **OR** achieved BW **< 40% of
  `BW_peak`** with compute at its ceiling ⇒ compute-bound. The byte savings don't convert; the
  serial state-walk / symbol-pop (or B.6 barrier overhead) is the wall. The density moat then stands on
  **determinism + on-device-fits alone**, not speed — the honest downside the roadmap prices in
  (paradigmshift.md:292 calls out exactly this Apple risk).

### 5. Sanity controls so a PASS is real
- Correctness gate (step 0) green before any timing is trusted.
- Peaks from step 1's microbenches, not the datasheet, feed `I*`.
- Identical host harness for STRAND vs Q4_K so the ≥25% claim is apples-to-apples.

---

## ⚠️ CORRECTION — the activation-RHT recipe above is BROKEN for multi-row tensors

The "RHT on the activation" section above documents a **single-rotation** recipe: compute
`x_rht = rht_forward(x, seed)` once per GEMV and dot every decoded row against it. **That
recipe is wrong for every row after the first.** The encoder's Rademacher signs are drawn
from the GLOBAL flat index (`rht.rs::sign_at(seed, row*in + col)`), so each row `r` is
rotated by a *different* signed Hadamard `R_r`, and `<R_r(w_r), R_0(x)> != <w_r, x>` for
`r > 0`. The correct per-row identity is

```text
y[r] = <q_r, FWHT(s_r ⊙ x)>      s_r = signs at flat indices r·in .. r·in + in
```

— implemented and pinned by **`outlier_mac::matvec_rht`** (which broadcasts `x` to every
row through the public `rht_forward_rows_inplace`, so the rotation uses exactly the
encoder's sign stream). The divergence of the single-rotation recipe is itself pinned by
the `single_rotation_recipe_diverges_for_multirow` test in `outlier_mac.rs` — do not
re-import the recipe above. A fused kernel wanting the one-rotation trick needs an encoder
that draws signs per *column*: an encode-bits change, gated and off-by-default if ever
attempted (will.md determinism law). The buffer-1 `x_rht` description above (and the
host shim of `gpu_matvec_named`) is therefore only exact for `rht_seed == 0` tensors or
single-row probes; the metal.rs identity tests use exactly those.

---

# `strand_bitslice.metal` — G4: the bitslice decode gate (MEASURED 2026-06-11 — REVIVAL)

The one untested GPU lever (will.md §4, roadmap §6) — and the first GPU kernel in project
history to clear 50% of measured peak. Structural inversion vs every dead kernel: grid =
ALL blocks (one thread owns one 256-weight block-stream end-to-end), 256 independent
streams per threadgroup, chain state in registers, the 2^L LUT staged once into
threadgroup memory, ONE barrier total, decode-only Q12 output (no MAC/reduction confound).
Uncoalesced-but-cache-resident payload reads were accepted and measured per the roadmap.

## Protocol (the refuse-perf-without-identity discipline, `bin/gate-bitslice.rs`)

1. **Waits for `pgrep -f strand-qat`** to clear before any GPU dispatch (`--no-wait` overrides).
2. **Probe**: GPU `sizeof(BitsliceEntry)` asserted against host `repr(C)` (80 B) at init.
3. **Identity matrix BEFORE perf, in-process**: 673 cells — the gate-kernels canonical
   matrix (k ∈ {2,3,4}, both fold branches, k2 L12 reopen, tail-biting × affine-min,
   24 edge lengths) + the vec-trellis `_with_lut` fallback — every cell byte-identical to
   `decode_tensor_fixed`. Affine-min is handled by pre-baked `off[8]` (zeros when off), so
   ONE kernel covers the whole lever matrix. Tail-bite recovery is host-side at bake.
4. **Bench**: machine-stamped (loadavg + co-running science jobs), best-of-10 dispatch on
   ffn_down 18944×3584 (67.9 Mw), judged vs the **empirically measured** streaming peak
   (`StrandGpu::bench_peak_bw`, 70.6–90.9 GB/s across runs), never the datasheet. The
   bench shape's own identity is re-asserted in-process before its perf line prints.

## Measured (M3 Pro 18 GB, 2026-06-11, 4 runs; co-running rustc from sibling builds —
stamped on every run; peak and kernel measured under the SAME contention)

| config | GPU decode | GB/s moved | % of co-measured peak | vs canon CPU 4.5 Gw/s |
|---|---|---|---|---|
| 3-bit (k3 L7) | 9.5–13.3 Gw/s | 44–62 | **51–70%** | 2.1–3.0× |
| 2-bit reopen (k2 L12, 16 KB TG LUT) | 12.1–14.0 Gw/s | 55–64 | **61–86%** | 2.7–3.1× |

**VERDICT: ≥50% of peak on every run/config ⇒ REVIVAL** (threshold committed in roadmap
§6 before running). Decode-only is write-bound (the 4 B/w Q12 store is ~85% of its
traffic) — i.e. the decode itself runs at the memory system, not the ALU: the per-row
kernels' ALU wall was their SHAPE, not the trellis math.

## The stretch (unlocked): fused batch-1 `y = W·x`

`strand_bitslice_gemv_partials` (same gated walk, one float partial per block) +
`strand_bitslice_reduce_rows` (per-row sum in FIXED ascending block order — no atomics).
Float accumulation order documented per-kernel: `y[r] = Σ_b (Σ_j w·(1/4096)·x)`, both
sums sequential (the fused-caveat pattern; Metal fast-math may contract mul+add to fma —
cross-device float-y equality is NOT claimed, the Q12 stream is the moat).

| config | fused B=1 | effective | 7B token-decode arithmetic |
|---|---|---|---|
| 3-bit | 1.89–2.03 ms | **33.4–35.9 Gw/s** | **~4.8–5.1 tok/s** (CPU two-pass: 0.38) |
| 2-bit L12 | 1.97–2.49 ms | 27.3–34.4 Gw/s | ~3.9–4.9 tok/s |

The fused path moves only 16–26 GB/s (payload + 80 B/block table) — it is NOT yet
bandwidth-saturated, and the **table is 43–53% of its traffic**: the banked lean 16 B
table is the named next lever, then a multi-token (batch-B) variant. Honest framing: this
is the first credible GPU speed path (≈13× CPU token decode at 7B by arithmetic), still
short of llama.cpp-class ~20–30 tok/s decode — parity is an open engineering question,
no longer a closed one.

Run: `cargo run -p strand-decode-kernel --release --bin gate-bitslice [-- --no-wait]`
