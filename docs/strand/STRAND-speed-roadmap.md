# STRAND speed roadmap — from 4.5 Gw/s to the 156 Gw/s flip

_Branch `speed-bitslice`, started 2026-06-10 while the quant marathon ran. The goal:
make STRAND inference-competitive with llama.cpp-class kernels WITHOUT giving up one
bit of the determinism contract. Every stage gates on byte-identity to
`decode_tensor_fixed` before its first perf number is believed (house rule §5.2)._

## 0. The honest starting line (measured)

- Single-thread fast decode: **~756 Mw/s** (≈5.4 cycles/weight on the M3 P-core).
- Rayon block-parallel: **~4.5 Gw/s** (6×, 12 cores, bit-identical).
- End-to-end implication TODAY: 7B ÷ 4.5 Gw/s ≈ **0.6 tok/s decode ceiling** —
  llama.cpp CPU does ~20-30 tok/s on the same silicon. We are NOT speed-competitive.
- The format's bandwidth flip: **156 Gw/s** @ 0.4375 B/w — the throughput where decode
  stops being compute-bound. Everything between 4.5 and 156 is kernel engineering.
- Dead GPU levers (do not re-run): fused per-block GEMV 8-25% peak (ALU-bound),
  occupancy/multirow (plateaus 18-29%), vecread (inert), predec (worse), tensor cores
  (float = off-moat), tropical-scan for decode (moot — no serial chain), per-weight
  windowed decode (8-10%; window re-derivation 3× ops). See will.md §4.
- Dead CPU lever: NEON `decode_q12_simd` — vectorised state-advance, but every LUT
  gather pays a vector→scalar crossing; measured slower than rayon.

## 1. Why the chain math says there is headroom

Per weight, the hot loop carries two interlocked serial chains: the bit-reader
accumulator (`acc >>= k`) and the trellis register (`state = ((state<<k)|sym)&mask`),
plus off-chain loads (LUT, sub-scale) and one i64 mul. Chain latency ≈ 4-5 cycles on
an 8-wide core ⇒ ~80% of issue slots idle. Filling them needs MORE INDEPENDENT
CHAINS per core, not vectors: blocks are independent by construction (per-block
`init_state` + bit offset = prefix sums) — the same property the rayon path exploits
across cores, exploited WITHIN a core.

## 2. G0 — interleaved multi-stream scalar (BUILT, 2026-06-10)

`crates/strand-decode-kernel/src/interleave.rs` + `bin/gate-interleave.rs`.
`decode_q12_interleave{,_par}::<S>`: blocks in chunks of S, one scalar loop advancing
S chains per iteration. Bit-identity: **7/7 variants byte-identical** at both deploy
points (k=3/L=7, k=2/L=12), 23/23 lib tests green, plus fold/vector fallback paths.

Two engineering lessons already paid for (recorded so they are never re-bought):
1. **The quadratic drain**: building per-chunk output slices with `Vec::drain(..S)`
   from the front memmoved ~2MB per chunk (~280GB per decode) — first bench read
   13-97 Mw/s and the linear-in-S pattern was the tell. O(n) chunked `split_at_mut`
   fixed it.
2. **Array-of-structs kills SROA**: lane state behind `&mut [Lane]` cannot be
   register-allocated; every step paid ~12 field load/stores ⇒ 0.83× of baseline.
   G0b moved hot state (`acc/have/widx/state/done`) into local primitive arrays
   owned by the function — with const S the loops unroll, indices const-propagate,
   and the chains live in registers.

**G0b advisory result (CONTENDED box — marathon requant co-running; definitive
re-bench queued for an idle window):** S=4 = **1.31-1.38× single-core** over the
co-measured baseline; S=8/16 regress (31 GPRs: 4 lanes fit, more spill). Par path ≈
parity at 12 cores (rayon already saturates; the interleave win is per-core
efficiency — which is exactly what G2's fused GEMV monetises, and what batch=1
inference on efficiency cores will feel).

**Gate to pass before G1:** idle-box bench ≥1.5× single-core at S=4 → adopt as the
inner loop of G2. If idle-box shows <1.2×, the chain model is wrong for this silicon
— stop, profile with instruments, re-derive.

## 3. G1 — NEON hybrid (state-advance vector, gathers scalar-interleaved)

The dead `decode_q12_simd` vectorised state math but serialised gathers through
vector→scalar moves. G1 inverts the split: keep G0b's register-resident scalar
chains, but pack the FOUR lanes' state advances into one NEON op-pair where the
compiler does not already (inspect the G0b asm first — LLVM may have done it). Only
profitable if the bottleneck after G0b is ALU issue, not load ports (3 loads/cycle on
M3 P-core; at S=4 we issue ~1.3 loads/cycle — headroom exists). Expected: modest
(1.1-1.3×). Skip if G2 makes it moot.

## 4. G2 — fused decode+GEMV with interleaved lanes (the product kernel)

Today `matvec` decodes to a Vec then multiplies — materialising 4 B/weight of Q12
traffic. The product kernel decodes S=4 row-blocks straight into f32 MACs against
the activation vector (batch=1) or activation tile (batch=B):
- batch=1: decode cost still per-token, but the 4 B/w write traffic disappears and
  the MAC hides under the chain stalls G0b leaves.
- batch=B (prompt phase): decode ONCE per B MACs — decode amortises to ~zero; this
  is where llama.cpp-parity arrives first. Target: prompt tokens/s within 1.5× of
  llama.cpp Q4 on the M3 at 3-bit, with bit-identical Q12 decode asserted.
Determinism note: the DECODE stays integer/bit-exact; float accumulation order in
the MAC is documented per-kernel (same caveat llama.cpp lives with) — the .strand
artifact and its Q12 stream remain bit-identical everywhere.

## 5. G3 — x86 AVX2 lane (the pod / commodity-server story)

`vpgatherdd` does 8 LUT gathers/instruction from L1 (the 16 KB L=12 table fits);
256-bit state advance for 8 lanes. The same interleave structure compiles to both
targets (G0b is pure scalar Rust — it already runs on the pod; G3 adds the
`target_feature(avx2)` specialisation). Bench on the 24-32 vCPU pod between chain
phases. This is also the cheap rehearsal for…

## 6. G4 — GPU bitslicing (the one untested GPU lever; post-marathon)

The research-verified route (bitsliced Viterbi-class decode, ~21 Gbps V100-era,
pure integer = moat-safe): many independent block-streams per threadgroup, the
16 KB L=12 LUT in threadgroup memory, lean 16-B block table (banked, 0.4609 B/w),
one stream per SIMD lane — amortising the ALU that killed every per-block kernel.
Target gate: ≥50% of the M3's 122 GB/s peak on ffn shapes (vs 8-29% for every dead
kernel). Pass → dismantle's Metal kernel spec (`STRAND-dismantle-wiring.md` grows a
kernel section). Fail at <35% after occupancy/ILP sweeps → GPU speed is closed for
good on this silicon generation and the CPU+prompt-GEMM story carries the product.

## 7. Bench protocol (the sweep harness, runs on "any length of uptime")

- `gate-interleave` (exists): identity-gated, best-of-3, both deploy points.
- Post-marathon idle sweep: S ∈ {2,4,6,8} × {3-bit, 2-bit} × {single, par} ×
  P/E-core pinning (`taskpolicy -c utility/background`) → CSV in research/, plotted
  into this doc. Add `--shape` flags for attn_o/ffn_up/ffn_down.
- Every number lands here with the machine state noted (idle vs co-running).

## 8. Decision tree

```
G0b idle bench ──≥1.5×──► G2 fused GEMV ──prompt ≥ llama.cpp/1.5──► ship CPU story
      │                        │
      <1.2× → profile          └─ batch=1 still ≫ llama.cpp behind → G4 gate
      │
G4 ≥50% peak ──► dismantle Metal kernel (the real "out of the gate" speed)
G4 <35%      ──► GPU closed; density+determinism+prompt-GEMM is the speed story
```

## G2 results (2026-06-11 — ADVISORY: co-running box; idle re-run queued)

_Built post-G0b-failure, per the §8 tree: no interleaved lanes — the win claimed here is
killing the materialized-Q12 traffic (4 B/w written + re-read) and batch amortization,
nothing else. `crates/strand-decode-kernel/src/fused.rs` + `bin/gate-fused.rs`._

**Machine state (be honest):** measured while rung-screen's 12-thread `quantize-model`
(800% CPU) and a 4-thread pv2 bake co-ran. strand-delta was waited out first (the gate
binary refuses to bench while it lives; `--no-wait` overrides). Treat every ratio below
as advisory; the §7 idle sweep re-runs `gate-fused` serially. G0b's lesson cuts both
ways: contention flattered the interleave, and here it likely *understates* the high-B
fused win (the baseline's 64 re-reads of 272 MB suffer less when memory is already the
bottleneck) while muddying B=1.

**Determinism gate (before any perf number, both deploy points): 3/3.**
`fused_matvec` y per-element **bit-equal** to `decode_q12_par` + reference MAC (same
accumulation order — per-row sequential, left-assoc `(q as f32) * (1/4096) * x[i]`,
documented in fused.rs); hidden-Q12 debug path **byte-identical** to `decode_q12_par`;
every fused_gemm B=4 column bit-equal to its fused_matvec (B=1 column equal by
construction — fused_matvec IS fused_gemm at B=1). Plus 4 lib tests over the full
encode-lever matrix (k∈{2,3,4}, fold/no-fold, tail-biting, affine-min, block-straddling
rows, vec-trellis fallback); crate suite 27/27 green.

**Numbers (ffn_down 18944×3584 = 67.9 Mw, synth, best-of-3, ms):**

| | 3-bit (k=3,L=7) | 2-bit reopen (k=2,L=12) |
|---|---|---|
| decode_q12_par (decode only) | 19.9 (3.42 Gw/s) | 18.2 (3.74 Gw/s) |
| baseline two-pass matvec (B=1) | 26.6 | 25.7 |
| **fused_matvec (B=1)** | 35.7 = **0.75×** | 27.3 = **0.94×** |
| baseline (B=4 / 16 / 64) | 70.5 / 252.0 / 882.5 | 51.2 / 208.6 / 866.7 |
| **fused_gemm (B=4 / 16 / 64)** | 45.0 / 56.6 / 137.7 = **1.57 / 4.45 / 6.41×** | 51.0 / 70.9 / 149.5 = **1.00 / 2.94 / 5.80×** |
| fused per-column @ B=64 | 2.15 ms = 31.6 GMAC/s | 2.34 ms = 29.1 GMAC/s |

**Verdicts vs the §4 targets:**
- **batch=1: FAILED its half of the thesis.** The "MAC hides under the chain stalls"
  bet is wrong on this silicon: fused B=1 is 0.75×/0.94× of the two-pass baseline.
  The materialized path wins because its second pass is an autovectorizable NEON
  matmul, while the fused scalar MAC rides the serial decode loop — the 272+272 MB
  of Q12 traffic it saves is cheaper than the SIMD it gives up. Token decode keeps
  `decode_q12_par` + matmul. (Same honest pattern as G0b: correct code, thesis dead.)
- **batch=B: the amortization is real and big.** 6.41×/5.80× over two-pass at B=64;
  per-prompt-token cost drops 26.6 → 2.15 ms (12.4×). The mechanism is exactly the
  bytes-not-re-read claim: baseline at B=64 re-reads the 272 MB Q12 tensor 64 times
  (~17 GB); fused reads each weight once into 64 contiguous MACs against an L1-resident
  activation tile.
- **At B≥16 the bottleneck is no longer decode — it is the scalar f32 MAC** (B=16→64:
  +81 ms for +48 columns ≈ 1.7 ms/column pure MAC; decode is ~13% of the B=64 wall).
  Next lever (G2b, cheap): NEON-vectorize the MAC inner loop at large B. That changes
  float accumulation order (must be re-documented per kernel — the llama.cpp caveat §4
  already accepts); the Q12 decode stays bit-exact and the .strand artifact untouched.

**What G2 means for tokens/s (arithmetic, not a measurement; 3-bit, this box, advisory):**
- 0.5B (~0.5 Gw/token): token decode ≈ 0.5e9/2.55e9 ≈ **5.1 tok/s** (two-pass path);
  prompt @B=64 ≈ 0.5e9/31.6e9 ≈ **63 tok/s**.
- 7B (~7 Gw/token): token decode ≈ **0.36 tok/s**; prompt @B=64 ≈ **4.5 tok/s**.
- llama.cpp Q4 on the same silicon does ~20–30 tok/s *decode* (§0) and prefill well
  above that — so the §8 "prompt ≥ llama.cpp/1.5" ship-gate is **not met** by the
  scalar-MAC kernel at 7B scale. The path that could close it: G2b SIMD MAC (≥4–8×
  on the now-MAC-bound prompt phase) + the G3/G4 swings. Token decode remains the
  format's honest weak leg on CPU; the G4 GPU bitslice gate is unchanged as the big
  lever.

**Queued:** idle-box `gate-fused` re-run (it self-waits on strand-delta); side-by-side
llama.cpp prefill measurement before any ship claim; G2b SIMD-MAC behind the same
identity gate.

## Harness + G2b (2026-06-11 — idle box, machine-stamped)

_Implements audit runtime.md §2.1 (kernel harness, harness scope) + §2.2 (G2b NEON
across-the-batch MAC). Files: `crates/strand-decode-kernel/src/block_walk.rs` (new),
`bin/gate-kernels.rs` (new), `fused.rs` (G2b), migrations in `gemv.rs`/`gemv_par.rs`/
`interleave.rs`, `bin/gate-fused.rs` extended._

### The harness (block_walk.rs) — shared parts, NOT a framework

ONE `WordReader`, ONE `BlockPlan`/`block_plans` (prefix sums), ONE `SideInfo::hoist`
(eff/off stack arrays), ONE `block_init_state` (tail-biting recovery), ONE
`MAX_SUB`/`exceeds_max_sub` fallback predicate — all plain `#[inline]` leaf
functions/POD structs, **no traits, no closures, no dispatch** (the G0 SROA 0.83× kill
and the S=8 GPR-spill cliff are law; hot loops stay hand-written and monomorphized).
The 4 in-crate copies (gemv, gemv_par, fused, interleave) migrated; interleave's
register-resident lockstep core still lifts the reader's fields into local primitive
arrays (the harness exposes `pub(crate)` fields for exactly that). `gate_proto`
(same file) gives every gate bin ONE synth-tensor builder + ONE machine-state stamp
(loadavg, co-running science jobs, pool width) — comparability and idle-vs-advisory
honesty by construction. **Migration gate: byte identity — crate suite 28/28 green
(27 untouched + the new G2b A/B test), zero number changes.**

### gate-kernels — THE identity registry

`bin/gate-kernels.rs`: every CPU decode path (fast, par, simd, interleave S∈{2,4},
interleave-par S=4, fused B∈{1,4} via the hidden-Q12 stream) asserted **byte-identical
to `decode_tensor_fixed`** on one canonical matrix — 7 configs (k∈{2,3,4} deploy, both
fold branches, the L=12 reopen geometry) × 4 encode-lever variants × 24 edge-sweeping
lengths + the vec-trellis `_with_lut` fallback dispatch = **5,380 cells, all green,
~7 s**. New kernels register with one `fn` + one row and inherit the whole matrix.
`--bench` (machine-stamped) is only reachable AFTER the identity matrix passes in the
same process — perf without identity is structurally impossible in this bin.
Idle stamp `loadavg {7.7…}`: par 4.48 Gw/s (canon 4.46 reproduced), simd 2.65,
interleave-s4 0.83–0.86, fast 0.76 — the ledger's relative picture, unchanged.

### G2b — NEON across-the-batch MAC: built, bit-equal, and the speed thesis is DEAD

Design as audited: broadcast `w=(q as f32)·(1/4096)` (`vdupq_n_f32`), MAC into
`B/4` `float32x4` per-column lane accumulators against the contiguous activation tile.
Per-column accumulation order UNCHANGED ⇒ the default non-fused `vmulq+vaddq` build is
**bit-equal to the scalar fused path** (asserted per element in the lib test
`g2b_neon_mac_bit_equal_scalar_mac` and in gate-fused before any perf line, B∈{4,16,64},
both deploy points). A `vfmaq` variant ships behind the **off-by-default `neon-fma`
cargo feature** — single rounding, faster and more accurate, but it CHANGES low-order
float bits of `fused_gemm` at B≥4 (per-kernel caveat documented in fused.rs; the
integer Q12 decode is bit-identical either way; strict float-equality tests compile
out under it, loudly). `fused_gemm_scalar_mac` stays public as the forever A/B.

**Numbers (ffn_down 18944×3584 = 67.9 Mw, synth, best-of-3, IDLE BOX — stamp:
`loadavg {3.65 5.85 6.13} | no co-running STRAND science jobs | rayon threads 12`,
M3 Pro 18 GB, 2026-06-11 ~13:00):**

| B | baseline two-pass | fused scalar-MAC | **fused NEON-MAC** | G2b gain | vs baseline | MAC rate |
|---|---|---|---|---|---|---|
| 3-bit (k=3,L=7): 4 | 39.4 ms | 25.2 ms | **23.0 ms** | 1.10× | 1.71× | 11.8 GMAC/s |
| 16 | 106.0 | 35.2 | **35.2** | 1.00× | 3.01× | 30.9 |
| 64 | 374.9 | 66.5 | **63.4** | 1.05× | 5.91× | **68.5** |
| 2-bit (k=2,L=12): 4 | 39.1 | 24.0 | **22.3** | 1.08× | 1.75× | 12.2 |
| 16 | 106.3 | 36.0 | **35.4** | 1.02× | 3.00× | 30.7 |
| 64 | 375.9 | 67.3 | **64.1** | 1.05× | 5.87× | **67.8** |

B=1 (token decode): fused 0.97–1.07× of two-pass — the G2 "keep two-pass at B=1"
verdict stands (within noise either way on the idle box; the advisory 0.75× loss was
contention-flattered against fused). `neon-fma` build: B=64 → 60.8 ms (**+5–8%** over
non-fused NEON) — measured, available behind the flag, not worth the bit caveat today.

**Honest verdict vs the audit's 3–4.5× claim: the explicit NEON MAC gains 1.00–1.10×.
The claim is DEAD at the gate.** Mechanism: the audit's arithmetic assumed the scalar
chunk MAC "gets NO autovectorization in the fused path" — wrong premise. The
const-width `[f32; B]` column chunks G2 shipped with are a straight per-lane map (no
cross-lane reduction), which LLVM already autovectorizes; and the fma probe (+5–8%,
where halved FP-op count would give ~2× if FP-issue-bound) says the kernel sits at the
**activation-tile load ceiling, not the FP ceiling**. The win the audit smelled was
real but already banked — and partly an artifact of the advisory baseline: the same
kernel re-measured idle dropped B=64 from 137.7 ms → ~64 ms (per-column 2.15 → ~1.0 ms,
31.6 → ~68 GMAC/s, i.e. **2.15× of the "expected 3–4.5×" came from measuring on an
idle box**, the G0b lesson again, in reverse).

**Prompt tokens/s arithmetic (this box, idle, 3-bit; arithmetic not measurement):**
- 0.5B (~0.5 Gw/token): prompt @B=64 ≈ 0.5e9/68.5e9·64-col ≈ **137 tok/s**; token
  decode (two-pass) unchanged ≈ 5.5 tok/s.
- 7B (~7 Gw/token): prompt @B=64 ≈ **~9.8 tok/s** (was 4.5 advisory); token decode
  ≈ 0.38 tok/s.
- **The §8 "prompt ≥ llama.cpp/1.5" ship gate is NOT MET** — llama.cpp Q4 prefill on
  this silicon class runs ≳100 tok/s at 7B; we are ~10. And G2b's parity result says
  the CPU fused-GEMM kernel is now at its load-bound ceiling: there is no remaining
  in-kernel MAC lever on this path. The honest remaining swings are **G4 GPU
  bitslicing** (the one untested GPU route) and **G3 AVX2** (different silicon, pod
  lane) — plus the queued side-by-side llama.cpp prefill measurement to make the gate
  number exact instead of class-estimated.

**Determinism/KAT note:** everything here is decode-path; emitted bits are untouched,
no KAT re-anchor needed. The default build is byte-identical (Q12) and bit-identical
(float y) to the pre-harness kernels — proven by the unchanged 27-test suite, the new
A/B test, and gate-kernels' 5,380-cell matrix. The only behavior-changing option,
`neon-fma`, is OFF by default and changes float low bits only, never the archive or
the Q12 stream.

## G4 FINAL — bitslice REVIVED + productionized (2026-06-11, idle box, machine-stamped)

_The one untested GPU lever fired. Files: `shaders/strand_bitslice.metal`,
`metal.rs` (`BitsliceGpu`/`BitslicePrepared`/`bake_bitslice_entries`), `bin/gate-bitslice.rs`.
Stamp: `loadavg {3.71 3.16 3.58} | no co-running STRAND science jobs | rayon threads 12`,
M3 Pro 18 GB, measured streaming peak **98.0 GB/s** (grid-stride f32 sum, 256 MB — the
empirical denominator; the 122 GB/s datasheet number is inadmissible per house rules)._

### Why this shape won where five kernels died

Every dead GPU kernel (§0) used a threadgroup per OUTPUT ROW: at real shapes (bpr =
cols/256 ∈ [14, 74]) most of 256 lanes idled, and busy lanes interleaved decode with
MAC + barriers + a tree reduce. G4 inverts the grid: **one thread per 256-weight
block-stream end-to-end** (grid = ALL blocks, 256 independent streams/TG — full
occupancy at ANY tensor shape), chain state in registers, the 2^L LUT staged once into
threadgroup memory, ONE barrier total, zero cross-lane traffic. Identity discipline:
**673 GPU config×variant cells byte-identical to `decode_tensor_fixed`** (k∈{2,3,4},
both fold branches, L=12 reopen, tail-biting × affine-min, 24 edge lengths, vec-trellis
fallback) before any perf print — perf is structurally unreachable until the matrix
passes in-process.

### The decode gate (the ≥50% revival threshold, committed before running): PASSED

ffn_down 18944×3584 = 67.9 Mw, best-of-10 on-GPU dispatch:

| config | GPU bitslice | % of measured peak | CPU rayon (12-core) | GPU/CPU |
|---|---|---|---|---|
| 3-bit deploy (k3 L7, 512 B TG LUT) | 5.36 ms = **12.66 Gw/s** (59.4 GB/s moved) | **60.6%** | 17.62 ms = 3.85 Gw/s | **3.29×** |
| 2-bit reopen (k2 L12, 16 KB TG LUT) | 4.27 ms = **15.89 Gw/s** (72.5 GB/s moved) | **74.0%** | 16.60 ms = 4.09 Gw/s | **3.88×** |

GPU decode is **revived** — within 1.35-1.65× of the streaming-bandwidth ceiling on the
4 B/w-write decode-only shape, vs 8-29% for every dead kernel. (Run-to-run band across
sessions: 11.9-15.9 Gw/s decode, 34.9-40.8 Gw/s fused B=1.)

### Fused B=1 (token decode) + the GEMM variants B∈{4,16,64} (prompt phase)

Fused y=W·x kills the 4 B/w Q12 write: **B=1 = 1.67/1.90 ms = 40.55/35.69 Gw/s
effective** (3-/2-bit) — ALU-bound now (29.8%/21.7% of peak moved; the 80 B/block table
is 43-53% of remaining traffic — the lean-table revival is the named next density lever
for GPU traffic).

GEMM (identity-gated per config×B via one-hot batch lanes — each lane recovers its
probed column's Q12 EXACTLY through the GEMM kernel, every row × lane asserted):

| B | 3-bit GPU | GMAC/s | ms/col | CPU fused-NEON | GPU/CPU | | 2-bit GPU | GMAC/s | ms/col | GPU/CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.36 ms | 114.8 | 0.591 | 23.34 ms | **9.87×** | | 2.07 ms | 131.4 | 0.517 | **11.50×** |
| 16 | 4.78 ms | **227.2** | **0.299** | 26.90 ms | 5.63× | | 4.25 ms | **255.5** | **0.266** | 6.25× |
| 64 | 32.88 ms | 132.2 | 0.514 | 45.79 ms | 1.39× | | 22.80 ms | 190.5 | 0.356 | 1.94× |

**Verdict: B=16 is the GPU prompt sweet spot** (227-255 GMAC/s, ~2.3-3.7× the CPU
fused-NEON ceiling of ~95-98 GMAC/s). **B=64 REGRESSES on GPU** — 64 f32 accumulators
per thread blow register pressure/occupancy (the shader predicted "trades occupancy for
amortization — measured, not argued"; measured: it loses). Prompt-phase dispatch rule:
tile B>16 prompts as ceil(B/16) B=16 passes, do NOT use the b64 kernel on M3-class.
Prompt arithmetic at B=16: 67.9 Mw tensor → 0.27-0.30 ms/col ⇒ 7B prompt ≈
**31-37 tok/s decode-primitive** on the M3 GPU (vs ~9.8 CPU @B=64) — the llama.cpp
prefill gap (~100 tok/s class) closes from ~10× to ~3×.

### Prepared integration (the load-time bake; per-token loop shape)

`BitsliceGpu::prepare` bakes + uploads ONCE (payload, 80 B/block `BitsliceEntry` table
with host-side tail-bite recovery, frozen LUT, resident out buffer); per-token dispatch
touches zero host derivation. `dispatch_prepared_all` batches all tensors into one
command buffer.

| measurement (ffn_down 67.9 Mw) | 3-bit | 2-bit |
|---|---|---|
| cold (bake+upload+dispatch per call) | 63.11 ms | 62.16 ms |
| **prepared dispatch** | **4.56 ms (14.88 Gw/s)** | **4.59 ms (14.80 Gw/s)** |
| rebuild tax eliminated | **13.8×** | 13.6× |
| one-time prepare | 26 ms | 26 ms |
| resident (incl. 4 B/w out buffer) | 318 MB = 4.69 B/w | 310 MB = 4.56 B/w |

(The 4 B/w out buffer dominates the resident bill — it is the decode-to-buffer shape's
cost; the fused/GEMM paths skip it and hold only payload+table+LUT ≈ 0.56-0.69 B/w.)

Token-decode loop, synthetic 24-tensor model (4 layers × 6 projections at 2048-dim
geometry, 134.2 Mw/token), identity asserted on every tensor after the batched dispatch:

| per token | 3-bit | 2-bit |
|---|---|---|
| GPU, 1 commit (dispatch_prepared_all) | **10.58 ms (12.69 Gw/s)** | **10.38 ms (12.93 Gw/s)** |
| GPU, 24 commits | 16.03 ms | 16.36 ms |
| commit overhead | ~0.23 ms/tensor | ~0.25 ms/tensor |
| CPU rayon | 36.10 ms | 36.11 ms |
| GPU/CPU | 3.41× | 3.48× |

**The integration rule dismantle inherits: batch ALL tensors of a token into one
command buffer** — per-tensor commits burn ~0.23-0.25 ms each (≈ 35% of the token wall
at 24 tensors; at 7B's 196 projections it would dominate).

### OUTL on GPU — decided by arithmetic: CPU-patch boundary

The artifact's outlier channel (1% of weights, `outlier_mac.rs` residual form
`y[row] += (val − w_bulk)·x[col]`, residuals precomputed at load): per token at 7B
scale, 68M residuals × 8-12 B ≈ 0.55-0.8 GB streamed + 136 MFLOP — at the measured
98 GB/s that is **5.6-8.3 ms ≈ 3-5% of the fused-GPU token wall (~172-196 ms at 7B)**,
and on unified memory the CPU sees y in place (zero readback cost). A GPU sparse add
would need CSR-per-row ordering to stay float-deterministic (atomics forbidden) to save
≤5% — not built, not worth it now. **Boundary: GPU bulk → CPU rayon sparse residual
add, original-x.** Revisit only if profiling shows the sync latency (not bandwidth)
binding.

### Tokens/s arithmetic (decode-primitive ceilings — NOT end-to-end inference)

Basis: measured M3 fused B=1 effective rates (40.6 / 35.7 Gw/s at 3-/2-bit); tok/s =
rate ÷ nominal Gw/token. 3090-class column = arithmetic only (no CUDA port exists):
the fused kernel is ALU-bound on M3, so scale by ~5-6× integer-ALU class ratio; the
936 GB/s bandwidth ceiling sits far higher and never binds.

| model | M3-class 3-bit | M3-class 2-bit | 3090-class 3-bit (~5-6×, arith.) | 3090-class 2-bit (arith.) |
|---|---|---|---|---|
| 0.5B (0.5 Gw/tok) | ~81 tok/s | ~71 | ~405-487 | ~357-428 |
| 7B (7 Gw/tok) | **~5.8** | ~5.1 | ~29-35 | ~25-31 |
| 14B (14 Gw/tok) | ~2.9 | ~2.5 | ~14-17 | ~13-15 |

Honest deductions from the ceiling: attention/KV/norms/sampling, the activation-RHT
question (per-row rotation costs ~12× the MAC in FLOPs — see the wiring doc's kernel
section for the three production routes), OUTL ~3-5%, and per-commit overhead unless
batched. llama.cpp Q4 token decode on the same silicon ≈ 20-30 tok/s ⇒ the CPU-era
~50× gap is now **~4-5×** at 7B. This kernel is dismantle's Metal seed
(`docs/STRAND-dismantle-wiring.md`, kernel section rewritten around it).

## COMPOSE — prepared × paired (2026-06-11; identity PROVEN, perf ADVISORY-contended)

_The two separately verified CPU winners — prepared-model flat side-info/start-states
(1.31-1.40× decode, 2.53-2.81× with tail-biting; `prepared.rs`) and the paired-step
2-symbol LUT (+12-19% single-thread at 3-bit; `paired_lut.rs`) — did not compose until
now. `prepared.rs::decode_q12_prepared_paired{,_par}` runs the paired replay (one
`pop(2k)`, one composed advance, ONE 8-byte load per two weights) off the prepared flat
arrays (no per-call hoist, no tail-bite prescan). Envelope: scalar trellis, 3-bit/L≤10
(8-64 KB pair table); correct (identity-gated) at every config through L=12._

**Identity (the part that is NOT advisory): `bin/gate-compose` — 1,344 compose×config×
variant cells byte-identical to `decode_tensor_fixed`** (7 canonical configs incl. both
fold branches and the L=12 reopen × tail-biting×affine-min × 24 edge lengths × single +
par), plus 3 new in-crate tests (full lever matrix, vec/custom-LUT fallbacks, and a
mismatched-LUT-table hard-fail — a wrong table must panic, never decode wrong bytes
deterministically). Crate suite 69/69 green. Perf in the bin is unreachable until the
matrix passes in-process.

**Perf (ADVISORY — both runs landed on a SATURATED box: stamped loadavg {111-160},
co-running debug test binaries + rustc from parallel agents; baselines read 7-10× below
canon. Per the G0b law these ratios are not verdict-grade):**

| config | lane | tail | paired alone | prepared alone | COMPOSE | reading |
|---|---|---|---|---|---|---|
| k3 L7 (envelope) | 1T | off | 1.10-1.42× | — | **1.50-2.07×** | compose > best solo |
| k3 L7 | 1T | ON | 1.07-1.29× | — | **2.85-3.85×** | prescan removal compounds with pair-halving |
| k3 L7 | par | ON | 0.39-0.98× | 0.93-2.65× | 1.23-2.94× | paired-alone still dies under rayon; compose survives |
| k3 L10 (edge) | 1T | ON | 0.92-0.97× | — | **3.78-5.00×** | biggest 1T composition win |
| k2 L12 (out of envelope) | 1T | ON | 0.96-1.06× | — | 4.72-8.14× | suspicious magnitude — contention-flattered baseline |
| k2 L12 | par | off | 0.78-1.34× | 1.30-2.78× | 1.21-2.98× | compose ≈ prepared alone (pair table L1-misses at 128 KB) |

**Provisional verdict (pattern stable across BOTH contended runs; magnitudes are not):
the composition is REAL and at minimum additive-to-multiplicative single-thread with
tail-biting ON — the only kernel in the matrix where both wins bind at once (prepared
kills the per-call prescan, paired halves the surviving chain). Under rayon, paired's
known L1-pressure death persists and the composition tracks prepared-alone — the
prepared win subsumes, no interference observed (no cell where compose < both solos).**
The decisive idle re-run is queued in the bin itself (`gate-compose --bench`,
machine-stamped); do not quote any ratio above without the idle stamp.

**Ops note (same wave): `quantize-model` grew the identity-skip integration the audit
deferred — `--skip-manifest <m.json> [--reuse-from <prior-recon>]` (encode_cache.rs
wired in; config_key pins every encode lever INCLUDING `STRAND_F32_METRIC` /
`STRAND_F32_SEARCH`). Verified: warm requant skips hash-identical tensors with
byte-identical output; f32-lane toggle / config change / single-ULP weight move all
break the skip; selective-PV shape skips exactly the frozen tensors. The old
`research/patches/identity-skip-quantize-model.patch` is marked SUPERSEDED.**
