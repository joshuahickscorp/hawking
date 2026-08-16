# HQ80BR1 — Implementation-Ready Specification
## Binary group base + fixed-count sparse residual for the Q80 routed expert table

---

## 0. Verdict first, because it changes what this spec is for

**The velocity thesis is not supported by the mapped evidence. Do not build this expecting speed.**

The arithmetic, using only numbers sealed in this repo:

- Measured DRAM ceiling on this box: **~700 GB/s** (two GPU-timestamped MTLBlit roofline receipts, `workspace/campaign/records/runs/deepseek-v4/metal-device-copy-roofline-receipt-v1.json`, 700.0 / 704.9 / 707.1 GB/s).
- Q80 total active weight traffic: **1.8956 GB/token**.
- At the quoted 2.9249 tok/s (341.9 ms/token) that is **5.55 GB/s = 0.79% of ceiling**. There is **~126x of bandwidth headroom** before bandwidth binds.
- Q30's sealed dispatch profile (`receipts/q30-dispatch-gap/Q30_S_BUCKET_MECHANISM_TABLE.md:11-19`) measured **51% of a token as GPU-idle inside the timestamp envelope**, and its GPU-busy window itself ran at 1.27 GB/s.
- The repo already sealed this conclusion for its own kernels: `crates/hawking-core/src/model/qwen30_complete_runtime.rs:859-863` records the R=8 variant hitting 79 GB/s in a microbench (11% of ceiling, the best kernel bandwidth ever measured here) and **losing wall clock** — "occupancy, not bandwidth."

A 3.09x cut on expert weights is a **1.40x cut on token bytes** (top-10-of-512 sparsity means the 97.03% of static mass that is expert weights is only **42.3% of per-token traffic**). At the roofline that is **0.775 ms off a 341.9 ms token = 0.23%**.

So this spec is for: **a ≤1.5 BPW complete artifact that executes at parity speed or slightly better**, with one conditional wall-clock win (the host staging memcpy, 802 → 259 MB/token) that is measurable *today* before any of this is built. Section 7 is the experiment that decides whether even that is real, and it costs about five lines of Rust.

**Design chosen:** candidate 3's *fixed-C-per-group* residual (kills every scan, sort, CSR pointer and ragged trip count), candidate 2's *32-bit sign-word broadcast* addressing and *fused gate+up* kernel (kills the 32-way threadgroup bank conflict that candidate 3's lane-window layout would have caused, and halves activation traffic), candidate 1's *leave non-experts on Q4* budget decision (buys 0.114 BPW of expert headroom for zero code) and *zero-ABI-change* discipline. Candidate 3's 4-bit residual value is rejected as under-precision; candidate 1's 32-bit-per-outlier residual is rejected as over-budget. The merged record is 16 bits.

---

## 1. On-device weight layout

### 1.1 Geometry

Every routed expert projection has 1,048,576 elements: `gate_proj`/`up_proj` are `[rows=512, cols=2048]`, `down_proj` is `[rows=2048, cols=512]` (already row-major over its output dim, which is why one matvec shape serves all three with no transpose).

```
GROUP_SIZE = 128 (over the flat row-major stream)
G          = rows*cols/128 = 8192 groups
gpr        = cols/128      = 16 (gate/up), 4 (down)
C          = residual slots per group (fixed, artifact-wide). Ship C=2.
```

`cols % 128 == 0` for all three projections, so a group never straddles a row and the packer's flat grouping and the kernel's per-row grouping are **the same grouping**. This must be a host-side assert at pack and at upload, not a comment — the two indexings diverge silently on any unaligned tensor.

### 1.2 One contiguous buffer per projection, three regions

| offset | size (B) | dtype | contents |
|---|---|---|---|
| `0` | `rows*cols/8` = 131,072 | `uchar[]`, read as `uint[]` | **SIGNS**. Bit `e` of the flat row-major stream at byte `e>>3`, bit `e&7`, LSB-first. `1 => +scale`, `0 => -scale`. |
| `131072` | `G*2` = 16,384 | `half[]` LE | **SCALES**. One fp16 per group, group-index-major. |
| `147456` | `G*C*2` = 16,384·C | `ushort[]` LE | **RESIDUAL**. `C` records per group, group-index-major, slot `j` of group `g` at index `g*C + j`. |

Totals: **C=1 → 163,840 B = 1.25000 BPW. C=2 → 180,224 B = 1.37500 BPW.** (Q4 today: 557,056 B = 4.25 BPW on device; 4.259241 with its file header.)

All three regions are 4-byte aligned from a 16-byte-aligned buffer base, and `row*cols` is a multiple of 32 for every projection, so the `uint` reinterpret of SIGNS is legal and every sign word read is naturally aligned.

### 1.3 The residual record — 16 bits, one aligned load

```
bits 15..9  offset7   : column offset within the 128-group, 0..127
bit  8      sign
bits 7..0   mag       : magnitude, 0..255, in units of (RESID_STEP * group_scale)
```

`correction = (sign ? -1 : +1) * float(mag) * RESID_STEP * float(scale[group])`

`mag == 0` is an exact no-op, so a group with fewer than `C` worthwhile outliers writes `0x0000` and ragged density is absorbed by the **encoding**, not by control flow. `RESID_STEP` is an **f32 in the file header and in `params.pad1`**, not a compile-time constant — it is the calibration knob. Default `1/64`: correction range `±3.98 * scale`, resolution `scale/64`, against a mean-|w| of exactly `1.0 * scale`. It must be fit by the Day-0 sweep (§7), not assumed.

Per-outlier cost is **16 bits**, versus 48 in `_residual_codec` today (u32 index + fp16 value). The 7-bit offset is only affordable because placement is per-group; a flat index would need 20 bits.

### 1.4 Reconstruction, and the four things that are load-bearing

```
W[row][col] = scale[g] * (sign_bit ? +1 : -1)
            + sum over the C records of group g whose offset == (col % 128) of
              (rec_sign ? -1 : +1) * mag * RESID_STEP * scale[g]
```

Get any of these wrong and the output is finite, plausible, and wrong:

1. **The group scale is `float64 mean(|w|)` over the 128 elements, ROUNDED TO FP16 before the base is formed.** The residual is computed against the fp16-rounded base and absorbs exactly that rounding. A decoder using the f64 mean desynchronizes every correction on the tensor.
2. **Sign rule is `w >= 0.0 -> +1`.** Exact zero is positive. Writing `> 0.0` flips those elements by `2*scale`.
3. **Bit order is LSB-first** (`np.packbits(bitorder="little")`), byte-identical to the shipping HQ30G1B1 sign stream.
4. **Zero-padding of the flat tail is retained** in the scale mean. A no-op for Q80 (`1,048,576 % 128 == 0`), but the packer must not diverge from the shared base code.

Points 1-4 are exactly the shipping `HQ30G1B1` base semantics (`lab/operators/ascension_qwen30_complete_gravity.py:160-186`, parser `crates/hawking-core/src/model/qwen_complete_binary/mod.rs:133`), so **the entire base half of the packer is reused verbatim** — same float64 mean-abs, same fp16 rounding, same sign rule, same packbits order. Only the residual is new.

### 1.5 File container: HQ80BR1

Clone the `HQ30GR2` grammar (`qwen_complete_binary/mod.rs:275-374`), do not invent one. Fixed 40-byte header, no JSON:

```
<8s magic "HQ80BR1\0", u32 version=1, u32 group_size=128, u32 rank,
 u32 resid_per_group(C), u32 elements, f32 resid_step, u32 payload_bytes>
then rank * u32 dims, then the three regions verbatim.
```

Header cost 40 + 4·rank = 48 B = 0.00037 BPW. **Do not ship the lab's `HGRAVR01` JSON container** — its 252-409 byte header is pure tax and is the entire gap between the sealed 1.1269226 and the true 1.125.

### 1.6 Device table ABI — unchanged, zero widening

```
tensor.primary_address   = buffer.gpuAddress            (SIGNS base)
tensor.secondary_address = 0                            (unused; kernel derives everything)
tensor.rows, tensor.cols = as today
tensor.rank              = C                            (verified free: written as literal 0 at
                                                         qwen80_device_expert_table.rs:149, and
                                                         never referenced in the Metal struct)
tensor.kind              = QWEN80_DEVICE_EXPERT_KIND_BINARY_RESID = 4
params.group_size        = 128
params.pad0              = C  (redundant with rank, used for the ≤32-per-lane fast path decision)
params.pad1              = bitcast<u32>(RESID_STEP)
```

`Qwen80DeviceExpertTensorRef` stays **40 bytes**, `Qwen80DeviceExpertTriplet` **128**, `Qwen80DeviceExpertMatvecParams` **48**. All three `const _: () = assert!` at `qwen80_device_expert_table.rs:110-112` and their Metal `static_assert` mirrors survive untouched. The existing `kind != QWEN80_EXPERT_KIND_UNIFORM_Q4` guards in all three Q4 kernels then mechanically reject a binres artifact — **failure mode is a silent all-zeros output**, which the parity test in §6 must cover explicitly.

Side effect: **3 buffers per expert instead of 6**, so the lease resource list and every `use_resources` declaration halve.

---

## 2. Kernel set

**Three kernels replace five.** All three go in the **existing** `crates/hawking-core/shaders/qwen80_device_expert_table.metal` — no new shader file, no new `SHADER_*` const, no `all_shader_sources()` edit, and the name-existence test at `src/metal/mod.rs:1911-1923` keeps passing for free.

| # | kernel | replaces |
|---|---|---|
| K1 | `qwen80_expert_table_binres_gate_up_swiglu` | gate matvec + up matvec + `qwen80_expert_table_silu_mul` |
| K2 | `qwen80_expert_table_binres_down` | down matvec |
| K3 | `qwen80_expert_table_weighted_sum` | **unchanged** — pure f32 activation math, codec-agnostic |
| K0 | `qwen80_expert_table_binres_matvec_serial` | parity oracle only, never dispatched in the hot path |

### 2.1 Shared addressing helper

```metal
struct BinResView {
    const device uint  *sign_words;   // primary, reinterpreted
    const device half  *scales;
    const device ushort*resid;
    uint gpr;      // cols / 128
    uint n_rec;    // gpr * C
};

inline BinResView binres_view(const device Qwen80DeviceExpertTensorRef *t, uint C) {
    const uint elems      = t->rows * t->cols;
    const uint sign_bytes = elems >> 3;
    const uint G          = elems >> 7;
    BinResView v;
    v.sign_words = (const device uint  *)(t->primary);
    v.scales     = (const device half  *)(t->primary + sign_bytes);
    v.resid      = (const device ushort*)(t->primary + sign_bytes + G * 2u);
    v.gpr        = t->cols >> 7;
    v.n_rec      = v.gpr * C;
    return v;
}
```

### 2.2 The two mechanisms that make this regular

**Sign-word broadcast.** For row `r`, group `gi`, sub-iteration `t`, the 32 columns `gi*128 + t*32 + [0..31]` occupy bits `(r*cols + gi*128 + t*32) .. +31` — exactly `uint32` word `(r*cols + gi*128 + t*32) >> 5`. **All 32 lanes load that same word** (one 4-byte broadcast transaction) and lane `L` consumes bit `L`. Weight traffic is exactly **1 bit per weight, zero amplification**, against Q4's helper (`qwen80_device_expert_table.metal:72-86`) which issues 1 uchar + 1 half per weight = 3 bytes of load issue per 4 bits (6x) and re-fetches the group scale on all 64 elements of its group. Do **not** bit-transpose or swizzle the layout — the natural LSB-first stream is what gives the broadcast.

**Hoisted scale per group.** A lane accumulates **unscaled** `±x` across the 4 sub-iterations of one group, then does one `fma(unscaled, scale, acc)`. 16 scale multiplies per row instead of 2048. And because the residual value is *already expressed in units of the group scale*, it folds into the same unscaled sum at zero extra multiplies.

**Why lanes stride columns rather than owning a contiguous window:** the alternative (lane `L` owns columns `L*64 .. L*64+63`) hoists the scale even harder but makes all 32 lanes read `xs[L*64 + k]` — stride 64 words, `64 % 32 == 0`, a **32-way threadgroup bank conflict on every activation read**. Column-interleaved lanes read `xs[base + lane]`: consecutive, conflict-free.

### 2.3 K1 — fused gate + up + SiLU

Threadgroup 256 threads = 8 simdgroups. **One simdgroup per (route, row)**, 8 rows per threadgroup. Threadgroup memory `float xs[2048]` = 8 KiB, staged once and reused by 8 rows × 2 projections (the wave passes `input_stride_elems = 0` for gate/up, so all 10 routes read the same hidden vector).

```metal
kernel void qwen80_expert_table_binres_gate_up_swiglu(
    const device uint *route_ids                    [[buffer(0)]],
    const device Qwen80DeviceExpertTriplet *table   [[buffer(1)]],
    const device float *input                       [[buffer(2)]],
    device float *activated                         [[buffer(3)]],
    constant Qwen80DeviceExpertMatvecParams &p      [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[2048];

    const uint groups_per_route = (p.max_rows + 7u) / 8u;      // 512/8 = 64
    const uint route = tgid / groups_per_route;
    if (route >= p.experts_per_token) return;
    const uint expert = route_ids[route];
    if (expert >= p.n_experts) return;

    const device Qwen80DeviceExpertTriplet &e = table[expert];
    const device Qwen80DeviceExpertTensorRef *G = &e.gate;
    const device Qwen80DeviceExpertTensorRef *U = &e.up;
    if (e.ready_mask != QWEN80_EXPERT_TRIPLET_READY ||
        e.generation != p.generation ||
        G->generation != p.generation || U->generation != p.generation ||
        G->kind != QWEN80_EXPERT_KIND_BINARY_RESID ||
        U->kind != QWEN80_EXPERT_KIND_BINARY_RESID ||
        G->primary == nullptr || U->primary == nullptr ||
        p.group_size != 128u) return;

    // stage x once per THREADGROUP. Only place the activation touches device memory.
    const device float *xsrc = input + p.input_base_elems + route * p.input_stride_elems;
    for (uint i = tid; i < G->cols; i += 256u) xs[i] = xsrc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = (tgid % groups_per_route) * 8u + sgid;
    if (row >= G->rows || row >= p.max_rows) return;

    const uint  C    = G->rank;                       // 2
    const float step = as_type<float>(p.pad1);
    BinResView vg = binres_view(G, C), vu = binres_view(U, C);

    const uint wbase = (row * G->cols) >> 5;          // sign word index of this row
    const uint gbase = row * vg.gpr;                  // scale/group index of this row

    float accg = 0.0f, accu = 0.0f;

    // ---- BASE: 16 groups x 4 sub-iterations of 32 lanes = 2048 columns ----
    for (uint gi = 0; gi < vg.gpr; ++gi) {
        float ug = 0.0f, uu = 0.0f;
        const uint c0 = gi << 7;
        for (uint t = 0; t < 4u; ++t) {
            const uint w  = (c0 + (t << 5)) >> 5;
            const uint sg = vg.sign_words[wbase + w];  // 4-byte broadcast, 32 weights
            const uint su = vu.sign_words[wbase + w];
            const float xv = xs[c0 + (t << 5) + lane]; // conflict-free tgmem
            ug += ((sg >> lane) & 1u) ? xv : -xv;      // select + add, no multiply
            uu += ((su >> lane) & 1u) ? xv : -xv;
        }
        accg = fma(ug, float(vg.scales[gbase + gi]), accg);
        accu = fma(uu, float(vu.scales[gbase + gi]), accu);
    }

    // ---- RESIDUAL: one contiguous 64-byte record block per (row, projection) ----
    // n_rec = 32 for gate/up at C=2 -> exactly one record per lane, one iteration.
    for (uint r = lane; r < vg.n_rec; r += 32u) {
        const uint gi = r / C;
        {   const ushort rec = vg.resid[row * vg.n_rec + r];
            const float  v   = float(rec & 0xFFu) * step * ((rec & 0x100u) ? -1.0f : 1.0f);
            accg = fma(v * float(vg.scales[gbase + gi]), xs[(gi << 7) + (rec >> 9)], accg); }
        {   const ushort rec = vu.resid[row * vu.n_rec + r];
            const float  v   = float(rec & 0xFFu) * step * ((rec & 0x100u) ? -1.0f : 1.0f);
            accu = fma(v * float(vu.scales[gbase + gi]), xs[(gi << 7) + (rec >> 9)], accu); }
    }

    const float g = simd_sum(accg);
    const float u = simd_sum(accu);
    if (lane == 0u) activated[route * p.max_rows + row] = (g / (1.0f + exp(-g))) * u;
}
```

**Why the residual has no divergence and no serialization:**

1. **Uniform trip count.** `n_rec` is a property of the tensor, identical for every lane, every simdgroup, every threadgroup. The loop runs `ceil(n_rec/32)` iterations — 1 for both gate/up (32) and down (8). No CSR row-pointer, no prefix sum, no scan, no ballot, no per-row count.
2. **Ragged density is encoded, not branched.** A group with fewer than `C` worthwhile outliers stores `mag = 0`, whose contribution is an exact arithmetic zero.
3. **Coalesced.** Lane `L` reads record `L` of a contiguous block — 32 consecutive `ushort` = one 64-byte transaction.
4. **Applied exactly once, no atomics.** Each record is owned by exactly one lane and lands in that lane's accumulator; `simd_sum` collects it. No scatter, no collision, no second reduction, no second dispatch.
5. **No unsafe load.** `xs[(gi<<7) + off]` is in-bounds for every lane by construction (`off < 128`, `gi*128 + 128 <= cols`), so it issues unconditionally with no guard.

Cost: 1 iteration of ~6 ops against 64 base iterations = **~3% ALU overhead**. Compare the existing Q30 sparse path (`shaders/qwen30_quality_repack_sparse_gate_up.metal:44-50`), which scans the entire residual list per output row and filters with `if ((flat / cols) == row)` — at Q80 scale that is 512 × 20,972 = **10.7M index loads to apply 20,972 corrections**. This design does zero wasted index loads.

### 2.4 K2 — down

Structurally identical with `cols = 512`, `gpr = 4`, `n_rec = 4·C = 8`, `xs[512]` (2 KiB), `rows = 2048`, input `activated` at `input_base_elems + route*512`, output `down[route*2048 + row]`, no SiLU. Sign load per group is one word per sub-iteration exactly as above (`512/32 = 16` sub-iterations, 4 groups).

### 2.5 K0 — serial oracle

One thread per `(route, row)`, strict ascending column order, group scale applied at each group boundary in flat order, then the residual walked ascending. Reuses the 8-columns-per-sign-byte unroll already written at `shaders/qwen_binary.metal:50-61`. **Never dispatched in the hot path** — it exists so §6 has a reference whose accumulation order matches the CPU decoder exactly.

### 2.6 Per-lane work and traffic

| | K1 (gate+up, per row) | K2 (down, per row) |
|---|---|---|
| base iterations / lane | 64 | 16 |
| ops / iteration | 2 broadcast word reads, 1 tgmem read, 2 select-adds | 1 word read, 1 tgmem read, 1 select-add |
| scale fma / lane | 32 | 4 |
| residual iterations / lane | 1 | 1 |
| **distinct DRAM / row** | 2×(256 signs + 32 scales + 64 resid) = **704 B** | 64 + 8 + 16 = **88 B** |

704 × 512 = 360,448 = 2 × 180,224 ✓ · 88 × 2048 = 180,224 ✓ — the arithmetic closes against the slab size, which is the check that the layout has no hidden re-read.

---

## 3. Dispatch structure per token

| | Q4 today | HQ80BR1 |
|---|---|---|
| dispatches / layer (expert wave) | 5 | **3** |
| dispatches / token (expert wave) | 240 | **144** |
| dispatches / token (whole) | 1,155 | **1,059** |
| MTLComputeCommandEncoder creations / token | 1,155 | 1,059 |
| command buffers + `commit_and_wait` fences / token | 98 | **98 (unchanged)** |
| host router readbacks / token | 48 | 48 (unchanged) |
| host staging memcpy / token | 802.2 MB | **259.5 MB** |
| Metal buffers per cached expert | 6 | **3** |
| `use_resources` slabs per matvec | 6 | 3 |

**Grid shapes** (top_k = 10, 60 GPU cores):

| kernel | grid_x | threadgroups | tgmem |
|---|---|---|---|
| K1 | `10 * ceil(512/8) * 256` = 163,840 | **640** | 8 KiB |
| K2 | `10 * ceil(2048/8) * 256` = 655,360 | **2,560** | 2 KiB |
| K3 | 2,048 | 8 | 0 |

**Against the current default, honestly labelled.** `qwen80_expert_table_kernel()` at `qwen80_device_expert_table.rs:51-61` returns `Rowblock` for anything unset. For gate/up, `max_rows = 512`, so its grid is `10 * ceil(512/1024) * 256 = 2,560` threads = **10 threadgroups** on a 60-core GPU, and `row0 = lid*4` kills every `lid >= 128`, so **half of each threadgroup exits immediately** — roughly 8% of the machine. K1 launches 640 threadgroups with every lane live.

**That 64x is a kernel-geometry win, not a codec win.** `qwen80_expert_table_uniform_q4_matvec_simdgroup` already has exactly this 640/2,560 shape and is one env var away (`HAWKING_QWEN80_Q4_KERNEL=simdgroup`). **Any A/B must be HQ80BR1 vs Q4-simdgroup.** Benchmarking against the Rowblock default would credit the codec with a win it did not earn; do not do it and do not accept a receipt that does.

**Gather is unchanged and deliberately so.** Device router matvec → host readback (`qwen80_uniform_q4_hybrid_decode.rs:2682`) → host top-10 (`:2684`) → `route_ids` written to the wave buffer (`:2715-2722`) → `write_compact_selected_table` memcpys the 10 selected experts into fixed-stride compact slabs → kernels index `table[route_ids[route]]`. Only the byte constants change. The 802 → 259 MB/token memcpy reduction falls out for free.

**Out of scope, larger, and where the tok/s actually lives** (all codec-independent, all doable on the Q4 vehicle today): one `begin_serial_group` per command buffer (machinery live for Q30 at `src/metal/mod.rs:4011-4047`, plus `use_resources_read_on_group` at `:4070` to hoist residency out of every matvec — zero calls on any qwen80 path); dispatching the already-compiled-but-never-dispatched `qwen80_postnorm_router_top10_select` (`shaders/qwen80_postnorm_router_top10.metal:153`) so the prefix and suffix command buffers can merge and the 98 fences collapse; and switching to `write_selected_expert_table` (`qwen80_device_expert_table.rs:529`, written, currently `#[allow(dead_code)]`) to delete the staging memcpy entirely.

---

## 4. Bytes per token, and the throughput prediction

### 4.1 Arithmetic

Active parameters per token (from `receipts/QWEN80_BIT_BUDGET_LEDGER.json` geometry): experts `48 × 10 × 3 × 512 × 2048 = 1,509,949,440`; non-expert `2,053,817,088` (attention/DeltaNet/norm/router 1,591,655,168 + lm_head 311,164,928 + shared expert 150,994,944 + embed row). **Expert weights are 42.3% of per-token traffic, not 97%** — the ledger identity weights by *static* mass, per-token traffic weights by *active* mass. This is the single most important number in the document.

| expert BPW | B / projection | expert B/token | + non-expert @ Q4 | total GB/token | vs Q4 |
|---|---|---|---|---|---|
| 4.259241 (Q4, baseline) | 557,056 | 802,160,640 | 1,093,462,743 | **1.8956** | 1.000x |
| 1.1269 (lab binary, no residual) | 147,708 | 212,699,520 | 1,093,462,743 | **1.3062** | 1.451x |
| **1.375 (this design, C=2)** | 180,224 | 259,522,560 | 1,093,462,743 | **1.3530** | **1.401x** |
| 1.45 (brief's hypothetical) | 190,054 | 273,678,336 | 1,093,462,743 | **1.3671** | 1.387x |

Expert-only ratios: 3.778x at 1.1269, **3.091x at 1.375**, 2.931x at 1.45. The brief's "3.8x" is the **no-residual** number; the residual costs about half of it back.

### 4.2 Complete-artifact budget

`complete_bpw = 0.97032*expert + 0.02968*nonexpert`

| expert | non-expert Q4 (4.259241) | non-expert 8-bit |
|---|---|---|
| 1.250 (C=1) | **1.3393 PASS** | 1.4503 PASS |
| **1.375 (C=2)** | **1.4606 PASS** (margin 0.039) | 1.5716 FAIL |
| 1.500 (C=3) | 1.5819 FAIL | — |

**Leave non-experts on the Q4 vehicle they already have.** It costs zero work, buys 0.114 BPW of expert headroom, and the ledger's own recommended operating point (non-experts at 8 bits) is actively harmful here: it costs **2.2994 GB/token, a 1.21x traffic regression versus Q4**, because it *raises* the 57.7% of active bytes that are non-expert. **The 1.5 BPW gate is an artifact-size gate. At the ledger's recommended mix it is anti-velocity.** Artifact size: 42.42 GB → **13.7 GB**.

### 4.3 Throughput prediction

**Assumptions, stated as assumptions:**
- A1: 2.9249 tok/s = 341.9 ms/token is the real device-path baseline. **This number appears nowhere in this repo.** Repo-wide grep finds it only in unrelated float literals; the only sealed Q80 velocity receipt (`receipts/QWEN80_UNIFORM_Q4_VELOCITY_BASELINE.json`) is a CPU-only run at 0.3262 tok/s with `metal_error: "metal: no Metal-capable GPU"`. Every percentage below takes A1 on faith.
- A2: the ~700 GB/s blit roofline bounds kernel weight reads. It is a *device-copy* ceiling; no sealed measurement exists on this box of what a Q4 or binary matvec kernel achieves against DRAM.
- A3: the expert working set is warm (cached in `expert_cache`, which never evicts). If cold, the binding constraint is the 8.76 GB/s `fs::read` path or the 0.448 GB/s first-touch sha256, not DRAM at all.
- A4: everything not in the expert wave is unchanged.
- A5: KV-cache traffic excluded (negligible at short prompts, grows linearly).

**Prediction from the byte cut alone:**

```
Q4 bandwidth floor        1.8956 GB / 700 GB/s = 2.708 ms/token  (369 tok/s)
HQ80BR1 bandwidth floor   1.3530 GB / 700 GB/s = 1.933 ms/token  (518 tok/s)
saving                                            0.775 ms/token
0.775 / 341.9 = 0.227%   ->   2.9249 -> 2.932 tok/s
```

**Predicted end-to-end speedup from the codec: +0.2%, i.e. unmeasurable.** At 1.1269 BPW it is +0.25%; at 1.45 BPW it is +0.22%. The three numbers are within noise of each other **and of zero**, which is the finding: at 126x off the bandwidth bound, expert BPW does not appear in the wall clock.

**What the design plausibly does buy, conditionally:** the host staging memcpy falls 802 → 259 MB/token. That runs on the CPU at memcpy speed on the critical path, not at DRAM speed. If `stages.moe_table_build_secs` (already accumulated at `qwen80_uniform_q4_hybrid_decode.rs:1330`, **never read**) turns out to be ~40 ms/token, this returns ~27 ms ≈ **8%**. That input is unmeasured and is experiment E0 below.

**Range, honestly:** **−3% to +10%.** The downside is real: this codec adds a second weight stream and is not bit-identical, so it can lose to a well-tuned Q4 simdgroup kernel.

**What would have to be true for 3.8x:** the 98 `commit_and_wait` fences, the ~1,155 per-dispatch encoders, the 48 host router round-trips and the staging memcpy would all have to be gone first, putting the token within ~2x of its 2.708 ms bandwidth floor. **Then, and only then**, a 1.40x byte cut buys ~1.4x. Even in that world it is 1.4x, not 3.8x — because 57.7% of the bytes are non-expert.

---

## 5. Rust integration points

All paths relative to `/Users/scammermike/Downloads/hawking/crates/hawking-core/`.

**`src/model/qwen80_device_expert_table.rs`** (the whole codec surface):

| line | change |
|---|---|
| `:13` | add `pub const QWEN80_DEVICE_EXPERT_KIND_BINARY_RESID: u32 = 4;` |
| `:45` | `enum Qwen80ExpertTableKernel` → add `BinRes` |
| `:51` | `qwen80_expert_table_kernel()` → add `"binres" => BinRes` to the `HAWKING_QWEN80_Q4_KERNEL` match |
| `:63` | `QWEN80_EXPERT_TABLE_KERNELS: [&str; 5]` → `[&str; 8]`, add the three new names |
| `:73-112` | ABI structs — **no change**, all three size asserts survive |
| `:128-130` | add `QWEN80_EXPERT_BINRES_SIGN_BYTES = 131_072`, `_SCALE_BYTES = 16_384`, `_RESID_BYTES = 16_384*C`, `_PROJ_BYTES` |
| `:136-152` | `tensor_ref()` → take `kind` and `rank`; today it hardcodes `rank: 0` and `kind: UNIFORM_Q4` |
| `:157` | `Qwen80ExpertGpuTriplet` — 6 buffers → 3 |
| `:428` | `upload_proj` — one `fs::read` + one buffer instead of two of each |
| `:459` | `upload_qwen80_expert_triplet` |
| `:529` | `write_selected_expert_table` (dead) — the follow-up that deletes the staging copy entirely |
| `:569-595` | `Qwen80CompactExpertSlabs` — 6 slabs → 3, new sizes |
| `:610` | `copy_buf` — unchanged, already generic |
| `:626-711` | `write_compact_selected_table` — 6 copies → 3, new strides, fill `rank = C` and `kind = 4` |
| `:729` | `matvec_dispatch_shape` — add the `BinRes` arm (identical grid arithmetic to the existing `Simdgroup` arm) |
| `:791-798` | `dispatch_qwen80_device_expert_table_wave` — `resources.len() != 6` → `!= 3`; branch to the 3-dispatch sequence |
| `:889-946` | `encode_matvec` — set `params.group_size = 128`, `pad0 = C`, `pad1 = step.to_bits()`; K1 binds the same 4 buffers + params |
| `:1004-1006` | test asserts on byte constants and `KERNELS.len()` |

**`src/model/qwen80_uniform_q4_hybrid_decode.rs`** (artifact ingest — three hard gates will otherwise reject the payload):

- `:175-183` — manifest `schema` equality check; accept `QWEN80_BINRES_SCHEMA` as well.
- `:241-300` — `read_payload` (`fs::read` + memoized sha256) is codec-agnostic, **reuse as-is**.
- `:293` and `:403-408` — two `header.group_size != UNIFORM_Q4_GROUP_SIZE` checks, both hardcoded to 64. Branch on the container magic.
- `:435-446` — `Qwen80Q4PackedTensor::codes()/scales()`; add a sibling `Qwen80BinResPackedTensor` (~60 lines) rather than generalizing the Q4 one.
- `:970` — `expert_cache: HashMap<(usize,u32), Qwen80ExpertGpuTriplet>`, never evicted. 3x smaller payloads move the worst case from ~39 GiB to ~13 GiB. **This does not fix the leak** — do not let the codec change be the reason nobody adds eviction.
- `:1288-1334` — `ensure_selected_expert_table`, unchanged in shape.
- `:2691-2698` — where `route_ids[10]` is filled. **This is the E1 probe site** (§7).
- `:2738-2744` — the live wave dispatch. Unchanged.

**`shaders/qwen80_device_expert_table.metal`** — add `QWEN80_EXPERT_KIND_BINARY_RESID = 4u`, `binres_view()`, K0/K1/K2. Nothing else in Metal-land changes: **no** new `SHADER_*` const at `src/metal/mod.rs:358`, **no** edit to `all_shader_sources()` at `:405-443`, and the kernel-name-existence test at `:1911-1923` passes unmodified.

**`src/model/qwen_complete_binary/mod.rs:275-374`** — `HQ30GR2` parser is the template for the `HQ80BR1` parser + CPU decoder. Reuse its validation shape.

**Packer** — `lab/operators/ascension_qwen30_complete_gravity.py:160-186` is the base packer, reusable verbatim. `lab/operators/ascension_qwen30_quality_repack.py:238-259` is the **deterministic tie rule** (`np.partition` boundary, strictly-greater, ties by ascending flat index) — use it, **not** the lab's bare `np.argpartition` at `ascension_dual_gravity_worker.py:739`, which has no tie rule and whose output depends on quickselect internals.

**Not touched, and must not be:** `forward_token_device` control flow, command-buffer structure, state-slot arithmetic; `DeviceActivationWorkspace` and `Qwen80DeviceExpertWaveWorkspace` (pure f32/u32, sized only by HIDDEN/MOE_INTERMEDIATE/TOP_K); the router matvec and host top-10; the 512-way `route_ids` indirection and the generation/ready_mask protocol; all 7 kernels in `qwen80_device_activations.metal`; the DeltaNet and GQA mixers; and **the entire non-expert weight path**, which stays on `qwen_uniform_q4_group64_matvec`.

---

## 6. Correctness plan

Bit-identity to the Q4 path is impossible (different artifact) and bit-identity of K1/K2 to a serial reference is impossible (`simd_sum` and the hoisted per-group scale both reassociate). The contract is therefore **staged**, with exactness required wherever it is actually achievable.

**Gate 1 — packer round-trip, EXACT.** Python packer emits HQ80BR1 → Rust `HQ80BR1` parser → CPU `decode_binres_f32` → compare against the packer's own reconstruction array. **Require bit-exact f32 equality on all 1,048,576 elements.** This is non-negotiable and it fixes the lab codec's standing sin: `_binary_codec` and `_residual_codec` both return a reconstruction built from encoder-local arrays and have **never round-tripped through their own bytes**, in direct contradiction of their module docstring at `ascension_dual_gravity_worker.py:569-575`. Every cosine in `receipts/QWEN80_REPRESENTATION_FRONTIER_SWEEP.json` was scored from encoder state, not from decodable bytes.

**Gate 2 — K0 vs CPU, EXACT weight decode, ≤1 ulp dot product.** K0 walks columns in the same ascending order with the same per-group scale application as the CPU reference. Weight *decode* is a sign select on an fp16→f32 scale plus `mag*step*scale` — the same ops in the same order on both sides, so require `max |Δ| == 0` on decoded weights (dump them from K0 in a test-only mode). For the dot product, require relative error ≤ 1e-6 per output element.

**Gate 3 — K1/K2 vs CPU reference, stated tolerance.** On ≥ 8 real organs spanning ≥ 4 layers: `max relative error ≤ 1e-4` and `per-row cosine ≥ 1 - 1e-6` against the CPU reference dot product. Seal as a receipt. This replaces bit-identity and **someone has to explicitly agree it is sufficient** — the Q80 lane's stated convention (`shaders/qwen_binary.metal:8-19`) is that the default kernel name is the bit-identical one. That is a process decision, not a technical one, and it should be made before the Metal is written, not after.

**Gate 4 — silent-zero guard.** Assert that dispatching a Q4 kernel against a `kind=4` table produces all zeros (it will — the existing guards early-return) and that the wave rejects the mismatch at the Rust level before dispatch. This is the one failure mode that produces plausible-looking garbage.

**Gate 5 — coherence, not tokens.** End-to-end, the binres artifact will not reproduce the Q4 token stream and should not be expected to. The gate is the 0.8604 output-space cosine bar per organ, plus a greedy-generation coherence read. Extend the existing Q80 table test at `qwen80_device_expert_table.rs:1030-1093` (which today pins `Rowblock` and checks correctness only) with a synthetic binres expert for Gates 2-4.

**Known gap:** where the 0.8604 bar is *defined* could not be located — it appears as the `"bar"` field in the frontier receipt, and grepping `lab/operators/doctor6/coherence.py` for it returns nothing. Whether it is per-organ, per-layer or a global aggregate is unresolved and should be settled before it is used as a ship gate.

---

## 7. The cheapest experiment that falsifies the throughput claim

**This is the most important section. Run E0 and E1 before writing one line of packer or Metal. Together they are under an hour and roughly five lines of code.**

### E0 — attribution, ZERO code, ~15 minutes

Run one generation with the existing instrumentation and read four numbers that already exist and have never been read:

- `Qwen80DecodeStageTimes` at `qwen80_uniform_q4_hybrid_decode.rs:740-753` — `moe_table_build_secs` (this is the 802 MB/token memcpy), `moe_routed_secs`, `moe_combine_secs`, `activation.metal_matvec_sync_secs`.
- `TokenCommandBuffer::dispatch_count()` (`src/metal/mod.rs:3914`) and `enable_structural_kernel_trace()` (`:3924`) — confirm the 98 CB / 1,155 dispatch topology, which is *derived* in this document, not measured.
- Same run with `HAWKING_QWEN80_Q4_KERNEL=simdgroup` vs default `rowblock`. **One env var.** This tests the occupancy half of the thesis today, on the Q4 vehicle, with no new codec.

**First, this also establishes the 2.9249 tok/s baseline, which does not exist in this repo.** If `moe_table_build_secs` is a large share of 341.9 ms, the memcpy is the story and the codec's only real win is the one it gets for free. If `simdgroup` beats `rowblock` substantially, that win belongs to geometry and must be subtracted from any later codec claim.

### E1 — the falsifier: the one-expert probe, ~5 lines, ~20 minutes

**The claim under test: "expert weight DRAM traffic is a material fraction of the token."**

Add an env-gated override immediately after the top-10 selection at `qwen80_uniform_q4_hybrid_decode.rs:2691-2698`:

```rust
// falsification probe only: collapse the distinct expert working set 10x while
// holding dispatch count, grid shape, memcpy volume, ALU and fence count identical.
if std::env::var("HAWKING_QWEN80_ONE_EXPERT_PROBE").is_ok() {
    let first = route_ids[0];
    route_ids = [first; 10];          // route_weights left untouched
}
```

This produces **wrong output on purpose**. What it holds *exactly* constant: dispatch count, grid geometry, threadgroup count, ALU per lane, `write_compact_selected_table`'s memcpy volume (ten copies of the same source is the same bytes moved), the 98 fences, the 48 host readbacks, and every activation buffer. What it changes: the *distinct* expert weight footprint, from 16.71 MB/layer (802 MB/token) to 1.67 MB/layer (80 MB/token) — a **10x cut in expert DRAM traffic**, which is a larger cut than this entire codec delivers (3.09x).

Measure steady-state tok/s after warm-up (so `expert_cache` is populated in both arms and no `fs::read`/sha256 confound is present), ≥ 30 tokens, 3 runs.

**Decision rule:**

| observation | conclusion |
|---|---|
| tok/s **unchanged** (within ±3%) | Expert weight bandwidth contributes ~0 to the token. **The velocity thesis is dead on this runtime.** HQ80BR1 is a footprint-and-quality change only. Do not build the kernel for speed; build it (if at all) for the 1.5 BPW gate, and spend the engineering on dispatch topology instead. |
| tok/s up **~1.6x** | Bandwidth-bound as the thesis predicts (cutting 42.3% of bytes by 10x cuts total ~38%). The codec is worth building for speed and should deliver ~1.4x. |
| tok/s up **1.05-1.3x** | Partially bandwidth-sensitive. Scale the expected codec win by the ratio: a 3.09x expert cut delivers roughly (measured gain) × 0.73. |

Given every mapped fact — 0.79% of the measured ceiling, Q30's 51% GPU-idle, and the repo's own sealed "occupancy, not bandwidth" note — **the predicted outcome is row 1**. E1 exists to make that prediction falsifiable at a cost of five lines instead of seven engineer-days.

### E2 — the quality gate, ~2 days, Python only, no Rust, no Metal

Independent of E0/E1 and equally capable of killing the design. Rerun `lab/operators/q80_representation_frontier_sweep.py` (fracs list at `:66`) with:

- **per-group top-C selection** at C ∈ {1, 2, 3} (densities 0.78%, 1.56%, 2.34%) instead of global top-|residual|;
- **the 16-bit quantized record** (7-bit offset, sign, 8-bit magnitude) instead of fp16 values, sweeping `RESID_STEP` ∈ {1/32, 1/64, 1/128};
- **more than 4 organs** — the entire existing frontier is gate/up of two `(layer, expert)` pairs, (10, 453) and (3, 494), out of 73,728 expert tensors;
- **activation-weighted selection** (score by `|residual| * E[|x_col|]`) as an arm. Selection today is weight-space while the pass/fail metric is output-space (`_mean_row_cosine(X@W.T, X@rec.T)`) — this is a free lever that spends the same bits on the corrections that actually move the output, and at ~1.5% density it may be the difference between pass and fail.

**Why this can kill it:** the only measured passing point is global top-2% with fp16 values (2.0881 BPW, 60% over budget), and even there the two `up_proj` organs cleared 0.8604 by **0.0047 and 0.0061**. The binary base alone fails on 3 of 4 organs. This design moves *three* parameters in the wrong direction at once — lower density, constrained per-group placement, quantized values.

**And `down_proj` has never been scored at any residual fraction**, because the sweep skips it: its input is the post-SwiGLU 512-dim intermediate, not the captured 2048-dim hidden (`q80_representation_frontier_sweep.py:37-38`). doctor6 treats it as the *discriminating* organ and doubles its residual fraction (`rungs.py:454`). **Capturing a post-SwiGLU X for Q80 is a prerequisite sub-project and is the real schedule risk** — the budget may be dominated by the one organ nobody has measured.

### Sequencing

```
E0  (0 code, 15 min)   attribute the token; establish the baseline that does not exist
E1  (5 lines, 20 min)  falsify or confirm that bytes matter at all
E2  (2 days, Python)   does 1.56% per-group with 16-bit records clear 0.8604, incl. down_proj
--- gate: if E2 fails, STOP. If E1 says row 1, build only for the size gate, if at all. ---
packer          1.5 d
HQ80BR1 parser  1.0 d   (clone HQ30GR2)
Metal K0/K1/K2  1.5 d   (assembling proven pieces: sign-word decode from qwen_binary.metal:50-61,
                         simdgroup skeleton from the existing Q4 simdgroup kernel,
                         fused gate/up from qwen30_device_expert_table.metal:1297)
Rust dispatch   1.5 d
parity receipt  1.0 d
```

~6.5 engineer-days behind a 2-day gate. The kernel is the cheap part; the packer and the coherence question are the schedule.

---

## What is unknown, stated plainly

- **The 2.9249 tok/s baseline is not sealed anywhere in this repo.** Every percentage here rests on it.
- **Nothing has been measured between 0% and 2% outlier density**, and the 2% points cleared the bar by ~0.005.
- **`down_proj` coherence is unmeasured at every fraction**, and doctor6 believes it is the organ that decides.
- **Residual value precision has never been swept** — only the fraction. The 16-bit record's clipping behaviour on true outliers is an open question and `RESID_STEP` is a fitted constant, not a derived one.
- **The frontier is n=4 organs out of 73,728**, and none of its cosines were computed from physically decodable bytes.
- **No GPU-timestamped per-kernel profile exists for Q80.** The dispatch-bound verdict is inferred from Q30's sealed profile plus Q80's command-buffer topology.
- **Whether the compact-slab memcpy is load-bearing.** Nothing documents why `write_compact_selected_table` was chosen over the already-written `write_selected_expert_table`. If it exists to avoid declaring residency over 30 separate buffers, part of the memcpy win reverses.
- **Whether the BF16 source for all 73,728 expert tensors is on disk.** Not verified. A prior campaign in this tree lost a source tree behind a guard that reported MISSING as a clean escape. Confirm before scheduling the repack, not after.