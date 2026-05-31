# Cheaper-decode Q3_K design — 2026-05-31 (DESIGN ONLY, no code/commits)

> **⛔ STATUS 2026-05-31 — NO-GO (clean-room confirmed). DO NOT BUILD.** The free
> pre-build oracle (clean `q3k_bytecut_bench`, Claude quit, `clean_room_batch.sh`
> §A) read f32-predec-Q3 at **33.3 GB/s = 22% of peak**, and Q3_K is *slower in
> absolute µs* than Q4_predec on all shapes despite ~half the bytes — the Q3_K
> GEMV is compute/residual-bound, NOT bandwidth-bound, so the f16-predec-Q3 128-B
> repack designed below cannot flip it to a tps win (it shaves bytes off a kernel
> that isn't on the bus). The byte-cut axis routes through **QTIP**
> (`plans/qtip_bytecut_design_2026_05_31.md`) only; Q3_K stays footprint-only.
> Kill: `reports/dead_levers.md` "Q3_K sub-Q4 decode byte-cut".

> **Scope.** A design doc, not an implementation. It evaluates a *speed-viable*
> Q3_K-class byte-cut for the dense decode GEMV (M=1) on Qwen2.5-3B / M3 Pro.
> Per bible §3.0, the decode-kernel-microopt track is exhausted (the Q4_K predec
> GEMV is at the Apple-GPU memory-model optimum) and **fewer bytes is the only
> remaining path to >34 dense tps** — but today's Q3 byte-cut is *footprint-only*
> because Q3 decode is too slow (A8: fastest Q3_K is 22–43% **slower** than
> Q4_K-predec; the Q3_K kernels run 6–33 GB/s, compute-bound on inline 6-bit
> scale decode). This doc designs the synthesis neither Lane 2 (inline, slow)
> nor A8 (f32-predec, +bytes) tested: **cheap-decode AND fewest-bytes.**
>
> No source edited. No cargo. No GPU. No commits. Output is this file.

---

## 0. TL;DR / recommendation

**Recommended format: `f16-predec-Q3` on a REPACKED 128-B block** (not the
174-B in-place predec the A8 bench measured), built by mirroring the existing
A6.5 f16s machinery. Contrast codec (QTIP trellis) is **deferred** — it is a
heavier, attended Colab+kernel project with no cheap in-tree oracle, and the
f16-predec-Q3 path reuses machinery we already shipped and trust.

**The honest verdict up front (the byte/decode math, worked in §3–§4):**

- The byte-cut **is real and large** once the block is repacked: **128 B/block**
  for f16-predec-Q3 vs **176 B/block** for Q4_K-predec-f16s (A6.5) and **208 B**
  for Q4_K-predec-f32. That is **−27% vs the f16s `_pair` kernel that is 46.6% of
  decode**, and −38% vs f32 predec. Bigger byte-cut than Q3-vs-Q4 weight-only
  (110 vs 144 = −24%), because predec *also* lets us drop Q3's dead scale bytes.
- The decode-cost premise **plausibly flips from compute-bound to BW-bound**:
  predec hoists the entire branchy 6-bit `q3_k_scale` unpack out of the hot loop
  (the measured bottleneck), leaving a hot loop that is *structurally simpler than
  Q4_K's* (a 2-bit + 1-high-bit unpack and an FMA, vs Q4_K's nibble unpack +
  sub-block-min subtract). If the f16-predec-Q3 hot loop reaches the same ~56% of
  peak that Q4_K-predec-`_pair` reaches, then at 128 B/block it beats Q4_K-predec
  on µs/call by the byte ratio (~0.73×, i.e. ~27% faster).
- **The risk that keeps this HONEST (not a foregone win):** the Q3 hot loop has
  *more integer ALU per element than Q4_K even after predec* — the high-bit gather
  from `hmask` (`(hmask[h_idx] & (1<<bit)) ? 0 : 4`) is a per-element branch/select
  that Q4_K has no analog for. If that residual per-element compute (not the
  hoisted scale decode) is itself enough to keep Q3 compute-bound at 128 B, the
  byte-cut still won't beat Q4_K on µs/call. **A8 measured the scale-decode was
  the dominant cost; it did NOT isolate how much the hmask gather alone costs.**
  That isolation is exactly what the microbench gate in §6 settles.

So this is a **Type-2 reframe of the A8 kill with a named, cheap on-GPU oracle**
(the §6 microbench, re-run with the new kernel). It is *alive* under the Kill
Protocol — but it is not pre-declared a win. The gate decides.

---

## 1. Where the A8 kill actually landed (the thing being reframed)

A8 (overnight queue, committed oracle) and Lane 2 (`gemm_q3_k_fused_2r`,
`c1f5275`) jointly established:

- **Q3_K decode is COMPUTE-bound, not BW-bound.** The fused/fused_2r kernels run
  7–21 GB/s (some shapes up to 33) on a 150 GB/s machine. A BW-bound kernel on the
  same shapes (Q4_K-predec) runs 18–50 GB/s. The Q3 kernel is starved on ALU, not
  on the bus.
- **The cost center is the inline 6-bit scale decode** — `q3_k_scale()` in
  `shaders/quant.metal:26-37`: a branch on `scale_idx < 8`, a low-nibble select, a
  high-2-bit shift+mask from a *second* byte region, an OR, and a `-32`. Called
  **once per element** inside `q3_k_value` (`:55`). That is ~256 invocations per
  block per row, each with a branch and two dependent byte reads.
- **Row-ILP (2r) attacks the wrong bottleneck.** 2r hides *DRAM latency* by
  giving the compiler 2 in-flight load streams. Q3 isn't waiting on DRAM; it's
  waiting on the scale-decode ALU chain. So 2r helped only the square shape (+8%)
  and *regressed* the wide FFN shapes (−5 to −30%) from doubled accumulator
  register pressure. fused_2r stayed −32 to −55% vs Q4_K-predec.
- **f32-predec exists and is fast-decode but byte-heavy.** `gemm_q3_k_v4_predec`
  (`:490`) + `predecode_q3_k_scale_table` (`quant/mod.rs:643`) hoist the decode out
  — but the bench reads the **full 110-B block + a 64-B f32 sidecar = 174 B/block**,
  *more* bytes than the 144-B Q4_K weights it's trying to undercut. Anti-byte-cut.

**A8's recorded verdict:** byte-cut SPEED is **Type-2** dead — "the fastest Q3_K
is 22–43% slower than Q4_K-predec," named oracle = "re-run this bench after a
Q3_K GEMV is rewritten to the Q4_K-predec standard." dead_levers.md:184 records
the live BW levers as "scale-byte volume (A6.5 f16-scales, shipped) and lower
weight precision (A8 Q3_K, footprint-only until a Q3_K predec/2r rewrite)."

**This doc is that rewrite's design.** The A8 bench tested f32-predec (fast
decode, +bytes) and inline-2r (fewest bytes, slow decode) — but **never the
diagonal cell**: fast decode (predec) AND fewest bytes (f16 scales + drop the now-
dead packed-scale bytes via a repack). That cell is f16-predec-Q3 on a 128-B
block.

---

## 2. The two candidate formats

### 2.1 — `f16-predec-Q3` (RECOMMENDED)

Mirror the A6.5 `f16s` win exactly, applied to Q3_K, **with the block repacked**
so the predec is a true byte-cut rather than a sidecar tax.

Two layout variants — the byte budget hinges entirely on which:

| variant | weight bytes read | scale bytes read | total/block | notes |
|---|---|---|---|---|
| **(a) in-place + f16 sidecar** | 110 (full Q3 block, incl. dead 12-B packed scales + 2-B d) | 32 (16 f16) | **142** | trivial to build (just `predecode_q3_k_scale_table_f16`); but reads 14 dead B/block |
| **(b) REPACKED + f16 inline** | 96 (hmask 32 + qs 64; drop packed scales + d) | 32 (16 f16) | **128** | the byte-cut-true layout; needs a repack pass like `q4k_fast` |

**Variant (b) is the recommendation.** The 12 packed-6-bit scale bytes
(`bytes[96..108]`) and the 2-byte super-block `d` (`bytes[108..110]`) are
**fully redundant once the scales are pre-decoded** — `predecode_q3_k_scale_table`
already folds `d * scale[i]` into the table, so the kernel never re-reads either.
Leaving them in the block (variant a) means streaming 14 dead B/block across the
bus on a bandwidth-bound kernel — directly anti-thetical to the byte-cut. The
repack drops them, giving the clean 128 B = 32 (hmask) + 64 (qs) + 32 (f16
scales).

(The Q4_K f16s kernels did NOT need a repack because Q4_K's packed scales sit in
the same 12 header bytes that the f32/f16 sidecar duplicates — but Q4_K kept the
in-place block and just added the sidecar, paying the same dead-byte tax: that's
why A6.5's `_pair_f16s` reads 144+32=176, not a repacked 128+. Q3 has the chance
to do better *because* we're designing the layout fresh, not retrofitting.)

**Hot-loop decode after predec** (per element, per the existing `gemm_q3_k_v4_predec`
inner loop `:513-531`, which already does this for the f32 table):
```
q = ((qs[q_idx] >> shift) & 0x3) - (hmask[h_idx] & high_mask ? 0 : 4);   // 2-bit + high-bit
dl = scales[sub];                                                         // ONE f16 read (predec'd d*scale)
partial += dl * (float)q * xv;                                            // FMA
```
No `q3_k_scale` call, no branch on `scale_idx`, no `-32`, no `d` multiply. The
entire 6-bit unpack is gone. What remains is the 2-bit dequant + the hmask
high-bit select + the scale read + the FMA.

### 2.2 — QTIP-class lookup-free trellis (CONTRAST / DEFER)

The bible's surviving deep-byte-cut codec (§2, §8.1 L1.5 resurrection): a
near-Gaussian-optimal sub-4-bit code whose decode is **arithmetic on contiguous
bits** (a bitshift trellis / "3INST" or "1MAD" computed-code), *no* LUT read —
hence Apple-GPU-friendly (no hardware gather needed). Targets ~3.0–3.25 bits.

**Why it's the right contrast:** it attacks the same goal (fewer bytes,
gather-free) by a *different* mechanism — instead of pre-decoding a tabular scale,
it makes the *code itself* cheap to decompute. It is the codec the Kill Protocol
already lists as the live byte-cut survivor (dead_levers.md:101, bible §8.3.1).

**Why DEFER it under the present (local, no-Colab, no-commit) constraints:**
1. **No cheap in-tree oracle.** QTIP needs a quantizer (incoherence
   pre-processing / a trellis fit) that lives on Colab (bible §5). Its quality
   gate is PPL/KL on a code corpus vs Q4_K_M — a Colab afternoon, not an M3
   microbench. We cannot stand it up read-only/local.
2. **Its decode-compute is a question mark, and A8 just taught us decode-compute
   is the binding constraint for sub-4-bit on this GPU.** A trellis decode spends
   *more* ALU per element than f16-predec-Q3 (it computes the code, vs reading a
   pre-decoded scale). The whole A8 lesson is that Q3's extra per-element ALU is
   what killed it; a trellis adds ALU in the same place. It could easily land
   compute-bound for the same reason — and we'd only find that out after the
   Colab quant + a new kernel. High effort, same risk.
3. **f16-predec-Q3 reuses shipped, trusted machinery** (the A6.5 f16s kernels +
   `predecode_*_scale_table_f16` + the predec parity/bench harnesses). QTIP is
   all-new.

**Recommendation:** build f16-predec-Q3 first (cheap, reuses A6.5, settles the
"is *any* Q3-class format speed-viable" question with one microbench). Only if it
**passes** the §6 gate does QTIP become worth the Colab investment (it would then
inherit a proven-BW-bound Q3-class kernel pattern to target — exactly P4's
"chained on a speed-viable Q3" framing in the queue). If f16-predec-Q3 *fails*
the gate (residual hmask-gather ALU keeps Q3 compute-bound even at 128 B), that is
strong evidence QTIP also fails for the same reason — a near-free kill of P4 by
proxy.

---

## 3. Byte-budget analysis (the numerator the bus must stream)

All numbers are **bytes per 256-element super-block**, the unit the decode GEMV
streams. Qwen2.5-3B dense decode reads ~1.9 GB of Q4_K_M weights/token; the
byte-cut acts proportionally on the dense per-token cost (bible axis 2).

| format | weight B | scale B | **total B/block** | vs Q4_K-predec-f32 | vs Q4_K-predec-f16s (A6.5) |
|---|---|---|---|---|---|
| Q4_K-predec-f32 (`_pair`/`_2r`, default) | 144 | 64 (16×f32) | **208** | — | +18% |
| Q4_K-predec-f16s (A6.5, opt-in, shipped) | 144 | 32 (16×f16) | **176** | −15% | — |
| Q3_K fused (A8, inline decode) | 110 | 0 | **110** | −47% | −38% |
| Q3_K predec-f32 (A8 bench, in-place + sidecar) | 110 | 64 (16×f32) | **174** | −16% | −1% |
| **Q3_K predec-f16, in-place (variant a)** | 110 | 32 (16×f16) | **142** | **−32%** | **−19%** |
| **Q3_K predec-f16, REPACKED 128 (variant b) ⭐** | 96 | 32 (16×f16) | **128** | **−38%** | **−27%** |

Plus the shape-constant `x` (cols×4) and `y` (rows×4), identical across all
kernels, so they wash out of any *relative* comparison.

> **Reconciling two byte accountings in the tree (read this or the numbers look
> contradictory).** This table uses the **full physical block + sidecar**
> accounting, matching `q3k_bytecut_bench.rs:204-205` (Q4_K-predec-f32 = 144 + 64
> = 208; Q3-predec-f32 = 110 + 64 = 174). The A6.5 f16s kernel comments
> (`quant.metal:2079,2220`) instead quote **"192→160 (−17%)"** — that is the
> *useful-bytes-read* accounting: it counts only the 128-B qs nibble plane Q4_K
> predec actually consumes + 64 f32 scales = 192, *dropping the 16 header/packed-
> scale bytes predec makes redundant but the kernel still streams.* Both are
> internally consistent; they differ only in whether the redundant header bytes
> are charged. **This distinction is the whole point of variant (b):** the f16s
> comment's "192" already concedes Q4_K predec streams 16 dead bytes it doesn't
> use — Q3 variant (b) does the analogous drop *explicitly* (repack to 96 useful
> weight B), so on the useful-bytes accounting Q3-predec-f16 is 96 + 32 = **128**
> vs Q4_K-f16s's 160 (still −20%), and on the full-block accounting it's 128 vs
> 176 (−27%). Either way the byte-cut holds; the table above uses the full-block
> form throughout for one consistent comparison.

**Key reads:**
- The recommended **128 B/block is −27% vs the A6.5 f16s `_pair` kernel that is
  46.6% of decode** and −38% vs the f32 default. This is a *larger* byte-cut than
  the naive Q3-vs-Q4 weight ratio (110/144 = −24%), because predec lets Q3 shed
  its packed-scale + d bytes that the fused kernel must keep.
- The **f16 scale table is half the f32 table** — the same A6.5 lever, and on Q3
  it matters *more* in relative terms: scales are 32/128 = 25% of the repacked
  block (vs 32/176 = 18% on Q4_K-f16s), because Q3's weight payload is smaller, so
  the scale bytes are a bigger slice. Halving them (f32→f16) is worth ~ the same
  −18% on the scale-table traffic that A6.5 banked.
- **The fused Q3 (110 B) is the fewest bytes of all** — but it's the one A8
  proved compute-bound (it pays the full inline 6-bit decode). The design bet is
  that 128 B with *no* inline scale decode beats 110 B *with* it, because the
  removed compute matters more than the 18 extra bytes when you're compute-bound.

**Footprint note (separate from speed):** on-disk / in-RAM, Q3_K_M is ~−11%
model size vs Q4_K_M (A8). The repacked 128-B variant is a *runtime* layout (like
the q4k_fast sidecar) — it does not change the on-disk Q3_K_M size; it's built at
load. So footprint stays the already-banked −11%; this doc is purely about the
*speed* of streaming it.

---

## 4. Decode-cost analysis (why the bottleneck plausibly flips)

The byte-cut only converts to a speed win **if the kernel is BW-bound** (bible
§2: byte-cut "raises the tps wall proportionally — no % ceiling"). A8 proved the
*current* Q3 kernels are not. The question is whether f16-predec-Q3 *is*.

### 4.1 — What predec removes (the measured bottleneck)

A8/Lane-2 isolated the cost to the **inline 6-bit scale decode** called per
element. Predec moves all of it to load time:

| per-element work | Q3 fused (A8, slow) | **Q3 f16-predec (proposed)** |
|---|---|---|
| `scale_idx < 8` branch | yes | **gone** (table built once) |
| low-nibble select from `bytes[96+]` | yes | **gone** |
| high-2-bit shift/mask from `bytes[104+]` | yes | **gone** |
| OR + `-32` | yes | **gone** |
| `d * scale` multiply | yes (per element) | **gone** (folded into f16 table) |
| 2-bit dequant `(qs >> shift) & 3` | yes | yes |
| hmask high-bit select `… ? 0 : 4` | yes | yes |
| scale read | (computed) | **1 f16 load + widen** |
| FMA | yes | yes |

This is precisely the transform that made Q4_K-predec the kernel "at the
memory-model optimum" — A4 shows the predec GEMVs at ~56% of peak BW, i.e.
genuinely BW-bound, whereas the pre-predec inline-decode Q4_K was compute-bound
(the +21% ffn_down→predec win, bible §0). **Predec did for Q4_K exactly what this
proposes for Q3_K.** The mechanism is proven on the sibling format.

### 4.2 — The honest residual: Q3's hot loop is NOT as cheap as Q4_K's even after predec

This is the crux the design must not gloss. After predec, compare the *surviving*
per-element hot loops:

**Q4_K-predec hot loop** (`gemm_q4_k_v4_predec_2r` `:2198-2205`), per nibble-pair:
- 1 byte load (`qb`), 2 nibble extracts (`& 0xF`, `>> 4`)
- 2 FMAs of form `(ds[k]*nib - dm[k]) * xl[k]` — note the **sub-block-min subtract
  `- dm[k]`** is Q4_K-specific work Q3 does NOT have (Q3 is symmetric, no min).

**Q3_K-predec hot loop** (`gemm_q3_k_v4_predec` `:513-531`), per element:
- index arithmetic: `half_idx, local, group16, j, second, lane, q_idx, h_idx,
  shift, high_mask` (≈8 shifts/masks) — **more index math than Q4_K's `pi`/`k0`/`k1`**
- 1 `qs` byte load + shift + `& 3`
- **1 `hmask` byte load + `& high_mask` + a select `? 0 : 4`** ← the Q3-only cost
- 1 f16 scale load + widen
- 1 multiply `dl * q` + 1 FMA `* xv`

So Q3-predec trades Q4_K's `-dm` subtract (cheap) for: (i) more per-element index
arithmetic, and (ii) a **second device byte load (`hmask`) plus a branch/select**
per element. The `hmask` high-bit is genuinely extra memory traffic *and* ALU that
Q4_K has no counterpart for.

**Two ways this resolves, and why the microbench is decisive:**
- **Optimistic (byte-cut wins):** the index math is all cheap integer ops the
  M3's ALU eats for free relative to memory, the `hmask` load is coalesced
  (32 contiguous bytes/simdgroup, same as qs — see A5's finding that Q3's loads
  are already simdgroup-coalesced), and once the *expensive branchy 6-bit decode*
  is gone the loop becomes memory-bound on the 128 B. Then µs/call ≈ Q4_K-predec ×
  (128/176) ≈ **0.73×** → ~27% faster, and the byte-cut converts.
- **Pessimistic (byte-cut still loses):** the per-element `hmask` select + the
  heavier index arithmetic keep the loop compute-bound even without the 6-bit
  decode — Q3 just has structurally more per-element work than Q4_K, and predec
  removes the *worst* part but not enough. Then µs/call stays > Q4_K-predec despite
  fewer bytes, and **f16-predec-Q3 dies the same Type-1-ish death** (Q3's
  per-element compute is irreducible without a different bit-packing).

A8 measured that the 6-bit *scale* decode dominated; it did **not** report the
counterfactual "Q3 decode cost with the scale decode removed but the hmask gather
kept." The f32-predec kernel A8 *did* run (`gemm_q3_k_v4_predec`) is the closest
data point — but A8 reported it only as "competitive [among Q3 kernels] but adds
64 B/block," i.e. it was judged on bytes (174 > 144), not isolated on decode-cost
at matched bytes. **Re-checking the f32-predec kernel's GB/s in the A8 bench
output is the cheapest possible pre-build sanity read** (see §6.0): if
`gemm_q3_k_v4_predec` already shows ~50% of peak GB/s, the bottleneck flip is
real and f16-predec on a 128-B repack is near-certain to win; if it's still
stuck at ~20–30 GB/s, the hmask/index residual is the wall and this lever is in
trouble *before* writing the repack.

---

## 5. Build plan (mirror the shipped f16s machinery)

All steps are behind an opt-in flag, default-off, single-purpose, gated. Mirrors
the A6.5 build exactly (which is the trusted template). **Nothing here is to be
built in this session — this is the plan for an attended build session.**

### 5.1 — Table builder (Rust) — `predecode_q3_k_scale_table_f16`
- **File:** `crates/dismantle-core/src/quant/mod.rs` (next to
  `predecode_q3_k_scale_table:643`).
- **Body:** literally the A6.5 one-liner pattern
  (`kernels/mod.rs:1086-1091`): map the existing f32 builder through
  `half::f16::from_f32`. Returns `Vec<f16>`, 16 halfs/block.
  ```rust
  pub fn predecode_q3_k_scale_table_f16(bytes: &[u8]) -> Vec<half::f16> {
      predecode_q3_k_scale_table(bytes).into_iter().map(half::f16::from_f32).collect()
  }
  ```
- Trivial; no new decode logic (reuses the validated f32 path).

### 5.2 — Repack builder (Rust) — `repack_q3_k_predec_128` (variant b)
- **File:** `crates/dismantle-core/src/quant/mod.rs` or `kernels/mod.rs`
  (alongside the q4k_fast repack, whichever houses that).
- **Output:** a `Vec<u8>` at **96 B/block** = `hmask[32] ++ qs[64]` (copy
  `bytes[off..off+32]` then `bytes[off+32..off+96]`; drop `[96..110]`). The f16
  scale table (§5.1) is the parallel sidecar. *Two* buffers, like predec today —
  NOT interleaved — so the kernel's two reads (weights, scales) stay independent
  and coalesced.
- **Why a repack at all:** §3 — without it the kernel streams the 14 dead
  scale+d bytes (variant a, 142 B). The repack is what gets to 128 B and the −27%.
- **Cost:** O(weights) at load, once, same as q4k_fast. RSS: the 96-B repacked
  buffer replaces nothing (the original Q3 mmap can be dropped post-repack if the
  dense path serves only the repacked form, like q4k_fast's sidecar) — watch the
  RSS sentinel (CLAUDE.md: >5 GB halts; q4k_fast was ~1.5 GB sidecar). A 96-B
  repack of a ~1.7 GB Q3 model is ~1.5 GB; **if both the original mmap and the
  repack are resident, that risks the ceiling** — the repack must replace, not
  augment, the served buffer (design note for the build session).

### 5.3 — Kernels (Metal) — two, mirroring `_2r_f16s` and `_pair_f16s`
- **File:** `crates/dismantle-core/shaders/quant.metal`.
- **`gemm_q3_k_v4_predec_2r_f16s`** — clone `gemm_q3_k_v4_predec` (`:490`, the f32
  predec Q3) and apply BOTH changes from the Q4_K `_2r_f16s` clone (`:2226`):
  1. `device const half* scales` instead of `float*`, widen with `(float)` at the
     load (exactly `:2257-2260`).
  2. 2-row ILP: two accumulator chains `p0/p1`, shared `x` load, 16 rows/TG —
     copy the `_2r` row-pairing structure (`:2164-2214`). **But** A8 showed 2r
     *regressed* wide-FFN Q3; include it as a *variant to A/B*, not a foregone
     default. The 2r question for f16-predec-Q3 is genuinely re-opened because the
     bottleneck (post-predec) is different from the inline-decode bottleneck 2r
     was tested against. Bench both 1-row and 2r.
  3. **Weight offsets for the 96-B repack:** the inner loop reads `hmask[h_idx]`
     at `bo + h_idx` and `qs[q_idx]` at `bo + 32 + q_idx` with block stride **96**
     (not 110). This is the one substantive divergence from the existing predec
     kernel, which uses stride 110 and reads the dead scale region.
- **`gemm_q3_k_v4_predec_pair_f16s`** — IF the dense Q3 path fuses any GEMVs the
  way Q4_K fuses gate+up. *Open design question:* a Q3_K-quantized Qwen would have
  its FFN gate+up as Q3 — so a Q3 `_pair` is the analog of the 46.6%-dominant
  Q4_K `_pair`, and is where most of the win lives. Clone `_pair_f16s` (`:2085`)
  with the same three changes. **This is the higher-priority kernel** (the `_pair`
  shape is the decode wall); the `_2r` covers q/o/ffn_down.
- Register both names in `metal/mod.rs` (the match arm at `:442-453`, alongside
  `gemm_q3_k_v4_predec`).

### 5.4 — Wrappers (Rust) — `gemv_q3_k_v4_predec_2r_f16s_pinned_tcb` (+ `_pair_f16s`)
- **File:** `crates/dismantle-core/src/kernels/mod.rs`.
- Clone `gemv_q3_k_v4_predec_pinned_tcb` (`:1321`) for the 2r, and
  `gemv_q4_k_v4_predec_pair_f16s_pinned_tcb` (`:1499`) for the pair. Changes vs the
  Q3 f32 wrapper:
  1. `scales_buf` validated as `… * 16 * sizeof(half)` (mirror `:1529-1533`), not
     `sizeof(f32)`.
  2. **`expected_bytes = rows * blocks_per_row * 96`** (the repack stride), not
     `* 110` (`:1342`). This is the byte-cut's load-bearing constant — the OOB/size
     guards must use 96 or they reject the repacked buffer.
  3. Grid/TG: 2r = `(ceil(rows/16)*256,1,1)/(256,1,1)`; pair = `ceil(rows/8)`.
- Opt-in env flag **`DISMANTLE_QWEN_Q3_PREDEC_F16SCALES`** (own flag, parallel to
  the Q4_K `DISMANTLE_QWEN_PREDEC_F16SCALES`), default-off, `OnceLock` pattern
  (`:1159-1166`).

### 5.5 — Dense-path wiring (the part A8 deferred as "too big unattended")
- A *full* Q3_K dense path (a Q3-quantized Qwen served end-to-end) is the large
  piece. The microbench gate (§6) does **not** require it — it validates the GEMV
  in isolation, which is the speed-premise question. **Wire the dense path only
  after the microbench passes** (don't build the big thing to discover the kernel
  loses). When wired, it mirrors the Q4_K predec dense path: build repack + f16
  table at load, route Q3 tensors through the new wrappers under the flag.

---

## 6. Gates (the two that decide GO/NO-GO)

### 6.0 — Free pre-build sanity read (do this FIRST, no new code)
Re-run the **existing** `q3k_bytecut_bench` (`--ignored --nocapture`) and read the
**`gemm_q3_k_v4_predec` (f32 predec) GB/s** line it already prints (the bench
dispatches it as "Q3_K predec"). This is the closest existing proxy for "Q3 decode
with the 6-bit scale decode hoisted out":
- If f32-predec-Q3 is **≳45–50% of peak** (~70+ GB/s) → the bottleneck already
  flipped to BW with predec; f16-predec on 128 B is near-certain to beat
  Q4_K-predec. **Strong GO signal; proceed to build.**
- If f32-predec-Q3 is still **~20–35 GB/s** → the hmask/index residual (§4.2) is
  the wall, predec alone didn't flip it, and fewer scale bytes won't save it.
  **Strong NO-GO signal; likely Type-1 (Q3 per-element compute irreducible) —
  record the kill, don't build, and by-proxy down-weight QTIP/P4.**

This costs one bench run (GPU, ~1 min) and gates the whole build cheaply — exactly
the discipline (oracle-before-kernel) the bible/CLAUDE.md mandate.

### 6.1 — Parity gate (correctness)
- **Test:** `crates/dismantle-core/tests/q3k_predec_f16s_parity.rs` (new), mirror
  `q4k_predec_pair_f16s_parity.rs`.
- **Reference:** `gemm_q3_k_fused_v2` (the inline-decode Q3, `:349`) — the
  canonical Q3 math. (Or `gemm_q3_k_v4_predec` f32; both are within fp16 tol of
  fused.)
- **Metric & bar:** **predec is NOT bit-identical** to fused — the table stores a
  pre-rounded `d*scale` and the compiler FMA-recontracts the inline form
  differently (documented at `quant.metal:483-487` and the q3k_predec_parity
  header: measured ~1 ULP / ~1e-4). f16 scales add ~5e-4 relative rounding on top.
  Gate at **atol 1e-3 fp16** (CLAUDE.md verification rule: "atol=1e-3 fp16 …
  tighter not required, looser forbidden"). For the whole-vector check, use the
  **relative-L2** form the f16s pair test uses (`rel_l2`, bar `< 1e-2`,
  `q4k_predec_pair_f16s_parity.rs:73,186`) since cancellation makes per-element
  atol noisy on long reductions — report both per-element max-abs and rel-L2.
- This is a **quality gate, not the greedy bit-identical gate** — f16-predec is a
  lossy-but-bounded lever, same class as A6.5. (The eventual *token-output* drift
  is a separate concern — see §7.)

### 6.2 — Microbench gate (the speed premise — the decisive one)
- **Test:** extend `q3k_bytecut_bench.rs` with the new kernel(s) as a 5th/6th
  bench line, on the same 3 Qwen shapes (attn-square 2048×2048, ffn-up 11008×2048,
  ffn-down 2048×11008), same `WARMUP=30`/`ITERS=200`, same TCB-commit-and-wait
  round-trip = one decode-path dispatch cost.
- **Verdict it must report** (the bench already computes verdict (b)/(c) — add the
  f16-predec line into the `q3_winner`/`bytecut` comparison):
  - **PASS:** fastest f16-predec-Q3 (1r or 2r) **beats Q4_K-predec µs/call** on
    the dominant shapes (the `_pair`-shaped ffn-up/down especially). The byte-cut
    converts: Q3 is now both fewer bytes AND faster.
  - **NO-GO:** f16-predec-Q3 is still slower than Q4_K-predec → the byte-cut
    remains footprint-only; the hmask/index residual is the wall (record Type-1 if
    GB/s confirms still-compute-bound, i.e. the 128-B kernel runs well under peak).
- **Also report the achieved GB/s** at 128 B/block — that's the BW-bound check
  (bible §1 invariant 1: GB/s ≤ 150). If the new kernel runs ~55%+ of peak like
  Q4_K-predec, the premise held; if it's still ~30%, it's still compute-bound and
  the byte-cut is moot regardless of the µs comparison.

### 6.3 — (Deferred) token-output / PPL gate — the Q3-vs-Q4 quality trade
- **Separate from this doc's speed question, and genuinely deferred.** Q3_K_M is
  ~3.4 effective bits vs Q4_K_M's ~4.5 — a real quality drop, independent of the
  kernel. Whether a Q3-quantized Qwen2.5-3B is *good enough* is a **PPL/KL on a
  code corpus** question (bible axis-2 oracle, Colab/local quant of a Q3 model +
  eval vs Q4_K_M), not a GEMV microbench. The A6.5 drift sweep is the cautionary
  precedent: f16-*scales* alone drifted 8.2% on open-ended gen (constrained 0%) —
  and Q3 *weights* are a far larger perturbation than f16 scales. **Do not flip a
  Q3 dense path default-on without that corpus quality eval.** This doc green-lights
  only the *kernel* (is the byte-cut fast?); the *should-we-serve-Q3* decision is a
  downstream, attended, quality-gated call.

---

## 7. Kill Protocol classification (CLAUDE.md / bible §8.3.1)

This doc reframes the **A8 "byte-cut SPEED" Type-2 kill**. Recording per the
protocol:

- **Type:** the A8 kill was **Type-2** ("died in the form tested" — inline-decode
  and f32-predec — "where a different formulation attacks the same goal"). The
  named reframe A8 itself recorded was "a Q3_K GEMV rewritten to the
  Q4_K-predec standard." **f16-predec-Q3 on a 128-B repack IS that reframe**,
  made concrete: cheap-decode (predec, the Q4_K-predec standard) + fewest-bytes
  (f16 scales + repack-drop the dead scale bytes).
- **The reframe is ALIVE** because it has a **named, cheap oracle**: the §6.0 free
  pre-build read (re-run the existing bench, read f32-predec-Q3's GB/s) and the
  §6.2 microbench (re-run with the new kernel). Both are M3-GPU minutes, no Colab.
- **What would make it Type-1 (dead for real):** if §6.0/§6.2 show f16-predec-Q3
  still runs well below peak BW (~30%) — i.e. predec removed the 6-bit decode but
  the *residual* per-element hmask-gather + index arithmetic keep Q3 compute-bound.
  Then Q3's per-element decode cost is an irreducible property of the
  2-bit+hmask+packed-6-scale layout (not fixable by *this* codec's cleverness), and
  the lever is Type-1 dead **for the f16-predec form** — the next reframe would be
  QTIP (a *different bit-packing*, §2.2), which the protocol already lists as the
  surviving codec.
- **No resurrection on vibes:** if the microbench says NO-GO, it stays dead unless
  a *new* named oracle appears (e.g., a QTIP trellis PPL+kernel result).

---

## 8. Honest bottom line

**The byte/decode math says f16-predec-Q3 has a genuine, large byte-cut (128 B vs
176 B = −27% vs the dominant A6.5 kernel) and a sound mechanism for flipping Q3
from compute-bound to BW-bound (predec removes the exact cost A8 measured as
dominant — the same transform that made Q4_K-predec optimal).** That makes it the
correct next thing to *try*, and a true Type-2 reframe of the A8 kill with a
cheap on-GPU oracle.

**But it is NOT a foregone win, and this doc refuses to pretend it is.** Q3's hot
loop has irreducible per-element work Q4_K lacks — the hmask high-bit gather and
heavier index arithmetic. A8 proved the scale-decode dominated; it did not prove
the *rest* of Q3's decode is cheap. If that residual is itself enough to keep the
128-B kernel compute-bound, the byte-cut stays footprint-only and the lever dies
Type-1 for this form (→ QTIP becomes the only live byte-cut, on Colab).

**The cheapest possible next action is the §6.0 free read** — re-run the existing
`q3k_bytecut_bench`, look at the f32-predec-Q3 GB/s already in its output. That one
number (is it ~50% of peak, or still ~30%?) forecasts the entire build's outcome
before a line of code is written. Do that first; build §5 only if it's a GO
signal; QTIP only if §5 passes (or is settled NO-GO, which also informs QTIP).

---

### Appendix — file/line index (for the build session)
- Q3_K decode helpers: `shaders/quant.metal:20-57` (`q3_k_fp16_at`, `q3_k_scale`,
  `q3_k_value`).
- Q3_K kernels: `gemm_q3_k_fused_v2:349`, `gemm_q3_k_fused_2r:397`,
  `gemm_q3_k_v4_predec:490`.
- Q3_K block layout (110 B): `quant/mod.rs:566-617` (`dequant_q3_k_into`) —
  hmask `[0..32]`, qs `[32..96]`, packed-6bit scales `[96..108]`, d `[108..110]`.
- Q3_K f32 table builder: `quant/mod.rs:643` (`predecode_q3_k_scale_table`).
- Q4_K f16s template — kernels: `gemm_q4_k_v4_predec_2r_f16s:2226`,
  `gemm_q4_k_v4_predec_pair_f16s:2085`.
- Q4_K f16s template — builders/wrappers: `predecode_q4_k_scale_table_f16`
  `kernels/mod.rs:1086`; `gemv_q4_k_v4_predec_pair_f16s_pinned_tcb:1499`;
  `gemv_q3_k_v4_predec_pinned_tcb:1321` (the Q3 wrapper to clone).
- Kernel-name registry: `metal/mod.rs:442-453`.
- Parity-gate template: `tests/q4k_predec_pair_f16s_parity.rs` (`rel_l2:73`,
  bars `:186/:190`); Q3 atol template: `tests/q3k_predec_parity.rs`.
- Microbench to extend: `tests/q3k_bytecut_bench.rs` (verdicts (a)/(b)/(c)
  `:230-259`).
- A4 profile (the `_pair` 46.6% @ 84 GB/s = 56% peak fact):
  `reports/a4_per_kernel_decode_profile_2026_05_31.md:23-38`.
- A8 kill record: `plans/overnight_build_queue_2026_05_31.md:85-99,155,198`;
  `reports/dead_levers.md:184`.
