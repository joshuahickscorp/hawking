# G1 entropy coding — Qwen3.8 quantized index streams

Lane: `12-entropy-coding`. Write scope: this file only.
No GPU, no pack, no artifact mutation, no resident-process contact.

**STATUS: MEASURED_NEGATIVE** as a G1 lever (index-entropy of the incumbent
uniform-Qn family cannot carry complete BPW from 4.25 to 1.5, and the
random-access-legal codecs recover little of the 0.52-bit gap at group-64).
The gap itself is real and measured.

Numbers below are tagged MEASURED, CROSS-CHECKED, ESTIMATED, or PROJECTED.
No complete-token or Metal-timed claim is made.

---

## 1. What was measured

**Quantizer (G0 production, generalized to other widths).**
`pack_uniform_q4_group64` in
`crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:233-248`:

```
scale = f16(max_abs / 7.0)
q     = rint_ties_even(x / scale).clamp(-8, 7)     # nibble = q+8
```

Generalized: `bound = 2^{n-1}-1`, `scale = f16(amax/bound)`,
`q ∈ [-2^{n-1}, bound]`. Group size 64, matching `UNIFORM_Q4_GROUP_SIZE`.

**CROSS-CHECKED.** Layer-0 `mlp.gate_proj` requant vs sealed HQ30UQ4 payload:
element match **1.0**, disagree **0**, H identical **3.4952326949731978**.
Command: python3 one-shot against
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`
and the BF16 source. The `-8` bin is unused on this tensor (amax/7 never
quantizes to -8 after f16 scale).

HGRAVU01 (`lab/operators/ascension_dual_gravity_worker.py:673-676`) clips
`[-bound, bound]` (no extra negative code). Used only as a packed-Qn
cross-check, not as the G0 definition.

**Artifacts (read-only).**

| artifact | path | role |
|---|---|---|
| BF16 source | `.../qwen38-27b/bf16` | 11 shards, language + vision |
| G0 uniform Q4 | `.../qwen38-27b/uniform-q4-v1` | 402 HQ30UQ4 GEMV + 353 f32 |
| mixed-q3mlp | `.../qwen38-27b/mixed-q3mlp-v1` | HGRAVU01 q3 g64 on all MLP |
| mixed-floor-q7/q8 | `.../qwen38-27b/mixed-floor-q7-v1`, `mixed-floor-q8-v1` | HGRAVU01 q7/q8 on non-MLP |

**Harness.** `/tmp/g1_entropy_measure.py` → `/tmp/g1_entropy_measure.json`
(schema `hawking.g1.entropy_coding.measure.v1`). Wall 268.17 s. Peak
working set one tensor at a time (embed/lm_head capped at 33 554 432
elements for the multi-width sweep; Q4 production histograms are complete).

Q4 production covers **every** language GEMV HQ30UQ4 file, 26 893 352 960
codes. That is the G0 index stream.

---

## 2. Tensor classes

Authority: `crates/hawking-core/src/model/qwen38_geometry.rs` and the sealed
Q4 manifest (`schema hawking.ascent.qwen38_language_uniform_q4.v1`).

Vision (`vision_tower.*`, 333 tensors) is not in the language pack.
Small vectors are stored f32 in G0 and have **no quantized index stream**.

### 2.1 GEMV classes that own a Q4 index stream (MEASURED census)

From `uniform-q4-v1/manifest.json` (755 catalog rows; 402 q4 + 353 f32).
`complete_physical_bpw` = **4.252735126866492**. `nominal_codec_bpw` = 4.25.
`source_weight_elements` = 26 895 998 464.
This independently confirms the G0 “~4.2527 complete BPW” *payload* claim
for this artifact. It is not a TPS or TOKEN_NS confirmation.

| class | tensors | elements | payload bytes | payload bpw |
|---|---:|---:|---:|---:|
| mlp.gate_proj | 64 | 5 704 253 440 | 3 030 387 200 | 4.2500 |
| mlp.up_proj | 64 | 5 704 253 440 | 3 030 387 200 | 4.2500 |
| mlp.down_proj | 64 | 5 704 253 440 | 3 030 387 200 | 4.2500 |
| linear_attn.in_proj_qkvz (fused) | 48 | 4 026 531 840 | 2 139 096 960 | 4.2500 |
| linear_attn.out_proj | 48 | 1 509 949 440 | 802 162 560 | 4.2500 |
| embed | 1 | 1 271 398 400 | 675 430 440 | 4.2500 |
| lm_head | 1 | 1 271 398 400 | 675 430 440 | 4.2500 |
| self_attn.q_proj | 16 | 1 006 632 960 | 534 774 400 | 4.2500 |
| self_attn.o_proj | 16 | 503 316 480 | 267 387 520 | 4.2501 |
| self_attn.k_proj | 16 | 83 886 080 | 44 565 120 | 4.2501 |
| self_attn.v_proj | 16 | 83 886 080 | 44 565 120 | 4.2501 |
| linear_attn.in_proj_ba (fused) | 48 | 23 592 960 | 12 535 680 | 4.2507 |
| **GEMV total** | **402** | **26 893 352 960** | | **4.2500** |

Fusion (`qwen38_geometry.rs:360-389` `fuse_in_proj_ba`): per key-head,
`b` rows then `a` rows. Concat `a||b` vs packed `ba` is a 64-aligned
permutation: **histograms identical, element order is not**. Entropy is
invariant. Split BF16 names `in_proj_{qkv,z,a,b}` were used for the
multi-width sweep; Q4 production uses the fused names.

### 2.2 Small f32 — no index stream (MEASURED)

353 tensors, 2 645 504 elements, 10 584 840 bytes, payload 32.01 bpw
(norms, `A_log`, `dt_bias`, `conv1d`, `q_norm`, `k_norm`). Mass fraction
2.6e6 / 2.690e10 = **9.8e-5**. Forcing Qn on them does not move complete
BPW. They are excluded from the entropy tables.

---

## 3. Empirical entropy of the G0 Q4 index stream

**MEASURED.** Order-0 Shannon H of the nibble stream `q = nibble-8` over
the entire language GEMV set. Source: `/tmp/g1_entropy_measure.json`
`q4_production.*`.

| class | elements | H (bits) | nominal | gap | mean group offset-width | mean group signed-width | H_group (8 192-group sample) | rice(zigzag q) k=2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embed | 1 271 398 400 | 3.4977 | 4 | 0.5023 | 4.000 | 4.000 | 3.298 | 3.661 |
| lm_head | 1 271 398 400 | 3.4708 | 4 | 0.5292 | 4.000 | 4.000 | 3.311 | 3.639 |
| mlp.gate_proj | 5 704 253 440 | 3.4846 | 4 | 0.5154 | 4.000 | 4.000 | 3.301 | 3.650 |
| mlp.up_proj | 5 704 253 440 | 3.4828 | 4 | 0.5172 | 4.000 | 4.000 | 3.309 | 3.648 |
| mlp.down_proj | 5 704 253 440 | 3.4783 | 4 | 0.5217 | 4.000 | 4.000 | 3.306 | 3.644 |
| linear_attn.in_proj_qkvz | 4 026 531 840 | 3.4732 | 4 | 0.5268 | 4.000 | 4.000 | 3.301 | 3.640 |
| linear_attn.in_proj_ba | 23 592 960 | 3.3384 | 4 | 0.6616 | 4.000 | 4.000 | 3.299 | 3.485 |
| linear_attn.out_proj | 1 509 949 440 | 3.4739 | 4 | 0.5261 | 4.000 | 4.000 | 3.277 | 3.640 |
| self_attn.q_proj | 1 006 632 960 | 3.4688 | 4 | 0.5312 | 4.000 | 4.000 | 3.289 | 3.636 |
| self_attn.k_proj | 83 886 080 | 3.3959 | 4 | 0.6041 | 4.000 | 4.000 | 3.266 | 3.571 |
| self_attn.v_proj | 83 886 080 | 3.4171 | 4 | 0.5829 | 4.000 | 4.000 | 3.276 | 3.594 |
| self_attn.o_proj | 503 316 480 | 3.4644 | 4 | 0.5356 | 4.000 | 4.000 | 3.294 | 3.632 |
| **element-weighted** | **26 893 352 960** | **3.4789** | **4** | **0.5211** | **4.000** | **4.000** | | |

`in_proj_ba` and `k_proj`/`v_proj` sit ~0.08–0.14 bits below the MLP/embed
cluster. They are 0.7 % of GEMV mass. They do not move the weighted mean.

**Layer stability (MEASURED, `q4_layer_probe`).** `mlp.gate_proj` H:
L0 3.4952, L31 3.4752, L63 3.4720. `mlp.down_proj` L63 is the low outlier
at 3.4159. All GEMV classes stay inside 3.23–3.50. One-class, one-layer
H is a valid proxy for that class.

**Histogram shape (MEASURED, all 5.704e9 `gate_proj` codes, q = -8..+7):**

```
[0, 66732944, 78027323, 150732173, 270470194, 435425558, 629162787,
 791180465, 858999270, 791432056, 629600242, 435863661, 270864996,
 150972976, 78126641, 66662154]
```

Unimodal at 0, near-symmetric, **zero mass at -8**, 1.17 % at +7
(f16 scale rounding). This is a scaled-by-group-max distribution, not a
uniform 4-bit code and not a sparse spike-plus-tail.

**Mechanism.** Group-absmax forces at least one |x| in the group to map
near `bound`. `signed_width_hist` for `gate_proj` is `[0,0,0,0,89128960]`:
**every group of 64 uses a signed width of 4**. Offset-span width is 4
for 89 128 817 / 89 128 960 groups (143 groups width-3). Unused MSBs do
not exist at the group grain. The 0.52-bit gap is *intra-group shape*
(more mass near 0 than a uniform [-7,7]), not idle high bits.

**Scale overhead (not an index bit, but part of complete BPW).**
16-bit fp16 scale / 64 = 0.250 bpw. Manifest 4.2527 = 4.000 codes +
0.250 scale + 0.0027 header/padding.

Shannon-limit complete BPW if every GEMV index were coded to H and scales
were left raw:

```
(3.478937682977414 * 26893352960 + 0.25 * 26893352960 + 32 * 2645504)
  / 26895998464
= 3.732
```

MEASURED ceiling on *this* index stream: **0.521 bpw recoverable in the
codes**, complete 4.253 → 3.732, **1.75 GB** on 26.89 B weights. That is
the entire prize of this lane under the incumbent quantizer.

---

## 4. Same quantizer at 2, 3, 4, 5, 6, 8 bits

**MEASURED.** BF16 requant, G0 convention, sample layers
{0,3,7,15,16,31,32,47,48,63} for per-layer classes; 33 554 432 elements
of embed and of lm_head (first 6 553.6 rows). `bf16_sweep` in the JSON.

H / gap / mean group-offset-width:

| class | n used | H2 / gap / W | H3 / gap / W | H4 / gap / W | H5 / gap / W | H6 / gap / W | H8 / gap / W |
|---|---:|---|---|---|---|---|---|
| mlp.gate_proj | 891 289 600 | 0.913 / 1.087 / 1.98 | 2.309 / 0.691 / 3.00 | 3.484 / 0.516 / 4.00 | 4.566 / 0.434 / 5.00 | 5.599 / 0.401 / 6.00 | 7.605 / 0.395 / 8.00 |
| mlp.up_proj | 891 289 600 | 0.912 / 1.088 / 1.98 | 2.308 / 0.692 / 3.00 | 3.483 / 0.517 / 4.00 | 4.565 / 0.435 / 5.00 | 5.598 / 0.402 / 6.00 | 7.604 / 0.396 / 8.00 |
| mlp.down_proj | 891 289 600 | 0.901 / 1.099 / 1.98 | 2.299 / 0.701 / 3.00 | 3.473 / 0.527 / 4.00 | 4.554 / 0.446 / 5.00 | 5.587 / 0.413 / 6.00 | 7.594 / 0.406 / 8.00 |
| in_proj_qkv | 209 715 200 | 0.896 / 1.104 / 1.98 | 2.294 / 0.706 / 3.00 | 3.469 / 0.531 / 4.00 | 4.550 / 0.450 / 5.00 | 5.583 / 0.417 / 6.00 | 7.589 / 0.411 / 8.00 |
| in_proj_z | 125 829 120 | 0.904 / 1.096 / 1.98 | 2.302 / 0.698 / 3.00 | 3.477 / 0.523 / 4.00 | 4.558 / 0.442 / 5.00 | 5.591 / 0.409 / 6.00 | 7.598 / 0.402 / 8.00 |
| in_proj_a | 983 040 | 0.857 / 1.143 / 1.95 | 2.259 / 0.741 / 3.00 | 3.432 / 0.568 / 4.00 | 4.512 / 0.488 / 5.00 | 5.545 / 0.455 / 6.00 | 7.551 / 0.449 / 8.00 |
| in_proj_b | 983 040 | 0.767 / 1.233 / 1.90 | 2.167 / 0.833 / 3.00 | 3.335 / 0.665 / 4.00 | 4.413 / 0.587 / 5.00 | 5.446 / 0.554 / 6.00 | 7.451 / 0.549 / 8.00 |
| out_proj | 125 829 120 | 0.901 / 1.099 / 1.98 | 2.299 / 0.701 / 3.00 | 3.473 / 0.527 / 4.00 | 4.554 / 0.446 / 5.00 | 5.587 / 0.413 / 6.00 | 7.594 / 0.406 / 8.00 |
| q_proj | 377 487 360 | 0.895 / 1.105 / 1.98 | 2.294 / 0.706 / 3.00 | 3.469 / 0.531 / 4.00 | 4.550 / 0.450 / 5.00 | 5.583 / 0.417 / 6.00 | 7.589 / 0.411 / 8.00 |
| k_proj | 31 457 280 | 0.839 / 1.161 / 1.95 | 2.242 / 0.758 / 3.00 | 3.413 / 0.587 / 4.00 | 4.493 / 0.507 / 5.00 | 5.526 / 0.474 / 6.00 | 7.532 / 0.468 / 8.00 |
| v_proj | 31 457 280 | 0.869 / 1.131 / 1.96 | 2.268 / 0.732 / 3.00 | 3.442 / 0.558 / 4.00 | 4.522 / 0.478 / 5.00 | 5.555 / 0.445 / 6.00 | 7.561 / 0.439 / 8.00 |
| o_proj | 188 743 680 | 0.883 / 1.117 / 1.98 | 2.283 / 0.717 / 3.00 | 3.457 / 0.543 / 4.00 | 4.538 / 0.462 / 5.00 | 5.571 / 0.429 / 6.00 | 7.577 / 0.423 / 8.00 |
| embed (32 768 k) | 33 554 432 | 0.886 / 1.114 / 1.96 | 2.282 / 0.718 / 3.00 | 3.456 / 0.544 / 4.00 | 4.537 / 0.463 / 5.00 | 5.570 / 0.430 / 6.00 | 7.576 / 0.424 / 8.00 |
| lm_head (32 768 k) | 33 554 432 | 0.945 / 1.055 / 1.99 | 2.335 / 0.665 / 3.00 | 3.511 / 0.489 / 4.00 | 4.593 / 0.407 / 5.00 | 5.626 / 0.374 / 6.00 | 7.633 / 0.367 / 8.00 |

Sweep H4 on `gate_proj` = 3.4843 vs production-all-64-layers H4 = 3.4846.
The 10-layer sample is not a biased proxy.

**CROSS-CHECKED against already-packed HGRAVU01 (clip [-bound,bound]).**

| stream | bits | H | nominal | gap | source |
|---|---:|---:|---:|---:|---|
| mlp.down L2 | 3 | 2.3140 | 3 | 0.686 | `mixed-q3mlp-v1/.../layers_2_mlp_down_proj...hq38seg` |
| mlp.gate L0/L10/L11 | 3 | 2.3196 / 2.3169 / 2.3190 | 3 | 0.681 | same pack, three layers |
| mlp.up L0/L10/L11 | 3 | 2.3199 / 2.3160 / 2.3172 | 3 | 0.682 | same |
| mlp.down L0/L10/L11 | 3 | 2.3120 / 2.3127 / 2.3147 | 3 | 0.687 | same |
| in_proj_qkv L0 | 7 | 6.6162 | 7 | 0.384 | `mixed-floor-q7-v1/segments/L00.hq38seg` offset 4897 |
| in_proj_z L0 | 7 | 6.6031 | 7 | 0.397 | same segment |
| out_proj L0 | 7 | 6.5864 | 7 | 0.414 | same |
| in_proj_qkv L0 | 8 | 7.6138 | 8 | 0.386 | `mixed-floor-q8-v1/segments/L00.hq38seg` |
| out_proj L0 | 8 | 7.5839 | 8 | 0.416 | same |

Q3 packed H matches the BF16 sweep (2.31 vs 2.309). Q7/Q8 packed H matches
the sweep to ~0.01. Two independent payloads, two clip conventions, same
gap-vs-width curve: **gap shrinks as width grows** (≈1.10 at 2, 0.69 at 3,
0.52 at 4, 0.44 at 5, 0.40 at 6, 0.40 at 8) and **per-group width stays
equal to nominal width**.

---

## 5. Coding schemes vs random-access

Constraint (from the contract, and from the live Q4 kernel): a Metal
thread must address an arbitrary weight without decoding a prefix of the
tensor. The incumbent kernel already does O(1) per weight:

```22:36:crates/hawking-core/shaders/qwen_uniform_q4.metal
static inline float qwen_uniform_q4_value(...)
{
    const uint group = element / group_size;
    const uint local = element % group_size;
    const uint code_base = group * (group_size >> 1u);
    const uchar packed = codes[code_base + (local >> 1u)];
    const uchar nibble = (local & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u);
    const int q = int(nibble) - 8;
    return float(q) * float(scales[group]);
}
```

ESTIMATED ops/weight for that path: 1 shift-div, 1 byte load, 1 nibble
extract, 1 sub, 1 half load, 1 fmul ≈ **6 ALU + 2 loads**. Grain: weight.

Three RA grains:

- **W** — O(1) in the weight index. Only fixed-width packs.
- **G** — decode at most one group (64 / 256 / 1024). Compatible with the
  current group-64 matvec walk. Not compatible with a thread that wants
  one isolated weight inside the group cheaper than scanning the group.
- **SEQ** — must decode everything before the target. **UNUSABLE.**

Codec samples (`codec_samples` in the JSON): one tensor per class, first
2 097 152 weights (or the whole tensor if smaller), G0 quantizer.
`mlp.gate_proj` L0 is the reference row below; other GEMV classes sit
within ±0.04 bpw at 4-bit.

### 5.1 Fixed-width + escape

Alphabet is 15 used codes ([-7,+7]). Reserve one k-bit slot as ESC.

Correct inlier mass for k=3 (7 most frequent symbols, computed from the
production `gate_proj` hist, **not** from the script `escape` object —
that object used `np.argsort` on negated uint64 and is discarded):

```
p_in(top-7) = 4 571 664 039 / 5 704 253 440 = 0.80145
p_esc       = 0.19855
```

| variant | bpw (4-bit gate) | RA | vs nominal 4 |
|---|---:|---|---|
| k=3 + 4-bit extra on ESC (naive) | 3 + 0.19855×4 = **3.794** | SEQ if tightly packed | −0.206 |
| k=3 + 2-bit leftover on ESC | 3 + 0.19855×2 = **3.397** | SEQ if tightly packed | −0.603 |
| same leftover, leftovers stored in group order, main is fixed 3-bit (scan 64 mains to pair leftovers) | **3.397** + 3/64 count ≈ **3.444** | G | −0.556 |
| same leftover + explicit 6-bit local index per ESC | 3 + 0.19855×(6+2) = **4.588** | W | **+0.588 LOSS** |

At 4-bit the only RA-legal escape that beats 4 is the group-scanned
leftover form (~3.44). That is just a clumsy ANS: you still walk 64
narrow codes. Weight-granularity escape **loses**.

At 2-bit the inlier set is tiny and p_esc is large; escape cannot beat
H≈0.91 without becoming a full entropy coder.

**Verdict:** not a win as a *weight-RA* codec. The group-scanned leftover
form is a G-grain coder at ~3.44 index bpw (complete ≈ 3.69 with scale).
ESTIMATED decode: 64 × (3-bit extract + compare) + p_esc × leftover
extract ≈ **70 ops** to isolate one weight, **~8 ops/weight** if the
thread already walks the group (matvec).

### 5.2 Rice of a sparse outlier stream

The Q4 *index* stream is not sparse.

| R (inlier \|q\|≤R) | p_out (gate 2 097 152 sample) | rice-gap pos bpw |  main + pos + value (seq) | RA group-local (6+4 per out) |
|---|---:|---:|---:|---:|
| 0 (everything is an “outlier”) | 0.852 | 1.000 | **4.409** | 8.521 |
| 1 | 0.577 | 1.000 | **5.310** | 7.774 |
| 3 | 0.202 | 0.746 | **4.556** | 5.025 |
| 7 (no outliers; Q4 never exceeds 7) | 0 | 0 | **4.000** | 4.000 |

Rice of zigzag(q) itself, closed form from the full `gate_proj` hist:
best k=2, **3.650 bpw**, SEQ.

Existing `HGRAVR02` / `rice_q1_rms_2pct` is a *different object*: binary
base plus a 2 % residual on the **values**, not rice of uniform-Qn
indices (`residual_compact_codec.py:9-12`, `INDEX_MODES = rice |
group_local | bitmap`). Its Metal decoder is explicitly sequential:

```117:134:crates/hawking-core/shaders/q80_mixed_decode.metal
// Serial rice decode + scatter-add into y. One lane does the whole stream so
// the add order matches the CPU oracle (increasing packed index). Grid may be
// any non-zero size; only thread 0 works.
kernel void q80_rice_q1_residual_apply(...)
{
    if (tid != 0u || outlier_count == 0u || cols == 0u) {
        return;
```

`group_local` / `bitmap` in the same Python module are the RA-legal
index layouts for *that* residual family. They are not a win on Qn
indices, because the “outliers” of q are 20–85 % of the stream.

**Verdict: KILLS** as a coding of uniform-Qn indices.
**UNUSABLE** if the rice stream is global (one-lane prefix).
Per-group rice of Qn indices still costs ≥ 4 bpw on this distribution.

ESTIMATED ops (global rice): unbounded in the target index; one lane
walks the whole residual. ESTIMATED ops (per-group rice of a 20 % tail):
unary scan of ~13 symbols/group + k LSB ≈ **15–40 ops/weight** amortized
inside the group, still G-grain.

### 5.3 Per-group variable width + small width table

**KILLS** on this quantizer.

`mean_group_offset_width = 4.000`, `mean_group_signed_width = 4.000`
on every GEMV class at 4-bit. Width table is 3 bits/group = 0.046875 bpw.
Coded size **4.047 > 4**.

The same holds at every measured width: W ≈ bits, table adds 3 or 4 bits
per group, so var-width is strictly worse than the fixed pack. Cause:
absmax scaling saturates the code range in essentially every group
(`signed_width_hist` all-mass on `bits`).

A width table would only pay if the *quantizer* stopped scaling each
group to its own max (shared row scale, larger blocks, residual after a
predictor). That is a different codec, not an entropy code of G0 indices.

ESTIMATED ops if it *had* paid: 1 width-table load, 1 group-pointer load
(or a 80-wide prefix of a row), bit extract, sub, scale mul ≈ **8 ops +
2–3 loads**, grain W if pointers are stored, else G.

### 5.4 ANS, per-group independent streams

Static 12-bit-freq rANS, one model shared across the tensor, one stream
per group, 4-byte state flush. MEASURED on the 2 097 152-weight samples.
Global rANS on the same sample is the sequential ceiling.

`mlp.gate_proj` L0 sample (H = 3.4950):

| grain | stream+flush bpw | flush term 32/G | vs H | vs 4 | RA |
|---|---:|---:|---:|---:|---|
| G=64 | 3.852 | 0.500 | +0.357 | −0.148 | G |
| G=256 | 3.573 | 0.125 | +0.078 | −0.427 | G |
| G=1024 | 3.509 | 0.031 | +0.014 | −0.491 | G |
| global (one stream) | 3.495 | ~0 | +0.000 | −0.505 | **SEQ UNUSABLE** |

Other classes at 4-bit, G=64: 3.49 (embed sample) … 3.85 (gate). Embed’s
2 M-weight slice is a colder prefix (H 3.233 vs full-embed 3.498); do
not treat 3.49 as the class number. Full-embed production H is 3.498;
expect G=64 ANS ≈ 3.80 on that class too.

At other widths, same tensor (`mlp.gate_proj` sample):

| bits | H | rANS G64 | rANS G256 | rANS G1024 | rANS global (SEQ) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.926 | 1.389 | 1.040 | 0.948 | 0.926 |
| 3 | 2.319 | 2.735 | 2.423 | 2.340 | 2.320 |
| 4 | 3.495 | 3.852 | 3.573 | 3.509 | 3.495 |
| 5 | 4.576 | 4.860 | 4.616 | 4.579 | 4.577 |
| 6 | 5.609 | 5.881 | 5.588 | 5.592 | 5.611 |
| 8 | 7.616 | 8.121 | 7.562 | 7.524 | 7.645 |

G=64 ANS keeps a slice of the gap (1.389 vs nominal 2 at 2-bit; 3.852 vs
4 at 4-bit) but the 0.50-bit flush eats most of a 0.52-bit 4-bit gap.
G=256 keeps most of the gap at every width. G=1024 sits on H.

Theoretical check: H_group (8 192-group sample) at 4-bit is ~3.30. Plus
0.50 flush = 3.80, matching the measured 3.78–3.85. The rANS implementation
is size-only; it is not a bit-exact sibling of `sideinfo_rans.rs`. Treat
the *measured* sizes as a coding-efficiency probe, the *H + 32/G*
formula as the clean bound.

`sideinfo_rans.rs` exists for STRAND *side-info* (scale_q, outlier
positions), not weight indices. Its own ledger (file header) quoted
0.084 + 0.148 recoverable bpw on Qwen2.5-0.5B q2 side-info. That is a
different model and a different stream. Stolen science: static integer
CDF, 14-bit freqs, self-describing table, decode is integer-only. The
same construction applies to Qn indices if anyone ships G≥256 streams.

ESTIMATED ops/symbol for rANS decode: 1 mul, 1 cdf binary-search
(4 probes on a 16-entry table, or a 16-wide load), 1 sub, occasional
byte refill ≈ **12–20 ALU**. Grain G: a seek to weight i decodes the
whole group from the end (rANS). For a 256-group that is **~4 000 ops**
to isolate one weight, or **~15 ops/weight** if the thread consumes the
group for a matvec. Not a win against 6-op Q4 unless the byte traffic
reduction is the thing being bought, and even then only G≥256.

**Global ANS is UNUSABLE** under the seek constraint, even though it
hits H.

---

## 6. Scoreboard (4-bit G0 index stream, language GEMV)

Index bpw only. Add +0.250 scale + ~0.003 header to get complete BPW.

| scheme | index bpw | complete BPW (est.) | vs G0 4.253 | RA grain | ops/weight (ESTIMATED) | usable? |
|---|---:|---:|---:|---|---:|---|
| incumbent HQ30UQ4 | 4.000 | 4.253 MEASURED | 0 | W | 6 | yes (live) |
| Shannon ceiling | 3.479 MEASURED | 3.732 | −0.521 | n/a (bound) | n/a | bound only |
| escape k=3 leftover, group-scanned | 3.444 | 3.697 | −0.556 | G | ~8 walk / ~70 seek | yes, G |
| escape + per-ESC local index | 4.59 | 4.84 | +0.59 | W | ~8 + 12 on ESC | no (larger) |
| rice of zigzag(q) | 3.650 | 3.903 | −0.350 | **SEQ** | unbounded prefix | **UNUSABLE** |
| rice of \|q\|>3 + 3-bit main | 4.556 | 4.81 | +0.55 | SEQ or G | 15–40 | no (larger) |
| per-group var-width + 3-bit table | 4.047 | 4.300 | +0.047 | W/G | ~8 | no (larger) |
| rANS G=64 | 3.85 | 4.10 | −0.15 | G | ~15 walk / ~1000 seek | wash |
| rANS G=256 | 3.57 | 3.82 | −0.43 | G | ~15 walk / ~4000 seek | modest |
| rANS G=1024 | 3.51 | 3.76 | −0.49 | G | ~15 walk / ~15000 seek | modest, fat groups |
| rANS global | 3.495 | 3.75 | −0.50 | **SEQ** | prefix of tensor | **UNUSABLE** |

No complete-token measurement. A linear “bytes / quoted GB/s” conversion
of the 0.43–0.52 bpw is **rejected** (standing rule). The only existing
wall-to-BPW formula on this model
(`mixed-sub15-v1/PACK_REPORT.json` `projection.formula`) assumes the Q4
kernel’s decode cost. An ANS or escape kernel is a different genome.
PROJECTED TPS from that formula is not a finding of this lane.

---

## 7. Adjacent: fp16 group-scale stream (not the assigned question)

Order-0 entropy of the raw fp16 bit-patterns of the Q4 scale array,
one tensor per class (first tensor of that class). MEASURED.

| class | H (of 16-bit codes) | unique | recoverable vs 16 | /64 bpw |
|---|---:|---:|---:|---:|
| embed | 7.449 | 1244 | 8.551 | 0.134 |
| lm_head | 7.376 | 731 | 8.624 | 0.135 |
| mlp.gate_proj | 7.143 | 480 | 8.857 | 0.138 |
| mlp.up_proj | 7.072 | 377 | 8.928 | 0.140 |
| mlp.down_proj | 7.097 | 607 | 8.903 | 0.139 |
| in_proj_qkvz | 7.372 | 659 | 8.628 | 0.135 |
| out_proj | 7.268 | 588 | 8.732 | 0.136 |
| q/k/v/o_proj | 7.05–7.55 | 396–525 | 8.45–8.95 | 0.132–0.140 |

Same shape as the STRAND side-info observation (`sideinfo_rans.rs:6-18`):
fixed-width scales sit well above their Shannon rate. Combined with the
index gap, the theoretical joint ceiling is ≈ 0.52 + 0.14 = **0.66 bpw**
(complete ≈ 3.59) if *both* streams are coded and the scale table stays
RA (one scale per group is already W-addressable; an entropy-coded scale
stream needs its own G-grain or a seek table). Not measured as a joint
coder. Still not a path to 1.5.

---

## 8. What this does not do

- Does not reach G1 complete BPW < 1.5. The Qn-index entropy ceiling
  under the live quantizer is 3.73 complete, 3.59 if scales are also
  coded. The missing 2.1–2.2 bpw is not in the index histogram.
- Does not produce a representation-specific kernel. Preferred shape
  remains “low-bpw consumed directly”; this lane did not build one.
- Does not re-evaluate `rice_q1` / HGRAVS01 / binary_g128 as *value*
  codecs. Those change the reconstruction, not the coding of an already
  chosen Qn index. They are out of this question. `mixed-sub15-v1`
  already recorded that attention rice_q1 at ~1.29 complete BPW is
  incoherent; that is a quality result, not an entropy-coding result.
- Does not measure TOKEN_NS or TPS. G0 26.4 TPS / 37.9 M TOKEN_NS
  remain unverified claims.

---

## 9. KILLS / REOPEN_IF

**KILLS (this mechanism, this quantizer, this RA constraint):**

1. Per-group variable width of G0-style absmax-Qn indices.
2. Rice of the Qn index stream (not sparse; global rice is SEQ).
3. Weight-RA escape tables (local index + extra bits exceed 4).
4. Per-group ANS at G=64 as a G1 lever (flush ≈ gap).
5. Any story that 0.52 free bits/weight, coded, is the road from 4.25
   to 1.5.

**REOPEN_IF:**

- The quantizer stops absmax-scaling every 64 weights, so unused MSBs
  actually appear (row scale, super-group scale, residual after a
  cheap predictor, or a learned codebook). Then var-width and escape
  get a new measurement, not this one.
- A G≥256 per-group ANS (or group-scanned leftover-escape) kernel
  consumes the coded stream *directly* and a serialized GPU lane
  measures complete-token TOKEN_NS. The 0.43 bpw is real; it is not
  known to be a net physical win.
- Someone jointly codes indices + scales with a RA-legal seek table
  and re-measures. Ceiling still ~3.59 complete.

Cheapest experiment that would change the verdict: change the
quantizer, not the coder. Re-histogram q under a *row-absmax* or
*block-256-absmax* rule on one `gate_proj` and one `q_proj`. If
`mean_group_offset_width` drops below `bits - 0.5`, reopen §5.3.

---

## 10. Evidence index

| claim | pointer |
|---|---|
| G0 packer scale=amax/7, clamp[-8,7] | `qwen80_uniform_q4.rs:233-248` |
| G0 Q4 O(1) seek | `qwen_uniform_q4.metal:22-36` |
| G0 complete BPW 4.252735 | `uniform-q4-v1/manifest.json` field `complete_physical_bpw` |
| Requant ≡ packed on gate L0 | python match rate 1.0, H 3.4952326949731978 |
| Production H table | `/tmp/g1_entropy_measure.json` `q4_production` |
| Weighted H 3.4789 | same, Σ H·n / Σ n over 26 893 352 960 codes |
| Width hist all-mass on 4 | `q4_production.mlp.gate_proj.signed_width_hist` |
| Multi-width H | `bf16_sweep` |
| Q3 packed H 2.31 | HGRAVU01 `mixed-q3mlp-v1` three organs × three layers |
| Q7/Q8 packed H | `mixed-floor-q7-v1` / `q8-v1` `L00.hq38seg` |
| rANS sizes | `codec_samples.*.rans_per_group` / `rans_global` |
| Rice not sparse | `codec_samples.mlp.gate_proj.4.rice_outlier` |
| Metal rice is one-lane SEQ | `q80_mixed_decode.metal:117-134` |
| HGRAVU01 clip | `ascension_dual_gravity_worker.py:673-676` |
| BA fuse layout | `qwen38_geometry.rs:360-389` |
| Scale-stream H | python over first Q4 tensor of each class |
| Harness wall | `/tmp/g1_entropy_measure.json` `wall_s` = 268.1715817451477 |

Command that produced the JSON:

```
python3 /tmp/g1_entropy_measure.py --phase all \
  --out /tmp/g1_entropy_measure.json --max-embed-elems 33554432
# exit 0, wall 268.17 s
```

---

## Completion report

STATUS is MEASURED_NEGATIVE. See headings after this file in the lane
return.
