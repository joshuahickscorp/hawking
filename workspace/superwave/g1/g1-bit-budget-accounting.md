# G1 bit-budget accounting — Qwen3.8 language body

Lane: `06-bit-budget-accounting`. Arithmetic only. No GPU, no pack, no generate.

Epistemic tags: **MEASURED** = integer or float read from an on-disk artifact or source header. **DERIVED** = exact arithmetic on MEASURED integers. **PROJECTED** = not used for any kill. **UNVERIFIED** = historical claim this lane did not re-measure.

---

## 1. Identity

Hawking complete physical BPW (the quantity G1 constrains):

```
complete_physical_bpw = 8 * tensor_payload_bytes / source_weight_elements
```

Evidence: `crates/hawking-core/src/model/qwen38_pack.rs:673-679`

```
    let source_weight_elements: u64 = rows.iter().map(|row| row.elements).sum();
    let tensor_payload_bytes: u64 = rows.iter().map(|row| row.bytes).sum();
    let complete_physical_bpw = if source_weight_elements == 0 {
        0.0
    } else {
        (tensor_payload_bytes as f64 * 8.0) / source_weight_elements as f64
    };
```

`tensor_payload_bytes` is the sum of on-disk codec payloads. It already contains codes, scales, zero-points (if any), codebooks, sparse indices, per-group metadata, per-tensor headers, and last-group alignment padding. It does **not** contain the JSON/binary catalog file. Catalog tax is a separate line (section 8).

G1 target is **strict**: `complete_physical_bpw < 1.5`.

The inversion identity used below (four classes):

```
complete = (E_mlp * b_mlp + E_attn * b_attn + E_tab * b_tab + E_small * b_small) / N
```

Solve for attention:

```
b_attn < (1.5 * N - E_mlp * b_mlp - E_tab * b_tab - E_small * b_small) / E_attn
```

`b_*` are **physical** class BPW, not nominal bit-widths.

---

## 2. What is counted

Language-only Qwen3.8-27B. Vision skipped. That is the G0 pack contract.

Evidence: `qwen38_pack.rs:104-107` skips `vision_tower.*`; `qwen38_pack.rs:474-664` packs `language_model.*` only. G0 manifest field `skipped_vision_tensors = 333`.

`config.json` on the BF16 source has `mtp_num_hidden_layers: 1`. The safetensors index contains **zero** tensors whose name contains `mtp`. MTP is not in N.

Tie embeddings is false. Embed and `lm_head` are two distinct `248320 × 5120` tables.

`attn_output_gate: true` is already inside GQA `q_proj` (`12288 × 5120` = `2 × 24 × 256 × 5120`). Not a hidden extra matrix.

---

## 3. Geometry constants (source of shapes)

Evidence: `crates/hawking-core/src/model/qwen38_geometry.rs:20-52`

```
pub const QWEN38_LAYERS: usize = 64;
pub const QWEN38_DELTANET_LAYERS: usize = 48;
pub const QWEN38_GQA_LAYERS: usize = 16;
pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_INTERMEDIATE: usize = 17_408;
pub const QWEN38_VOCAB: usize = 248_320;
pub const QWEN38_IN_PROJ_QKV_ROWS: usize = 10_240;
pub const QWEN38_IN_PROJ_Z_ROWS: usize = 6_144;
pub const QWEN38_QKVZ_ROWS: usize = 16_384;
pub const QWEN38_BA_ROWS: usize = 96;
pub const QWEN38_Q_PROJ_ROWS: usize = 12_288;
pub const QWEN38_KV_PROJ_ROWS: usize = 1_024;
pub const QWEN38_O_PROJ_ROWS: usize = 5_120;
pub const QWEN38_O_PROJ_COLS: usize = 6_144;
```

Confirmed against live `config.json` at
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/config.json`
(`text_config.num_hidden_layers=64`, `hidden_size=5120`, `intermediate_size=17408`, `vocab_size=248320`, `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`, `full_attention_interval=4`, `tie_word_embeddings=false`, `attn_output_gate=true`).

---

## 4. Per-class element census — MEASURED

Source: 11-shard BF16 tree, `model.safetensors.index.json` (1184 names) + every shard header.

Command (this lane): parse `weight_map`, read each shard's safetensors JSON header, sum `shape` products by class.

| class | tensors | shape each | elements (MEASURED) |
|---|---:|---|---:|
| mlp.gate_proj | 64 | `[17408, 5120]` | 5,704,253,440 |
| mlp.up_proj | 64 | `[17408, 5120]` | 5,704,253,440 |
| mlp.down_proj | 64 | `[5120, 17408]` | 5,704,253,440 |
| **MLP** | **192** | | **17,112,760,320** |
| linear_attn.in_proj_qkv | 48 | `[10240, 5120]` | 2,516,582,400 |
| linear_attn.in_proj_z | 48 | `[6144, 5120]` | 1,509,949,440 |
| linear_attn.in_proj_a | 48 | `[48, 5120]` | 11,796,480 |
| linear_attn.in_proj_b | 48 | `[48, 5120]` | 11,796,480 |
| linear_attn.out_proj | 48 | `[5120, 6144]` | 1,509,949,440 |
| self_attn.q_proj | 16 | `[12288, 5120]` | 1,006,632,960 |
| self_attn.k_proj | 16 | `[1024, 5120]` | 83,886,080 |
| self_attn.v_proj | 16 | `[1024, 5120]` | 83,886,080 |
| self_attn.o_proj | 16 | `[5120, 6144]` | 503,316,480 |
| **attention GEMV** (fused qkvz+ba) | **208 fused / 304 source** | | **7,237,795,840** |
| embed_tokens | 1 | `[248320, 5120]` | 1,271,398,400 |
| lm_head | 1 | `[248320, 5120]` | 1,271,398,400 |
| **tables** | **2** | | **2,542,796,800** |
| linear_attn.conv1d | 48 | `[10240, 4, 1]` | 1,966,080 |
| input_layernorm | 64 | `[5120]` | 327,680 |
| post_attention_layernorm | 64 | `[5120]` | 327,680 |
| linear_attn.norm | 48 | `[128]` | 6,144 |
| final norm | 1 | `[5120]` | 5,120 |
| self_attn.q_norm | 16 | `[256]` | 4,096 |
| self_attn.k_norm | 16 | `[256]` | 4,096 |
| linear_attn.A_log | 48 | `[48]` | 2,304 |
| linear_attn.dt_bias | 48 | `[48]` | 2,304 |
| **small (f32 in G0)** | **353** | | **2,645,504** |
| **language N** | **851 source / 755 fused** | | **26,895,998,464** |
| vision_tower (excluded) | 333 | mixed | 460,730,096 |

Cross-checks (all match):

- `N = 17_112_760_320 + 7_237_795_840 + 2_542_796_800 + 2_645_504 = 26_895_998_464`
- G0 manifest `source_weight_elements = 26895998464` (`uniform-q4-v1/manifest.json`)
- G016 receipt `total_params = 26895998464`, `mlp_params = 17112760320`, `embed_plus_lm_head = 2542796800`, `attention_and_norms = 7240441344` (`E_attn + E_small`)
- `git show HEAD:receipts/ascent-2026-08-16/G016_BPW_FEASIBILITY.json` geometry block

Fused catalog (G0 pack) merges `in_proj_qkv+z → qkvz [16384,5120]` and `in_proj_b+a → ba [96,5120]`. Element count is identical. G0 has 755 tensors (402 q4 + 353 f32); mixed packs keep the 851 source names.

Mass fractions **DERIVED** (`e / 26895998464`):

| class | fraction | percent |
|---|---:|---:|
| MLP | 0.6362567406785528 | 63.625674% |
| attention GEMV | 0.2691030730719185 | 26.910307% |
| tables | 0.0945418257442090 | 9.454183% |
| small | 0.0000983605053198 | 0.009836% |

G016 printed `0.6363 / 0.2692 / 0.0945`. Those are 4-decimal roundings of the same integers. This file uses the integers.

---

## 5. Non-weight costs a real artifact pays

Nominal bit-width is not complete BPW. Every proposal must bill the items in this section.

### 5.1 Uniform-Qn group-64 (incumbent family, HQ30UQ4 / HGRAVU01)

Evidence: `crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:47-48,195-273` and `uniform_q4.rs:15-18`.

```
pub const UNIFORM_Q4_NOMINAL_BPW: f64 = 4.0 + 16.0 / UNIFORM_Q4_GROUP_SIZE as f64;  // 4.25
```

Per tensor:

| item | bytes | bpw on a tensor with `n % 64 == 0` |
|---|---|---:|
| 4-bit offset-binary codes (`q ∈ [-8,7]`, nibble packed) | `n / 2` | 4.0 |
| FP16 scale per group of 64 (`f16(max_abs/7)`) | `2 * n / 64` | 0.25 |
| zero point | **none** | 0 |
| header: magic 8 + ver 4 + group 4 + rank 2 + reserved 2 + elements 8 + reserved 4 = 32, plus `rank * 4` dims | 40 for rank-2 | `320 / n` |
| last-group code/scale pad | `ceil(n/64)` groups | 0 on this model |

All language GEMVs and both tables have `n % 64 == 0`. **Group padding on this body is zero.** MEASURED.

Qn generalisation used by mixed-floor packs: same header + FP16 scale / 64 + `ceil(64 * bits / 8)` code bytes per group.

| bits | physical body BPW | + rank-2 header on one table (`n = 1_271_398_400`) |
|---:|---:|---:|
| 2 | 2.25 | 2.250000251691366 |
| 3 | 3.25 | 3.250000251691366 |
| 4 | 4.25 | 4.250000251691366 MEASURED embed/lm_head |
| 7 | 7.25 | 7.250000251691366 |
| 8 | 8.25 | 8.250000251691366 |

Adding an asymmetric zero-point (AWQ-style int16/fp16 zp per group) would add another `16/64 = 0.25` BPW. HQ30UQ4 does not.

`uniform_qn.rs` (HQ30UQ2/UQ3) uses **group 128**, not 64: body BPW = `bits + 16/128 = bits + 0.125`. A proposal that says "q3" must name the group size. Descent-sweep `uniform_q3_g64` is the 3.25 number; the Rust `HQ30UQ3` path is 3.125.

### 5.2 Binary g128 (HGRAVB01)

Body: 1 sign bit + FP16 scale / 128 = 1.125 BPW.

MEASURED on one `17408×5120` gate (`receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` and mixed-2p0 `mlp_gate_proj`):

- `elements = 89128960`
- `payload_bytes = 12534021`
- `physical_bpw = 1.1250234267290902`
- body = `89128960/8 + 89128960/128*2 = 12533760`
- container header = **261 bytes**

### 5.3 Rice residual (HGRAVR02, `rice_q1_rms_2pct`)

Evidence: `lab/operators/residual_compact_codec.py` header comment + `encode_residual_compact(..., outlier_ratio=0.02, index_mode="rice", value_bits=1, value_scale="rms")`.

Pays, on top of the binary base:

- rice-coded sorted indices of `ceil(0.02 n)` outliers
- 1-bit sign per outlier + one stored residual scale
- container header

MEASURED on all 64 `up_proj` (mixed-2p0): `physical_bpw = 1.2875108157887178`. Residual tax over binary = **0.1624873890596277 BPW**.

MEASURED on 304 attention GEMVs (sub15): `bytes = 1165098376`, `elements = 7237795840`, `physical_bpw = 1.2877935788805008`.

Raising the outlier ratio raises the index bill. mixed-floor-q8-up10 (`outlier_ratio=0.1`) MEASURED `up_proj` at `1.7091418182148652`.

### 5.4 HGRAVS01 r160_b3 (activation-weighted SVD)

Evidence: `lab/operators/hgravs01_adapter.py:80-149`. Payload = magic 8 + JSON header length 4 + JSON + left factor (fp16 scales + codes) + right factor (fp16 scales + codes). JSON header is billed.

`down_proj` is `[5120, 17408]`. Rank-160 factors: `5120×160 + 160×17408 = 3,604,480` elems = 4.044% of the matrix. Naive `3.25 * 0.040441 = 0.131434`.

MEASURED on all 64 downs (mixed-2p0): `bytes = 93847197`, `elements = 5704253440`, `physical_bpw = 0.13161714918473189`. Header + scale tax = 0.000183 BPW above the naive factor product.

This number is **not** transferable to attention without a new fit. It is a measured MLP-down point only.

### 5.5 f32 small-tensor container

Evidence: `qwen38_pack.rs:442-463`. Payload = `u64` length + `n * 4` LE f32. **8-byte header, 32 BPW body.**

MEASURED G0 / sub15 small class: `bytes = 10584840`, `elements = 2645504`, `physical_bpw = 32.00853977162764`.

| tensor | n | physical BPW |
|---|---:|---:|
| conv1d | 40960 | 32.0015625 |
| residual / final RMS | 5120 | 32.0125 |
| q/k_norm | 256 | 32.25 |
| DeltaNet norm | 128 | 32.5 |
| A_log / dt_bias | 48 | 33.333... |

These 2.65 M elements are 0.0098% of N. At 32 BPW they still contribute **0.003147 BPW** to complete. Leaving them f32 is not a 1.5-killer. Quantizing them cannot save more than that 0.003.

### 5.6 Codebooks, two-stage, additive

Descent-sweep `additive_q2q2_g64` MEASURED `physical_bpw = 4.500040929457721` on one gate (two q2 stages + two scale sets + header). That is **worse** than uniform-q4 4.25. Any additive/codebook proposal must bill every codebook and every index table.

### 5.7 Catalog / manifest

G0 `complete_physical_bpw` does **not** include `manifest.json`.

mixed-2p0 does publish both:

- `tensor_payload_bytes = 7011580330` → DERIVED tensor BPW `2.0855385872764454`
- `all_required_weight_artifact_bytes = 7011764637` → MEASURED artifact BPW `2.0855934079220506`
- catalog tax = `184307` bytes = **0.0000548206456055 BPW**

Catalog tax cannot flip a 1.5 verdict. Conservative G1 billing may use artifact-complete; this file's inversion uses the pack.rs tensor law and notes the +5.5e-5.

### 5.8 Resident RAM alignment

Host admission MEASURED child RSS `8.77 GB` against an `8.5 GB` artifact (`qwen38_host_admission.rs:3-7,17-20`). That gap is process overhead + workspace, not a per-tensor pad in the artifact. This lane does not convert RSS slop into BPW. A representation proposal is killed by **artifact** complete BPW, not by RSS.

### 5.9 Vision

`460,730,096` elements, `921,460,192` BF16 bytes. Not in N. Loading vision at q4 would add ~245 MB and move complete by ~0 (N and bytes grow together toward 4.25). Loading vision at bf16 while compressing text cannot be hidden inside language complete BPW. G1 language-only artifacts must keep skipping it.

---

## 6. G0 incumbent — MEASURED complete BPW

Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`

```
schema                  hawking.ascent.qwen38_language_uniform_q4.v1
source_weight_elements  26895998464
tensor_payload_bytes    14297694680
complete_physical_bpw   4.252735126866492
nominal_codec_bpw       4.25
q4_tensors              402
f32_tensors             353
skipped_vision_tensors  333
```

This lane re-summed every catalog row:

| class | bytes (MEASURED) | physical BPW (DERIVED) |
|---|---:|---:|
| mlp.gate / up / down (each) | 3,030,387,200 | 4.250003590303309 |
| MLP total | 9,091,161,600 | 4.250003590303309 |
| attention GEMV | 3,845,087,360 | 4.250009196169866 |
| tables | 1,350,860,880 | 4.250000251691366 |
| small f32 | 10,584,840 | 32.00853977162764 |
| **complete** | **14,297,694,680** | **4.252735126866492** |

`8 * 14297694680 / 26895998464 = 4.252735126866492` exactly. The historical G0 figure `~4.2527` is this integer ratio. **MEASURED, this lane.**

Why not 4.25: 402 rank-2 q4 headers (40 B) + 353 f32 headers (8 B) + 2.645 M small elements stored at 32 BPW instead of 4.25. Nominal 4-bit accounting hides 9,195,496 bytes.

G0 TPS `~26.4` and TOKEN_NS `~37,900,000` remain **UNVERIFIED** here (GPU lane owns them). Closest written complete-wall figures are UNVERIFIED receipts (`QWEN38_RUNG_TARGETS.json` wall 38.217 ms / 26.17 tok/s; `RUNG_QWEN38_MEASURED.json` 38.216792 ms / 26.1665 tok/s). Not used in any kill.

---

## 7. Inversion — max attention BPW for complete < 1.5

Fixed in this section:

- `E_mlp = 17112760320`
- `E_attn = 7237795840` (GEMVs only)
- `E_tab = 2542796800`
- `E_small = 2645504`
- `b_small = 32.00853977162764` (G0/sub15 MEASURED; the must-pay f32 tax)
- ceiling bits = `1.5 * N = 40343997696`
- table bits at G0 Q4 = `8 * 1350860880 = 10806887040`

### 7.1 Assigned question, tables held at G0 Q4 (incumbent)

`b_tab = 4.250000251691366` MEASURED.

| b_mlp | rem bits after mlp+tables+small | **max b_attn** (must be strictly below) |
|---:|---:|---:|
| 0.5 | 20,896,051,776 | **2.8870739432186028** |
| 0.8 | 15,762,223,680 | **2.1777657215597834** |
| 1.0 | 12,339,671,616 | **1.7048935737872375** |

DERIVED. Fractions: `20896051776/7237795840`, `15762223680/7237795840`, `12339671616/7237795840`.

Equality produces complete = 1.5. G1 needs `< 1.5`, so a live assignment must sit strictly under the cell.

If attention is defined the G016 way (GEMV + small, `E = 7240441344`) and small is **not** pinned at f32:

| b_mlp | max b_attn_wide @ tables Q4 |
|---:|---:|
| 0.5 | 2.897714310383 |
| 0.8 | 2.188665254934 |
| 1.0 | 1.715965884634 |

Same kill band. The f32 small tax does not move the decimal that matters.

### 7.2 Same inversion, other table budgets

`b_small` still f32. `b_attn` is GEMV physical.

| tables physical | mlp 0.5 | mlp 0.8 | mlp 1.0 |
|---|---:|---:|---:|
| 0 (info lower bound, not a codec) | 4.380192 | 3.670884 | 3.198012 |
| binary_g128 1.125023 MEASURED | 3.984947 | 3.275639 | 2.802767 |
| rice_q1 1.287511 MEASURED | 3.927862 | 3.218553 | 2.745681 |
| q2 g64 2.2500002517 | 3.589718 | 2.880410 | 2.407538 |
| q3 g64 3.2500002517 | 3.238396 | 2.529088 | 2.056216 |
| **q4 g64 MEASURED 4.2500002517** | **2.887074** | **2.177766** | **1.704894** |
| q7 g64 7.2500002517 | 1.833108 | 1.123800 | 0.650928 |
| q8 g64 8.2500002517 | 1.481786 | 0.772478 | 0.299606 |
| int8 no-scale 8.0 | 1.569617 | 0.860308 | 0.387436 |
| **bf16 16.0** | **IMPOSSIBLE** | **IMPOSSIBLE** | **IMPOSSIBLE** |
| **f32 32.0** | **IMPOSSIBLE** | **IMPOSSIBLE** | **IMPOSSIBLE** |

### 7.3 Reverse: max table BPW if attention stays G0 Q4

`b_attn = 4.250009196169866` MEASURED.

| b_mlp | max b_tab |
|---:|---:|
| 0.0 | 3.735501 |
| 0.131617 (HGRAVS01-all-mlp, hypothetical) | 2.849731 |
| 0.5 | 0.370553 |
| 0.8 | **negative — IMPOSSIBLE** |
| 1.0 | **negative — IMPOSSIBLE** |

KILLS: any proposal that keeps attention at uniform-Q4 and tables at ≥4.25, with MLP at the assigned 0.5/0.8/1.0 points.

---

## 8. Combination grid — REACH vs KILL

`b_small` fixed at f32. Cell is DERIVED complete BPW. `<` means `complete < 1.5` (arithmetically possible). `>` means `complete ≥ 1.5` (**KILL**, codec quality irrelevant).

### 8.1 Tables at G0 Q4 (4.2500002517) — default incumbent

```
mlp \ attn     0.1316   0.5000   0.8000   1.0000   1.1250   1.2878   2.2500   3.2500   4.2500
0.1316 HGRAVS  0.5241<  0.6232<  0.7040<  0.7578<  0.7914<  0.8352<  1.0942<  1.3633<  1.6324>
0.5000 assign  0.7585<  0.8576<  0.9384<  0.9922<  1.0258<  1.0696<  1.3286<  1.5977>  1.8668>
0.5346 Q80mean 0.7805<  0.8796<  0.9604<  1.0142<  1.0478<  1.0916<  1.3506<  1.6197>  1.8888>
0.8000 assign  0.9494<  1.0485<  1.1292<  1.1831<  1.2167<  1.2605<  1.5194>  1.7885>  2.0576>
0.8481 mixed   0.9799<  1.0791<  1.1598<  1.2136<  1.2473<  1.2911<  1.5500>  1.8191>  2.0882>
1.0000 assign  1.0766<  1.1758<  1.2565<  1.3103<  1.3440<  1.3878<  1.6467>  1.9158>  2.1849>
1.1250 binary  1.1562<  1.2553<  1.3360<  1.3899<  1.4235<  1.4673<  1.7262>  1.9953>  2.2644>
1.2875 rice    1.2596<  1.3587<  1.4394<  1.4932<  1.5269>  1.5707>  1.8296>  2.0987>  2.3678>
2.2500 q2      1.8720>  1.9711>  2.0518>  2.1056>  2.1393>  2.1831>  2.4420>  2.7111>  2.9802>
3.2500 q3      2.5082>  2.6074>  2.6881>  2.7419>  2.7755>  2.8194>  3.0783>  3.3474>  3.6165>
4.2500 q4      3.1445>  3.2436>  3.3243>  3.3781>  3.4118>  3.4556>  3.7145>  3.9836>  4.2527>
```

Reading the assigned rows against incumbent table Q4:

| MLP | attention must be | first KILL on this grid |
|---:|---|---|
| 0.5 | < 2.887 | q3 attention (3.25) already dead; q2 attention (2.25) still alive |
| 0.8 | < 2.178 | q2 attention (2.25) dead; rice attention (1.288) alive |
| 1.0 | < 1.705 | q2 attention dead; rice attention alive |

Uniform-q3 on **both** MLP and attention is dead at table Q4 (`2.74–2.82`). Uniform-q2 on **both** is dead (`2.44`). Uniform-q4 on anything large plus another q4 class is dead.

### 8.2 Tables at q2 g64 (2.2500002517)

Opens more of the high-attention columns. MLP=0.5 + attn Q4 = `1.6777>` still **KILL**. MLP=0.5 + attn q3 = `1.4086<` REACH. Crushing tables to q2 does **not** save a Q4-attention + Q4-MLP pair (`4.06`).

### 8.3 Tables at q8 g64 (8.2500002517)

| MLP | attention must be |
|---:|---|
| 0.5 | < 1.481786 |
| 0.8 | < 0.772478 |
| 1.0 | < 0.299606 |

Q8 tables + MLP 0.8 already forbids every incumbent attention codec except an HGRAVS01-class structured code (unmeasured on attention). Q8 tables + MLP 1.0 forces attention to 0.30 physical.

int8-without-scale (8.0 exactly) is only slightly looser (1.570 / 0.860 / 0.387).

### 8.4 Byte ceiling

`1.5 * N / 8 = 5,042,999,712` bytes. Strictly under that.

| artifact | payload bytes | complete BPW | vs 5.043 GB |
|---|---:|---:|---|
| G0 uniform-q4 | 14,297,694,680 MEASURED | 4.252735126866492 | 2.835× over |
| mixed-2p0 tensor | 7,011,580,330 MEASURED | 2.0855385872764454 | 1.390× over |
| mixed-2p0 artifact | 7,011,764,637 MEASURED | 2.0855934079220506 | 1.390× over |
| mixed-sub15 | 4,340,604,637 MEASURED | 1.2910781930062503 | under, **incoherent** |

---

## 9. Hard kills — impossible regardless of codec quality

These die from mass × bits. No kernel and no codebook resurrects them.

**KILL 1 — tables stay bf16 or f32.**
`16 * E_tab / N = 1.5126692119073435`. Tables alone exceed 1.5. Deficit = `340,751,104` bits = 42.6 MB. Rest-of-model at 0 bits still fails.
`REOPEN_IF`: tables are not stored (tied embeddings + on-the-fly lm_head, or a smaller vocab). They are not tied. Vocab is 248,320.

**KILL 2 — crush MLP only, leave attention + tables at Q4.**
`complete(0, 4.250009196, 4.250000252, 32.00854) = 1.548641694627961`.
Even a 0-bit MLP misses 1.5 by 0.0486 BPW. G016 already said this in words; the integer form is the kill.
`REOPEN_IF`: attention or tables leave the Q4 family.

**KILL 3 — assigned MLP 0.5/0.8/1.0 + attention Q4 + tables Q4.**
Completes `1.8668 / 2.0576 / 2.1849`. All dead. This is the "keep the incumbent attention kernel, just squeeze MLP" proposal.
`REOPEN_IF`: attention physical < 2.887 / 2.178 / 1.705 respectively, **or** tables drop far below 4.25 (section 7.3: at MLP 0.5 tables would need < 0.371, which is below binary).

**KILL 4 — uniform-q3 or coarser on both MLP and attention, tables at Q4.**
q3+q3+q4 = 2.74. q2+q2+q4 = 2.44. Ternary-everywhere at 2.25 body is still 2.18 with Q4 tables.
`REOPEN_IF`: tables also drop, or one of the large classes uses a structured codec well below 2.

**KILL 5 — G0 itself.**
4.2527 is 2.835× the 1.5 byte ceiling. Not a proposal; the incumbent.

**KILL 6 — treat Qwen3.8 like Q80.**
Q80 ledger (`receipts/QWEN80_BIT_BUDGET_LEDGER.json`): `complete = 0.97032 * expert + 0.02968 * nonexpert`. Experts are 97% of mass, so expert 1.3 + 8-bit sensitive organs still clears 1.5.
Qwen3.8 MLP is 63.6%, not 97%. The Q80 subsidy does not exist here. A 1.3-BPW MLP + 8-bit everything else is `complete ≈ 3.54` (mixed-floor-q8 MEASURED 3.540559).

---

## 10. Measured packed points (not new this lane)

| artifact | recipe | complete BPW | class |
|---|---|---:|---|
| uniform-q4-v1 | all GEMV+tables HQ30UQ4 g64; small f32 | 4.252735126866492 MEASURED | G0 incumbent |
| mixed-2p0-v1 | gate binary 1.125; up rice 1.288; down HGRAVS01 0.132; non-MLP Q4 | 2.0855934079220506 MEASURED artifact | above 1.5; **KILL for G1 complete**. MLP physical 0.8480504639008466 |
| mixed-q4down-v1 | mixed-2p0 but down is Q4 not r160 | 2.959042928357003 MEASURED | KILL |
| mixed-q3mlp-v1 | MLP all q3; non-MLP Q4 | 3.613864737317677 MEASURED | KILL |
| mixed-floor-q7-v1 | mixed-2p0 MLP; non-MLP q7 | 3.176815835796740 MEASURED | KILL |
| mixed-floor-q8-v1 | mixed-2p0 MLP; non-MLP q8 | 3.540559152227054 MEASURED | KILL |
| mixed-sub15-v1 | mixed-2p0 MLP + rice attention + Q4 tables | 1.291078193006250 REACH | **MEASURED incoherent** (`QWEN38_SUB15_INCOHERENT.json`: token cycle `220/264`, 0 fallbacks) |

sub15 is the existence proof that **< 1.5 is arithmetically reachable** with currently packed codecs, and that **this particular assignment is not a capability-preserving point**. Arithmetic REACH ≠ promote.

G016 scenario A (Q80 organ BPWs on Q38 mass, non-MLP 4.25): DERIVED with exact fractions `complete(0.5345848147860832, 4.250595793629995, 4.250595793629995, 4.250595793629995) = 1.8862587599050495`. Matches their 1.8863. That was a 2.0-target envelope, not a 1.5 envelope. **KILL for G1 < 1.5.**

G016 scenario C (`1.2404`) is their 4-decimal mix of `mlp_mean` on attention **and** tables at 8. Recomputed exact: `complete(0.534585, 0.534585, 8.0, 4.25) = 1.240744`. That cell assumes attention physical ≈ 0.53. No Qwen3.8 attention artifact in this tree has that BPW except sub15 rice at 1.29, which is incoherent. Scenario C is an envelope, not a packed point.

---

## 11. Active BPW is the wrong ceiling

Dense decode reads every weight except the embedding table (one row gathered). Greedy `lm_head` is fully read.

| | elements | G0 bytes | BPW |
|---|---:|---:|---:|
| complete (pack.rs) | 26,895,998,464 | 14,297,694,680 | 4.252735126866492 |
| active (exclude embed table) | 25,624,600,064 | 13,622,264,240 | 4.252870821312968 |

Evidence: `QWEN38_ACTIVE_BUDGET_MEASURED.json` `active_bytes_per_token = 13622264240` matches `G0 bytes - embed bytes`.

Active and complete differ by 0.00014 BPW on this model because the embed table is itself Q4. On a small model with f32/bf16 tables the gap is the whole story; here the tables are 9.45% of N already inside complete. **G1's 1.5 is complete, and the inversion above is the one that binds.**

Census `active_weights_per_token: 23611000000` (`QWEN38_ARCH_CENSUS.json`) omitted both `out_proj` families (DeltaNet 1.510 G + GQA 0.503 G). Do not use 23.611 G. The header census and the G0 manifest are the authority.

---

## 12. What a G1 representation proposal must show

Minimum bill of materials, or it is not a proposal:

1. Per-class physical BPW, not nominal bits. Formula: `8 * payload_bytes / elements` including every item in section 5.
2. Tables named. Silent "weights at X bits" that leaves embed/lm_head at bf16 is KILL 1.
3. Attention named separately from MLP. Silent "MLP at 0.8" with incumbent Q4 attention is KILL 3.
4. Small tensors: f32 (this file's default) or a billed alternative. Saving them cannot buy more than 0.003 BPW.
5. Catalog bytes if the runtime needs one.
6. A representation-specific consume path (standing rule). Expand-to-Q4-then-generic-GEMV is out of scope here; this file only kills on bytes.

Cheap next measurement that would tighten a REACH cell (not required to accept this lane): pack one attention GEMV family at the inversion cap for the chosen MLP/table pair and run the existing coherence prompts. sub15 already falsified rice-on-attention at 1.29. The open band is `1.29 < b_attn < 2.18` at MLP 0.8 + tables Q4, and `1.29 < b_attn < 2.89` at MLP 0.5 + tables Q4.

---

## 13. Claim boundary

- Element counts: MEASURED from BF16 safetensors headers and re-confirmed on the G0 manifest.
- G0 / mixed / sub15 complete BPW: MEASURED from those artifacts' byte sums.
- Inversion cells: DERIVED from those integers. No fit, no generate, no GPU.
- Coherence of every REACH cell except sub15 (negative) is **untested**.
- Q4K / AWQ / strand-quant physical BPW was not re-derived. Those families still pay the identity in §1; they do not get to quote a nominal 4.0. Cheapest experiment: encode one `17408×5120` with the candidate encoder, print `8*len(payload)/n`.
- G0 TPS / TOKEN_NS: UNVERIFIED this lane.

---

## Required test output (this lane)

```
$ test -s workspace/superwave/g1/g1-bit-budget-accounting.md && echo PASS
PASS

$ wc -l workspace/superwave/g1/g1-bit-budget-accounting.md
     591 workspace/superwave/g1/g1-bit-budget-accounting.md

$ git status --porcelain
?? workspace/superwave/g1/g1-bit-budget-accounting.md
```

---

```
STATUS
SUPPORTED

CLAIMS
C1. Language N = 26,895,998,464. MLP 17,112,760,320 (63.625674%). Attention GEMV 7,237,795,840 (26.910307%). Tables 2,542,796,800 (9.454183%). Small 2,645,504 (0.009836%). Vision 460,730,096 excluded. MEASURED.
C2. Complete BPW is 8 * payload_bytes / N, payload includes scales, headers, indices, codebooks, group pad. pack.rs:673-679.
C3. G0 complete BPW = 4.252735126866492 = 8 * 14,297,694,680 / 26,895,998,464. MEASURED this lane from uniform-q4-v1/manifest.json.
C4. Tables at bf16 already produce complete 1.5126692119073435 with the rest at 0 bits. KILL any bf16/f32 table proposal. DERIVED.
C5. MLP = 0 and attention+tables at G0 Q4 produces complete 1.548641694627961. KILL crush-MLP-only. DERIVED.
C6. With tables at G0 Q4 and small at f32, max attention physical BPW for complete < 1.5 is 2.8870739432186028 (MLP 0.5), 2.1777657215597834 (MLP 0.8), 1.7048935737872375 (MLP 1.0). Strictly below. DERIVED.
C7. With tables at q8 g64 those caps become 1.481786, 0.772478, 0.299606. DERIVED.
C8. mixed-2p0 artifact 2.0855934079220506 is above 1.5 (KILL for the complete target). mixed-sub15 1.291078193006250 is under 1.5 and MEASURED incoherent. MEASURED.

EVIDENCE
E1. BF16 header census command output, this lane (class elements table in §4). Path: .../qwen38-27b/bf16/model.safetensors.index.json + 11 shard headers.
E2. crates/hawking-core/src/model/qwen38_geometry.rs:20-52.
E3. crates/hawking-core/src/model/qwen38_pack.rs:673-679, 442-463, 104-107, 474-664.
E4. crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:47-48, 195-273; uniform_q4.rs:15-18; mod.rs:41 COMPLETE_BINARY_HEADER_BYTES=32.
E5. .../uniform-q4-v1/manifest.json fields listed in §6; per-class byte sum this lane.
E6. .../mixed-2p0-v1/PACK_REPORT.json complete_physical_bpw 2.0855934079220506, organ_breakdown.
E7. .../mixed-sub15-v1/PACK_REPORT.json complete_physical_bpw 1.2910781930062503, per_tensor_class.
E8. git show HEAD:receipts/ascent-2026-08-16/G016_BPW_FEASIBILITY.json geometry + scenarios.
E9. git show HEAD:receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json RESULT INCOHERENT.
E10. git show HEAD:receipts/QWEN80_BIT_BUDGET_LEDGER.json identity 0.97032*expert+0.02968*nonexpert.
E11. Inversion command, this lane: rem_bits = 1.5*N - E_mlp*b_mlp - 8*1350860880 - 8*10584840; b_attn = rem_bits/7237795840.

CHANGES
created workspace/superwave/g1/g1-bit-budget-accounting.md
no tracked file modified

TESTS
test -s workspace/superwave/g1/g1-bit-budget-accounting.md
wc -l workspace/superwave/g1/g1-bit-budget-accounting.md
git status --porcelain

RISKS
R1. Defining "attention" as GEMV-only vs GEMV+small changes the third decimal, not the kill band.
R2. Catalog tax (+5.5e-5 BPW) is outside pack.rs complete. Using artifact-complete does not flip any cell in §8.
R3. REACH cells are arithmetic. sub15 is the only packed <1.5 point and it is incoherent.
R4. Q4K/AWQ/strand physical BPW not instantiated; they still cannot beat the identity.

UNRESOLVED
U1. Cheapest attention codec that is coherent in the open band (1.29, 2.18) at MLP 0.8 + tables Q4. Not this lane.
U2. Whether tables can go below Q4 without a coherence cliff. Not this lane.
U3. G0 TPS / TOKEN_NS not re-measured (GPU lane).

NEXT
N1. Any G1 representation proposal must land in a `<` cell of §8 and name physical, not nominal, BPW.
N2. Do not spend a pack on KILL 1–6.
N3. If a proposal needs Q4K/AWQ/strand, measure 8*payload/n on one 17408×5120 before claiming a cell.
```
