# G1-ARCH-Q80 — transferable mechanism from the dead Q80 campaign

Vehicle: Qwen3-Coder-Next (`qwen3_next`, 48 layers, 512 experts, top-10, hidden 2048, moe_intermediate 512). Sealed as Odyssey reference; weights deleted. Do not resurrect.

Target genome: Qwen3.8-27B (`qwen3_5`, **dense**, 64 layers, hidden 5120, intermediate 17408, hybrid 48 linear-attn + 16 GQA). Receipt: `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json` field `text.DENSE_NOT_MOE`.

Every number below is labeled **measured** (receipt field), **design** (screen / identity arithmetic, not a packed or generated token), or **constructed** (arithmetic from named constants). Component microbenchmarks are not token claims.

Binding (standing, not a new measurement): a low-BPW mechanism that expands to float/Q4 then generic GEMV is rejected unless a complete-token measurement proves a net win. Preferred shape is a representation-specific Metal kernel consuming the packed codes.

---

## 1. Mixed per-component representation (~1.43 complete BPW)

### Mechanism

One codec family does not win on gate / up / down. The working recipe assigns a different family to each routed expert organ and leaves the sensitive ~3% at 8-bit.

| Organ | Codec | Magic | Design expert BPW | What is stored |
|---|---|---|---|---|
| routed `gate_proj` `[512,2048]` | `binary_group` g=128, fp16 mean-abs scale, 1-bit sign | `HGRAVB01` | 1.1269 | per-group scale + packed signs; no activation fit |
| routed `up_proj` `[512,2048]` | binary + `rice_q1_rms` residual @ 2% outliers | `HGRAVR02` | 1.2918 | binary base + Rice-coded index deltas + 1-bit sign × stored RMS |
| routed `down_proj` `[2048,512]` | `hgravs01_r160_b3` activation-weighted SVD, 3-bit factors g=64 | `HGRAVS01` | 1.27 (design) / 1.28587 (packed) | `W ≈ L[out,160] @ R[160,in]`; decode `y = L @ (R @ x)`; never densify W |
| non-expert ~3% | uniform Q8 g=64 | `HGRAVU01` | 8.0 design / 8.2506 packed | embed, lm_head, attention, DeltaNet, norms, router, **shared expert** |

Identity (design, source geometry):

```
complete_bpw = 0.97032 * expert_bpw + 0.02968 * nonexpert_bpw
```

Mass fractions **measured** from Q80 source inventory: `f_routed_expert = 0.9703169371044981`, `f_non_expert = 0.029683062895501933` (`receipts/QWEN80_BIT_BUDGET_LEDGER.json` `mass_fractions`). That 97% is unread-at-batch-1 expert storage, not traffic.

### Numbers

**Design screen** (`receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`, status `SCREEN_PASSED_NOT_YET_PACKED_OR_GENERATED`):

- `mixed_expert_bpw`: **1.22957** (design)
- `complete_bpw.nonexpert_8bit`: **1.43051** (design)
- `complete_bpw.nonexpert_6bit`: 1.37115 (design)
- `complete_bpw.nonexpert_4bit`: 1.31179 (design)
- organ cosine bar: 0.8604 (D23 residual-identity break-even; **not** a capability certificate)
- `claim_boundary.artifact_packed`: false in this receipt
- `claim_boundary.coherence_generation_tested`: false in this receipt

**Packed artifact** (`receipts/ascent-2026-08-16/Q80_PACK.json` `on_disk_ledger`, label DIRTY_ENGINEERING):

- `all_required_execute_bytes`: 14,385,668,506
- `source_weight_elements`: 79,674,391,296
- `complete_physical_bpw`: **1.44445** (measured on-disk bytes / elements)
- `payload_only_physical_bpw`: 1.44313
- `expert_physical_bpw`: 1.2348805110280712
- `nonexpert_physical_bpw`: 8.250600705299505 (fp16 scale / 64 → 8.25 nominal)
- routed gate 1.126922607421875, up 1.2918486054986715, down **1.2858703201636672** (HGRAVS01 JSON header billed; 0.016 above the 1.27 design figure)

**Controller-verified fuse run** used the packed vehicle at `complete_physical_bpw` **1.4444457** (`receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_FUSE.json`).

### Evidence

`receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`:

```json
"identity": "complete_bpw = 0.97032*expert_bpw + 0.02968*nonexpert_bpw",
"components": {
  "gate_proj": { "codec": "binary_group", "expert_bpw": 1.1269, "cosine_range": [0.8586, 0.8932], "clears_bar": true },
  "up_proj":   { "codec": "binary + rice_q1_rms sparse residual @2% outliers", "bits_per_outlier": 8.24, "expert_bpw": 1.2918, "cosine_range": [0.86416, 0.86524], "clears_bar": true },
  "down_proj": { "codec": "hgravs01_r160_b3 activation-weighted low-rank", "expert_bpw": 1.27, "cosine_range": [0.8862, 0.8978], "clears_bar": true,
                 "note": "scored on post-SwiGLU intermediate; first time down_proj was measurable" }
},
"mixed_expert_bpw": 1.22957,
"complete_bpw": { "nonexpert_8bit": 1.43051, "nonexpert_6bit": 1.37115, "nonexpert_4bit": 1.31179 }
```

Physical contract: `docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md` recipe table (gate `HGRAVB01`, up `HGRAVR02`, down `HGRAVS01` fit on post-SwiGLU, non-expert `HGRAVU01`).

Frontier that justified **per-component** (not the r160 cosine): `receipts/QWEN80_REPRESENTATION_FRONTIER_SWEEP.json`

| organ | codec | cosine (measured) | expert_bpw | verdict |
|---|---|---|---|---|
| L10E453 gate | `binary_g` | 0.8932464137407012 | 1.12692 | PASS |
| L10E453 up | `binary_g` | 0.8275162668981674 | 1.12692 | fail |
| L10E453 up | `binary+resid_2pct` | 0.865143437435382 | 2.08814 | over-budget (legacy 48-bit outliers) |
| L3E494 gate | `binary_g` | 0.8585935762823004 | 1.12692 | **fail** (below 0.8604) |
| L3E494 up | `binary_g` | 0.8258422168325341 | 1.12692 | fail |

NS-012 (`receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json`): single-family across gate/up/down is **INSUFFICIENT**.

### Caveats that travel with the number

1. The 1.43051 figure is a **design screen**, not a packed complete-physical BPW. Packed is **1.44445**.
2. Cross-adversarial P1-MIXED-CLEARS-BAR-FALSE: the mixed receipt sets `gate_proj.clears_bar: true` while citing cosine_range whose low end is L3E494 `binary_g` **0.858594**, which the frontier marks FAIL. Organ cosine is a screen; this screen does not pass on both cited gate organs.
3. Cross-adversarial P1-DOWN-R160-UNSOURCED: `hgravs01_r160_b3` at 1.27 / cosine [0.8862, 0.8978] **does not appear** in `QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json` (that file has r192 @ 1.5368 BPW over-budget and r256 @ 2.045). The exact 0.886230… value appears in `QWEN80_RESIDUAL_ENCODING.json` as `rice_q4_absmax` on **gate_proj** L3E494, not as a down_proj r160 score. Treat r160/1.27/0.886–0.898 as a **pack-lane contract**, not as a measured operating point.
4. Standing null for raw activation cosine is **0.898**. The 0.8604 bar sits **below** that null (`_Q80_DENSITY_COMMON.md`; NS-013). Generation is the gate.
5. `Q80_MIXED_GENERATE.json` generated English on mixed-1p5-v1 (`Here’s a function that reverses a string`) and graded against the **artifact oracle**, not the BF16 parent. Continuation diverges from the Q4 vehicle (748 vs 594). `coherence_generation_tested` against teacher is not a GO.

### Transfer to Qwen3.8 dense

| Piece | Transfers? | Why |
|---|---|---|
| Per-component codec assignment (gate ≠ up ≠ down) | **YES — method** | Qwen3.8 has the same three SwiGLU organs on every layer (`QWEN38_ARCH_CENSUS.json` `tensor_naming.mlp_64x`). NS-012 is a ranking law, not an MoE law. |
| Identity `0.97*expert + 0.03*nonexpert` and the 1.43 complete-BPW arithmetic | **NO — MoE-specific** | 97.03% of Q80 **storage** is 512 routed experts, of which batch-1 decode reads 10. Qwen3.8 is dense: every MLP/attn/lm_head weight except one embed row is traffic (`QWEN38_ACTIVE_BUDGET_MEASURED.json` `embed_excluded_why`). Qwen3.8 activates **23.611 G** weights/token vs Q80 **3.562 G** (`QWEN38_ARCH_CENSUS.json` `DENSITY_VS_TPS_ARITHMETIC`). There is no unread-expert subsidy. |
| Leave a “sensitive 3%” at 8-bit while crushing the rest | **NO as a budget trick** | On Qwen3.8 the MLP is the mass (17.11 G of 23.61 G). Crushing a 3% tail does not buy a 1.5 complete BPW. |
| Packed 1.44445 as a Qwen3.8 target | **do not copy the number** | Different geometry, different active set. A Qwen3.8 complete-physical BPW must be re-ledgered from its own tensors. |
| Native per-codec kernel (no expand-to-float) | **YES — binding** | Same law. Qwen3.8 is already bandwidth-bound at ~406 GB/s of a 411.51 unique-once ceiling (`QWEN38_BANDWIDTH_BOUND.json`); expansion would add bytes. |

`QWEN80_BIT_BUDGET_LEDGER.json` `structural_note` (quoted): “97.03%% of Q80 mass is routed experts, so expert representation almost entirely determines complete BPW. This is the subsidy a dense model cannot offer.”

**KILLS** if G1 treats 1.43 as a Qwen3.8 complete-BPW that has already been demonstrated. **REOPEN_IF** a Qwen3.8 per-organ screen on real post-norm / post-SwiGLU X produces a mixed recipe whose **on-disk complete-physical** BPW is remeasured and then executed by native kernels.

---

## 2. Rice + one-bit outlier

### Mechanism

Same operation as the incumbent residual: binary sign/scale base plus a sparse additive correction at the **same** global top-k positions by `|W − binary(W)|`. Only the **storage** of those positions and values changes.

Index (`lab/operators/residual_compact_codec.py`):

- sort selected flat indices
- store `uint32 first_index`
- Rice-code **positive deltas** (`k` chosen by `_best_rice_k`: unary quotient LSB-first, terminator 0, then `k` remainder LSBs)

Value at 1-bit / `rms`:

- one fp16 scale = RMS of the selected residual values
- 1-bit packed signs; reconstruction `sign * stored_rms_scale`
- a selected outlier is never a zero code

Bits-per-outlier definition (receipt field, not folklore):

```
8 * (codec_payload_bytes - binary_payload_bytes) / outlier_count
```

payload includes binary base, container header, and residual body (`receipts/QWEN80_RESIDUAL_ENCODING.json` `bits_per_outlier_definition`).

### Numbers (measured, two busy up_proj organs, 2% outliers, 20,972 positions, 1,048,576 elements)

Chosen operating point `rice_q1_rms` @ 0.02, 8-bit non-expert (`QWEN80_RESIDUAL_ENCODING.json` `verdict.chosen` / `findings.chosen_operating_point`):

| organ | bits_per_outlier | residual_body_bits_per_outlier | expert_bpw | cosine | Δcosine vs fp16 residual |
|---|---|---|---|---|---|
| L10E453 up | **8.244135037192446** | 8.12283044058745 | 1.29180908203125 | 0.8641625485550589 | −0.0009808888803231053 |
| L3E494 up | **8.245279420179287** | 8.12397482357429 | 1.2918319702148438 | 0.8652372408594525 | −0.0012269222802221424 |

Index-only floor at 2% (fp16 values still stored) — `findings.index_only_floor_bits_per_outlier_at_2pct`:

- `rice_fp16`: **23.231356093839405**
- `group_local_fp16`: 24.70150677093267
- `bitmap_fp16`: 63.51363723059317

So Rice is the cheapest lossless-index packing and **still cannot hit 12 bits** while storing fp16 values. Twelve bits/outlier is reachable only by quantizing the residual **value**. 1.5% outliers does **not** clear 0.8604 on up_proj even with fp16 values (`up_proj_1p5pct_min_cosine_fp16`: 0.8588867552073517). Per-group bitmaps lose at 2% density.

Incumbent comparison (same L10E453 gate, 0.25% for scale): `legacy_u32_fp16` 48.48 bits/outlier vs `rice_fp16` 26.99 vs `rice_q1_rms` ~8.24 at 2%.

### Token-path constraint (measured, Q80 mixed decode)

Serial per-token Rice bitstream expand: **15,597,000 ns / organ** — rejected. Token path is bind-time expand + CSR apply at **78–82 µs** (`receipts/ascent-2026-08-16/q80-decode-kernels.json`; NS-031).

Fusing binary+CSR into one occupancy-tile dispatch deleted **480** residual kernels/token (`Q80_RECONSTRUCTION_FUSE.json` `dispatches_per_token` 2893 → 2413).

### Evidence

`receipts/QWEN80_RESIDUAL_ENCODING.json` `verdict.statement`:

> up_proj clears 0.8604 inside the 1.5 complete-BPW ceiling at encoding=rice_q1_rms, outlier_frac=0.02, nonexpert_bits=8 (expert_bpw=1.2918/1.2918, cosine=0.864163/0.865237, bits_per_outlier=8.2441/8.2453).

Code: `lab/operators/residual_compact_codec.py` `_pack_rice_indices`, `_quantize_residual_values` (`value_bits == 1`, `codebook: sign_times_stored_scale`), magic `HGRAVR02`. Format: `docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md` § `HGRAVR02`.

### Transfer to Qwen3.8 dense

| Piece | Transfers? | Why |
|---|---|---|
| Binary + sparse residual as a family | **YES — method** | Independent of MoE. Selection and reconstruction are weight-space. |
| Rice on sorted index deltas | **YES — method** | Same. |
| 8.24 bits/outlier | **NO as a number** | Function of outlier density, matrix length (Q80 expert 2^20 elements), and header amortization. Qwen3.8 gate is `[17408, 5120]` (recon-measured `rows=17408, cols=5120`). Remeasure. |
| 2% as the operating fraction | **NO as a number** | 1.5% failed the Q80 up_proj bar; 2% cleared. Qwen3.8 organs need their own curve. |
| Serial Rice on the token path | **KILLS** | NS-031. 15.6 ms/organ. Bind-time expand + in-register CSR apply is the path. |
| Reconstruction time of rice_q1 CSR-in-register at tpr64 | **already remeasured on Qwen3.8** | `rice_q1_rms_2pct/csr_inregister` median **15,125 ns** on gate vs f32 control **15,125 ns** (`QWEN38_RECON_MEASURED.json`). Component microbenchmark, not a token. |

**REOPEN_IF** a Qwen3.8 residual sweep on real X reports bits/outlier, cosine vs a stated null, and a native CSR-in-register kernel on the complete token.

---

## 3. down_proj inverts codec ranking and needs post-activation X

### Mechanism

`down_proj` is `y = W @ silu(x @ W_gate.T) * (x @ W_up.T)`. Its input is the **post-SwiGLU intermediate** of width 512, not the 2048-d layer hidden / router input. Scoring it on hidden X is the wrong operator.

On that X, the gate/up ranking **inverts**: `binary_g` (the in-budget winner on busy gate) **fails** the 0.8604 bar; activation-weighted low-rank **clears** it.

SwiGLU formula verified on the same pairs used for the down sweep (`receipts/QWEN80_SWIGLU_INTERMEDIATE_VERIFY.json`): three pairs, cosine ≥ 0.999999999997 against `silu(X @ W_gate.T) * (X @ W_up.T)`, `hidden_act: silu`, `x_kind: swiglu_hidden_routed`, width 512.

### Numbers (measured, `receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json`)

Bar 0.8604. Expert allowances: 1.30116 (8-bit non-expert) / 1.42352 (4-bit). Four organs, post-SwiGLU X.

L1E265 down (`W` `[2048,512]`, `X` `[341,512]`, `n_fit_rows=341`):

| codec | cosine | expert_bpw | clears_bar | vs 1.3012 |
|---|---|---|---|---|
| `binary_g` | **0.8264830117545535** | 1.12692 | false | fail |
| `binary+resid_0.25pct` | 0.83730 | 1.24815 | false | fail |
| `binary+resid_1pct` | 0.85501 | 1.60813 | false | fail |
| `binary+resid_1.5pct` | 0.86350 | 1.84814 | true | over-budget |
| `uniform_b2` | 0.81649 | 2.25207 | false | fail |
| `hgravs01_r192_b3` | **0.9132099561676162** | **1.53677** | true | over-budget |
| `hgravs01_r256_b3` | 0.92093 | 2.04459 | true | over-budget |

Same inversion on L32E179, L46E428, L35E330: `binary_g` cosine 0.806–0.813 (fail); `hgravs01_r192_b3` cosine 0.874–0.918 (clears, over-budget).

**Zero** candidates in this file both clear 0.8604 **and** fit the 1.3012 8-bit-nonexpert allowance. That is the budget hole r160 was invented to fill — without a measured r160 row in this receipt.

Compare gate on the same campaign (`QWEN80_REPRESENTATION_FRONTIER_SWEEP.json` L10E453): `binary_g` cosine **0.89325** PASS at 1.1269 BPW. That is the inversion: binary wins on gate, loses on down.

### Evidence

- `receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json` `activation_provenance.x_kind = "swiglu_hidden_routed"`, `packed_swiglu: true`, `swiglu_width: 512`
- `receipts/QWEN80_SWIGLU_INTERMEDIATE_VERIFY.json` pairs
- `docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md`: “fit on **post-SwiGLU** `silu(X@G.T)*(X@U.T)`, never layer hidden”
- NS-012 `what_was_measured.down_proj`
- `Q80_SEALED_LOSER_SCIENCE_RETAINED.json` `science_preserved_elsewhere.negative_science[1]`
- Historical skip: `lab/operators/q80_representation_frontier_sweep.py` (down skipped because input is post-SwiGLU 512, not captured 2048 hidden)

### Transfer to Qwen3.8 dense

| Piece | Transfers? | Why |
|---|---|---|
| Score down_proj on post-SwiGLU X, not hidden | **YES** | Qwen3.8 MLP is dense SwiGLU, `hidden_act: silu` (`QWEN38_ARCH_CENSUS.json` `text.hidden_act`). Same operator. Width is 17408, not 512. |
| “low-rank beats binary on down” | **remeasure** | Measured on Q80 routed `[2048,512]` with sparse expert X (277–341 rows). Qwen3.8 down is `[5120, 17408]` (`QWEN38_RECON_MEASURED.json` organ `down`). Ranking can invert again. |
| `hgravs01_r160_b3` as the Qwen3.8 down codec | **do not copy** | r160 was an unsourced interpolation on Q80. On Qwen3.8 down, `hgravs01_r160_q3` median GPU **71,458 ns** vs f32 tpr64 **7,083 ns** — **~10× slower**, two-stage algebra (`QWEN38_RECON_MEASURED.json` variant `hgravs01_r160_q3`, note “two-stage algebra, never reconstructs W”). That is a component microbenchmark: low-rank at r160 is **not free** on this dense shape. |
| Expert-routing sparsity of X | **NO** | Dense down sees every token. Capture is one sequence of hiddens, not 24,576 (layer, expert) reservoirs. |

**KILLS** transferring Q80’s r160 down kernel as a Qwen3.8 velocity win. **REOPEN_IF** a Qwen3.8 down sweep on real post-SwiGLU X names a codec that clears a stated null **and** a native kernel that is not slower than the tpr64 f32 control on that shape.

---

## 4. Ceremony eliminations: serial extract 867 → 36.6 ms; command buffers 337 → 49

These are **two different wins** (plus two more). None is a compute/codec change.

### 4a. Serial 1-thread-per-row extract → in-register occupancy tiles

**Mechanism.** Shipping path was still dispatching a serial one-thread-per-row extract doing 8 bit-loads per weight. Occupancy tiles that consume codes in-register already existed. Wiring them onto the token path (binary tg256, 8 signs from one byte, Q8 one-byte load + `(code-127)*scale` FMA, binary+CSR fused, HGRAVS 3-bit simd3 unpack) replaced the extract. Codecs unchanged.

**Measured** (`receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_WON.json`, controller-run under `tools/gpu_lane_lock.sh`):

| arm | gpu_matvec_ns | wall_ns_per_token | tok/s |
|---|---|---|---|
| base (serial extract) | **867,040,696** | 1,376,263,011 | 0.726605 |
| ours (occupancy tiles) | **36,598,269** | 301,418,659 | 3.317645 |

- gpu_matvec speedup **23.7×** (component of the token, GPU timestamps)
- wall speedup **4.57×** (complete-token wall, same generate)
- greedy ids identical; silent fallbacks 0
- after: gpu_matvec is 11% of the token (was 74%)

Paired 3-rep fuse receipt (`Q80_RECONSTRUCTION_FUSE.json`, DIRTY_ENGINEERING, same vehicle mixed-1p5-v1):

- base gpu_matvec median **863.725 ms** `[862.975, 864.581]`
- ours gpu_matvec median **36.218 ms** `[35.903, 36.630]`
- speedup_x **23.848**
- CBs still **337** on both arms (this win is not the CB collapse)
- achieved GB/s 2.57 → **61.24** of 411.51 honest unique-once ceiling
- 480 residual dispatches deleted

The rounded slogan “867.0 → 36.6 ms” is the controller-verified pair in `Q80_RECONSTRUCTION_WON.json` / `Q80_FOUR_WINS_LANDED.json`. The cleaner paired-rep median is **863.7 → 36.2 ms**.

Independent pre-result diagnosis (`Q80_RECONSTRUCTION_WON.json` `independent_diagnosis`): qkvz x-only 138 µs, load-only 175, f32 serial 250, byte-serial 287, shipped 1586 — ALU busy on the extract, not stalled on DRAM.

### 4b. Command buffers 337 → 49

**Mechanism.** Keep RMSNorm, DeltaNet conv/recurrent, GQA rope+attn, SwiGLU, residual add, and the shared/combine tail **device-resident in the same `MTLCommandBuffer` as the mixed GEMVs**. Fuse `suffix_i + prefix_{i+1}`. The remaining split is the **512-logit router readback** that binds ten mixed expert payloads.

**Measured** (`receipts/ascent-2026-08-16/G003_Q80_CB_COLLAPSE.json`):

| field | value | label |
|---|---|---|
| `cbs_per_token_was` | **337** | host-activation path (`HAWKING_Q80_DEVICE_ACTIVATIONS=0`) |
| `cbs_per_token` | **49** | fuse path |
| topology | 1 × L0 mixer+prefix, 47 × (suffix_i + mixer/prefix_{i+1}), 1 × (last suffix + terminal RMS + lm_head) | constructed from 48 layers |
| opt-out | 97 CBs if collapse_fuse=0 (prefix+suffix+terminal) | |
| `median_wait_minus_gpu_ns` | 15,864,347 (**15.864 ms**) | measured, GPU timestamps |
| paired host-activation wait−gpu | 78.684 ms | measured |
| G003-closed wait−gpu | 83.375 ms | measured |
| recovered vs paired base | 62.819 ms | derived |
| `dispatches_per_token` | 2834 (was 2413) | GPU now covers work previously host-exclusive |
| greedy ids | `[8420, 748, 264, 729, 429, 17431, 288, 264, 914, 320, 72, 1734]` bit-identical | measured |
| silent fallbacks | 0 | measured |
| median wall | 156.989 ms | DIRTY_ENGINEERING, decode-only, 11 steady steps |

`Q80_FOUR_WINS_LANDED.json` slogans the wait as 83.4 → 15.9 ms (G003-closed vs ours).

### 4c. The other two ceremony wins (same four-win set)

`Q80_FOUR_WINS_LANDED.json` `all_four_merged_and_verified`:

2. catalog re-parse: `host_preparation` **84.79 → 25.0 ms**; geometry cached at first touch; 289 `packed()` calls/token eliminated
4. DeltaNet host recurrence: **42.376 → 4.449 ms**; wall 160.479 → 111.733 ms, 6/6 pairs

Complete-token trajectory (same receipt, **not** the same run as 4a’s 301 ms — later genome):

- morning serial reconstruction: **1376.3 ms**/token
- after reconstruction: 301.4
- after catalog + CB: 160.5
- after DeltaNet: 111.7
- sealed-science 12.7× figure: **1376.3 → 108.306 ms** (`Q80_SEALED_LOSER_SCIENCE_RETAINED.json` `the_12_7x_result`; also `CORRECTION_ROOF_IS_CONDITIONED.json`)

108.306 ms is a **complete-token wall**, not gpu_matvec. Do not convert 36.6 ms → 27 TPS (that fallacy is C5 in the genesis tournament).

### Evidence

- `receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_WON.json`
- `receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_FUSE.json` `gpu_matvec_ns_per_token`, `mechanisms.won`
- `receipts/ascent-2026-08-16/G003_Q80_CB_COLLAPSE.json` `measurement.cbs_per_token`, `synchronization`
- `receipts/ascent-2026-08-16/Q80_FOUR_WINS_LANDED.json`
- `receipts/ascent-2026-08-16/Q80_SEALED_LOSER_SCIENCE_RETAINED.json` `the_12_7x_result`
- `receipts/ascent-2026-08-16/CORRECTION_ROOF_IS_CONDITIONED.json`: “Q80 went 1376.3 -> 108.3 ms today, 12.7x, and BPW was never the lever. All four eliminations were execution-genome changes.”

### Transfer to Qwen3.8 dense

| Piece | Transfers? | Why |
|---|---|---|
| In-register consumption; never ship a 1-thread-per-row bit-walk | **YES — already remeasured** | Qwen3.8 tpr64 lands at f32 speed (finding 5). The 867→36.6 number is Q80 mixed-organ shaped; do not quote it as a Qwen3.8 delta. |
| Occupancy-tile **tg256** as the Qwen3.8 launch | **KILLS as a copy** | On Qwen3.8 gate, tg256 median **26,541 ns** vs tpr64 **15,125 ns** (`QWEN38_RECON_MEASURED.json`). Q80-won geometry is the wrong geometry here (`G002_Q80_HOT_PATH_RESOLUTION.json`). |
| Cache catalog geometry at first touch; do not reparse per token | **YES** | Genome-general. Immutable-identity recomputation has been the wall on every Hawking vehicle. |
| Collapse host/device round-trips; keep activations on device | **YES — method** | |
| 337 → 49 | **NO as a number; MoE-specific topology** | 337 is the host-activation Q80 path (48 layers × host RMS/DeltaNet/GQA + GEMV CBs). 49 remains because **router logits must return to the host** to bind 10 of 512 mixed expert buffers (`G003_Q80_CB_COLLAPSE.json` `remaining_cb_split_readbacks`). Qwen3.8 has **no router**. Census: “the entire Q80 expert machinery — moe_table_build, expert residency, device top-k, expert address tables, first-touch upload — is NOT NEEDED.” Qwen3.8 already sits in a 1-CB band (`G003` note: “wait−gpu cannot fall to the Qwen38 0.384 ms (1 CB) band”). |
| DeltaNet host → GPU | **YES — parametric reuse** | Qwen3.8 linear_attn tensor names match Q80 gated-DeltaNet; only projection fusion differs (`QWEN38_ARCH_CENSUS.json` `DECISIVE_REUSE_FINDING`). |

**KILLS** any G1 plan that spends time on Q80 expert_bind / moe_table_build / 10-of-512 gather as if they existed on Qwen3.8. **REOPEN_IF** a Qwen3.8 generate still opens more than one CB per decode step — then the fuse method applies, with a different split.

---

## 5. Reconstruction is free at one threadgroup geometry

### Mechanism

Once codes are consumed in-register at a launch that already saturates the organ, the codec is not a time term. The penalty that was blamed on rice/binary/low-rank was **launch geometry** (serial 1-thread-per-row, or the wrong threads-per-row), not density.

Two measurements, opposite directions, same conclusion.

### 5a. Qwen3.8 — already the target architecture (component microbenchmark)

`receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json`:

- activation: **real** captured BF16 post-norm hidden, token 192 of a 256-token holdout
- organs: 2 (gate, down)
- variants: **33**
- GPU authority: `MTLCommandBuffer.GPUEndTime-GPUStartTime` after wait
- f32 control tpr64: gate **15,125 ns**, down **7,083 ns**
- codecs at tpr64: **15,124–15,541 ns** for q4/q3/q2/binary/ternary/additive_q2q2/hadamard/rice — “the SAME as uncompressed f32”
- `recon_excess_ns_zero_on`: **32 of 33**
- same codecs at tg256: **~26,500 ns** — penalty is launch geometry

Primary table: `receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json`  
`launch_primary`: “64 threads/row, TG 128, 2 rows/TG (production Qwen3.8 q4 winner)”

Gate `[17408, 5120]` tpr64 medians (measured, 5 GPU reps):

| variant | median_gpu_ns |
|---|---|
| `f32_tpr64` | **15125** |
| `prod_q4_nibble_g64` | 15500 |
| `uniform_q4_g64/disc_uniform_bits_tpr64` | 15208 |
| `uniform_q3_g64/disc_uniform_bits_tpr64` | 15125 |
| `uniform_q2_g64/disc_uniform_bits_tpr64` | 15374 |
| `binary_g128/disc_binary_tpr64` | 15416 |
| `ternary_t0.7_g128` | 15541 |
| `additive_q2q2_g64` | 15125 |
| `rice_q1_rms_2pct/csr_inregister` | **15125** |
| `hadamard_q2_g128` | **17333** (the 33rd; WH is O(cols log 128); this is the excess) |
| `uniform_q4_g64/disc_uniform_bits_tg256` | **26541** |
| `hgravs01_r160_q3` (down only) | **71458** vs down f32 7083 — not free |

This is an **isolated-organ** measurement, not a token.

### 5b. Q80 corroboration (complete-token gpu_matvec)

Same conclusion from the other direction: in-register occupancy tiles took gpu_matvec **867.0 → 36.6 ms without changing a single codec** (`QWEN38_RECONSTRUCTION_IS_FREE.json` `corroboration`; finding 4a).

`Q80_RECONSTRUCTION_WON.json` `CORRECTIONS_TO_MY_EARLIER_RECORD`: “I recorded ‘density is costing speed — mixed is 5.9x slower per byte than Q4 because binary/rice/low-rank reconstruction is expensive’. The codecs were never the cause.”

NS-006 (density is velocity / 5.9× reconstruction penalty) is **REFUTED** once reconstruction is in-register. The 5.9× figure compared mixed serial extract to a **different** Q4 simdgroup kernel.

### Evidence

- `receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json`
- `receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json` `organs[].variants[].median_gpu_ns`
- `receipts/ascent-2026-08-16/G002_Q80_HOT_PATH_RESOLUTION.json`: “at tpr64 every codec runs at the uncompressed f32 speed, while the same codecs at tg256 cost 26,500 ns against 15,125. Launch geometry is the shared lesson.”
- `receipts/ascent-2026-08-16/Q80_SEALED_LOSER_SCIENCE_RETAINED.json` `reconstruction_is_free` — **mis-cites** `Q80_RECONSTRUCTION_WON.json` for the 15,124–15,541 ns table. That table lives in `QWEN38_RECONSTRUCTION_IS_FREE.json` / `QWEN38_RECON_MEASURED.json`. The Q80 receipt is the 867→36.6 corroboration only.

### Transfer to Qwen3.8 dense

**Already transferred and remeasured on Qwen3.8.** Do not re-derive.

Consequences for G1:

- Codec selection under 1.5 BPW may select on **quality**, not reconstruction time, **at tpr64, for bit-packed families** (q2/q3/q4/binary/ternary/additive/rice-CSR).
- Exceptions that are **not** free on Qwen3.8 shapes (component, measured):
  - `hadamard_q2_g128` gate 17,333 ns (WH overhead)
  - `hgravs01_r160_q3` down 71,458 ns (two-stage)
  - any tg256 launch (~26.5 µs gate)
- A roof built from “bytes / 411.51 GB/s” is still a **conditioned** roof (`CORRECTION_ROOF_IS_CONDITIONED.json`). Q80 already falsified “density costs speed” as a ceiling. Qwen3.8 being at 98.7% of the unique-once control means **bytes** are the remaining lever, not kernels — provided the kernel stays at the free geometry.

**KILLS** a G1 plan that rejects rice/binary/q3 because of the retired 5.9× Q80 serial-extract penalty. **KILLS** a plan that ports Q80 tg256 occupancy tiles onto Qwen3.8 production. **REOPEN_IF** a new codec family (or a fused two-stage low-rank) is slower than the tpr64 f32 control on the real Qwen3.8 organ shapes.

---

## 6. `rank = min(budget, n_fit_rows)` made the working rank unreachable by construction

### Mechanism

Two independent clamps composed:

1. **Codec law** (`lab/operators/hgravs01_adapter.py` `clamp_rank`):
   ```python
   def clamp_rank(budget_rank: int, n_fit_rows: int) -> int:
       """rank = min(budget_rank, n_fit_rows) — same law as the repack operator."""
       return min(int(budget_rank), int(n_fit_rows))
   ```
   A requested r192 fit on 85 rows is billed and scored as **r85**. The score is not the codec’s score (NS-014).

2. **Capture-budget law** (retired ×48 guard). Streamed RSS hard cap 16 GiB held **all 48 layers** of retained hidden at once:

```
N = floor( 16 GiB / (QWEN80_EXPERTS × QWEN80_LAYERS × QWEN80_HIDDEN × 4) )
  = floor( 17179869184 / (512 × 48 × 2048 × 4) )
  = floor( 17179869184 / 201326592 )
  = 85
```

**Constructed** from named constants in `crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs`:

- `QWEN80_LAYERS = 48` (line 44)
- `QWEN80_HIDDEN = 2048` (line 45)
- `QWEN80_EXPERTS = 512` (line 49)
- `STREAMED_PEAK_RSS_HARD_CAP_BYTES = 16 << 30` (line 58)

Test `per_layer_budget_guard_admits_384_and_512_and_refuses_over_cap` (same file, ~3894):

```rust
let before_layers_factor_ceiling = cap
    / (QWEN80_EXPERTS.saturating_mul(QWEN80_LAYERS)
        .saturating_mul(QWEN80_HIDDEN).saturating_mul(4));
// The old ×48 guard capped N at 85. Per-layer must raise that ceiling.
```

Python check of that formula: `old x48 N 85 exact 85.333…`. Then `rank = min(192, N≤85) = 85`. r192 is unreachable **by construction**, not by quality.

Prior campaign also reported median **92 rows against 2048 dims** (standing law in `workspace/ops/ascent-lanes/q80-capture-coverage.md` and `_Q80_DENSITY_COMMON.md`). 92 < 2048 ⇒ every full-dim score garbage. I did not find a separate sealed JSON whose field is exactly `median_rows: 92`; the 92 is standing-law text, the 85 is reconstructed from the constants the test names.

Later capture flushed per layer, so the cap became `16 GiB / (512 × 2048 × 4) = 4096` and N=512 fits. Later **pack** policy **un-clamped** rank: `docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md` “Rank is **not** clamped to `n_fit_rows`”; ridge-regularized 512×512 Gram at requested 160. `Q80_PACK.json` `down_proj_fit.rank_clamped_to_n_fit: false`, rank 160 on every down_proj (221 never-routed experts use weight-space SVD, reported).

The coherence probe still clamped: **395 / 2048** organs `rank = min(160, rows)` (`Q80_COHERENCE_LAYER_DRIFT_PROBE.json` `analysis.reconstruction.hgravs_rank_clamped`; CROSS_ADVERSARIAL P1-COHERENCE; NS-014 / NS-017).

25258-token capture census (`receipts/ascent-2026-08-16/Q80_CAPTURE_COVERAGE.json`, DIRTY_ENGINEERING): p50 retained rows **258**, 24326/24576 gate/up pairs rows < 2048, 221 never-routed, 10041/24576 down pairs n_fit < 160 after 25% holdout. Even the un-clamped later capture is underdetermined for a 2048-d Gram on typical experts.

### Evidence

- `lab/operators/hgravs01_adapter.py` lines 10–12, 68–72
- `crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs` lines 44–58, 3894–3902
- `receipts/ascent-2026-08-16/Q80_SEALED_LOSER_SCIENCE_RETAINED.json`: “rank = min(budget, n_fit_rows) with a per-layer capture budget caps N <= 85, so r192 is unreachable BY CONSTRUCTION”
- `workspace/ops/ascent-lanes/q80-capture-coverage.md` lines 80–84 (median 92; N<=85; r192)
- NS-014 `receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json`
- `receipts/ascent-2026-08-16/Q80_CAPTURE_COVERAGE.json` `organs[]`
- `receipts/ascent-2026-08-16/Q80_PACK.json` `down_proj_fit` (the later un-clamp)

### Transfer to Qwen3.8 dense

| Piece | Transfers? | Why |
|---|---|---|
| Never trust a rank-r score when `n_fit < r` | **YES — law** | NS-014 is model-agnostic. |
| Never implement `rank = min(budget, n_fit)` as a silent fallback | **YES — law** | Starves the codec; the score is not the requested point. Report achieved rank or refuse. |
| N ≤ 85 | **NO — MoE ×48 construction** | 85 = 16 GiB / (512 experts × 48 layers × 2048 × 4). Qwen3.8 is dense: one MLP per layer, not 512. Hidden is 5120. A 16 GiB all-layers budget would be `16 GiB / (64 × 5120 × 4) ≈ 12,800` rows/layer if you made the same mistake — not 85 — and you should flush per layer anyway. |
| r192 as “the rank that actually worked” | **Q80-specific** | Working rank on Qwen3.8 down/gate must be re-fit. r160 on Qwen3.8 down is already slow (finding 3/5). |
| Underdetermined typical expert | **MoE-specific severity** | 24,576 (layer, expert) pairs share a token budget. Dense Qwen3.8 has 64 MLPs; a short capture already gives n_fit ≫ rank. Still require `n_fit ≥ rank` and `n_fit ≥ dim` for a full-dim claim. |

**KILLS** any G1 activation-weighted SVD that clamps rank to the row count and then quotes the requested rank’s BPW/cosine. **REOPEN_IF** a Qwen3.8 capture publishes `n_fit` vs claimed rank per organ and refuses underdetermined scores.

---

## Transfer ledger (one line each)

| # | Mechanism | Number | Primary receipt | To Qwen3.8 |
|---|---|---|---|---|
| 1 | Per-component mixed recipe | design complete **1.43051** / packed **1.44445** | `QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`, `Q80_PACK.json` | method YES; 1.43 identity NO (no 97% expert subsidy) |
| 2 | Rice + 1-bit residual | **8.244–8.245 bits/outlier** @ 2% | `QWEN80_RESIDUAL_ENCODING.json` | method YES; 8.24 NO; serial expand KILLS; CSR-in-register already free at tpr64 |
| 3 | down_proj ranking invert + post-SwiGLU X | binary_g cosine **0.806–0.826** fail; r192 cosine **0.874–0.918** over-budget | `QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json`, `QWEN80_SWIGLU_INTERMEDIATE_VERIFY.json` | post-SwiGLU X YES; r160 kernel KILLS on Qwen3.8 down (71.5 vs 7.1 µs) |
| 4a | Serial extract → in-register | gpu_matvec **867.041 → 36.598 ms** | `Q80_RECONSTRUCTION_WON.json` | method YES; 867/36.6 NO; tg256 KILLS on Qwen3.8 |
| 4b | CB collapse | **337 → 49**; wait−gpu 83.4 → 15.9 ms | `G003_Q80_CB_COLLAPSE.json` | method YES; 337/49 NO (router/MoE); Qwen3.8 is 1-CB |
| 4c | Catalog cache + DeltaNet GPU | 84.79→25.0 ms; 42.376→4.449 ms | `Q80_FOUR_WINS_LANDED.json` | YES |
| 5 | Reconstruction free at one geometry | tpr64 **15,124–15,541 vs 15,125 ns**, 32/33 | `QWEN38_RECONSTRUCTION_IS_FREE.json`, `QWEN38_RECON_MEASURED.json` | already on Qwen3.8 |
| 6 | rank = min(budget, n_fit) + ×48 cap | **N ≤ 85** constructed; r192 unreachable | `qwen80_source_bf16_layer_major.rs`, `hgravs01_adapter.py` | law YES; N=85 NO |

## Negative science that must not be re-paid on G1

- Cross-expert shared basis: REFUTED. L10 96-expert pairwise cosine mean gate **0.00414**, up **−6.0e-5** (`QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json`). MoE-specific; Qwen3.8 has no expert set to share.
- Storage BPW ≠ active BPW on MoE (NS-002). On **dense** Qwen3.8 they nearly coincide (embed row excluded).
- “Lower BPW is faster” unqualified (NS-006): REFUTED on Q80 serial extract; retired once reconstruction is free.
- Organ cosine 0.86–0.90 as a capability certificate (NS-013): REFUTED. Null 0.898. Generation is the gate.
- Isolated-organ product as a token (NS-034). 192 ms = 48×10×(gate+up+down) is not a token.
- 403 ms Q80 “baseline” (NS-035): not GPU, not 0-fallback, not paired.
- Roof-as-physics from bytes/quoted-bandwidth (`CORRECTION_ROOF_IS_CONDITIONED.json`).

## What G1 should steal

1. Assign codecs **per organ** on real X (gate/up on post-norm hidden, down on post-SwiGLU). Do not pick one family.
2. If a residual is needed, Rice-index + value-quantized correction; expand off the token path; consume CSR in-register at **tpr64**, not tg256.
3. Never clamp rank to n_fit. Publish n_fit vs rank. Refuse underdetermined scores.
4. Hunt ceremony (reparse, extra CBs, host recurrence) with a named-component ledger. Do not hunt Q80’s expert_bind.
5. Treat reconstruction as free only at the geometry that measured free, and only for the families that measured free. Low-rank two-stage is not in that set on Qwen3.8 down.

## What G1 must not steal

- The 1.43 complete-BPW identity or the 97/3 mass split.
- r160 / 1.27 / cosine [0.8862, 0.8978] as a measured down_proj point.
- 8.24 bits/outlier as a Qwen3.8 constant.
- 337→49 or 867→36.6 as Qwen3.8 targets.
- MoE router-split, 10-of-512 bind, expert first-touch, moe_table_build.
- Silent `rank = min(budget, n_fit)`.

---

## Claim boundary for this document

No GPU run, no pack, no generate, no live-Genesis interference. All numbers are recovered from git-resident receipts and source. Sparse checkout: receipts were read via `git show HEAD:<path>`, not from a materialized working tree.
