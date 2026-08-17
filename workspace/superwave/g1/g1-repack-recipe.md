# G1 packing recipes — Qwen3.8 language body

Lane: `40-repack-recipe`. Synthesis only. No GPU, no pack, no generate, no resident touch.

This file is the recipe a packing run consumes. Three assignments, one recommended.

Epistemic tags: **MEASURED** = integer or float read from an on-disk artifact, receipt, or this-lane recomputation that reproduced a sealed integer. **DERIVED** = exact arithmetic on MEASURED integers and on-disk container identities. **PROJECTED** = unused for any kill. **UNMEASURED** = generate / coherence not run.

---

## 0. What a packer must emit

Catalog: `catalog.hq38m20` (`HQ38M20\0` v1). Fused G0 names (`in_proj_qkvz`, `in_proj_ba`). Vision skipped. Small tensors stay exact f32 (codec 4, change **X0**).

`complete_physical_bpw = 8 * tensor_payload_bytes / 26895998464` — `qwen38_pack.rs:673-679`. Payload includes codes, scales, JSON/binary headers, group pad. Catalog is billed separately as `artifact_complete_bpw`.

G0 reproduction (this lane, same formulas): `14297694680` bytes, `4.252735126866492` BPW. Delta vs `uniform-q4-v1/manifest.json` = **0**. Formulas below are the ones that hit that integer.

Denominator **MEASURED**: `N = 26895998464`. MLP `17112760320`. Attention GEMV `7237795840`. Tables `2542796800`. Small `2645504`. Evidence: `g1-bit-budget-accounting.md` §4; `g1-artifact-inventory.md` §2.

---

## 1. Codec cards (only these five)

| id | container | bits | group | scale rule | body + header identity | consume kernel | kernel exists? | Qwen38 hybrid today |
|---|---|---:|---:|---|---|---|---|---|
| **UQ4** | `HQ30UQ4\0` v1 | 4 | 64 flat | `s = fp16(absmax / 7)`, `q ∈ [-8,7]` | `40 + n/2 + 2n/64` (rank-2) | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` | YES, G0 production | YES (`q4` map, or HQ38M20 codec 3) |
| **HU3** | `HGRAVU01` | 3 | 64 flat | `s = fp16(absmax / 3)`, clamp `[-3,3]` | `12 + json + 2n/64 + 3n/8` | `q80_hgravs01_factor_matvec_simd3` | YES (`q80_mixed_decode.metal:845`, `bits != 3` early-out) | parse YES; **MLP role lock REFUSES** |
| **HU2** | `HGRAVU01` | 2 | 64 flat | `s = fp16(absmax / 1)`, clamp `[-1,1]` | `12 + json + 2n/64 + n/4` | `q80_hgravs01_factor_matvec_simd` | YES (`:499`, bits ∉ {3,8}) | parse YES; embed YES (`qwen38_hgravu_embedding_lookup`); MLP lock REFUSES |
| **HB1** | `HGRAVB01` | 1 | 128 per-row | `s = fp16(mean_abs)` , 1 sign bit | `12 + json + n/8 + 2n/128` | `q80_binary_group_matvec_tg256` | YES | YES on gate only under lock |
| **F32** | f32v2 | 32 | — | identity | `8 + 4n` | uploaded as f32 buffer (norms / islands) | YES | uniform-Q4 path only; HQ38M20 needs **X0** |

UQ4 embed: `qwen_uniform_q4_embedding_lookup`. HU2/HU3 embed: `qwen38_hgravu_embedding_lookup`. lm_head is a GEMV, not a gather.

Not used, with cause:

| family | why absent |
|---|---|
| HGRAVS01 r160_b3 | mixed-2p0 down at `0.131617` **MEASURED**; native generate INCOHERENT (`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`). Generator+residual **FALSIFIED** (`g1-generator-residual.md`). **KILLS**. REOPEN_IF a down-only HGRAVS01 pack generates English with 0 fallbacks. |
| HGRAVR02 rice | in mixed-2p0 up and sub15 attention; both fail (native / expand). **KILLS** this family. |
| `ternary_t0.7_g128` | hetero-1.5 rung. No Qwen38 kernel. Not a recipe codec. |
| HQ30UQ2/UQ3 g128 | `qwen_uniform_qn_matvec` exists, 1 thread/row, Qwen30 diagnostic, **not** wired to `Qwen38HybridDecodeSession`. Do not consume. |
| expand-to-Q4 then `geo_tpr64` | binding. sub15 INCOHERENT is this confound (`tools/qwen38_sub15_pack.py:13-15`). |
| g=48 | forensics: straddling g=48 beat g=64 on L0 out Q3 (`0.9602` vs `0.9531`). No production kernel for g=48. Keep g=64 because that is what UQ4 / HGRAVU01 tiles assume. |

Worked payloads (DERIVED, compact JSON as in `q80_mixed_decode.rs` test `wrap_container`):

| tensor | n | UQ4 | HU3 | HU2 | HB1 |
|---|---:|---:|---:|---:|---:|
| gate/up `17408×5120` | 89128960 | 47349800 | 36208866 | 25067746 | 12533976 |
| down `5120×17408` | 89128960 | 47349800 | 36208866 | 25067746 | 12533976 |
| qkvz `16384×5120` | 83886080 | 44564520 | 34078946 | 23593186 | 11796696 |
| out/o `5120×6144` | 31457280 | 16711920 | 12779743 | 8847582 | 4423893 |
| ba `96×5120` | 491520 | 261160 | 199894 | 138454 | 69323 |
| q `12288×5120` | 62914560 | 33423400 | 25559265 | 17694945 | 8847574 |
| k/v `1024×5120` | 5242880 | 2785320 | 2130140 | 1474780 | 737489 |
| embed/lm_head `248320×5120` | 1271398400 | 675430440 | 516505832 | 357581032 | n/a (embed refuses binary) |

HB1 gate vs mixed-2p0 MEASURED `12534021`: this JSON is 45 B smaller (`representation` key set). 256 binary tensors × 45 B = 11520 B = `3.4e-6` BPW. Does not move a decimal that matters.

HU3 gate JSON (214 B), used for every HU3 header length:

```
{"schema":"hawking.gravity.uniform_group.v1","representation":"uniform_q3_group_scale","shape":[17408,5120],"elements":89128960,"bits":3,"group_size":64,"groups":1392640,"scale_bytes":2785280,"code_bytes":33423360}
```

---

## 2. Named reader / kernel changes

HEAD `Qwen38HybridWeights::load_mixed` + `assert_mixed_mlp_native` cannot open any of these three recipes as written. mixed-2p0 is the only HQ38M20 shape that loads today.

**X0 — HQ38M20 codec 4 = f32v2.** `load_mixed` (`qwen38_hybrid_decode.rs:659-664`) refuses codec ∉ {0,1,2,3}. Small tensors and island rows must stay exact f32 (`g1-doctor-tensor-map.md` §5.4). Add codec 4: magic/payload `u64 numel + n f32 LE` (`qwen38_pack.rs:442-463` / `read_qwen38_f32_payload`), upload into `f32s`. Without X0, norms in an HQ38M20 catalog are either missing or silently 8-bit dequant.

**X1 — drop the mixed-2p0 MLP role lock.** `qwen38_hybrid_decode.rs:958-1003` requires every layer `gate=Binary`, `up=Residual`, `down=Hgravs`. Accept `Uniform` (HU2/HU3) and `Binary` (HB1) on any MLP role, and treat HQ30UQ4 in the `q4` map as a valid MLP GEMV (`encode_named_matvec:1213-1215` already dispatches it). This is the change that makes C, R, and A loadable. It does not add a kernel.

**X2 — island overwrite (R and A only).** After each write GEMV (`mlp.down_proj`, `linear_attn.out_proj`, `self_attn.o_proj`), if catalog row `{name}::island_row_3994` exists (codec 4, length = in_dim), dispatch a 1-row f32 dot into `out[3994]`. New kernel `qwen38_exact_row_dot_f32` (does not exist). GEMV geometry stays regular; the Qn row is double-stored (waste billed).

No other extension. HU2/HU3 GEMV and HU2 embed already have kernels.

---

## 3. Hard constraints honored

1. **Not mixed-2p0 allocation.** mixed-2p0 left attention at `4.250142713483966` and crushed `down_proj` to `0.13161714918473189` (`mixed-2p0-v1/PACK_REPORT.json`). Native generate INCOHERENT, 0 fallbacks, 0 dense-W. All three recipes keep `gate=up=down` at the **same** codec inside a layer. None uses HGRAVS01. Aggressive compresses attention (attn physical `1.772325`), it does not leave it at Q4.

2. **Direct kernel, no expand-to-Q4.** Each codec in §1 names the tile that consumes packed bytes.

3. **Generator+residual not proposed.** Zero hits at lower total BPW (`g1-generator-residual.md`). **KILLS**. REOPEN_IF activation-weighted SVD on a Q4-vehicle capture beats Q4 error at lower BPW.

4. **Sparse |W| islands not proposed as the attention path.** `g1-sparse-exact-islands.md` FALSIFIED. The residual **output row 3994** island is a different mechanism (one row, activation-persistent, doctor MEASURED) and is only in R and A.

5. **Act-colscale not in any recipe.** Density probe `HGRAVU01_q4_g64_act_colscale` dropped L0 out_proj output cosine `0.99224 → 0.91865` (`g1-doctor-tensor-map.md` §3.1). Forensics exact-|X| columns moved `0.9531 → 0.97616`. Unsettled. Pack-time output-MSE group scale (`argmin_s ||X(w−q(w,s))||`) is the named next codec experiment, **not** a recipe field. REOPEN_IF that search is scored on the 192/64 split and beats absmax on residual-proxy without the colscale collapse.

---

## 4. Headline numbers

| recipe | role | tensor_payload_bytes | catalog_bytes | artifact_bytes | complete_physical_bpw | artifact_complete_bpw | mlp | attn | tab |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 UQ4 | incumbent MEASURED | 14297694680 | n/a (JSON manifest) | — | **4.252735126866492** | — | 4.2500035903 | 4.2500091962 | 4.2500002517 |
| **G1-C** | conservative | 12158635352 | 141051 | 12158776403 | **3.616489008437207** | 3.616530962930977 | 3.2500202852 | 4.2500091962 | 4.2500002517 |
| **G1-R** | recommended | 12354061566 | 166503 | 12354228069 | **3.6746169754689051** | 3.6746665004568615 | 3.3385604737 | 4.2500091962 | 4.2500002517 |
| **G1-A** | aggressive <1.5 | 4900690968 | 166503 | 4900857471 | **1.4576714003191282** | 1.4577209253070844 | 1.1250193876 | 1.7723246927 | 2.7500014598 |
| mixed-q3mlp | packed sibling MEASURED | 12149632429 | — | — | 3.613864737317677 | — | q3 MLP + q4 rest | 4.25 | 4.25 |
| mixed-2p0 | KILL sibling MEASURED | 7011580330 | 158970 | 7011764637 | 2.085538587276445 | 2.085593407922051 | 0.8480504639 | 4.250 | 4.250 |
| 1.5 ceiling | DERIVED | 5042999712 | — | — | 1.5 | — | — | — | — |

G1-C vs mixed-q3mlp: +9,002,923 B. Cause: this file bills HGRAVU01 JSON headers on 192 MLP tensors; mixed-q3mlp's on-disk header family was not re-parsed this lane (packer PRESERVE_ONLY `11be05969`). Gap = `0.002624` BPW. Does not flip any cell vs 1.5 or vs G0.

G1-A slack vs 1.5: `142308744` B.

Calculator: `/tmp/g1_repack_bpw.py`. G0 match printed below §11.

---

## 5. G1-C — conservative

**Intent.** One change from the only generate-proven artifact: MLP leaves Q4, nothing else does.

**Should-certainly-be-coherent, not generate-proven.** Q4 whole-model is COHERENT (`QWEN38_COHERENCE_SEAL.json`). Q3 MLP hold-cosine min on 18 scored organs = **0.9679** (L31 up), all ≥ descent 0.95 bar (`g1-doctor-tensor-map.md` §5.1, `QWEN38_BPW_DESCENT.json`). Attention, embed, lm_head stay at the last generate-proven cheap codec (Q4). No organ is uniquely starved. mixed-q3mlp is this assignment packed, **NEVER_EVALUATED** (`g1-arch-gravity-runs.md` §3.6). Floor remains `(2.0856, 4.2527]`.

### 5.1 Class rule (all 64 layers)

| class | n | codec | bits | group | scale | kernel | exists |
|---|---:|---|---:|---:|---|---|---|
| mlp.gate_proj | 64 | HU3 | 3 | 64 | absmax/3 | `q80_hgravs01_factor_matvec_simd3` | YES |
| mlp.up_proj | 64 | HU3 | 3 | 64 | absmax/3 | same | YES |
| mlp.down_proj | 64 | HU3 | 3 | 64 | absmax/3 | same | YES |
| dn.in_proj_qkvz / ba / out | 48+48+48 | UQ4 | 4 | 64 | absmax/7 | `geo_tpr64_tg128` | YES |
| gqa.q / k / v / o | 16×4 | UQ4 | 4 | 64 | absmax/7 | `geo_tpr64_tg128` | YES |
| embed | 1 | UQ4 | 4 | 64 | absmax/7 | `qwen_uniform_q4_embedding_lookup` | YES |
| lm_head | 1 | UQ4 | 4 | 64 | absmax/7 | `geo_tpr64_tg128` | YES |
| all small (353) | 353 | F32 | 32 | — | identity | f32 buffer | YES after X0 |

No islands. No per-layer variation.

### 5.2 Bytes (DERIVED)

| bucket | bytes | physical BPW |
|---|---:|---:|
| MLP | 6952102272 | 3.250020285213695 |
| attention GEMV | 3845087360 | 4.250009196169866 |
| tables | 1350860880 | 4.250000251691366 |
| small | 10584840 | 32.00853977162764 |
| **tensor payload** | **12158635352** | **3.616489008437207** |
| catalog HQ38M20 (755 rows, 66 segs) | 141051 | +0.000041954493770 |
| **artifact** | **12158776403** | **3.616530962930977** |

Reader: X0 + X1. Kernels all exist.

### 5.3 Per-layer GEMV (packer CSV)

```
layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o
0,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
1,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
2,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
3,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
```

Rows 4–63 repeat the mixer pattern: DeltaNet `HU3,HU3,HU3,UQ4,UQ4,UQ4`; GQA iff `(layer+1)%4==0` → `HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4`. Full 64-row table is identical in structure to §7.3 with every `mlp.down` = HU3.

---

## 6. G1-A — aggressive (complete < 1.5)

**Intent.** Land strictly under 1.5 without mixed-2p0's two sins: do not leave attention at Q4, do not uniquely starve down.

**Coherence: UNMEASURED, and the cosine screen says this is below the MLP floor.** Mid-depth binary hold-cosine on up/down is `0.73–0.82` (`g1-doctor-tensor-map.md` §5.1). L0 out Q2 mixer-output `0.7063` (`g1-out-proj-forensics.md` §Full Q3/Q4 table) — that is why writes stay HU3, not HB1. Native fail at 2.0856 used a *different* MLP family (rice+HGRAVS01) with Q4 attention; it does not measure this assignment, but it does say “sub-2 complete with crushed MLP” has already died once.

G1-A is the arithmetic target, not the first pack.

### 6.1 Class rule (all 64 layers; gate=up=down)

| class | n | codec | bits | group | scale | kernel | exists |
|---|---:|---|---:|---:|---|---|---|
| mlp.gate / up / down | 64×3 | HB1 | 1 | 128 mean-abs | `q80_binary_group_matvec_tg256` | YES |
| dn.in_proj_qkvz | 48 | HB1 | 1 | 128 | mean-abs | same | YES |
| gqa.q_proj | 16 | HB1 | 1 | 128 | mean-abs | same | YES |
| dn.out_proj / gqa.o_proj | 48+16 | HU3 | 3 | 64 | absmax/3 | `factor_matvec_simd3` | YES |
| gqa.k / v | 16+16 | HU3 | 3 | 64 | absmax/3 | same | YES |
| dn.in_proj_ba | 48 | HU3 | 3 | 64 | absmax/3 | same | YES |
| embed | 1 | HU2 | 2 | 64 | absmax/1 | `qwen38_hgravu_embedding_lookup` | YES |
| lm_head | 1 | HU3 | 3 | 64 | absmax/3 | `factor_matvec_simd3` | YES |
| small | 353 | F32 | 32 | — | identity | f32 | YES after X0 |
| island row 3994 on 128 write tensors | 128 | F32 sidecar | 32 | — | exact | `qwen38_exact_row_dot_f32` | **NO — X2** |

Why this split, not hetero-1.5: hetero-1.5 puts 53 downs at 1-bit and early out_proj at 1-bit, and uses ternary (no kernel). G1-A spends the scarce high bits on **writes + k/v + ba + lm_head** (forensics: write-gain and mixer-output live there) and keeps MLP organs equal. Embed HU2 because `encode_embed_mixed` refuses HB1 (`qwen38_hybrid_decode.rs:2906-2908`) and tables at Q4 make complete <1.5 **KILL** with this MLP (`g1-bit-budget-accounting.md` §7.3).

### 6.2 Bytes (DERIVED)

| bucket | bytes | physical BPW |
|---|---:|---:|
| MLP (192 × HB1) | 2406523392 | 1.125019387637868 |
| attention GEMV | 1603465536 | 1.772324692706447 |
| tables (embed HU2 + lm HU3) | 874086864 | 2.750001459809923 |
| small | 10584840 | 32.00853977162764 |
| island sidecars (128 × f32v2 row) | 6030336 | +0.0017936753 on complete |
| **tensor payload** | **4900690968** | **1.4576714003191282** |
| catalog (883 rows) | 166503 | +0.000049524987956 |
| **artifact** | **4900857471** | **1.457720925307084** |
| 1.5 ceiling | 5042999712 | slack **142308744** B |

vs mixed-2p0: attn `1.772` vs `4.250`; down `1.125` vs `0.132`; gate=up=down.

### 6.3 Per-layer GEMV (packer CSV)

Every layer is the same class rule. Mixer only changes which columns are live.

```
layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o
0,delta_net,HB1,HB1,HB1,HB1,HU3,HU3,,,,
1,delta_net,HB1,HB1,HB1,HB1,HU3,HU3,,,,
2,delta_net,HB1,HB1,HB1,HB1,HU3,HU3,,,,
3,gqa,HB1,HB1,HB1,,,,HB1,HU3,HU3,HU3
```

Rows 4–63: DeltaNet `HB1,HB1,HB1,HB1,HU3,HU3`; GQA `HB1,HB1,HB1,,,,HB1,HU3,HU3,HU3`.

Reader: X0 + X1 + X2.

---

## 7. G1-R — recommended

**This is the recipe a G1 pack run should emit.**

### 7.1 Why R, not C or A (evidence, not preference)

1. **C is already on disk.** mixed-q3mlp is G1-C's sibling at MEASURED `3.613864737317677` BPW, `generation_is_the_gate: true`, no `GENERATE.json` (`g1-arch-gravity-runs.md` §3.6). Packing C again does not add information. After X1, the cheapest C-test is native greedy on that artifact, not a new pack.

2. **A sits below every measured token fact.** Token floor is `(2.0856, 4.2527]` (`QWEN38_COHERENCE_FLOOR_BRACKETED.json`). A at `1.4577` is under the native fail. Doctor binary hold on mid-depth MLP is `0.73–0.82`. Bit-budget inversion: tables at Q4 + MLP at 1.0 already caps attention at `1.705` for complete <1.5 (`g1-bit-budget-accounting.md` §7.1); A meets the byte target by putting 17.11 G MLP weights at 1-bit. That is a target envelope, not a coherence claim.

3. **Wave 1 named two cheap corrections that do not leave Q4 attention.**
   - Residual channel **3994** is activation-hot in 54/64 layers and a top-5 output row on all 128 write tensors; L0 `lin_o` kurtosis `149.36` (`g1-doctor-tensor-map.md` §3.1; `g1-out-proj-forensics.md`). Mass ≪ 0.01%. Sidecar tax **6030336** B = `0.001794` BPW.
   - Late-layer write-gain: L63 `down` write/R = `2.792`, Q3 residual-proxy `0.97324`; L47 down binary hold `0.7802` (`g1-out-proj-forensics.md` §D/E; descent). R keeps `down` L47–63 at UQ4 (17 tensors). Cost vs C: `17 * (47349800 − 36208866) = 189395578` B = `0.056335` BPW.

4. **R does not repeat mixed-2p0.** Attention stays UQ4 (generate-proven). MLP organs stay equal inside each layer. down is *raised* in late layers, not crushed. No HGRAVS01, no rice.

5. **Act-MSE scale is not in R.** The two wave-1 measurements on “use X, not |W|” contradict each other (colscale kills L0 out; exact-|X| columns help). R takes the discriminator that is cheap and uncontradicted (one exact output row) and leaves group-scale search as the next pack flag, not as this recipe.

If mixed-q3mlp / C generate **fails**, R is also dead (it is C plus bits). If it **passes**, R is the first new pack: same family, two measured patches, still `3.67` complete — inside the unmeasured bracket, above 2.0856.

### 7.2 Class rule

Same as C except:

| class | layers | codec | note |
|---|---|---|---|
| mlp.down_proj | 0–46 (47 tensors) | HU3 | same as C |
| mlp.down_proj | 47–63 (17 tensors) | UQ4 | late write-gain |
| island row 3994 | all 64 down + 48 lin_o + 16 o | F32 sidecar | overwrite after GEMV |

gate/up all HU3. All attention GEMVs UQ4. embed+lm_head UQ4. small F32.

### 7.3 Bytes (DERIVED)

| bucket | bytes | physical BPW |
|---|---:|---:|
| mlp.gate (64×HU3) | 2317367424 | 3.250020285213695 |
| mlp.up (64×HU3) | 2317367424 | 3.250020285213695 |
| mlp.down (47×HU3 + 17×UQ4) | 2506763302 | 3.515640850628123 |
| MLP total | 7141498150 | 3.338560473685171 |
| attention GEMV | 3845087360 | 4.250009196169866 |
| tables | 1350860880 | 4.250000251691366 |
| small | 10584840 | 32.00853977162764 |
| island 128×f32v2 | 6030336 | (no new source elements) |
| **tensor payload** | **12354061566** | **3.6746169754689051** |
| catalog (883 rows) | 166503 | +0.000049524987956 |
| **artifact** | **12354228069** | **3.6746665004568615** |

Island in_dim: down 17408 → `8+4*17408 = 69640` B × 64 = `4456960`. out/o 6144 → `24584` B × 64 = `1573376`. Sum `6030336`.

Reader: X0 + X1 + X2. UQ4 / HU3 kernels exist. Island kernel does not (X2).

### 7.4 Per-layer GEMV (packer CSV)

```
layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o
0,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
1,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
2,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
3,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
4,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
5,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
6,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
7,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
8,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
9,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
10,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
11,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
12,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
13,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
14,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
15,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
16,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
17,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
18,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
19,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
20,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
21,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
22,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
23,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
24,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
25,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
26,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
27,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
28,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
29,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
30,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
31,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
32,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
33,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
34,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
35,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
36,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
37,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
38,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
39,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
40,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
41,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
42,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
43,gqa,HU3,HU3,HU3,,,,UQ4,UQ4,UQ4,UQ4
44,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
45,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
46,delta_net,HU3,HU3,HU3,UQ4,UQ4,UQ4,,,,
47,gqa,HU3,HU3,UQ4,,,,UQ4,UQ4,UQ4,UQ4
48,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
49,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
50,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
51,gqa,HU3,HU3,UQ4,,,,UQ4,UQ4,UQ4,UQ4
52,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
53,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
54,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
55,gqa,HU3,HU3,UQ4,,,,UQ4,UQ4,UQ4,UQ4
56,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
57,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
58,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
59,gqa,HU3,HU3,UQ4,,,,UQ4,UQ4,UQ4,UQ4
60,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
61,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
62,delta_net,HU3,HU3,UQ4,UQ4,UQ4,UQ4,,,,
63,gqa,HU3,HU3,UQ4,,,,UQ4,UQ4,UQ4,UQ4
```

Plus 128 island rows, name `{gemm}::island_row_3994`, codec 4, shape `[in_dim]`.

---

## 8. Pin every small tensor (all recipes)

| class | n | shape | codec | physical BPW (MEASURED G0 / DERIVED identical) |
|---|---:|---|---|---:|
| input_layernorm | 64 | `[5120]` | F32 | 32.0125 |
| post_attention_layernorm | 64 | `[5120]` | F32 | 32.0125 |
| final_norm | 1 | `[5120]` | F32 | 32.0125 |
| dn.conv1d | 48 | `[10240,4,1]` | F32 | 32.0015625 |
| dn.norm | 48 | `[128]` | F32 | 32.5 |
| dn.A_log | 48 | `[48]` | F32 | 33.333… |
| dn.dt_bias | 48 | `[48]` | F32 | 33.333… |
| gqa.q_norm | 16 | `[256]` | F32 | 32.25 |
| gqa.k_norm | 16 | `[256]` | F32 | 32.25 |

RMS pack rule unchanged: MLX `(1+δ)` → HF `δ` (`qwen38_hybrid_decode.rs:51-62`). Quantizing this class cannot buy more than `0.003147` complete BPW (`g1-bit-budget-accounting.md` §5.5).

---

## 9. Killed allocations (do not pack)

| proposal | verdict | pointer |
|---|---|---|
| mixed-2p0 shape (binary gate, rice up, r160 down, Q4 attn) | **KILLS** native generate | `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`; down `0.1316` |
| crush MLP only, leave attn+tables Q4, complete <1.5 | **KILLS** arithmetic | even 0-bit MLP → complete `1.54864` (`g1-bit-budget-accounting.md` KILL 2) |
| tables bf16/f32 | **KILLS** | tables alone `1.51267` |
| generator + residual | **KILLS** | `g1-generator-residual.md` zero hits |
| |W|-sparse islands as attention codec | **KILLS** | `g1-sparse-exact-islands.md` |
| act-colscale from this capture | **KILLS** as written | L0 out `0.992→0.919` |
| hetero-1.5 ternary + 1-bit early out | not executable (no ternary kernel); early out Q2 cosine `0.706` | `g1-heterogeneous-allocation.md` §7; forensics Q2 table |
| complete 1.0 | **KILLS** this codec family | binary floor `1.128069` (`g1-heterogeneous-allocation.md` §5) |

---

## 10. Cheapest next experiment (not this lane)

GPU lock owner is the measurement lane.

1. **X1**, then native `ascension_qwen38_hybrid_greedy` on existing `mixed-q3mlp-v1` (3 sealed prompts). Isolates Q3-MLP. If that artifact's catalog is not HU3/UQ4 (unknown header), pack G1-C instead.
2. If (1) PASSES: pack G1-R, native generate vs Q4 oracle.
3. If (1) FAILS: floor is above Q3 MLP. Do not pack A. Next bisect is mixed-q4down (`2.959`) / mixed-floor-q7 (`3.177`), already packed, never generated.
4. Output-MSE group scale on L0/L32 out and L3/L63 o, UQ4 codes, same `geo_tpr64`. Pack-time only. Score hold-out residual-proxy. Do not fold column RMS.

---

## 11. Calculator evidence (this lane)

Command: `python3 /tmp/g1_repack_bpw.py`

G0 identity check (formulas vs MEASURED `14297694680` / `4.252735126866492`):

```
==== G0_recompute ====
tensor_payload_bytes     14297694680
complete_physical_bpw    4.2527351268664919
mlp_bpw                  4.2500035903033089
attn_bpw                 4.2500091961698656
tab_bpw                  4.2500002516913664
small_bpw                32.0085397716276390
G0 MEASURED bytes 14297694680 delta 0
match True
```

Recipe extracts:

```
G1-C  tensor_payload_bytes 12158635352  complete_physical_bpw 3.6164890084372070
G1-R  tensor_payload_bytes 12354061566  complete_physical_bpw 3.6746169754689051  island 6030336
G1-A  tensor_payload_bytes  4900690968  complete_physical_bpw 1.4576714003191282  slack_vs_1.5 142308744
```

`8 * 12158635352 / 26895998464 = 3.616489008437207`
`8 * 12354061566 / 26895998464 = 3.6746169754689051`
`8 *  4900690968 / 26895998464 = 1.4576714003191282`

Catalog identity: HQ38M20 prefix 32 + 66 × (`44 + len("seg_XX.bin")`) + `128 * n_rows` + name-blob. C: 755 rows → `141051`. R/A: 883 rows → `166503`.

---

## 12. Machine-readable packer block

```json
{
  "schema": "hawking.superwave.g1_repack_recipe.v1",
  "n": 26895998464,
  "recommended": "G1-R",
  "reader_changes": ["X0_hq38m20_codec4_f32v2", "X1_relax_assert_mixed_mlp_native", "X2_island_row_3994_dot"],
  "codec_map": {
    "UQ4": {"magic": "HQ30UQ4\\0", "bits": 4, "group": 64, "scale": "absmax/7", "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128", "exists": true},
    "HU3": {"magic": "HGRAVU01", "bits": 3, "group": 64, "scale": "absmax/3", "kernel": "q80_hgravs01_factor_matvec_simd3", "exists": true},
    "HU2": {"magic": "HGRAVU01", "bits": 2, "group": 64, "scale": "absmax/1", "kernel": "q80_hgravs01_factor_matvec_simd", "exists": true},
    "HB1": {"magic": "HGRAVB01", "bits": 1, "group": 128, "scale": "mean_abs", "kernel": "q80_binary_group_matvec_tg256", "exists": true},
    "F32": {"magic": "f32v2", "bits": 32, "kernel": "f32_buffer", "exists": true, "hq38m20": "X0"}
  },
  "recipes": {
    "G1-C": {
      "complete_physical_bpw": 3.616489008437207,
      "tensor_payload_bytes": 12158635352,
      "mlp": "HU3x192",
      "attn": "UQ4",
      "tables": "UQ4",
      "islands": false
    },
    "G1-R": {
      "complete_physical_bpw": 3.6746169754689051,
      "tensor_payload_bytes": 12354061566,
      "mlp_gate_up": "HU3x128",
      "mlp_down": "HU3 L0-46; UQ4 L47-63",
      "attn": "UQ4",
      "tables": "UQ4",
      "islands": "row_3994_f32x128"
    },
    "G1-A": {
      "complete_physical_bpw": 1.4576714003191282,
      "tensor_payload_bytes": 4900690968,
      "mlp": "HB1x192",
      "attn_in_q": "HB1",
      "attn_write_kv_ba": "HU3",
      "embed": "HU2",
      "lm_head": "HU3",
      "islands": "row_3994_f32x128"
    }
  }
}
```

---

## 13. Claim boundary

- Element counts, G0 bytes, mixed sibling BPWs: MEASURED (wave 1 + this-lane G0 recompute).
- Recipe payload bytes and complete BPW: DERIVED from container identities in §1. Not a packed artifact.
- Coherence of C and R: UNMEASURED. Cosine screen only.
- Coherence of A: UNMEASURED, and the cosine screen is a fail on mid-depth binary MLP.
- `q80_hgravs01_factor_matvec_simd3` on a `17408×5120` GEMV (not a rank-160 factor) is a **direct** kernel but has **no** Qwen3.8 token-level receipt. Component existence ≠ token claim.
- Catalog bytes DERIVED from the HQ38M20 layout in `qwen38_hybrid_decode.rs:31-38,96-174`. mixed-2p0 MEASURED catalog `158970` is the same family, different row/segment count.

---

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. G0 complete BPW 4.252735126866492 = 8*14297694680/26895998464. This lane recomputed every HQ30UQ4+f32v2 payload; delta 0. Evidence: §11 calculator; uniform-q4-v1/manifest.json; qwen38_pack.rs:673-679.
C2. Three executable-by-direct-kernel recipes: G1-C 3.6164890084372070, G1-R 3.6746169754689051, G1-A 1.4576714003191282 complete physical BPW, catalog billed. Evidence: §4–§7, §11.
C3. None of the three allocates like mixed-2p0 (attn 4.250 / down 0.1316). C and R keep attn+tables at UQ4 and equalize MLP; A sets attn physical 1.772 and down 1.125 = gate = up. Evidence: §3, §6.2 vs mixed-2p0-v1/PACK_REPORT.json.
C4. Recommended is G1-R, because C is already packed as mixed-q3mlp, A is below the unmeasured token floor and below the doctor binary-MLP screen, and R is C plus the two cheap wave-1 corrections (late down UQ4, row 3994). Evidence: §7.1; g1-arch-gravity-runs.md §3.6; g1-doctor-tensor-map.md §3.1; g1-out-proj-forensics.md §D/E.
C5. Native reader cannot load C/R/A until X1 (and X0 for f32, X2 for islands). Evidence: qwen38_hybrid_decode.rs:958-1003, :659-664.
C6. Every codec in C/R/A names an existing in-register kernel except the island 1-row f32 dot (X2) and HQ38M20 f32v2 (X0). Evidence: §1 kernel table; q80_mixed_decode.metal:499,845; qwen_uniform_q4.metal:183; qwen38_hybrid_decode.rs:1396-1406.
C7. HGRAVS01 / rice / generator-residual / |W|-islands / act-colscale are not in any recipe. KILLS as cited in §9.

EVIDENCE
E1. workspace/superwave/g1/g1-bit-budget-accounting.md §§1,4,5,7,8
E2. workspace/superwave/g1/g1-heterogeneous-allocation.md §§1,5,7 (1.5 table not used: ternary kernel missing)
E3. workspace/superwave/g1/g1-out-proj-forensics.md named root cause + Q3/Q4 table + |W|∩|X|=0
E4. workspace/superwave/g1/g1-arch-gravity-runs.md ledger + NEVER_EVALUATED floor packs
E5. workspace/superwave/g1/g1-kernel-inventory.md §3, §6.2, §6.5
E6. workspace/superwave/g1/g1-doctor-tensor-map.md §§3,5
E7. workspace/superwave/g1/g1-generator-residual.md STATUS FALSIFIED
E8. workspace/superwave/g1/g1-sparse-exact-islands.md STATUS FALSIFIED
E9. workspace/superwave/g1/g1-lm-head-and-tails.md tables 9.45% / lm_head Q3 top-1 regress
E10. crates/hawking-core/src/model/qwen38_hybrid_decode.rs:508-673,958-1003,1213-1221,1377-1416,2858-2908
E11. crates/hawking-core/src/model/qwen38_pack.rs:673-679
E12. crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:185-229,529-576,1134-1168
E13. /tmp/g1_repack_bpw.py output in §11

CHANGES
created workspace/superwave/g1/g1-repack-recipe.md
no tracked file modified

TESTS
test -s workspace/superwave/g1/g1-repack-recipe.md
wc -l workspace/superwave/g1/g1-repack-recipe.md
git status --porcelain

RISKS
R1. HU3 on a 17408×5120 GEMV uses the factor simd3 tile, never token-measured at that geometry. Direct, but not a G0-class receipt.
R2. X1 is a lock relaxation, not a new numeric path; a bug in the lock replacement could reopen silent Q4 fallback. Implementation must keep the "refuse unknown / refuse reconstruct" branches.
R3. G1-C vs mixed-q3mlp 9.0 MB header-family gap. If that pack already used HU3, generate it; do not assume the catalogs are byte-identical.
R4. Island double-stores row 3994. Waste is billed. A geometry that omits the row from the GEMV needs an irregular kernel; not proposed.
R5. Doctor coverage is 6/64 layers, 256 tokens, no Q4-vehicle X. Late-down UQ4 band (L47-63) interpolates L47/L63 write-gain. REOPEN_IF a 64-layer write-gain census moves the cut.

UNRESOLVED
U1. Native generate of mixed-q3mlp / G1-C. Cheapest floor bisect from above.
U2. Whether HGRAVU01 bits=3 on full GEMVs matches HQ30UQ4-family Q3 error (bound 3 vs nibble-8). Same bit width, different signed alphabet. Organ cosine for HU3-as-packed was not re-scored this lane. Cheapest: encode one L31 up both ways, print hold cosine.
U3. Output-MSE group scale (the unsettled forensics mechanism).
U4. G1-A coherence. Do not pack until U1 is a pass and a mid-band (2.96-3.18) generate also passes.

NEXT
N1. Measurement lane: X1 then greedy on mixed-q3mlp-v1.
N2. Pack G1-R only after N1 PASSES.
N3. Do not pack G1-A, mixed-2p0-shaped MLP, or generator+residual.
```
