# G1 packing recipe v2 — what the measurements now license

Lane: `101-g1-recipe-v2`. Synthesis + integer arithmetic on on-disk catalogs. No GPU. No pack. No generate. No resident touch.

Tags: **MEASURED** on-disk integer/float or sibling-lane command output. **DERIVED** exact arithmetic on MEASURED integers. **CITED** prior report, not re-run. **ESTABLISHING** sibling lane 100, not re-measured here.

`complete_physical_bpw = 8 * tensor_payload_bytes / N` with `N = 26895998464`. Catalog billed separately as artifact BPW. Law: `qwen38_pack.rs:673-679`.

G0 sealed: `4.252735126866492` = `8 * 14297694680 / 26895998464`. CITED `g1-bit-budget-accounting.md` §4; this-lane recompute delta 0 at float64.

---

## 0. Verdict

Sub-1.5 and ≤2.0 are closed for this model under every native codec that has been scored. The one coherent native artifact is mixed-q3mlp-v1 at complete **3.6138111608720234**. Build inside that band.

Three recipes a packing run may consume:

| id | role | complete BPW (G0 def) | vs G0 | pack? |
|---|---|---:|---:|---|
| **G1-C** | coherent baseline = q3mlp exactly | **3.6138111608720234** MEASURED | −0.6389239659944685 | already packed; do not re-pack |
| **G1-Q** | MSE scale on the C tile (free) | **3.6138111608720234** DERIVED (= C) | −0.6389239659944685 | **first new pack** |
| **G1-L** | lowest that still respects every scored floor | **3.5801733386059729** DERIVED | −0.6725617882605190 | second pack, density only |

Recommended: **G1-Q**. Same assignment as the only COHERENT native body. Same bytes. Same kernels. The scale plane is the only change. G1-L cuts 113,090,352 B (−0.0336378222660505 vs C) by moving attention to Q4 g=128 MSE, which is the measured 0.99 attention floor.

All three land 192 MLP GEMVs on `q80_hgravs01_factor_matvec_simd3`. That is the slow Uniform bits-3 path. q3mlp complete-token wall is 148.588917 ms / 6.7300 TPS (DIRTY_ENGINEERING) against G0 39.326090 ms / 25.4284 TPS: ratio **3.77838**. Sibling lane 100 is establishing a 3.78× kernel-class penalty for that tile. These are coherence/density recipes, not 100 TPS recipes.

---

## 1. Binding measurements (do not re-derive)

### 1.1 Census

| class | tensors | elements MEASURED |
|---|---:|---:|
| MLP gate/up/down | 192 | 17,112,760,320 |
| attention GEMV | 304 unfused | 7,237,795,840 |
| embed + lm_head | 2 | 2,542,796,800 |
| small | 353 | 2,645,504 |
| language N | 851 | **26,895,998,464** |

CITED `g1-bit-budget-accounting.md` §4. Reconfirmed this lane against mixed-q3mlp-v1 catalog (851 rows, sum nbytes 12,149,632,429).

### 1.2 MLP — 0.95-all-192 native floor is Uniform q3

CITED `g1-mlp-floor-locate.md` `/tmp/g1_mlp_floor.json` (192 organs, wall 983.1 s, rss 4.257 G). Hold = last 64 of 256.

| codec | down hold min | at | n≥0.95 / 64 | class physical BPW |
|---|---:|---|---:|---:|
| Binary | **0.300575** | L58 | 0 | 1.1250234267290902 |
| Residual 2% | **0.364203** | L58 | 0 | 1.2875108157887178 |
| Uniform q2 | **0.752931** | L62 | 1 | 2.25 + header |
| HGRAVS r160_b3 honest (192/64) | **0.730175** | L9 | 1 | 0.13161714918473189 |
| HGRAVS packed-2p0 in-sample | 0.919630 | L8 | 49 | 0.13161714918473189 |
| **Uniform q3** | **0.966253** | **L62** | **64** | **3.2500251321231617** |
| Uniform q4 | 0.993738 | L62 | 64 (also 0.99-all-192) | 4.2500251321231617 |

Uniform q3 is the first native codec with hold ≥ 0.95 on all 192. Min 0.9662531725591231 at L62 down. Packed 2p0 HGRAVS 0.9196 is in-sample (`n_fit_rows=256`); honest hold is 0.7302.

L54/L58/L59 binary 0.36/0.30/0.38 are not in the old 6-layer sample. **KILLS** Binary / Residual / Q2 / HGRAVS as a down codec. REOPEN_IF a capture with rows/dim ≫ 0.05 moves those three above 0.95.

### 1.3 Attention — 0.99 floor is Q4 g=128 MSE

CITED `g1-attention-2bpw-stack.md` `/tmp/g1_attention_2bpw_stack.json` (29 GEMVs).

| cfg | b_attn | min hold | n≥0.99 |
|---|---:|---:|---:|
| best ≤ ~2.07 (Q2 g=256 MSE k=1) | 2.065634 | 0.818095 L0.out | **0 / 29** |
| Q4 g=64 **absmax** | 4.250009 | **0.989797** L47.o | 28 / 29 |
| Q4 g=64 **MSE** | 4.250009 | **0.991123** L47.o | 29 / 29 |
| Q4 g=128 **absmax** | 4.125009 | 0.987709 | 27 / 29 |
| **Q4 g=128 MSE** | **4.125009196169866** | **0.990088** L47.o | **29 / 29** |

Nothing holds 0.99 at or under 2.07. Q4 g=64 absmax fails the bar on L47 o_proj (0.9897967539, matches `g1-group-partition-geometry.md`). MSE at identical BPW and identical g=64 tile is the free quality win (0.991123). MSE is also the only reason g=128 clears.

### 1.4 Generate — codec identity, not bits

CITED `g1-mlp-family-generate.md`. Native `ascension_qwen38_hybrid_greedy`, 0 fallbacks, 0 expand-to-Q4 / dense-W.

| artifact | complete (G0 def) | MLP | verdict |
|---|---:|---:|---|
| mixed-q3mlp-v1 | 3.6138111608720234 | 3.2500251321231617 all U01 q3 | **COHERENT** (France⊃Paris ×8, 17×19⊃323 ×3) |
| mixed-q4down-v1 | 2.9589935339460913 | **2.2208531248803234** (B01 gate 1.125 / R02 up 1.288 / U01 q4 down 4.250) | **INCOHERENT** |

Q4 down does not rescue Binary gate + Residual up. mixed-q3mlp is not a 6/6 oracle-32 seal (loops after the fact). It is the campaign-gate coherent point.

### 1.5 Island — nearly free, subadditive, L0-write tool

CITED `g1-channel-3994-island.md`, stacked in `g1-attention-2bpw-stack.md` §6.

- Honest compile-time set `{3994, 3456, 310}`. 3994 is identically 0 on L7 (`L07.f32` n_nonzero=0; L7 post_attn γ[3994]=0).
- k=1 cost 5,252,608 bf16 elems, 0 index bits, **+0.002490** vs a Q3 body. PROJECTED.
- Q3 + island does not reach 0.99 on L0.out (0.95310 → 0.96195) or L47.o (0.94831 → 0.94858).
- Stacked with MSE: L0.out Q3 Δstack +0.02309 < ΔMSE+Δisl +0.03035 (subadditive). Late GQA island Δ ≤ 0.00027. Dies after layer-0 writes.

Not in G1-C / G1-Q / G1-L. Does not buy a bit-width drop. Do not put `if (row==3994)` in TPR64.

### 1.6 Scale rule (the free win)

CITED `g1-mse-scale-rule.md` `/tmp/qwen38_mse_scale_rule.json` (732 cells, 0 LOSEs vs absmax).

```
s = argmin_m ||X_g (w − q(w, s0·m))||
m ∈ {0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00}
s0 = f16(absmax(w) / bound)     # bound = 2^(bits-1)-1
fit = even 128 of the 256-token capture; hold = odd 128
```

Same BPW as absmax at the same bits and group. Kernel consumes `float(q)*float(scale)`; it does not know how s was chosen. α=1 AWQ still dead (L0 out Q4 0.99225 → 0.91865). Embed gather is not a production MSE plane (63 / 248320 vocab rows observed).

256-token X is underdetermined (rows/dim 0.0417 at K=6144, 0.0147 at 17408). Sign of the rule is determined. The numbers on a frozen plane are not. Packing run fits; do not treat the 256-token plane as sealed.

---

## 2. Native codecs and kernels

Reader accepts catalog 0/1/2/3 (`load_mixed`). MLP admits Uniform (lane 91). HGRAVS locked r160_b3. HQ30UQ4 on an MLP GEMV still refuses (lands in `q4` map → lock sees missing).

| family | magic | bits | group | scale (absmax default) | consume kernel | slow? |
|---|---|---:|---:|---|---|---|
| Binary | `HGRAVB01` | 1 | 128 | mean-abs | `q80_binary_group_matvec_tg256` (K≤2048) / `q80_binary_group_matvec_simd_bytes` (K>2048) | no (not used) |
| Residual | `HGRAVR02` | rice+1 | 128 | rms 2% | `q80_binary_group_csr_matvec_tg256` / `_simd_bytes` | no (not used) |
| Hgravs | `HGRAVS01` | 3 | 64 | factor | two-stage `q80_hgravs01_factor_matvec_simd3` | yes, and quality-dead on down |
| **Uniform q3** | `HGRAVU01` | 3 | 64 | absmax/3 or MSE | **`q80_hgravs01_factor_matvec_simd3`** | **YES — bits-3 Uniform. ESTABLISHING 3.78× vs G0 class. Flag on every recipe.** |
| Uniform q2,4–7 | `HGRAVU01` | 2,4–7 | 64 or 128 | absmax/(2^{b-1}−1) or MSE | `q80_hgravs01_factor_matvec_simd` | Uniform path, not G0 `geo_tpr64`. Not the 3.78× simd3 tile. |
| Uniform q8 | `HGRAVU01` | 8 | 64 | absmax/127 | `q80_uniform8_matvec_tg256` (K≥2048) / `_simd_bytes` | not used |
| HQ30UQ4 | `HQ30UQ4\0` | 4 | **64 only** | absmax/7 or MSE | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` | **G0 fast path.** Not used by q3mlp (attention is U01). MLP role-lock refuses this magic on gate/up/down. |
| embed U01 | `HGRAVU01` | 4 | 64 | absmax/7 | `qwen38_hgravu_embedding_lookup` | gather, not GEMV |
| small U01 | `HGRAVU01` | 4 | 64 | absmax/7 | dequant → f32 buffer at load | not a GEMV |

Dispatch: `qwen38_hybrid_decode.rs:96-110, 1600-1760`. simd3 early-outs unless `bits==3` (`q80_mixed_decode.metal:845+`). `group_size` is a kernel argument; g=128 Uniform is consumable. HQ30UQ4 production tile is hardcoded g=64.

q3mlp load census MEASURED: `tensors=851 binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0`. 498 = 192 MLP + 304 attn + embed + lm_head. 353 small U01 dequant to f32.

---

## 3. On-disk q3mlp identity (G1-C)

Root: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`

This-lane catalog parse (HQ38M20 v1, 851×HGRAVU01, 258 segs, catalog 180124 B):

| class | n | bytes MEASURED | physical BPW |
|---|---:|---:|---:|
| mlp.gate U01 q3 g64 | 64 | 2,317,370,880 | 3.2500251321231617 |
| mlp.up U01 q3 g64 | 64 | 2,317,370,880 | 3.2500251321231617 |
| mlp.down U01 q3 g64 | 64 | 2,317,370,880 | 3.2500251321231617 |
| attn U01 q4 g64 | 304 | 3,845,162,320 | 4.2500920501233699 |
| embed U01 q4 g64 | 1 | 675,430,686 | 4.2500017995932664 |
| lm_head U01 q4 g64 | 1 | 675,430,686 | 4.2500017995932664 |
| small U01 q4 g64 | 353 | 1,496,097 | 4.5241950116121536 |
| **tensor payload** | **851** | **12,149,632,429** | **3.6138111608720234** |
| catalog | | 180,124 | +0.0000535764456533 |
| artifact | | 12,149,812,553 | 3.6138647373176767 |

One MLP tensor: 36,208,920 B. Header MEASURED:

```
{"bits":3,"code_bytes":33423360,"elements":89128960,"group_size":64,"groups":1392640,"representation":"uniform_q3_group_scale","retained_padding_elements":0,"scale_bytes":2785280,"scale_dtype":"float16","schema":"hawking.gravity.uniform_group.v1","shape":[17408,5120]}
```

`12 + 268 + 2785280 + 33423360 = 36208920`. 192 × that = 6,952,112,640. Reconstruction delta vs catalog nbytes: **0** on all 851 U01 payloads.

Attention/embed/norm copied from mixed-2p0-v1 (hardlinked). MLP re-encoded from BF16. `reconstruct_to_q4: false`.

---

## 4. Recipes

### 4.1 G1-C — reproduce q3mlp exactly

Already on disk. COHERENT. Do not pack again.

| field | value |
|---|---|
| MLP 192 | HGRAVU01 bits=3 group=64 scale=absmax/3 |
| attn 304 | HGRAVU01 bits=4 group=64 scale=absmax/7 (copied 2p0) |
| embed, lm_head | HGRAVU01 bits=4 group=64 absmax/7 |
| small 353 | HGRAVU01 bits=4 group=64 absmax/7 → f32 at load |
| island | none |
| complete | **3.6138111608720234** MEASURED |
| vs G0 | **−0.6389239659944685** |
| kernels | MLP `q80_hgravs01_factor_matvec_simd3` **SLOW**; attn `q80_hgravs01_factor_matvec_simd`; embed `qwen38_hgravu_embedding_lookup` |

### 4.2 G1-Q — MSE everywhere it is free (recommended pack)

Identical tile, identical bits, identical group, identical nbytes to G1-C. Replace absmax with the §1.6 search on every GEMV that has a site-correct X.

| tensor class | apply MSE? | why |
|---|---|---|
| mlp.gate / up / down ×64 | **yes** | same 3.25 BPW; 34-layer sweep 0 LOSEs (CITED mse-scale) |
| attn in/out/q/k/v/o ×304 | **yes** | same 4.25 BPW; takes L47.o 0.98980 → 0.991123 |
| lm_head | **yes** | same 4.25; Q4 Δ +0.00022 NEUTRAL, still write the plane |
| embed | **no** | gather; 63/248320 rows; not an activation-scale rule |
| small 353 | **no** | dequant to f32; not a GEMV |
| island | **no** | subadditive with MSE; dead after L0 writes |

Fit after any qkv/z conceptual grouping (q3mlp stores qkv and z split; search each stored tensor on its own X). Snap each candidate s to f16 before rint (stored f16 is authority).

| complete | **3.6138111608720234** DERIVED (= C; scale plane same bytes) |
| vs G0 | **−0.6389239659944685** |
| vs C | 0 bytes |
| kernels | identical to C, including **slow simd3 on 192 MLP** |

Do not switch attention to HQ30UQ4 in this recipe (that is a different tile). A later speed pack may retarget attn+lm_head to `geo_tpr64` at the same 4.25 BPW; MLP q3 cannot follow (lock + no 3.25 HQ30 container).

### 4.3 G1-L — lowest complete that respects every scored floor

Floors honored: MLP Uniform q3 (0.95-all-192); attention Q4 g=128 MSE (0.99-all-29); tables stay Q4 g=64 (generate-proven on C; g=128 tables UNMEASURED).

| class | codec | bytes DERIVED | physical BPW |
|---|---|---:|---:|
| MLP 192 | U01 q3 g64 MSE | 6,952,112,640 | 3.2500251321231617 MEASURED (= C) |
| attn 304 | U01 q4 g128 MSE | **3,732,071,968** | **4.1250922800276166** |
| embed + lm_head | U01 q4 g64 MSE (lm) / absmax (embed) | 1,350,861,372 | 4.2500017995932664 |
| small 353 | U01 q4 g64 absmax | 1,496,097 | 4.5241950116121536 |
| **payload** | | **12,036,542,077** | **3.5801733386059729** |
| catalog (851 rows, same layout) | | 180,124 | artifact **3.5802269150516262** |
| vs G0 | | | **−0.6725617882605190** |
| vs C | −113,090,352 B | | −0.0336378222660505 |

Attention-lane floor quote 4.125009196169866 is HQ30-style 40-B headers on the fused set. This recipe keeps the q3mlp U01 JSON container; header tax makes class BPW 4.1250922800276166. Same RTN operator. Kernel: `q80_hgravs01_factor_matvec_simd` with `group_size=128` (not simd3, not `geo_tpr64`). MLP still **slow simd3**.

Tables at g=128 would save another 39,731,200 B (complete **3.5683556103879468**, −0.6843795164785451 vs G0). UNMEASURED at generate and on embed gather. Not in G1-L.

Q4 g=128 **absmax** does not clear 0.99. G1-L attention **must** be MSE.

---

## 5. KILLS / REOPEN_IF

| ID | KILL | REOPEN_IF |
|---|---|---|
| K1 | Targets A/B/C (≤2.0 coherent native) | a new native codec, or a non-underdetermined capture that lifts Q2/Residual/HGRAVS above 0.95 on L54/L58/L59/L62 **and** a 0.99 attention encoding ≤2.07 |
| K2 | mixed-sub15 1.291 / mixed-2p0 2.086 as floors | already confounded; do not cite as the Qwen3.8 floor |
| K3 | mixed-q4down 2.221 MLP | generate INCOHERENT with Q4 down still on Binary+Residual. REOPEN_IF a native generate of Uniform-all-MLP below q3 is English |
| K4 | Binary / Residual / Q2 / HGRAVS as down | §1.2 |
| K5 | Q4 g=64 absmax as 0.99 attention | L47.o 0.98980. MSE at same tile clears. |
| K6 | Island as a bit-width drop | +0.00249, does not reach 0.99, subadditive with MSE |
| K7 | Act-colscale α=1 | L0 out 0.992→0.918 |
| K8 | Treating q3mlp 6.73 TPS as a speed win | 3.778× slower than G0; simd3 tax |
| K9 | HGRAVU01 q3 without pricing simd3 | sibling ESTABLISHING 3.78×; token ratio MEASURED 3.77838 |

---

## 6. Packer parameter set

Packer that produced C: `lab/operators/qwen38_mlp_not_r160_pack.py` (commit `00939c186`, branch `grok/auto-q80-qwen3-complete-still-bpw-20260816-191937`). **Not on this HEAD.** Absmax, g=64, MLP-only replace, hardlink the rest from mixed-2p0. Restore that file; do not rewrite the catalog format.

```json
{
  "schema": "hawking.superwave.g1_recipe.v2",
  "n": 26895998464,
  "g0_complete_physical_bpw": 4.252735126866492,
  "recommended": "G1-Q",
  "do_not_pack": ["G1-C"],
  "container": "HGRAVU01 + HQ38M20",
  "scale_search": {
    "rule": "per_group_argmin_eTGe",
    "multipliers": [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0],
    "fit": "even_128",
    "hold": "odd_128",
    "snap": "f16_before_rint",
    "skip": ["embed_tokens", "small_f32_dequant"]
  },
  "recipes": {
    "G1-C": {
      "complete_physical_bpw": 3.6138111608720234,
      "tensor_payload_bytes": 12149632429,
      "save_vs_g0": 0.6389239659944685,
      "mlp": {"magic": "HGRAVU01", "bits": 3, "group": 64, "scale": "absmax", "kernel": "q80_hgravs01_factor_matvec_simd3", "slow": true},
      "attn": {"magic": "HGRAVU01", "bits": 4, "group": 64, "scale": "absmax", "kernel": "q80_hgravs01_factor_matvec_simd"},
      "tables": {"magic": "HGRAVU01", "bits": 4, "group": 64, "scale": "absmax"},
      "artifact_already": "mixed-q3mlp-v1"
    },
    "G1-Q": {
      "complete_physical_bpw": 3.6138111608720234,
      "tensor_payload_bytes": 12149632429,
      "save_vs_g0": 0.6389239659944685,
      "delta_vs_C_bytes": 0,
      "mlp": {"magic": "HGRAVU01", "bits": 3, "group": 64, "scale": "mse", "kernel": "q80_hgravs01_factor_matvec_simd3", "slow": true},
      "attn": {"magic": "HGRAVU01", "bits": 4, "group": 64, "scale": "mse", "kernel": "q80_hgravs01_factor_matvec_simd"},
      "lm_head": {"scale": "mse"},
      "embed": {"scale": "absmax"},
      "island": false
    },
    "G1-L": {
      "complete_physical_bpw": 3.5801733386059729,
      "tensor_payload_bytes": 12036542077,
      "save_vs_g0": 0.6725617882605190,
      "delta_vs_C_bytes": -113090352,
      "mlp": {"magic": "HGRAVU01", "bits": 3, "group": 64, "scale": "mse", "kernel": "q80_hgravs01_factor_matvec_simd3", "slow": true},
      "attn": {"magic": "HGRAVU01", "bits": 4, "group": 128, "scale": "mse", "kernel": "q80_hgravs01_factor_matvec_simd", "must_not_be_absmax": true},
      "tables": {"magic": "HGRAVU01", "bits": 4, "group": 64, "scale": "mse_lm_head_absmax_embed"},
      "island": false
    }
  }
}
```

---

## 7. Exact commands (do not run in this lane)

`ART=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b`

G1-C historical command (already executed 2026-08-16, wall 182.7 s). Re-running overwrites a COHERENT artifact.

```
python3 lab/operators/qwen38_mlp_not_r160_pack.py \
  --gate-bits 3 --up-bits 3 --down-bits 3 \
  --tag mixed-q3mlp-v1 \
  --mixed $ART/mixed-2p0-v1 \
  --model-dir $ART/bf16 \
  --root $ART/mixed-q3mlp-v1
```

G1-Q (packing run restores the packer and adds `--scale-rule mse`; nbytes must equal 12149632429):

```
python3 lab/operators/qwen38_mlp_not_r160_pack.py \
  --gate-bits 3 --up-bits 3 --down-bits 3 \
  --scale-rule mse --fit-split even \
  --capture $ART/activation-capture-v1 \
  --tag mixed-g1q-q3mlp-mse-v1 \
  --mixed $ART/mixed-2p0-v1 \
  --model-dir $ART/bf16 \
  --root $ART/mixed-g1q-q3mlp-mse-v1
```

G1-L (same restore; adds `--attn-bits 4 --attn-group 128`; attn must be MSE; payload must equal 12036542077):

```
python3 lab/operators/qwen38_mlp_not_r160_pack.py \
  --gate-bits 3 --up-bits 3 --down-bits 3 \
  --attn-bits 4 --attn-group 128 \
  --scale-rule mse --fit-split even \
  --capture $ART/activation-capture-v1 \
  --tag mixed-g1l-q3mlp-q4g128mse-v1 \
  --mixed $ART/mixed-2p0-v1 \
  --model-dir $ART/bf16 \
  --root $ART/mixed-g1l-q3mlp-q4g128mse-v1
```

`--scale-rule`, `--fit-split`, `--capture`, `--attn-bits`, `--attn-group` are packing-run extensions. Current preserved packer is absmax / g=64 / MLP-only. Search implementation is `g1-mse-scale-rule.md` §Recipe (Gram `e^T G e`, 8 multipliers, f16 snap). Do not AWQ-fold. Do not expand to Q4 or float. Do not mutate mixed-q3mlp-v1, mixed-2p0-v1, uniform-q4-v1, or the resident.

Acceptance of a new pack: HQ38M20 opens, census `refused=0 expanded_to_q4=0 expanded_to_float_gemv=0`, payload bytes match the table, then native greedy vs the C gate (France 128 Paris, 17×19 256 323) with 0 fallbacks. A load is not coherence.

---

## 8. This-lane calculator

```
N=26895998464
G0_bytes=14297694680
C_bytes=12149632429          # mixed-q3mlp catalog sum nbytes MEASURED
L_attn_bytes=3732071968      # 304 U01 q4 g=128, same JSON schema, DERIVED
C_attn_bytes=3845162320      # MEASURED
tables=1350861372            # MEASURED
small=1496097                # MEASURED
mlp=6952112640               # MEASURED
L_bytes=mlp+L_attn+tables+small=12036542077

8*G0_bytes/N = 4.2527351268664919   # sealed 4.252735126866492
8*C_bytes/N  = 3.6138111608720234
8*L_bytes/N  = 3.5801733386059729
C-L          = 113090352
```

Catalog reconstruction: 851/851 U01 payloads match `12+json_len+scale_bytes+code_bytes`.

---

## 9. Claim boundary

- C complete BPW and organ bytes: MEASURED this lane on the live catalog.
- C coherence: CITED lane 92 generate, not re-run.
- MLP / attention floors: CITED lanes 81 / 80 / 74. Not re-scored.
- L bytes: DERIVED from C headers with `group_size` 64→128 (scale plane halves; JSON digit lengths held on embed, checked per attn class).
- simd3 3.78×: ESTABLISHING (lane 100). Token ratio 3.77838 is MEASURED complete-token on C vs sealed G0, DIRTY.
- G1-Q / G1-L coherence: UNMEASURED. Cosine screen + C generate only.
- Mixer X is still the derived proxy. Capture is still 256 tokens.

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. G0 complete BPW 4.252735126866492 = 8*14297694680/26895998464. Evidence: qwen38_pack.rs:673-679; g1-bit-budget-accounting.md §4; this-lane recompute.
C2. mixed-q3mlp-v1 catalog sum nbytes 12149632429, complete 3.6138111608720234, MLP 3.2500251321231617, 851×HGRAVU01. Evidence: this-lane parse of mixed-q3mlp-v1/catalog.hq38m20; PACK_REPORT.json; §3.
C3. mixed-q3mlp-v1 is COHERENT native (France Paris, 17×19 323, 0 fallbacks, 0 expand). Evidence: g1-mlp-family-generate.md:10-20,310-315.
C4. mixed-q4down-v1 MLP 2.2208531248803234 is INCOHERENT. Codec identity, not bits. Evidence: g1-mlp-family-generate.md:10-23,389-407.
C5. Uniform q3 is the first native MLP codec with hold≥0.95 on 192/192; min 0.966253 at L62 down. Q2 down min 0.752931 (1/64). Binary down min 0.300575 L58. Residual down min 0.364203 L58. HGRAVS honest min 0.730175; packed-2p0 0.919630 is in-sample. Evidence: g1-mlp-floor-locate.md:86-93,307-310.
C6. Attention 0.99 floor is Q4 g=128 MSE, b_attn 4.125009196169866, 29/29, min 0.990088. Nothing ≤2.07 holds 0.99. Q4 g=64 absmax fails (L47.o 0.9897967539); Q4 g=64 MSE clears (0.991123). Evidence: g1-attention-2bpw-stack.md:16-35,176-187.
C7. Three recipes: G1-C 3.6138111608720234 (−0.6389239659944685 vs G0); G1-Q same bytes, MSE; G1-L 3.5801733386059729 (−0.6725617882605190). Evidence: §3–§4, §8.
C8. Island +0.00249, set {3994,3456,310}, L7 3994 ≡ 0, subadditive with MSE, dead after L0 writes. Not in any recipe. Evidence: g1-channel-3994-island.md:22,50-60,210-214; g1-attention-2bpw-stack.md:200-214.
C9. Every recipe's 192 MLP GEMVs consume q80_hgravs01_factor_matvec_simd3 (slow Uniform bits-3). q3mlp TOKEN_NS/G0 = 148588917/39326090 = 3.77838. Sibling ESTABLISHING 3.78× kernel class. Evidence: hybrid_decode.rs:1687-1688; g1-mlp-family-generate.md:29-42,343-385; metal simd3 bits!=3 early-out.
C10. Native kernels named in §2. HQ30UQ4 MLP refused. g=128 Uniform is a group_size argument, not a new shader. Evidence: hybrid_decode.rs:96-110,1600-1760,3143-3165.

EVIDENCE
E1. crates/hawking-core/src/model/qwen38_pack.rs:673-679
E2. crates/hawking-core/src/model/qwen38_hybrid_decode.rs:96-110,1600-1760,1687-1690,3143-3165
E3. crates/hawking-core/shaders/q80_mixed_decode.metal simd3 (bits != 3 early-out)
E4. .../mixed-q3mlp-v1/catalog.hq38m20 + PACK_REPORT.json (this-lane parse, §3, §8)
E5. .../mixed-q4down-v1/PACK_REPORT.json mlp_physical_bpw 2.2208531248803234
E6. /Users/scammermike/.claude-grok/worktrees/81-mlp-floor-locate-20260817-115439/workspace/superwave/g1/g1-mlp-floor-locate.md
E7. /Users/scammermike/.claude-grok/worktrees/80-attention-2bpw-stack-20260817-115433/workspace/superwave/g1/g1-attention-2bpw-stack.md
E8. /Users/scammermike/.claude-grok/worktrees/74-mse-scale-rule-20260817-113802/workspace/superwave/g1/g1-mse-scale-rule.md
E9. /Users/scammermike/.claude-grok/worktrees/92-mlp-family-generate-20260817-121917/workspace/superwave/g1/g1-mlp-family-generate.md
E10. workspace/superwave/g1/g1-channel-3994-island.md
E11. workspace/superwave/g1/g1-scale-contradiction.md
E12. workspace/superwave/g1/g1-group-partition-geometry.md (L47.o Q4 g=64 absmax 0.9897967539)
E13. workspace/superwave/g1/g1-bit-budget-accounting.md §4
E14. workspace/superwave/g1/g1-mlp-rolelock-unlock.md (Uniform on MLP admitted)
E15. lab/operators/qwen38_mlp_not_r160_pack.py @ 00939c186 (C command)
E16. This-lane python catalog parse + g=128 byte rewrite, §8

CHANGES
created workspace/superwave/g1/g1-recipe-v2.md
no tracked file modified

TESTS
test -s workspace/superwave/g1/g1-recipe-v2.md
wc -l workspace/superwave/g1/g1-recipe-v2.md
git status --porcelain

RISKS
R1. G1-Q/L coherence UNMEASURED. C loops after the gate. MSE can move tokens either way. Capture is thin.
R2. simd3 3.78× is ESTABLISHING at kernel grain; token 3.778× is DIRTY complete-token on the whole C body (attn is also Uniform, not geo_tpr64).
R3. Packer flags for MSE / attn-group do not exist on HEAD. Packing run must add them without changing C's catalog law.
R4. 30 DeltaNet layers unswept for MSE (18 were). Sign consistent; unswept scale planes UNMEASURED.
R5. G1-L g=128 Uniform has no Qwen3.8 token receipt. group_size is wired; that is not a generate.

UNRESOLVED
U1. Native generate of G1-Q vs C oracle. Cheapest next GPU.
U2. Whether HQ30UQ4 retarget of attn+lm_head (same 4.25, geo_tpr64) recovers a slice of the 3.778× without touching MLP q3.
U3. Lane 100's 3.78× kernel microbench (ESTABLISHING). Do not substitute projected TPS.
U4. Thicker mixer-site capture. Could move L47.o Q4 g=256 MSE (0.98935) across 0.99 and drop the family floor from 4.125 to 4.063. Will not drop it to 2.07.

NEXT
N1. Pack G1-Q. Do not re-pack G1-C.
N2. Native greedy G1-Q vs C gate, 0 fallbacks. Then optional complete-wall (DIRTY ok if labeled).
N3. Pack G1-L only if N2 passes and density is the next question.
N4. Do not pack sub-1.5, mixed-2p0-shaped MLP, q4down-shaped MLP, or island-as-codec.
```
