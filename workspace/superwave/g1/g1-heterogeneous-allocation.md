# G1 heterogeneous allocation — Qwen3.8-27B language-only

STATUS: IMPLEMENT_READY for the discrete cheap-codec recipes at 2.0 and 1.5 complete BPW, and for the unconstrained 1.2 recipe. 1.0 is FALSIFIED for this codec family.

Objective solved: minimize `J = sum_i (s_i * e_i(b_i))^2` subject to `8 * sum payload_bytes / 26895998464 <= T` for `T in {2.0, 1.5, 1.2, 1.0}`.

Generation was not run (lane forbid). Organ-level holdout rel-L2 is a screen, not a token-level claim.

---

## 1. Complete BPW accounting (MEASURED catalog, not a bandwidth floor)

Definition, packer authority:

```
// crates/hawking-core/src/model/qwen38_pack.rs:673-678
let source_weight_elements: u64 = rows.iter().map(|row| row.elements).sum();
let tensor_payload_bytes: u64 = rows.iter().map(|row| row.bytes).sum();
complete_physical_bpw = (tensor_payload_bytes as f64 * 8.0) / source_weight_elements as f64
```

Uniform-Q4 catalog on disk:

```
// /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json:2-15
complete_physical_bpw: 4.252735126866492
source_weight_elements: 26895998464
tensor_payload_bytes: 14297694680
q4_tensors: 402
f32_tensors: 353
tensor_count: 755
```

This is the G0 complete-BPW number (`4.252735126866492`). HQ30UQ4 group-64 payload formula reproduces catalog Q4 bytes exactly: `q4_formula_bytes - q4_catalog_bytes = 0` (solver check against every catalog Q4 row).

Language-only, vision skipped (333 tensors). In-proj fused to `qkvz`+`ba` as the packer stores them.

### Inventory (from that catalog)

| class | n | elements | mass % | G0 kind | G0 payload BPW |
|---|---:|---:|---:|---|---:|
| mlp.gate_proj | 64 | 5704253440 | 21.209 | q4 | 4.25000 |
| mlp.up_proj | 64 | 5704253440 | 21.209 | q4 | 4.25000 |
| mlp.down_proj | 64 | 5704253440 | 21.209 | q4 | 4.25000 |
| dn.in_proj_qkvz | 48 | 4026531840 | 14.971 | q4 | 4.25000 |
| dn.out_proj | 48 | 1509949440 | 5.614 | q4 | 4.25001 |
| embed | 1 | 1271398400 | 4.727 | q4 | 4.25000 |
| lm_head | 1 | 1271398400 | 4.727 | q4 | 4.25000 |
| gqa.q_proj | 16 | 1006632960 | 3.743 | q4 | 4.25001 |
| gqa.o_proj | 16 | 503316480 | 1.871 | q4 | 4.25001 |
| gqa.k_proj | 16 | 83886080 | 0.312 | q4 | 4.25006 |
| gqa.v_proj | 16 | 83886080 | 0.312 | q4 | 4.25006 |
| dn.in_proj_ba | 48 | 23592960 | 0.088 | q4 | 4.25065 |
| dn.conv1d | 48 | 1966080 | 0.007 | f32 | 32.00156 |
| input_layernorm | 64 | 327680 | 0.001 | f32 | 32.01250 |
| post_attention_layernorm | 64 | 327680 | 0.001 | f32 | 32.01250 |
| dn.norm / A_log / dt_bias / gqa.q_norm / gqa.k_norm / final_norm | 145 | 24064 | ~0 | f32 | 32–33.3 |
| TOTAL | 755 | 26895998464 | 100 | 402 q4 + 353 f32 | 4.252735 |

Mass fractions in the descent receipt (`mlp 0.6363 / attention_norms 0.2692 / embed_lm_head 0.0945`) match this inventory to the printed 4 digits. Evidence: `receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json:28-31`.

---

## 2. Error-vs-bits curves

Source: `receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` seal `6269ed05963950e19a526c75bc9037db994e6b3a618b94b4d3e519949dfa3735`.

MEASURED: holdout `hold_output_rel_l2` = `||X W^T − X Ŵ^T|| / ||X W^T||` on 64 tokens never used as codec parameters. Capture: 256 real BF16 post-norm hiddens, `fit_n=192`, `hold_n=64`, layers `{0,3,15,31,47,63}`. Roles scored in output space: `gate_proj, up_proj, down_proj, attn_in`. `attn_out` is weight-space only (`out_proj` in-dim 6144 ≠ captured 5120).

2-bit rung is **ternary_t0.7_g128**, not affine q2. Descent finding: at identical ~2.25 BPW, ternary hold means 0.90–0.93 vs affine-q2 0.82–0.87. Affine q2 is dominated; the allocator never wants it.

### MEASURED hold_output_rel_l2 (selected layers)

| layer | role | b=1 binary | b=2 ternary | b=3 q3 | b=4 q4 |
|---:|---|---:|---:|---:|---:|
| 0 | gate | 0.5221 | 0.3645 | 0.1922 | 0.0821 |
| 0 | up | 0.5258 | 0.3671 | 0.1976 | 0.0849 |
| 0 | down | 0.4617 | 0.3444 | 0.1263 | 0.0538 |
| 0 | attn_in (DN qkv) | 0.5487 | 0.3949 | 0.2067 | 0.0883 |
| 3 | up | 0.5733 | 0.4084 | 0.2351 | 0.1008 |
| 31 | up | 0.6593 | 0.4298 | 0.2585 | 0.1157 |
| 31 | down | 0.5801 | 0.4209 | 0.2319 | 0.0991 |
| 47 | down | 0.6257 | 0.4766 | 0.2382 | 0.1021 |
| 47 | up | 0.6198 | 0.4323 | 0.2537 | 0.1105 |
| 63 | gate | 0.3931 | 0.2499 | 0.1100 | 0.0471 |
| 63 | up | 0.3884 | 0.2304 | 0.0930 | 0.0407 |
| 63 | down | 0.6917 | 0.5391 | 0.2366 | 0.1009 |
| 63 | attn_in (GQA q) | 0.4231 | 0.2727 | 0.1353 | 0.0586 |

Late-layer `down_proj` is the hard organ (L63 binary hold L2 0.692, cosine 0.730). Early `down` and late `gate/up` are easy. Uniform bits cannot spend on that split.

### PROXY labels (not measured output error)

| tensor | proxy | justification |
|---|---|---|
| unmeasured MLP layers | linear interp in layer index among `{0,3,15,31,47,63}` | same role, same shape |
| all 48 DN `in_proj_qkvz` / `out_proj` | L0 `attn_in` / `attn_out` | only L0 DN was scored; qkvz is fused qkv+z, scored as qkv only |
| GQA unmeasured layers | interp among `{3,15,31,47,63}` | GQA interval |
| `gqa.k_proj`, `gqa.v_proj` | GQA `attn_in` (q_proj) curve | same X, different W |
| `dn.in_proj_ba` | DN `attn_in` curve | same X; 0.088% mass |
| `dn.out_proj`, `gqa.o_proj` | `attn_out` `weight_rel_l2` | no matching activation width |
| embed, lm_head | mean `weight_rel_l2` over all scored organs at that rung | descent `lm_head_not_output_scored: true` |

`corr(weight_rel_l2, hold_output_rel_l2) = 0.9559` on 96 measured (layer, role, rung) pairs. Weight-L2 is an adequate stand-in for `attn_out`. It is still a proxy.

Mean weight-L2 used for embed/lm_head: b1=0.6074, b2=0.4429, b3=0.2587, b4=0.1110.

No per-tensor Hessian / Fisher exists in the materialized tree or in the Qwen3.8 receipts. The descent hold-L2 **is** the per-tensor error curve. Activation RMS below is the sensitivity weight, not a substitute for the curve.

---

## 3. Sensitivity weights (PROXY)

Activation capture: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json`

```
schema: hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1
status: CAPTURED_REAL_BF16_POST_NORM_HIDDEN
n_tokens: 256  n_layers: 64  hidden: 5120
not_synthetic: true
```

Hidden RMS computed from `hidden/L00.f32` … `L63.f32` (each 256×5120 f32, 5242880 bytes).

| | value |
|---|---|
| mean RMS | 0.64955 |
| min RMS | 0.09979 (L0), rms_norm=0.154 |
| max RMS | 1.25902 (L61), rms_norm=1.938 |
| L63 RMS | 1.16680, rms_norm=1.796 |
| range | 12.6× late/early |

Primary weight: `s_i = rms_norm[input_layer] * class_prior`.

class_prior is the Gravity evidence-derived organ prior (`tools/condense/gravity_global_allocator.py` GPT-OSS table), **not** a Qwen3.8 Jacobian. Values used: `lm_head 2.0, embed 1.5, down 1.4, attn in/out 1.3, k/v 1.2, up 1.1, gate 1.0, ba 0.4`.

Input layer for embed = L0 hidden; for lm_head = L63 hidden; for layer tensors = that layer's captured post-norm hidden.

Protection floors (PROXY policy, not measured): `lm_head min=3`, `embed min=2`, `dn.in_proj_ba min=2`. Small tensors pinned f32. Unconstrained (all GEMV min=1) is also solved.

Robustness check: `s_i = 1` (unit weight) still beats uniform by 27.8% at 2.0 and 22.9% at 1.5. Direction of the gap does not depend on the Gravity priors.

---

## 4. Optimizer

Discrete rungs a packing run can consume:

| bits | codec | payload |
|---:|---|---|
| 1 | `binary_g128` | header 261 + f16 scale / 128 + 1-bit signs (calibrated to descent L0 gate 12534021 bytes / 89128960) |
| 2 | `ternary_t0.7_g128` | 2-bit codes + two f16 tables / 128 (calibrated to descent 25067853 / 89128960) |
| 3 | `uniform_q3_g64` | HQ30UQ4-faithful: `32+4*rank + groups*2 + groups*64*bits/8` |
| 4 | `uniform_q4_g64` | same; matches catalog byte-for-byte |
| 32 | `f32v2` | `8 + 4*n` |

Algorithm: gravity greedy marginal (`tools/condense/gravity_global_allocator.py:400-436`). Start at floors. Repeatedly raise the organ with largest `ΔJ / Δbytes` that still fits. On concave separable `J` this weakly dominates any other discrete assignment in the same budget, uniform included.

Uniform comparator at exact T: every free tensor uses the same two-rung mix `(1-p)*lo + p*hi` so complete BPW = T. `J_uniform` uses linearly mixed `e`. This is the equal-BPW uniform baseline, not "nearest integer bits".

Solver: `/tmp/g1_hetero_alloc.py` → `/tmp/g1_hetero_alloc_out.json`. CPU only. No pack, no GPU, no weight reload.

---

## 5. Feasibility

| assignment | complete BPW | bytes |
|---|---:|---:|
| all binary + f32 small | 1.128069 | 3792567522 |
| protected floor (lm_head=3, embed=2, ba≥2) | 1.282686 | 4312390661 |
| all ternary + f32 | 2.252958 | 7574445282 |
| all q3 + f32 | 3.252833 | 10936025560 |
| all q4 + f32 (= G0 catalog) | 4.252735 | 14297694680 |

| T | budget bytes | primary floors | unconstrained |
|---:|---:|---|---|
| 2.0 | 6723999616 | FEASIBLE 1.999998 | FEASIBLE 1.999983 |
| 1.5 | 5042999712 | FEASIBLE 1.499991 | FEASIBLE 1.499990 |
| 1.2 | 4034399769 | INFEASIBLE (floor 1.282686) | FEASIBLE 1.199997 |
| 1.0 | 3361999808 | INFEASIBLE | INFEASIBLE (floor 1.128069) |

**1.0 KILLS** for the cheap integer-bit family. Binary is the densest in-register codec in the descent catalog. Hitting 1.0 requires a sub-1-bit organ on substantial mass.

HGRAVS01 r160_b3 on `down_proj` is the only measured sub-bit codec: `0.131617` physical BPW, 93847197 bytes / 5704253440 elements (`mixed-2p0-v1/PACK_REPORT.json:28-32`). Putting all 64 downs there and everything else at binary projects ~0.917 complete BPW, which would then have slack to 1.0. That is **not** a recommended recipe: mixed-2p0 used HGRAVS01 down + binary gate + rice up + Q4 attention at **2.085593** complete BPW and is **INCOHERENT** on the native reader (`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:7-10`). A 1.0 pack that also 1-bits attention and embed/lm_head is a stricter version of a recipe that already failed. Isolated HGRAVS01 hold_output_rel_l2 was never scored. REOPEN_IF that number is measured and a native generate of a down-only-HGRAVS01 pack is coherent.

Rice (`~1.288` BPW) is denser-worse than binary and was in the incoherent sub-1.5 pack. Excluded from the rungs.

---

## 6. How much hetero beats uniform at equal complete BPW

J is the screen objective (sum of squared weighted organ rel-L2). Not generation. Not TPS.

| T | policy | J_hetero | J_uniform | abs Δ | rel reduction | uniform mix |
|---:|---|---:|---:|---:|---:|---|
| 2.0 | primary floors | 67.433 | 144.838 | 77.405 | **53.44%** | 70.5% ternary / 29.5% binary |
| 2.0 | unconstrained | 59.031 | 140.436 | 81.405 | **57.97%** | 77.5% ternary / 22.5% binary |
| 2.0 | unit-weight floors | 58.427 | 80.933 | 22.506 | **27.81%** | same mix as floors |
| 1.5 | primary floors | 133.712 | 197.109 | 63.396 | **32.16%** | 21.4% ternary / 78.6% binary |
| 1.5 | unconstrained | 110.573 | 187.864 | 77.291 | **41.14%** | 33.1% ternary / 66.9% binary |
| 1.5 | unit-weight floors | 83.722 | 108.607 | 24.885 | **22.91%** | same mix as floors |
| 1.2 | unconstrained | 180.319 | 219.707 | 39.388 | **17.93%** | 6.4% ternary / 93.6% binary |
| 1.0 | — | n/a | n/a | n/a | TARGET_MISSED | — |

The gap is not small. On this screen, the heterogeneous family is worth its complexity at 2.0 and 1.5. At 1.2 the gap shrinks because almost everything is already at the 1-bit floor.

Capability caveat (MEASURED, not this lane): a *different* heterogeneous recipe at 2.0856 complete BPW is generation-incoherent. This allocator's 2.0 table is not that recipe (no rice, no HGRAVS01, late-layer downs at 3–4 bits instead of 0.13). Generation remains the gate. Cheapest experiment: pack the 2.0 table into HQ38M20 and run the existing native generate harness under the GPU lane.

vs uniform q4 (not equal-BPW): J_q4 = 6.240. The 2.0 hetero J=67.4 is 10.8× worse than G0 q4 on the screen — expected, we removed more than half the bits. The relevant comparison is equal-BPW uniform, above.

---

## 7. Published recipes

Pinned in every recipe (not listed per layer):

```
input_layernorm              f32v2   32
post_attention_layernorm     f32v2   32
final_norm                   f32v2   32
dn.conv1d                    f32v2   32
dn.A_log                     f32v2   32
dn.dt_bias                   f32v2   32
dn.norm                      f32v2   32
gqa.q_norm                   f32v2   32
gqa.k_norm                   f32v2   32
```

Empty GQA cells on DeltaNet layers and empty DN cells on GQA layers are structural, not omissions. Mixer rule: GQA iff `(layer+1)%4==0` (`qwen38_geometry.rs:20-23`).

### Target 2.0

primary_floors. embed=2 ternary, lm_head=3 q3. Discretionary bits go to late-layer down / out_proj / qkvz / GQA k,v,o. Early MLP stays 1-bit. 461 raises, slack 5634 bytes.

- policy: `primary_floors`
- feasible: `True`
- achieved complete physical BPW: `1.999998324211683`  **ESTIMATED** from payload formulas
- payload bytes: `6723993982`
- budget bytes: `6723999616`
- slack bytes: `5634`
- J: `67.43295100810795`
- embed bits: `2`
- lm_head bits: `3`

Class bit histogram (count of tensors):

```
{
  "dn.A_log": {
    "32": 48
  },
  "dn.conv1d": {
    "32": 48
  },
  "dn.dt_bias": {
    "32": 48
  },
  "dn.in_proj_ba": {
    "3": 4,
    "4": 44
  },
  "dn.in_proj_qkvz": {
    "1": 27,
    "2": 2,
    "3": 19
  },
  "dn.norm": {
    "32": 48
  },
  "dn.out_proj": {
    "1": 15,
    "3": 8,
    "4": 25
  },
  "embed": {
    "2": 1
  },
  "final_norm": {
    "32": 1
  },
  "gqa.k_norm": {
    "32": 16
  },
  "gqa.k_proj": {
    "2": 1,
    "3": 3,
    "4": 12
  },
  "gqa.o_proj": {
    "1": 4,
    "2": 1,
    "3": 2,
    "4": 9
  },
  "gqa.q_norm": {
    "32": 16
  },
  "gqa.q_proj": {
    "1": 6,
    "2": 2,
    "3": 8
  },
  "gqa.v_proj": {
    "1": 1,
    "3": 3,
    "4": 12
  },
  "input_layernorm": {
    "32": 64
  },
  "lm_head": {
    "3": 1
  },
  "mlp.down_proj": {
    "1": 31,
    "2": 3,
    "3": 26,
    "4": 4
  },
  "mlp.gate_proj": {
    "1": 59,
    "2": 5
  },
  "mlp.up_proj": {
    "1": 33,
    "2": 31
  },
  "post_attention_layernorm": {
    "32": 64
  }
}
```

Per-layer GEMV bit widths (packer recipe):

```csv
layer,mixer,mlp.gate_proj,mlp.up_proj,mlp.down_proj,dn.in_proj_qkvz,dn.in_proj_ba,dn.out_proj,gqa.q_proj,gqa.k_proj,gqa.v_proj,gqa.o_proj
0,delta_net,1,1,1,1,3,1,,,,
1,delta_net,1,1,1,1,3,1,,,,
2,delta_net,1,1,1,1,3,1,,,,
3,gqa,1,1,1,,,,1,2,1,1
4,delta_net,1,1,1,1,3,1,,,,
5,delta_net,1,1,1,1,4,1,,,,
6,delta_net,1,1,1,1,4,1,,,,
7,gqa,1,1,1,,,,1,3,3,1
8,delta_net,1,1,1,1,4,1,,,,
9,delta_net,1,1,1,1,4,1,,,,
10,delta_net,1,1,1,1,4,1,,,,
11,gqa,1,1,1,,,,1,3,3,1
12,delta_net,1,1,1,1,4,1,,,,
13,delta_net,1,1,1,1,4,1,,,,
14,delta_net,1,1,1,1,4,1,,,,
15,gqa,1,1,1,,,,1,3,3,1
16,delta_net,1,1,1,1,4,1,,,,
17,delta_net,1,1,1,1,4,1,,,,
18,delta_net,1,1,1,1,4,1,,,,
19,gqa,1,1,1,,,,1,4,4,2
20,delta_net,1,1,1,1,4,3,,,,
21,delta_net,1,1,1,1,4,3,,,,
22,delta_net,1,1,1,1,4,3,,,,
23,gqa,1,1,1,,,,1,4,4,3
24,delta_net,1,1,1,1,4,3,,,,
25,delta_net,1,1,1,1,4,3,,,,
26,delta_net,1,1,1,1,4,3,,,,
27,gqa,1,1,1,,,,2,4,4,3
28,delta_net,1,1,1,1,4,3,,,,
29,delta_net,1,1,1,1,4,3,,,,
30,delta_net,1,1,1,1,4,4,,,,
31,gqa,1,1,2,,,,2,4,4,4
32,delta_net,1,1,2,1,4,4,,,,
33,delta_net,1,2,2,1,4,4,,,,
34,delta_net,1,2,3,2,4,4,,,,
35,gqa,1,2,3,,,,3,4,4,4
36,delta_net,1,2,3,2,4,4,,,,
37,delta_net,1,2,3,1,4,4,,,,
38,delta_net,1,2,3,3,4,4,,,,
39,gqa,1,2,3,,,,3,4,4,4
40,delta_net,1,2,3,3,4,4,,,,
41,delta_net,1,2,3,3,4,4,,,,
42,delta_net,1,2,3,3,4,4,,,,
43,gqa,1,2,3,,,,3,4,4,4
44,delta_net,1,2,3,3,4,4,,,,
45,delta_net,1,2,3,3,4,4,,,,
46,delta_net,1,2,3,3,4,4,,,,
47,gqa,1,2,3,,,,3,4,4,4
48,delta_net,1,2,3,3,4,4,,,,
49,delta_net,1,2,3,3,4,4,,,,
50,delta_net,1,2,3,3,4,4,,,,
51,gqa,1,2,3,,,,3,4,4,4
52,delta_net,1,2,3,3,4,4,,,,
53,delta_net,1,2,3,3,4,4,,,,
54,delta_net,1,2,3,3,4,4,,,,
55,gqa,1,2,3,,,,3,4,4,4
56,delta_net,1,2,3,3,4,4,,,,
57,delta_net,1,2,3,3,4,4,,,,
58,delta_net,2,2,3,3,4,4,,,,
59,gqa,2,2,4,,,,3,4,4,4
60,delta_net,2,2,4,3,4,4,,,,
61,delta_net,2,2,4,3,4,4,,,,
62,delta_net,2,2,4,3,4,4,,,,
63,gqa,1,2,3,,,,3,4,4,4
```

<details><summary>machine-readable JSON for this target</summary>

```json
{
  "target_complete_bpw": 2.0,
  "policy": "primary_floors",
  "feasible": true,
  "achieved_complete_physical_bpw": 1.999998324211683,
  "tensor_payload_bytes": 6723993982,
  "budget_bytes": 6723999616,
  "slack_bytes": 5634,
  "objective_J": 67.43295100810795,
  "reason": null,
  "codec_map": {
    "1": "binary_g128",
    "2": "ternary_t0.7_g128",
    "3": "uniform_q3_g64",
    "4": "uniform_q4_g64",
    "32": "f32v2"
  },
  "globals": {
    "embed": 2,
    "lm_head": 3,
    "final_norm": 32
  },
  "pinned_f32_classes": [
    "input_layernorm",
    "post_attention_layernorm",
    "final_norm",
    "dn.conv1d",
    "dn.A_log",
    "dn.dt_bias",
    "dn.norm",
    "gqa.q_norm",
    "gqa.k_norm"
  ],
  "layers": [
    {
      "layer": 0,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 1,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 2,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 3,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 2,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 4,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 5,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 6,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 7,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 8,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 9,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 10,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 11,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 12,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 13,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 14,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 15,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 16,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 17,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 18,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 19,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 20,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 21,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 22,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 23,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 24,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 25,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 26,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 27,
      "mixer": "gqa",
      "gqa.q_proj": 2,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 28,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 29,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 30,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 31,
      "mixer": "gqa",
      "gqa.q_proj": 2,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 2
    },
    {
      "layer": 32,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 2
    },
    {
      "layer": 33,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 2
    },
    {
      "layer": 34,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 2,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 35,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 36,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 2,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 37,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 38,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 39,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 40,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 41,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 42,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 43,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 44,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 45,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 46,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 47,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 48,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 49,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 50,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 51,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 52,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 53,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 54,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 55,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 56,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 57,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 58,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 2,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    },
    {
      "layer": 59,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 2,
      "mlp.up_proj": 2,
      "mlp.down_proj": 4
    },
    {
      "layer": 60,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 2,
      "mlp.up_proj": 2,
      "mlp.down_proj": 4
    },
    {
      "layer": 61,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 2,
      "mlp.up_proj": 2,
      "mlp.down_proj": 4
    },
    {
      "layer": 62,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 2,
      "mlp.up_proj": 2,
      "mlp.down_proj": 4
    },
    {
      "layer": 63,
      "mixer": "gqa",
      "gqa.q_proj": 3,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 2,
      "mlp.down_proj": 3
    }
  ]
}
```

</details>

### Target 1.5

primary_floors. embed=2, lm_head=3. Almost all MLP stays 1-bit; last ~11 downs raised to 3. Attention out_proj and GQA k/v take the rest. 254 raises, slack 30504 bytes.

- policy: `primary_floors`
- feasible: `True`
- achieved complete physical BPW: `1.4999909268287501`  **ESTIMATED** from payload formulas
- payload bytes: `5042969208`
- budget bytes: `5042999712`
- slack bytes: `30504`
- J: `133.71247380873697`
- embed bits: `2`
- lm_head bits: `3`

Class bit histogram (count of tensors):

```
{
  "dn.A_log": {
    "32": 48
  },
  "dn.conv1d": {
    "32": 48
  },
  "dn.dt_bias": {
    "32": 48
  },
  "dn.in_proj_ba": {
    "2": 7,
    "3": 8,
    "4": 33
  },
  "dn.in_proj_qkvz": {
    "1": 44,
    "2": 1,
    "3": 3
  },
  "dn.norm": {
    "32": 48
  },
  "dn.out_proj": {
    "1": 22,
    "2": 2,
    "3": 17,
    "4": 7
  },
  "embed": {
    "2": 1
  },
  "final_norm": {
    "32": 1
  },
  "gqa.k_norm": {
    "32": 16
  },
  "gqa.k_proj": {
    "1": 3,
    "3": 3,
    "4": 10
  },
  "gqa.o_proj": {
    "1": 7,
    "2": 1,
    "3": 5,
    "4": 3
  },
  "gqa.q_norm": {
    "32": 16
  },
  "gqa.q_proj": {
    "1": 14,
    "2": 2
  },
  "gqa.v_proj": {
    "1": 3,
    "3": 3,
    "4": 10
  },
  "input_layernorm": {
    "32": 64
  },
  "lm_head": {
    "3": 1
  },
  "mlp.down_proj": {
    "1": 53,
    "3": 11
  },
  "mlp.gate_proj": {
    "1": 64
  },
  "mlp.up_proj": {
    "1": 64
  },
  "post_attention_layernorm": {
    "32": 64
  }
}
```

Per-layer GEMV bit widths (packer recipe):

```csv
layer,mixer,mlp.gate_proj,mlp.up_proj,mlp.down_proj,dn.in_proj_qkvz,dn.in_proj_ba,dn.out_proj,gqa.q_proj,gqa.k_proj,gqa.v_proj,gqa.o_proj
0,delta_net,1,1,1,1,2,1,,,,
1,delta_net,1,1,1,1,2,1,,,,
2,delta_net,1,1,1,1,2,1,,,,
3,gqa,1,1,1,,,,1,1,1,1
4,delta_net,1,1,1,1,2,1,,,,
5,delta_net,1,1,1,1,2,1,,,,
6,delta_net,1,1,1,1,3,1,,,,
7,gqa,1,1,1,,,,1,1,1,1
8,delta_net,1,1,1,1,2,1,,,,
9,delta_net,1,1,1,1,2,1,,,,
10,delta_net,1,1,1,1,3,1,,,,
11,gqa,1,1,1,,,,1,1,1,1
12,delta_net,1,1,1,1,3,1,,,,
13,delta_net,1,1,1,1,3,1,,,,
14,delta_net,1,1,1,1,3,1,,,,
15,gqa,1,1,1,,,,1,3,3,1
16,delta_net,1,1,1,1,3,1,,,,
17,delta_net,1,1,1,1,3,1,,,,
18,delta_net,1,1,1,1,3,1,,,,
19,gqa,1,1,1,,,,1,3,3,1
20,delta_net,1,1,1,1,4,1,,,,
21,delta_net,1,1,1,1,4,1,,,,
22,delta_net,1,1,1,1,4,1,,,,
23,gqa,1,1,1,,,,1,3,3,1
24,delta_net,1,1,1,1,4,1,,,,
25,delta_net,1,1,1,1,4,1,,,,
26,delta_net,1,1,1,1,4,1,,,,
27,gqa,1,1,1,,,,1,4,4,1
28,delta_net,1,1,1,1,4,1,,,,
29,delta_net,1,1,1,1,4,2,,,,
30,delta_net,1,1,1,1,4,2,,,,
31,gqa,1,1,1,,,,1,4,4,2
32,delta_net,1,1,1,1,4,3,,,,
33,delta_net,1,1,1,1,4,3,,,,
34,delta_net,1,1,1,1,4,3,,,,
35,gqa,1,1,1,,,,1,4,4,3
36,delta_net,1,1,1,1,4,3,,,,
37,delta_net,1,1,1,1,4,3,,,,
38,delta_net,1,1,1,1,4,3,,,,
39,gqa,1,1,1,,,,1,4,4,3
40,delta_net,1,1,1,1,4,3,,,,
41,delta_net,1,1,1,1,4,3,,,,
42,delta_net,1,1,1,1,4,3,,,,
43,gqa,1,1,1,,,,1,4,4,3
44,delta_net,1,1,1,1,4,3,,,,
45,delta_net,1,1,1,1,4,3,,,,
46,delta_net,1,1,1,1,4,3,,,,
47,gqa,1,1,1,,,,1,4,4,3
48,delta_net,1,1,1,1,4,3,,,,
49,delta_net,1,1,1,1,4,3,,,,
50,delta_net,1,1,1,1,4,3,,,,
51,gqa,1,1,1,,,,1,4,4,3
52,delta_net,1,1,1,1,4,3,,,,
53,delta_net,1,1,3,1,4,3,,,,
54,delta_net,1,1,3,1,4,4,,,,
55,gqa,1,1,3,,,,1,4,4,4
56,delta_net,1,1,3,1,4,4,,,,
57,delta_net,1,1,3,1,4,4,,,,
58,delta_net,1,1,3,2,4,4,,,,
59,gqa,1,1,3,,,,2,4,4,4
60,delta_net,1,1,3,3,4,4,,,,
61,delta_net,1,1,3,3,4,4,,,,
62,delta_net,1,1,3,3,4,4,,,,
63,gqa,1,1,3,,,,2,4,4,4
```

<details><summary>machine-readable JSON for this target</summary>

```json
{
  "target_complete_bpw": 1.5,
  "policy": "primary_floors",
  "feasible": true,
  "achieved_complete_physical_bpw": 1.4999909268287501,
  "tensor_payload_bytes": 5042969208,
  "budget_bytes": 5042999712,
  "slack_bytes": 30504,
  "objective_J": 133.71247380873697,
  "reason": null,
  "codec_map": {
    "1": "binary_g128",
    "2": "ternary_t0.7_g128",
    "3": "uniform_q3_g64",
    "4": "uniform_q4_g64",
    "32": "f32v2"
  },
  "globals": {
    "embed": 2,
    "lm_head": 3,
    "final_norm": 32
  },
  "pinned_f32_classes": [
    "input_layernorm",
    "post_attention_layernorm",
    "final_norm",
    "dn.conv1d",
    "dn.A_log",
    "dn.dt_bias",
    "dn.norm",
    "gqa.q_norm",
    "gqa.k_norm"
  ],
  "layers": [
    {
      "layer": 0,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 1,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 2,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 3,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 4,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 5,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 6,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 7,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 8,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 9,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 2,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 10,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 11,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 12,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 13,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 14,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 15,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 16,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 17,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 18,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 19,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 20,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 21,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 22,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 23,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 24,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 25,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 26,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 27,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 28,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 29,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 30,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 31,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 32,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 33,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 34,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 35,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 36,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 37,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 38,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 39,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 40,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 41,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 42,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 43,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 44,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 45,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 46,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 47,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 48,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 49,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 50,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 51,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 52,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 53,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 54,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 55,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 56,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 57,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 58,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 2,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 59,
      "mixer": "gqa",
      "gqa.q_proj": 2,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 60,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 61,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 62,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 3,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 63,
      "mixer": "gqa",
      "gqa.q_proj": 2,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 4,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    }
  ]
}
```

</details>

### Target 1.2

primary_unconstrained — floors (lm_head=3, embed=2) make 1.2 infeasible. embed=1, lm_head=1. MLP is 1-bit except L61 down=3. Slack 11184 bytes.

- policy: `primary_unconstrained`
- feasible: `True`
- achieved complete physical BPW: `1.19999667323003`  **ESTIMATED** from payload formulas
- payload bytes: `4034388585`
- budget bytes: `4034399769`
- slack bytes: `11184`
- J: `180.31865535741125`
- embed bits: `1`
- lm_head bits: `1`

Class bit histogram (count of tensors):

```
{
  "dn.A_log": {
    "32": 48
  },
  "dn.conv1d": {
    "32": 48
  },
  "dn.dt_bias": {
    "32": 48
  },
  "dn.in_proj_ba": {
    "1": 12,
    "3": 7,
    "4": 29
  },
  "dn.in_proj_qkvz": {
    "1": 48
  },
  "dn.norm": {
    "32": 48
  },
  "dn.out_proj": {
    "1": 30,
    "2": 6,
    "3": 12
  },
  "embed": {
    "1": 1
  },
  "final_norm": {
    "32": 1
  },
  "gqa.k_norm": {
    "32": 16
  },
  "gqa.k_proj": {
    "1": 4,
    "3": 5,
    "4": 7
  },
  "gqa.o_proj": {
    "1": 10,
    "2": 2,
    "3": 4
  },
  "gqa.q_norm": {
    "32": 16
  },
  "gqa.q_proj": {
    "1": 16
  },
  "gqa.v_proj": {
    "1": 4,
    "3": 5,
    "4": 7
  },
  "input_layernorm": {
    "32": 64
  },
  "lm_head": {
    "1": 1
  },
  "mlp.down_proj": {
    "1": 63,
    "3": 1
  },
  "mlp.gate_proj": {
    "1": 64
  },
  "mlp.up_proj": {
    "1": 64
  },
  "post_attention_layernorm": {
    "32": 64
  }
}
```

Per-layer GEMV bit widths (packer recipe):

```csv
layer,mixer,mlp.gate_proj,mlp.up_proj,mlp.down_proj,dn.in_proj_qkvz,dn.in_proj_ba,dn.out_proj,gqa.q_proj,gqa.k_proj,gqa.v_proj,gqa.o_proj
0,delta_net,1,1,1,1,1,1,,,,
1,delta_net,1,1,1,1,1,1,,,,
2,delta_net,1,1,1,1,1,1,,,,
3,gqa,1,1,1,,,,1,1,1,1
4,delta_net,1,1,1,1,1,1,,,,
5,delta_net,1,1,1,1,1,1,,,,
6,delta_net,1,1,1,1,1,1,,,,
7,gqa,1,1,1,,,,1,1,1,1
8,delta_net,1,1,1,1,1,1,,,,
9,delta_net,1,1,1,1,1,1,,,,
10,delta_net,1,1,1,1,1,1,,,,
11,gqa,1,1,1,,,,1,1,1,1
12,delta_net,1,1,1,1,1,1,,,,
13,delta_net,1,1,1,1,1,1,,,,
14,delta_net,1,1,1,1,1,1,,,,
15,gqa,1,1,1,,,,1,1,1,1
16,delta_net,1,1,1,1,3,1,,,,
17,delta_net,1,1,1,1,3,1,,,,
18,delta_net,1,1,1,1,3,1,,,,
19,gqa,1,1,1,,,,1,3,3,1
20,delta_net,1,1,1,1,3,1,,,,
21,delta_net,1,1,1,1,3,1,,,,
22,delta_net,1,1,1,1,3,1,,,,
23,gqa,1,1,1,,,,1,3,3,1
24,delta_net,1,1,1,1,3,1,,,,
25,delta_net,1,1,1,1,4,1,,,,
26,delta_net,1,1,1,1,4,1,,,,
27,gqa,1,1,1,,,,1,3,3,1
28,delta_net,1,1,1,1,4,1,,,,
29,delta_net,1,1,1,1,4,1,,,,
30,delta_net,1,1,1,1,4,1,,,,
31,gqa,1,1,1,,,,1,3,3,1
32,delta_net,1,1,1,1,4,1,,,,
33,delta_net,1,1,1,1,4,1,,,,
34,delta_net,1,1,1,1,4,1,,,,
35,gqa,1,1,1,,,,1,3,3,1
36,delta_net,1,1,1,1,4,1,,,,
37,delta_net,1,1,1,1,4,1,,,,
38,delta_net,1,1,1,1,4,1,,,,
39,gqa,1,1,1,,,,1,4,4,1
40,delta_net,1,1,1,1,4,2,,,,
41,delta_net,1,1,1,1,4,2,,,,
42,delta_net,1,1,1,1,4,2,,,,
43,gqa,1,1,1,,,,1,4,4,2
44,delta_net,1,1,1,1,4,2,,,,
45,delta_net,1,1,1,1,4,2,,,,
46,delta_net,1,1,1,1,4,2,,,,
47,gqa,1,1,1,,,,1,4,4,2
48,delta_net,1,1,1,1,4,3,,,,
49,delta_net,1,1,1,1,4,3,,,,
50,delta_net,1,1,1,1,4,3,,,,
51,gqa,1,1,1,,,,1,4,4,3
52,delta_net,1,1,1,1,4,3,,,,
53,delta_net,1,1,1,1,4,3,,,,
54,delta_net,1,1,1,1,4,3,,,,
55,gqa,1,1,1,,,,1,4,4,3
56,delta_net,1,1,1,1,4,3,,,,
57,delta_net,1,1,1,1,4,3,,,,
58,delta_net,1,1,1,1,4,3,,,,
59,gqa,1,1,1,,,,1,4,4,3
60,delta_net,1,1,1,1,4,3,,,,
61,delta_net,1,1,3,1,4,3,,,,
62,delta_net,1,1,1,1,4,3,,,,
63,gqa,1,1,1,,,,1,4,4,3
```

<details><summary>machine-readable JSON for this target</summary>

```json
{
  "target_complete_bpw": 1.2,
  "policy": "primary_unconstrained",
  "feasible": true,
  "achieved_complete_physical_bpw": 1.19999667323003,
  "tensor_payload_bytes": 4034388585,
  "budget_bytes": 4034399769,
  "slack_bytes": 11184,
  "objective_J": 180.31865535741125,
  "reason": null,
  "codec_map": {
    "1": "binary_g128",
    "2": "ternary_t0.7_g128",
    "3": "uniform_q3_g64",
    "4": "uniform_q4_g64",
    "32": "f32v2"
  },
  "globals": {
    "embed": 1,
    "lm_head": 1,
    "final_norm": 32
  },
  "pinned_f32_classes": [
    "input_layernorm",
    "post_attention_layernorm",
    "final_norm",
    "dn.conv1d",
    "dn.A_log",
    "dn.dt_bias",
    "dn.norm",
    "gqa.q_norm",
    "gqa.k_norm"
  ],
  "layers": [
    {
      "layer": 0,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 1,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 2,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 3,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 4,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 5,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 6,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 7,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 8,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 9,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 10,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 11,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 12,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 13,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 14,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 15,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 16,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 17,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 18,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 19,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 20,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 21,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 22,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 23,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 24,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 3,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 25,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 26,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 27,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 28,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 29,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 30,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 31,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 32,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 33,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 34,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 35,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 3,
      "gqa.v_proj": 3,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 36,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 37,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 38,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 39,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 1,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 40,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 41,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 42,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 43,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 44,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 45,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 46,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 47,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 2,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 48,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 49,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 50,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 51,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 52,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 53,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 54,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 55,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 56,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 57,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 58,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 59,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 60,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 61,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 3
    },
    {
      "layer": 62,
      "mixer": "delta_net",
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 4,
      "dn.out_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    },
    {
      "layer": 63,
      "mixer": "gqa",
      "gqa.q_proj": 1,
      "gqa.k_proj": 4,
      "gqa.v_proj": 4,
      "gqa.o_proj": 3,
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1
    }
  ]
}
```

</details>

### Target 1.0

TARGET_MISSED. Table is the unconstrained 1-bit floor (complete BPW 1.128069). Not a 1.0 pack.

- policy: `primary_unconstrained`
- feasible: `False`
- achieved complete physical BPW: `1.1280689287891834`  **ESTIMATED** from payload formulas
- payload bytes: `3792567522`
- budget bytes: `3361999808`
- slack bytes: `None`
- J: `227.7197422056343`
- embed bits: `1`
- lm_head bits: `1`

Per-layer GEMV bit widths (packer recipe):

```csv
layer,mixer,mlp.gate_proj,mlp.up_proj,mlp.down_proj,dn.in_proj_qkvz,dn.in_proj_ba,dn.out_proj,gqa.q_proj,gqa.k_proj,gqa.v_proj,gqa.o_proj
0,delta_net,1,1,1,1,1,1,,,,
1,delta_net,1,1,1,1,1,1,,,,
2,delta_net,1,1,1,1,1,1,,,,
3,gqa,1,1,1,,,,1,1,1,1
4,delta_net,1,1,1,1,1,1,,,,
5,delta_net,1,1,1,1,1,1,,,,
6,delta_net,1,1,1,1,1,1,,,,
7,gqa,1,1,1,,,,1,1,1,1
8,delta_net,1,1,1,1,1,1,,,,
9,delta_net,1,1,1,1,1,1,,,,
10,delta_net,1,1,1,1,1,1,,,,
11,gqa,1,1,1,,,,1,1,1,1
12,delta_net,1,1,1,1,1,1,,,,
13,delta_net,1,1,1,1,1,1,,,,
14,delta_net,1,1,1,1,1,1,,,,
15,gqa,1,1,1,,,,1,1,1,1
16,delta_net,1,1,1,1,1,1,,,,
17,delta_net,1,1,1,1,1,1,,,,
18,delta_net,1,1,1,1,1,1,,,,
19,gqa,1,1,1,,,,1,1,1,1
20,delta_net,1,1,1,1,1,1,,,,
21,delta_net,1,1,1,1,1,1,,,,
22,delta_net,1,1,1,1,1,1,,,,
23,gqa,1,1,1,,,,1,1,1,1
24,delta_net,1,1,1,1,1,1,,,,
25,delta_net,1,1,1,1,1,1,,,,
26,delta_net,1,1,1,1,1,1,,,,
27,gqa,1,1,1,,,,1,1,1,1
28,delta_net,1,1,1,1,1,1,,,,
29,delta_net,1,1,1,1,1,1,,,,
30,delta_net,1,1,1,1,1,1,,,,
31,gqa,1,1,1,,,,1,1,1,1
32,delta_net,1,1,1,1,1,1,,,,
33,delta_net,1,1,1,1,1,1,,,,
34,delta_net,1,1,1,1,1,1,,,,
35,gqa,1,1,1,,,,1,1,1,1
36,delta_net,1,1,1,1,1,1,,,,
37,delta_net,1,1,1,1,1,1,,,,
38,delta_net,1,1,1,1,1,1,,,,
39,gqa,1,1,1,,,,1,1,1,1
40,delta_net,1,1,1,1,1,1,,,,
41,delta_net,1,1,1,1,1,1,,,,
42,delta_net,1,1,1,1,1,1,,,,
43,gqa,1,1,1,,,,1,1,1,1
44,delta_net,1,1,1,1,1,1,,,,
45,delta_net,1,1,1,1,1,1,,,,
46,delta_net,1,1,1,1,1,1,,,,
47,gqa,1,1,1,,,,1,1,1,1
48,delta_net,1,1,1,1,1,1,,,,
49,delta_net,1,1,1,1,1,1,,,,
50,delta_net,1,1,1,1,1,1,,,,
51,gqa,1,1,1,,,,1,1,1,1
52,delta_net,1,1,1,1,1,1,,,,
53,delta_net,1,1,1,1,1,1,,,,
54,delta_net,1,1,1,1,1,1,,,,
55,gqa,1,1,1,,,,1,1,1,1
56,delta_net,1,1,1,1,1,1,,,,
57,delta_net,1,1,1,1,1,1,,,,
58,delta_net,1,1,1,1,1,1,,,,
59,gqa,1,1,1,,,,1,1,1,1
60,delta_net,1,1,1,1,1,1,,,,
61,delta_net,1,1,1,1,1,1,,,,
62,delta_net,1,1,1,1,1,1,,,,
63,gqa,1,1,1,,,,1,1,1,1
```

<details><summary>machine-readable JSON for this target</summary>

```json
{
  "target_complete_bpw": 1.0,
  "policy": "primary_unconstrained",
  "feasible": false,
  "achieved_complete_physical_bpw": 1.1280689287891834,
  "tensor_payload_bytes": 3792567522,
  "budget_bytes": 3361999808,
  "slack_bytes": null,
  "objective_J": 227.7197422056343,
  "reason": "floor_exceeds_budget",
  "codec_map": {
    "1": "binary_g128",
    "2": "ternary_t0.7_g128",
    "3": "uniform_q3_g64",
    "4": "uniform_q4_g64",
    "32": "f32v2"
  },
  "globals": {
    "embed": 1,
    "lm_head": 1,
    "final_norm": 32
  },
  "pinned_f32_classes": [
    "input_layernorm",
    "post_attention_layernorm",
    "final_norm",
    "dn.conv1d",
    "dn.A_log",
    "dn.dt_bias",
    "dn.norm",
    "gqa.q_norm",
    "gqa.k_norm"
  ],
  "layers": [
    {
      "layer": 0,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 1,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 2,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 3,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 4,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 5,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 6,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 7,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 8,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 9,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 10,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 11,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 12,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 13,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 14,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 15,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 16,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 17,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 18,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 19,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 20,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 21,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 22,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 23,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 24,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 25,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 26,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 27,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 28,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 29,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 30,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 31,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 32,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 33,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 34,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 35,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 36,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 37,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 38,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 39,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 40,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 41,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 42,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 43,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 44,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 45,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 46,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 47,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 48,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 49,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 50,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 51,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 52,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 53,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 54,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 55,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 56,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 57,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 58,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 59,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    },
    {
      "layer": 60,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 61,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 62,
      "mixer": "delta_net",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "dn.in_proj_qkvz": 1,
      "dn.in_proj_ba": 1,
      "dn.out_proj": 1
    },
    {
      "layer": 63,
      "mixer": "gqa",
      "mlp.gate_proj": 1,
      "mlp.up_proj": 1,
      "mlp.down_proj": 1,
      "gqa.q_proj": 1,
      "gqa.k_proj": 1,
      "gqa.v_proj": 1,
      "gqa.o_proj": 1
    }
  ],
  "note": "TARGET_MISSED. Table is unconstrained 1-bit floor (all GEMV binary_g128). Complete BPW 1.128069 > 1.0."
}
```

</details>

---

## 8. Pattern (why the tables look like this)

Activation RMS grows ~12.6× from L0 to L61. Primary `s_i` therefore spends first on late layers. Measured `e(b)` is also worse for mid/late `down` and `up` than for late `gate`. Result at 2.0:

- L0–L30 MLP: almost all 1-bit
- L31–L63 `down`: 2 then 3 then 4
- L33–L63 `up`: 2-bit ternary
- L58–L62 `gate`: 2-bit
- late `out_proj` / GQA `k,v,o`: 3–4 bit (small, cheap to protect, high class prior)
- early `in_proj_qkvz` / `q_proj`: stay 1-bit (15% of mass, low act RMS)

This is the opposite of mixed-2p0 (`gate=binary, up=rice, down=HGRAVS01 0.13, attn=q4 everywhere`). mixed-2p0 spent its density on MLP and left attention at 4.25. This allocator, given equal complete BPW, spends density on early/easy MLP and keeps bits on late residual writers.

Unit-weight (`s_i=1`) instead maxes small tensors (ba/k/v/o at 4) because they are byte-cheap. That is the signature of an unweighted relative-error objective. Primary is the intended recipe.

---

## 9. Historical hetero packs (not this recipe)

| artifact | complete BPW | recipe | generate |
|---|---:|---|---|
| uniform-q4-v1 | 4.252735 MEASURED | all GEMV q4, small f32 | COHERENT (controller, same harness) |
| mixed-q3mlp-v1 | 3.613865 MEASURED | MLP q3, attn/embed q4 | not this lane |
| mixed-2p0-v1 | 2.085593 MEASURED | gate binary, up rice, down HGRAVS01, rest q4 | INCOHERENT native, 0 fallbacks |
| mixed-sub15-v1 | 1.291078 MEASURED | mixed-2p0 MLP + rice attention | INCOHERENT |

```
// receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-10
4.2527_BPW_q4_oracle: COHERENT
2.0856_BPW_mixed-2p0-v1: INCOHERENT (native, verified twice)
1.2910_BPW_mixed-sub15-v1: INCOHERENT
conclusion: floor with current codecs lies between 2.0856 and 4.2527 BPW
```

Descent screen already said no cheap in-register codec is both below 2.0 and quality-intact (`QWEN38_BPW_DESCENT.json` coherence_floor). The 2.0 table below still places a majority of early MLP at binary. Treat it as a packer recipe whose generation result is unknown, not as a capability claim.

---

## 10. What is not claimed

- No TOKEN_NS, no TPS. Byte-count invariant `ms ≈ 33.537 * (bpw/4.2527)` is a PROJECTED bandwidth story from `QWEN38_BPW_DESCENT.json:21-26` / `QWEN38_BANDWIDTH_BOUND.json`. This lane did not measure it.
- No generation. Organ hold-L2 ≠ token drift.
- `s_i` is a proxy. Embed/lm_head curves are proxies. DN attention curves are L0 copies.
- HGRAVS01 / rice are not in the published rungs.
- Kernels: binary and ternary need representation-specific Metal. Expanding to float/Q4 then generic GEMV is forbidden unless a complete-token measurement (other lane) shows a net win.

Cheapest experiment that would convert the 2.0 table from IMPLEMENT_READY to MEASURED_WIN or MEASURED_NEGATIVE: pack it, native-generate the coherence prompt set, compare against the Q4 oracle. GPU lane owns that.

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. Complete BPW is 8*payload_bytes/26895998464. G0 uniform-q4 catalog is 4.252735126866492. MEASURED. Evidence: qwen38_pack.rs:673-678; uniform-q4-v1/manifest.json:2-15.
C2. Per-tensor error curves exist for 6 layers × {gate,up,down,attn_in} as hold_output_rel_l2 at rungs {1,2,3,4}. MEASURED. Evidence: QWEN38_BPW_DESCENT.json organs + seal 6269ed05.
C3. Remaining tensors use labeled proxies (layer interp, L0-DN copy, weight-L2, mean weight-L2). ESTIMATED. Evidence: descent claim_boundary.lm_head_not_output_scored / attention_census_not_exhaustive; corr(weight,hold)=0.9559 on 96 pairs.
C4. Sensitivity s_i = hidden_rms_norm * Gravity class prior. PROXY. Evidence: activation-capture-v1 hidden/L*.f32 RMS (mean 0.64955, range 0.100-1.259); gravity_global_allocator.py organ priors.
C5. At equal complete BPW 2.0, greedy hetero J is 53.44% below uniform two-rung mix (67.433 vs 144.838) under primary floors. ESTIMATED from the screen. Evidence: /tmp/g1_hetero_alloc_out.json targets.2.0.policies.primary_floors.gap.
C6. At 1.5 the same gap is 32.16%. ESTIMATED. Evidence: targets.1.5.policies.primary_floors.gap.
C7. At 1.2, floors are infeasible; unconstrained hetero J is 17.93% below uniform. ESTIMATED. Evidence: targets.1.2.
C8. 1.0 complete BPW is unreachable with {binary,ternary,q3,q4,f32}. Floor is 1.128069. FALSIFIED for this family. Evidence: baselines.all_binary_plus_f32; budget 3361999808 < 3792567522.
C9. mixed-2p0 at 2.085593 is a different hetero recipe and is generation-incoherent. MEASURED (other lane). Evidence: mixed-2p0-v1/PACK_REPORT.json:10; QWEN38_COHERENCE_FLOOR_BRACKETED.json:7-10.
C10. Published 2.0/1.5/1.2 tables are concrete per-layer bit widths a packer can consume. IMPLEMENT_READY. Evidence: section 7 CSV+JSON.

EVIDENCE
- crates/hawking-core/src/model/qwen38_pack.rs:673-678 complete BPW formula
- crates/hawking-core/src/model/qwen38_geometry.rs:20-45 shapes
- workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json:2-15 catalog
- receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json organs, seal, claim_boundary, coherence_floor
- workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json + hidden/L00-L63.f32
- workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/PACK_REPORT.json:10-32
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-10
- tools/condense/gravity_global_allocator.py:400-436 greedy
- /tmp/g1_hetero_alloc.py solver; /tmp/g1_hetero_alloc_out.json numbers

CHANGES
- workspace/superwave/g1/g1-heterogeneous-allocation.md (this file only)

TESTS
- test -s workspace/superwave/g1/g1-heterogeneous-allocation.md
- wc -l workspace/superwave/g1/g1-heterogeneous-allocation.md
- git status --porcelain

RISKS
- Screen J is not generation. mixed-2p0 already showed a 2.09 hetero pack can be garbage.
- DN attention curves are a single-layer copy. If late DN qkvz is as fragile as late down, the 2.0 table under-protects it.
- embed/lm_head unscored. Floors (2/3) are priors. Unconstrained dumps both to 1-bit.
- Ternary and binary need native kernels. No kernel in this lane.
- 1.0/1.2 sit in the Q30 graveyard zone the descent receipt already named.

UNRESOLVED
- Generation of these tables.
- Isolated HGRAVS01 hold_output_rel_l2.
- Output-space score for attn_out, embed, lm_head.
- Per-layer DN attention curves for layers 1-62.
- Whether 53% J reduction survives a Jacobian / residual-gain model.

NEXT
- GPU lane: pack the 2.0 CSV and native-generate the coherence set.
- If 2.0 generate holds, pack 1.5. If not, raise the binary-MLP early-layer floor and re-solve.
- Do not pack 1.0 with this family.
```

