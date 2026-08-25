# G11 — 2-tier Matryoshka NR for Qwen3.8 MLP

Lane: `g11-matryoshka`. CPU numpy. Real BF16 parent + real v2 `post_swiglu`.
No GPU. No generate. No runtime / receipt / artifact mutation.

STATUS: **MEASURED_WIN**. Base-only reconstructs. Base+correction is closer to
the bf16 parent in weight space on all 6 tensors and on
`down_proj` activations on all 3 layers.

Every number is **MEASURED** (this process) unless tagged **CITED** or **DERIVED**.

---

## 0. Verdict

A real 2-tier hierarchy exists for this organ set:

- **BASE** = HGRAVU01 uniform-q3 group-64 of the BF16 MLP weight.
  This is the packing scheme already in `mixed-q3mlp-q3attn-v1`
  (CITED complete BPW 3.344772, MLP BPW 3.250025,
  coherent on the campaign gate; see `g1-mlp-family-generate.md` for
  `mixed-q3mlp-v1` and the `mixed-q3mlp-q3attn-v1` PACK_REPORT / Genesis.nr).
  The base is a valid standalone: that artifact loads and generates without a
  correction plane.
- **CORRECTION** = HGRAVU01 uniform-q2 group-64 of
  `residual = bf16 − q3_dequant`. Same grouping. Own f16 scale plane.
- **One hierarchy**, not two models: base codes + correction codes + two scale
  planes that share the group-64 index. Native-decode of the correction plane
  is **future work**; this lane proves the stored structure and the error drop.

| space | n | base-only rel-fro (mean) | base+correction rel-fro (mean) | drop |
|---|---:|---:|---:|---:|
| weight | 6 | 0.25616898 | 0.12515527 | 0.13101371 |
| down_proj on post_swiglu | 3 | 0.22742779 | 0.11110732 | 0.11632046 |

Correction is strictly lower on every row below. Fail-closed: the packer exits
non-zero if any tensor violates that.

---

## 1. Weight-space reconstruction

Relative Frobenius `||W − Ŵ||_F / ||W||_F` against the BF16 parent.
Full tensor, not the stride-17 packer screen.

| tensor | shape | base-only rel-fro | base+corr rel-fro | drop | base cos | +corr cos |
|---|---|---:|---:|---:|---:|---:|
| `language_model.model.layers.15.mlp.gate_proj.weight` | 17408×5120 | 0.25278002 | 0.12345044 | 0.12932958 | 0.96964094 | 0.99247445 |
| `language_model.model.layers.15.mlp.down_proj.weight` | 5120×17408 | 0.25451112 | 0.12429434 | 0.13021678 | 0.96924202 | 0.99237170 |
| `language_model.model.layers.31.mlp.gate_proj.weight` | 17408×5120 | 0.25744784 | 0.12577337 | 0.13167447 | 0.96855659 | 0.99219114 |
| `language_model.model.layers.31.mlp.down_proj.weight` | 5120×17408 | 0.25736740 | 0.12574241 | 0.13162499 | 0.96857276 | 0.99219526 |
| `language_model.model.layers.47.mlp.gate_proj.weight` | 17408×5120 | 0.25718788 | 0.12571864 | 0.13146923 | 0.96861022 | 0.99219814 |
| `language_model.model.layers.47.mlp.down_proj.weight` | 5120×17408 | 0.25771964 | 0.12595244 | 0.13176720 | 0.96849104 | 0.99216975 |

Stride-17 weight cosine. The packer screen (CITED PACK_REPORT) uses f32 group
scales and does not snap to f16; `packer-screen` below is that exact operator.
`f16-recon` is the stored HGRAVU01 path (scale snapped to f16, then subsampled).

| tensor | packer-screen base | CITED mixed-q3mlp | Δ | f16-recon base | f16-recon +corr |
|---|---:|---:|---:|---:|---:|
| `language_model.model.layers.15.mlp.gate_proj.weight` | 0.969732758307 | 0.969732758307 | 0.000e+00 | 0.9696822304 | 0.9924805691 |
| `language_model.model.layers.15.mlp.down_proj.weight` | 0.969391449267 | 0.969391449267 | 0.000e+00 | 0.9693413812 | 0.9923977756 |
| `language_model.model.layers.31.mlp.gate_proj.weight` | 0.968605987134 | 0.968605987134 | 0.000e+00 | 0.9685523603 | 0.9921923784 |
| `language_model.model.layers.31.mlp.down_proj.weight` | 0.968574370704 | 0.968574370704 | 0.000e+00 | 0.9685199755 | 0.9921913357 |
| `language_model.model.layers.47.mlp.gate_proj.weight` | 0.968649590089 | 0.968649590089 | 0.000e+00 | 0.9685974972 | 0.9921995142 |
| `language_model.model.layers.47.mlp.down_proj.weight` | 0.968446687096 | 0.968446687096 | 0.000e+00 | 0.9683909495 | 0.9921417314 |

A packer-screen Δ on the order of float64 rounding is the proof that this BASE
is the mixed-q3mlp scheme, not a new q3. The f16-recon column is slightly lower
because the stored scale is f16; that is the real artifact, not the screen.

---

## 2. `down_proj` on real `post_swiglu` activations

Y = X @ Wᵀ. X is the v2 parent-BF16 `post_swiglu` capture
(CITED schema `hawking.ascension.qwen38_activation_capture.v2`,
status `CAPTURED_REAL_BF16_MULTI_SITE`, n_tokens=23216).
This process used a linspace of 512 rows across the full capture so the
measurement is a reconstruction screen on real X, not a generate claim and not
a 256-token v1 leftover.

| layer | X rows used / avail | base-only rel-fro | base+corr rel-fro | drop | base cos | +corr cos |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 512 / 23216 | 0.22173017 | 0.10834863 | 0.11338153 | 0.97638859 | 0.99418645 |
| 31 | 512 / 23216 | 0.22257579 | 0.10869505 | 0.11388074 | 0.97618290 | 0.99414595 |
| 47 | 512 / 23216 | 0.23797740 | 0.11627829 | 0.12169911 | 0.97287822 | 0.99331279 |

Capture files:
- L15: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_swiglu/L15.f16`
- L31: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_swiglu/L31.f16`
- L47: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_swiglu/L47.f16`

---

## 3. Bytes per tier

HGRAVU01 body = packed codes + f16 scales. Container = magic + JSON header + body.
The hierarchy is one envelope around both planes, not two HGRAVU01 files.

| tensor | n | base body | corr body | hierarchy (est) | two separate files | bf16 |
|---|---:|---:|---:|---:|---:|---:|
| `language_model.model.layers.15.mlp.gate_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| `language_model.model.layers.15.mlp.down_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| `language_model.model.layers.31.mlp.gate_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| `language_model.model.layers.31.mlp.down_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| `language_model.model.layers.47.mlp.gate_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| `language_model.model.layers.47.mlp.down_proj.weight` | 89128960 | 36208640 | 25067520 | 61276416 | 61276720 | 178257920 |
| **sample total** | 534773760 | 217251840 | 150405120 | 367658496 | 367660320 | 1069547520 |

On-disk mixed-q3mlp HGRAVU01 container is **36208920** B per MLP
tensor (CITED PACK_REPORT `nbytes` = body + ~280 B magic/JSON). Per-tensor body
breakdown (identical for every 5120×17408 / 17408×5120 MLP matrix):

| plane | bits | code bytes | scale bytes | body | container | physical BPW |
|---|---:|---:|---:|---:|---:|---:|
| BASE q3 | 3 | 33423360 | 2785280 | 36208640 | 36208920 | 3.250000000 |
| CORRECTION q2 | 2 | 22282240 | 2785280 | 25067520 | 25067800 | 2.250000000 |
| hierarchy (sum of bodies) | 3+2 | 55705600 | 5570560 | 61276160 | 61276416 | 5.500000000 |

DERIVED: hierarchy / bf16 = 34.38% of the parent bytes for that tensor. Base alone is 20.31%.
A second full q3 model would have been 36208920 extra bytes; the correction plane is 25067520 (69.23% of one base).

Projected to all 192 MLP tensors if the same two-tier recipe were packed
(DERIVED from the measured per-tensor bodies, not a packed 192-tensor artifact):

- base bodies: 6952058880 B
- correction bodies: 4812963840 B
- hierarchy bodies: 11765022720 B
- vs mixed-q3mlp MLP payload CITED 3 × 2_317_370_880 = 6_952_112_640 B (exactly 192 × 36208920)

---

## 4. Tiered NR container spec

The document below is the NR. It names decoder *families*, not kernels.
A field that could only be true of one machine belongs in NX and is rejected.
G103 left `correction_planes` empty; this is the first filled plane on this patient.

```json
{
  "nr_version": "1.1.0-matryoshka",
  "nr_kind": "hawking.nos.noetic_representation",
  "schema": "hawking.nr.matryoshka_mlp.v1",
  "magic": "HGRAVM01",
  "semantic_provenance": {
    "parent_model": "Qwen3.8-27B (Genesis patient, abliterated)",
    "parent_revision": "bf16",
    "parent_path": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16",
    "parameter_count": 26895998464,
    "sample_layers": [
      15,
      31,
      47
    ],
    "sample_organs": [
      "gate_proj",
      "down_proj"
    ],
    "base_artifact": "mixed-q3mlp-q3attn-v1",
    "base_complete_physical_bpw": 3.3447723007722434,
    "base_mlp_physical_bpw": 3.2500251321231617,
    "patient_note": "abliterated parent; Tabula drift is a Doctor axis and is NOT recorded here because NR states what the representation IS, not how it scored"
  },
  "representation": {
    "kind": "two_tier_residual_matryoshka",
    "one_hierarchy_not_two_models": true,
    "shared_structure": {
      "group_size": 64,
      "flatten": "row_major C-order of the stored [out, in] matrix",
      "scale_dtype": "float16",
      "scale_layout": "two f16 planes of identical length (n_groups). They share the group index; they do NOT share numeric values. Sharing the base absmax as the residual scale maps q2 codes to 0 (residual < 0.5 base-step) and the correction plane vanishes."
    },
    "base": {
      "family": "grouped_absmax",
      "codec": "HGRAVU01",
      "representation": "uniform_q3_group_scale",
      "bits": 3,
      "group": 64,
      "bound": 3,
      "scale": "f16(absmax(group) / 3)",
      "codes": "int in [-3, +3], packed unsigned as code+3, 3 bits/elem",
      "standalone": "the existing mixed-q3mlp-q3attn-v1 catalog is this base for every MLP GEMV; it loads and generates without the correction plane",
      "decode": "W_base[g] = q3[g] * s_base[g]"
    },
    "correction": {
      "family": "grouped_absmax",
      "codec": "HGRAVU01",
      "representation": "uniform_q2_group_scale",
      "bits": 2,
      "group": 64,
      "bound": 1,
      "of": "residual = bf16_weight - dequant(base)",
      "scale": "f16(absmax(residual_group) / 1)",
      "codes": "int in [-1, +1], packed unsigned as code+1, 2 bits/elem",
      "optional_at_runtime": true,
      "decode": "W_hat[g] = W_base[g] + q2[g] * s_corr[g]",
      "native_decode": "FUTURE WORK - not required for this obligation"
    },
    "byte_ledger_per_mlp_tensor": {
      "n_elem": 89128960,
      "base_code_bytes": 33423360,
      "base_scale_bytes": 2785280,
      "base_body_bytes": 36208640,
      "base_hgravu01_container_bytes": 36208920,
      "corr_code_bytes": 22282240,
      "corr_scale_bytes": 2785280,
      "corr_body_bytes": 25067520,
      "hierarchy_bytes_est": 61276416,
      "bf16_bytes": 178257920,
      "physical_bpw_base": 3.25,
      "physical_bpw_corr_plane": 2.25,
      "physical_bpw_hierarchy": 5.5
    },
    "entropy_streams": [],
    "shared_structures": [
      {
        "name": "group64_layout",
        "group": 64,
        "applies_to": [
          "base",
          "correction"
        ]
      }
    ],
    "generated_structures": [],
    "latent_codes": [],
    "correction_planes": [
      {
        "plane": "mlp_residual_q2",
        "of": "base_uniform_q3_group64",
        "family": "grouped_absmax",
        "bits": 2,
        "group": 64,
        "decode": "W_hat = dequant(base) + dequant(correction)",
        "optional_at_runtime": true
      }
    ],
    "exact_islands": [],
    "route_graph": null
  },
  "kernel_requirements": [
    {
      "requires": "grouped_absmax_decoder",
      "bits": 3,
      "group": 64,
      "note": "the mixed-q3mlp decoder family. Naming a specific kernel, threadgroup geometry or device here would make this NX, not NR."
    },
    {
      "requires": "grouped_absmax_decoder",
      "bits": 2,
      "group": 64,
      "note": "the correction-plane decoder; same family, 2-bit. Native fused decode (Y += X @ dequant(corr).T, or a two-scale unpack in one tile) is future work."
    },
    {
      "requires": "gated_delta_recurrence",
      "note": "the DeltaNet mixer family the representation assumes"
    }
  ]
}
```

Physical layout (one blob per tensor, plane-major so the base is a byte
prefix: a base-only reader stops after `q_base` and never sees the correction):

```
HGRAVM01                  # 8 B magic
u32 header_len
JSON header               # schema, shape, bits, group, byte ledger
f16 s_base[n_groups]      # shared group index
u3  q_base[n_padded]      # little-endian packed, HGRAVU01 bit order
                         # ---- base-only reader stops here ----
f16 s_corr[n_groups]      # same group index, residual absmax
u2  q_corr[n_padded]      # little-endian packed, HGRAVU01 bit order
```

Decode:

```
W_base[i] = q_base[i] * s_base[i // 64]          # standalone
W_hat[i]  = W_base[i] + q_corr[i] * s_corr[i // 64]
```

A reader that does not implement the correction plane stops after `W_base`.
That reader is the existing mixed-q3mlp path.

---

## 5. Base is a runnable standalone

| claim | evidence | tag |
|---|---|---|
| `mixed-q3mlp-q3attn-v1` exists and is the q3 MLP organ set | `workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-q3attn-v1/` PACK_REPORT status PACKED, 851 tensors, MLP organs HGRAVU01 q3 g64 | CITED |
| complete physical BPW | 3.3447723007722434 | CITED PACK_REPORT |
| MLP physical BPW | 3.2500251321231617 | CITED PACK_REPORT |
| coherent generate | `mixed-q3mlp-v1` (same MLP recipe, richer attention) cleared the campaign gate (France/Paris, 17×19); `mixed-q3mlp-q3attn-v1` is the 3.34 BPW sibling with attention also at q3 | CITED `g1-mlp-family-generate.md`, `claude-generate/q3mlp-generate.json` |
| this BASE equals that scheme | packer-screen (f32, stride-17) cosine on the six sample tensors reproduces the PACK_REPORT column to float64 noise | MEASURED this process |
| native-decode of the correction plane | not implemented; numpy dequant only | SCOPE |

---

## 6. Method

```
W                  ← BF16 safetensors, name language_model.model.layers.N.mlp.{gate,down}_proj.weight
s_b, q_b           ← per-64 absmax / 3, f16 snap, rint, clip [-3,3]     # HGRAVU01 q3
W_base             ← q_b * float32(s_b)
R                  ← W - W_base
s_c, q_c           ← per-64 absmax(R) / 1, f16 snap, rint, clip [-1,1]  # HGRAVU01 q2
W_hat              ← W_base + q_c * float32(s_c)
weight rel-fro     ← ||W - Ŵ||_F / ||W||_F
act rel-fro        ← ||X W^T - X Ŵ^T||_F / ||X W^T||_F     # down_proj only
```

Codec identity with mixed-q3mlp is the HGRAVU01 rule in
`lab/operators/qwen38_mlp_not_r160_pack.py:encode_uniform_payload` and
`lab/operators/ascension_dual_gravity_worker.py:_uniform_codec` (group 64,
bound = 2^{bits-1}-1, scale stored f16). This file reimplements that rule so
the worktree does not have to materialize `lab/`.

Why the correction plane has its own scale: a q3 residual lives in
(-0.5 s_b, 0.5 s_b]. Feeding that residual to q2 *with s_b* rounds every
code to 0. The shared thing is the **group index**, not the numeric scale.

---

## 7. Code histograms (sanity)

q3 codes should occupy {-3..+3}. q2 residual codes should occupy {-1,0,+1}
and must not be all-zero (that would be a dead plane).

- `language_model.model.layers.15.mlp.gate_proj.weight`
  - base q3 hist: `{"-1": 20561697, "-2": 7253547, "-3": 1774220, "0": 29957404, "1": 20550637, "2": 7256086, "3": 1775369}`
  - corr q2 hist: `{"-1": 22267054, "0": 44595851, "1": 22266055}`
- `language_model.model.layers.15.mlp.down_proj.weight`
  - base q3 hist: `{"-1": 20549316, "-2": 7135593, "-3": 1755817, "0": 30255608, "1": 20539246, "2": 7138675, "3": 1754705}`
  - corr q2 hist: `{"-1": 22266102, "0": 44591524, "1": 22271334}`
- `language_model.model.layers.31.mlp.gate_proj.weight`
  - base q3 hist: `{"-1": 20501056, "-2": 6966441, "-3": 1732283, "0": 30743568, "1": 20491061, "2": 6965246, "3": 1729305}`
  - corr q2 hist: `{"-1": 22264228, "0": 44599741, "1": 22264991}`
- `language_model.model.layers.31.mlp.down_proj.weight`
  - base q3 hist: `{"-1": 20494002, "-2": 6920552, "-3": 1720851, "0": 30868600, "1": 20492538, "2": 6913340, "3": 1719077}`
  - corr q2 hist: `{"-1": 22263870, "0": 44600581, "1": 22264509}`
- `language_model.model.layers.47.mlp.gate_proj.weight`
  - base q3 hist: `{"-1": 20472504, "-2": 7006984, "-3": 1739409, "0": 30669884, "1": 20495361, "2": 7011482, "3": 1733336}`
  - corr q2 hist: `{"-1": 22258277, "0": 44606900, "1": 22263783}`
- `language_model.model.layers.47.mlp.down_proj.weight`
  - base q3 hist: `{"-1": 20503322, "-2": 6915881, "-3": 1720741, "0": 30854007, "1": 20499939, "2": 6914545, "3": 1720525}`
  - corr q2 hist: `{"-1": 22259317, "0": 44610161, "1": 22259482}`

---

## 8. Run identity

```
argv       tools/matryoshka_pack.py --layers 15,31,47 --out workspace/superwave/g1/g11-matryoshka.md
parent     /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16
acts       /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_swiglu
layers     [15, 31, 47]
organs     ['gate_proj', 'down_proj']
act_rows   512
wall_s     17.592
numpy      2.4.6
capture    schema=hawking.ascension.qwen38_activation_capture.v2 sha256_self=6acdc97d1c408b43c409981633f74ab3fc18103299b53a62419c56c997b12229
```

Future work (explicitly out of scope): a native HGRAVM01 reader and a fused
correction-plane GEMV so base+correction is a generate vehicle, not only a
numpy reconstruction.

