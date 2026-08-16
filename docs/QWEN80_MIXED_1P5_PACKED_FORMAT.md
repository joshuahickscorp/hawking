# Qwen80 mixed ≤1.5 packed format (kernel-lane contract)

This is the physical byte contract for
`quality-candidates/mixed-1p5-v1`. Packing owns this document. The
decode-kernel lane consumes it. Generation is a different lane and is
**not** claimed here.

Packing alone is **not** a ≤1.5 coherence claim.

## Identity

```
complete_bpw = 0.97032 * expert_bpw + 0.02968 * nonexpert_bpw
```

Complete physical BPW is computed from **bytes on disk** (every byte
required to execute: codes, scales, headers, rank factors, residual
indices, outlier payloads, catalog, manifest, segment padding).

- Schema: `hawking.ascension.qwen80_mixed_representation_candidate.v1`
- Branch: `qwen80-mixed-1p5-v1`
- Model id: `Qwen3-Coder-Next-mixed-1p5-v1`
- Artifact prefix: `QWEN80_MIXED_1P5_V1`
- Expected tensor count: `74391` (same source inventory as uniform-Q4)

## Recipe (receipt `QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`)

| Organ | Codec | Magic | Notes |
|---|---|---|---|
| routed `gate_proj` | `binary_group` (group 128, fp16 mean-abs scale, 1-bit sign) | `HGRAVB01` | no activation fit |
| routed `up_proj` | binary + `rice_q1_rms` sparse residual @ 2% | `HGRAVR02` | same top-\|r\| selection as incumbent residual |
| routed `down_proj` | `hgravs01_r160_b3` activation-weighted SVD | `HGRAVS01` | fit on **post-SwiGLU** `silu(X@G.T)*(X@U.T)`, never layer hidden |
| non-expert (sensitive 3%) | uniform Q8 group-64 | `HGRAVU01` bits=8 | embed, lm_head, attention, DeltaNet, norms, router, shared expert. Untouched by the expert crush. |

Shared expert is non-expert (8-bit). It is not packed with the routed recipe.

## Directory layout

```
mixed-1p5-v1/
  QWEN80_MIXED_1P5_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json   # small sealed manifest
  QWEN80_MIXED_1P5_V1_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json
  FORMAT.md                                                    # copy of this contract
  catalog.hq80m15                                              # compact mmap catalog
  fit_rows.u16le                                               # 24576 little-endian u16, layer-major
  fit_kind.u8                                                  # 24576 bytes, down_proj fit class
  segments/
    00_embed.hq80seg
    L00.hq80seg … L47.hq80seg
    99_terminal.hq80seg
```

No per-tensor JSON array. No `capture-result.json`-class index.

## Physical layout (execution order)

Bytes inside each segment are concatenated in the order the token graph
consumes them. Offsets live in `catalog.hq80m15`.

1. `00_embed`: `model.embed_tokens.weight`
2. `Lxx` (layer `xx`):
   - `input_layernorm`
   - mixer, DeltaNet (`linear_attn.*`) or GQA (`self_attn.*`) in kernel order
   - `post_attention_layernorm`
   - `mlp.gate` (router)
   - shared expert `gate_proj`, `up_proj`, `down_proj`
   - `mlp.shared_expert_gate`
   - routed experts `0..511`, each as `gate_proj`, `up_proj`, `down_proj`
3. `99_terminal`: `model.norm.weight`, `lm_head.weight`

GQA layers are `layer % 4 == 3` (12 layers). Others are DeltaNet (36).

Experts are stored 0..511. Runtime gather is by route id; the ten live
experts are not a pack-time order.

## Container envelope (all four magics)

Every tensor payload is:

```
[8] magic
[4] header_len   uint32 le
[header_len]     UTF-8 JSON object, canonical
                 `json.dumps(..., sort_keys=True, separators=(",", ":"))`
[body]           codec-specific
```

This is the existing Gravity `_container` envelope
(`lab/operators/ascension_dual_gravity_worker.py`). Do not invent a
second header.

### `HGRAVB01` — binary group (gate)

Header schema `hawking.gravity.binary_sign_scale.v1`.

Body = `groups * fp16 scale` (little-endian) + packed sign bits
(`numpy.packbits(..., bitorder="little")`, 1 = non-negative).

- `group_size = 128`
- scale = mean \|w\| of the group, stored as fp16 (the stored bits are
  authority)
- reconstruction: `sign * scale`

### `HGRAVR02` — binary + rice_q1_rms residual @ 2% (up)

Header schema `hawking.gravity.binary_outlier_residual.v2`.

Body, in order:

1. binary scales + sign bits (same as `HGRAVB01`)
2. rice index blob: `uint32 first_index` + Rice-coded positive deltas
   (`k` in header `rice_k`; unary quotient LSB-first, then `k` remainder
   LSBs)
3. residual scale: one fp16 (`value_scale = "rms"`)
4. residual signs: 1-bit packed unsigned (`bitorder` via
   `_pack_unsigned`, LSB-first). A selected outlier is never a zero
   code. Reconstruction is `sign * stored_rms_scale`.

Selection is **global top-k** by \|W − binary(W)\| with
`k = ceil(elements * 0.02)`. Same selection as
`_residual_codec` / `select_outlier_indices`.

### `HGRAVS01` — rank-160, 3-bit factors (down)

Header schema `hawking.gravity.activation_weighted_svd_low_rank.v1`.
Representation string (Rust parser requires this exact value):
`activation_weighted_svd_low_rank_q`.

`W ≈ L @ R` with `L` shaped `[out, 160]`, `R` shaped `[160, in]`.
Each factor is uniform-q3 group-64 (`factor_bits=3`,
`factor_group_size=64`). Body = left uniform body + right uniform body.

Uniform factor body = `groups * fp16 absmax/bound scale` + packed
unsigned codes, `code = signed + bound`, `bound = 3` for 3-bit,
`bitorder` LSB-first (`_pack_unsigned`).

Required header keys for `parse_hgravs01_header`:

- `schema`, `representation`, `shape`, `matrix_shape`, `elements`,
  `rank`, `factor_bits`, `factor_group_size`
- `left`, `right` (full `hawking.gravity.uniform_group.v1` metadata)
- `left_body_bytes`, `right_body_bytes`
- `activation_capture.sha256` (64 hex)
- `activation_capture.fit_kind` = `real_routed_activation_capture`

Native decode is `y = L @ (R @ x)`. Do **not** materialize dense `W`
on a token path. See
`hgravs01_matvec_f64` / `ascension_qwen30_hgravs01_packed_matvec_parity`.

`down_proj` is `[2048, 512]`. `X_fit` is `[N, 512]` post-SwiGLU.

Fit policy (not a silent fallback):

| `n_fit_rows` | What is packed | `fit_kind.u8` |
|---|---|---|
| `>= 1` | activation-weighted SVD at **requested rank 160** (ridge-regularized 512×512 Gram). Rank is **not** clamped to `n_fit_rows`. | `0` if `N >= 160`, else `1` |
| `0` (never routed in the bound capture) | weight-space truncated SVD of `W` at rank 160, same `HGRAVS01` body. Reported loudly; activation-weighted SVD is undefined without X. | `2` |

The 25258-token capture has 221 never-routed `(layer, expert)` pairs.
Those 221 still have to exist in the artifact (a later prompt may
route to them).

### `HGRAVU01` bits=8 — non-expert

Header schema `hawking.gravity.uniform_group.v1`, `bits=8`,
`group_size=64`. Body = fp16 absmax/127 scales + packed 8-bit unsigned
codes (`signed + 127`).

## Compact catalog `catalog.hq80m15`

Little-endian. mmap-friendly. Not JSON.

```
[8]   magic              "HQ80M15\0"
[4]   version            u32 = 1
[4]   n_tensors          u32 = 74391
[4]   n_segments         u32
[4]   flags              u32
[4]   name_blob_bytes    u32
[4]   reserved           u32 = 0

segment_table[n_segments]:
  [2] id                 u16
  [2] name_len           u16
  [8] bytes              u64
  [32] sha256
  [name_len]             UTF-8 file name relative to `segments/`

tensor_table[n_tensors]   fixed 128-byte records, execution order:
  +0   name_off          u32 into name_blob
  +4   name_len          u16
  +6   codec             u8  (0 HGRAVB01, 1 HGRAVR02, 2 HGRAVS01, 3 HGRAVU01-q8)
  +7   organ             u8  (0 gate, 1 up, 2 down, 3 nonexpert)
  +8   ndim              u8
  +9   pad               u8 = 0
  +10  reserved          u8[2] = 0  (dims aligned to +12)
  +12  dims              4 × u32 (unused = 0)
  +28  elements          u64
  +36  segment_id        u16
  +38  achieved_rank     u16  (down_proj only; else 0)
  +40  offset            u64  into the segment file
  +48  nbytes            u64
  +56  sha256            32-byte content digest of the payload
  +88  flags             u32
  +92  n_fit_rows        u32  (down_proj only)
  +96  codec_bpw         f32
  +100 reserved          u8[28] = 0

name_blob                 concatenated UTF-8 tensor names
```

Tensor flags:

- bit 0: `SENSITIVE_UNTOUCHED` (non-expert 8-bit)
- bit 1: `DOWN_GRAM_RANK_DEFICIENT` (`n_fit_rows < 512`)
- bit 2: `DOWN_WEIGHT_SPACE_SVD` (`n_fit_rows == 0`)
- bit 3: `DOWN_ACTIVATION_WEIGHTED`

Admission verifies each payload sha256 **once**. Never re-hash per token.

## Manifest

Small sealed JSON (Q4 structure, no `tensors` array):

- `schema`, `status`, `branch_id`, `model_id`, `artifact_prefix`
- `source_*` seals and the revalidation path
- `representation` (the four-family recipe)
- `complete_physical_bpw_ledger` (on-disk bytes, not design)
- `catalog` `{path, sha256, bytes, format}`
- `segments` `[{id, path, bytes, sha256}]`
- `claim_boundary`
- `seal_sha256`

`seal_sha256` is SHA-256 of the canonical JSON with the seal field
removed.

## Sensitive 3%

Left at 8-bit. Not reassigned. Changing the protected set is a
representation decision and must be reported with its BPW cost.

## What this format is not

- Not a Metal kernel.
- Not a generation / coherence certificate.
- Not the abandoned uniform-Q4 4.259-BPW vehicle. Q4 remains a
  correctness reference only.
