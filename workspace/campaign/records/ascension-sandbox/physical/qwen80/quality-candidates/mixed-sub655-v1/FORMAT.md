# Qwen80 mixed sub-0.655 packed format

Physical byte contract for
`quality-candidates/mixed-sub655-v1`. Same container envelope as
`docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md`. Only the per-organ codec
assignment changes.

Packing alone is **not** a coherence claim. Generation is the gate.

## Identity

```
complete_bpw = 0.97032 * expert_bpw + 0.02968 * nonexpert_bpw
```

Complete physical BPW is computed from **bytes on disk** (every byte
required to execute: codes, scales, headers, rank factors, catalog,
manifest, terminal, format spec, fit tables, segment padding).

- Schema: `hawking.ascension.qwen80_mixed_representation_candidate.v1`
- Branch: `qwen80-mixed-sub655-v1`
- Model id: `Qwen3-Coder-Next-mixed-sub655-v1`
- Artifact prefix: `QWEN80_MIXED_SUB655_V1`
- Expected tensor count: `74391`

## Recipe (recalibrate `best_sub_0_6552_by_min_organ`)

| Organ | Codec | Magic | Notes |
|---|---|---|---|
| routed `gate_proj` | `hgravs01_r16_b3` on **layer hidden** X | `HGRAVS01` | W is `[512, 2048]`; X_fit is router-input hidden, never post-SwiGLU |
| routed `up_proj` | `hgravs01_r40_b3` on **layer hidden** X | `HGRAVS01` | same X as gate |
| routed `down_proj` | `binary_group` 128 | `HGRAVB01` | no activation fit |
| non-expert | uniform Q4 group-64 | `HGRAVU01` bits=4 | embed, lm_head, attention, DeltaNet, norms, router, shared expert |

Never-routed experts (no X) pack HGRAVS organs via weight-space truncated
SVD at the requested rank, same on-disk layout. Reported, not a silent
fallback.

Shared expert is non-expert (4-bit). It is not packed with the routed recipe.

## Directory layout

Same as mixed-1p5-v1 (`catalog.hq80m15`, `fit_rows.u16le`, `fit_kind.u8`,
`segments/00_embed.hq80seg`, `L00`…`L47`, `99_terminal.hq80seg`).
No per-tensor JSON array.

Catalog codec ids are unchanged: 0 binary, 1 residual, 2 hgravs01, 3 uniform.
This recipe uses 2 on gate/up, 0 on down, 3 on non-expert. Runtime must
dispatch by the catalog codec, not by organ.

## Container envelope

Unchanged Gravity `_container` (`HGRAVB01` / `HGRAVR02` / `HGRAVS01` /
`HGRAVU01`). HGRAVS01 factor bits stay 3, group 64; rank is 16 (gate) or
40 (up). HGRAVU01 `bits` is 4.

Native HGRAVS path remains `y = L @ (R @ x)`. Do not materialize dense `W`.
