# Qwen80 mixed-1p5-ne4 packed format

Physical byte contract for
`quality-candidates/mixed-1p5-ne4-v1`. Same container envelope and
catalog layout as `docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md`. Only the
non-expert uniform width changes: `HGRAVU01` `bits=4` instead of 8.

Packing alone is **not** a ≤1.5 coherence claim. Generation is the gate.

## Identity

```
complete_bpw = 0.97032 * expert_bpw + 0.02968 * nonexpert_bpw
```

Complete physical BPW is computed from **bytes on disk** (every byte
required to execute: codes, scales, headers, rank factors, residual
indices, outlier payloads, catalog, manifest, segment padding, fit
tables, format spec).

- Schema: `hawking.ascension.qwen80_mixed_representation_candidate.v1`
- Branch: `qwen80-mixed-1p5-ne4-v1`
- Model id: `Qwen3-Coder-Next-mixed-1p5-ne4-v1`
- Artifact prefix: `QWEN80_MIXED_1P5_NE4_V1`
- Expected tensor count: `74391`

## Recipe

| Organ | Codec | Magic | Source of packed bytes |
|---|---|---|---|
| routed `gate_proj` | `binary_group` 128 | `HGRAVB01` | mixed-1p5-v1 |
| routed `up_proj` | binary + `rice_q1_rms` residual @ 2% | `HGRAVR02` | mixed-1p5-v1 |
| routed `down_proj` | `hgravs01_r160_b3` post-SwiGLU | `HGRAVS01` | mixed-1p5-v1 |
| non-expert | uniform Q4 group-64 | `HGRAVU01` bits=4 | mixed-sub655-v1 |

This is a **sealed composition** of already-packed payloads. It does not
re-encode the BF16 source. Every payload sha256 must match the source
catalog row. Isolate lane `auto-q80-isolate-q4-non-expert` showed the
same organ-3 overlay is bit-identical to mixed-1p5 on the certified
12-token reverse-string generate.

Do **not** confuse this with crushed-expert + q4-DeltaNet
(`mixed-sub655-v1`). Isolate showed q4 on the 252 `linear_attn` tensors
collapses that recipe. Incumbent experts + q4 on all 663 non-experts
(including DeltaNet) does not.

## Directory layout

Same as mixed-1p5-v1 (`catalog.hq80m15`, `fit_rows.u16le`, `fit_kind.u8`,
`segments/00_embed.hq80seg`, `L00`…`L47`, `99_terminal.hq80seg`).
No per-tensor JSON array.

Catalog codec ids are unchanged: 0 binary, 1 residual, 2 hgravs01, 3
uniform. Runtime dispatches by the on-disk `HGRAVU01` `bits` field, not
by organ.

Native HGRAVS path remains `y = L @ (R @ x)`. Do not materialize dense `W`.
