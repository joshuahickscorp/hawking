# Condense Naming Migration (2026-06-22)

Decision: **Condense** is the public name for the former STRAND low-bit
compression/quantization line. Keep `strand` as a legacy/internal implementation
term until a compatibility-safe migration exists.

Why: "Condense" says what the feature does without making every identifier a
joke. It pulls a large parent model into a smaller verified artifact and
composes naturally with Model Press:

```text
hawking press       # product command
hawking condense    # public verb / alias for memory-budgeted pressing
Condense artifact   # public artifact class
Condense ladder     # 4/3/2/1-bit target ladder
```

## Canonical Public Names

Use this naming stack going forward:

| Concept | Canonical name | Notes |
|---|---|---|
| Overall product/runtime | Hawking | Keep this stable. |
| Artifact creation pipeline | Model Press | The user-facing pipeline: bake, verify, publish. |
| Low-bit compression family | Condense | Replaces public STRAND language. |
| Memory-budgeted large-parent flow | Condense Planner | Normal, direct name for the dry-run planner. |
| Per-tensor/shard execution plan | Press Plan | Prefer this over themed names. |
| Output provenance | Artifact Manifest | Plain and durable. |
| Quality/footprint report | Quality Card | Already used elsewhere; keep it. |
| 4/3/2/1-bit targets | Condense Ladder | Slightly branded, still clear. |
| Old STRAND/strand terms | Legacy/internal | Keep only where needed to point at existing code. |

Avoid introducing more one-off names. In docs, "black-hole" can describe the
mental model; code, CLI, env vars, and file names should be boring enough to
survive contact with users.

## Current `strand` Function Symbols

These are the code symbols found by `rg` on 2026-06-22. Do not rename them
mechanically without tests; many touch wire format, fixtures, or compatibility.

| Current symbol | Location | Public rename target |
|---|---|---|
| `write_strand` | `vendor/strand-quant/src/format.rs` | `write_condense` alias |
| `read_strand` | `vendor/strand-quant/src/format.rs` | `read_condense` alias |
| `write_strand_v2` | `vendor/strand-quant/src/format.rs` | `write_condense_v2` alias |
| `write_strand_v2_rht` | `vendor/strand-quant/src/format.rs` | `write_condense_v2_rht` alias |
| `write_strand_v2_packed` | `vendor/strand-quant/src/format.rs` | `write_condense_v2_packed` alias |
| `write_strand_v2_inner` | `vendor/strand-quant/src/format.rs` | internal; rename last |
| `read_strand_v2_header` | `vendor/strand-quant/src/format.rs` | `read_condense_v2_header` alias |
| `read_strand_v2` | `vendor/strand-quant/src/format.rs` | `read_condense_v2` alias |
| `strand_v1_to_v2` | `vendor/strand-quant/src/format.rs` | `condense_v1_to_v2` alias |
| `read_strand_v2_header_applied` | `vendor/strand-quant/src/sideinfo_wire.rs` | `read_condense_v2_header_applied` alias |
| `read_strand_v2_applied` | `vendor/strand-quant/src/sideinfo_wire.rs` | `read_condense_v2_applied` alias |
| `strand_recon` | `vendor/strand-quant/src/bin/gate-debias.rs` | `condense_recon` |
| `strand_delta_alive` | `vendor/strand-decode-kernel/src/bin/gate-histogram.rs` | `condense_delta_alive` |
| `strand_delta_alive` | `vendor/strand-decode-kernel/src/bin/gate-fused.rs` | `condense_delta_alive` |
| `from_strand_tensor` | `crates/hawking-core/src/tq_gpu.rs` | `from_condense_tensor` alias |
| `read_strand` | `crates/hawking-core/src/tq.rs` | `read_condense` alias |
| `strand_bitslice_entry_sizeof` | `crates/hawking-core/src/kernels/mod.rs` | `condense_bitslice_entry_sizeof` |
| `decode_strand_bitslice` | `crates/hawking-core/src/kernels/mod.rs` | `decode_condense_bitslice` |
| `strand_bitslice_gemv` | `crates/hawking-core/src/kernels/mod.rs` | `condense_bitslice_gemv` |
| `strand_bitslice_gemv_tcb` | `crates/hawking-core/src/kernels/mod.rs` | `condense_bitslice_gemv_tcb` |
| `strand_bitslice_gemm` | `crates/hawking-core/src/kernels/mod.rs` | `condense_bitslice_gemm` |
| `strand_recon` | `tools/strand/scripts/isobpw-headtohead.sh` | `condense_recon` |
| `strand_requant` | `tools/strand/scripts/strand-qat.py` | `condense_requant` |

Test-only symbols can move with the same pattern after the production aliases
exist.

## Broader Public Surface

Also affected:

- crates: `strand-quant`, `strand-decode-kernel`
- package imports: `strand_quant`, `strand_decode_kernel`
- env vars: `STRAND_NO_GPU`, `STRAND_F32_METRIC`, `STRAND_F32_SEARCH`,
  `STRAND_TROPICAL_TIMING`, `STRAND_EVAL_SMOKE`, `STRAND_ROOT`,
  `STRAND_HF_BACKEND`, `STRAND_PYTHON`
- scripts: `strand-qat.py`, `strand-7b-ppl.sh`, `strand-eval`,
  `strand-act2-*.sh`, `strand-delta`, `bake-attested.sh`
- artifacts/extensions: `.strand`, STR2, `.sa`, `application/vnd.strand.archive`
- paths/docs: `tools/strand/*`, archived STRAND docs, packaging assets

## Migration Order

1. **Docs/product language now:** say Condense publicly; describe STRAND as the
   legacy/internal codec name where needed for code references.
2. **CLI aliases next:** add `hawking condense` and keep old scripts working.
3. **Function aliases:** add `*_condense*` wrappers next to `*_strand*`
   functions, with tests proving both paths match.
4. **Env aliases:** accept `CONDENSE_*` env vars while still honoring
   `STRAND_*`.
5. **Artifact identity:** decide whether `.strand`/STR2 becomes a legacy wire
   format inside a `.hawking` or `.condense` container. Do not break existing
   archive verification.
6. **Internal rename last:** only after aliases and gates are green should code
   modules/crates/paths be renamed.

## Light Theme Vocabulary

Use these sparingly. Prefer the normal names above unless a themed term adds
clarity.

- `condense` — main verb: produce a smaller verified artifact.
- `horizon` — optional shorthand for a hard memory/quality boundary.
- `mass` — optional shorthand for model/artifact size.
- `density` — useful for bpw/footprint discussions.
- `collapse` — useful as a verb for reducing a parent into an artifact, but
  avoid using it as a command name.

Do not use forced names like `singularity_manifest`, `hawking_radiation`,
`redshift`, or `accretion_plan` for public interfaces. They make the system feel
less serious and harder to remember.
