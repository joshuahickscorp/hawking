# Flash meta-representation frontier

The Flash campaign now tracks two different quantities:

| Quantity | Meaning | Current state |
|---|---|---|
| `physical_ebpw` | Bits represented by serialized, loadable model bytes | Not measured for a complete Flash successor |
| `meta_bpw` | Pre-registered executable information budget for a teacher-constrained functional representation | Prospective target `0.8871807728336929` |

`meta_bpw` is deliberately detached from traditional BPW. It is a design
budget for a learned program, latent symbols, shared decoder, exact islands,
and residual repair—not a byte count. The normalized byte-equivalent in the
receipt is bookkeeping only. It must never be reported as physical residency,
active bytes/token, or complete-model EBPW.

The target is weighted over the indexed Flash organ census:

| Family | Local target | Program boundary |
|---|---:|---|
| Routed experts | 0.88 | Expert-local teacher-distilled latent code, shared tile decoder, route-margin repair |
| N-gram embedding | 0.70 | Frequency tiers, hot exact islands, measured residual symbols |
| Linear attention / HC | 2.50 | Protected recurrent/state island |
| Embedding / lm_head | 3.50 | Protected vocabulary and terminal-logit island |
| Full attention | 3.00 | KV-sensitive protected island |
| Other state/weights | 2.50–16.00 | Exact or source-approved islands |

The prospective search ladder is `1.0 → 0.9 → 0.8871807728336929 → 0.8 →
0.7 → 0.5 → 0.35 → 0.25` meta-BPW. Every rung is `NOT_MEASURED` until the
same held-out coherence, routing, state, generation, capability, and latency
gates pass; `physical_ebpw` is null at every rung and the lowest useful target
remains unset.

The dominant expert and n-gram programs must consume codes directly. Dense
rematerialization is forbidden because it would trade representation savings
for latency and working-set expansion.

Admission gates are hard: held-out teacher distillation, hidden/routed-output
and terminal-logit fidelity, exact top-k membership and order, low-margin
router protection, recurrent/KV semantics, short-horizon agreement, long-run
no-collapse, zero fallback, and capability-suite parity. The accelerator gate
also requires resident decoder state, route-before-payload, fused generated
tile → gate/up/SwiGLU/down accumulation, explicit dispatch/synchronization
accounting, and GPU plus complete wall latency no worse than the matched
control.

The source-independent serializer, loader, native consumer, and protected
complete-token path do not exist yet. The corresponding HCLI row
`flash-meta-sub1-coherent` therefore remains `BLOCKED`; this is intentional
frontier registration, not a promotion claim. The controlling receipt is
`receipts/headless/FLASH_META_REPRESENTATION_SUB1.json`.

The first development screen is implemented by
`tools/flash_meta_coherence_screen.py`. Its legacy unsafe probe used four
rows from a layer-3 HyperConnection state, which is not the post-HyperConnection
`mlp_input` actually consumed by the layer-4 expert bank. That probe therefore
remains a diagnostic failure, not coherence evidence: held-out relative error
was `0.8994` and `0.7447`, with cosine `0.4371` and `0.6679`, while the matched
per-expert Q4 reference was `0.1008`.

The corrected capture path is
`crates/hawking-core/examples/flash_meta_teacher_trace.rs`. It uses dense
source-BF16 layers 0–3, preserves stateful routing/KV semantics, captures the
exact layer-4 `mlp_input`, records per-row top-K IDs, and emits a hash-bound
teacher receipt with source specimen/index/config identity, per-row input and
layer-output hashes, an explicit route union, and one unique token ID per row.
The capture refuses fewer than 256 rows or duplicate token IDs/MLP-input rows;
the screen independently rechecks the row hashes and route union. The screen
now requires that receipt for a normal fit, and requires at least 256
non-duplicate rows before any sub-1 surface can enter the frontier. Until that
capture is run, the existing diagnostic receipt
`receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json` remains
`UNSAFE_SMALL_PROBE_NOT_PROMOTABLE`, with `physical_ebpw` null and no runtime
artifact. If the host cannot provide a Metal-capable GPU, the capture emits
`receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json` instead; that
boundary is a refusal receipt and never substitutes for teacher rows.

Each fitted rank carries an ordered `surface_failure_gates` list and a
`first_surface_failure` field. A failed surface therefore records whether the
first break was held-out function error, cosine, or failure to beat the
per-expert Q4 comparator; it does not silently lower the global meta target.

The capture/screen handoff is:

```console
cargo run -p hawking-core --example flash_meta_teacher_trace -- \
  --token-start 0 --count 256 \
  --out receipts/headless/FLASH_META_TEACHER_L4.json \
  --state-out receipts/headless/FLASH_META_TEACHER_L4_MLP_INPUT.f32

python3 tools/flash_meta_coherence_screen.py \
  --state receipts/headless/FLASH_META_TEACHER_L4_MLP_INPUT.f32 \
  --teacher-receipt receipts/headless/FLASH_META_TEACHER_L4.json \
  --out receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json
```

The capture is source-teacher evidence only and should be run in a separate
diagnostic window from protected accelerator measurements. It does not emit a
student artifact or authorize a sub-1 physical representation.

The screen also keeps accounting domains separate: the selected ten-expert
BF16 source slice is `65,536,000` bytes, while the temporary float32 CPU load is
`131,072,000` bytes. Neither number is a whole-model active-bytes/token claim.
