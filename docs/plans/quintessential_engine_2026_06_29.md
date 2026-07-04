# Hawking — the quintessential local inference engine (unified plan, 2026-06-29)

> Size, quality, and speed are treated as one economy, not three tradeoffs. This is the canonical
> plan that unifies the condense/serve program. Discipline throughout: EFFECTIVE bpw only (baker
> AGGREGATE), output-space + multi-eval gates (never weight-space RMSE), and no WIN cell without a
> receipt. `MEASURED` / `GATED` (on the serve build) / `UNPROVEN` tags are load-bearing.

## Thesis

Hawking is the quintessential local engine because it treats **size, quality, and speed as one
economy**. The **SIZE** axis stops requiring the whole model resident (the event horizon: weights
live on SSD, only what a token touches streams through RAM), so a 96GB M2 Max pulls in 235B/671B
MoE that llama.cpp cannot hold. The **QUALITY** axis is an expanding **Doctor registry**, not one
script - every point the Doctor recovers is a point re-spent on more compression, driving the
measured bit-floor down with scale. The **SPEED** axis (native `.tq` GEMV, residual two-part decode,
STKV tiered KV, spec-decode on the condensed substrate) turns the small artifact into interactive
tok/s. The single gate unifying all three is the **serve build**; until it lands, the frontier
quality and RAM-cliff numbers stay honestly GATED.

## The Doctor, as a registry (tools/condense/doctor.py registry)

"The doctor" is the *name for restoring quality at low bits*, not a function. It is now a pluggable
`REGISTRY` of `RecoveryMethod`s + an auto-selector. Register with `@register(...)`; a method's
`build_fn` returns a BakeSpec (baker argv + optional shadow-weights step + eff-bpw hooks) - the
driver runs it, so composition/ordering lives in one place. 11 methods today:

| L | method | stage | train-free | status |
|--:|---|---|---|---|
| 0 | calib (domain-matched) | local | yes | MEASURED |
| 1 | awq pre-scale / learned_rotation (QuaRot/SpinQuant) | local | yes | MEASURED / UNPROVEN |
| 2 | mixed_precision (output-sensitivity water-fill) / expert_alloc (MoE per-expert) | local/studio | yes | MEASURED / GATED |
| 3 | residual (two-part) / outlier_channel | local | yes | MEASURED |
| 4 | block_qat (BRECQ-lite full-rank) | studio | no | GATED |
| 5 | gptq_hessian (codec-native error-feedback; NO uniform STE = DEAD) | studio | no | UNPROVEN |
| 6 | deep_kd / big_teacher_kd | studio | no | GATED / UNPROVEN |

**Auto-selection:** start from the SUBBIT-0 measured entropy floor (else the redundancy heuristic)
-> target rung; always run the train-free stack L0->L3 (they compose, cheap); L2 water-fills bits by
output-sensitivity at the target avg bpw; add training layers L4-L6 only if the train-free stack
leaves a gap above the +2% gate and the model is size-appropriate; insert expert_alloc for MoE. The
selection is a STARTING recipe the floor-search confirms/steps up (2->3->4) - never a silent GO.
**Ledger:** recovery_ledger emits per-method {eff_bpw, degr, recovered_vs_prev, cost} so it reports
recovered-points-per-GPU-hour and flags the tier with the largest remaining gap = where the next
Doctor effort pays. This closes the loop: the ledger tells the selector what to add next.

## The unified chain (studio_run.py `go`)

- **P0 STAGE+ADVISE** - auto_bits (bpw + regime) + size_frontier (device ceilings) + doctor_registry
  (recovery chain) per model, before any bake.
- **P1 CONDENSE** - the Doctor chain via audit_ladder -> floor-search (<=+2% ppl AND multi_eval) ->
  scaling_law --floor -> receipt. RAM-packed.
- **P2 SUBBIT** - SUBBIT-0 entropy gate -> sub-1/sub-2-bit lane (outlier / residual / codec-native).
- **P3 SPEC** - spec-decode revival on the condensed substrate (lossless gate -> capture-retrain ->
  governor). Latency x density.
- **P4 FRONTIER** - 235B/405B/671B/744B, serve-oriented on streamed shards; quality+cliff GATED on
  the serve build. The prize.
- **P5 EVAL + LONG-CONTEXT** - eval_suite (capability+NIAH) + ctx_extend (YaRN) + kv_frontier
  (int2/trellis KV, SSD-paging, SSM) + kv_hybrid (STKV: exact recall + unbounded reach).
- **P6 BASELINE** - wedge gate vs IQ1_S/IQ2/MLX-4bit at matched bpw.
- **P7 CLIFF** - RAM-cliff tok/s + energy J/tok (the headline + the energy moat).
- **P8 CODEC** - STRAND vs QTIP/QuIP#/AQLM (where we rank).
- **P9 SYNTH+SCORECARD** - fit both curves + extrapolate 70B/405B + the receipt-gated competitive matrix.

## Serve-build critical path (the one gate on real wins, in dependency order)

1. **Residual two-part GPU decode** - sum base + residual passes; parity vs CPU `matvec_rht`. GATE: parity green.
2. **All-tensor `.tq` loader** - generalize `read_strand` (today an FFN-hybrid overlay on Q4_K) to a full-model `.tq` incl. attention GEMV. GATE: 7B all-`.tq` at parity.
3. **Per-expert `.tq` writer** + MoE construction - addressable per-expert shards. GATE: 235B `.tq` loads.
4. **MoE expert-paging OOC pager** - mmap cold-expert store + hot LRU cache + prefetch; only active experts resident. GATE: 235B served on 96GB, warm tok/s measured (interactive vs batch verdict).
5. **Frontier native quality + RAM-cliff** - with 1-4 landed, measure real `.tq` quality + the cliff. GATE: P4/P7 flip GATED -> MEASURED.
6. **Spec-decode governor** on the condensed substrate - gated on the greedy-kernel losslessness property test.

## Top over-engineering opportunities (ranked)

1. **Residual two-part `.tq` serve** (M) - converts the measured ~1:1 residual quality into a shippable artifact; every quality point becomes a served point. Kill: summed-pass GEMV can't hit parity or doubles latency.
2. **MoE expert-paging OOC pager** (L) - the black-hole SIZE unlock (235B/671B on 96GB); the one thing llama.cpp can't do. Kill: warm tok/s < ~1 after cache+prefetch -> batch/async-only.
3. **All-`.tq` model loader** (L) - drop the Q4_K base, let attention go sub-4-bit, the substrate the pager needs. Kill: attention sub-4-bit fails the gate -> keep attention higher-bit via mixed-prec.
4. **L2 exact output-space metric + per-tensor residual depth** (M) - lowers avg eff-bpw at a quality target = pushes the floor down. Kill: exact metric matches the cheap proxy within noise.
5. **Codec-native GPTQ-Hessian at scale** (L) - the sub-residual ceiling-breaker on big models. Kill: doesn't beat train-free residual+block-QAT on 7B -> don't scale the most expensive lane.

## Testing plan (every new piece ships with its gate)

- PARITY: CPU reference vs GPU bit-identity (residual two-part must match base+residual CPU sum before it serves).
- GREEDY-KERNEL LOSSLESSNESS: batched verify != greedy at near-ties is a foundational prior finding; spec-decode ships with the property gate that caught it.
- EFFECTIVE-BPW RECEIPT: every bake -> schema-valid receipt with AGGREGATE eff-bpw. No floor point without one.
- OUTPUT-SPACE + MULTI-EVAL: ppl-delta vs f16 AND the multi_eval tripwire (MULTIWINDOW=4); judge on 7B+.
- REGRESSION: cargo check --workspace + crate suites green; RAM-safety leash (the 18GB/6000MB death can't recur).
- WEDGE + CLIFF honesty: no WIN vs IQ/MLX without a matched-bpw receipt; label MEASURED/GATED/UNPROVEN on every number.
