# Hawking — the quintessential local inference engine (unified plan, 2026-06-29)

> Capability, energy, bytes, quality, and wall time are treated as one economy. This is the canonical
> plan that unifies the condense/serve program. Discipline throughout: EFFECTIVE bpw only (baker
> AGGREGATE), output-space + multi-eval gates (never weight-space RMSE), and no WIN cell without a
> receipt. `MEASURED` / `GATED` (on the serve build) / `UNPROVEN` tags are load-bearing.

## Thesis

Hawking is the quintessential local engine because it treats **capability per joule, byte, parameter,
and second as one economy**. The delivered target is an **M3 Ultra Mac Studio with 96 GB unified
memory, 819 GB/s advertised memory bandwidth, and a 1 TB SSD**. The GPT/Codex app is an interactive
tenant, so no experiment owns the full memory envelope. Storage plans use current free space, never the
nominal 1 TB: leave 150 GB untouched and charge 64 GB scratch plus 32 GB HF/Xet cache. The **SIZE**
axis stops requiring every model resident, but storage and paging bytes are charged rather than hidden.
The **QUALITY** axis is an expanding **Doctor registry**, not one
script - every point the Doctor recovers is a point re-spent on more compression, driving the
measured bit-floor down with scale. The **SPEED** axis (native `.tq` GEMV, residual two-part decode,
STKV tiered KV, locality, and persistent state) turns the small artifact into useful wall-clock work.
The single gate unifying all axes is the **serve build plus the capability-efficiency receipt**; until
both land, the frontier
quality and RAM-cliff numbers stay honestly GATED.

## Efficiency charter

[`computational_efficiency_paradigms_2026_07_11.md`](computational_efficiency_paradigms_2026_07_11.md)
is part of this plan and precedes the training ladder. FLOPS and raw tok/s are measurements, not the
objective. Every promoted comparison fixes the capability/quality contract and reports useful or correct
completions per joule, capability per byte moved/resident byte, SLO goodput, TTFT and p95, output length,
peak unified memory, pressure/swap state, and discarded/recomputed work. Cold load, prefill, warm decode,
long-context decode, and server concurrency remain separate regimes. Estimated bytes and energy must be
labeled separately from measured counters.

**E0 baseline comes first.** Capture the current engine and same-box baselines before training. Target
less than 2% accounting overhead and phase-level closure within 10%. Later Phase E pilots are default-off:
E1 useful-token/locality scheduling, E2 content-addressed copy-on-write state, E3 typed all-position KV
with lossless backing, E4 a predictive-innovation code-length oracle, and E5 a retrieval/speculation oracle.
Each keeps the GO/KILL gates in the research agenda; none may convert shifted work or quality loss into a win.

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
output-sensitivity at the target avg bpw. Run L0-L3 first. Add L4 block-QAT only if the train-free stack
leaves a reproducible gap above the +2% gate on 7B+ and the checkpoint/memory-pressure oracle is green.
Add L5 GPTQ-Hessian only if that baseline is reproducible and the predicted recoverable gap clears the
measured cost ceiling. L6 remains a separately costed polish step; insert expert_alloc for MoE. The
selection is a STARTING recipe the floor-search confirms/steps up (2->3->4) - never a silent GO.
**Ledger:** recovery_ledger emits per-method {eff_bpw, degr, recovered_vs_prev, cost} so it reports
recovered-points-per-GPU-hour and flags the tier with the largest remaining gap = where the next
Doctor effort pays. This closes the loop: the ledger tells the selector what to add next.

## The unified chain (studio_run.py `go`)

- **E0 CAPABILITY-EFFICIENCY BASELINE** - same-box quality, joule, byte, memory-pressure, and wall-time
  controls before training.
- **P0 STAGE+ADVISE** - auto_bits (bpw + regime) + size_frontier (device ceilings) + doctor_registry
  (recovery chain) per model, before any bake.
- **P1 CONDENSE** - the Doctor chain via audit_ladder -> floor-search (<=+2% ppl AND multi_eval) ->
  scaling_law --floor -> receipt. Safe order: 0.5B, 1.5B, 7B; 14B alone; 32B blocked until a measured
  peak proves the interactive reserve or a streamed/blockwise checkpoint path is green.
- **P2 SUBBIT** - SUBBIT-0 entropy gate -> sub-1/sub-2-bit lane (outlier / residual / codec-native).
- **P3 SPEC** - DEAD/BLOCKED by default. Existing EAGLE (`tau≈0.877`) and n-gram (`tau≈1.43`) results
  miss the `tau>=2.5` resurrection threshold. Reopen only after an offline oracle clears that threshold
  and a genuinely one-pass batched verifier is measured; then require exact/distribution-preserving output,
  lower completion energy/latency, and non-regressing p95.
- **P4 FRONTIER** - 235B-A22B, 405B, 671B, DeepSeek-V4 Flash/Pro, GLM-5.2, and Kimi-K2 controls;
  serve-oriented on verified streamed shards where whole sources do not fit; quality+cliff GATED on the
  serve build. The prize.
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
4. **Resident frontier proof** - 235B-A22B (~39 GB), 405B (~68 GB), and V4-Flash (~48 GB) must pass
   native quality, measured peak-memory, and RAM-cliff gates with the interactive reserve intact.
5. **MoE expert-paging OOC pager** - 671B (~84 GB) is a pressure-sensitive edge/paging target; GLM,
   Kimi, and V4-Pro overflow the interactive resident budget. GATE: useful tasks complete at a documented
   latency/energy SLO; paging is not reported as resident-speed throughput.
6. **Frontier native quality + RAM-cliff** - with the relevant resident or paging path landed, measure real
   `.tq` quality + the cliff. GATE: P4/P7 flip GATED -> MEASURED.

## Top over-engineering opportunities (ranked)

1. **Residual two-part `.tq` serve** (M) - converts the measured ~1:1 residual quality into a shippable artifact; every quality point becomes a served point. Kill: summed-pass GEMV can't hit parity or doubles latency.
2. **MoE expert-paging OOC pager** (L) - the 671B+/overflow capacity unlock; the one thing llama.cpp
   cannot do locally at this footprint. Kill: warm tok/s < ~1 after cache+prefetch -> batch/async-only.
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
- CAPABILITY EFFICIENCY: no energy/byte/goodput win without the same frozen quality suite, output-length
  accounting, memory-pressure trace, and cold/warm separation.
- CHECKPOINT/MOVE: every download shard, training save interval, model/config completion, and phase
  transition has a durable checkpoint. Before power removal, drain new launches, finish or gracefully
  stop the active writer, verify the latest cache/artifact and ledger, and require `SAFE TO UNPLUG`.
