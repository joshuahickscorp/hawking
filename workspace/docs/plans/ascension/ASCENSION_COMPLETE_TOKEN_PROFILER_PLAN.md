# Ascension complete-token profiler and FLOPS ledger plan

**Authority:** `HAWKING_ASCENSION_BIBLE.md` §11  
**Status:** plan + scaffold (no live model work)  
**Gate:** Proto-Frankenstein offload — this lane does not wait on Frankenstein code; it also does not run live Qwen/Gravity capture until that programme gate and local weights allow.  
**Scaffold code:** `lab/operators/ascension_complete_token_profiler.py`  
**Tests:** `lab/tests/test_ascension_complete_token_profiler.py`

---

## 1. Purpose

Explain **≥98%** of complete-token latency with a **named per-stage ledger** and six **global FLOPS / traffic ledgers**. Higher achieved FLOPS is **not** automatically better: a candidate that performs fewer useful operations can be faster with lower FLOPS.

This is not a greenfield design. A working DSV4F profiler already sealed a real per-stage breakdown with **zero unexplained “other”**. Ascension generalizes that shape to family-agnostic stage inventories (attention **or** DeltaNet, router top-k variants, etc.).

---

## 2. Existing profiler audited (working reference)

### Source files (read-only — do not edit for this lane)

| Piece | Path |
|-------|------|
| Stage inventory | `lab/operators/deepseek_v4_gravity.py` → `_COMPLETE_TOKEN_PROFILE_STAGES` |
| Stage labels | `_COMPLETE_TOKEN_PROFILE_STAGE_LABELS` |
| Per-token profiler | `_DiagnosticTokenProfiler` |
| Aggregation | `_aggregate_complete_token_profile` |
| CLI / seal path | `profile_diagnostic_complete_token` / `profile-complete-token` |
| Unit contracts | `tools/condense/tests/test_deepseek_v4_complete_token_profile.py` |
| Related thin scaffold | `lab/operators/ascension_tg_gauntlet.py` → early `CompleteTokenProfiler` |
| Family stage lists | `lab/operators/ascension_parity_ladder.py` → `QWEN3_MOE_STAGES`, `QWEN3_NEXT_STAGES`, `DEEPSEEK_V4_STAGES_REFERENCE` |

**Rust example note:** this worktree has no `crates/hawking-core/examples/gravity_deepseek_v4_complete_token_profile` binary. The live capture that produced tonight’s sealed breakdown is the **Python diagnostic** path above, not a Metal example binary.

### Sealed receipt (real Layer-4 CPU diagnostic)

Path in parent hawking tree:

```text
workspace/campaign/records/runs/deepseek-v4/complete-token-profile-receipt-v3.json
```

| Field | Value |
|-------|--------|
| Schema | `hawking.gravity.deepseek_v4.complete_token_profile_receipt.v1` |
| Status | `SEALED_REAL_LAYER4_CPU_DIAGNOSTIC_PROFILE_NOT_BASE_TRUE_TPS` |
| Seal prefix | `be80cfee31f6a6b4…` |
| Created | `2026-08-04T19:51:39Z` |
| Forwards | 2 real diagnostic forwards |
| Wall p50 | ~1545.88 ms |
| `other_share_percent` | **0.0** |
| Timing status | `PASS_NO_UNEXPLAINED_OTHER_BUCKET` |
| GPU counters | `CPU_NUMPY_DIAGNOSTIC_NO_GPU_COUNTERS_OR_DISPATCHES` |
| `claim_boundary.base_true_tps` | **false** |

### Stage wall-share p50 (named stages; no OTHER bucket)

| Stage | Label | Wall share p50 |
|-------|-------|----------------|
| `lm_head` | lm_head | **~41.99%** |
| `expert_gather` | expert gather | **~28.76%** |
| `compressed_sparse_attention` | compressed/sparse attention | **~17.81%** |
| `qkv` | QKV | **~9.77%** |
| `gate_up` | gate/up | ~0.37% |
| `kv_state_read_write` | KV read/write | ~0.35% |
| remaining named stages | … | each &lt; 0.3% |
| unexplained other | — | **0.00%** |

Top four named stages already sum to **~98.33%**. That explicitly satisfies bible §11 (“≥98% explained”, “no unexplained other bucket above 2%”).

### DSV4F concrete stage inventory (20 stages)

```text
tokenizer_template
embedding
mhc_state_control
norm
qkv
compressed_sparse_attention
index_heads_topk_index
kv_state_read_write
router_top6
expert_gather
gate_up
activation
down
shared_expert
route_combine
residual
lm_head
topk_sampling
endpoint_hcli_streaming
runtime_bookkeeping
```

### Patterns that must be preserved

1. **Every millisecond has a name.** Residual wall is reconciled into `runtime_bookkeeping`, never into a silent OTHER.
2. **Aggregate rejects** any record with non-zero unexplained wall/CPU.
3. **Percentiles** p50/p95/p99 on complete-token and per-stage walls.
4. **Honest claim boundary:** diagnostic ≠ full runtime ≠ `BASE_TRUE_TPS`.
5. **GPU fields explicit:** zero / `not_available_*` when the path is CPU diagnostic — no invented Metal counters.
6. **HCLI stream** may be `unavailable_in_inprocess_profile` rather than faked.

---

## 3. What generalizes vs what is DSV4F-specific

### Generalizes (family-agnostic scaffold)

| Concept | Ascension locus |
|---------|-----------------|
| Named stage inventory + zero OTHER | `CompleteTokenProfilerV1` + `aggregate_complete_token_profiles` |
| ≥98% explained / ≤2% other budget | `TARGET_EXPLAINED_PERCENT`, `MAX_OTHER_SHARE_PERCENT` |
| Bookkeeping reconciliation | `finish(reconcile_bookkeeping=True)` |
| Per-stage field set (bible §11) | `StageMetricRecord` |
| Aggregate p50/p95/p99 + wall share | `aggregate_complete_token_profiles` |
| Claim-boundary honesty | `default_claim_boundary` + profile claim fields |
| Attention **or** DeltaNet as one role | `StageRole.ATTENTION_OR_DELTANET` |
| Router top-k parameterization | `router_top_k` / `router_top6` / `router_top10` → role `router` |
| Six global ledgers | `GlobalLedgerAccumulator` |
| FLOPS-not-automatically-better rule | `compare_mechanism_candidates` / `assert_not_flops_only_ranking` |

### DSV4F-specific (do not hardcode into Qwen paths)

| Item | Why it stays DSV4F |
|------|--------------------|
| `mhc_state_control` | Hyper-connections control path |
| `compressed_sparse_attention` + `index_heads_topk_index` | Sparse/compressed attention + indexer |
| `router_top6` | V4 routed top-6 (not top-8 / top-10) |
| Schemas `hawking.gravity.deepseek_v4.*` | Family-scoped receipt lineage |
| Layer-4 diagnostic scope | Not full 43-layer runtime |
| `cpu_numpy_diagnostic` device | No GPU dispatch counters in that run |
| Shared expert “accounted by gate_up/activation/down” | DSV4F diagnostic accounting choice |
| `streamed-layer4-diagnostic.gravity` artifact binding | One-layer diagnostic artifact |

### Family stage mapping (concrete → bible role)

Bible §11 abstract roles:

```text
tokenizer/template · embedding · norm/state · QKV · attention or DeltaNet ·
KV/state · router · expert selection · expert gather · gate/up · activation ·
down · shared expert · combine · residual · head · sampling · readback · HCLI stream
```

(+ `runtime_bookkeeping` for named residual wall — required by the DSV4F honesty pattern.)

| Family | Attention path | Router | Inventory constant |
|--------|----------------|--------|--------------------|
| `QWEN3_MOE` | `attention` (+ `rope` under QKV role) | `router_top_k` (top-8 at build time) | `QWEN3_MOE_STAGES` |
| `QWEN3_NEXT` | `gated_deltanet_*` + `hybrid_schedule_slot` + `gated_attention` | `router_top10` | `QWEN3_NEXT_STAGES` |
| `DEEPSEEK_V4` | `compressed_sparse_attention` + `index_heads_topk_index` | `router_top6` | `DEEPSEEK_V4_STAGES_REFERENCE` |

Role map tables live in `FAMILY_STAGE_ROLE_MAP` so rollups by bible role stay comparable across families while concrete stage IDs stay exact-model honest.

---

## 4. Generalized receipt schema

### Single-token profile — `hawking.ascension.complete_token_profile.v1`

```json
{
  "schema": "hawking.ascension.complete_token_profile.v1",
  "family": "QWEN3_MOE",
  "phase": "decode",
  "token_ordinal": 0,
  "position": 0,
  "stage_metrics": [ /* StageMetricRecord */ ],
  "timing_accounting": {
    "observed_complete_token_wall_elapsed_ms": 0.0,
    "observed_complete_token_cpu_duration_ms": 0.0,
    "named_stage_wall_elapsed_ms": 0.0,
    "unexplained_other_wall_elapsed_ms": 0.0,
    "other_share_percent": 0.0,
    "explained_percent": 100.0,
    "target_explained_percent": 98.0,
    "max_other_share_percent": 2.0,
    "status": "PASS_ALL_TIME_EXPLICITLY_NAMED"
  },
  "ops_accounting": {
    "theoretical_ops_total": 0,
    "executed_ops_total": 0,
    "useful_ops_total": 0,
    "redundant_ops_total": 0,
    "achieved_flops_token_estimate": null,
    "bytes_total": 0
  },
  "gpu_dispatch_accounting": {
    "dispatches_total": 0,
    "command_buffers_total": 0,
    "waits_total": 0,
    "gpu_duration_ms_total": 0.0
  },
  "claim_boundary": { "base_true_tps": false, "scaffold_profile_shape_only": true },
  "seal_sha256": "…"
}
```

### Per-stage fields (bible §11)

| Field | Notes |
|-------|--------|
| `gpu_duration_ms` / `cpu_duration_ms` / `cpu_wall_elapsed_ms` | Durations |
| `bytes_read` / `bytes_written` | Traffic estimates or counters |
| `theoretical_ops` / `executed_ops` / `useful_ops` / `redundant_ops` | Ops ledger |
| `achieved_flops` | Derived; **not** a ranking key alone |
| `occupancy` | Device occupancy when measured |
| `arithmetic_intensity` | ops / byte |
| `reuse_factor` | executed/theoretical or measured reuse |
| `dispatches` / `command_buffers` / `waits` | Queue accounting |
| `p50_ms` / `p95_ms` / `p99_ms` | Filled on aggregate |
| `fallback` | Explicit fallback tag |
| `role` | Bible abstract role for cross-family rollup |

### Aggregate — `hawking.ascension.complete_token_profile_aggregate.v1`

Same spirit as DSV4F `_aggregate_complete_token_profile`: per-stage percentiles, wall-share p50, `PASS_NO_UNEXPLAINED_OTHER_BUCKET` when `other_share ≤ 2%`.

### Taxonomy — `hawking.ascension.stage_taxonomy.v1`

Sealed family inventory + `stage_to_role` map + DSV4F reference pointers.

---

## 5. Global ledgers design

### Names (exactly six)

| Ledger | Unit | Meaning |
|--------|------|---------|
| `PEAK_UTILIZATION` | fraction 0..1 | Peak observed occupancy / util over the token window |
| `FLOPS_PER_TOKEN` | flops/token | Executed ops per complete token (**descriptive only**) |
| `BYTES_PER_FLOP` | bytes/flop | Traffic per executed op (inverse intensity) |
| `REUSE_FACTOR` | ratio | Mean reuse of loaded work/bytes |
| `CRITICAL_DEPTH` | stages or ms | Serial critical-path depth |
| `STATE_TRAFFIC` | bytes/token | KV / DeltaNet / mHC / state bytes |

### Persistent JSON — `hawking.ascension.global_flops_ledger.v1`

Produced by `GlobalLedgerAccumulator.snapshot()`:

```json
{
  "schema": "hawking.ascension.global_flops_ledger.v1",
  "family": "QWEN3_MOE",
  "device": "scaffold",
  "token_count": 0,
  "ledgers": {
    "PEAK_UTILIZATION": { "name": "…", "value": null, "status": "UNMEASURED_SCAFFOLD", "p50": null, "p95": null, "p99": null, "samples": [] },
    "FLOPS_PER_TOKEN": { "higher_is_automatically_better": false, "…": "…" },
    "BYTES_PER_FLOP": {},
    "REUSE_FACTOR": {},
    "CRITICAL_DEPTH": {},
    "STATE_TRAFFIC": {}
  },
  "comparison_rule": { "id": "hawking.ascension.flops_not_automatically_better.v1", "…": "…" },
  "claim_boundary": { "higher_flops_is_automatically_better": false },
  "seal_sha256": "…"
}
```

### Accumulator interface

```text
GlobalLedgerAccumulator(family, device)
  .observe_token(profile)   # ingest one complete-token receipt
  .snapshot() -> sealed JSON
```

UNMEASURED stays distinct from zero until at least one observation supplies the cell.

---

## 6. Comparison rule (encoded as a real check)

Document id: `hawking.ascension.flops_not_automatically_better.v1`

```text
primary:   minimize complete-token wall_ms
secondary: minimize useful_ops when wall is not worse
forbidden: rank solely by achieved_flops / FLOPS_PER_TOKEN
```

Code:

| Function | Behaviour |
|----------|-----------|
| `assert_not_flops_only_ranking(decision)` | Raises if `rank_key` is FLOPS and latency ignored |
| `compare_mechanism_candidates(a, b)` | Wall first; useful_ops second; **never** higher FLOPS alone |
| `prefer_lower_useful_ops_example()` | Canonical: low-FLOPS fewer-ops candidate beats high-FLOPS slower one |

Explicit trap: higher FLOPS with **worse** wall is annotated `TRAP_AVOIDED` and not preferred.

---

## 7. Implementation map (this scaffold)

| Deliverable | Path |
|-------------|------|
| Family-agnostic profiler + ledgers | `lab/operators/ascension_complete_token_profiler.py` |
| Tests | `lab/tests/test_ascension_complete_token_profiler.py` |
| This plan | `workspace/docs/plans/ascension/ASCENSION_COMPLETE_TOKEN_PROFILER_PLAN.md` |
| Prior thin profiler (TG harness) | `lab/operators/ascension_tg_gauntlet.py` (kept; v1 is the full §11 shape) |
| Family stage inventories | `lab/operators/ascension_parity_ladder.py` |

**Not modified (read-only reference):**

- `lab/operators/deepseek_v4_gravity.py`
- `tools/condense/tests/test_deepseek_v4_complete_token_profile.py`
- sealed DSV4F receipts under parent hawking `workspace/campaign/records/runs/deepseek-v4/`
- `lab/operators/frankenstein_*`

---

## 8. Live capture path (post-gate; out of scope now)

When Proto-Frankenstein is offloaded and a family runtime is eligible:

1. Build `CompleteTokenProfilerV1` with that family’s stage inventory.
2. Instrument each concrete stage around real Metal dispatches (GPU timestamps, bytes where counters exist).
3. Seal single-token profiles; aggregate; feed `GlobalLedgerAccumulator`.
4. Compare mechanism candidates with `compare_mechanism_candidates` — never FLOPS-only.
5. Feed Self-TG (`profile_own_complete_token` step in `ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md`).

Eligibility mirrors TG / child-baseline honesty: fallback=0, real GPU dispatch, no `BASE_TRUE_TPS` from CPU diagnostic alone.

---

## 9. Non-goals (this session)

- No edits to DSV4F profiler / live capture infra.
- No Qwen download, Gravity pack, or Metal run.
- No Frankenstein operator / evidence touches.
- No push / PR / remote.
- No claim of live FLOPS or TG rungs.

---

## 10. Acceptance (scaffold)

- [x] DSV4F profiler audited with real sealed stage breakdown numbers.
- [x] Family-parameterized stage taxonomy (MoE / Next / DSV4F reference).
- [x] Per-stage bible §11 field set on sealed profile shape.
- [x] ≥98% / ≤2% other budget encoded; scaffold profiles pass with other=0.
- [x] Six global ledgers as persistent sealed JSON + accumulator.
- [x] Higher-FLOPS-not-better encoded as raising check + comparison verdicts.
- [x] Plan document cites working reference and generalizes vs DSV4F-specific split.
- [ ] Live GPU complete-token profile on Qwen (post-gate).
- [ ] Wire Metal counters into `achieved_flops` / occupancy (post-gate).

---

## 11. Related plans

| Plan | Relation |
|------|----------|
| `ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md` | Self-TG consumes complete-token profiles |
| `ASCENSION_30B_PARITY_LADDER_PLAN.md` | Qwen3-MoE stage inventory + parity rungs |
| `ASCENSION_80B_HYBRID_ARCHITECTURE_PLAN.md` | DeltaNet / top-10 stage inventory |
| `ASCENSION_PROGRAM_OVERVIEW.md` | Bible §11 index row |
