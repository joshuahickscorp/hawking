# Ascension Self-TG Gauntlet + Measurement Harness Plan

**Status:** PLAN + HARNESS SCAFFOLD ONLY  
**Gate:** Eligible BASE_TRUE_TPS only after full-runtime parity on a given model  
**Bible:** HAWKING_ASCENSION_BIBLE §10 (Self-TG gauntlets), §11 (profiler), §29–§31  
**Scaffold code:**  
- Python: `lab/operators/ascension_tg_gauntlet.py`  
- Rust metrics: `crates/hawking-speculate/src/metrics_sep.rs`  
**Scaffold tests:** `lab/tests/test_ascension_tg_gauntlet_scaffold.py`

---

## 1. Purpose

Each Qwen model optimizes **itself** before becoming sandbox workforce. Models
**cannot promote themselves**. This plan designs the measurement harness and
loop inventory so that when weights exist, TG rungs are earned with the same
honesty as tonight’s DSV4F work — not with blended TPS or CPU-proxy “base”
numbers.

---

## 2. Self-TG loop (bible §10)

```text
profile own complete token
→ rank bottleneck
→ retrieve prior failures
→ propose three materially different mechanisms
→ select the cheapest distinguishing experiment
→ implement in isolated worktree
→ protected parity
→ protected capability
→ CLEAN benchmark
→ adversarial review
→ report
→ continue
```

Scaffold: `SELF_TG_LOOP_STEPS` + sealed `hawking.ascension.self_tg_loop.v1`.

| Step | Guard |
|------|-------|
| profile | Complete-token profiler; ≥98% wall named (bible §11) |
| rank bottleneck | Roofline / stage p99; no vibes |
| retrieve prior failures | Graveyard / negative-science (bible §32) |
| three mechanisms | Materially different; not three knobs on one idea |
| cheapest experiment | Smallest distinguishing test first |
| isolated worktree | No mutation of sealed serve path |
| protected parity | Parity harness; fallback=0 where required |
| protected capability | Capability suite must not regress |
| CLEAN benchmark | Batch-1 base runtime; same model + tier |
| adversarial review | Human or protected controller |
| report | Sealed receipt only |
| continue | Never self-promote past TG3 |

---

## 3. Temporal Gravity ladder

| Rung | Target TPS | Note |
|------|------------|------|
| TG32 | 31.25 | Entry |
| TG20 | 50 | |
| TG16 | 62.5 | |
| TG12 | 83.3 | |
| TG10 | 100 | |
| TG8 | 125 | |
| TG5 | 200 | |
| TG4 | 250 | |
| **TG3** | **333** | **Mandatory stop-for-human-review** |
| TG2 | 500 | Human-authorized only after TG3 |
| TG1 | 1000 | Final scientific target class |

**TG3 is a review threshold, not the final scientific target.**

On TG3 clear:

```text
stop autonomous promotion
checkpoint
seal complete evidence
emit TG3_REVIEW_REQUIRED
notify human
```

Human / protected controller may then promote, continue toward TG2/TG1,
authorize representation changes, or generalize into family megakernels.

---

## 4. Every TG rung requires

From bible §10 (also `TG_RUNG_REQUIREMENTS`):

```text
same model
same capability tier
complete-token timing
batch 1 base runtime
CLEAN benchmark
fallback=0
real GPU dispatch
stable p99
prompt-dependent coherent generation
```

Harness evaluation: `evaluate_tg_rung()` — withholds or rejects rather than
inflating PASS.

---

## 5. Separated metrics (never blend)

| Scoreboard | Meaning | Type locus |
|------------|---------|------------|
| `BASE_TRUE_TPS` | Decode base path, speculation **off** | `BaseTrueTps` / Python cell |
| `BLOCK_EXECUTED_TPS` | Device-block work only | `BlockExecutedTps` |
| `ACCELERATED_ACCEPTED_TPS` | Accepted tokens / (draft+verify+rollback) | `AcceleratedAcceptedTps` |
| `PREFILL_TPS` | Prefill phase only | `PrefillTps` |
| `TTFT` | Time to first token (seconds) | `TtftSeconds` |
| `HCLI_TOOL_AUGMENTED_THROUGHPUT` | Product path with tools | Python cell (product later) |

**Forbidden:** `mean_tps`, averages across scoreboards, draft-only as
“accepted TPS”, CPU diagnostic TPS as `BASE_TRUE_TPS`.

Rust generalization of tonight’s pack:

- Prior: `BaseTrueTps` + `AcceleratedAcceptedTps` in `metrics_sep.rs`
- Now: + `BlockExecutedTps`, `PrefillTps`, `TtftSeconds`, `SeparatedTgScoreboard`
- Python: `empty_separated_scoreboard()` + `assert_no_blended_tps()`

DSV4F pattern to preserve: child baseline scoreboard with
`BASE_TRUE_TPS: { value: null, status: WITHHELD }` until eligible
(`tools/condense/tests/test_deepseek_v4_child_baseline.py`).

---

## 6. Complete-token profiler (generalize DSV4F)

### DSV4F reference

- `lab/operators/deepseek_v4_gravity.py` — `_DiagnosticTokenProfiler`,
  `_COMPLETE_TOKEN_PROFILE_STAGES`, `_aggregate_complete_token_profile`
- `tools/condense/tests/test_deepseek_v4_complete_token_profile.py`
  — all stages present, `PASS_ALL_TIME_EXPLICITLY_NAMED`, no OTHER bucket

### Ascension scaffold

- `CompleteTokenProfiler` parameterized by family stage inventory
- `scaffold_complete_token_profile(family)` zero-fills with correct stages
- Timing accounting fields mirror DSV4F:
  - `observed_complete_token_wall_elapsed_ms`
  - `named_stage_wall_elapsed_ms`
  - `unexplained_other_wall_elapsed_ms`
  - `other_share_percent`
  - target explained ≥ **98%** (bible §11)

Stages for MoE vs Next differ (see 30B / 80B plans); aggregation path is shared.

---

## 7. Measurement protocol (when eligible)

1. **Eligibility gate:** full-stack runtime on that model, fallback=0,
   real GPU, sealed parity at the capability tier under test.
2. **Prefill trial:** record `PREFILL_TPS` and `TTFT` separately.
3. **Decode base:** speculation off → `BASE_TRUE_TPS`; also log
   `BLOCK_EXECUTED_TPS` from GPU timestamps if available.
4. **Accelerated (if any):** full draft+verify+rollback ledger →
   `ACCELERATED_ACCEPTED_TPS` only.
5. **p99:** multi-trial complete-token wall; require stability contract.
6. **Coherence:** prompt-dependent generation checks (not null/collapse).
7. **Compare** `BASE_TRUE_TPS` to next TG target; seal rung receipt.
8. If rung is TG3 and base clears → `TG3_REVIEW_REQUIRED`, stop autonomy.

No eligible runtime → entire scoreboard remains `METRIC_WITHHELD` /
`BASE_TRUE_TPS_WITHHELD` (current scaffold state).

---

## 8. Rotation interaction (bible §31)

Self-TG results feed rotation:

| Trigger | Condition |
|---------|-----------|
| A | Model **descends** ≥1 named TG rung → rotate candidate |
| B | Two failed architectures + measured roofline + sealed bottleneck + named next change |
| TG3 | Always human review (not an auto-rotate) |

Winning kernel grammars freeze by geometry before source eviction.

---

## 9. Receipt schemas

| Schema | Purpose |
|--------|---------|
| `hawking.ascension.tg_gauntlet_receipt.v1` | Full gauntlet inventory |
| `hawking.ascension.tg_rung_receipt.v1` | One TG rung |
| `hawking.ascension.complete_token_profile.v1` | One token profile |
| `hawking.ascension.separated_tps_scoreboard.v1` | Separated metrics |
| `hawking.ascension.self_tg_loop.v1` | Loop step cursor |

---

## 10. Dual-model operation

| Model | Family | Gauntlet |
|-------|--------|----------|
| Qwen3-Coder-30B | `QWEN3_MOE` | Independent Self-TG |
| Qwen3-Coder-Next 80B | `QWEN3_NEXT` | Independent Self-TG |

Neither may promote the other into sandbox workforce without human/controller
review. Cross-model review (30B proposes, 80B adversarially reviews) is a
product pattern layered **on top of** these per-model harnesses.

---

## 11. Non-goals (this document / this session)

- No live TPS measurement (Qwen not local).
- No Self-TG mechanism implementation against real bottlenecks.
- No promotion of any TG rung.
- No blending of metrics “for convenience.”
- No Frankenstein / DSV4F forward-lane edits.

---

## 12. Acceptance for the scaffold (done when)

- [x] TG32…TG1 inventory with correct targets.
- [x] TG3 emits `TG3_REVIEW_REQUIRED`, not silent final PASS.
- [x] Separated scoreboard cells; blend fields rejected.
- [x] Complete-token profiler shape + family stages.
- [x] Self-TG loop step inventory sealed.
- [x] Rust newtypes for BLOCK / PREFILL / TTFT.
- [ ] Live profile against Qwen GPU runtime (post-gate).
- [ ] First earned TG rung sealed with fallback=0 evidence.
