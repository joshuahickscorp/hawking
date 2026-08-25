# Ascension Complete-Token Profiler and FLOPS Ledger Contract (Bible §11)

**Status:** CONTRACT SEALED (planning only) — implementation **NOT_STARTED**  
**Bible:** `HAWKING_ASCENSION_BIBLE.md` §11  
**Machine-readable:** [`ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.json`](./ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.json)  
**Scaffold code:** `lab/operators/ascension_tg_gauntlet.py` (`CompleteTokenProfiler`)  
**Related plans:** [`ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md`](./ASCENSION_TG_GAUNTLET_HARNESS_PLAN.md), 30B/80B parity plans  
**Schedule steps:** 13 (30B profiler), 20 (80B profiler) — both remain `NOT_STARTED`

---

## 1. Purpose

Define the **exact required observations** and **refusal conditions** for a
complete-token profile and FLOPS ledger so Self-TG and parity promotion cannot
accept vibes, blended TPS, or unmeasured GPU claims.

This contract **does not claim any live Qwen performance**. No Qwen3-Coder-30B
or Qwen3-Coder-Next-80B complete-token values, FLOPS ledgers, or BASE_TRUE_TPS
numbers currently exist. DSV4F / Metal receipts may inform mechanism class only.

---

## 2. Primary rule (bible §11)

> Explain at least **98%** of complete-token latency.

```text
target_explained_percent = 98.0
timing_pass_status       = PASS_ALL_TIME_EXPLICITLY_NAMED
timing_fail_status       = FAIL_UNEXPLAINED_OTHER_WALL
```

Scaffold accounting fields (must appear on sealed profiles):

```text
observed_complete_token_wall_elapsed_ms
observed_complete_token_cpu_duration_ms
named_stage_wall_elapsed_ms
unexplained_other_wall_elapsed_ms
other_share_percent
status
target_explained_percent
```

---

## 3. Required stage inventory (semantic)

Bible §11 requires the complete token to name:

```text
tokenizer/template
embedding
norm/state
QKV
attention or DeltaNet
KV/state
router
expert selection
expert gather
gate/up
activation
down
shared expert
combine
residual
head
sampling
readback
HCLI stream
```

Family-parameterized stage **keys** (scaffold) live in
`lab/operators/ascension_parity_ladder.py`:

| Family | Constant |
|--------|----------|
| `QWEN3_MOE` (30B) | `QWEN3_MOE_STAGES` |
| `QWEN3_NEXT` (80B) | `QWEN3_NEXT_STAGES` (adds DeltaNet / hybrid / state accounting) |

Missing a required semantic stage on an **authority** profile is a refusal
condition, not a silent OTHER bucket.

---

## 4. Required per-stage observations

Every named stage on an authority profile must record:

```text
GPU duration
CPU duration
bytes read
bytes written
theoretical operations
executed operations
useful operations
redundant operations
achieved FLOPS
occupancy
arithmetic intensity
reuse factor
dispatches
command buffers
waits
p50
p95
p99
fallback
```

Scaffold stage rows today carry a **subset** (wall/cpu/gpu/dispatches/bytes
estimates + status placeholders). Filling the full observation set is
**implementation work after** eligible runtime — not a claim that Qwen values
exist.

---

## 5. Required global ledgers

Maintain programme-level ledgers:

```text
PEAK_UTILIZATION
FLOPS_PER_TOKEN
BYTES_PER_FLOP
REUSE_FACTOR
CRITICAL_DEPTH
STATE_TRAFFIC
```

| Ledger | Role |
|--------|------|
| `PEAK_UTILIZATION` | Fraction of peak useful work attained |
| `FLOPS_PER_TOKEN` | Executed / useful ops per complete token |
| `BYTES_PER_FLOP` | Bandwidth pressure relative to compute |
| `REUSE_FACTOR` | Weight / activation reuse across work |
| `CRITICAL_DEPTH` | Serial depth of the token graph |
| `STATE_TRAFFIC` | KV / DeltaNet / hybrid state bytes moved |

**Current values for Qwen:** none. Ledgers must remain null / withheld until a
sealed full-stack runtime produces them.

---

## 6. Ranking rule (not FLOPS-max)

> Higher achieved FLOPS is **not** automatically better.  
> A candidate that performs fewer useful operations can be faster with lower FLOPS.

Primary ranking for mechanism promotion:

```text
complete wall time + p99
```

Not primary:

```text
achieved FLOPS alone
microbench without complete token
```

---

## 7. Refusal conditions (exact)

| ID | When | Action |
|----|------|--------|
| `REFUSE_UNEXPLAINED_OTHER_ABOVE_TARGET` | `other_share_percent > 2%` (i.e. explained &lt; 98%) | Reject profile as complete-token authority |
| `REFUSE_OTHER_BUCKET_AS_NAMED_STAGE` | Catch-all OTHER hides wall time | Reject profile |
| `REFUSE_LIVE_QWEN_CLAIM_WITHOUT_RUNTIME` | Qwen FLOPS/TPS/p99 asserted without sealed full stack | `METRIC_WITHHELD` / `REJECT_CLAIM_BOUNDARY` |
| `REFUSE_BASE_TRUE_TPS_FROM_PROFILE_ALONE` | Profile without TG eligibility gates | `BASE_TRUE_TPS_WITHHELD` |
| `REFUSE_HIGHER_FLOPS_AS_AUTOMATIC_WIN` | Rank by FLOPS without complete wall + p99 | Reject ranking |
| `REFUSE_BLENDED_TPS` | mean/avg/blended TPS across scoreboards | Reject scoreboard |
| `REFUSE_MISSING_GLOBAL_LEDGER` | Any of the six global ledgers missing on authority profile | Reject as incomplete ledger |
| `REFUSE_MISSING_PER_STAGE_OBSERVATION` | Required per-stage field absent on authority profile | Reject as incomplete stage observation |

TG eligibility (from bible §10 / TG harness) still applies before any
`BASE_TRUE_TPS` cell may leave withheld: same model, same capability tier,
complete-token timing, batch-1 base runtime, CLEAN benchmark, fallback=0, real
GPU dispatch, stable p99, prompt-dependent coherent generation.

---

## 8. Honesty boundary (mandatory)

```text
live_qwen_profile          = false
live_qwen_flops_ledger     = false
base_true_tps_claimed      = false
performance_claims         = false
stage_ready                = false
```

Do **not**:

- paste synthetic Qwen TPS or FLOPS into plans or receipts,
- treat DSV4F measured FLOPS as Qwen FLOPS,
- mark schedule steps 13 or 20 complete from scaffold shape alone.

---

## 9. Integration

| Surface | Role |
|---------|------|
| TG harness plan §6 | Measurement shape and separated scoreboards |
| 30B / 80B parity plans | Family stage inventories |
| Master schedule steps 13, 20 | Companion doc for profiler work |
| Blocked-state registry | Profiler evidence is a **required blocker** before unblocking either Qwen bootstrap model |

---

## 10. Acceptance for this planning contract

- [x] 98% explained-latency rule recorded.
- [x] Bible stage inventory + per-stage observations + global ledgers listed.
- [x] Refusal conditions enumerated with actions.
- [x] Explicit: no live Qwen values; no performance claims.
- [x] Machine-readable JSON + structural validation tests.
- [ ] Live Qwen authority profiles (post full-stack runtime).
- [ ] Global ledgers populated from sealed measurements.

---

## 11. Non-goals

- No live Qwen download, serve, or Metal profile in this wave.
- No BASE_TRUE_TPS or TG rung claims.
- No change to DSV4F sealed receipts.
- No schedule / completion-state status flips.
