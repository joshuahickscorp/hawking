# Step 2B setup — chain-K=4 acceptance-distribution harness

**Session:** continuation of Step 2A (same session — user accelerated to 2B after seeing K=1 tax was spread across all phases with no single fat target). See [phase_step2a_setup.md](phase_step2a_setup.md) for the K=1 verdict.
**Status:** SETUP COMPLETE — measurement deferred to user post-Cmd-Q.
**Goal:** Decide whether chain-K=4's 7.52-dec_tps regression vs off=26.87 is **head architectural** (acceptance too low) or **glue cost** (acceptance fine, outer_step bloated).

## The math we're testing

Per [path_to_100_repath.md §Step 2B math floor](../plans/path_to_100_repath.md):

```
chain_dec_tps = (1 + mean_accept) / outer_step_seconds
```

Break-even with off (26.87 tps) requires:

| outer_step inflation | break-even mean_accept | path-to-50 mean_accept | path-to-100 mean_accept |
|---|---|---|---|
| 1.5× off_step | 0.50 | 2.00 | 4.47 |
| 2.0× off_step | 1.00 | 3.00 | 6.46 |
| 3.0× off_step | 2.00 | 5.00 | 10.20 |
| 5.0× off_step | 4.00 | 9.00 | 18.40 |

The current 7.52 tps means one (or both) of:
- mean_accept ≈ 0 (head almost never predicts correctly) → **F.2 already ruled out a similar architecture; we're testing whether eagle4_v3 is also capped**
- outer_step >> 1.5× off_step (chain glue costs more than amortization can recover) → **L5 Lever B (chain-step pipelining) becomes the target**

## What's been built

### Harness — `tools/bench/path_to_100_step2b.sh`

Uses the **existing** `[spec/eagle4-chain]` log at [deepseek_v2.rs line ~1762](../../../../../crates/dismantle-core/src/model/deepseek_v2.rs) — no new instrumentation. The chain path already emits per outer iter:

```
[spec/eagle4-chain] K=4 accept=N/draft_actual_k step=X.Yms emit=M tps=Z
```

Two-phase clean-window run, refuses if `pgrep -i Claude` returns alive (exit 2):

**Phase 1** — off baseline. 3 prompts × 1 trial × sequential profile. Captures mean `dec_tps` → derives `off_step_ms` for break-even comparison.

**Phase 2** — eagle4 chain. 3 prompts × `--max-new-tokens 64` × `parallel-k` profile × `EAGLE4_CHAIN_K=4` (override via env). `DISMANTLE_SPEC_LOG=1` captures every outer iter. Per-iter rows parsed into `chain_steps.csv`:

```csv
prompt_idx,accept,draft_actual_k,step_ms,emit,inst_tps
```

**Analysis** — Python summarizer prints:

- Accept histogram across bins `0..=K` (with ASCII bar chart)
- `mean_accept`
- `median_outer_step_ms`
- `step_inflation` = `median_step / off_step_ms`
- `acceptance_to_break_even` = `step_inflation − 1`
- `implied_chain_tps` = `(1 + mean_accept) / median_step` — should match measured `chain_dec_tps`

### Gates

| Gate | Condition | Verdict |
|---|---|---|
| **Gate 1 — head wall** | `mean_accept < 0.5` | F.2 already ruled out medusa K=8 acceptance. If eagle4_v3 also fails this threshold, the draft-head architecture is the wall. Path-to-100 via chain-K=4 is closed. Options: F.3 (medusa Rust port retry), F.5 (hybrid tree), new head training. **Not a kernel problem.** |
| **Gate 2 — glue cost** | `mean_accept ≥ 1.0` AND `step ≥ 1.5× off_step` | Head is producing usable predictions but the outer step is too expensive. L5 Lever B (chain-step pipelining) + argbuf rollup + persistent threads become the implementation track. **Kernel/host problem.** |
| **Gate break-even** | `mean_accept ≥ step_inflation − 1` | Chain beats off. Focus shifts to further amortization (bigger K, head distillation, persistent threads). |

The break-even gate is a **derived metric** — even at low acceptance, if step inflation is also low (chain is cheap), we win. Conversely high acceptance can't save us from a bloated step.

## Allocation matrix

| Observation pattern | Implicated cause | Next lever | Realistic chain tps target |
|---|---|---|---|
| `mean_accept < 0.5` (Gate 1 fails) | Head architectural wall — eagle4_v3 K=4 acceptance is below floor | F.3 Rust port of medusa heads + retrain; F.5 hybrid tree | Chain track CLOSED at this gate; rescope to path-to-30/40 (off-mode only) |
| `mean_accept ≥ 1.0` AND `step_ms ≥ 1.5× off_step` (Gate 2 fails) | Glue cost — seed forward + K head proposes + verify batch + accept-or-reject overhead exceeds what acceptance recovers | L5 Lever B (chain-step pipelining), argbuf rollup, persistent threads, batched-K verify amortization | 17-30 chain dec_tps if glue is halved (median_step → 1.25× off_step) |
| `mean_accept ≥ break_even` (chain wins) | Already amortizing | Push K higher; head distillation for higher per-K acceptance; persistent threads | 30-60+ depending on head ceiling |
| Mixed: 0.5 ≤ mean_accept < break_even | Partial wins, head signal is real but insufficient at K=4 | Try K=2 (lower per-K cost, may have proportionally higher acceptance); retrain head with longer context window | 10-17 chain dec_tps at K=2 |
| Heavy-tail in accept distribution: lots of accept=0 AND lots of accept=K | Head has confidence-correlated correctness | Calibrated K-truncation (use K=4 when confident, K=1 when not) | Marginal; head distillation more impactful |

## What success unlocks

Per [path_to_100_repath.md §Sequencing recommendation](../plans/path_to_100_repath.md):

- **Gate 1 fails (head wall):** write `phase_step2b_negative.md` documenting the dead lever. Combined with Step 2A's "K=1 tax architectural" finding, this CLOSES path-to-100 via spec-decode. Track 1 (off-mode kernels) is the only remaining knob. Rescope to path-to-30/40.
- **Gate 2 fails (glue cost):** L5 Lever B becomes the next implementation target. With outer_step → 1.25× off_step at current acceptance, chain-K=4 lands at 17-30 dec_tps. Combined with Track 1 acceleration of off → 35, dual-track ceiling lands at 50-65.
- **Chain wins (break-even):** rare but plausible if eagle4_v3's K=4 acceptance is higher than F.2's medusa K=8 result. Focus shifts to amortization tuning.

## How to run

```bash
# 1. Cmd-Q Claude.app
# 2.
cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
./tools/bench/path_to_100_step2b.sh

# Optional: try K=8 (deeper chains)
EAGLE4_CHAIN_K=8 ./tools/bench/path_to_100_step2b.sh

# Expected wall-clock: ~3-5 min (3 off trials + 3 chain trials with spec_log overhead)
# Output: reports/path_to_90/_bench_step2b_<TS>/summary.txt
```

Paste back the **accept histogram** and the **gate verdict** block. The gate that fires determines next session's scope.

## Verification this session

| Check | Result |
|---|---|
| `bash -n tools/bench/path_to_100_step2b.sh` | syntax OK |
| `tools/bench/path_to_100_step2b.sh --help` | exits 0, prints config table |
| Claude-running refusal | exits 2 with explanatory error |
| No source code changes | Step 2B uses the existing `[spec/eagle4-chain]` emit at deepseek_v2.rs:1762 — zero new instrumentation. Step 2A's K=1 diagnostic block is untouched. |
| Step 2A diagnostic edits intact | `git diff --stat` confirms +10/+13/+4 in engine.rs / kernels/mod.rs / deepseek_v2.rs (test helper) |

## Step 2A finding carried into Step 2B

Step 2A measured chain prerequisites:
- **Capture forward (= off-mode forward at this position) = 37.3 ms** in metal at 32 tokens. Chain's seed forward IS this same call.
- **h_shared GPU path works** (0/31 fallback). Chain inherits the working GPU h_shared.
- **Head propose (Metal, single-call) = 1.3 ms**. Chain calls this K+1 times per outer iter = ~6.5 ms for K=4.
- **Head argmax = 3.6 ms**. Chain calls this K times per outer iter = ~14.4 ms for K=4 (plus the verifier's lm_head dispatch).

Rough composite outer_step prediction at K=4:
```
seed_capture (37 ms) + K+1 head proposes (6.5 ms) + K head argmaxes (14 ms)
  + batched verifier (37 ms × parallelization factor) + accept/rollback glue
≈ 80-110 ms / outer iter at mean_emit=1-2
```

That's ~2.2-3.0× off_step_ms (37.3 ms). For break-even, need mean_accept ≥ 1.2-2.0. **If the head can't clear that, the architectural-wall verdict lands.**
