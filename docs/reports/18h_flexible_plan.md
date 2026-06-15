# 18h flexible compute plan — short bursts, architectural work between

## Philosophy (the lesson from today)

The rigid 24h overnight plan FAILED twice — eagle5 v2/v3 silent OOM kills, 9-hour training jobs producing nothing usable. **Today's microbench pattern WORKED** — 2 minutes of compute produced more actionable data than the 9-hour eagle5 run.

**New rule:** compute happens in **~10-30 min targeted windows**. Architectural work fills the gaps. Every compute window has a single specific hypothesis it's testing.

## 18h budget allocation

| Block | Type | Time | Purpose |
|---|---|---|---|
| 1 | compute | 30 min | Round-2 microbench: K=16, n-gram K=8 — captures Session B's design point |
| 2 | architectural | 1-2 h | Investigate Session B's L4 K=4 regression, decide whether to ship K=16 vs scrap |
| 3 | compute | 15 min | Targeted parity: re-run Session A's parity test (handwritten since A's auto-test missing) |
| 4 | architectural | 1 h | Wire `--q8-kv` CLI flag (Session C's hidden lever) |
| 5 | compute | 15 min | Microbench with `--q8-kv` to measure Session C's real gain |
| 6 | architectural | 2-3 h | Build smaller draft head architecture (Session G prep) using V2-Lite-as-teacher |
| 7 | compute | 1-2 h | Train small draft head (5M params); periodic checkpoints, early stop on plateau |
| 8 | architectural | 1-2 h | Wire small head into spec-decode runtime |
| 9 | compute | 15 min | Microbench with small head + spec-decode |
| 10 | architectural | 2-3 h | MLA Phase 4 resurrection from branch |
| 11 | compute | 15 min | Microbench with MLA Phase 4 + all other levers |
| 12 | architectural | 1-2 h | Pre-baking different tier maps for next-round comparison |
| 13 | compute | 30 min | Microbench across 4 tier maps × 3 stacked configs |
| 14 | architectural | 1-2 h | Wrap memo with all findings |

**Total architectural:** ~10-13 h
**Total compute:** ~3-5 h
**Total:** ~14-18 h with slack

## Compute discipline (lessons from today)

1. **NEVER launch a >2 h compute job blindly.** Either checkpoint every 15 min or break it into smaller targeted measurements.

2. **Microbench > matrix.** A single-binary microbench (no cargo build, no parity gate, no overnight wrapper) finishes in 2 min and tells you 80% of what the full matrix would.

3. **Paired-delta is fine with Claude live.** Absolute numbers contaminate; relative deltas don't.

4. **Pause-aware everywhere.** Every compute script checks `artifacts/runs/PAUSE` between trials so a fast architectural insight can interrupt mid-flight.

5. **No silent failure.** If a training job hasn't written log.jsonl in 30 min, KILL IT. Today's eagle5 sat in a silent OOM-prep state for 2 hours.

6. **RAM ceiling.** M3 Pro 18 GB. Worktree agents + cargo build + dismantle binary + Claude = 4-5 GB peak each. **Maximum 2 concurrent heavy workloads.** No 3rd cargo, no 3rd model load.

## Decision points (NOT pre-committed work — depends on data)

After block 1 (K=16 microbench):
- IF Session B at K=16 ≥ baseline: ship spec-decode with K=16 default
- IF Session B at K=16 still net-negative: scrap B's changes, revert spec-decode to off-mode default, focus on small head (block 6+)

After block 5 (Q8 KV microbench):
- IF +2 tps or more: keep, add to STACK
- IF <+1 tps: deprioritize, focus on bigger levers

After block 9 (small head):
- IF small head + spec-decode net-positive on 3 of 5 prompt types: ship
- ELSE: deprioritize spec-decode entirely, focus on kernel work (F, MoE GEMM)

After block 11 (MLA Phase 4):
- IF +1 tps: ship
- ELSE: park (memory note says it had bench contention issues; might not show up)

## What this plan AVOIDS (lessons from today)

- ❌ No 9-hour eagle5-style training jobs
- ❌ No "full matrix" benches that need Claude quit
- ❌ No autonomous chain that runs without checkpoints
- ❌ No commitments to architectural work that isn't grounded in measured bottleneck data

## Pause discipline

Every script in the plan honors `artifacts/runs/PAUSE` between stages/trials. If you spot a faster path mid-block:
1. `bash tools/bench/pause_bench.sh`
2. Make the architectural change
3. `bash tools/bench/resume_bench.sh`

Pause cost: max one trial's worth of wasted compute (~30 sec). No lost data.

## Realistic 18h outcome

**Worst case** (Session B at K=16 fails, Session C is small, small-head training inconclusive):
- 27 → 30 tps clean (vocab-prune + tier_default + ngram K=4)
- Memory wrap with what was tried

**Mid case** (Session B at K=16 net-positive, small head works at modest level):
- 27 → 36-40 tps clean
- Spec-decode shipped at K=16 with eagle4 or small head

**Best case** (everything lands incrementally):
- 27 → 42-48 tps clean
- Clear path to 50 in the next 1-2 days of supervised work

## NOT in this plan (deferred to later)

- Sessions F (RMSNorm-matmul fusion) and G (small head full pipeline) flagged as ≥1 week each — they need decisions from this 18h's data before committing
- Long-context corpus rebuild — old corpus was deleted; new one only makes sense if eagle5/small-head training warrants it
- MoE GEMM custom shaders — biggest potential win but biggest time commitment; pick after path-to-50 lands
