# Path-to-75 v2 — updated with this session's empirical findings

Supersedes the prior path-to-75 plans. Built from M9 variance hunt + M1
stack matrix + M5 K-sweep + M7 final-confidence data. **Everything else
is speculation grounded in memory notes, not measurement.**

## Today's empirical floor

| Config | dec_tps (Claude-live, paired) | Clean equivalent | Notes |
|---|---:|---:|---|
| baseline | 24.88 | ~27 | M7 |
| **L1 vocab-prune** | **26.43** | **~28.7** | **shipped, σ=0.12** |
| best stacked attempt | 26.43 | ~28.7 | no compound past L1 (M1) |

**Honest start point for path-to-75: ~28.7 clean. Gap to 75 = ~46 tps (1.6× multiplier).**

### Update — 2026-05-23 overnight 6h chain M5

Chain M5 ran TRIALS=20 × 3 prompts × 4 configs (paired, Claude-live).
L1's deployable floor confirmed across 3 prompts:

| prompt | baseline | L1 | Δ |
|---|---:|---:|---:|
| "Once upon a time" | 22.039 | 23.954 | +1.92 |
| "def fibonacci(n):" | 24.894 | 26.457 | +1.56 |
| "Explain photosynthesis briefly:" | 24.881 | 26.530 | +1.65 |

**Mean L1 Δ = +1.71 dec_tps** (vs the +1.55 prior). Floor verified.

L1+Jw2 within noise on top of L1 (+0.19/+0.05/−0.10), σ on prompt 1
ballooned 7× — keep env-gated.

L1+M4 candidate-profile is **prompt-bimodal**: +0.50 on prompt 1,
−2.05 on prompt 2, −2.12 on prompt 3. **REJECTED** — do not adopt
`M4_candidate_profile.json`. See memory `m5_stack_matrix_2026_05_23.md`
and `m4_autotune_2026_05_23.md`.

## The 5 levers that actually exist (ranked by realistic gain × confidence)

| # | Lever | Confidence | Realistic gain | Cost | Status |
|---|---|---|---|---|---|
| 1 | **MoE GEMM custom shaders** | HIGH (50.5% of decode) | +5-15 tps | 2-4 weeks Metal | Sketch only (J w2 +1.33%) |
| 2 | **RMSNorm + matmul fusion** | HIGH (24% of decode) | +3-7 tps | 2-3 weeks Metal | 1/6 sites wired (F) |
| 3 | **Q8 KV runtime wiring (real)** | MEDIUM | +1-3 tps short / +3-5 tps long-context | 3-5 days Rust | **patch broken; needs full debug** |
| 4 | **Smaller draft head (5M params)** | MEDIUM | +5-10 tps (requires #5 first) | 1-2 weeks training + wiring | Never attempted; eagle5 failed 4× |
| 5 | **Spec-decode runtime cost reduction** | LOW (B's attempt regressed) | +5-15 tps if #4 lands | 1-2 weeks profiling | Reverted (K); root cause undiscovered |

**Sum at midpoint:** ~22 tps. **From 28.7 + 22 = ~50.7 tps.**

**75 tps requires either:**
- Compounding beyond midpoint estimates (gains arrive bigger than projected), OR
- A 6th lever we haven't identified, OR
- A 2-3× spec-decode win once draft cost is lowered

## Workstream ordering by ROI / risk

### Tier 1 — Highest ROI, do first (1-3 days each)
1. **Commit the safe sequence** (vocab + K + J w2 + infra) — locks the baseline. Per session_wrap_2026-05-23.md.
2. **Debug Q8 KV patch** — measured wiring failure; root-cause +flag fix. 1-2 days. Unlocks +1-5 tps.

### Tier 2 — Multi-week, parallel-safe
3. **MoE GEMM kernel work** — biggest lever. Start with one shape (1408×2048 routed MoE down). Each kernel iteration is 1 day. Stop when you have a parity-clean variant ≥10% faster.
4. **RMSNorm fusion completion** — wire remaining 5/6 sites. Each site ~1 day. Memory has `rmsnorm_fusion_sketch.md`.

### Tier 3 — Speculative, defer until 1+2+3+4 land
5. **Spec-decode runtime cost reduction** — second attempt after B's failure. Profile first, fix the actual hot spot, not the imagined one.
6. **Smaller draft head** — only after spec-decode is net-positive at all. Otherwise even a perfect head doesn't help.

## What this session's autonomous run can deliver TONIGHT

Realistic single-run scope:

- **Tier 1 item 1 (commits)** — automate via a chain that stages each commit, runs `cargo test`, commits if green. ~15 min compute, low risk.
- **Tier 1 item 2 (Q8 KV debug)** — chain: re-apply patch, rebuild, smoke `--q8-kv` flag, run parity, run microbench. ~30 min compute, may fail (that's data).
- **MoE GEMM characterization** — chain: trace-dispatch on baseline, identify top-3 kernel call hot spots by total ms, dump to report. ~10 min compute. Foundation for the eventual kernel sketch.
- **Stack confirmation TRIALS=30** — repeat M9 variance hunt at TRIALS=30 on 3 prompt types. ~30 min. High-confidence number for the release notes.

Total: **~85 min focused compute, ~2-3h with all module overhead.**

What this CANNOT deliver tonight:
- Writing custom Metal shaders (multi-day; not appropriate unsupervised)
- Training a small draft head (5-10h training + risk of silent failure)
- Operator fusion of new sites (per-site Rust + parity test; risk of mid-flight parity break)

## Lessons banked from this session

1. **TRIALS=3 is noise; TRIALS=15 is signal; TRIALS=30 gives 95% CI ±0.04**
2. **"Stacks" don't always compound** — measure pair-wise before assuming additive
3. **Patches that "apply" can be no-ops** — always verify the artifact (flag presence, kernel registered, etc.)
4. **Silent failures kill overnight value** — eagle5 ran 2h with zero rows; current chain logs every transition
5. **Worktree agents OOM-kill under parallel load on M3 Pro 18 GB** — serialize cargo work
6. **Q8 KV is the highest-EV "should work" lever** — kernels exist, parity passes, wiring fails — top of the bug list
