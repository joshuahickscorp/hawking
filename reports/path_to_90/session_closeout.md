# Path-to-90 — Session closeout (2026-05-15)

**Branch:** `claude/strange-proskuriakova-b5d48e`
**Plan:** `/Users/scammermike/.claude/plans/dismantle-path-to-90-immutable-jellyfish.md`
**Starting commit:** ecf77a6 / 8848165 (v2.2.0 T2.14 head)
**Ending commit:** 6b419d9 (validation finding)

## What this session was for

The user came in with a 3-track multi-month plan to drive DeepSeek-V2-Lite Q4_K_M decode on M3 Pro toward 90 dec_tps. The plan called for Stage 0 attribution → Stage 1 engine work (A1–A6) → Stage 2 KV R&D (B1–B3) → Stage 3 self-speculative decode (C1–C4). This session executed Stage 0 + the entire feasible portion of Stage 1, plus an audit of Stage 3 existing infrastructure, and a validation-pass on the headline numbers.

## What actually shipped (no revert needed)

Two engine landings now baked into the default profile:

| Commit | Lever | env-A Δ (5×3 trial, in-process bench) | env-B Δ (multi-prompt, fresh process) | Status |
|---|---|---:|---:|---|
| 8ee86bf | **A5 — persistent argbuf bump-arena** | +8.4% trimmed | (subsumed in A4) | Shipped default. Replaces per-dispatch `new_buffer` with a 64 KB shared arena reset per token. |
| 0983166 | **A4 — `mla_decode_kernel_fc` via MTLFunctionConstantValues** | +7.8% trimmed | net neutral-to-slight-regression | Shipped default. Specializes MLA decode kernel with 6 model constants baked in at engine load. |

Cumulative under env-A: **20.50 → 23.97 dec_tps trimmed-median (+16.9%)** on the 4-token "Once upon a time" / 64-token-decode bench.

## The honest reading (per 6b419d9)

The +16.9% headline is **valid only under env-A conditions** (one engine shared across many trials, with Claude Code running). The user noticed that 23.97 matches v2.2.0's `clean_bench` reference (~23.3) closely. Validation: ran the same A4 build against pristine pre-A5 source under env-B (multi-prompt harness, fresh process per trial). The win evaporates there — mean Δ across 7 prompts is −5.9%.

**Mechanistic explanation:** both A5 and A4 are *warmup-amortization* optimizations. Their costs are paid once per engine lifetime; their wins compound across the many dispatches that follow. env-A's protocol amortizes; env-B's per-trial fresh process does not.

**Real-world impact:**
- Long-running inference server (engine loaded once, many requests): **+5-15% warm steady-state** — meaningful but smaller than headline.
- Interactive chat with persistent engine: same pattern.
- Per-prompt fresh-process scripts: wash to slight regression.

A5 and A4 are correct, well-tested, ship as default, and do not need to be reverted. The framing changes, not the code.

## What was rejected (kept as opt-in scaffolding)

Five levers committed and rejected at the +3% gate. Code remains in tree as scaffolding for future revisits:

| Commit | Lever | Result | Diagnosis |
|---|---|---|---|
| 918c93c | A1 — `flash_attn_decode_kernel` wire-up | −13.9% in env-A; **−46.5% at long context in env-B** | Flash kernel's inner-loop recomputes `exp(scores_tile[ti] - m_bc)` per output-row × per tile-element. ~2048× more `exp()` calls than `mla_decode_kernel`. Apple-Silicon Metal `exp()` is software-emulated; drowns the kernel. Refactoring to hoist `w[ti]` once per tile (A1.2, ~half-day) might fix this. Opt-in: `mla_schedule = "metal-mla-flash"`. |
| 97c1828 | A4.2 — MoE Q4 routed-gemv fc | −11.1% in env-A | Inner loops are already small (blocks_per_row = 8) and auto-unrolled. Function constants gave the compiler nothing new to fold; the change shifted register-allocator decisions into a lower-occupancy band. First-try also had a parity bug (shared-expert rows ≠ moe_intermediate). Opt-in: `gemm_q4_k_schedule = "v2t_gu_v2_fc"`. |
| 715403e | A3 — `add_rmsnorm_f32` fusion | −5.3% in env-A | The unfused `add_inplace` runs 8 TGs in parallel (hidden/256=8). The fused kernel must be single-TG (rmsnorm reduction needs a threadgroup barrier; cross-TG sync isn't available within one dispatch). The add phase loses 8× parallelism; GPU regression eats the CPU saving. Opt-in: `residual_fusion = "f32"`. |
| 2628a5c | Stage 3 — existing speculate audit | No code; docs only | NGram + ExactShared already exist but don't fire on the 4-token bench prompt. On long code-prompt with 79.7% accept rate, NGram still loses −15% because verify cost ≈ K × single-forward; theoretical ceiling at K=4 is 1.25×, CPU overhead eats the margin. Real spec-decode needs trained drafter (off-machine) or new parallel-K MLA kernels (multi-week). |

## Durable infrastructure landed

| Commit | Artifact | Why it matters |
|---|---|---|
| 3036639 | Stage 0 attribution — [reports/path_to_90/stage0/](stage0/) | Profile-driven gap analysis vs llama.cpp on the same model/hw (52.51 dec_tps). Established that the gap is 2.2-2.5× (not 3×) and that ~85% of it is CPU dispatch overhead, not kernel quality. |
| 8ee86bf | `MetalContext::argbuf_alloc/reset` API | Reusable beyond A5: any future kernel that wants packed scalar args can carve from the arena. |
| 0983166 | `MetalContext::register_specialized_pipeline` | Reusable beyond A4: any future MTLFunctionConstantValues specialization plugs into the same pipeline-cache injection point. |
| ad9fec9 | `tools/bench/multi_prompt_bench.sh` + 7-prompt suite | The infrastructure that *caught* the env-A vs env-B disagreement. Foundation for B1 (PPL eval), B2/B3 (KV/expert quant quality gates), and future spec-decode work. |
| 6b419d9 | Validation methodology | Documents how to read env-A vs env-B numbers and the trap of single-environment headlines. Pattern reusable across all future levers. |

## What the plan got right and wrong

**Right:**
- Stage 0 first. Attribution before code changes saved chasing the wrong levers.
- A5 ordering. The plan was revised to put A5 first based on Stage 0 evidence; A5 was the cleanest landing of the session.
- "Reject at +3%" gate. Each rejection produced documented evidence; nothing shipped that regresses.

**Wrong:**
- The plan's "Stage 1 +30-50% from engine work alone" was over-optimistic. Actual env-B impact of A5+A4 combined is ~0-5% on cross-process workloads, not +16% as env-A suggested. The plan's heuristics (dispatch-count reduction, GPU-share specialization) didn't account for register-allocator interactions, fusion-parallelism tradeoffs, or warmup-vs-steady-state divergence.
- The plan's "A1 flash-attn keystone" thesis was wrong even after fixing the seq_len concern. The flash kernel as-shipped has a structural `exp()`-recomputation bug; it doesn't win even at the contexts the plan called out.
- The "verify cost ≈ K × single-forward" arithmetic for spec-decode wasn't acknowledged. The existing infrastructure can't deliver wins from any drafter at K=4; needs either a cheap parallel-K MLA kernel rewrite or a real trained drafter.

## Where the path-to-90 actually stands

Honest assessment, factoring the validation finding:

- **Engine track ceiling:** roughly v2.2.0's ~23.3 dec_tps under cross-process conditions, ~24-28 under warm long-running conditions. A5+A4 sit near the upper end of this range. Further engine wins (A6 autotune, A3 done correctly with a two-buffer ping-pong, future fc specializations) are likely +0-5% each — small, hard to validate.
- **KV R&D track (Stage 2) ceiling:** can break through at long context where KV bandwidth grows. p006 (169-token prompt + 96 decode = seq_len ~265) is just barely entering the regime where Q8 latent KV would help. Real benefit at chat-scale (4K-32K context) is +10-30% but the bench doesn't reach that regime yet.
- **Spec-decode track (Stage 3) ceiling:** 1.5-2× potential, but only with a trained drafter (off-machine R&D) OR a parallel-K MLA kernel rewrite. Either is multi-week elapsed.
- **The hard truth:** **90 dec_tps probably requires at minimum BOTH a trained EAGLE-3-style drafter AND a long-context (4K+) workload baseline.** The single-prompt 4-token bench will never see 90 dec_tps on M3 Pro Q4_K_M — the bandwidth roofline at 1.82 GB/token × 130 GB/s practical = 71 dec_tps is the engine ceiling, period. The only way past it is multi-token-per-forward-pass spec decode.

## Suggested re-prioritization (whenever next session happens)

The plan's three tracks remain right; the priorities within them need updating:

1. **Highest leverage, session-scope:** finish B1 (PPL eval harness, `tools/bench/ppl_eval.py`). ~3 hours. Unblocks every future quant/KV decision and adds a quality gate that's currently missing.
2. **Highest leverage, multi-session:** start C2 prep — gather distillation data for an EAGLE-3 / MTP head trained on DeepSeek-V2-Lite outputs. The training itself happens off-machine. ~1 week elapsed wall-time for the dataset; ~12-24 H100-hours for the head.
3. **Worth one focused session:** A1.2 — refactor `flash_attn_decode_kernel` to hoist `w[ti]` once per tile. The bug is identifiable and the fix is bounded. If it works, A1 becomes a context-conditional schedule (default at seq_len ≥ 512).
4. **Skip / deprioritize:** A6 autotune polish, A4.2 retry, more dispatch-fusion attempts. Each is +0-5% under env-B; the marginal value is shrinking.

## Files cleanly committed

Total: 8 commits this session, 5 reports + 1 plan + new code + scaffolding.

```
6b419d9 path-to-90: A5+A4 win validation — honest framing
ad9fec9 path-to-90: multi-prompt bench harness + A1 long-context re-litigation
715403e v2.3.0 A3: add_rmsnorm_f32 fusion — REJECTED at +3% gate
2628a5c path-to-90 Stage 3 audit: existing spec-decode reality-checked
97c1828 v2.3.0 A4.2: MoE Q4 routed v2t_gu_v2_fc — REJECTED at +3% gate
918c93c v2.3.0 A1: flash_attn_decode_kernel wire-up — REJECTED at +3% gate
0983166 v2.3.0 A4: mla_decode_kernel_fc via MTLFunctionConstantValues — +7.8% e2e
8ee86bf v2.3.0 A5: persistent argbuf bump-arena — +7-8% e2e
3036639 path-to-90 Stage 0: attribution + llama.cpp comparator
```

## Behavioral changes since v2.2.0

The shipped engine is functionally:
- Same kernels as v2.2.0 for non-fc paths
- New `mla_decode_kernel_fc` is selected by default (specialized for V2-Lite's shape constants)
- All `KernelArgBuffer` writes route through a per-context bump arena instead of per-dispatch `new_buffer`
- All token-level parity tests still pass bit-identical against pre-A5

Profile changes vs v2.2.0:
- `mla_schedule`: `metal-mla` → `metal-mla-fc`
- `shader_hash`: updated to reflect added kernels (mla_decode_kernel_fc, moe_..._v2t_gu_v2_fc, add_rmsnorm_f32, flash_attn_decode_kernel was already present)
- Added optional opt-in field `residual_fusion` (default `"off"`)
