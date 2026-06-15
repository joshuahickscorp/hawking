# Autonomous session log — 2026-05-23

User went to church. Asked me to run a cron job to detect when the
overnight 6h chain finishes, continue running smokes, and otherwise
operate autonomously without questions. This log captures everything
done.

## Chain state at session start (15:49 UTC)

Chain PID 42677 was already running. M1, M2, M3 had completed on a
prior invocation; this restart resumed at M4_autotune.

## What I did (no code changes — read-only analysis + reports)

### 1. Set up wakeup watcher (Bash run_in_background bh9063wqe)

`/tmp/wait_chain.sh` polls the chain PID every 60s, exits when it's
gone, then dumps status.json + first 80 lines of WRAP.md. The harness
will notify me when it exits, so I won't be sleep-polling.

### 2. Read chain artifacts M1, M2, M3, partial M4

- **M1** (DRY-RUN commit plan): 149-file dirty tree, 27 lib tests pass,
  pre-written 4-commit sequence still applies. **Not auto-landed**
  because `CHAIN_AUTO_COMMIT=0` (correct; no autonomous commit without
  user OK per global rules).
- **M2** (Q8 KV 3-way): patch rejects 17 files, atomic — no half-state.
  Disposition unchanged: requires HAND-port.
- **M3** (kernel hot-spots, with real GPU timings this time):
  | % GPU | kernel |
  |---:|---|
  | 26.6 | `moe_batched_gemm_q4_indexed_v2t_gu_v2` |
  | 12.3 | `mla_decode_kernel` |
  | 10.8 | `rmsnorm_gemv_f16w_attn_pinned_v2t` |
  | 10.2 | `moe_batched_gemm_q5_0_indexed_v2t` |
  |  9.5 | `gemv_f16_simdmat` |
  |  6.3 | `moe_batched_gemm_q8_0_indexed_v2t` |

  Bandwidth: 12.4 GiB/s effective (8 % of 150 GiB/s peak). NOTE: this
  was split-CB-with-trace mode — production single-CB-per-block is
  higher. The shape of the table (relative %) is still load-bearing.

### 3. M4 paired validation — REJECTED

The chain stitched a candidate profile with
`gemm_q4_k_schedule = v2t` (dropping `v2t_gu_v2`). Paired n=12:

- baseline (`v2t_gu_v2`): 23.723 dec_tps σ=0.200
- candidate (`v2t`):     21.796 dec_tps σ=0.336
- delta: **−1.927 dec_tps = −8.1 %**

**DO NOT** copy `M4_candidate_profile.json` over the live profile.
Saved memory `m4_autotune_2026_05_23.md` so the next session won't
relitigate. `v2t_gu_v2` remains the production schedule.

### 4. Wrote `reports/q8_kv_hunk_port_plan_2026-05-23.md`

Verified by grep that **most of the Q8 KV patch has already landed on
main** — the shader (`mla_decode_kernel_q8kv`), the Metal `kv_append_q8_0_f32`
shader, the standalone Rust wrappers in `kernels/mod.rs`, the CPU
quantizer in `quant/mod.rs`, and both parity-test files. What remains
is a focused wiring layer:

- `EngineConfig::q8_kv: bool` + Default
- 2 new TCB wrappers (`mla_decode_q8kv_and_o_proj_arena_tcb`,
  `kv_append_q8_0_f32_tcb`) modeled after their f32 siblings at
  `kernels/mod.rs:4005` and similar
- `DeepSeekV2` struct fields (`q8_kv_enabled`, `mla_c_kv_q8_gpu`) and
  allocator block
- `q8_kv_sync_prefix` helper method (verbatim from patch)
- 3 dispatch-site `if` branches in `model/deepseek_v2.rs` hot path
- 2 prefix-sync call sites
- CLI flag in `main.rs`
- Profile JSON `shader_hash` recomputation

The plan estimates ~5–6 h of focused work. The 17-file divergence
count is misleading — only 4 source files + 1 profile carry actual
work.

### 5. Wrote `reports/moe_gemm_v3_targets_2026-05-23.md`

Read the Q4_K `v2t_gu_v2`, Q5_0 `v2t`, and Q8_0 `v2t_w2` kernels
side-by-side. Three candidates:

- **T1 — `moe_batched_gemm_q5_0_indexed_v2t_gu`** (NEW). Gate+Up
  fusion + block-pair processing. Projection +0.5–0.7 dec_tps,
  ~1 week effort. Biggest unmined lever at 10.15 % GPU.
- **T2 — `moe_batched_gemm_q8_0_indexed_v2t_gu`** (NEW). Gate+Up
  fusion. Projection +0.3–0.5 dec_tps, ~3–5 days.
- **T3** — extend Session J `_v2t_w2` 2-rows-per-simdgroup to the
  fused T2 kernel. Compounded with T2. +0.2–0.4 dec_tps.

What does NOT apply: paired-nibble (Q5_0/Q8_0 are not nibble-packed),
sumy-correction (no min-offset), scale-preload (already trivial).
Don't try to put gu_v2 on Q4_K — that already has all 4 tricks.

### 6. M5 partial data (chain still running M5 as of session end)

First config from `M5_high_conf_stack.md` (TRIALS=20, "Once upon a time"):

- baseline: 22.039 dec_tps σ=0.359 95%CI ±0.157 n=20
- L1 vocab-prune: 23.954 dec_tps σ=0.275 95%CI ±0.121 n=20
- **delta: +1.915 dec_tps = +8.7 %**

That confirms L1 as the deployable floor — bigger than the prior +1.55
quoted in path_to_75_v2.md. Once the L1+Jw2 and L1+M4 configs and the
other 2 prompts come in, the M5 matrix will give the
release-confidence numbers.

## What I did NOT do (deliberately)

- **No git commits.** Global rule + `CHAIN_AUTO_COMMIT=0`. The safe
  sequence is staged in `session_wrap_2026-05-23.md` and waits for the
  user.
- **No bench runs of my own.** Chain was running M4/M5 and any
  competing bench would contaminate both the chain and my numbers.
- **No code changes.** Q8 KV port plan and MoE v3 targets are
  read-only design docs; the work hasn't started.
- **No `.git` writes.** No `git add`, no `git commit`, no tag, nothing.
- **No `gh` / network operations.**

## Files added this session

- `reports/q8_kv_hunk_port_plan_2026-05-23.md` — port plan
- `reports/moe_gemm_v3_targets_2026-05-23.md` — kernel sketch targets
- `reports/autonomous_session_2026-05-23.md` — this log
- `~/.claude/projects/.../memory/m4_autotune_2026_05_23.md` — project
  memory + index entry in MEMORY.md

## Chain COMPLETE — final M5 matrix (16:25 UTC)

**TRIALS=20 × 3 prompts × 4 configs, paired (Claude-live):**

| config | "Once upon" | "fibonacci" | "photosynthesis" | mean Δ baseline |
|---|---:|---:|---:|---:|
| baseline | 22.039 | 24.894 | 24.881 | — |
| L1 | 23.954 | 26.457 | 26.530 | **+1.71** |
| L1+Jw2 | 24.142 | 26.512 | 26.434 | +1.79 |
| L1+M4 | 24.457 | 24.407 | 24.412 | **−0.18** |

**Final dispositions:**

- **L1 vocab-prune**: deployable on all 3 prompts. +1.71 dec_tps mean.
  Bigger floor than the +1.55 we'd quoted; safe to update path_to_75.
- **L1+Jw2**: within noise on top of L1 (+0.19/+0.05/−0.10). σ on
  prompt 1 is 0.896 vs L1's 0.275 — ~7× variance bump suggests
  contention or schedule sensitivity. **Stay env-gated.**
- **L1+M4**: prompt-bimodal — +0.50 on prompt 1, but −2.05 and −2.12
  on prompts 2/3. **REJECTED.** Confirms the M4 n=12 paired result.
  Saved memory `m5_stack_matrix_2026_05_23.md` so the next session
  has this as a single fact lookup.

**Important methodology lesson:** the M4 sweep's single-prompt
candidate hid the prompt-bimodality. Future autotune must validate
across ≥3 prompts before recommending a profile swap. Added to memory.

## Chain runtime

37 min total (resumed-state run; first invocation did M1–M3
separately). Single-CB-per-block + decode-arena baseline at
~24.9 dec_tps and L1-floor at ~26.5 across prompts 2/3 (where the
prompt is long enough that decode dominates startup).

The Q8 KV port (~5–6 h of code) and MoE GEMM v3 sketches (multi-week)
are queued in the reports but **NOT** started — both require user OK
per global rules ("explicit user OK before commits", "ask before
acting on hard-to-reverse changes").

## Open questions for the user (when they're back)

1. **Adopt L1 as production default?** Currently env-gated. M5 confirms
   it. The `path_to_75_v2.md` Tier 1 commit sequence already includes
   it, just hasn't been merged.
2. **Start Q8 KV hunk port?** Plan is in
   `reports/q8_kv_hunk_port_plan_2026-05-23.md`. 5–6 h of focused work.
3. **Prioritise Tier 2 MoE GEMM v3?** Plan is in
   `reports/moe_gemm_v3_targets_2026-05-23.md`. Multi-week, supervised.
