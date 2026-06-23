# Autonomous Run Log (campaign 2026-06-21)

Chronological. Newest at bottom. Recovery: if disconnected, read `findings_summary.md` + `roadmap.md` +
the IN-FLIGHT section below, then resume the top unstatused roadmap item. Watchdog: a `sleep 1700` heartbeat
re-invokes the manager ~every 28 min; job/agent completions also re-invoke it.

## Phase 0 — pivot from spec-decode
- Spec-decode (EH free-market + trained EAGLE) concluded **net-negative for speed** (per-cycle overhead wall).
  Lossless cost-aware router committed `73fc5b4` (default-OFF). Pivot to bandwidth-bound decode levers.

## Phase 1 — config/KV sweep (warm)
- Established warm baseline ~40 tps; discovered cold single-runs = PSO-compile artifact (the campaign-wide method fix).
- `FFN_DOWN_Q4K` "+29%" → **cold artifact**, warm ~0%. REJECTED.
- `--profile fast` → initially measured as +7.5%, later refined to **~+3–7% at the bench noise floor**,
  83–90% argmax-identity. Speed-priority config, mild trade.
- `F16_KV` → −50% KV, +1.9%@2.5k (scales), 88% identity. Clean long-ctx/footprint lever.
- `int4-KV` (per-row) → SLOWER + 0% quality. REJECTED (per-row).
- Long-ctx KV wall measured: 40 → 18.8 (2.5k) → 8.6 (8k) = 4.6× drop.
- Deliverable written: `docs/plans/ratios_roadmap_2026_06_21.md`.

## Phase 2 — adversarial red-team (ULTRACODE workflow `w69da85gs`, 8 agents)
- BROKE the "engine near ceiling" conclusion. Surfaced + I then VERIFIED each (kill-ledger rule):
  - Lever #1 Q6_K row-blocking: red-team OVER-CLAIMED "2r unreachable" → **2r is already default + optimal** (A/B 2r=40.48 best). CLOSED.
  - Lever #2 per-channel int4-KV: **CONFIRMED real** (built + validated + dead-called, cosine 0.998). → wiring.
  - Lever #3 1.6× gap: **MLX-diffable** (not MST-only). → research.
- Roadmap hardened with corrections (F16_KV is the *weaker* compression option; int4-KV NO-GO is *per-row only*; Q6_K predec confirmed null).

## Phase 3 — build/validate + infrastructure (current)
- **RWKV-7 SSM moat CONFIRMED (decisive):** 118.6 / 110.6 / 119.4 tps (short / 2.5k / 8k) = FLAT; vs Qwen 40 / 18.8 / 8.6
  (−78%). **~14× faster @8k.** The long-context differentiator is the SSM path. (mamba2-370M corroboration bench running `bv2zdsqso`.)
- **Architecture pass:** decode-kernel micro-opt is genuinely tapped (bible §3.0 A5/A6/A7/A10 + my Q6_K 1r/2r/4r A/B);
  dead-called wrapper scan found per-channel int4-KV (the live lever) + ~30 parity-tested A/B variants (intentional).
- **Quality lane:** adversarial 10-prompt suite (code/math/JSON/multilingual/format) → `--profile fast` 90%, `F16_KV` 100%.
- **INFRASTRUCTURE BUILT (goal rule 6):** `tools/bench/ratios.sh` (warm-median tps + qual harness, self-validated),
  `docs/env_flags.md` (284-flag curated reference), `docs/architecture.md` (subsystem map), the 5 `docs/campaign/` artifacts.

## IN-FLIGHT (for recovery)
- ❌ **Agent `a5e71cfac57e799e2` (int4-KV wiring) FAILED** — died on process restart, BEFORE editing the hot path (no leak;
  tree verified clean). int4-KV-PC NOT wired. DEFERRED: the RWKV-7 SSM moat is the better long-ctx answer (no KV wall at all),
  which lowers this lever's strategic urgency. Documented as top ready-to-wire lever (roadmap #1) with exact steps from `dead_levers.md:401`.
- ❌ **Agent `af8c77df9de78b596` (MLX-diff) FAILED** — died on process restart, no result. Lower-value loss: bible §3.0 already
  shows decode-kernel micro-opt is tapped (simdgroup-matrix A7 dead at M=1). Re-run only if attacking the 1.6× gap becomes priority.
- ✅ **Bench `bs76ri0np` (RWKV-7 moat) DONE** — 118.6/110.6/119.4 tps = FLAT (recorded in findings + test_matrix).
- **RECOVERY (this process):** tree verified clean — the only working-tree code change is the pre-existing 114-line
  `mod bsize_verify_diag` (`#[cfg(test)]`, non-hot-path). `cargo check` running (`bv7v6knjt`). Pivot → testing lane.
- **Heartbeat `bd1vyu73c`** — watchdog ticker (~28 min).

## NEXT (when the above return)
1. Review the int4-KV-PC patch → run parity (decode==CPU) + long-ctx tps + a real-model PPL spot-check. Gate before any commit.
2. Review the MLX-diff design → if a structural delta is flag-gateable + low-risk, prototype behind a flag + GB/s A/B.
3. Record the RWKV-7 flatness in `test_matrix.md` + `findings_summary.md`.
4. Run `cargo test` (parity/regression) → record in `test_matrix.md`; fix in-scope failures.
5. Build `tools/bench/ratios.sh` (reusable warm-bench harness). Document all env flags.
6. Continue down `roadmap.md` (f16-activations, GQA-coalesced MHA, density consolidation).

## RAILS (unattended)
- No destructive git; no revert of user changes; no large deletions without parity+tests. Commit code only behind a passed
  parity/quality gate (or leave as a reviewed patch). GPU jobs SEQUENTIAL (no concurrent model loads → no OOM).

## Phase 4 — production-hardening continuation (current Codex goal)
- Added **`tools/ci/overnight_hardening.sh`**: non-destructive timestamped runner for unattended validation. It records
  git state, cargo check, lib tests, representative parity, local preflight, adversarial quality, and sequential SSM/Qwen
  benches into `reports/overnight/<timestamp>/`. Syntax validated with `bash -n`; orchestration smoke passed with
  `RUN_CARGO_CHECK=0 RUN_LIB_TESTS=0 RUN_PARITY=0 RUN_PREFLIGHT=0 RUN_GPU=0 OUT=reports/overnight/smoke-codex`.
  Full run not launched yet because an existing mamba2 GPU bench is active.
- Added **`docs/campaign/claude_goal_prompt.md`**: copy/paste goal prompt for restarting Claude/Sonnet on the campaign.
  It encodes the rails, priority order, model-quality guidance, command ladder, and stop conditions so the run can continue
  without relying on chat memory.
- Hardened **`tools/ci/overnight_hardening.sh`** after the first smoke:
  - SSM benches now emit explicit `short` / `mid` / `long` rows instead of labeling a ~2k prompt as short.
  - Bench steps now fail the run if no `dec_tps` can be parsed, rather than silently recording `ERR`.
  - Re-validated with `bash -n`, `git diff --check`, and a no-compile/no-GPU smoke:
    `RUN_CARGO_CHECK=0 RUN_LIB_TESTS=0 RUN_PARITY=0 RUN_PREFLIGHT=0 RUN_GPU=0 OUT=reports/overnight/smoke-codex-runner-hardened-2 tools/ci/overnight_hardening.sh`.
- Added **`docs/campaign/change_manifest.md`** to classify the dirty tree into campaign-infra, test-diagnostic,
  and training/KD lanes. Use it before staging so unrelated agent/user work does not get mixed into one commit.
- Added **`tools/ci/README.md`** so future agents can use `preflight.sh` and `overnight_hardening.sh` from docs
  without reading shell internals.
- Added **`docs/campaign/ssm_productionization.md`**: concrete ship criteria for turning the validated RWKV/SSM
  long-context speed moat into a model-selection/serving path with quality gates and fallback rules.
- Ran CPU-only hardening pass while the mamba2 GPU bench was active:
  `RUN_GPU=0 RUN_PREFLIGHT=0 tools/ci/overnight_hardening.sh`.
  Report: `reports/overnight/20260622T020448Z/`. Result PASS:
  `cargo check` ✅, `cargo test -p hawking-core --lib` ✅ (182 passed, 0 failed, 1 ignored, 524.85s),
  `mha_decode_perchannel_int4kv_parity` ✅, `q6k_swiglu_2r_parity` ✅, `q6k_swiglu_4r_parity` ✅.
- Corrected artifact drift:
  - `roadmap.md`: SSM productionization is now the top validated frontier; int4-KV-PC is ready-to-wire, not agent-active;
    MLX-diff is deferred research.
  - `test_matrix.md` / `docs/env_flags.md`: `--profile fast` consistently recorded as ~+3–7%, noisy, mild quality trade.
  - infrastructure checklist now reflects actual built artifacts (`ratios.sh`, `preflight.sh`, `overnight_hardening.sh`,
    adversarial quality suite, env-flag reference).
- Current running work observed by process table: a **mamba2-370M SSM corroboration bench** is active (`hawking generate`
  against `models/mamba2-370m-Q4_K_M.gguf`). Do not start another GPU/model job until it exits or `GPU_WAIT=1` runner
  can wait for it.
- Added **`tools/ci/ssm_serve_smoke.sh`** and wired it into `tools/ci/overnight_hardening.sh` as the default RWKV
  serve-path gate after the SSM bench matrix. Syntax validated with `bash -n`; no-GPU runner smoke passed at
  `reports/overnight/smoke-codex-ssm-serve-gate/`.
- The earlier Claude-launched mamba2 corroboration process is no longer visible in `ps`, but no saved stdout/log was found
  under `/tmp`, `reports`, or the app terminal. Treat its final numbers as **unverified** unless recovered from Claude output.
- Ran focused RWKV serve smoke:
  `tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf`.
  Report: `reports/serve-smoke/20260622T022233Z/`. Result **FAIL/ATTENTION**:
  `/healthz` ✅, `/v1/models` ✅, `/metrics` ✅, but native `/v1/hawking/generate` timed out after 180s. Metrics after the
  request show `hawking_queued_requests 1`, `hawking_requests_admitted_total 0`, `hawking_tokens_generated_total 0`.
  Interpretation: RWKV/SSM model loading through `hawking serve` works, but the server request admission/decode path is not
  production-ready for this SSM model yet. This is now the top SSM productionization bug.

## Phase 5 — SSM serve admission gap: ROOT-CAUSED + FIXED (patch applied, verifying)
- **ROOT CAUSE (confirmed by code read):** RWKV (`impl Engine for RwkvSeven`, rwkv7.rs:816) did NOT override
  `encode_prompt_for_batch` → it used the Engine-trait DEFAULT (engine.rs:310) which returns `Err(Unimplemented)`. The HTTP
  admit path (`http.rs:429`) does `driver.admit(...).ok().flatten()`, which **silently maps that `Err` to `None`** — identical
  to "no free slot" — so the request is pushed to `wait_queue` and never admitted (admitted=0, queued=1, 180s hang). The trait
  comment claims such engines "fall back to generate", but that fallback was never implemented.
- **Why it's a small fix:** RWKV already had everything else — `prefill_slot` + `forward_multiseq_batched` (per-slot GPU arena
  state, isolated via `reset_slot`); the default `forward_multiseq_greedy_tokens` works via that. Only the 3 tokenizer-wrapper
  trait methods were missing.
- **FIX (rwkv7.rs, applied, cargo check ✅):** added `encode_prompt_for_batch`, `decode_token_for_batch`, `eos_id_for_batch` to
  `impl Engine for RwkvSeven`, mirroring `qwen_dense` and guarding the `Option<Tokenizer>` as `generate()` does. Additive,
  RWKV-only, not hot-path, not output-changing (RWKV serve produced nothing before → cannot regress).
- **VERIFICATION ✅ (`reports/serve-smoke/20260622T023814Z/`):** admission gap FIXED — `requests_admitted 1` (was 0),
  `queued 0`, **NO 180s hang** (request returns in ~7s, decode-stepped). **The goal's top task (admission gap) is DONE.**
- **NEW downstream bug (SEPARATE from admission, NOT hacked):** the RWKV serve path now produces only **1 token with empty
  text** (`{"text":"","tok_index":0}` = immediate EOS) vs single-stream `generate`'s coherent 119-tps output. `prefill_slot`
  reads correctly (zeroes slot → CPU `forward_token_core` over the prompt → `copy_cpu_state_to_gpu_slot` → returns first
  prediction), so the fault is downstream: a prefill→GPU-slot recurrent-state handoff, an EOS/chat-template mismatch on the
  native `/v1/hawking/generate` path, or the multiseq GPU decode (`forward_token_gpu_multiseq`) producing a degenerate first
  token. **Higher-care (hot-path RWKV/Metal state) → CHARACTERIZED + flagged for review, not auto-patched.** Secondary: the SSE
  also omits `[DONE]`/stats on early-stop (serve streaming-format follow-up).
- **DOWNSTREAM ROOT CAUSE PINNED (diagnostic, NOT fixed — higher-care):** RWKV `generate()` runs prefill+decode **entirely on
  the GPU single-stream path** (`forward_token_routed`→`forward_token_gpu`; "GPU used end-to-end", "prefill is the same per-step
  kernel as decode"). The serve `prefill_slot` instead prefills on **CPU** (`forward_token_core` into a temp `RwkvState`) then
  **copies CPU→GPU multiseq slot** (`copy_cpu_state_to_gpu_slot`, rwkv7.rs:1235) before GPU multiseq decode. **That CPU→GPU
  recurrent-state handoff is the fault** — the GPU slot decodes from a corrupt/mismatched state → degenerate first token. RWKV's
  prefill→multiseq handoff has NO active test (the only `prefill_slot_into_multiseq_parity` is Qwen + `#[ignore]`; the qwen
  multiseq decode/mha/churn/kv-scatter parity tests PASS, so the kernel infra is fine — it's the RWKV CPU→GPU state copy).
  **Recommended fix (higher-care review): correct `copy_cpu_state_to_gpu_slot`'s CPU-`RwkvState`→GPU-arena layout (or give RWKV a
  per-slot GPU prefill that preserves cross-slot isolation), gated by a NEW RWKV `prefill_slot_into_multiseq` parity test
  (mirror the Qwen one) that asserts multiseq-after-prefill == single-stream `generate`.** Left for review, not auto-patched.
- **FOLLOW-UP (flagged, not done here):** `http.rs:429`'s `.ok().flatten()` still silently swallows `Unimplemented` for ANY
  future non-batch engine → distinguish `Err(Unimplemented)` (→ 501 or generate-fallback) from `Ok(None)` (→ queue) so the
  server never hangs on an unsupported engine. Higher-care (touches the request handler); left for review.
- **Moonshot handoff added:** `docs/campaign/claude_moonshot_goal_prompt.md` gives Claude a longer SSM production campaign:
  harvest the active overnight run, fix RWKV serve correctness with a parity gate, harden HTTP admission errors, build an SSM
  product gate, add long-context quality evaluation, document model-selection policy, and keep commit lanes clean.

## Phase 6 — SSM productionization moonshot (multi-lane, in progress)
- **Lane 0 (harvest) DONE:** overnight `20260622T025008Z` completed. RWKV-7-SFT flat moat CONFIRMED clean (warm 3-trial):
  ~114.6 / ~113.9 / ~110.5 tps (short/mid/long). mamba2_long = **0.00/0.00/0.00** (the 8k kernel bug, now in the matrix).
  `preflight_fast` FAILED on `cargo fmt --check` = **repo-wide PRE-EXISTING drift** (~190 locations across committed files:
  json_constrain, speculate/*, olmoe, mamba2, qwen_dense, dozens of tests, src/main.rs, tools — almost none touched by SSM work).
  NOT fixed (global `cargo fmt` would reformat unrelated lanes; rails forbid). The repo's real CI bar is clippy+build, not fmt.
- **Lane 1 (serve correctness) — parity gate WRITTEN + REPRODUCES THE BUG ✅:** `rwkv7_prefill_slot_multiseq_parity` FAILS as
  intended: `solo=[37138,47,11]` vs `multi=[37138,45,21265]`. **multi[0]==solo[0]=37138 (first token CORRECT, not EOS)** →
  `prefill_slot` returns the right prediction (from CPU `last_logits`); divergence is at **token 2** (first decode off the slot
  state) → `copy_cpu_state_to_gpu_slot` leaves a WRONG recurrent state in the GPU slot. NARROWED (kernel-confirmed, my first guess CORRECTED):
  the multiseq WKV kernel (`rwkv7.metal:411-435`) is "byte-for-byte the single-stream kernel with a stream index added" and the
  wkv plane is `[stream][layer][head][hs][hs]` (same `so=head*hs*hs` as single-stream) → the element ORDER MATCHES (so my earlier
  "element-order mismatch" was WRONG). Thus the multiseq decode is bit-exact-single-stream FOR A GIVEN STATE; the real fault is
  that the **COPIED CPU state ≠ the GPU single-stream state**. Candidates (focused next step, gated): (a) CPU-prefill
  `forward_token_core` vs GPU-prefill `forward_token_gpu` diverge beyond the near-tie f32 tolerance and compound; (b) an INCOMPLETE
  per-slot copy (some GPU per-slot state beyond wkv/att_shift/ffn_shift not covered/zeroed); (c) a token-shift representation
  convention difference. **Decisive diagnostic (gated): prefill a slot via CPU-copy vs via a GPU pass and diff the slot state
  element-wise; the cleanest fix may be to build `prefill_slot` on the GPU path (single-slot) rather than CPU+copy.** Higher-care
  Metal; reproduced + isolated + gated. (Serve "empty token" is a superset symptom; this state-handoff is the root.)
- **Lane 2 (HTTP admission robustness) — DONE + COMPILES:** `http.rs` now matches `driver.admit` → `Err` returns a clear SSE
  error (`code: admit_unsupported`) instead of silently queueing forever. `cargo check -p hawking-serve` ✅. Qwen path unchanged.
- **Lane 3 (SSM product gate) — WRITTEN:** `tools/ci/ssm_product_gate.sh` (serve smoke + speed matrix + quality probes +
  isolation → one report under `reports/ssm-product/`). `bash -n` ✅.
- **Lane 4 (quality suite) — WRITTEN + ran → KEY HARNESS FINDING:** `tools/ci/ssm_quality_suite.sh` (fixed an `rg`→`grep`
  portability bug — `rg` isn't in the non-interactive PATH, so the first run silently scored 0/0). The valid run then exposed that
  **`hawking generate` does RAW COMPLETION — it does NOT apply a chat template** (no such CLI flag). Instruct/SFT Q&A prompts
  produce garbage (Qwen direct: "Brainly\nMath…", "multiply 11 by 222222222"); earlier "Write X…" benches only *looked* coherent
  because raw-completing an imperative is plausible. → a generate-based suite is NOT a valid instruct quality gate; its raw run
  (SSM 3/5, Qwen 0/5) is UNTRUSTWORTHY. **Valid quality eval needs the chat template — via the serve `/v1/chat/completions`
  endpoint (which applies it) → GATED ON LANE 1 (RWKV serve fix), or via manual per-model-templated prompts.** Ties Lane 4 to Lane 1.
- **Lane 5 (model selection) — WRITTEN:** `docs/campaign/ssm_model_selection.md` (operator decision table: ctx×priority×class →
  Qwen/RWKV/hybrid; honest caveats: serve not yet correct, RWKV-0.4B is small, mamba2 not viable).
- Code additions are fmt-clean (the 2 RWKV trait methods rewrapped to rustfmt); pre-existing repo-wide drift left untouched.
- **Final hardening/condensation prompt added:** `docs/campaign/claude_final_hardening_condensation_goal_prompt.md`
  is the recommended last autonomous goal before stopping the loop. It asks Claude to produce a project-standing packet,
  pruning inventory, commit plan, summary seed for a new chat, and final recommendation while avoiding destructive cleanup.

## Phase 7 — final hardening/condensation pass (project-standing packet)
- Produced the **project-standing packet** (the deliverable): `docs/campaign/project_standing_snapshot.md` (1-page state +
  speed table + ready/experimental/blocked/dead + final recommendation), `open_risks_and_gates.md` (R1–R8 + evidence-to-close
  + best next gate), `pruning_inventory.md` (doc-condensation candidates; **no code deletion** — codebase is curated/entangled),
  `commit_plan.md` (5 lanes; **NOT executed** — attended decision), `summary_seed_for_new_chat.md` (paste-able standing seed).
- Stretch A checks: all `tools/ci/*.sh` + `tools/bench/ratios.sh` `bash -n` ✅; `git diff --check` clean; GPU idle.
  `preflight` fmt failure = pre-existing repo-wide drift (R6; NOT fixed — rails forbid global `cargo fmt`).
- Stretch C: marked `change_manifest.md` **SUPERSEDED** by `commit_plan.md` (no deletions).
- **Verdict:** the campaign is SUMMARIZED + ready for a fresh project-standing review. The single highest-leverage next fix
  is RWKV serve decode correctness (R1, gated by `rwkv7_prefill_slot_multiseq_parity`) — it converts the measured SSM moat
  into a shippable long-context serve product and unblocks valid instruct quality eval. Not committed (tree mixes lanes).

### Next safe commands
```bash
# First reproduce/characterize the SSM serve admission gap:
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf

# Full unattended validation is available after/around that investigation.
# To skip the known-failing serve gate while collecting the bench matrix:
RUN_SERVE_SMOKE=0 tools/ci/overnight_hardening.sh
```

## Phase 8 — RWKV serve DECODE FIXED + end-to-end coherent (committable checkpoint)
- **R1 CLOSED.** Diagnostic workflow (`w2fef3c7w`, 4 probes) pinned the root cause: the bundle-wide `RwkvGpu.fresh` flag is
  STALE after `prefill_slot` copies the recurrent state into the GPU slot → the first multiseq token-shift runs `sx=-att_in`,
  DISCARDING the copied `att_shift`/`ffn_shift` → divergence at token 2. (NOT a layout/stride bug — those were proven identical.)
- **Fix (tiny, gated):** clear `g.fresh=false` in `prefill_slot` after the copy (`rwkv7.rs:1222`). Parity gate
  `rwkv7_prefill_slot_multiseq_parity` is now **GREEN** (both slots: solo==multi==[37138,47,11]).
- **Serve stream-terminator fix:** the native + OpenAI SSE forwarders emitted `{stats}`/`[DONE]` only on the EOS signal, so a
  max_tokens-bounded request ended without a terminator. Moved the terminator to fire on ANY stream end (`http.rs`).
- **End-to-end PROVEN:** `ssm_serve_smoke.sh` (continuation prompt) is **fail=0** — 16 coherent tokens (" Paris. The capital of
  the United Kingdom is…") + `{stats}` + `[DONE]`; admitted=1, tokens_generated=16, queued=0.
- The earlier "1 empty token" on the "Summarize…" prompt = RWKV raw-completion predicting eos (the chat-template gap, R3) — NOT a bug.
- **Serve perf NOTE (follow-up, not correctness):** serve dec_tps ~7.8 vs single-stream ~119 — the B=8 multiseq arena does
  8-stream work for 1 active stream. Size the arena to active slots to recover it. Separate from this checkpoint.
- **Checkpoint:** serve admission + R1 decode + stream terminator all FIXED + TESTED (parity green, serve smoke fail=0). Committing the clean lanes.

## Phase 9 — regression gates wired (Lane A.5: make the speed/compression/quality wins persist)
- **Why this lane:** master-plan §7 ranks "no perf/compression/quality regression gates" as the #2 critical-path risk and
  the owner's explicit "ensure our wins persist" ask. Correctness was already locked (193 golden hashes); the missing half
  was an enforced floor. Per the prompt's own algorithm ("add the gate before changing risky behavior") this is the
  prerequisite for the next lane (serve-throughput hot-path change, R1b).
- **Built (narrow, reversible, no product behavior changed):**
  - `tools/ci/baselines/regression_baseline.json` — committed floors. Footprints are MEASURED exact bytes (Qwen-3B
    1,929,903,264; RWKV-7-SFT 300,694,944). tps/quality floors sit ~10–15% below the measured warm median so the gate
    catches CATEGORY regressions (a lever silently disabled) without flapping on the documented ±several-% noise floor.
  - `tools/ci/regression_gate.sh` — measures (reusing `tools/bench/ratios.sh` + `stat` + `jq`), compares to baseline, and
    EXITS NON-ZERO on a breach. Footprint check is CPU-safe/always-on; tps + quality run when the GPU/binary/model are
    available (else reported SKIPPED, never a false PASS). Honestly lists `pending_not_enforced` (serve-tps, int4-KV
    perplexity, instruct-quality).
  - Wired into `tools/ci/preflight.sh` (footprint always; full gate in the bench block) and `tools/ci/overnight_hardening.sh`
    (`RUN_REGRESSION=1` default; a breach sets the run's FAIL=1).
- **Evidence (MEASURED 2026-06-22):**
  - Full gate GREEN end-to-end — `reports/regression/20260622T140213Z/summary.md`: footprint 2/2 PASS; `tps qwen3b_short=39.42
    ≥34.0`; `tps rwkv7_sft_short=119.42 ≥100.0`; `quality f16_kv=100% ≥0.85`; `quality profile_fast=90% ≥0.80`. **6 enforced,
    fail=0, exit 0.** Numbers reproduce the baseline's cited sources (test_matrix), so the baseline is honest.
  - Red path PROVEN — a deliberately-too-tight ceiling makes the gate exit 1 (footprint FAIL). A gate that cannot go red is useless.
  - Wiring PROVEN — `bash -n` clean on both scripts; CPU-only overnight smoke ran the regression phase and nested its report
    under the run's OUT (`reports/overnight/smoke-regression-wire/regression/summary.md`), rc=0.
- **Lane hygiene:** touches only `tools/ci/` (2 edits + 2 new files); no source/kernel/output change; `reports/` is gitignored.
- **Exact commands:**
  ```bash
  FOOTPRINT_ONLY=1 tools/ci/regression_gate.sh      # CPU-safe, always enforceable
  tools/ci/regression_gate.sh                        # full (needs free GPU + release binary + models)
  RUN_REGRESSION=1 tools/ci/overnight_hardening.sh   # enforced inside the unattended runner
  ```
- **NEXT:** with the floors in place it is now safe to take the serve-throughput hot-path change (R1b) — size the RWKV/Qwen
  multiseq decode arena to ACTIVE slots, keep `rwkv7_prefill_slot_multiseq_parity` green, and add a `serve_decode_tps` floor
  to the baseline to lock the recovered throughput.

## Phase 10 — Condense frontier added (post-finalization Model Press expansion)
- **Why this lane:** the owner asked to carry the condensation mentality beyond runtime finalization: make Hawking
  able to quantize/condense models that ordinary local machines cannot fully load, with GLM-5.2-class open weights as the
  current market signal.
- **Built (docs/strategy only; no code behavior changed):**
  - `docs/plans/condense_frontier_2026_06_22.md` — new post-finalization frontier plan: memory-budgeted
    press planner, out-of-core resumable tensor pipeline, output-damage-ranked bit allocation, 4/3/2/1-bit ladder,
    condense-then-recover loop, GLM-class MoE dry-run support, and artifact proof requirements.
  - `docs/plans/condense_naming_migration_2026_06_22.md` — public rename plan: Condense replaces STRAND in product
    language; current `strand` function symbols and aliases are inventoried.
  - `docs/campaign/hawking_ship_finalization_prompt.md` — P2 renamed/framed as Model Press / Condense; added
    condensation doctrine and Lane G.
  - `docs/campaign/hawking_ship_goal_prompts.md` — added a copy-paste Lane G prompt and strengthened Lane B/C wording.
  - `docs/plans/hawking_shippability_masterplan_2026_06_22.md` — added §3.8-§3.13 and §3.A; sequenced Phase E after core
    finalization.
  - Standing packet updated: `project_standing_snapshot.md`, `open_risks_and_gates.md`, `summary_seed_for_new_chat.md`,
    `roadmap.md`, and `commit_plan.md`.
- **Evidence checked:** external GLM-5.2 state was verified on 2026-06-22 from Z.ai/Hugging Face/Together/arXiv sources:
  public weights/MIT, 1M context, GLM-5 backbone around 744B total / 40B active MoE. Treat GLM-5.2 as the current proof of
  demand, not as the only target.
- **Gate/rails:** no frontier downloads, paid cloud, or public derivatives without owner approval. Start with dry-run planner
  and small-model out-of-core proofs before GLM-class work.

## Phase 10 — R1b serve throughput: hypothesis tested + DISPROVEN (evidence-driven pivot)
- **Lane:** with the regression floors in place (Phase 9), took the top runtime gap R1b (serve tps ≪ single-stream).
- **Hypothesis (from the standing docs):** the B=8 multiseq decode arena does 8-stream work for 1 active stream → size it to
  active slots to recover throughput.
- **Built + parity-proven, then REVERTED:** a surgical, parity-safe change bounding decode dispatch to `active=max(region)+1`
  while keeping the arena `SERVE_BATCH`-wide (absolute slot state preserved; strides are config-derived). 3 edits in `rwkv7.rs`
  (`forward_token_gpu_multiseq` gained an `active` bound; the serve path computed it; the compacted method path passed
  `tokens.len()` unchanged). Gates GREEN: `rwkv7_prefill_slot_multiseq_parity` (slot0 active=1, slot3 active=4) + all 11
  `rwkv7_multiseq_parity` (b1/b2 isolation, CPU-oracle, serial-GPU).
- **MEASURED (fresh binary, 96-token continuation via `/v1/hawking/generate`):** serve dec_tps **9.94** — essentially
  unchanged. With 1 active stream at slot 0 → `active=1` → 1-stream dispatch, yet still ~10 tps vs `generate` ~119 for the same
  0.4B model. **Conclusion: stream width is NOT the bottleneck; the cost is per-token FIXED overhead independent of `b`.**
  (First measurement also caught a process trap: building `-p hawking-core` does NOT rebuild the `hawking` serve binary — must
  `cargo build --release -p hawking`; the parity TESTS recompile the lib so they were always valid.)
- **Action:** reverted the change (rwkv7.rs == HEAD; verified empty diff; parity re-confirmed GREEN). No measured benefit →
  no unmeasured hot-path delta left in the tree (rails). The deliverable is the corrected diagnosis.
- **Re-aimed R1b (open_risks_and_gates.md R1b):** the multiseq decode path is ~12× slower per token than single-stream even at
  b=1. Next diagnostic: profile ONE serve token vs ONE generate token (dispatch count + `commit_and_wait` + serve-loop CPU),
  and diff the multiseq layer GEMVs vs the single-stream predec/fused Q4K GEMVs — fix the dominant term, then (only if
  multi-stream matters) re-land active-sizing WITH a measured aggregate gain.
- **Lane hygiene:** engine source unchanged from HEAD; the only tracked deltas this session are the Phase-9 regression-gate
  tooling (`tools/ci/`) + campaign docs.

## Phase 11 — R3 valid quality eval: chat path works, harness bug (no quality recorded)
- **Built** `tools/ci/ssm_quality_chat.sh` — a valid instruct eval driving serve `/v1/chat/completions` (vs the raw-`generate`
  suite that is explicitly NOT valid), comparing RWKV-7 vs Qwen-3B across 5 task classes (bash 3.2-safe).
- **KEY FINDING:** RWKV-7's chat endpoint ALREADY produces coherent instruct output under the generic template — manual
  `/v1/chat/completions` returned "The capital of France is Paris." So R3 was less blocked than assumed; no RWKV chat-template
  fix is needed for a valid eval (RWKV falls to `render_chat_generic` and still answers).
- **HARNESS BUG (honest):** the suite returned all-empty (0/5) for BOTH models — a measurement artifact, NOT a quality result
  (a manual curl with the same body returns valid JSON with non-empty content). Root cause not fully pinned (likely the
  script's curl/parse or the long retrieval prompt's CPU-prefill timeout on RWKV). **No quality numbers recorded.** Also spotted
  a separate concern: a raw Qwen-3B chat response showed garbage + apparent template leakage ("caffein\nuser\n…multiply 1 by 2").
  Both flagged for a separate session (`spawn_task` task_e50ba5ae). R3 remains open pending the harness fix.

## Phase 12 — Lane G / Condense: C1 memory-budgeted Press Planner (dry-run) LANDED
- **Gate-safety check (required before Lane G):** core finalization gates are safe to leave — R1 serve-correctness CLOSED,
  R9 regression gates green+wired, parity green, engine source == HEAD (R1b probe cleanly reverted), lib 182/182. Proceeded.
- **Built** `hawking press --dry-run --memory-budget <SIZE> --target <BITS> --weights <gguf>` (`crates/hawking/src/main.rs`
  `press_main` + `parse_size_arg`/`parse_tier_arg`/`fmt_bytes_h`/`fmt_count_h`). GGUF-metadata-only: no weights resident, no
  GPU, no network, no writes. Prints a truthful **Press Plan**: arch, params/tensors, weight bytes + bpw, largest tensor, peak
  CREATION memory (out-of-core tensor-at-a-time vs full-resident-f32), the Condense ladder (4/3/2/1-bit) output sizes, and a
  budget verdict naming the **wedge**.
- **MEASURED (truthful):** Qwen-3B-Q4_K_M — 3.09B params, largest `token_embd` 311.2M elems (f32 1.16 GiB), out-of-core peak
  1.32 GiB vs full-resident 11.50 GiB (~9×); at `--memory-budget 2gb` → out-of-core FITS, full-resident EXCEEDS → **WEDGE**
  (the "press a parent that can't fit fully resident" product claim, demonstrated). RWKV-7 — 450.9M, 292 MiB vs 1.68 GiB.
  Weight-bytes vs file `stat` differ only by the ~6 MB GGUF header (excluded). Non-`--dry-run` prints a gated notice (bake is
  owner-gated, not implemented).
- **safetensors (fp16/bf16 PARENT) support added same session (✅):** `read_inventory` now auto-detects GGUF vs safetensors
  (8-byte LE header + JSON, metadata only). MEASURED on local HF parents (no download): `rwkv7-g1-04-hf` (BF16×795, 450.8M,
  ~16 bpw) → out-of-core 292 MiB vs full-resident 1.68 GiB, WEDGE at `--memory-budget 1gb`, ladder 3.56×/5.33×/8.00×;
  `mamba2-370m-hf` (F16×434) also planned; weight-bytes vs file `stat` differ by exactly the 84,584-byte safetensors header.
- **Gates:** `cargo test -p hawking --bins press_tests` GREEN (3: parse_size_arg, parse_tier_arg, safetensors_header); build
  clean; the planner code is clippy-clean (PRE-EXISTING `hawking-core/json_constrain.rs` dead-code is unrelated — flagged in R8b).
- **Remaining (C1):** multi-shard safetensors (`*-00001-of-000NN.safetensors` + index.json — frontier parents are sharded;
  reader handles a single file today); scratch/thread/resume fields; then C2 out-of-core writer. The bake stays owner-gated.
  Docs: `condense_frontier_2026_06_22.md` C1 STATUS, `open_risks_and_gates.md` R8b.
- **Lane hygiene:** Lane G change is confined to `crates/hawking/src/main.rs` (new `press` subcommand + GGUF/safetensors
  inventory readers + 3 test fns) + docs; no engine/kernel/output change; read-only dry-run with no side effects.

## Phase 13 — Lane H / Apple Fit: A1 Mac profiler + A2 `hawking fit` (capability-first) LANDED
- **Gate-safety check:** prior lanes safe to leave (R9 regression gates green, Condense C1 done+gated, R1 closed, engine ==
  HEAD). Proceeded to Lane H.
- **A1 (Mac profiler):** `detect_mac()` reads chip / unified memory / OS via `sysctl` (hw.memsize, machdep.cpu.brand_string,
  kern.osproductversion). MEASURED here: "Apple M3 Pro | 18.00 GiB | macOS 26.6". Wired into `hawking doctor`: the swap-risk
  was **hardcoded to an 18 GB M3 Pro** (`m3_pro_18gb_swap_risk`) → now machine-relative (`fit_zone` vs detected RAM) + a
  `machine:` line. REMAINING: `doctor --json`, live pressure/thermal (A4), Metal limits.
- **A2 (`hawking fit`):** `hawking fit --weights <gguf> [--intent <I>] [--max-context N] [--concurrency C]` (`fit_main`).
  CPU-only (metadata + sysctl; no weights/GPU/network). Prints the **context × KV-policy fit envelope** (FITS/TIGHT/SWAP/OOM
  vs detected RAM), envelope ceilings (longest stable / f32-quality / safe-comfort / native), and an **intent recommendation**
  (capability-first default) with explicit alternatives + override hints. SSM models get the **flat-KV moat row**.
- **Capability doctrine honored:** reports the MAX envelope, never caps; safety-biased intents (safe-fit/max-battery) print an
  explicit **anti-throttle note** naming the stronger max-capability option. (The enforcing anti-throttle GATE — A8 — comes
  with `serve --auto` A3, which isn't built yet; `fit` is informational/read-only so there's no run to throttle.)
- **MEASURED:** Qwen2.5-3B on M3-Pro-18GB → 4k–32k all FITS (f16 KV 144 MiB→1.12 GiB; f32 up to 2.25 GiB), max-capability =
  native 32k @ f32; longest stable 262k @ f16. RWKV-7 → ~887 MiB at ANY context (flat). `doctor` swap_risk now machine-aware.
- **Gates:** `cargo test -p hawking --bins` GREEN — `fit_tests` (kv_cache_bytes scaling + precision/concurrency, fit_zone
  thresholds) + `press_tests` (3); build clean; clippy clean for this code (fixed a `useless_format`; the PRE-EXISTING
  `hawking-core/json_constrain.rs` dead-code remains unrelated).
- **Remaining (Lane H):** A3 `serve --auto` (intent policy over the planner) + A8 anti-throttle regression gate; A4 memory-
  pressure engine; A5 long-context routing (transformer vs SSM, measured); A6 energy/thermal cards; A7 Mac-native model
  experience (pull/registry/hawkingd). Docs: `apple_fit_frontier_2026_06_22.md` A1/A2 STATUS.
- **Lane hygiene:** Lane H change confined to `crates/hawking/src/main.rs` (new `fit` subcommand + `detect_mac`/`kv_cache_bytes`/
  `fit_zone` helpers + 2 test fns + the doctor real-RAM wire) + docs; no engine/kernel/output change; read-only.

## Phase 14 — Lane H continued: doctor --json (A1) + serve --auto (A3) + anti-throttle invariant (A8)
- **Required-docs note:** read `docs/env_flags.md` + `docs/architecture.md` in full this turn (the prior turn had not). Key:
  hawking-serve honors the same `HAWKING_*` flags as the CLI; KV policy = `HAWKING_QWEN_F16_KV` (f32 default), energy =
  `HAWKING_ENERGY_EFFICIENT`/`--energy-mode`, speed = `--profile fast` — so `serve --auto` chooses among these.
- **A1 finish — `doctor --json`:** added `--json` to `doctor`; emits machine{chip,total_unified_bytes,os} + model{layers,
  kv_heads,head_dim,context,…} + kv_cache_estimate + swap_risk for repeatable fit decisions. Validated on Qwen-3B.
- **A3 — `hawking serve --auto [--intent]`:** `auto_serve_pick` + Serve dispatch. On `--auto` it detects the Mac, reads model
  facts (`read_model_facts`), picks the strongest stable config for the intent, ANNOUNCES machine + chosen (ctx/KV/profile/
  energy) + anti-throttle verdict, and APPLIES the safe levers (KV policy, fast profile for max-speed, efficient energy for
  max-battery) ONLY where the user didn't set them (expert flags win). MEASURED e2e: Qwen `max-capability` → ctx 32768 @ f32
  ("anti-throttle OK"); `safe-fit` → ctx 32768 @ f16 + "EXPLICIT DOWNGRADE: max-capability would serve … @ f32 KV"; RWKV-7 →
  flat SSM full native. (Fixed a logic bug found in validation: safe-fit/max-battery now cap context at native — "comfortable"
  context can exceed native on a roomy Mac, which is not safer — and the downgrade note names the real giveback, f16-vs-f32.)
- **A8 — anti-throttle invariant (pick level):** baked into `auto_serve_pick` — a non-safety intent never returns below
  max-capability without `safety_downgrade` set; a hard-RAM reduction is the capability ceiling (not a flagged downgrade).
  Unit-tested (`fit_tests::auto_pick_is_capability_first_and_anti_throttle`): roomy → native@f32 no-downgrade; safety intents
  flag + cap ≤ native; tight-RAM → f16 forced (no false flag); SSM flat. The MEASURED regression gate (auto vs best-manual
  serve) needs A6 + a serving harness — deferred.
- **Gates:** `cargo test -p hawking --bins` **15/15 GREEN**; build + clippy clean (fixed a `useless_format`); serve --auto
  validated e2e across 3 intents/2 models. PRE-EXISTING `hawking-core/json_constrain.rs` dead-code still unrelated.
- **Remaining (Lane H):** A4 live memory-pressure engine; A5 measured long-context routing; A6 energy/thermal cards; A7
  Mac-native model experience (pull/registry/hawkingd); the measured A8 serving regression gate; serve context-cap wiring.
- **Lane hygiene:** confined to `crates/hawking/src/main.rs` (doctor --json, fit/auto helpers, serve --auto wiring, 1 test fn)
  + docs; `--auto` is opt-in so default serve is unchanged; engine/kernels untouched.

## Phase 14 — finish gates + clean the SOTA rebench (goal: "finih all gates and lanes for a cleaner rebench")
- **GATE: workspace clippy GREEN.** The one blocker was a pre-existing dead variant `ValidFirstBytes::None`
  (`hawking-core/json_constrain.rs`) — the state machine never produces it. Removed the variant + its dead match arm →
  `cargo clippy --workspace -- -D warnings <CI allowlist>` is now clean (commit 55dac35). Non-behavioral.
- **BENCH CLEANED (cleaner rebench):** diagnosed from the owner's runs (`reports/sota-compare/20260622T204811Z`):
  - Hawking showed **NA** in §5 speed AND §7 long-ctx (all rows) — root cause: `hawking generate` prints `dec_tps=` to
    **stderr** and `hawking_tps()` discarded it (`2>/dev/null`). Fix: capture `2>&1` (commit 158116e). Verified 20.59 tps.
  - §4 bit-ladder showed **EXCEEDS/EXCEEDS** (no wedge) because the press demo hardcoded `--memory-budget 2gb`, below the 7B
    out-of-core peak (2.5 GiB). Fix: `PRESS_BUDGET` (default 8gb, sits between out-of-core and full-resident) → WEDGE shows
    (out-of-core FITS 2.32 GiB vs full-resident EXCEEDS 28.37 GiB) (commit bd96f64).
  - §11 energy is a deliberate pointer (macmon present) — not a gap.
- **GATES re-verified GREEN:** workspace clippy ✅; `cargo build` ✅; `cargo test -p hawking --bins` ✅ (17, incl. A8
  `serve_auto_tests`); `regression_gate.sh` footprint ✅ (2 enforced); `rwkv7_prefill_slot_multiseq_parity` ✅ (slot0+slot3);
  full lib suite (182) re-running to confirm the json_constrain change.
- **Result:** the rebench now produces complete data — Hawking tps in §5/§7, the wedge in §4, all engines (llama.cpp +
  MLX 7B). Rerun clean (Claude closed): `BIT_TARGETS=8,6,5,4,3,2,1 TRIALS=5 TOK=256 bash tools/bench/compare_sota.sh`.
- **Lane hygiene:** json_constrain (1 dead-variant removal) + bench .sh + this log; engine decode/serve paths untouched.

## Phase 15 — AUDIT of Codex's uncommitted work + fair-comparison push (goal: "make Hawking win the fair comparison")
Audited Codex's uncommitted parallel work BEFORE building (per owner). Findings + actions:
- **RoPE interleaved→NEOX (CPU + all Metal kernels)** — Codex's biggest change. VERDICT: **correct, keep.** Validated: Qwen-7B
  via `/v1/chat` returns correct math/reasoning (391 / 40 / Paris); `cpu_backend_parity` (Metal≡CPU) ✅; lib 182/182 ✅; Q6_K +
  RWKV parity ✅. NEOX is the right convention for Qwen/Llama/Gemma/DeepSeek GGUF. Committed `f2fb65e`.
- **The §8 quality "miss" was a HARNESS bug, not Hawking quality.** Raw `generate` was fed a manual `<|im_start|>` template
  whose special tokens aren't honored → 241/491 (wrong). Via `/v1/chat` (correct special tokens) Hawking gets 391/40/Paris.
  Codex's bench upgrade routes quality through `/v1/chat` + llama-server chat — the fair path. Committed `14193fb`.
- **7B kernel profile (goal: short-decode speed)** — Codex's profile was shader-stale (won't load). Rebaked via autotune at
  the current shader hash (loads; `runtime_profile=fast`). **Honest result: default 20.43 → profile 20.53 tps (+0.5%).**
  autotune picked `metal-default` — the 7B short-decode gap (~0.7× llama 28.8) is **structural Q4_K GEMV efficiency, not
  kernel selection**; a profile cannot close it (needs a GEMV kernel rewrite — multi-week, out of scope). Committed `51a4ea5`.
- **tokenize subcommand** added (inspect special-token handling) — committed `720618a`.
- **OPEN findings (documented, not yet fixed):**
  - **8k transformer drop (goal 3):** Codex's change was RoPE-only — the drop is UNADDRESSED. Hawking Qwen-7B 20.9→11.3 @8k
    vs llama flat 28.7. Lever under test: F16_KV (halves KV read). [measuring]
  - **Llama serve `/v1/chat` returns EMPTY** — per-arch chat-template gap in serve (raw generate is coherent). Fair-quality
    eval currently only valid for Qwen; non-Qwen needs the serve chat template fixed.
  - **`integration_greedy_64` golden gate is RED** — the NEOX change legitimately altered output + the shader hash changed, so
    the 3B golden profile + token hashes are stale. Needs a 3B autotune + golden-token rebake (rebake-after-validated-change,
    not a regression). Live correctness gates (lib 182, cpu_backend_parity, kernel/RWKV parity, /v1/chat) all pass.
- **Honest scorecard (fair, 7B):** quality = WIN/parity (correct via fair path); long-context = WIN (RWKV-7 SSM 119 flat);
  unique caps (fit/press/serve --auto) = WIN; short 7B decode = LOSE ~0.7× (structural GEMV); 8k transformer = LOSE (KV-path).
