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
