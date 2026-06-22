# Hawking — Open Risks & Gates (2026-06-22)

Each risk: what it is, evidence required to close it, and the current best next gate. Ordered by leverage.

## R1 — RWKV serve decode — ✅ CLOSED (2026-06-22)
- **RESOLVED.** Root cause = stale bundle-wide `RwkvGpu.fresh` flag (NOT a layout/stride bug): after `prefill_slot` copies
  the recurrent state into the GPU slot, the first multiseq token-shift ran `sx=-att_in`, discarding the copied
  `att_shift`/`ffn_shift`. **Fix:** clear `g.fresh=false` in `prefill_slot` after the copy (`rwkv7.rs:1222`) + emit the SSE
  `{stats}`/`[DONE]` terminator on any stream end (`http.rs`). **Parity gate GREEN; `ssm_serve_smoke.sh` fail=0** (16 coherent tokens).

## R1b — RWKV serve THROUGHPUT (NEW top gate) 🟡 — hypothesis CORRECTED 2026-06-22
- **Risk:** single-stream serve dec_tps ~10 vs ~119 single-stream `generate` (~12×) for the SAME 0.4B model.
- **❌ DISPROVEN hypothesis (was the assumed cause):** "the B=8 multiseq arena does 8-stream work for 1 active stream; size
  it to active slots." A parity-safe change was built that bounds decode dispatch to `active = max(region)+1` (so 1 stream at
  slot 0 → dispatch 1 stream, not 8), with `rwkv7_prefill_slot_multiseq_parity` GREEN (slot0 active=1, slot3 active=4).
  **Measured result: serve stayed ~9.9 tps (96-token continuation, fresh binary) — NO improvement.** If stream-width were the
  bottleneck, `active=1` would have approached ~119. It did not → **the cost is per-token FIXED overhead, independent of `b`.**
  The change was REVERTED (no measured benefit; rails forbid unmeasured hot-path deltas). rwkv7.rs == HEAD.
- **Real bottleneck (re-aimed):** the multiseq decode path is ~12× slower PER TOKEN than the single-stream `forward_token_gpu`
  path even at b=1. Candidate causes, in order to check: (a) the multiseq layer GEMVs may not use the optimized predec/fused
  Q4K kernels the single-stream path uses (generic per-stream `gemv_batched`/bi-loop dequant GEMV); (b) per-token dispatch
  count / `commit_and_wait` latency in the multiseq glue; (c) serve-loop CPU overhead per token (scheduler, channel sends, SSE,
  detok). NOTE: prefill is CPU (`prefill_slot` runs the prompt through `forward_token_core`) — separate TTFT issue.
- **Evidence to close:** profile ONE serve decode token vs ONE `generate` token — count kernel dispatches + commit_and_wait +
  CPU serve overhead, and diff the multiseq GEMV kernels against the single-stream predec GEMV. Fix the dominant term; re-bench
  serve dec_tps toward single-stream; THEN (if multi-stream matters) re-land active-sizing WITH a measured aggregate-tps gain.
  Keep `rwkv7_prefill_slot_multiseq_parity` green throughout.

<details><summary>(historical R1 detail, pre-fix)</summary>

- **Risk:** `hawking serve` admits an RWKV request but emits one empty token / no final stats; output is incoherent vs
  single-stream `generate`. Root: `prefill_slot` returns the correct first token but the recurrent state it copies into the
  GPU multiseq slot (`copy_cpu_state_to_gpu_slot`) diverges from the single-stream GPU state → decode diverges at token 2.
- **Signature (reproduced):** `solo=[37138,47,11]` vs `multi=[37138,45,21265]`; multi[0]==solo[0] (first token correct).
- **Diagnosis (kernel-confirmed):** the multiseq WKV kernel is byte-for-byte the single-stream kernel + a stream index, and
  the wkv plane is `[stream][layer][head][hs][hs]` — so element ORDER matches; the fault is the COPIED state ≠ GPU
  single-stream state (CPU↔GPU prefill divergence beyond f32 tolerance / incomplete per-slot copy / token-shift convention).
- **Gate (exists, reproduces):** `cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1`
- **Evidence to close:** parity gate GREEN, then `ssm_serve_smoke.sh` shows coherent SSE text + final stats + `[DONE]`,
  `admitted>=1`, `queued=0`, `tokens_generated>1`. **Higher-care Metal** — keep the fix tiny, rerun the gate every change.
- **Best fix:** build `prefill_slot` on the GPU path (single-slot) instead of CPU-`forward_token_core`+copy, OR element-wise
  diff the copied vs GPU-built slot state to find the exact discrepancy.
</details>

## R2 — per-channel int4-KV not wired end-to-end 🟡
- **Risk:** the −75% KV compression lever is built + numerics-validated (cosine ~0.998, parity test passes) but DEAD-CALLED.
- **Evidence to close:** wire behind `HAWKING_QWEN_INT4_KV_PC` (5-step spec in `roadmap.md`), then (a) parity (decode==CPU),
  (b) long-ctx coherence @8k, (c) real-model perplexity vs f32 KV. **Strategically lower urgency** — the SSM path beats it
  for long-ctx (no KV at all).

## R3 — no valid instruct quality gate 🟡
- **Risk:** `hawking generate` is raw completion (no chat template) → instruct/Q&A eval is invalid; only argmax-identity
  lever gates and "Write X…" prompts are trustworthy today.
- **Evidence to close:** run a quality suite through `/v1/chat/completions` (applies the template) — **R1 is now CLOSED, so
  this is UNBLOCKED** — or add manual per-model chat templates to `ssm_quality_suite.sh`.

## R4 — mamba2 long-context path is broken 🟡
- **Risk:** mamba2-370M returns 0.00 tps at 8k (a pure SSM has no context cap → it's a kernel bug). Secondary model.
- **Evidence to close:** fix the mamba2 long-ctx kernel; re-bench the matrix. **Low priority** (RWKV-7 is the primary SSM).

## R5 — RWKV-7-0.4B answer quality is unquantified 🟡
- **Risk:** RWKV-7 is a 0.4B model; its raw quality vs Qwen-3B is unknown per task class. Routing must not assume parity.
- **Evidence to close:** a valid instruct quality suite (gated on R3/R1) per class (retrieval/JSON/math/instruction/
  multilingual/long-ctx-retrieval); record per-class pass/fail → fill `ssm_model_selection.md`'s class overlay.

## R6 — `preflight_fast` fails on `cargo fmt --check` (pre-existing) 🟢
- **Risk:** repo-wide rustfmt drift (~190 committed locations across many modules) fails the preflight fmt gate. NOT from
  the SSM work; the repo's real CI bar is clippy+build, not fmt. **Do NOT run global `cargo fmt`** (reformats unrelated lanes).
- **Evidence to close:** an attended, isolated repo-wide fmt commit, OR relax preflight's fmt check to a warning.

## R7 — dirty tree mixes lanes (commit hygiene) 🟢
- **Risk:** the working tree spans SSM-serve, training/KD, CI/harness, diagnostics, and docs. A careless commit mixes them.
- **Evidence to close:** follow `commit_plan.md` — commit one narrow lane at a time; never stage `reports/`.

## R8 — KD drafts are undertrained 🟢
- **Risk:** KD beat SFT (75M ~19.4% vs 17.7% top-1 agreement) but both are far from converged (~60% target). Not a no-go.
- **Evidence to close:** a separate, longer KD training campaign with more corpus (out of scope for the inference runtime).

## R8b — Condense Model Press is documented but not built 🟢 — C1 PLANNER LANDED (2026-06-22)
- **Risk:** the condensation frontier depends on a memory-budgeted dry-run planner, resumable out-of-core shard/tensor
  pressing, and output-damage-ranked bit allocation. Today the legacy STRAND/TQ ingredients exist; the product claim
  ("quantize a parent that cannot fit fully resident on this machine") is being built piece by piece.
- **DONE (C1, the first piece):** `hawking press --dry-run --memory-budget <SIZE> --target <BITS> --weights <gguf>` is
  implemented (`crates/hawking/src/main.rs` `press_main`) — GGUF-metadata-only (no weights/GPU/network), prints a truthful
  Press Plan and a budget verdict that names the **wedge** (out-of-core peak vs full-resident). MEASURED: Qwen-3B at a 2 GB
  budget → out-of-core press FITS (1.32 GiB) while naive full-resident EXCEEDS (11.50 GiB). Parser unit tests `press_tests`
  GREEN; the planner code is build- + clippy-clean. See `condense_frontier_2026_06_22.md` C1 STATUS.
- **Still open:** safetensors (fp16 HF parent) metadata reader (today GGUF-only); scratch/thread/resume fields; then C2
  out-of-core writer, C3 damage-ranked allocation, C4 condense-then-recover. **Owner-gated:** no frontier downloads, cloud
  spend, or published derivatives. The bake path itself is NOT implemented (`--dry-run` only; non-dry-run prints a gated notice).
- **Naming:** Condense is the public name; STRAND stays legacy/internal until aliases land
  (`condense_naming_migration_2026_06_22.md`).
- **Pre-existing (unrelated to Condense):** `cargo clippy -D warnings` flags dead-code in `hawking-core/json_constrain.rs`
  (`ValidFirstBytes::None` never constructed) — on HEAD, not from this work; would fail a strict workspace clippy gate.

## R8c — Apple Fit (Lane H): A1/A2/A3 + A8-invariant LANDED 🟢 (2026-06-22)
- **DONE:** `hawking fit` (A2) per-Mac context/KV envelope + intent recs; `detect_mac()` (A1) reads real chip/RAM/OS via
  sysctl, wired into `hawking doctor` (was hardcoded M3-Pro) + `doctor --json`; **`hawking serve --auto [--intent]` (A3)** —
  picks + announces + applies the strongest stable config (KV/profile/energy), expert flags override; **anti-throttle
  invariant (A8, pick level)** baked into `auto_serve_pick` (non-safety intent never below max-capability without an explicit
  `safety_downgrade`). Gates: `fit_tests` (kv_cache, fit_zone, **auto_pick anti-throttle**) GREEN; serve --auto validated e2e
  (max-capability/safe-fit/SSM). CPU-only/opt-in; default serve unchanged. See `apple_fit_frontier_2026_06_22.md` A1/A2/A3/A8.
- **Risk / still open:** the A8 gate is enforced at the *chooser* level (unit-tested), NOT yet as a *measured serving*
  regression (run auto vs best-manual and fail on unstated material tps/quality/context loss) — that needs A6 measurements +
  a serving harness. Also pending: A4 live memory-pressure engine, A5 measured long-ctx routing, A6 energy/thermal cards, A7
  Mac-native model experience (pull/registry/hawkingd), serve context-cap wiring (today the cap is announced/advisory).
- **Evidence to close:** add the measured A8 serving gate; e2e expert-override tests; A4/A5/A6/A7. Do NOT let auto become a
  performance ceiling.

## R9 — speed/compression/quality could regress silently — 🟢 PARTIALLY CLOSED (2026-06-22)
- **Was:** CI enforced only fmt/clippy/build/test; correctness was locked by 193 golden hashes but **speed, footprint, and
  quality had no enforced floor** — any could regress between manual overnight measurements. Master-plan §7 ranked this the
  #2 critical-path risk; the owner flagged it explicitly ("ensure our wins persist").
- **Now (built + proven):** `tools/ci/regression_gate.sh` + committed `tools/ci/baselines/regression_baseline.json` enforce
  footprint ceilings (measured exact bytes), decode_tps floors, and lever argmax-identity floors; exit non-zero on breach;
  wired into `preflight.sh` + `overnight_hardening.sh` (`RUN_REGRESSION=1`). Floors are CATEGORY-regression thresholds
  (~10–15% below the warm median) calibrated to the noise floor so the gate does not flap. Live GREEN (6 enforced, fail=0,
  `reports/regression/20260622T140213Z/`); red path proven (too-tight ceiling → exit 1). This now protects the decode path,
  making the R1b hot-path change safe to take.
- **Still open (in the baseline's `pending_not_enforced`):** serve_decode_tps floor (add after R1b), int4-KV-PC perplexity
  (R2), instruct-quality (R3/R5), and wiring the perf job into GitHub CI (currently local/overnight only).

## Current best next gate
**R1b RWKV serve throughput.** R1's correctness parity is already green; keep it green while resizing the multiseq decode
arena to active slots, then re-run `ssm_serve_smoke.sh` and `ssm_product_gate.sh`. The parallel unblocked gate is R3/R5:
run valid chat-templated quality through `/v1/chat/completions` so RWKV routing decisions rest on measured quality, not tps
alone.
