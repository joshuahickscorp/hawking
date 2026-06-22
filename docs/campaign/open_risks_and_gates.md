# Hawking — Open Risks & Gates (2026-06-22)

Each risk: what it is, evidence required to close it, and the current best next gate. Ordered by leverage.

## R1 — RWKV serve decode — ✅ CLOSED (2026-06-22)
- **RESOLVED.** Root cause = stale bundle-wide `RwkvGpu.fresh` flag (NOT a layout/stride bug): after `prefill_slot` copies
  the recurrent state into the GPU slot, the first multiseq token-shift ran `sx=-att_in`, discarding the copied
  `att_shift`/`ffn_shift`. **Fix:** clear `g.fresh=false` in `prefill_slot` after the copy (`rwkv7.rs:1222`) + emit the SSE
  `{stats}`/`[DONE]` terminator on any stream end (`http.rs`). **Parity gate GREEN; `ssm_serve_smoke.sh` fail=0** (16 coherent tokens).

## R1b — RWKV serve THROUGHPUT (NEW top gate) 🟡
- **Risk:** serve dec_tps ~7.8 vs ~119 single-stream — the B=8 multiseq arena does 8-stream work for 1 active stream.
- **Evidence to close:** size the multiseq decode arena to the number of ACTIVE slots; re-bench serve tps toward single-stream.
  **Higher-care** (the multiseq arena) — keep `rwkv7_prefill_slot_multiseq_parity` green through the change.

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

## Current best next gate
**R1's parity test.** It is the lynchpin: green → coherent RWKV serve → unblocks the SSM serve product (R-snapshot) AND a
valid quality gate (R3/R5).
