# Roadmap — ranked remaining opportunities (autonomous campaign, 2026-06-21)

Ranked by (impact × validation-confidence ÷ effort/risk). Full detail + evidence in
`docs/plans/ratios_roadmap_2026_06_21.md`. Status: 🔵 in-flight · 📐 design-ready · 🧭 research · 🟡 blocked-on-gate.

| # | Lever | Ratio | Impact | Effort | Risk | Status | Gate / next |
|---|---|---|---|---|---|---|---|
| 1 | **RWKV-7 / SSM long-ctx production path** (flat decode, no KV wall) | speed (long-ctx) | huge at depth (Qwen 4.6× drop @8k; RWKV ~14× @8k) | low-med | low | ✅ validated, 📐 productionize | serving docs, quality gate, workload bench, model-selection defaults |
| 2 | **Per-channel int4-KV wiring** (`HAWKING_QWEN_INT4_KV_PC`) | compression | **−75% KV** (3.5× @32K) + long-ctx tps | med (kernels built; hot-path wiring remains) | med | 📐 ready-to-wire | parity (decode==CPU) + **real-model PPL** + long-ctx tps @8-16k |
| 3 | **MLX-diff → Q4_K GEMV structural fix** (split-K / 128-bit loads / register-block) | speed | the **~1.6× gap** (56%→~70% BW) | high | high (A10 −16.8% prior) | 🧭 deferred research | busy-time GB/s oracle, then prototype top delta behind a flag |
| 4 | **f16 activations in decode GEMV** (read x as half) | speed | byte-cut never tried on x | med (new kernel) | med (x may be L2-resident) | 📐 design | f16-x twin of ffn_down GEMV; argmax-gate ≥90%; GB/s delta |
| 5 | **GQA group-coalesced MHA** (1 TG/KV-head, K/V read once) | speed (long-ctx) | ~8× attention KV-read cut | med (new kernel) | low | 📐 design | attention-only µs/token @{2.5k,8k,16k}; atol ~1e-3 |
| 6 | **Density consolidation** (`eagle5*` ~1.6k LoC) | density | LoC reduction | high | high | 🔴 BLOCKED (audited) | ENTANGLED: `eagle` on ~140 lines of `qwen_dense.rs` + 17 in `main.rs`. The spec/proposal-market FRAMEWORK is the committed lossless layer (`73fc5b4`), NOT dead. Only the trained-EAGLE *proposer* is NO-GO, but it's wired into the market → untangling needs an attended pass + full parity. Not a safe autonomous removal. |
| 7 | **Column-split GEMV** for under-occupied k/v_proj | speed | +17% (SpQt) on a small byte share | med | low | 📐 design | e2e delta (k/v_proj is small fraction); atol 1e-4 |
| 8 | **Trellis sub-4-bit full bake** (`tq_bake --bpw 3.34 --match ffn_`) | compression | ~30% smaller weights | high | high (decode-SLOWER) | 🟡 deferred | size/quality/speed; existing `.tq` is a 19 MB partial |
| 9 | **Fused-epilogue default-on** (GEMV+add/rmsnorm/silu) | speed | small (intermediate traffic) | low | low | 📐 design | dispatch-count + GB/s; bounded win |

## Infrastructure / testing backlog (goal-mandated)
- ✅ Durable artifacts created (`docs/campaign/{findings_summary,kill_ledger,roadmap,test_matrix,autonomous_run_log}.md`)
  plus `claude_goal_prompt.md`, `change_manifest.md`, and `ssm_productionization.md`.
- ✅ Reusable warm-bench harness script (`tools/bench/ratios.sh`): config → warm-median tps + adversarial argmax-identity, plus interleaved `abi` for sub-10% deltas.
- ✅ Local CI mirror (`tools/ci/preflight.sh`): fmt, clippy allowlist, build, test compile, parity subset, optional bench smoke.
- ✅ Overnight runner (`tools/ci/overnight_hardening.sh`): timestamped logs, non-destructive checks, sequential GPU benches, recovery summary.
- ✅ Run the existing lib test suite; record pass/fail in `test_matrix.md`; fix in-scope failures. Current evidence: `cargo test -p hawking-core --lib` = 182/182 pass.
- ▢ Add parity test for `HAWKING_QWEN_INT4_KV_PC` once wired (decode==CPU + real-K/V cosine).
- ✅ Adversarial prompt suite (code/math/JSON/multilingual/formatting) for quality gating of quant-trade levers. Current evidence: `--profile fast` 90%, `F16_KV` 100% on 10 prompts.
- ✅ Document user-facing wired/near-wired env flags discovered (`docs/env_flags.md`).

## Verdict
The ENV/CONFIG speed ceiling is small and noisy (`--profile fast` ~+3–7%, mild quality trade). The live frontier is
**(1) productionizing the RWKV-7 / SSM long-context path** (validated moat), **(2) per-channel int4-KV compression**
(concrete, mostly-built, hot-path wiring gated), and **(3) the MLX-diffable 1.6× kernel gap** only as a deferred,
high-risk research lane.

## Wiring spec — per-channel int4-KV (roadmap #1, ready-to-implement)
Kernels parity-validated (`mha_decode_perchannel_int4kv_parity` ✅). Wire behind a NEW `HAWKING_QWEN_INT4_KV_PC`
(default-OFF → scoped + reversible). Mirror the disabled per-ROW path in `qwen_dense.rs`:
1. **Flag + incompat** (≈:4732 / :5240): `int4_kv_pc = env_on("HAWKING_QWEN_INT4_KV_PC")`; copy the F16_KV / W4A8 /
   FLASH_ATTN / BATCH_PREFILL guards (:4754-4769). Do NOT touch the disabled per-row `INT4_KV`.
2. **Arena** (≈:5295): add `k_chan_scales` / `v_chan_scales` f16 `PinnedBuffer`s sized `n_kv_heads * head_dim`
   (extend `ensure_int4_kv` → `ensure_int4_kv_pc`).
3. **Calib at prefill→decode boundary** (once, post-prefill): `kv_int4_calib_max_tcb(tcb, &k_chan_scales, &v_chan_scales,
   n_kv_heads, head_dim)` — running-max fold → finalized per-(layer,kvh,channel) scales.
4. **Append swap** (:6255 → `_pc`): `kv_quant_int4_append_pc_tcb(tcb, &k_chan_scales, &v_chan_scales, n_kv_heads, head_dim)`.
5. **Decode swap** (:6330 → `_pc`): `mha_decode_flash_int4kv_pc_tcb(tcb, &k_chan_scales, &v_chan_scales, seq_len, head_dim, …)`.

**Gate before default-on:** (a) `cargo test -p hawking-core --test mha_decode_perchannel_int4kv_parity` (passes); (b) long-ctx
coherence (`HAWKING_QWEN_INT4_KV_PC=1` @8k — no "The The The" collapse); (c) real-model perplexity vs f32 KV.
**Strategic note:** primarily a long-ctx FOOTPRINT lever (−75% KV); for long-ctx SPEED the RWKV-7 SSM path dominates (no KV at all).
