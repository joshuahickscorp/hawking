# Q8 latent KV — kernel-level microbench

Captured 2026-05-20 from [crates/dismantle-core/tests/mla_q8kv_microbench.rs](../crates/dismantle-core/tests/mla_q8kv_microbench.rs).
Production V2-Lite shapes (n_heads=16, qk_nope=128, qk_rope=64, v_head=128,
kv_lora=512). c_kv values scaled ×0.1 to simulate post-rmsnorm range.

## Per-kernel cost

| seq_len | mla_decode (f32 c_kv) µs/call | mla_decode_q8kv µs/call | speedup |
|---|---|---|---|
| 256 | 918 | 469 | **1.96×** |
| 512 | 786 | 611 | 1.29× |
| 1024 | 1230 | 958 | 1.28× |
| 2048 | 2320 | 1629 | 1.42× |
| 4096 | 4912 | 2980 | **1.65×** |

Same input data, same V2-Lite shapes, back-to-back invocations. Warmup 5
iters, measured 40–200 iters per shape (smaller seq → more iters).

## Caveats

**This microbench includes per-call buffer setup.** Each iteration allocates
fresh c_kv / k_pe / q / out buffers. The f32 path uploads `seq×512×4` bytes
of c_kv per call; the Q8 path uploads `seq×544/4` bytes (4× less). Some of
the measured speedup is from that upload-side bandwidth, not the kernel's
on-GPU work. In production, c_kv lives in a persistent pinned buffer that
doesn't get re-uploaded per call.

**Parity already verified** at every benched shape — see
[crates/dismantle-core/tests/q8_kv_parity.rs](../crates/dismantle-core/tests/q8_kv_parity.rs).
Realistic-scale data (c_kv ~ 0.1 × U[-1,1]): max-abs-diff vs f32 reference
is 2–8 × 10⁻⁴. Worst-case uniform U[-1,1]: max-abs-diff 0.1–0.18, within
the analytical Q8 noise bound.

## Expected production impact

The kernel's pure-GPU work IS bandwidth-bound on c_kv (it reads the entire
seq_len × kv_lora_rank latent twice — once for scores, once for the
weighted accumulation). Q8's 4× reduction in c_kv bandwidth is what pays
off. Even with persistent buffers eliminating the per-call upload cost,
the kernel's GPU-side reads of the cache shrink by ~75%.

At long context (seq~1300, current 14 dec_tps clean baseline), the
attention block carries a meaningful share of per-token time. Honest
estimate after accounting for buffer-setup overhead in the microbench:
**+5 to +10% e2e at seq ≥ 1024** if the production cache is migrated
fully to Q8.

The lever STACKS with the existing `attn_block_schedule = "flash"` path
(which last week showed +1.77% at long context but missed the +5% gate
on its own): flash + Q8 KV together could clear the gate.

## What's not done (next session)

The lever above is the per-kernel signal. To convert it into a dec_tps
improvement the production engine has to:

1. Allocate `mla_c_kv` as `Vec<Vec<u8>>` (Q8 byte layout) when
   `mla_schedule == "metal-mla-q8kv"` — see
   [crates/dismantle-core/src/model/deepseek_v2.rs:635](../crates/dismantle-core/src/model/deepseek_v2.rs:635)
   for the current f32 allocation.
2. Mirror the same change in `mla_c_kv_gpu` — the per-layer pinned
   buffer at [deepseek_v2.rs:797](../crates/dismantle-core/src/model/deepseek_v2.rs:797)
   needs the smaller Q8 byte size.
3. Replace `kv_append_f32` dispatches with `kv_append_q8_0_f32` at the
   TCB sites (`kv_append_f32_tcb` callers in deepseek_v2.rs).
4. Route attention decode through `mla_decode_q8kv_metal` when the
   profile gate is selected. Mirror in the arena/TCB flow
   (`mla_decode_and_o_proj_arena_tcb`) — those wrappers currently take
   the c_kv `PinnedBuffer` typed as f32; need a Q8-aware variant.
5. Profile gate value: extend `mla_schedule` to accept
   `"metal-mla-q8kv"`. Default stays `"metal-mla"`.
6. End-to-end parity test: load V2-Lite, decode 32 tokens with the gate
   off vs on, hash-compare token streams. Greedy-decoding tolerance is
   binary (same token or different); set ATOL via logit-level instead.
7. Bench gate at long context (the harness exists from earlier session,
   [crates/dismantle-bench/src/suites/decode.rs](../crates/dismantle-bench/src/suites/decode.rs)
   reads `DISMANTLE_BENCH_PROMPT_FILE`).

Step 1+2+3+4 are the engineering work. Step 6+7 are quick once the wire-in
holds. Reasonable estimate: 1–2 sessions to land the production gate.
