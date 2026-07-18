# Hawking architecture map (2026-06-21)

Apple-Silicon-first Rust/Metal LLM inference runtime. ~92.4k Rust LoC + 13.1k Metal across 4 crates.

## Crates
| Crate | LoC | Role |
|---|---|---|
| `hawking-core` | 82.5k | the engine: model forward paths, Metal kernels, KV/cache, spec-decode, tokenizer |
| `hawking-serve` | 5.1k | OpenAI-style server (`/v1/chat`, `/v1/completions`); reads the same env flags |
| `hawking` | 3.7k | CLI bin (`generate`, `--profile`, `--trace-dispatch`, …) |
| `hawking-bench` | 1.2k | benchmark harnesses |

## `hawking-core` subsystems
| Dir | LoC | Role |
|---|---|---|
| `model/` | 21.7k | per-arch forward paths. **Hot:** `qwen_dense.rs` (9.6k — the Qwen decode path). Also `deepseek_v2.rs` (4.4k), `rwkv7.rs` (2.7k — the SSM moat), `mixtral.rs` (1.1k) |
| `kernels/` | 12.9k | Metal kernel wrappers + dispatch. **`mod.rs` is 12.4k** (125 `*_tcb` wrappers; the kernel registry) |
| `speculate/` | 5.9k | spec-decode / Event-Horizon proposal market + `eagle5.rs` (1.1k). **Net-negative for speed; lossless layer only** |
| `metal/` | 3.5k | Metal device/queue/encoder; `rwkv_decode_arena.rs` (1.2k, the SSM state arena) |
| `stateful/` | 2.1k | `prefix_cache.rs` (KV prefix reuse), `usage_capture.rs` |
| `backend/` | 1.5k | the backend seam (Metal/CPU routing, `HAWKING_BACKEND_SEAM`) |
| `cache/` | 0.9k | KV cache management |
| `tq.rs` (0.99k) | — | trellis-quant (STRAND) CPU decode reference; GPU path in `tq_gpu.rs` + `strand_bitslice.metal` |

## Shaders (13.1k Metal)
`quant.metal` (5.7k — **the Q4_K/Q6_K/Q3_K decode GEMVs, the core hot path**), `moe.metal` (1.3k), `common.metal` (1.3k),
`megakernel_qwen3b.metal` (1.1k — POC), `mha.metal` (1.1k — attention decode), `rwkv7.metal` (0.9k — SSM time/channel mix),
`strand_bitslice.metal` (0.5k — trellis), `attn.metal`, `sample.metal`, `matmul.metal`.

## The decode hot path (Qwen-3B, the optimization target)
`hawking generate` → `qwen_dense.rs::forward_token_greedy_tcb` → per layer: rmsnorm → q/k/v_proj GEMV → RoPE → MHA
(`mha.metal`) → o_proj → rmsnorm → gate/up GEMV → **ffn_down GEMV (`gemm_q6_k_fused_v2_swiglu_2r`, the largest single
read, 20% of bytes)** → LM-head. Bandwidth-bound (~1.8 GiB weights/token, ~56% of peak BW; the predec Q4_K GEMV is at the
batch-1 memory-model optimum — see `docs/campaign/kill_ledger.md`, archived, see `docs/ARCHIVE_INDEX.md`).

## Density / consolidation candidates (parity-gated, NOT yet removed)
- **`speculate/eagle5.rs` + EH neural scaffolds (~1.6k+ LoC)** — trained-EAGLE is now-dead (spec is net-negative for speed).
  CAUTION: kernels are name-string-referenced; the committed cost-aware router may reference spec infra. Needs a reference-audit.
- **`megakernel_qwen3b.metal` (1.1k) + the megakernel module** — POC, measured 4.4× SLOWER (don't wire); a documented dead experiment.
- ~~`deepseek_v2.rs` + `mixtral.rs`~~ — **NOT candidates (verified):** reachable multi-arch support via `model/mod.rs`
  dispatch (llama+mixtral / deepseek2 / qwen2 / qwen-moe / gemma2 / phi3 / rwkv7 / mamba2 / olmoe), with maintained tests
  (`cpu_backend_parity_deepseek.rs`, `v1_mixtral_smoke.rs`, `phase2_mla_metal_parity.rs`). Intentional — keep.
- 125 `*_tcb` kernel wrappers; ~30 are unwired from forward (most are parity-tested A/B variants or `*_off` ablation twins —
  intentional, not dead). The genuinely-dead-called optimized path was **per-channel int4-KV** (now being wired).

## Notes
- Env-flag surface = 284 `HAWKING_*` flags (see `docs/env_flags.md`).
- The architectural long-context differentiator is the **SSM path** (`rwkv7.rs`, flat decode) — see `docs/campaign/findings_summary.md` (archived, see `docs/ARCHIVE_INDEX.md`).
