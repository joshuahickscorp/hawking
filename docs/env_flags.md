# Hawking env-flag reference (decode / compression / profiles)

The engine exposes **284 `HAWKING_*` env flags** (`rg -oN 'HAWKING_[A-Z0-9_]+' crates --glob '*.rs' | sort -u`).
This curates the **user-facing decode + compression** subset, with effects **measured in the 2026-06-21 campaign**
(warm 5-trial median, Qwen2.5-3B-Q4_K_M; see `docs/campaign/test_matrix.md`). To find any flag's default in code:
`crate::env_on("X")` = **default-OFF** (on iff set & != "0"); a `var_os(...).map_or(true, …)` = **default-ON** (opt-out via `=0`).

## Profiles (preferred entry point)
| Flag / CLI | Effect | Measured |
|---|---|---|
| `--profile fast` | bundles vocab-prune + Q4K-LM-head + Q4K-FFN-down + predec + f16-scales | **~+3–7% warm (noisy — at the bench noise floor; see `docs/campaign/test_matrix.md`), 83–90% argmax-identity** (mild quant trade) — speed-priority |
| (no profile) = default | the bit-identical fast levers (predec, TCB, 2r Q6_K) without the quality-trade ones | ~40 tps warm baseline, bit-identical reference |

## Speed levers
| Flag | Default | Effect | Measured / note |
|---|---|---|---|
| `HAWKING_QWEN_Q4K_PREDEC` | ON | pre-decoded Q4_K sub-block scales (the +34% headline) | bit-identical; the core decode win |
| `HAWKING_QWEN_Q6K_SWIGLU_2R` | ON | Q6_K ffn_down at 2 rows/simdgroup | **2r=40.5 is optimal** (1r=39.9, 4r=40.0); leave on |
| `HAWKING_QWEN_Q6K_SWIGLU_4R` | OFF | 4 rows/TG variant | −1% vs 2r; not better |
| `HAWKING_QWEN_PREDEC_F16SCALES` | OFF | f16 predec scale stream | +6–9% historically; **~0% on this binary**; failed an earlier quality oracle → opt-in |
| `HAWKING_QWEN_FFN_DOWN_Q4K` | OFF | requant ffn_down Q6_K→Q4_K | **cold +29% = artifact; warm ~0%** (REJECTED as a speed lever) |
| `HAWKING_QWEN_VOCAB_PRUNE`, `_Q4K_LMHEAD` | OFF (in `fast`) | prune LM-head vocab / Q4_K LM-head | part of `--profile fast` |
| `HAWKING_QWEN_PREDEC_2R/4R`, `_OPROJ_4R`, `_PAIR_2R_INLINE` | OFF | per-shape GEMV row-blocking variants | micro-levers, mostly below gate |
| `HAWKING_QWEN_FLASH_ATTN`, `_CONCURRENT_QKV` | OFF | flash attention / concurrent QKV encode | long-ctx / small; see dead_levers |

## Compression / KV-cache
| Flag | Default | Effect | Measured / status |
|---|---|---|---|
| `HAWKING_QWEN_F16_KV` | OFF | f16 KV cache | **−50% KV footprint; +1.9%@2.5k (scales); 88% identity** — clean long-ctx lever |
| `HAWKING_QWEN_INT4_KV` | DISABLED (fail-loud) | per-ROW int4 KV | **NO-GO** (per-row outlier collapse, 0% identity) |
| `HAWKING_QWEN_INT4_KV_EXPERIMENTAL` | OFF | force-enable the broken per-row path | for redesign only |
| `HAWKING_QWEN_INT4_KV_PC` | (being wired) | **per-CHANNEL int4 KV** | **−75% KV, cosine 0.998** — the live compression lever; needs PPL gate |
| `HAWKING_QWEN_TQ` | OFF | trellis (sub-4-bit) FFN from `<weights>.tq` | ~3.34 bpw weights but **decode-slower**; existing `.tq` is a 19 MB partial |
| `HAWKING_QWEN_W4A8`, `HAWKING_QWEN_AWQ` | OFF | 4-bit weight / 8-bit activation | quality-blocked (held) |

## Spec-decode / Event-Horizon (NET-NEGATIVE for speed — default OFF)
| Flag | Effect |
|---|---|
| `HAWKING_QWEN_USER_DRAFT` | user-ngram draft path (set `=0` to force no-spec canonical greedy) |
| `HAWKING_QWEN_EVENT_HORIZON`, `HAWKING_EH_SAM`, `HAWKING_EH_PARALLEL_DRAFT` | the EH proposal market (lossless, but slower — see kill_ledger) |
| `HAWKING_QWEN_EAGLE5[_K/_CAPTURE/_BATCHED]` | trained EAGLE head (net-negative for speed) |

## Prefix cache / stateful / capture / debug
| Flag | Effect |
|---|---|
| `HAWKING_QWEN_PREFIX_CACHE`, `HAWKING_PREFIX_CACHE_DIR` | prefill KV prefix cache (moat for repeated prefixes) |
| `HAWKING_QWEN_USAGE_CAPTURE`, `_CAPTURE_FFN_PATH`, `_EAGLE5_CAPTURE` | activation / path capture for training/analysis |
| `HAWKING_TCB_TRACE`, `HAWKING_TRACE_DISPATCH` (`--trace-dispatch`) | Metal dispatch tracing + structural counters |
| `HAWKING_BACKEND_SEAM`, `HAWKING_FORCE_CPU`, `HAWKING_RWKV7_F32_GGUF`, `HAWKING_ENERGY_EFFICIENT` | backend routing / CPU fallback / RWKV f32 / energy mode |

**Validated bottom line:** for speed use `--profile fast` (~+3–7%, noisy, mild trade) or the bit-identical default; for long-context
footprint use `F16_KV` (and soon `INT4_KV_PC`). Everything else above is either already-default-on, opt-in micro, or a
documented dead end (`docs/campaign/kill_ledger.md`).
