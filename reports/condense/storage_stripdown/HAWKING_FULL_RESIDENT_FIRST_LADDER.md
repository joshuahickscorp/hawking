# Hawking resident-first ladder

generated 2026-07-21T02:59:36Z

- volume: 926.4 GiB
- free used for fit: 555.3 GiB (measured live after stripdown)
- free measured now: 555.3 GiB
- selected next parent: **deepseek-ai/DeepSeek-V4-Flash-DSpark**

Ordering law: every parent that fits fully runs before any parent that must stream, regardless of parameter count.

| # | parent | slot | params | active | source GiB | largest shard | reserve | headroom | fit class | margin GiB |
|--:|---|---|--:|--:|--:|--:|--:|--:|---|--:|
| 1 | deepseek-ai/DeepSeek-V4-Flash-DSpark | procurement | 284B | 13B | 155.4 | 3.4 | 32.0 | 33.1 | FULL_RESIDENT_COMFORTABLE | 334.8 |
| 2 | MiniMaxAI/MiniMax-M2 | off-ladder | 230B | 10B | 214.4 | 3.4 | 32.0 | 26.8 | FULL_RESIDENT_COMFORTABLE | 282.2 |
| 3 | openai/gpt-oss-20b | off-ladder | 21B | 4B | 38.5 | 12.8 | 32.0 | 16.0 | FULL_RESIDENT_COMFORTABLE | 468.8 |
| 4 | moonshotai/Kimi-Linear-48B-A3B-Instruct | off-ladder | 48B | 3B | 91.5 | 4.7 | 32.0 | 16.0 | FULL_RESIDENT_COMFORTABLE | 415.8 |
| 5 | ibm-granite/granite-4.0-h-small | off-ladder | 32B | 9B | 60.0 | 4.6 | 32.0 | 16.0 | FULL_RESIDENT_COMFORTABLE | 447.3 |
| 6 | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 | off-ladder | 49B | 0B | 92.9 | 4.7 | 32.0 | 16.0 | FULL_RESIDENT_COMFORTABLE | 414.4 |
| 7 | Qwen/Qwen3-Next-80B-A3B-Instruct | off-ladder | 80B | 3B | 151.5 | 3.7 | 32.0 | 16.0 | FULL_RESIDENT_COMFORTABLE | 355.8 |
| 8 | Qwen/Qwen3-VL-235B-A22B-Instruct | off-ladder | 235B | 22B | 439.0 | 4.6 | 32.0 | 27.4 | FULL_RESIDENT_COMFORTABLE | 57.0 |
| 9 | openai/gpt-oss-120b | F0 | 117B | 5B | 182.3 | 60.8 | 121.5 | 60.8 | FULL_RESIDENT_COMFORTABLE | 190.7 |
| 10 | moonshotai/Kimi-K2.7-Code | off-ladder | 1070B | 32B | 554.3 | 9.1 | 32.0 | 124.6 | DOES_NOT_FIT_FULLY | -155.6 |
| 11 | moonshotai/Kimi-K2.6 | F4 | 1070B | 32B | 554.3 | 9.1 | 32.0 | 124.6 | DOES_NOT_FIT_FULLY | -155.6 |
| 12 | deepseek-ai/DeepSeek-V3.2 | F3 | 671B | 37B | 642.2 | 6.2 | 32.0 | 78.1 | DOES_NOT_FIT_FULLY | -197.0 |
| 13 | meta-llama/Llama-4-Maverick-17B-128E-Instruct | off-ladder | 400B | 17B | 748.0 | 20.0 | 40.0 | 46.6 | DOES_NOT_FIT_FULLY | -279.3 |
| 14 | Qwen/Qwen3.5-397B-A17B | F2 | 396B | 17B | 751.4 | 9.0 | 32.0 | 46.1 | DOES_NOT_FIT_FULLY | -274.2 |
| 15 | deepseek-ai/DeepSeek-V4-Pro-DSpark | OPT_1_6T | 1600B | 49B | 831.4 | 13.1 | 32.0 | 186.3 | DOES_NOT_FIT_FULLY | -494.4 |
| 16 | Qwen/Qwen3-Coder-480B-A35B-Instruct | OPT_480B | 480B | 35B | 894.4 | 3.7 | 32.0 | 55.9 | DOES_NOT_FIT_FULLY | -427.0 |
| 17 | moonshotai/Kimi-K2-Instruct | off-ladder | 1000B | 32B | 958.5 | 18.1 | 36.2 | 116.4 | DOES_NOT_FIT_FULLY | -555.8 |
| 18 | zai-org/GLM-5.2 | off-ladder | 753B | 39B | 1,403.2 | 5.0 | 32.0 | 87.7 | DOES_NOT_FIT_FULLY | -967.6 |

## Why the leader leads

- **deepseek-ai/DeepSeek-V4-Flash-DSpark** (deepseek_v4, fp8/e4m3, 43 layers, 256 experts, top-6): native fp8 e4m3 frontier MoE, 1M context, DeepSeek family; maximal distance from the just-sealed Qwen3-235B and the only frontier-class parent whose source is ALREADY sub-8-bit
- **MiniMaxAI/MiniMax-M2** (minimax_m2, fp8/float8_e4m3fn, 62 layers, None experts, top-8): native fp8 MoE, distinct family; second native-low-bit parent
- **openai/gpt-oss-20b** (gpt_oss, mxfp4, 24 layers, None experts, top-4): MXFP4 native small control; same family as the sealed F0
- **moonshotai/Kimi-Linear-48B-A3B-Instruct** (kimi_linear, bfloat16, 27 layers, 256 experts, top-None): Kimi Delta Attention linear-attention hybrid; cheapest distinct-architecture probe

## Honesty

- Every source_bytes figure is a live HfApi files_metadata sum at the pinned sha, not a nominal or model-card size.
- total_params is used ONLY to project the compact-checkpoint headroom term. It is name-derived for most rows and must not be used as a ledger denominator; bind original_weight_count from the resident tensor index at admission.
- free_bytes_used_for_fit may be a PROJECTED post-cleanup figure. The fit class is re-computed against measured free space before any download starts.
- Fit class is a STORAGE verdict. It says nothing about whether the parent is compressible, and nothing about whether the Rust engine can execute the artifact.
