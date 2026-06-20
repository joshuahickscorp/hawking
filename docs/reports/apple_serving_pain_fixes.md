# Apple Serving Pain Fixes

This report is the public-claim ledger for the Dismantle/Hawking serving work.
Rows stay `planned` until a benchmark reproduces the pain and a measured fix lands.

| Status | Pain | Source | Reproduction | Baseline | Fix | Post-fix | Claim |
|---|---|---|---|---|---|---|---|
| planned | Apple/Metal backend gap | [vLLM](https://github.com/vllm-project/vllm/issues/1441) | `apple_backend_gate` | pending | `metal_first_defaults` | pending | pending |
| planned | Apple/Metal backend gap | [vLLM](https://github.com/vllm-project/vllm/issues/19073) | `apple_backend_gate` | pending | `metal_first_defaults` | pending | pending |
| planned | Spec decode regression | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/23752) | `spec_decode_gate` | pending | `net_positive_spec_gate` | pending | pending |
| planned | Memory cliff | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/20697) | `memory_budget_matrix` | pending | `resident_memory_ledger` | pending | pending |
| planned | Install/runtime confusion | [vLLM](https://github.com/vllm-project/vllm/issues/38591) | `fresh_machine_setup` | pending | `one_command_apple_path` | pending | pending |
| planned | Long-context reprocessing | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/19794) | `shared_agent` | pending | `detached_prefix_state` | pending | pending |
| planned | Long-context reprocessing | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/19838) | `shared_agent` | pending | `detached_prefix_state` | pending | pending |
| planned | Long-context reprocessing | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/20133) | `shared_agent` | pending | `detached_prefix_state` | pending | pending |
| planned | Cache invalidation opacity | [llama.cpp](https://github.com/ggml-org/llama.cpp/issues/21681) | `cache_miss_taxonomy` | pending | `reason_coded_prefix_cache` | pending | pending |
| planned | Memory cliff | [vLLM](https://github.com/vllm-project/vllm/issues/42024) | `memory_budget_matrix` | pending | `resident_memory_ledger` | pending | pending |
| planned | Cache invalidation opacity | [vLLM](https://github.com/vllm-project/vllm/issues/37168) | `cache_miss_taxonomy` | pending | `reason_coded_prefix_cache` | pending | pending |
| planned | High-concurrency collapse | [vLLM](https://github.com/vllm-project/vllm/issues/37729) | `high_concurrency_decode` | pending | `continuous_batching_scheduler` | pending | pending |

## Claim Rules

- Do not mark a row `fixed` without a before/after JSONL result.
- Prefer P95/P99 and cache-hit metrics over single-stream tokens/sec.
- Separate 16GB-class and 96GB-class Apple hardware tiers.
- Treat Hawking claims as release-direction until the model/runtime artifact exists.
