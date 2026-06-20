# Dismantle/Hawking Public Pain Radar

Generated from `docs/research/pain_radar/ledger.jsonl`.

This file tracks public complaints that can become Apple-local benchmark cases.

## Cluster Summary

| Pain class | Count | Benchmark | Feature |
|---|---:|---|---|
| Cache invalidation opacity | 2 | `cache_miss_taxonomy` | `reason_coded_prefix_cache` |
| High-concurrency collapse | 2 | `high_concurrency_decode` | `continuous_batching_scheduler` |
| Install/runtime confusion | 1 | `fresh_machine_setup` | `one_command_apple_path` |
| Long-context reprocessing | 3 | `shared_agent` | `detached_prefix_state` |
| Memory cliff | 2 | `memory_budget_matrix` | `resident_memory_ledger` |
| Apple/Metal backend gap | 2 | `apple_backend_gate` | `metal_first_defaults` |
| Spec decode regression | 1 | `spec_decode_gate` | `net_positive_spec_gate` |

## Cache invalidation opacity

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 14 | llama.cpp | unknown | maybe | yes | [Prompt-cache state drift in multi-turn conversations](https://github.com/ggml-org/llama.cpp/issues/21681) | `reason_coded_prefix_cache` |
| 12 | vLLM | server_gpu | maybe | yes | [Agent context mutability vs prefix/KV cache assumptions](https://github.com/vllm-project/vllm/issues/37168) | `reason_coded_prefix_cache` |

## High-concurrency collapse

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 12 | vLLM | server_gpu | maybe | yes | [Engine core deadlock under concurrent load](https://github.com/vllm-project/vllm/issues/37729) | `continuous_batching_scheduler` |
| 5 | vLLM | server_gpu | no | no | [Context parallelism and sequence parallelism RFC](https://github.com/vllm-project/vllm/issues/22693) | `continuous_batching_scheduler` |

## Install/runtime confusion

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 16 | vLLM | apple | maybe | yes | [Apple-local install/runtime failure on newer Qwen model path](https://github.com/vllm-project/vllm/issues/38591) | `one_command_apple_path` |

## Long-context reprocessing

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 15 | llama.cpp | unknown | maybe | yes | [Context truncation/rebuild pain during long chats](https://github.com/ggml-org/llama.cpp/issues/19838) | `detached_prefix_state` |
| 15 | llama.cpp | unknown | maybe | yes | [Prompt cache forces full re-processing on hybrid long-context turns](https://github.com/ggml-org/llama.cpp/issues/19794) | `detached_prefix_state` |
| 15 | llama.cpp | unknown | maybe | yes | [Second-turn performance drop with long context and multimodal state](https://github.com/ggml-org/llama.cpp/issues/20133) | `detached_prefix_state` |

## Memory cliff

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 18 | llama.cpp | apple | maybe | yes | [Disk-based context checkpoint offloading for long-context inference](https://github.com/ggml-org/llama.cpp/issues/20697) | `resident_memory_ledger` |
| 13 | vLLM | server_gpu | maybe | yes | [KV connector changes visible capacity/concurrency behavior](https://github.com/vllm-project/vllm/issues/42024) | `resident_memory_ledger` |

## Apple/Metal backend gap

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 21 | vLLM | apple | yes | yes | [Mac/Metal/MPS support gap](https://github.com/vllm-project/vllm/issues/1441) | `metal_first_defaults` |
| 21 | vLLM | apple | yes | yes | [Metal support request/thread](https://github.com/vllm-project/vllm/issues/19073) | `metal_first_defaults` |

## Spec decode regression

| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |
|---:|---|---|---|---|---|---|
| 18 | llama.cpp | apple | maybe | yes | [MTP speculative decoding degrades throughput on Metal](https://github.com/ggml-org/llama.cpp/issues/23752) | `net_positive_spec_gate` |
