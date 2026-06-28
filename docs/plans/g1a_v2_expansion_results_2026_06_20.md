# G1a V2 Expansion Chain Results
**Date:** 2026-06-20 21:23 UTC

## Gate Context

| | |
|---|---|
| Final PPL | 3.4489 |
| Gate result | pass |
| Phase2 report | none |
| Artifact dir | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion |

## Results

| Step | Status | Log |
|---|---|---|
| cargo check hawking-core | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/cargo_check_core.log |
| cargo check hawking-serve | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/cargo_check_serve.log |
| cargo check hawking-bench | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/cargo_check_bench.log |
| cargo check hawking-core tq | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/cargo_check_core_tq.log |
| json constraint unit tests | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/json_constraint_tests.log |
| mamba2 smoke | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/mamba2_smoke.log |
| rwkv7 metal parity | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/rwkv7_metal_parity.log |
| rwkv7 flatness quick 16k | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/rwkv7_flatness_16k.log |
| tq trellis synthetic parity | PASS | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/tq_trellis_parity.log |
| rwkv7 flatness full 64k | skipped: set G1A_V2_FULL_BENCH=1 | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/rwkv7_flatness_64k.log |
| rwkv7 tq loader | skipped: no TQ artifact at /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/export/g1a/model.tq | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/rwkv7_tq_loader.log |
| rwkv7 tq bench | skipped: no TQ artifact at /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/export/g1a/model.tq | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/rwkv7_tq_bench.log |
| llama.cpp qwen3b head-to-head | SOFT-FAIL exit 3 | /Users/scammermike/Downloads/hawking/artifacts/lowbit_rwkv7/v2_expansion/llama_qwen_head_to_head.log |

## Interpretation

This chain is deliberately wider than the G1a promote ladder. It keeps
result-dependent TQ work behind artifact checks, while still advancing the
independent surfaces that improve Dismantle against llama.cpp: JSON-mode
constraint scaffolding, Mamba2 architecture breadth, core/serve/bench compile
health, synthetic TQ parity, RWKV-7 parity, and context-depth flatness.

Set `G1A_V2_FULL_BENCH=1` for the full 64k flatness sweep. The clean-room
Qwen3B llama.cpp comparison runs by default as a soft-fail claim gate; set
`G1A_V2_LLAMA_BASELINE=0` only when you intentionally want to skip it.
