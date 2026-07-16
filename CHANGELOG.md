# Changelog

Git history and signed receipts are the detailed archive. This file records
only durable release-level changes.

## Unreleased

- Consolidated repository operations around canonical domain entry points.
- Preserved CPU/Metal parity, golden decode, receipt, and low-bit promotion
  gates while reducing retired wrappers and dated handoffs.
- Continued Doctor V5, STRAND/TQ, speculative decode, SSM, continuous batching,
  and HIDE work behind their existing evidence and release boundaries.

## v2.0.0

- Added a pure-Rust/Metal inference path for dense and MoE GGUF models on Apple
  silicon.
- Added OpenAI-compatible serving, benchmark, doctor, autotune, fit, press, and
  statistics workflows.
- Added Q4_K/Q6_K/Q3_K kernels, mmap-backed weight buffers, explicit CPU
  references, parity tests, golden hashes, profile drift checks, and resource
  limits.
- Added Mixtral and DeepSeek-V2-Lite paths, speculative-decode infrastructure,
  prefix caching, batched execution foundations, and model-persistent
  benchmarking.
- Established correctness-before-speed and matched-baseline methodology.

Historical phase logs, individual commit narratives, and retired file lists are
available with:

```sh
git log --stat --decorate
git show <commit>
```
