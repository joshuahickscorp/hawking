# Independent Hawking runtime and acceleration audit

## Role

Read-only audit. Do not edit files. Reconcile the latest committed Temporal
Gravity/base-runtime work and the existing speculative/parallel-token subsystem
against the final-ascent directive.

## Inputs

Inspect:

- current HEAD and all relevant committed branches since 2026-07-26;
- `HAWKING_RESUME_CHECKPOINT.md`;
- runtime, parity, TPS, profiler, acceleration, and artifact-bound receipts;
- `crates/hawking-core`, `crates/hawking-serve`, runtime examples/tests;
- unintegrated Grok worktrees only to identify candidate work, never to claim it
  landed;
- Numeric Parity V2.1 evidence and the latest artifact hash each measurement
  actually binds to.

## Required assessment

1. State which base-runtime features are committed and genuinely exercised:
   one-time verification, persistent tensor registry, native packed execution,
   BF16 head on device, GPU logits/top-k/sampling, device activations/routing/KV,
   routed expert waves, attention/IndexShare fusion, command-buffer collapse,
   replayable token graph, persistent causal loop.
2. Report evidence-backed `BASE_TRUE_TPS`, TTFT, prefill, bytes/token,
   operations/token, command buffers/token, cold/warm and 2K/8K/32K status. Name
   the exact artifact/hash and hardware for each number.
3. Distinguish simulator, microbenchmark, projected roofline, partial block, and
   true complete-token results.
4. Audit Numeric Parity V2.1: byte-exact semantics, same-backend determinism,
   condition-aware cross-backend metrics, exact router/top-k/token/tool
   decisions, near-tie fallback.
5. Reconcile the generic provider registry for feature-space/EAGLE, MTP,
   self-speculation, tree verification, n-gram/suffix/prompt lookup, prefix/KV
   caching, and Fabric-assisted draft/verify. Separate base TPS from accepted
   verified TPS and identify any unsafe path where unverified tokens could enter
   canonical context/tools/files/actions.
6. Identify stale/contradictory receipts and unintegrated committed work.

Return a dependency-ordered bounded worklist with exact files/tests and promotion
gates. Separate work executable before a capable GLM artifact exists from work
that is artifact-blocked. Do not run a heavy profiler, touch MOP, or call a
projection a measurement.
