# Phase 1 prep — debt + lessons from Phase 0

Distilled from `NOTES.md` "Bug parade" and the Phase 0 closeout.
Phase 1 hauls should weigh these without re-litigating them.

## Lessons from the Phase 0 bug parade

1. **Modern GGUF MoE packs experts into one 3D tensor**
   (`blk.{li}.ffn_gate_exps.weight`), not one tensor per expert.
   Loader slices by byte-range; per-expert size lands on a quant-
   block boundary. Phase 1 GEMV ports keep this layout — Metal
   dispatches receive the same per-expert byte slices the CPU path
   uses today.

2. **Dense FFN intermediate ≠ MoE intermediate.** For
   DeepSeek-V2-Lite they're 10944 vs 1408. Field
   `cfg.ffn_intermediate` exists; respect it. Don't conflate.

3. **Shared experts are stored as ONE fused MLP** with
   `intermediate = n_shared × moe_intermediate`. Modeled as a
   length-1 `Vec<Expert>` with the wider intermediate. Phase 1
   doesn't change this; just keep the convention.

4. **Q4_K_M-tagged GGUFs mix in Q5_0 / Q8_0 / Q6_K** for specific
   tensors. The dequant table covers all of them. Phase 1's
   numerical parity tests should at least exercise Q4_K (the bulk
   of weight bytes).

5. **K-quant bit-pack is the most fragile dequant.** `decode_q_k_scale_min`,
   `Q5_K qh layout`, `Q6_K qhi shifts` were all wrong on first
   write; ggml-quants.c is canonical. Phase 1 GEMV ports MUST NOT
   change quant code; the parity test from G1.1 catches drift.

6. **Lazy expert dequant is the memory-survival path.**
   `DeepSeekV2.gguf` keeps the mmap alive; `TensorRef` byte-pointers
   per expert; dequant on-demand into reusable scratch. Resident
   working set ~2 GB. Phase 1 must preserve this — eager
   materialization at any layer is a halt condition.

7. **ByteLevel pre-tokenizer + decoder are non-negotiable** for the
   GGUF-fallback BPE path. Encode and decode both need it.

8. **Auto-shutoff lessons:** Ctrl-C handler with two-stage protocol
   (graceful → exit(130)). `--max-stall-ms` watchdog. `StopReason::Aborted`.
   These are wired and used by the haul runner.

## Open debt (not blocking Phase 1)

- The mlx-lm bench infra was stripped 2026-04-27. If we ever want
  to re-bench against MLX, restore `crates/dismantle-bench/src/competitors/mlxlm.rs`
  from git history and re-add the SKIP_MLXLM env switch in
  `tools/competitors/smoke.sh`. Out of scope for Phase 1.
- llama.cpp baseline numbers vary by ~30% between thermal runs
  (38–56 tok/s decode observed). The audit doc's `~48` is a
  median-of-2 placeholder. A proper full-matrix run is overnight
  work for a future haul. Out of scope for Phase 1.
- `crates/dismantle-bench/src/competitors/dismantle.rs` loads the
  engine in-process while `llamacpp.rs` spawns a subprocess. The
  asymmetry biases startup-cost comparisons. Document but don't
  fix in Phase 1.

## What Phase 0 closeout did NOT address

These are recognized gaps that Phase 1 hauls might surface but are
not in scope to fix unless they block an item:

- **Numerical parity test infrastructure.** Phase 0 had no parity
  tests against ggml-quants reference dequant. Phase 1 G1.1 builds
  this layer (`tests/correctness/phase1_kernel_parity.rs`).
- **End-to-end token regression.** Phase 0 captured one greedy
  output by hand. Phase 1 G1.2+ will capture the locked-baseline
  via `tools/haul/capture-baseline.sh`.
- **Metal pipeline state caching across runs.** `MetalContext`
  caches pipelines per-process. Cross-process caching (mtl_compile
  output to disk) is a future optimization, not blocking.
- **Model card / config validation.** `DeepSeekConfig` has
  `or_else` fallbacks for several GGUF metadata keys. Some fall to
  hardcoded defaults that may drift if the GGUF schema evolves.
  Acceptable for Phase 1; may need attention if a different exporter
  is tried.

## Phase 1 design implications

Reading the bug parade, the Phase 1 GEMV ports must:

- **Not change quant code paths.** The Q4_K dequant is correct
  (verified by Phase 0 producing coherent text); a parity test
  guards it.
- **Not change the lazy-dequant invariant.** Every Metal GEMV that
  reads expert weights still goes through `dequant_ref_into` on a
  reusable scratch buffer. The Metal kernel reads the dequanted
  fp32 (or f16) buffer; it does not read the GGUF mmap directly.
  (Direct-mmap reading is wedge 2 work, not Phase 1.)
- **Not change the tokenizer.** ByteLevel encoder/decoder are
  load-bearing; don't touch.
- **Not change the engine API.** `EngineConfig`, `GenerateRequest`,
  `Engine` trait are stable from Phase 0 onwards.

If a Phase 1 item appears to require changing any of the above,
that's a halt condition: write the blocked doc, escalate to
attended session.

## Co-existence prep

Phase 1's first haul runs **with another GPU/RAM-heavy process
active** (slm training). The protections are:

- `tools/haul/coexist.sh probe` per-item check + 30s retry × 5.
- `tools/haul/coexist.sh watch` sidecar — SIGSTOP/SIGCONTs the active
  gate's tree on pressure transitions so it yields CPU/GPU to slm
  without losing state. Started automatically by `coexist.sh launch`.
- All dismantle subprocesses launched with `nice -n 19 taskpolicy -b`.
- RSS sentinel (5 GB ceiling).
- Synthetic-first parity tests (no model load) where possible.
- 30s inter-item cool-down.
- 3-token (not 5-token) regression validation.

These are not negotiable per CLAUDE.md `§ Memory-coexist rule`.
Phase 1 will be slower than ideal. Correctness over speed.
