# Lever 3 — Phase F3 (async verify-start) audit

## What it is
Overlap the last Eagle4 head's draft step (CPU + AMX) with the first
V2-Lite verifier layer's expert prefetch (Metal command queue).

## What it needs
1. **Multi-queue Metal context** — the existing MetalContext uses ONE
   command queue. Async overlap requires either (a) a second queue
   for the verifier or (b) commit-while-encode patterns. Neither is
   currently supported.
2. **Eagle4 head's last step boundary** — the head's last propose
   call needs to commit-and-wait BEFORE the verify can start (because
   verify needs the head's last draft_token as batch[1]). True
   overlap is only possible if we run head's last step IN PARALLEL
   with verifier's BEGIN-OF-WORK (e.g., kv_append at slot K-1
   and pre-load of layer 0 expert weights).
3. **Profile flag** — `verify_kernels = "parallel-k-union-async"`
   would gate this. 24+ hours of dispatch-graph refactor.

## Audit verdict
F3 in its full form requires significant Metal-API refactoring (new
command queue lifecycle, multi-queue synchronization primitives, and
careful scheduling). Estimated 24-48 hours of focused work for
production-grade. Projected gain (per AUTONOMOUS_PLAN.md): +5-8 dec_tps.

For path-to-125: lower ROI/effort ratio than:
  - Continuing Branch 3 head re-arch experiments (1-3 hrs / iter)
  - Phase F1 AMX extension (4-8 hrs)
  - Stage 0.5 MLX rewrites (6-10 hrs)

DEFER F3 until the easier levers have been exhausted.
