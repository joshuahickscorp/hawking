# Phase 1.1 spec — GPU greedy sampling default (scout ac3750cc, 2026-06-01)

> Full transcript in agent ac3750cc. This is the actionable distillation.

## Premise correction (the brief was inverted)
GPU argmax ALREADY exists and is wired. It runs whenever `DISMANTLE_QWEN_TCB=1`
(which is in the locked fast-path / bench baseline). `forward_token_greedy_tcb`
keeps the ~600KB logits on-GPU and returns a 4-byte token id. The ~600KB
host-copy + CPU argmax is on the OTHER path: `forward_token` (qwen_dense.rs:2807;
`vec![0.0f32; vocab]` at :2973) + `sampler.sample` (sample/mod.rs:43 → `argmax`
:105). Selector: qwen_dense.rs:2602; `use_tcb = env_on("DISMANTLE_QWEN_TCB") &&
temp==0` (:1547-1548). The 0.2 trace shows `sample_argmax_f32` 1×/tok BECAUSE
the trace ran with TCB=1.

**⇒ Phase 1.1 = flip the DEFAULT so greedy (temp==0) takes the TCB/GPU-argmax
path without the opt-in.** Kernel work is already done.

## Change (default-flip; 2 sites, lockstep)
- Decode qwen_dense.rs:1547-1548 → `use_tcb = req.sampling.temperature == 0.0 &&
  crate::env_opt_out("DISMANTLE_QWEN_TCB")` (`env_opt_out` lib.rs:41 = true unless
  set to 0/false/off/no). Keeps `TCB=0` escape hatch (needed for the A/B bench);
  KEEP the `temp==0` guard (temp>0 must stay CPU — GPU kernel is pure argmax).
- Prefill qwen_dense.rs:1421 → couple to the SAME condition:
  `use_tcb_prefill = req.sampling.temperature == 0.0 && env_opt_out("DISMANTLE_QWEN_TCB")`.
  This is the R5 mitigation (eliminates the new mixed mode). No structural buffer
  changes; `forward_token`'s host logits vec stops being allocated on greedy.

## Gate (bit-identical) — the subtle part
- **Correct gate = Path-A-vs-Path-A:** capture `dismantle batch-hash` b3sum on
  `main`/pre-flip with `DISMANTLE_QWEN_TCB=1` + the same prod env (vocab-prune,
  Q4K-lmhead, predec), then re-run post-flip with NO flag. Both are Path A →
  must be byte-identical (hash column).
- CPU-vs-GPU (TCB=0 vs default) is **NOT** byte-identical when vocab-prune is on
  OR the LM head is Q4_K/predec (different arithmetic: f16 CPU head vs Q4_K GPU
  head; pruned vs full-vocab argmax). Only exact for an f16 full LM head.
- `tests/v1e_gpu_argmax_parity.rs` (`sample_argmax_f32_tcb` == CPU argmax) is
  necessary-not-sufficient.

## Risks
- **R5 (highest): new mixed mode** TCB-prefill + CPU-decode (temp>0) → the KV
  mirror (`mirror_arena_kv_into_self` :1497) must run or output is wrong (CPU
  decode reads host `self.kv`; TCB prefill wrote only the GPU arena). MITIGATED by
  coupling prefill to `temp==0` (above).
- temp>0 / top-k / top-p: guarded by `&& temp==0.0` — keep it.
- `ffn_capturing`, spec-decode (eagle5/user-draft), non-macOS, other model
  families (mixtral/gemma2/phi3 still CPU-sample): unaffected.

## tps
- Paired vs the **fast-path baseline (already TCB=1) ≈ 0** — not a win over the
  bench baseline. Paired vs the **CPU path (TCB=0)**: small but real (removes the
  608KB copy+sync, 150K-element CPU argmax, per-token alloc). Caveat: A (TCB=0)
  and B (default) use DIFFERENT LM-head GEMVs, so the delta includes the GEMV
  diff, not just sampling — measure, don't assert.
- **Ship gate:** bit-identical (Path-A-vs-Path-A) + B ≥ A (no regression).

## Strategic note
Phase 1.1 banks a win for the **default user** (no env flags) and is a
bit-identity/cleanliness step — it is NOT new tps over the fast-path baseline.
The only genuinely-new Phase-1 tps lever is **f16-scales** (1.2). Real tps over
the fast-path is Phase 2 (GEMV efficiency = the 1.55× gap).
