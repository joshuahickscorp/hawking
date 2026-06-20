# Cloud selective-PV run — orchestration + GPU + cost (built 2026-06-13, grounded in run data)

THE RUN: selective-PV on the RED set (up/v/gate, KL-routed) of the q2 7B recon, KD-distilled,
to push 2-bit loss-tax 0.464(7B) toward the high-confidence ~0.20-0.24 (the headline:
"trained deterministic 2-bit beats AQLM-class ~8.9"). 32B is DEFERRED.

## GPU DECISION — grounded in the memory math (the empirical fp32-shadow finding)
7B selective-PV VRAM budget:
- frozen base (forward): bf16 = **14GB** (fp32 = 28GB — the BUG-A path that OOMs a 24GB card).
- trainable RED shadow (only ~25 tensors, NOT all): fp32 ~4-8GB; 8-bit Adam state ~1-2GB.
- KD teacher: **CACHED via --kd-cache (top-128)** → ~0 VRAM (the enabler). Live teacher = +14GB = OOM.
- activations (ctx512, batch1): ~1-2GB.

| GPU | $/hr (typ) | 7B selective-PV fit | needs |
|---|---|---|---|
| **RTX 3090 24GB** (current pod) | ~$0.30-0.45 | **YES, tight (~22GB)** | bf16 base + 8-bit Adam (BUG-A fix) + `--kd-cache` + selective |
| A100 40GB | ~$1.10-1.50 | YES, comfortable | nothing (fp32 base 28GB fits) — the no-code-fix fallback |
| A100/H100 80GB | ~$2-4 | trivial | overkill for 7B |

**VERDICT: stick with the 24GB 3090** (cheapest) once the bf16-shadow + 8-bit-Adam fix lands and
`--kd-cache` is on. A100-40 is the zero-code-change fallback if we don't want to touch the harness.
We do NOT need an 80GB card for 7B. (32B WOULD need ≥80GB OR the bf16 fix — deferred.)

## COST — anchor + the cost-safety
- Recipe anchor: ~$30 for the 7B canary. On a 3090 with selective(small shadow)+cached-teacher,
  plausibly **~$3-15** depending on step count and the real per-step rate.
- **COST-SAFETY (do this): the 0.5B KL-routed selective-PV gate calibrates the real per-step rate.**
  Extrapolate gate_step_rate × (14× model / GPU-speedup) × steps → the 7B cost is KNOWN before any
  pod dollar. Commit only if the extrapolated cost ≤ budget. Never run 7B blind.
- Cost levers (empirical): `--kd-cache` (kills the 2nd 7B forward), selective-PV (small shadow/requant),
  flip cloud-GPU→OK so PV uses the GPU not the slow CPU-canonical path, community/spot pricing,
  TERMINATE-on-done (the killer cost is idle: $10 ≈ ~25h idle = the whole gate wait).

## ORCHESTRATION SEQUENCING (the gated runbook)
PRE (now, free, no pod): 
  P1. land the bf16-shadow + 8-bit-Adam fix in strand-qat.py (needed for 3090; also unblocks 32B later).
  P2. run the 0.5B KL-routed selective-PV gate locally → (a) go/no-go (≥80% RED-vs-full recovery),
      (b) the per-step rate that calibrates the 7B cost.
GATE GREEN (the 0.5B gate passes):
  1. PROVISION the 3090 (or A100-40 fallback) — ~5 min. Fresh pod (don't pre-pay idle).
  2. DEPLOY (scripted ~20-30 min): cloud-selective-pv.sh + tools/ + the 4 PV scripts + the q2 7B
     recon + actmean; re-download 7B weights (15GB); cargo build (or the auto-router binary); verify.
  3. CACHE TEACHER: one 7B bf16 forward over the train windows → top-128 logit cache (--kd-cache).
  4. RUN: selective-PV, RED set, ctx512, recipe steps/LR, --kd-cache, gated by require_promoted.
  5. MIRROR DOWN: pv-7b.json → promote.py stamp (loss-tax vs the 6.629 bf16 anchor).
  6. INTERPRET: does trained 2-bit clear ~8.9 (AQLM-class)? bank the headline or the honest miss.
  7. TERMINATE the pod immediately (idle is the cost killer).

DO-NOT: run on CPU-canonical if avoidable (slow → expensive); hold a live KD teacher on 24GB (OOM);
leave the pod idle waiting for the gate (terminate, resurrect when green); promise <=0.15 (it's ~0.22).
