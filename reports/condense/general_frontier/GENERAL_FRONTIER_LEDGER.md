# GENERAL FRONTIER LEDGER

updated 2026-07-19T02:36:44Z

## Authoritative
- commit: e2609f94 (accepted handoff) ; Gate-F branch codex/general-frontier-gate-f
- Second-Light baseline b4655883 (0.770 BPW, NEGATIVE) accepted as prior, NOT rerun
- 32-trial geometry search accepted as bounded PROXY prior

## Active
- parent: openai/gpt-oss-120b (the only present source; the real compute lane)
- backend: apple / MPS (one heavy local lease)
- controller: none live (Gate-F runs are bounded scripts; durable frontier controller from prior generation is idle/COMPLETE)

## Gate-F progress (120B)
- G0 reproduction: PASS (deterministic)
- G1 larger expert reproduction: PASS-proxy. mlp1 robust (~0.005 all PQ); mlp2 pq_protected_islands wins decisively (val 0.149 vs rvq 0.447, plain PQ 0.610), stable calib/val, CPU/Metal parity within tol. capability_parity False.
- G2-G5: pending

## Frontier (proxy priors, NOT champions)
- expert_mlp1: pq_islands ~0.005 ; expert_mlp2: pq_protected_islands 0.149

## Blockers (honest)
- CUDA/RunPod: BLOCKED - no sealed cloud budget (HAWKING_CLOUD_BUDGET absent); provisioning plan + budget schema only; NO paid launch
- 685B / 1T / 1.6T: source ABSENT locally - source-authority/adapter prep only (contracts exist); no giant run possible
- capability tier: needs an HF-validated forward + true-residual + holdout (G3/G4)

## Next exact edits
- G2 complete-layer gate (router + all expert paths + weighted combine + residual, layer 0, full byte accounting)
- G3 cross-layer (early/mid/late) + holdout ; then G4 short end-to-end (real Harmony logits/NLL)
- fix program-hash to exclude timestamp for stable generation identity
- when a sealed cloud budget appears: activate the CUDA lane per the provisioning plan
