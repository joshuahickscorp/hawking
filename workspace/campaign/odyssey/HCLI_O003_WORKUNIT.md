/mission HCLI WorkUnit: Odyssey O003 hard-specimen Gravity continuation

You are HCLI, the scientist and executor for the continuing Odyssey campaign.
Codex is only a detachable shadow supervisor. Do the engineering, tool calls,
experiments, mutations, checkpoints, receipts, Laws/Scars, continuation, and
release yourself. Do not answer with a plan when a tool-backed experiment can
be run. Work from the current checkout and durable receipts; preserve unrelated
dirty worktree changes and never rewrite historical evidence.

## Objective

Continue the same HCLI Odyssey campaign on specimen O003
(moonshotai/Kimi-VL-A3B-Instruct, small multimodal MoE). This is the first
scientifically serious Gravity continuation after the shallow canaries. Perform
one residency session if possible: acquire/stage once, do all earned Doctor,
Gravity, and Odyssey-III work, seal the session, release NR/model memory, and
verify release. If O003 is not resident, use the existing HCLI acquisition and
streaming machinery; do not confuse static shard inspection with capability
execution. Do not evict a promising specimen at an arbitrary phase boundary.

The search must try to move through these measured milestones:

    current ~3.x -> <2.5 -> <=2.0 -> <=1.5 -> pursue <=1.0

No current result is close enough to normalize. A budget stop is not a bound.
Gravity may report only TARGET_HIT, PROVEN_UNABLE, or BUDGET_EXHAUSTED, and the
last one must name the exhausted representation level.

## Current measured authority

Doctor/census for O003:

- total parameters 16,407,657,776; source stored bytes 32,815,315,552;
  source density 16.0 bpw.
- complete-byte anatomy from workspace/campaign/odyssey/patients/O003/census.json:
  expert 28,789,702,656 bytes (87.73%); attention 958,386,816 (2.92%);
  shared_expert 899,678,208 (2.74%); embed 681,882,880 (2.08%);
  mlp_dense 674,193,120 (2.05%); lm_head 671,088,640 (2.05%);
  other 133,080,832 (0.41%); router 6,819,072; norm 483,328.
- O003_SENSITIVITY.json is measured in-place ablation evidence: zeroing any
  named organ destroys the battery, so it does not rank organs; round8 gives
  11/12 and is nearly a quantization identity. Do not call this evidence of
  equal importance. Explain this limitation and use byte contribution,
  execution frequency, routing, and new discriminators to rank organs.
- Existing measured Gravity receipts include:
  q2-g128: complete 2.1918, DEGRADED;
  q2-g32: 2.9211, DEGRADED;
  q2-g32-experts: 3.0639, CANDIDATE_PASS;
  mixed-q2q3: 2.9642, DEGRADED;
  mixed-q2q4: 3.0073, DEGRADED;
  q3-g64-experts: 3.5026, CANDIDATE_PASS.
  The external patient runner's q2-g32-experts receipt is not independent
  execution proof and does not have measured magnitude/utilization; it is a
  search signal, not a target hit.
- Existing canary checkpoints O001/O003/O005 are in
  workspace/campaign/odyssey/gauntlets/. O003's previous two-candidate loop
  exhausted only the local quantization neighborhood. Do not repeat q2/q3
  ladders as if they were deep search.
- Read workspace/campaign/odyssey/SUB1_SYNTHESIS.md and all relevant O003
  receipts. Its ranked hypotheses are clues, not accepted facts.

## Required scientific action

1. Run/verify complete Doctor byte anatomy for O003 and write a durable current
   anatomy/diagnostic receipt. Explain why the prior ~3 bpw plateau remains and
   rank the largest promising organs by measured bytes, redundancy evidence,
   sensitivity evidence, reconstructability, execution frequency, and expected
   leverage. If the old sensitivity map is non-discriminating, explicitly add
   cheaper discriminators rather than pretending it ranked organs.

2. Generate at least four materially different, falsifiable representation
   hypotheses and rank them by information gain and physical cost. At minimum
   consider and measure cheap discriminators for:

   - expert shared basis plus per-expert low-rank/delta coefficients;
   - routing-aware hot/cold expert treatment, heterogeneous precision, and
     selective expert residuals/drop controls;
   - mixed per-organ/per-expert precision with sensitive router/attention/
     shared-expert protection and globally allocated error;
   - transform/dictionary/codebook or generated-coefficient representation with
     explicit residual accounting;
   - a composed route (for example shared basis + mixed precision + sparse or
     generated residual) if the discriminators support it.

   Do not create branded subsystems. These are Gravity experiments. Do not
   spend all candidates on one knob. Every candidate must state its mechanism,
   complete byte accounting, expected effect, cheapest falsifier, and whether
   the representation has a valid native/streamed execution path or only static
   accounting.

3. Run a bounded but genuinely deep Gravity search on the strongest surviving
   hypothesis. Use the existing HCLI Gravity/search machinery where possible;
   extend it through HCLI if it cannot express the needed candidate. Measure
   components before compositions and then search combinations. Allocate bits
   globally by measured leverage rather than equal fidelity. Target the expert
   organ first because it dominates bytes, but protect/control routing,
   attention, shared experts, and modality-sensitive paths.

4. Required controls for every serious candidate:

   - complete bytes, metadata, headers, corrections, router state, and all
     representation-attributable state in the denominator;
   - native or valid streamed execution with activation/state continuity, not
     static shard inspection mislabeled as execution;
   - independent verifier and capability battery, including multimodal or
     route-relevant cases where available;
   - measured magnitude adequacy; retain the permanent 0.01*W
     magnitude-destroying negative control and reject it even if direction
     similarity is high;
   - measured physical utilization, prefill/decode or wall cost, and resident
     memory behavior;
   - checkpoint and NR release evidence for each candidate/session.

   If a candidate reaches <2.5, <=2.0, <=1.5, or <=1.0, immediately run
   Odyssey-III attacks before any promotion: ablations, hidden cases,
   anti-transfer, restart/replay, corrupted state, stronger oracle or alternate
   seed where available, mutation attacks, and the 0.01*W control. A small
   artifact that loses the organism is not a success.

5. Record new Laws only from adequate cross-candidate evidence and record Scars
   for failed assumptions. Cross-check the O003 result against O001/O005 and
   ask whether any pattern is compound across specimens (expert-basis behavior,
   sensitive organs, quantization failures, or utilization). Do not promote a
   cross-specimen claim from one receipt.

6. Seal the complete O003 session record with Odyssey-I findings, Doctor
   anatomy, Gravity trajectory, best complete EBPW, retained capability,
   accepted/rejected learning, Laws, Scars, utilization, execution method,
   checkpoint state, and future reopen conditions. Release NR/resident memory
   and verify memory return before moving on. Leave a durable handoff naming
   the exact next WorkUnit if the target is not reached.

## Terminal/verifier rules

TARGET_HIT requires measured complete EBPW <=1.0, complete accounting,
capability survival, complete execution, an independent verifier, magnitude
adequacy, and validated utilization. PROVEN_UNABLE requires a real measured
bound with limiting mechanism, assumptions, search region, and reopen
condition. Otherwise report BUDGET_EXHAUSTED with the exact representation
level exhausted and do not imply broader inability. Never promote an external
candidate or a static-only receipt to NX.

Use the available machine time aggressively and use long, explicit search-driver
prompts if useful: ask for competing explanations, 20 hypotheses where it buys
information, self-falsification, cheaper discriminators, compound compression,
and experiments ranked by information gain. Record prompt/context and wall
cost. Fix implementation gaps through HCLI; Codex must not become the
permanent manual executor.
