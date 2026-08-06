# Odyssey Launch Packet

Odyssey is **prepared and not started**. `ODYSSEY_LAUNCH_AUTHORIZED` is `false`.

## What is ready

- the full package tree under `odyssey/`, content-addressed where the artifacts exist
- training plan T0-T5, objective contract, checkpoint contract, evaluation contract
- sandbox policy: network denied by default, filesystem allowlisted, one heavy lane
- the nine Ramanujan roles plus Adversary, Tribunal and verifier, with promotion rights
  held only by verifier events and the Tribunal
- the four-tier verification lattice, seven memory stores, branch economics, Graveyard
- Lean and Mathlib pinned; a Tier-3 proof that needs a different Mathlib is a different proof

## Odyssey input

`GLM-5.2-H0.98-Math-Preserve.gravity` is complete, content-addressed, and selected
as the mandatory training substrate.  Its PASS3 receipt verifies complete official
tensor coverage, every shard hash, the frozen per-tensor allocation, and actual
whole-package compliance with the one-bit law.


## To start Odyssey in the next session

Authorize deliberately, then run the stage runner:

```bash
printf 'true\n' > workspace/campaign/governance/odyssey/program/launch/ODYSSEY_LAUNCH_AUTHORIZED && python3.12 workspace/campaign/governance/odyssey/program/training/run.py T0
```

To halt any running loop at the next checkpoint boundary:

```bash
touch workspace/campaign/governance/odyssey/program/launch/STOP
```
