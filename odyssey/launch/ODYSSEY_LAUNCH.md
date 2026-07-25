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

## What is not ready, and why

The Odyssey training substrate is the compact Math artifact Prometheus selects, and that
selection is gated on the flagship traversal (gate M11). Odyssey trains that artifact and
nothing else; substituting an available model would be a different experiment wearing this
one's name.

## To start Odyssey in the next session

Authorize deliberately, then run the stage runner:

```bash
printf 'true\n' > odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED && python3.12 odyssey/training/run.py T0
```

To halt any running loop at the next checkpoint boundary:

```bash
touch odyssey/launch/STOP
```
