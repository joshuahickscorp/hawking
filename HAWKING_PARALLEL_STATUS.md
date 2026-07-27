# HAWKING PARALLEL CONTINUATION STATUS

Generated from live evidence by `tools/campaign/parallel_status.py`. Do not hand-edit.
Ownership and the DAG are hand-authored architecture and are read, never written, by this tool.

    at:       2026-07-27T03:07:26Z
    endpoint: RAMANUJAN_SANDBOX_READY (not reached)
    width:    0/6 lanes running
    fences:   ODYSSEY_LAUNCH_AUTHORIZED=False RAMANUJAN_RESEARCH_AUTHORIZED=False HIDE_KERNEL_TURN=False

## Running

- none

## Finished, awaiting controller review

- none

## Integrated by the controller

- `L01` hide-archaeology-v2 (grok)
- `L02` hide-memory-six (grok)
- `L03` odyssey-t0 (grok)
- `L04` fabric-software (grok)
- `L05` bridge-events-adapters (grok)
- `L06` consolidation-inventory (grok)
- `L18` gravity-serve (grok)
- `L07` speculation-safety (grok)
- `L09` ramanujan-migration-prep (claude)
- `L19` odyssey-data (grok)
- `L20` memory-writers (grok)
- `L21` gravity-degenerate-attribution (claude (controller, run inline -- Grok slots held by a concurrent campaign; user authorized))
- `L22` odyssey-readiness-supersession (claude)
- `L23` representation-escalation-preregistration (claude)
- `L24` ramanujan-governance-and-cognition (claude)
- `L25` ramanujan-environment-lock (claude)

## Queued

- `L08` hide-os-wiring (grok)

## Blocked (real data dependencies, see the DAG)

- `L12` consolidation-execute (unassigned)
- `L13` ramanujan-migrate (unassigned)
- `L14` ramanujan-cognition (unassigned)
- `L15` local-forge-training (unassigned)
- `L16` search-governance (unassigned)
- `L17` q0-q6-qualification (unassigned)

## Hard walls on the critical path

- **L10**: training a 92 GB artifact on a 96 GB machine that is already at load ~29 of 28 cores because a separate campaign (MOP) holds most of them
- **L10**: ODYSSEY_LAUNCH_AUTHORIZED is false and only the controller may flip it, after G1-G4 of ODYSSEY_PROMOTION_GATE.md
- **L15**: F0-F9 local training requires the frozen Math-Frozen Director, which is downstream of the L10 wall

## Next action

    launch L08 (hide-os-wiring) -- a slot is free

```bash
# see HAWKING_PARALLEL_LANE_OWNERSHIP.json for the lane's contract
```
