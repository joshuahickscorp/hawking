# HAWKING PARALLEL CONTINUATION STATUS

Generated from live evidence by `tools/campaign/parallel_status.py`. Do not hand-edit.
Ownership and the DAG are hand-authored architecture and are read, never written, by this tool.

    at:       2026-07-27T01:42:43Z
    endpoint: RAMANUJAN_SANDBOX_READY (not reached)
    width:    6/6 lanes running
    fences:   ODYSSEY_LAUNCH_AUTHORIZED=False RAMANUJAN_RESEARCH_AUTHORIZED=False HIDE_KERNEL_TURN=False

## Running

- `L01` hide-archaeology-v2 (grok) -- outputs missing: HIDE_ARCHAEOLOGY_V2.md, HIDE_ARCHAEOLOGY_V2.json
- `L02` hide-memory-six (grok) -- outputs missing: HIDE_MEMORY_CLASSES.json
- `L03` odyssey-t0 (grok) -- outputs missing: ODYSSEY_T0_RECEIPT.json, ODYSSEY_CONTRACT_CLOSURE.json, ODYSSEY_FEASIBILITY.json
- `L04` fabric-software (grok) -- outputs missing: FABRIC_SOFTWARE_STATUS.json, FABRIC_QUALIFICATION_LADDER.json
- `L05` bridge-events-adapters (grok) -- outputs missing: HAWKING_CANONICAL_EVENTS.json, HAWKING_ADAPTER_REGISTRY.json, HAWKING_BRIDGE_SURFACE.json
- `L06` consolidation-inventory (grok) -- outputs missing: HAWKING_CONSOLIDATION_INVENTORY.md, HAWKING_CONSOLIDATION_INVENTORY.json

## Finished, awaiting controller review

- none

## Queued

- `L07` speculation-safety (grok) -- outputs missing: HIDE_SPECULATION_SAFETY.json
- `L09` ramanujan-migration-prep (claude)

## Blocked (real data dependencies, see the DAG)

- `L08` hide-os-wiring (grok)
- `L10` odyssey-t1-t7 (unassigned)
- `L11` math-frozen (unassigned)
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

    6 lanes running at width 6/6; do not poll, wait on a milestone

```bash
~/.claude-grok/bin/grok-run wait
```
