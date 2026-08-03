# D-SEAM-COLLAPSE-V1 — builder report (in-tree)

**Rung:** D-SEAM-COLLAPSE-V1  
**Base:** `bd3c04fc6a2c92f92ea5d64a0c5707a52e8bedce` — Release residual historical evidence and densify retained pins  
**Kind:** Honest **elimination-only** product dual-authority collapse  
**External full report:** `/private/tmp/HAWKING_D_SEAM_POST_AX_BUILDER_20260730.md`

## What changed

1. **Sole product SubmitTurn** is `run_turn_core` (via `generate_submit_turn` / `generate_and_publish`).
2. **Deleted** incomplete default-off `HIDE_KERNEL_TURN` product branch:
   - `kernel_turn_enabled`, `DEFAULT_KERNEL_TURN_MAX_STEPS`
   - `run_turn_kernel` and exclusive helpers (answer derive, approval announce/resolve, kernel-only context_manifest publisher)
   - dual branch in `spawn_submit_turn_generation`
3. **Preserved** AgentKernel, `build_turn_kernel` / `build_fleet_kernel`, `turn_kernel_autonomy`, fleet `KernelRunLauncher`, Context OS, objects, automations, lenses, ACP, serve.
4. **Memory authority clarified:** host `MemoryLedger` is intent/KV projection only; classed forget/export remains sole.
5. **Not done (honest):** host_support_N semantic absorption (risk of megafile / relocated=0); `connector_abi_impls.rs` delete blocked without editing non-owned `connector_abi.rs`.

## Accounting note

Preferred complete-active delta was −2000..−5500. Full densify rewrite of host_support/host_cmds was **not** certified under measured S4 rates; contract prefers smaller honest elimination-only. Source diff before apparatus: **−615 lines** (59 insertions / 674 deletions on product sources). Final six-bucket + topology from committed-tree instruments in companion JSON.

## Gates (summary)

| Gate | Result |
|------|--------|
| hide-backend tests | PASS |
| sibling HIDE crates + hawking-context | PASS |
| cargo check --workspace | PASS |
| case_extract --check | PASS 4623 |
| blackbox --only-runnable | PASS 86/86 |
| generation audit | PASS 0 generated |
| HIDE interaction PERF | UNAVAILABLE (not claimed) |

## Six buckets (pre-measure sketch)

| Bucket | Value |
|--------|------:|
| eliminated | product dual-turn branch LOC (measured post-commit) |
| rewritten | 0 |
| generated | 0 |
| relocated | 0 |
| facade | 0 |
| added_apparatus | control/D-SEAM-COLLAPSE-V1-* only |
