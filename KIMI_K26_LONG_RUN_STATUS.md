# Kimi K2.6 Long-Run Status

- Status: **STOPPED_DISK_FLOOR**
- Started: `2026-07-21T16:52:21Z`
- Updated: `2026-07-21T18:16:14Z`
- Wall-clock managed: `5033.3s`
- Experiments completed: `13`
- Active experiment: `none`
- Current best: `P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY` / `0.9085909525553385` BPW
- Primary diagnosis: `UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER`
- Next experiment: `AFTER_DISK_RECOVERY: NONLINEAR_REPRESENTATION_SIDE_STRUCTURAL_ALLOCATION_F0_F1`

## Guards

- Controller PID/lease: `13332` / `True`
- Heartbeat age: `2.289114` seconds
- Free disk/headroom: `39.25` / `-42.75` GiB
- MOP protected: `True`
- Sole Kimi source: `True`

## Latest result

```json
{
  "decision": "STOP_EARLY_PRESERVE_SOURCE_CONTROLLER_AND_MOP",
  "deficit_bytes": 45904011264,
  "event": "LR14_DISK_FLOOR_POLICY",
  "floor_increase_bytes": 53687091200,
  "free_disk_bytes": 42142818304,
  "guard_failure": "DISK_FLOOR_RISK",
  "new_floor_bytes": 88046829568,
  "old_floor_bytes": 34359738368,
  "region_closure_decision": "TESTED_LINEAR_REPAIR_REGION_CLOSED",
  "region_closure_seal_sha256": "4a599fab04e1f6ba758c4314bd9e3cb9354b4205e2ab4d754e87ab5b2440282d",
  "telegram_delivered": true,
  "telegram_receipt_seal_sha256": "5674951e3b0a9d6343056c564ee50396062993a4192ed4ca8ee7b10ebbd18ccf"
}
```
