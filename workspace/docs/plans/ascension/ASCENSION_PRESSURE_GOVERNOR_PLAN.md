# Ascension Pressure Governor Plan

**Bible:** HAWKING_ASCENSION_BIBLE.md §27  
**Status:** plan + scaffold (gated on Proto-Frankenstein offload)  
**Scaffold:** `workspace/ops/ascension/pressure_governor.py`, `signals.py`  
**Tests:** `workspace/ops/ascension/tests/test_pressure_governor.py`

---

## What tonight's code already proves

| Live / lab file | Proven capability | Scaffold mapping |
|-----------------|-------------------|------------------|
| `v0_notifier.py` → `free_gib()` / `shutil.disk_usage` | Disk free GiB, floor alerts (`FLOOR_GIB = 25`) | `signals.free_disk_bytes`, disk thresholds (yellow 40 / red 25 / critical 12) |
| `v0_notifier.py` → `gpu_pct()` via `ioreg -c IOAccelerator` | Live Metal GPU Device Utilization % | `signals.parse_gpu_util_ioreg` |
| `lab/operators/glm52_grounding.py` → `parse_darwin_memory` | `vm_stat` + `sysctl hw.memsize` + `vm.swapusage` | `signals.parse_vm_stat` / `parse_swapusage` |
| `lab/operators/bounded_cache.py` → `available_ram_bytes` / `free_disk_bytes` | Fail-soft memory + disk probes | same signal surface |
| `reclaim_storage_keep_proto.py` free-space print | Prove recovery after reclaim | governor CRITICAL → recommend reclaim path |
| `crates/hawking-speculate/src/governor.rs` | Hysteresis (asymmetric thresholds + dwell) | `PressureGovernor` escalate-immediate / deescalate-dwell |

Session precedent: `ioreg`, `vm_stat`, and `df` / `shutil.disk_usage` were used all night for operational safety. The governor codifies those probes into a state machine rather than ad-hoc floor checks.

---

## Levels and actions (bible §27)

| Level | Campaign behaviour |
|-------|--------------------|
| **GREEN** | full campaign, normal downloads, normal residency, normal concurrency |
| **YELLOW** | reduce sessions, batch reviews, pause new large downloads, shrink KV/session count |
| **RED** | checkpoint models, unload reviewer, evict leased cache, pause heavy benchmark, preserve active receipt |
| **CRITICAL** | unload target model, stop downloads, preserve rollback + evidence, return resources to user, emit urgent report |

---

## Signal sources (real)

| Bible input | Command / API | Parser |
|-------------|---------------|--------|
| Disk floor | `shutil.disk_usage("/")` (≡ `df`) | `free_disk_bytes` |
| Unified-memory pressure | `sysctl -n hw.memsize` + `vm_stat` | `parse_vm_stat` → available RAM + pressure ratio |
| Swap | `sysctl -n vm.swapusage` | `parse_swapusage` |
| GPU contention | `ioreg -r -d 1 -c IOAccelerator` | `parse_gpu_util_ioreg` (max Device Utilization %) |
| Thermal throttling | `pmset -g therm` | `parse_thermal_pmset` (`CPU_Speed_Limit < 100`) |
| Foreground user activity | injectable probe (`lsappinfo` / osascript later) | `foreground_user_active` → YELLOW yield |

Unknown optional sensors **do not escalate** (fail-open on missing thermal/GPU). Hard floors (disk, RAM, swap) escalate when measured.

---

## Default thresholds (aligned with tonight)

```text
disk free GiB:   YELLOW < 40   RED < 25   CRITICAL < 12
RAM free GiB:    YELLOW < 12   RED < 6    CRITICAL < 2.5
RAM used ratio:  YELLOW ≥ 0.80 RED ≥ 0.90 CRITICAL ≥ 0.96
swap used GiB:   YELLOW ≥ 2    RED ≥ 8    CRITICAL ≥ 16
GPU util %:      YELLOW ≥ 85   RED ≥ 95
thermal:         any throttle → RED
foreground user: active → YELLOW (yield), never sole CRITICAL
```

Thresholds are a `GovernorThresholds` dataclass — override in tests or future policy files.

---

## State machine

```text
evaluate_pressure(signals) → worst-of(disk, ram, swap, gpu, thermal, foreground)

PressureGovernor.step(signals):
  if proposed.rank > current.rank  → escalate immediately
  if proposed.rank < current.rank  → require deescalate_dwell consecutive samples
  if proposed.rank == current.rank → clear pending dwell
```

Hysteresis prevents flapping around the disk floor (the exact failure mode of a single threshold bouncing 24↔26 GiB).

---

## Integration plan (post offload)

1. Supervisor poll loop (not a detached daemon in this scaffold) samples `collect_host_signals()` every N seconds  
2. On level change → `notifications.build_notification(MEMORY_DISK_PRESSURE, ...)`  
3. RED → request garbage ecosystem to reclassify expired LEASED caches as EVICTABLE candidates  
4. CRITICAL → stop downloads, preserve rollback/evidence, surface `HUMAN_DECISION_REQUIRED` if user resources cannot be returned automatically  
5. Do **not** auto-unload models until Frankenstein offload + sealed supervisor wiring  

---

## Non-goals (this scaffold)

- No model unload / process kill  
- No download cancellation  
- No detached launchd plist  
- No mutation of `v0_notifier.py` disk floor  

---

## Remaining work

- [ ] Policy file for thresholds per machine profile (M3 Pro 18 GB vs larger)  
- [ ] Wire `foreground_user_active` to a real macOS frontmost-app probe  
- [ ] Aggregate sustained GPU util (windowed) not single-sample spikes  
- [ ] Connect RED `evict_leased_cache` → garbage ecosystem  
- [ ] Connect CRITICAL → notification bus + campaign pause receipt  
