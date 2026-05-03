# Hardware audit — published numbers reference

Every benchmark number in dismantle's README, launch post, or
`dismantle bench` output is produced on the configuration recorded
here. Update this file in the same PR as any change to the
published-numbers hardware.

## Hardware

| Field | Value |
|---|---|
| Machine | MacBook Pro (Mac15,7), Model Number MRW13LL/A |
| CPU | Apple M3 Pro, 11-core |
| GPU | Apple M3 Pro, 14-core |
| Unified RAM | 18 GB |
| Memory bandwidth (theoretical peak) | 150 GB/s |
| Storage | TODO (capture from `system_profiler SPSerialATADataType`) |

## Software (Phase 0 baseline — update as deps land)

| Field | Value |
|---|---|
| OS | TODO — capture `sw_vers -productVersion` + `-buildVersion` on next bench run |
| Xcode CLT / Metal SDK | TODO — `xcrun --show-sdk-version` |
| Rust | TODO — `rustc --version` |
| llama.cpp comparison commit | TODO — auto-captured by `tools/competitors/run_competitors.sh` into `tools/competitors/versions.json` |
| dismantle | 0.0.1 |

The competitor versions are auto-pinned by
`tools/competitors/run_competitors.sh` into
`tools/competitors/versions.json` (gitignored). Numbers cited in
[docs/competitive_audit.md](competitive_audit.md) reference that
file.

## Thermal policy for benchmarks

- Benchmarks run with the laptop on a hard surface, lid open, power
  adapter connected.
- Between benchmark suites: 5 minutes idle to return to baseline
  thermals.
- Each suite reports first-trial AND median-of-three numbers so
  thermal degradation is visible.
- macOS Low Power Mode is *off* during benchmarks.

## Bandwidth target

The North Star metric: **measured GB/s on the weight read of the MoE
block, divided by 150 GB/s theoretical peak**. A 90% number means
there is no further perf to extract on this hardware without changing
the math, only the algorithm. dismantle Phase 5 publishes this number
for every model.

## What changes invalidate the published numbers

- Any change to RAM, GPU core count, or CPU core count → invalidates.
- macOS major version update → re-run before claiming.
- Metal SDK major version → re-run before claiming.
- llama.cpp comparison commit advance → numbers re-published.
- dismantle Phase change → numbers must be re-published with the new
  dismantle version annotation.

When numbers are re-run, append a dated section under "History"
rather than overwriting; old numbers remain interpretable in the
context of the dismantle version that produced them.

## History

(empty — Phase 0 hasn't published any numbers yet)
