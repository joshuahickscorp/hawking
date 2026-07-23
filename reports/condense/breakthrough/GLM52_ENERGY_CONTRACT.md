# GLM-5.2 energy measurement contract (mandate section 10)

Sealed 2026-07-23T01:39:02Z on branch `campaign/glm52-inference-breakthrough`.
Machine of record: Mac15,14 / Apple M3 Ultra / 28 CPU cores / 60 GPU cores / 96 GiB / macOS 27.0 (26A5378j).
Companion artifact: `GLM52_ENERGY_CONTRACT.json` (authoritative; this file is the readable summary).

## Verdict: IOREPORT

Energy is measurable on this machine today, unprivileged, but only in one domain.

The IOReport `Energy Model` group is readable as user `scammermike` with no sudo and no
entitlement. It enumerates 565 energy channels. **Exactly one is populated:**

```
Energy Model ::  :: GPU Energy (nJ) = 1.62W
```

Everything else reads zero:

| Domain | Status | Evidence |
|---|---|---|
| GPU | AVAILABLE | `GPU Energy` (nJ), nonzero on 60/60 samples |
| CPU | UNAVAILABLE | 565 DIE_0/DIE_1 x EACC/PACC cluster+core+SRAM channels all 0.00; `cpu_power` 0.0 on 60/60 |
| ANE | UNAVAILABLE | `ANE0_0`, `ANE0_1` = 0.00 |
| DRAM | UNAVAILABLE | `DRAM0_0`, `DRAM0_1` = 0.00 |
| GPU SRAM | UNAVAILABLE | `GPU SRAM0_0`, `GPU CS SRAM0_0` = 0.00 |
| Package total | UNAVAILABLE | no populated total channel |
| Wall AC | UNAVAILABLE | no meter, no UPS, no battery |

The CPU zeros are not a sampling artifact. Those channels are mJ-granular; over a 1.13 s
sample at load average 20.85 with a busy P-cluster, even 5 W would deposit ~5600 mJ.
They are simply not populated on this part and this OS build.

**`all_power` is a trap.** macmon computes it as cpu+gpu+ane. With cpu and ane at zero it
was numerically identical to `gpu_power` on all 60 samples. It is not a system total and
must never be reported as one.

## What was tested

**powermetrics** — present (`/usr/bin/powermetrics`, root:wheel, 341056 bytes, Jul 3).
`--help` runs fine unprivileged. Sampling does not:

```
$ /usr/bin/powermetrics -n 1 -i 1000 --samplers cpu_power
powermetrics must be invoked as the superuser
```

No sudo was run. No password was requested, printed, or inferred.

Its own help text caps the value of ever unlocking it:

> "Average power values reported by powermetrics are estimated and may be inaccurate -
> hence they should not be used for any comparison between devices, but can be used to
> help optimize apps for energy efficiency."

**asitop** — present at `~/.local/bin/asitop`, but it is a powermetrics wrapper and shells
out to sudo:

```
sudo: a terminal is required to read the password; ...
sudo: a password is required
```

**istats** — not installed. **macmon 0.7.2** — installed at `/opt/homebrew/bin/macmon`,
sudoless, IOReport-backed. This is the only working energy reader on the box.
A scan of `/opt/homebrew/bin` (928 entries) for `power|energy|watt|stat|mon|therm|smc|sensor`
returns only `dbus-daemon`, `dbus-monitor`, `h5stat`, `llama-lookup-stats`, `macmon`,
`suitesparse_mongoose`. `/usr/local/bin` has nothing. Python side: `pyRAPL`, `codecarbon`,
`zeus`, `psutil` all ABSENT from the glm52 venv (and pyRAPL is Intel-RAPL only anyway).

**External meter** — none. `system_profiler SPUSBDataType` returns **0 bytes** (no USB
devices at all, so no UPS and no HID power device). `ioreg -c IOPMPowerSource` has no
power-source node. `pmset -g ps` says `Now drawing from 'AC Power'`.

## The one number that is not what it looks like

macmon also emits `sys_power`, 56-110 W. It is **not** a sum of any Energy Model channel
(those read zero while it reads 59 W), so it comes from somewhere else, most likely an SMC
power key. **This probe did not confirm which key.** It is recorded as
`AVAILABLE_BUT_UNVALIDATED` and may be used as a relative indicator only. It is not a
measured joule until someone puts a wall meter inline.

## Measured noise, and why the baseline is blocked

Two windows, minutes apart, same machine, live campaign running throughout:

| window | n | interval | `gpu_power` W (min/med/max, sd) | `sys_power` W (min/med/max, sd) |
|---|---|---|---|---|
| A | 20 | 500 ms req / 622 ms actual | 1.112 / 1.559 / 26.337, **sd 10.517** | 55.90 / 77.44 / 110.64, **sd 19.85** |
| B | 40 | 1000 ms req / 1134 ms actual | 1.295 / 1.740 / 2.205, **sd 0.196** | 56.30 / 57.96 / 71.96, **sd 3.13** |

The noise floor is **not stationary**. Same machine, same reader, and the GPU standard
deviation moved by 54x between windows. Detection floor for now is the worse of the two:
**10.52 W at 1 sigma.**

Requested 1000 ms, got a 1134 ms median grid — 13.4% overhead, non-uniform. Power itself is
unbiased by this (the reader divides by actual elapsed), but **no energy sample can be
attributed to a code phase shorter than ~1.2 s.** The campaign's kernel calls are 0.21-0.51 ms.
That is three orders of magnitude below the grid; single-call energy is impossible and always
will be with this instrument.

**Idle baseline: UNAVAILABLE, and this is a blocking condition, not a caveat.**
At seal time: load average 20.85 / 27.74 / 29.63, six Python processes at 411.9%, 210.8%,
197.1%, 190.3%, 98.0%, 84.1% CPU, GPU Device Utilization 13-16%, 1.42 GB swap in use.
Five of those are the live campaign and must not be touched. There is no idle to measure.
The correct response is to **wait**, never to subtract an assumed idle.

Baseline unblocks only when: all five campaign processes are gone, load average below 1.0
for 300 consecutive seconds, GPU Device Utilization below 2%, then a 600 s capture at 1 Hz.

## What gets logged instead

None of these is energy. None may be converted into energy.

| Observable | Exact source | Status |
|---|---|---|
| wall time | `time.perf_counter_ns()`, impl `mach_absolute_time()`, resolution 4.1667e-08 s, **measured min nonzero delta 41 ns** over 1869 samples | MEASURED |
| bytes moved | DERIVED from shapes: `ndarray.nbytes`, Metal buffer sizes. No hardware counter. `getrusage` returns `ru_inblock=0 ru_oublock=0` on Darwin and must not be used | DERIVED |
| executed operations | DERIVED from shapes and loop counts in the harness | DERIVED |
| dispatch counts | DERIVED from the harness; anchored to measured 215.8 us command-buffer fixed cost and 0.71 us marginal dispatch | DERIVED |
| thermal state | macmon `temp.cpu_temp_avg` / `temp.gpu_temp_avg`; raw SMC keys (`TCMz` 81.8, `TCMb` 74.6, `TVD0` 73.5, `TCDX` 64.9); IOHID `PMU tcal/tdev1-8/tdie1-10`; `pmset -g therm` | MEASURED |
| GPU utilization | `ioreg -c IOAccelerator -r -d 1` -> `PerformanceStatistics` -> `Device Utilization %` (13-16), Renderer (12-15), Tiler (13-16) | MEASURED |
| P-state residency | IOReport `CPU Stats` (46 ch) and `GPU Stats` (`GPUPH`: OFF 157882, P1 22280, P3 2443593) | MEASURED |
| memory / swap | macmon `memory{ram_total, ram_usage, swap_total, swap_usage}` | MEASURED |

Two notes worth carrying forward. `pmset -g therm` reports *"No thermal warning level has
been recorded"* and the same for performance warnings and CPU power status — so there is no
throttle log to correlate against; thermal state gates run validity and never feeds a number.
And `IOAccelerator` reports `GPUConfigurationVariable num_cores=80` while `gpu-core-count=60`;
80 is the die-config maximum, 60 is this part, matching ground truth.

The `bytes_moved` figure must be the **true device-read** figure, not the naive one:
with `groups=8` and `stage_x` on, true device reads are 1,785,856 B (7.10% of BF16 dense),
while `gravity_metal.py:258-262` under-reports 1,572,864+2048 (6.26%) by ignoring
per-threadgroup re-reads of the codebook and of x. Log the true figure and name the source line.

## Forbidden (mandate 10)

- No joules from a utilization percentage. Not `cpu_usage_pct`, not `Device Utilization %`.
- No joules from a P-state residency histogram times a voltage table. Residency x voltage is a model.
- No joules from a temperature. Not an SMC key, not `PMU tdie`, not `cpu_temp_avg`.
- No reporting `all_power` as a system total. On this box it *is* `gpu_power`.
- No reporting `sys_power` as measured energy until a wall meter validates it.
- No filling an UNAVAILABLE domain with a vendor TDP, a datasheet figure, or another Apple part.
- No estimating a dark domain by difference (`cpu = sys - gpu`); `sys` provenance is unverified, so the difference is undefined.
- No carrying a noise floor forward from a previous session. Sigma is re-measured in-session.

## Protocol, if and when energy becomes usable

Status: **DEFINED_BUT_BLOCKED** on five conditions.

- **B1 LIVE_CAMPAIGN_ACTIVE** — blocks the idle baseline and therefore every delta defined against it. Blocking, not a caveat.
- **B2 NO_CALIBRATION** — blocks every absolute joule claim, every cross-machine claim, anything past 2 significant figures. A/B deltas of identical workload shape inside one thermal window survive.
- **B3 SUB_GRID_WORKLOAD** — blocks per-call energy. Clears by construction once work is looped into a >= 30 s band.
- **B4 CPU_DRAM_DOMAINS_DARK** — blocks any total-energy-per-token claim and any claim about a CPU-side fix (e.g. the `glm52_pack.py:61` unpack_indices 36.6 ms / 190.6 MiB defect cannot be energy-justified). GPU-domain claims unaffected.
- **B5 GPU_PHASE_NOT_AUTHORIZED** — this phase is CPU-only, and the GPU is the only lit domain. The only measurable target is out of scope right now.

Steps once clear:

1. **Preflight** — assert B1..B5 clear; record load average, GPU utilization, swap, all thermal sources. If any block holds, abort and record the abort. Never degrade to an estimate.
2. **Idle baseline** — 600 s at 1 Hz, benchmark not started. Retain every sample. This is the subtrahend and it is measured, never assumed.
3. **Warmup** — 120 s of the workload, discarded. Code, cache, allocator, clock ramp.
4. **Thermal steady state** — continue until `cpu_temp_avg` and `gpu_temp_avg` each drift under 0.5 C across a 120 s window. Record entry temperatures. A run whose entry temperature differs from its pair's by more than 1.0 C is void.
5. **Paired repetitions** — interleave **ABBA ABBA**, never blocked AAAA BBBB, so drift cancels to first order. Minimum 8 blocks per arm, each >= 30 s of sustained work. Report the paired per-block delta distribution, not two independent means.
6. **Matched output contract** — identical input tensor, identical shapes (`[2048,6144]` gate/up, `[6144,2048]` down), identical row count, identical boundary dtype, plus a recorded parity figure. The custom kernel's parity is **2.1e-4**, caused by `gravity_metal.py:202` casting the codebook to fp16. That 2.1e-4 is part of the contract and is restated with every energy claim, never dropped.
7. **Report** — joules per matched unit with measured idle subtracted, the 3-sigma band, block count, entry temperatures, parity figure. Two significant figures. Raw JSONL attached.

Significance rule: a delta is reportable only if it exceeds **3 sigma** of the concurrently
measured baseline, with sigma measured inside the same session.

### Matched-quality rule (non-negotiable)

**A lower-quality model or kernel may not claim an energy win against a higher-quality one.**

A comparison is admissible only against a fixed matched-quality point: either bit-identical
output, or both arms driven to an identical pre-registered quality target on an identical
eval set, with the quality receipt sealed *before* the energy run. If the cheaper arm cannot
reach the target, the report is "cheaper arm does not reach the quality point" and **no energy
number is published at all.**

Concretely for this campaign: R0 (`packed_bpw` 0.87633) is **not** quality-matched to BF16
dense, so no joules-per-token comparison between them may be published under any framing until
a matched-quality receipt exists. The point is currently moot in the other direction too — the
custom kernel is *slower* than dense fp16 (0.727x at down, 0.329x at gate/up), so an energy win
would be doubly unsupported.

One disclosure that must ride along with any energy-per-byte figure: the Metal kernel uploads
8-bit indices while R0 bills 7-bit, so the executed index stream is **14.3% fatter** than the
artifact. State which of the two is being billed.

## Upgrade path (user action only)

Unlocking powermetrics would light the CPU / ANE / DRAM domains. It would **not** produce
calibration — powermetrics says its own numbers are estimates unsuitable for cross-device
comparison, so B2 would still hold.

The single command, **for the user to run themselves, in their own terminal**:

```
sudo /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 5
```

I will not run it. I did not run any sudo command during this probe, and I will not request,
print, store, or infer a password.

## Retention

Raw JSONL, one sample object per line, verbatim. **Aggregation before storage is forbidden** —
medians and standard deviations are computed at report time from retained lines, never in place
of them. Destination `reports/condense/breakthrough/energy_raw/<run_id>.jsonl`, **not yet
created**: no benchmark run has been energy-logged. An energy claim without its raw log is void.

The samples backing this document live in the session scratchpad (`macmon20.jsonl`,
`macmon40.jsonl`, `macmon_debug.txt`) and are **not durable**. They are evidence for this
contract, not campaign artifacts.

## What this probe did not do

No sudo. No password requested, printed, or inferred. No Python file created or edited.
No process signalled, paused, or killed — PID 59093 and the four `mop.temporal.runs.e2`
workers were observed via `ps` only. No `launchctl` or `com.hawking.*` job touched. Nothing
under `/Users/scammermike/Downloads/hawking` modified. Nothing in
`/Users/scammermike/Desktop/GLM52-Gravity-SubBit` opened, written, moved, or renamed.
No GPU or Metal work.
