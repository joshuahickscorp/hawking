# Gravity gauntlet cost — target-seeking search, not a static plan

`ODYSSEY_48H_PLAN.md` priced Odyssey I as a census plus a light probe battery:
read each specimen once, run a short prompt battery, done. The operator has
corrected that (2026-09-05): every specimen must be driven through a gravity
**gauntlet** toward sub-bit complete EBPW — a search, not one sample — and
the schedule below prices that search instead.

**Bottom line up front: the honest total is UNKNOWN, not a number.** Every
gravity mix this repo has a receipt for went through two stages — build
(`mlx_lm.convert`, quantize the weights) then probe (run a prompt battery
against the result) — and only the second stage has a timestamp anywhere.
Everything below that says "known" is I/O + probe only; it is a **lower
bound**, and the missing build stage is very likely the larger of the two for
anything past a 7B model. Where a figure can't be measured it says UNKNOWN.
It is not filled in with a guess.

Source of truth: `python3 tools/odyssey/lake_phases.py --gauntlet` (byte
totals are the live lake, same as `ODYSSEY_48H_PLAN.md`) combined with the
evidence in `tools/odyssey/lake_phases.py`'s `GRAVITY_PROBE_WALL_S` /
`GRAVITY_BATTERY_MIXES` constants, which `test_lake_phases.py` recomputes
live from the raw receipts so this document can't quietly drift from them.

## 1. What does one gravity mix cost today?

**MEASURED (partial): probe time only.**
`receipts/odyssey-i/O0{01,03,04,05,06}_GRAVITY_*.json`, field
`doctor.wall_s`, 108 completed mixes across 5 specimens (Falcon-H1-7B,
Kimi-VL-A3B, Mistral-Small-3.1-24B, Qwen3-30B-A3B, Qwen3-VL-30B-A3B — 7B to
30B params, tiers A_tiny/B_mid):

| | value |
|---|---|
| n | 108 |
| min | 6.271 s |
| median | 15.986 s |
| mean | 16.363 s |
| max | 45.579 s |

**UNKNOWN: the build step.** `tools/odyssey_patient_runner.py`'s
`convert_gravity()` calls `mlx_lm.convert(...)` to produce each mix (reads
the full canonical bf16 weights, quantizes, writes the result) *before* the
probe battery runs. No receipt field, no log line in `drive.log` /
`frontier.log`, and no code path in `convert_gravity()` brackets that call
with a timer. Grepped for any GB/s or seconds figure tied to `mlx_lm.convert`
or "quantize throughput" anywhere in `receipts/` and `tools/` — nothing.
**One full gravity-mix cost = UNKNOWN(build) + ~16.4 s(probe, measured).**
Do not multiply the probe number by a specimen count and call the result the
gauntlet cost; that would be reporting an estimate as a measurement.

## 2. How many mixes does a search plausibly need?

**MEASURED: a real battery already ran, on 5 specimens, and it did not find
sub-bit.** The same 108 receipts are one grid battery per specimen — bits in
{2,3,4} × group-size in {32,64,128} × an attn/mlp- or expert-protected
variant, plus 3 mixed-bit combos:

| specimen (oxx) | model | mixes tried | best complete_bpw found |
|---|---|---|---|
| O001 | Falcon-H1-7B-Instruct | 23 | 4.6419 |
| O003 | Kimi-VL-A3B-Instruct | 23 | 2.1918 |
| O004 | Mistral-Small-3.1-24B-Instruct | 21 | 2.2092 |
| O005 | Qwen3-30B-A3B | 18 | 2.2502 |
| O006 | Qwen3-VL-30B-A3B-Instruct | 23 | 2.2112 |

Mean battery size: **21.6 mixes**. Every one of the 5 batteries is a full
grid sweep, and **none reached the FLASH_COMPLETE_EBPW_LE_1 bar** (best
2.19–4.64, all several times over the ≤1.00 target) — consistent with the
task brief's "nobody has hit sub-bit on a complete system."

This means 21.6 is a **lower bound on the wrong thing**: it is what one pass
of the existing (bits × group-size) grid costs, and that grid already proved
insufficient. How many mixes a search that can actually reach sub-bit needs —
one that also varies organ-level heterogeneous allocation via
`tools/gravity_allocator.py` / `gravity_bpw_family.py`, not just uniform bits
and group size — is **UNKNOWN**. No receipt in this repo records any
specimen approaching the target through a driven search; there is nothing to
extrapolate from.

## 3. Per-tier gauntlet cost

Tier byte/count totals are `tools/odyssey/lake_phases.py`'s live
`snapshot()` (identical source to `ODYSSEY_48H_PLAN.md`); the operator's
brief figures (79 / 323 / 294 GB) are ~7-13% above what the live lake
measures in GiB, the same rounding gap `ODYSSEY_48H_PLAN.md` already flagged
— the live numbers are used here, not the brief's.

Per tier: `io_hours = bytes / 118,000,000 Bps`, `probe_hours = n ×
21.6 mixes × 16.363 s`, `known_hours_lower_bound = io_hours + probe_hours`,
`build_hours = UNKNOWN`.

| tier | n | bytes | size (GiB) | I/O hours | probe hours (lower bound) | build hours | known lower bound |
|---|---|---|---|---|---|---|---|
| A_tiny | 21 | 73,456,681,480 | 68.4 | 0.173 | 2.062 | UNKNOWN | **2.235 h** |
| B_mid | 19 | 337,064,183,935 | 313.9 | 0.793 | 1.865 | UNKNOWN | **2.659 h** |
| C_large | 5 | 314,263,794,536 | 292.7 | 0.740 | 0.491 | UNKNOWN | **1.231 h** |
| D_giant (−3 deferred) | 8 | 1,534,938,184,222 | 1,429.5 | 3.613 | 0.785 | UNKNOWN | **4.399 h** |
| **total** | **53** | **2,259,722,844,173** | **2,104.5** | **5.319** | **5.203** | UNKNOWN | **10.522 h** |

Known lower bound across all four tiers is **10.52 h of a 48 h budget
(~22%)** — but that excludes the build stage entirely, and the build stage
is what actually varies with model size (loading + dequantizing + writing
full weight tensors 21.6 times per specimen), unlike the probe stage above,
which was flat-ish (16.4 s mean) across 7B–30B params because the probe is a
short, fixed prompt set. Extrapolating the probe stage's flatness onto the
build stage would be exactly the kind of magnitude-blind assumption the
adequacy-gate scar warns against — it is not made here.

## 4. Which tiers can complete a target-seeking gauntlet in 48h?

| tier | verdict |
|---|---|
| A_tiny | **Plausible.** Known lower bound 2.24 h; models are 1–8 GiB, comfortably resident alongside the ~11 GiB resident body on a 96 GiB machine even with build overhead included. The unresolved risk is not time or memory, it's whether *any* achievable search finds sub-bit at all (§2). |
| B_mid | **Plausible, tighter.** Known lower bound 2.66 h. Up to 40 GiB specimens; `mlx_lm.convert` needs the source weights loaded plus the quantized output materializing, which for the largest B_mid specimens starts to compete with the ~85 GiB of headroom on this machine. Flagged, not ruled out — same posture `ODYSSEY_48H_PLAN.md` took for C_large. |
| C_large | **Flagged, tight.** Known lower bound 1.23 h (cheap — only 5 specimens), but the largest specimens run to ~77 GiB; converting one (loading bf16 source + writing a quantized copy) can approach or exceed the ~85 GiB headroom on a 96 GiB machine with the resident body running. `ODYSSEY_48H_PLAN.md` already flagged this same tier as "plausible, not comfortable" for execution probes alone; a build step adds more resident pressure than a probe does, not less. |
| D_giant (−3 deferred) | **Cannot execute at all, independent of the clock.** `ODYSSEY_48H_PLAN.md` already established this: `G011_odyssey_streaming` has no receipt and all 5 of `tools/odyssey/test_odyssey_streaming_runtime.py`'s tests fail (re-confirmed here — reading that test file is a read, it regenerates nothing). `mlx_lm.convert` loads the full model to quantize it; every one of these 8 specimens (89.5–335.3 GiB) is bigger than the ~85 GiB of usable headroom before *any* KV cache or activation memory, the same ceiling that blocks plain inference. A build step needs at least as much residency as inference, so this blocker applies at least as hard to a gravity gauntlet as it did to the probe-only plan. The 48h clock is irrelevant until G011 lands. |

The 3 deferred giants (Kimi-K3, Inkling-Small, Qwen3.8-Flash-Next) are
excluded here by the same operator policy `ODYSSEY_48H_PLAN.md` encodes —
Kimi-K3 is data-only and never an execution target; the other two stay in
scope for a future pass, not this one.

## 5. Single highest-leverage change

**Instrument `convert_gravity()` to record wall-clock around the
`mlx_lm.convert(...)` call, and write it into every gravity receipt as
`build_s`.** This is a one-line-of-timing change (bracket the existing call
in `tools/odyssey_patient_runner.py`, same pattern as the `t0 =
time.perf_counter()` / `wall_s` already used for the probe battery a few
hundred lines away) and it closes the single largest gap in this whole cost
model. Right now nobody can say whether the gauntlet is I/O-bound,
probe-bound, or build-bound — and every throughput optimization aimed at the
wrong stage buys nothing (Amdahl's law: the probe stage is only ~5.2 of the
10.5 known hours, and the untimed build stage could easily dwarf both). This
is exactly the "timing-only harness shipped a wrong answer" failure mode
already on record for this campaign: reporting §3's 10.52 h as *the* gauntlet
cost, instead of a lower bound, would be presenting an estimate as a
measurement. Measure the build stage before optimizing anything.

A secondary, code-grounded (not measured) observation for once that number
exists: `convert_gravity()` re-invokes `mlx_lm.convert` against the same
canonical `hf_path` for every spec in a battery — a 21.6-mix battery reads
the specimen's full original weights from disk 21.6 separate times. If the
build stage does turn out to dominate, sharing one loaded copy of the
original weights across every mix in a specimen's battery (convert once,
requantize the in-memory array per spec) is the obvious next lever — but
that is a hypothesis to verify against a real timing, not a claim to act on
yet.

## Checked snapshot (2026-09-05)

Emitted by `python3 tools/odyssey/lake_phases.py --gauntlet`.
`tools/odyssey/test_lake_phases.py::test_gauntlet_cost_doc_agrees_with_the_live_lake`
re-derives this live and fails the moment it drifts by more than 3%.

```json
{
  "A_tiny": {
    "n": 21,
    "bytes": 73456681480,
    "io_hours": 0.17292062495291902,
    "probe_hours_lower_bound": 2.0616936666666663,
    "build_hours": null,
    "known_hours_lower_bound": 2.2346142916195855
  },
  "B_mid": {
    "n": 19,
    "bytes": 337064183935,
    "io_hours": 0.7934655930673258,
    "probe_hours_lower_bound": 1.8653418888888886,
    "build_hours": null,
    "known_hours_lower_bound": 2.6588074819562144
  },
  "C_large": {
    "n": 5,
    "bytes": 314263794536,
    "io_hours": 0.7397923600188324,
    "probe_hours_lower_bound": 0.49087944444444437,
    "build_hours": null,
    "known_hours_lower_bound": 1.2306718044632767
  },
  "D_giant_minus_deferred": {
    "n": 8,
    "bytes": 1534938184222,
    "io_hours": 3.613319642707156,
    "probe_hours_lower_bound": 0.7854071111111111,
    "build_hours": null,
    "known_hours_lower_bound": 4.398726753818267
  }
}
```
