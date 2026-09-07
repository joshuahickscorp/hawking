# Odyssey 48h schedule — phase arithmetic, checkable against the live lake

Machine: M3 Ultra, 96 GiB unified. Resident body: ~11 GiB. Lake: 56 specimens
on `/Volumes/corpdrive/hawking-modellake`, USB, 118 MB/s sequential.

Units: tier boundaries and sizes below are **GiB (2^30 bytes)**, not decimal
GB — that is what the operator's own figures turn out to be (the 3
deferred-giant sizes match the live lake to within 0.1 GiB in binary units,
0.9 GiB gap in each in decimal GB). I/O time uses **118,000,000 bytes/s**
(decimal, standard USB throughput convention) against the exact byte count,
not a rounded GiB figure.

## Where these numbers come from

Every table below is the live output of `python3 tools/odyssey/lake_phases.py`,
taken 2026-09-05. That script is the source of truth, not this document —
`tools/odyssey/test_lake_phases.py` re-runs it and fails the moment the two
disagree by more than 3%. If the lake has moved on since, **trust the script,
regenerate this file, and re-run the test** — do not hand-edit the numbers
below into place. The exact snapshot behind this plan is embedded verbatim
in the "Checked snapshot" section at the bottom.

Note on the operator-supplied MEASURED FACTS in the task brief: this plan's
tier *counts* (21 / 19 / 5 / 11) and the top-3 deferred sizes match those
facts exactly once read as GiB. The per-tier *byte* totals differ from the
brief by up to ~13% (A_tiny: 68.4 GiB measured here vs. 79 GiB stated) —
consistent with `du`-style block-rounding on many small files, or the lake
having shifted a few specimens' bytes since the brief was written. Either
way, that gap is exactly why this plan is generated from a script and
checked by a test instead of typed once and trusted forever.

## Phase 0 — static census, all 56, header-only

Reads `config.json` + safetensors headers only (no weight bytes): ~127 KB
per shard header × 764 shards ≈ 97 MB total, across the whole 4.3 TiB lake.
At 118 MB/s that is under 1 second of I/O — negligible next to the phases
below. **Fits in 48h trivially**, and should run first because it costs
nothing and it is the input every later phase's tier assignment depends on.

## Phase 1 — Odyssey I execution probes, tiers A_tiny + B_mid

| tier   | n  | bytes            | size      |
|--------|----|------------------|-----------|
| A_tiny | 21 |    73,456,681,480 |  68.4 GiB |
| B_mid  | 19 |   337,064,183,935 | 313.9 GiB |
| **total** | **40** | **410,520,865,415** | **382.3 GiB** |

I/O time at 118 MB/s: **0.966 h** (58 min). **Fits in 48h** with immense
headroom — I/O is not the constraint here; probe compute time per specimen
is not measured and is not estimated in this document (no fabricated
numbers). All 40 specimens in this range are small enough to be fully
resident on a 96 GiB machine with an 11 GiB resident body already running.

## Phase 2 — Odyssey I, tier C_large

| tier    | n | bytes           | size      |
|---------|---|-----------------|-----------|
| C_large | 5 | 314,263,794,536 | 292.7 GiB |

I/O time at 118 MB/s: **0.740 h** (44 min). **Fits in 48h.** Residency is
tighter here: the largest (`arcinstitute/evo2_40b`, 76.6 GiB) plus the 11 GiB
resident body leaves under 9 GiB of headroom on a 96 GiB machine for KV
cache, activations, and OS — plausible, not comfortable. Flagged, not ruled
out.

## Phase 3 — tier D_giant, minus the 3 deferred giants

Operator decision 2026-09-05 (encoded as policy, not re-litigated): the top
3 by size — `moonshotai/Kimi-K3` (1453.8 GiB), `thinkingmachines/Inkling-Small`
(495.4 GiB), `windowsxp811203/Qwen3.8-Flash-Next-Abliterated` (335.3 GiB) —
are deferred from execution-class Odyssey work. They stay in the phase-0
static census and remain eligible for a later pass; they are excluded only
from phases 1-3 below.

| tier                    | n  | bytes             | size        |
|-------------------------|----|-------------------|-------------|
| D_giant (all)           | 11 | 3,987,905,281,873 | 3,714.0 GiB |
| deferred (top 3)        |  3 | 2,452,967,097,651 | 2,284.5 GiB |
| **D_giant minus deferred** | **8** | **1,534,938,184,222** | **1,429.5 GiB** |

I/O time at 118 MB/s for the remaining 8: **3.613 h**. By the clock, this
**fits in 48h** — I/O alone is cheap even at 1.4 TiB.

**But I/O time is not the binding constraint, and this is the part that
does not fit today:** every one of the remaining 8 specimens (89.5 GiB to
335.3 GiB — `mistralai/Mistral-Small-3.1-24B-Instruct-2503` through
`Qwen/Qwen3.8-Flash-Next`) is larger than the ~85 GiB left on a 96 GiB
machine once the 11 GiB resident body is running, before any KV cache or
activation memory. None of them can be loaded whole. The only path around
that is streaming/offload — `G011_odyssey_streaming` has **no receipt** and
**all 5 of `tools/odyssey/test_odyssey_streaming_runtime.py`'s tests fail**
(re-confirmed while writing this plan: `receipts/sovereign/G011_odyssey_streaming.json`
does not exist; running that pre-existing test file is a read, it does not
regenerate any receipt or campaign state). So phase 3's real
schedule risk is not the 48h clock, it is a missing execution path. Until
G011 lands, phase 3 does not execute at all, regardless of how much of the
48h budget is spent.

## Odyssey II — on Odyssey I survivors only

No fixed byte total: II's input is whatever subset of phases 1-3 comes back
as a survivor, which is a phase-1/2/3 *outcome*, not a static lake fact.
This document does not guess a survivor count. Once phase 1-3 receipts
exist, II's cost is `lake_phases`-computable the same way — sum the survivor
specimens' bytes and divide by 118 MB/s — but that number does not exist
before those receipts do.

## Odyssey III — on finalists only

Same shape as II, one level further downstream: III's input is II's
finalists. No fixed byte total for the same reason.

## Summary — what fits in 48h and what does not

| phase | I/O time | fits in 48h? |
|-------|----------|--------------|
| 0 (census)            | <1 s     | yes |
| 1 (A+B)               | 0.97 h   | yes |
| 2 (C)                 | 0.74 h   | yes |
| 3 (D minus deferred)  | 3.61 h   | yes, **by the clock only** — no execution path exists yet (G011 absent) |
| II (survivors)        | unknown  | cannot be scheduled until phase 1-3 receipts exist |
| III (finalists)       | unknown  | cannot be scheduled until II's receipt exists |

Total I/O across the whole 4.3 TiB lake, everything included, is 11.09 h —
under a quarter of the 48h window. **I/O was never going to be the
bottleneck.** The real gates are: G011 streaming (blocks all of tier D,
not just the deferred 3, from ever executing on this machine as-is), and
the survivor counts out of phases 1-3 (which bound how big II and III even
are).

## Checked snapshot (2026-09-05)

The exact numbers every table above was built from, as emitted by
`python3 tools/odyssey/lake_phases.py`. `tools/odyssey/test_lake_phases.py`
re-derives this same block live and asserts agreement within 3% on every
field — that test is what keeps this document from silently going stale.

```json
{
  "n_specimens": 56,
  "total_bytes": 4712689941824,
  "tiers": {
    "A_tiny": {"n": 21, "bytes": 73456681480},
    "B_mid": {"n": 19, "bytes": 337064183935},
    "C_large": {"n": 5, "bytes": 314263794536},
    "D_giant": {"n": 11, "bytes": 3987905281873}
  },
  "deferred_n": 3,
  "deferred_bytes": 2452967097651,
  "phase1_n": 40,
  "phase1_bytes": 410520865415,
  "phase1_hours": 0.9663862180202448,
  "phase2_n": 5,
  "phase2_bytes": 314263794536,
  "phase2_hours": 0.7397923600188324,
  "phase3_n": 8,
  "phase3_bytes": 1534938184222,
  "phase3_hours": 3.613319642707156,
  "total_hours": 11.093902876233521
}
```
