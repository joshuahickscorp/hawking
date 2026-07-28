# Speed research: where the wall clock goes, and what removes it

**Date:** 2026-07-27  
**Scope:** Research + bounded microbenchmarks only. Did not modify the running pack under `~/Library/Application Support/Hawking/GLM52Gravity/`.  
**Note:** `_HEAVY.md` is not in this worktree; ground truth taken from `HAWKING_HEAVY_CONTINUATION_STATUS.json` and live measurement.

---

## Executive answer

| Rank | Lever | Measured effect | Effort | Risk |
|---:|---|---|---|---|
| 1 | **Eliminate the second full fetch** (measure→evict→pack re-fetch). One-pass or retain bodies through pack inside a sliding window | **~2× network bytes** on a full campaign (2×1.40 TiB → 1×1.40 TiB). Wall ≈ halves while fetch-dominated | Medium (orchestrator redesign; allocation stays global) | **Low** if SHA-256 still verified and allocation still global |
| 2 | **Multi-file concurrency 4–8** (outer axis; Xet already splits one file) | Sustained **1698 → 1984 → 2001 Mbit/s** at 1/4/8 files (**+18%** over serial, then flat). Live 4-wide pack ≈ **1.5 Gbit/s** aggregate | Low (already partially live) | **Low** (disk residency ×N; floor must cover N×5.36 GiB) |
| 3 | Prefetch depth >1 + pack-side parallel workers once bodies resident | Hides **~3.5 s** pack compute behind fetch; small vs network | Low–medium | Low |
| 4 | Measure-score optimization (rank cascade / fewer ranks / cheaper probe) | Measure work **~51 s/shard**, of which **score ~41 s (80%)**. Pack path only **~3.5 s** | Medium | **Medium** if allocation curves change |
| — | HTTP range / partial tensor download | **No meaningful win**: pack needs **~100%** of payload (AA 93.4% + PT 6.5% of weights; both written) | High | **High** if verification weakened |
| — | `hf_transfer` | **No-op** (hub 1.24 unused) | — | — |
| — | `HF_XET_HIGH_PERFORMANCE` alone | **No-op vs link ceiling** once Xet is already on; solo files already 0.8–2.0 Gbit/s | — | — |

### Single highest-value change

**Stop fetching every shard twice.**  
The program is `measure (fetch+evict) → allocate global → pack (fetch+evict)`. Parent payload is **1.403 TiB**; two streaming passes ≈ **2.8 TiB** on the wire. Disk cannot hold the parent (**~152–170 GiB free**, floor 30–141 GiB), but a **sliding window that measures and packs before evict** (or reorders so each body is admitted once) cuts network in half without changing codecs.

**Cost:** Rewrite the pack orchestrator so a body is fetched once, measured, held until its pack step (or pack immediately after a frozen global allocation is available). Global allocation still needs a full measure pass first, so the clean design is:

1. Measure all shards with fetch/evict (unavoidable first touch), **persist MEASUREMENT only** (already done — 80 MB).  
2. Allocate (already done).  
3. Pack with concurrent fetch — **only one more touch** (current state).

For a **future cold run**, merge (1) and (3) only if allocation can be deferred differently; with **global** allocation, you cannot pack before measure completes. The real 2× win is **not re-measuring and not re-fetching for a third reason** — i.e. ensure pack is the **only** second pass, and make that pass concurrent at 4–8 files.

**Remaining work right now (measure already paid):** ~184 shards, ~1.0 TiB, lower bound **~1.2 h at 2 Gbit/s**, observed **~2.5–3 h** at ~70 shards/h. Concurrency is the remaining live lever; one-pass savings on measure are already sunk.

---

## 1. Fetch: real ceiling

### What the “~780 Mbit/s” number was

It is a **lower operating point**, not the per-TCP or per-file physics limit. Full ledger:

| Sample | N | Mbit/s |
|---|---:|---:|
| REHYDRATE_LEDGER solo-ish verifies | 368 | min 170, **median 1213**, mean 1324, max **2062** |
| Solo p10 / p50 / p90 | 368 | 751 / 1213 / 1991 |
| Concurrent mid-overlap (4 files) | 10 | per-file median **391** (share of pipe) |

**Command:**
```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import median, mean
rows=[json.loads(l) for l in open(Path.home()/"Library/Application Support/Hawking/GLM52Gravity/pilot_source/REHYDRATE_LEDGER.jsonl") if l.strip()]
v=[r for r in rows if r.get("event")=="VERIFIED" and "megabits_per_second" in r]
print(len(v), min(r["megabits_per_second"] for r in v), median([r["megabits_per_second"] for r in v]), max(r["megabits_per_second"] for r in v))
PY
```

### Official concurrency ladder (already on disk, 2026-07-23)

Source: `reports/condense/glm52_generation_b/GLM52_GENERATION_B_XET_AUTOTUNE.json`

| Concurrent whole files | Sustained aggregate Mbit/s | Per-shard median Mbit/s | Wall for 3×~5.36 GB |
|---:|---:|---:|---:|
| **1** (serial) | **1698** | 1895 | 75.7 s |
| **4** | **1984** | 690 | 64.9 s |
| **8** | **2001** | 697 | 64.3 s |

- **Gain 8 vs 1: 1.18×**, then flat.  
- Autotune interpretation (receipt): *“gain near 1.0 means the link, not the client, is the ceiling.”*  
- Selected: **8 files**, **2000.6 Mbit/s**.

So the **aggregate ceiling on this path is ≈ 2.0 Gbit/s (~20% of 10Gbase-T)**, not 10 Gbit/s. Client concurrency past ~4–8 only adds disk pressure.

### Per connection vs per file vs per stream

| Claim | Evidence |
|---|---|
| Cap is **not** a hard 780 Mbit/s per TCP | Solo files regularly **1.2–2.0 Gbit/s** |
| **Xet already range-splits one file** into chunk fetches | Autotune method text: *“Xet parallelises chunk ranges within a file”*; outer axis is whole files |
| Range-splitting one file further in userland | **Redundant** with Xet; not measured as better than Xet solo ~1.9 Gbit/s |
| N concurrent **files** | Helps until ~2 Gbit/s aggregate; 4≈8 |

### Live passive observation (did not start competing downloads)

Incomplete-file growth under `pilot_source/.cache/huggingface/download/` while the pack ran 4-wide:

- 50 s window: peak window **1392 Mbit/s**, overall messy because of completion gaps.  
- 90 s window: nonzero-window median **~1.7 Gbit/s**, mean **~2.2 Gbit/s**, spurious peaks to **6.5 Gbit/s** (chunky writes / multiple `.incomplete` names — treat as upper noise, prefer autotune sustained).  
- Ledger concurrent batch of 8 shards: **sustained 1488 Mbit/s** wall-accounted.

### Levers that did **not** work (keep visible)

| Lever | Result | Evidence |
|---|---|---|
| `HF_HUB_ENABLE_HF_TRANSFER` | No-op | hub **1.24** deprecates; rehydrate comments + campaign notes |
| `HF_XET_HIGH_PERFORMANCE=1` alone | Does not lift past ~2 Gbit/s aggregate | Solo already fast; autotune used HP=1 and still capped ~2 Gbit/s |
| Expecting 10G line rate from one HF/Xet client | **False** | Ceiling ~2 Gbit/s measured |

---

## 2. Do we need the bytes? (partial / range reads)

### What the pack actually reads

`measure_shard` / `pack_shard` iterate **every** tensor in the safetensors header and `read_bf16_tensor` for each (`tools/condense/glm52_activation_aware_pack.py`).

### Byte fractions (ALLOCATION + official weight map)

| Quantity | Value |
|---|---|
| Source payload | **1403.19 GiB** (282 shards) |
| Activation-aware weight bytes | **1311.80 GiB (93.4%)** |
| Pass-through weight bytes | **91.39 GiB (6.5%)** |
| Header share of file | **0.0005%** (`GLM52_SOURCE_FORMAT_LEDGER`) |
| Typical MoE shard (281): AA span of payload | **99.26%**; gaps if AA-only **0.74%** |

Pass-through is **not** skippable for the artifact: it includes `lm_head`, `embed_tokens`, dense MLP tensors that failed beats-null, etc. Pack **writes them through**.

### Conclusion on range GETs

- Network reduction from “only AA ranges”: **≲ 1–7%** typical, **and** pack still needs PT.  
- Xet CAS reconstruction may still pull overlapping chunks.  
- Skipping full-file SHA-256 to allow sparse reads is a **correctness risk** (rehydrate currently verifies full sealed sha256).

**Partial reads are not the large win.** The large win is **not downloading the same full payload twice**, not downloading half of each file.

---

## 3. Re-fetch / one-pass schedule

### Why two fetches exist

```
measurement shardwise (fetch → measure → evict)
    → allocation global
    → packing shardwise (fetch → pack → evict)
```

Parent **1.507 TB** cannot stay resident. Disk now: **~152–170 GiB free**; usable above 30 GiB floor ≈ **120–140 GiB** ≈ **24–26 shards** of 5.36 GB (matches `GLM52_STREAMING_SCHEDULE` max resident 26).

### Measure pass already done

| Metric | Value |
|---|---|
| MEASUREMENT.json | 59585 tensors, **15639 s (~4.34 h)** wall |
| Sum of per-shard work | **5425 s (~1.51 h)** |
| Fetch/idle overhead in measure | **~2.84 h** |
| ALLOCATION.json | present; BPW **1.0924** |

So **half the double-fetch tax is already paid.** Remaining is pack re-fetch only.

### Is a one-pass schedule feasible with 170 GiB free?

| Design | Feasible? | Notes |
|---|---|---|
| Keep all 282 bodies | **No** | Needs ~1.4 TiB |
| Sliding window measure+pack before global allocate | **Correctness risk** | Allocation is **global** budget over all tensors |
| Measure-all (evict) → allocate → pack with concurrent prefetch | **Yes (current)** | Second touch only for pack |
| After allocate: multi-file fetch 4–8, prefetch, pack, evict | **Yes** | Matches autotune; live pack already multi-process ~4-wide |
| True single touch end-to-end | Only if allocation becomes streaming/local | Changes science; rank separately |

**Network for remaining ~184–197 shards:** ≈ **1.0 TiB** once.  
Lower bound at 2 Gbit/s: **~1.2 h**. At serial 780 Mbit/s: **~3.0 h**.

---

## 4. Compute profile (local body `model-00281`, read-only)

**Environment:** `/Users/scammermike/Downloads/hawking/.venv/glm52/bin/python`, numpy 2.2.6.  
**Shard:** 5.32 GB, 216 tensors, AA 212 / PT 4.

| Stage | Seconds | Notes |
|---|---:|---|
| Header parse | 0.001 | |
| Raw sequential read | 0.79 | ~6.8 GB/s disk |
| Read all tensors + BF16→f32 widen | 2.3–3.2 | |
| Build one layer basis (eig path) | 7.0–7.4 | once per layer, cached |
| **Full measure_shard** | **51.2** | |
| — read | 3.2 | |
| — project+reconstruct (6 ranks) | 7.1 | |
| — **functional_score** | **40.8 (80%)** | `X @ W.T` + row cosines + null + recon norm |
| **Pack-path sim (rank-16 only)** | **3.50** | read 3.00 + proj 0.30 + serialize 0.06 |
| Full-file SHA-256 | 2.52 | rehydrate verifies this |

**Single expert gate_proj [2048,6144] (input side):**

| Rank | project | recon | score | serialize |
|---:|---:|---:|---:|---:|
| 16 | 0.0019 s | 0.0042 s | 0.0305 s | 0.0003 s |
| 256 | 0.0032 s | 0.0046 s | 0.0279 s | 0.0014 s |

### Once fetch stops dominating

1. **Measure phase:** `functional_score` over 6 ranks — not SVD of W (projection is `W @ B` / `B.T @ W`).  
2. **Pack phase:** disk read + single projection; **~3.5 s** vs **~25–60 s** fetch → still fetch-bound even at 2 Gbit/s (~21 s/shard for 5.36 GB).  
3. **SHA-256** of each body (~2.5 s) is real but smaller than fetch.  
4. **Idle cores:** pack path is single-threaded over shards; numpy releases GIL inside GEMM so `--workers` on measure helps; `phase_pack` is still serial with `Queue(maxsize=1)` prefetch.

---

## 5. Everything else (campaign wall clock)

| Sink | State | Scale vs pack |
|---|---|---|
| **GLM pack (this job)** | Dominant live heavy: ~4.3 h measure + multi-hour pack | **Primary** |
| Teacher capsules | 33 capsules, **~86 GB**, already captured | Sunk |
| Cargo `target/` | **~22 GB** under Downloads/hawking | Rebuilds minutes–tens of min; not 1.5 TB network |
| Elan/Lean | elan present; **~5.3 GB** `~/.elan`; Lean 4.32.1 pinned in status | Mathlib fetch/build can be large once; Q0 unblocked |
| Ramanujan data | **~7.1 GB** under Application Support; tree mostly JSON/codegen | Local generation lane; not network-of-parent scale |
| Container / Odyssey | Gated on Math-Preserve-v2 artifact; Q0 container replay already green | Waiting on pack, not burning equal wall |
| Desktop apps / load | Status: MOP stopped; governor conservative on loadavg | Irrelevant to pack bytes |

Nothing else in the campaign currently rivals **~2.8 TiB of GLM shard traffic** for wall clock.

---

## Ranked levers (detail)

| # | Lever | Speedup (measured or bounded) | Effort | Risk | Safe? |
|---:|---|---|---|---|---|
| 1 | One body admission per shard for pack after measure (no accidental triple fetch; concurrent 4–8) | Remaining wall → **~1.2–2 h** vs **~3 h** serial-780 | Low | Low | Yes |
| 2 | Full-campaign single-touch design (future cold runs) | **~2×** network | Medium | Low if verify kept | Yes |
| 3 | Multi-file concurrency 4–8 | **+18%** sustained vs serial Xet | Low | Low (disk) | Yes |
| 4 | Prefetch depth ≥2–4 while packing | Hide **3.5 s** work; better link fill | Low | Low | Yes |
| 5 | Measure workers >1 (already supported) | Measure work 1.5 h parallelizable | Low | Low | Yes |
| 6 | Cheaper measure scores (early exit ranks, subsample rows) | Up to **~0.8×** of measure compute (~1.2 h) | Medium | Medium (allocation) | Separate |
| 7 | Skip SHA-256 | ~2.5 s/shard | Trivial | **High** | No |
| 8 | HTTP range sparse tensors | **≲7%** bytes even optimistically; PT still needed | High | High if no full hash | No |
| 9 | hf_transfer | **0** | — | — | n/a |
| 10 | Xet HP flag alone | **0** vs already-on path | — | — | n/a |

---

## Microbenchmark commands (reproducible)

### A. Ledger rates
```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import median, mean
p=Path.home()/"Library/Application Support/Hawking/GLM52Gravity/pilot_source/REHYDRATE_LEDGER.jsonl"
v=[json.loads(l) for l in p.read_text().splitlines() if l.strip() and '"VERIFIED"' in l]
v=[r for r in v if "megabits_per_second" in r]
xs=[r["megabits_per_second"] for r in v]
print(dict(n=len(xs), min=min(xs), med=median(xs), mean=mean(xs), max=max(xs)))
PY
```
**Raw:** n≈374+, med≈1169–1213, max≈2062 Mbit/s (grows as pack runs).

### B. Autotune table
```bash
python3 -c "import json;d=json.load(open('reports/condense/glm52_generation_b/GLM52_GENERATION_B_XET_AUTOTUNE.json'));
print([(c['concurrent_files'],c['sustained_megabits_per_second']) for c in d['configurations']])"
```
**Raw:** `[(1, 1698.1), (4, 1983.7), (8, 2000.6)]`

### C. Passive incomplete growth (read-only; do not write into pack dirs)
```bash
# sample sizes of pilot_source/.cache/huggingface/download/*.incomplete every 5s; compute delta*8/dt
```
**Raw (90 s):** peak noisy 6547 Mbit/s; median nonzero ~1715; mean all ~1584 Mbit/s.

### D. Compute profile on resident shard only
```bash
/Users/scammermike/Downloads/hawking/.venv/glm52/bin/python tools/condense/...  # see session: measure 51.2s, pack_sim 3.5s
```
**Raw:** score 40.8 / proj 7.1 / read 3.2; pack_sim read 3.0 proj 0.30 ser 0.06; sha256 2.52 s.

### E. Consumption fractions
```bash
# ALLOCATION dispositions + weight_map n_weights*2
```
**Raw:** AA 1311.80 GiB / PT 91.39 GiB / total 1403.19 GiB; shard281 AA coverage 99.26%.

---

## Bottom line

1. **Wall clock is still network**, with a **~2 Gbit/s path ceiling** (not 10G, not 780 Mbit/s forever).  
2. **Xet already multi-streams within a file**; outer concurrency only buys **~18%**.  
3. **Partial tensor downloads are a dead end** for this pack — nearly all bytes are consumed, PT included.  
4. **The structural waste is the second full traversal** of 1.4 TiB. Measure is done; don’t invent a third. Make pack concurrent and never re-fetch a packed shard.  
5. **Compute is fine** until fetch is fixed: pack is ~3.5 s/shard; measure is score-heavy (~41 s) but already finished for this run.  
6. **Highest-value change:** orchestration so each shard body is admitted **once** for pack after a frozen allocation (and concurrent 4–8 while doing it). **Cost:** medium engineering on the fetcher/orchestrator; **not** a codec change; keep full SHA-256.
