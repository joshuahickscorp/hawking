# aud08 — can the code run a complete Odyssey I+II+III cycle in ≤48h, overlapped?

**No.** The strongest honest state is: unit-tested 48h *admission library* + proposal-shaped I/II/III listeners + one live ModelLake download watcher. There is no started 48h mission, no law stream, and no overlapping I+II+III science job.

The ≤48h objective applies to **one complete recurring I+II+III cycle**, not to first useful output. That cycle has not started.

Machine-readable twin: `receipts/audit/aud08-odyssey-48h.json`.

---

## Verdict (13-state vocabulary, strongest supported)

| Capability | Classification | Why that is the ceiling |
|---|---|---|
| Streaming laws continuously | **SCAFFOLDED** | `record_law` exists and is registered; nothing in production has called it. `HCLI_LEDGER.json` is absent. G011 receipt absent. |
| Listeners / watchers | **CALLABLE** | ModelLake watch is a live process (pid 4183). Law listener is a one-shot CLI that returns dicts. |
| Law-availability events | **SCAFFOLDED** | `phase_listeners.listen()` can emit II and III WorkUnit dicts from the II law store. `record_law` does not fire it. Units are not enqueued. |
| Specimen readiness | **CALLABLE** | `hcli.specimens.registry()` and live `modellake_events.consume()` see the lake. Odyssey-I `ODYSSEY_STATE.json` does not. |
| Predictive prefetch | **SCAFFOLDED** | G102 measured USB cold/warm rates and `specimen_scheduler.rank` prices them. Nothing prefetches bytes. |
| Multi-model scheduling | **CALLABLE** | `cycle_tick` + memgate can admit multiple Odyssey-I lanes. launchd driver is **disabled**. |
| Overlap I/II/III | **TESTED** | Unit tests prove probes may be *recorded* the moment a law is claimed. No overlapped science run. |
| 48h mission controller | **TESTED** | `horizon`/`admits`/`phase_entry` are real functions with 14 tests. Horizon is `NOT_STARTED`. No production caller. |
| Odyssey I (WHAT IS TRUE) | **CALLABLE** | August harvests and a 9-rule gravity rulebase exist. The loop is not running. |
| Odyssey II (WHAT DID HAWKING LEARN) | **SCAFFOLDED** | Transfer library is STATIC_ONLY. Acceptance **BLOCKED** (`evaluations_avoided=-8`). |
| Odyssey III (WHERE IS HAWKING WRONG) | **SCAFFOLDED** | Adversary emits STATIC_ONLY specs. Acceptance **BLOCKED** (synthetic REFUTED). |
| **One complete I+II+III cycle ≤48h** | **ABSENT** | No start receipt, no seal, G011 red, II and III blocked. |

Definitions without callers were not counted. Imports were not counted as call sites. Tests were not counted as board reality.

---

## What is actually wired

**Odyssey I patient controller** (`tools/odyssey_ctl.py`, ~8132 lines) is a real harvest/retire/acquire/launch loop. `cycle_tick` reaps lanes, harvests completions, retires eligible patients, acquires the next download when the frontier is empty, and fills lanes. That is Odyssey **I work-units**, including a TRANSFER *phase* that copies `TRANSFER_MATRIX.json` into a patient packet. It is not streams II and III.

HCLI exposes the driver and a separate ledger:

- inspect: `odyssey.status/queue/value/economics/ingest` (read-only)
- mutate driver: `odyssey.cycle` (confirm=True)
- own campaign: `odyssey.record_law`, `create_transfer_probe`, `create_adversarial_probe` (confirm=True)

The own-campaign verbs write `workspace/campaign/odyssey/HCLI_LEDGER.json`. That file is not in git and is not on disk. The only callers of `record_law` / `create_transfer_probe` / `create_adversarial_probe` *themselves* are `hcli/test_odyssey.py` (tmp ledger) and the registry wrappers. G011's producer (`tools/sovereign/g011_streaming.py`) will refuse until a resident-owned ledger exists with overlapping I/II/III timestamps.

**Law listener** (`tools/future/phase_listeners.listen`) is the closest thing to "II and III start when a law exists." In one call it can spawn both transfer and attack WorkUnit dicts, with `phase_ii_depends_on_phase_iii=False`. It sets `performs_science=False`, `evidence_class=STATIC_ONLY`. `emit_hcli_workunit` builds a dict and returns it; it does not write a queue. `lifecycle_events.py` *names* `listen()` as the `LAW_UPDATED` consumer; the cited call site is `build()` / CLI `--listen` — self-invocation, not a bus.

**48h controller** (`tools/future/odyssey_mission_controller.py`, G099) is honest about its own gap: every horizon question is `NOT_STARTED` until `receipts/future/ODYSSEY_CYCLE_1_START.json` is stamped, because reporting "hour 0 of 48" would be fake progress. `admits()` knows that a five-hour experiment is fine at hour 4 and not at hour 44 — **in tests**, via monkeypatch. Nothing that launches work calls it. `CYCLE_HOURS = 48.0` is the only live 48h number in source. H-ROADMAP on corpdrive still schedules 24h / **72h** / 1 week.

**Live process (MEASURED, no signals sent):**

- `com.hawking.modellake.watch` **running**, pid **4183**, poll 0.10s, seal-event emit every 300s
- `com.hawking.odyssey` **disabled** (`com.hawking.odyssey.plist.disabled`)
- HCLI resident at Downloads/hawking: `state=FAILED`, `worker_live=False`, mission cancelled, evacuated below 12 GiB RAM reserve
- `org.substrate.odyssey7d.telegram` registered, not running (7-day leftover)

---

## Timing — four clocks, not one

| Clock | State | Evidence |
|---|---|---|
| Time to first useful law | **ABSENT** as a 48h-cycle clock (**INFERRED** historical I artifacts) | `GRAVITY_RULEBASE.json` (9 rules) committed 2026-08-19 01:12. First harvest filename `...-20260819-015556`. Zero HCLI ledger laws. No cycle start, so elapsed_hours is undefined. |
| Time to first transfer result | **ABSENT** as Odyssey II (**INFERRED** I-phase transfer-control receipt; II acceptance BLOCKED) | O006 `transfer-control` harvest `...-20260819-131420`. `O006_TRANSFER.json` has 2 `TRANSFERRED_UNCHANGED` cells vs O005. Acceptance: COLD arm 2 evals, TRANSFER arm 10, `evaluations_avoided=-8`. |
| Time to first adversarial falsification | **ABSENT** | Novelty-adversarial harvest `...-20260819-183202` is an Odyssey I novelty lane (`physics_not_measured`). III selftest is `synthetic=True`. |
| Time to **one complete I+II+III cycle** | **ABSENT** (MEASURED absence) | No `ODYSSEY_CYCLE_1_START.json`, no `ODYSSEY_CYCLE_1.json`, no G011 receipt. Horizon `NOT_STARTED`. |

August 19 harvests show Odyssey I patient science could produce a transfer-control receipt ~11.5 h after the first harvest filename, and a novelty-adversarial harvest ~16.5 h later, **serialized inside I**. That is not a 48h overlapped cycle and not II/III as canonical streams.

---

## 55 specimens — which Odyssey blockers actually disappeared

**MEASURED on this machine (read-only):** `/Volumes/corpdrive/hawking-modellake/specimens` has **55** directories, **4.35 TB**. `receipts/future/MODELLAKE_INDEX.json`: `n_specimens=55`, `n_partial=0`, `n_families=39`, **49 SEALED / 6 UNSEALED**.

**Surprise (loud):** the campaign sentence "55 sealed" overcounts. The six UNSEALED bodies include the Odyssey I teachers:

- Falcon-H1-7B (O001)
- Kimi-VL-A3B (O003)
- Qwen3-30B-A3B (O005)
- Qwen3-VL-30B-A3B (O006)
- DeepSeek-V4-Flash (O011)
- Qwen3.8-Flash-Next

`odyssey2_transfer.require_sealed` will refuse those aliases.

**Acquisition blockers that disappeared *if consumers read the lake*:**

- Second specimen for II — gone as a download problem (49 sealed B-sides, 39 families).
- Hostile architectures for III — gone as a download problem (Dream, iLLaDA, RWKV7, BitNet, Mamba3, evo2, …).
- HF download of O001/O003/O004/O005/O006/O007/O009/O010/O011/O013 bodies — the bytes are on the USB volume.

**Still present:**

- Gemma O000/O002: no gemma slug in the 55 (license gate in ODYSSEY.md).
- O008 Jamba-1.5-Mini: lake has `AI21-Jamba2-3B`, not that specimen.
- O012 GLM-4.5 355B: absent (Air + GLM-5.3-Flash only).
- **I-controller producer is stale:** git `ODYSSEY_STATE.json` still has O001 ACQUIRING `on_disk=False` while Falcon-H1 sits in the lake. `cycle_tick` will not treat those patients as ready.
- Cold USB load: G102 receipt (stale n=8) already 2.8 h / 77 min for Flash-Next. Full 4.35 TB has no prefetch loop. Largest bodies cannot be UMA-resident (Kimi-K3 1.56 TB).
- II still blocked on *saving experiments*, not on missing models.
- III still blocked on a *non-synthetic law-scope move*, not on missing models.

`hcli/specimens.py` docstring still says "47 sealed". `SPECIMEN_LOAD_COST.json` still says 8 sealed + 29 unsealed. Three counts (47 / 8 / 55 / 49) are in circulation; 55 dirs / 49 SEALED is the index+disk fact.

---

## Acceptance receipts vs producers

| Gate | Verdict | What was actually invoked |
|---|---|---|
| `ODYSSEY_I_DISCOVERY` | ACCEPTED | STATIC `pick_acquire_candidate(mutate=False)` + safetensors header census |
| `ODYSSEY_II_TRANSFER` | BLOCKED | `load_qualification_queue`; numeric bar failed |
| `ODYSSEY_III_ADVERSARIAL_META_SCIENCE` | BLOCKED | `odyssey3_adversary.selftest` synthetic + `scars()` registry |
| HMF fusion manifest | 1 accepted / 4 blocked | I only |
| `ODYSSEY_LAUNCH_GATE` / `ODYSSEY_I_LAUNCH` | LAUNCH 16/16, `phase_transition=STARTED` | STATIC sidecar, `gpu_authority=false`. Contradicts controller `NOT_STARTED` and disabled launchd. |

Roadmap prose and STARTED sidecars were not treated as evidence that the cycle exists.

---

## G011 (sovereign streaming gate)

Verifier: `tools/odyssey/test_odyssey_streaming_runtime.py`  
Producer: `tools/sovereign/g011_streaming.py`  
Receipt: `receipts/sovereign/G011_odyssey_streaming.json` — **absent** (MEASURED).

The gate is correctly red. A separate honesty problem: it would accept overlapping `recorded_at` on `laws` / `transfer_probes` / `adversarial_probes` with `hcli_owned=true`. Those II/III lists are **PROPOSED** bookkeeping. A resident that only appended proposals could green G011 without transfer or attack running. Flagged as a surprise; settling it needs protected review of the gate, not a silent edit.

---

## What would have to exist for a yes

1. `ODYSSEY_CYCLE_1_START.json` stamped by a process that calls `admits()` on every experiment.
2. A law producer that calls `record_law` (or the II law store) **and** notifies `listen()` without a human CLI.
3. `listen()` (or equivalent) persisting II/III units onto a scheduler that runs them, not returning dicts.
4. II producing `evaluations_avoided > 0` on a second lake specimen.
5. III moving a named law's scope down with non-STATIC, non-synthetic evidence.
6. Odyssey-I `on_disk` reading the same lake the rest of the campaign uses.
7. A prefetch/residency policy that actually starts loads, because 4.35 TB cold from USB will eat the 48h budget if it waits.
8. G011 green on *jobs*, not proposals — after protected review.

Until those have call sites and durable artifacts, the 48h overlapped cycle is **ABSENT**.

---

## Evidence tiers used

- **MEASURED:** lake directory count, index seal_status, file absences (G011, CYCLE_1_START, HCLI_LEDGER), launchd/pgrep, resident state.json.
- **SOURCE_INSPECTION / STATIC_VERIFICATION:** all source and test archaeology, acceptance JSON, launch sidecars.
- **INFERRED:** August harvest filename deltas as historical Odyssey-I timing.
- **Never MEASURED here:** GPU/ANE/FPGA runtime, 48h wall of a cycle that did not run, specimen load times (cited from G102 receipt only).

Did not touch `hcli/`, `tools/`, `crates/`, `civilization/`, or ModelLake specimens. Did not signal any process.
