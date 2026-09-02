# aud20 — conceptual load on HCLI

Discovery only. No implementation. No H-ROADMAP rewrite.
Machine-readable twin: `receipts/audit/aud20-conceptual-load.json`.
HEAD: `04193ccbc`. Evidence class: **SOURCE_INSPECTION**, plus one **STATIC_VERIFICATION** (live `ToolRegistry` instantiate = 68 tools).
Nothing here is **MEASURED** hardware.

## Verdict

**Hawking is not getting conceptually smaller as it gets more capable.** The live spine HCLI actually runs is small. The search set it must retrieve to touch that spine is large, aliased, and still branded by a vestigial campaign.

Live spine (call sites, not imports):

- `WorkUnit` / `GoalCompiler.compile` / `Mission` / `Scheduler` / `WorkUnitExecutor`
- `start_resident` / `hawkingd`
- `tools.odyssey.modellake_promote.promote`
- `hawking_bench::run` from the hawking CLI
- `run_protected_accelerator_benchmark`

Search set that taxes it:

| surface | HEAD count |
|---|---|
| `tools/future` Python (non-test) | 262 |
| unique `hawking.ascension.*` schema strings in source | 631 |
| `hawking-core` examples named `ascension_*` | 131 / 233 |
| `lab/operators` named `ascension_*` | 127 / 269 |
| registered HCLI tools | 68 |
| Goal compilers | 2 (`GoalCompiler` live, GoalIR tested-not-wired) |
| classes named `WorkUnit` | 2 |
| classes named `ResidentSupervisor` | 2 |
| FPGA/physical primitive vocabularies | 3 |
| EBPW producers | 2 (acceptance gate talks to only one) |
| physical-graph compilers | 3 |
| doctor-named files outside receipts | 75 |

Constitution held as data: five eras, three Odysseys, no Era VI, FPGA inside Accelerator/Fusion, Theia is not a civilization, coordinate `0.7` in `civilization/ROADMAP_STATE.json`. This audit does not propose changing any of that.

## How HCLI is taxed

HCLI does not fail because a module is missing. It fails because **several modules answer to the same job**, and the catalog often points at the envelope rather than the producer.

Definitions without callers were not counted as capability. `import X` is not a call site. Tests are not board reality. Roadmap prose was treated as data.

This worktree is clean at HEAD. The live launchd daemon is described as holding ~110 uncommitted `hcli/` files; **that overlay was not diffed**. Claims below are about `04193ccbc`.

## Conceptual fan-out (seven tasks)

After-compression counts are proposals, not an implemented state.

### 1. Understand WorkUnit — live path INTEGRATED; parallel stacks TESTED_NOT_WIRED

**Now: 8 must-read files + 10 trap files. After: 4.**

Must-read: `hcli/workunit.py` (576), `hcli/goal.py` (1553, **live** `GoalCompiler().compile` from Mission/Controller/Engine/AgentOS.runtime), `mission.py` (1582), `scheduler.py` (427), `executors.py` (623), `dag_store.py` (405), `resources.py`, `tool_registry.py` (2193).

Traps:

- GoalIR stack (`goal_ir.py` / `goal_graph.py` / `goal_tokenizer.py` / `goal_compile.py`, ~2260 lines). `goal_compile.py` claims it is the only contact with `WorkUnitDAG`. Production never calls `ingest`/`schedule` (only tests do).
- `lab/hcli/special_unit.py` (2597) — different product per `docs/CURRENT_ARCHITECTURE.md`.
- `tools/future/improvement_metabolism.py` defines a **second class named `WorkUnit`** (“Scientific work, not an HCLI WorkUnit”).
- `tools/future/workgraph.py` (1990), `workunit_species.py`.

HCLI tax: answering “what is a unit of work” requires distinguishing two compilers and three unit types.

### 2. Modify EBPW — policy CALLABLE; complete-system EBPW SCAFFOLDED / ABSENT as a physical number

**Now: 10 files, six distinct meanings. After: 3.**

- Policy ceiling: `hcli.flash_next.COMPLETE_SYSTEM_EBPW_MAX = 1.00`.
- Flash writer: `flash_executable._ebpw_budget` → `FLASH_EBPW_BUDGET.json` with `measured.complete_system_ebpw: null`, status `PLANNED_UNTIL_VERIFIED_BODY`.
- Calculator: `tools/future/complete_ebpw.py` (`STATIC_ONLY` arithmetic over `MIX_REPORT`). **Does not import** `flash_executable`; **not imported by** it.
- Acceptance `FLASH_COMPLETE_EBPW_LE_1` calls `complete_ebpw.build`, not `_ebpw_budget`.
- Tensor `effective_bits_per_value` in Flash organ experiments is a different quantity.
- `EBPW_NAMESPACE_SEPARATION` already named `ARTIFACT_PHYSICAL_*` vs `DESIGN_EXPECTED_*`; producers still emit overlapping keys. `scoreboard.py` reads either.

HCLI tax: changing the 1.00 ceiling does not move the gate that “proves” EBPW ≤ 1.

### 3. Add an FPGA primitive — SCAFFOLDED, hardware BLOCKED_HARDWARE

**Now: 11 files, three ontologies. After: 3.**

| vocabulary | count | home |
|---|---|---|
| `NR_PRIMITIVES` | 8 | `hcli/physical_graph.py` (HCLI serializes these) |
| `ATLAS_PRIMITIVES` | 17 | `tools/future/physical_primitives.py`, fingerprint-locked to the architecture atlas |
| HWIR `NODE_KINDS` | 7 | `tools/future/hwir.py` (6279 lines) |

`hwir.py` **does** call `physical_primitives.instantiate` (real call site). Adding a name only to `NR_PRIMITIVES` is invisible to HWIR; adding it only to HWIR is invisible to the atlas lock. `fpga_preboard` is a `[S]` mock. No U50 on this host.

Also in the path: `FpgaHwirBackend`, `lower_fpga_domain_to_hwir`, `fpga_engines.py` (no HDL), `tools/odyssey/physical_graph_compiler.py`, `recompile_physical_graph`.

### 4. Change resident lifecycle — live path INTEGRATED

**Now: 8 must-read + 12 traps. After: 3.**

Live: `hcli/agentos/resident.py` (2396) — `ResidentDaemon`, `ResidentSupervisor`, `start_resident` (callers: `agentos_cli.py`, lifecycle tests, native smoke). `hawkingd` imports `daemon_main` from that module.

Trap: `tools/future/resident_supervisor.py` (2270) defines a **second `class ResidentSupervisor`**, constructed from its own `__main__` and tests, not imported by the live module. Plus `resident_{api,health,identity,install,optimizer,provider}.py`, `super_resident.py`, `fallback_resident.py`, `tools/hcli_resident/`, `genesis_resident.py`, and `ascension_qwen38_resident.rs` (vestigial campaign name on a body binary).

Do not signal the live daemon to “verify” this.

### 5. Add a ModelLake transition — `promote()` is INTEGRATED

**Now: 9 files. After: 3.**

Canonical mutation: `tools.odyssey.modellake_promote.promote`. Callers (not imports): `modellake_watch.promote_if_needed` (`go=True`), `hcli/acquisition.py`, `tools/acceptance/lake/promotion.py`. A test already asserts watch and promote share one module object.

`modellake_gate.run_modellake_census` is census-only (never downloads, never promotes). HCLI tools `specimens.registry`, `acquisition.propose`, `modellake.status` are three views. `tools/future/specimen_*` look like peers.

Blast radius if the wrong module grows a second `promote`: a 4.35 TB lake.

### 6. Run a benchmark — several CALLABLE surfaces, easy to invert

**Now: 9 surfaces. After: 3.**

| name | what a call actually does |
|---|---|
| `hawking bench` | `hawking_bench::run` (live from `crates/hawking/src/main.rs`) |
| `hcli agentos protected-accelerator-bench` | `run_protected_accelerator_benchmark` (exclusive window) |
| tool `benchmark.run` | pytest/unittest/cargo, not hawking-bench |
| tool `accelerator.benchmark` | authorizes a window |
| tool `benchmark.inspect` | `_receipt_read` |
| `lab.bench_harness` | **trap** — foundry tests + authority census only |
| `tools/bench/*.sh` | shell competitive harnesses |
| `kernel_bench` / `bench-q4k-shapes` | kernel microbench |

Complete-system TPS/EBPW remain unmeasured in this lane (not run).

### 7. Add a representation candidate — Flash experiments CALLABLE; the HCLI tool is CONCEPT_ONLY

**Now: 12 files. After: 3.**

Live science: `run_flash_representation_experiment` and the Flash organ CLI family (12+ `add_parser` entries). Qwen80 mixed pack is a second specimen with schema `hawking.ascension.qwen80_mixed_representation_candidate.v1`.

The catalog tool named `gravity.experiment` sets `execute=True` → `REFUSED_NOT_IMPLEMENTED`. It does not call the Flash runner.

## Ranked consolidations (see JSON for files, tests, risk)

**Near-zero-risk (do these without a campaign):**

1. Catalog `alias_of` — stop listing `fs.read`/`filesystem.read` (and the other six alias groups, including `git.checkout/revert-safe`) as first-class tools.
2. Rename `improvement_metabolism.WorkUnit`.
3. `pytest` `testpaths` currently `tools/haider` only — collect `hcli/`.
4. Stamp `doctor.query` / `gravity.experiment` results with `invoked_producer=null`.
5. Cross-link the two EBPW receipts (producer + sibling + `evidence_class`); do not rewrite sealed bytes.
6. Mark `hawking-eval` / `hawking-perception` / `hawking-comms` as SCAFFOLDED (zero external `use` sites). Confirm default-members before changing Cargo.
7. Stop minting new `hawking.ascension.*` schemas (forward-only).
8. Rename future `ResidentSupervisor`.
9. Map `NR_PRIMITIVES` → atlas names or `UNMAPPED`.
10. Stop treating `lab.bench_harness` as a product bench.

**High-leverage (one job, one symbol):**

1. One Goal compiler on the live path (wire GoalIR or mark it SUPERSEDED — not both looking live).
2. One EBPW calculator; namespaced fields; gate imports the writer.
3. Atlas 17 as the only FPGA/physical primitive names.
4. One `ResidentSupervisor` (future sidecar constructs the live class).
5. `doctor.query` invokes one Doctor producer.
6. `gravity.experiment` dispatches to `run_flash_representation_experiment` (or specimen pack).
7. Quarantine `tools/future` by **import-graph triage**, not a mass `git mv` (`CONSOLIDATION.json` already BLOCKED that pattern).
8. Collapse Flash AgentOS CLIs into one family with stages; keep receipt filenames.
9. ModelLake tools are views of `promote()` / `registry()`.
10. One protected-window lock shared by physical benches; `benchmark.run` cannot impersonate it.

**Nomenclature retirements (forward-only; sealed receipts stay):**
Ascension as a live identifier; Condense/Press as public identity (already Gravity); the duplicate `WorkUnit` / `ResidentSupervisor` names; `NR_PRIMITIVES` as a peer ontology; un-namespaced “EBPW”; Haider as the HCLI path; `wax` bench alias; Goal “tokenizer”; unqualified “artifact”.

**Harness generalizations:** default pytest collects live HCLI; unique catalog names; tools must call a producer or refuse; shared protected lock; acceptance gates call the producer symbol (lake promotion already does this); one WorkUnit fixture; Flash stages; one `diagnose()` facade; Goal DAG golden; no new `ascension_*` operators.

**Directory/file collapses:** triage `tools/future`; collect `hcli/` tests; re-export or drop empty `hcli.doctor`/`hcli.gravity` markers; classify `hawking-core/examples` vs bins (verify launchd callers before moving); no new `lab/operators/ascension_*`; keep headless/accelerator/odyssey as three live roots; `lab/hcli` is SpecialUnit, not HCLI; Doctor is a facade not a directory merge; unused hawking crates leave default-members; **do not collapse `receipts/`**.

## Surprises (flagged, not smoothed)

1. **GoalIR is not on the live compile path** despite a docstring that says it is the only contact with `WorkUnitDAG`. Settle: a production `ingest()` caller, or SUPERSEDED plus a test that Mission does not import it.
2. **`doctor.query` does not call Doctor.** Settle: monkeypatch of `tools.doctor.engine` changes the tool result (today it would not).
3. **`gravity.experiment(execute=true)` is refused** next door to a live Flash experiment runner.
4. **Three FPGA primitive lists**, none a subset of another. A board is not required to settle the naming.
5. **Two EBPW producers, one gate.** Negative control: mutate the unused writer; the gate must not move.
6. **Default pytest does not collect `hcli/`.** Settle: `pytest --collect-only` lists `hcli/test_workunit_can_call_tools.py`.
7. **Three workspace crates with zero external `use` sites** (`hawking-eval`, `hawking-perception`, `hawking-comms`). `hide-backend` *does* use `hawking-orch` / `hawking-research`.
8. **Ascension is VESTIGIAL and still the largest schema brand** (631 vs Gravity 173).
9. **A live tool is named `git.checkout/revert-safe`.**
10. **This audit is HEAD, not the daemon dirty tree.**
11. **`complete_ebpw` incumbent arithmetic is not Flash complete-system EBPW** (the latter is null). STATIC_ONLY must not be spoken as PHYSICALLY_MEASURED.

## Do not do

Mass-rename sealed receipts. `git mv` `tools/future` as one lane. Merge SpecialUnit into WorkUnit. Kill/signal the live daemon. Write MEASURED for EBPW/TPS/FPGA/ANE/GPU from this lane. Add Era VI. Rewrite the 0.7 coordinate. Collapse `receipts/`. Chase LOC.

Semantic density, not line count: ten explicit readable lines may beat one generic metaprogram. The defect is **duplicate names for the same job**, and **parallel stacks that HCLI must distinguish before it can act**.

## Commit blocker (this sandbox)

`git add --sparse -A` failed: `index.lock: Operation not permitted` (worktree git dir and common `.git/objects` / `.git/refs` are not writable here). Both artifacts exist on disk:

- `receipts/audit/aud20-conceptual-load.json` (valid JSON)
- `receipts/audit/aud20-conceptual-load.md`

The launcher must stage and commit them before cleanup, or they will be destroyed. No nested git dir or bundle was created.
