# aud16 — file and directory compression

Discovery audit. No files were moved, merged, or deleted. No `hcli/`, `tools/`, `crates/`, or `civilization/` edits. This note is the human summary of `receipts/audit/aud16-file-directory-compression.json`.

Evidence tier: **STATIC_VERIFICATION**. Nothing in this audit is `PHYSICALLY_MEASURED`. Tests were not run. Call sites were taken from `git grep HEAD` (the worktree is sparse; worktree grep is not the tree). An import is not a call site.

HEAD: `04193ccbc` (`04193ccbc8ef9fdd2dfd595d65f656760829dddc`).

The question is not whether files can be concatenated. It is: does **one concept** currently require navigating many files that have no independent lifecycle, authority, or reuse reason?

---

## Answer

Most of the tree's bulk is **not source**. 12,232 of 15,913 files at HEAD are artifacts (receipts, campaign records, experiment dumps). Source is 3,681 files / 85 MB.

Among source, the EBPW-shaped defect (many sibling files for one concept, no authority boundary) is real in a handful of families — and in almost every case the right collapse is **one package with fewer public surfaces**, not one file.

The six-way EBPW split used as the model (`ebpw_base` / `ebpw_complete` / `ebpw_report` / `ebpw_accounting` / `ebpw_helpers` / `ebpw_schema`) **does not exist at HEAD**. EBPW is two sibling modules plus tests, and that split is an authority boundary (the bill vs the judge). Treat that as the template, not as a concatenation mandate.

Do not collapse:

- `hcli.goal_*` (F24 identity: one file, two dotted names, two class objects)
- `hcli/doctor`, `hcli/gravity`, `hcli/vmcp` one-file ownership markers
- `ramanujan/` (blocked non-authorizing scaffold)
- `tools/roadmap/` (already a package with non-test callers)
- `tools/doctor/` and `lab/operators/doctor6/` (two live Doctor generations, already packages)
- `hide-*` vs `hawking-*` crate boundary (`Cargo.toml` default-members)
- Theia (`tools/theia/`) — one bounty model, not a civilization
- FPGA as a new top-level — it lives inside Accelerator/Fusion

---

## Surprises

Flag these loudly. Each names what would settle it.

1. **EBPW is two files, not six.** `git ls-tree -r --name-only HEAD | grep -i ebpw` returns `complete_ebpw.py`, `ebpw_categories.py`, their tests, and receipts. The prompt's six-way split is not on disk.

2. **Duplicate basenames are rivals, not copies.** `organ_bandwidth`, `hardware_doctor`, `contamination`, `protected_window`, `tournament`, `decoding_gravity`, `state_gravity` all differ in SHA-256. `tools/conftest.py` already skips `tools/headless/test_organ_bandwidth.py` during a wide `pytest tools/` because `import organ_bandwidth` hits `tools/future/organ_bandwidth.py`. That is an authority hazard from naming, not a reason to keep both.

3. **The tree is mostly evidence.** `workspace/` 6,061 files, `receipts/` 4,121, `hawking-experiments/` 1,953. Source compression and directory compression are different jobs.

4. **`crates/hawking-core/examples` holds 233 `.rs` campaign probes**, not examples. Cargo will not let them be one file (each is an example binary). The defect is the directory name and organ home.

5. **Four workspace crates are TESTED and not INTEGRATED.** `hawking-eval`, `hawking-perception`, `hawking-comms`, `hide-gateway` have `#[test]` callers of their own symbols and **zero** other `Cargo.toml` dependents. `tools/eval/support_halo_gate.py` reimplements evaluation in Python; it does not call `hawking_eval`.

6. **`ebpw_categories.py` is missing from `disk_truth_modules`** in `civilization/CAPABILITY_GRAPH.json`, which lists `complete_ebpw.py`. `ebpw_categories` has more non-test callers (`tools.doctor.engine.validate`, `tools.audit.reachability_triage.can_promote`, `judge_dense_rematerialization` in mlp/deltanet).

7. **`hcli/agentos/` is not only re-exports.** `docs/CURRENT_ARCHITECTURE.md` says AgentOS implementations stay at `hcli.goal` etc. so F24 cannot recur. `hcli/agentos/` still contains 50 files (`flash_*`, `resident.py`, `fpga_preboard.py`, `modellake_*`). The F24 rule was applied to `GoalCompiler`, then a nested package grew beside it.

8. **Support-halo exists in two languages without a call across them.** Rust `hawking-eval::score` / `wilson_interval` vs Python `tools/eval/support_halo_gate.py` vs `tools/odyssey/contamination.py` corpus seal.

9. **H-ROADMAP.md is outside this git tree** (`civilization/ROADMAP_STATE.json` `roadmap_path` = `/Users/scammermike/Downloads/H-ROADMAP.md`). This audit did not read it as instructions and did not rewrite it.

10. **Sparse-checkout lists repo-root `src/bin/*.rs` and `src/lib.rs` that do not exist at HEAD.** Those bins live under `crates/`.

---

## CURRENT TREE

Top-level at HEAD mixes durable organs with campaign names:

| Path | Files | Character |
|---|---|---|
| `workspace/` | 6061 | Campaign records/evidence/ops (source: 121) |
| `receipts/` | 4121 | Evidence, filed by campaign (`future`, `headless`, `ascent-2026-08-16`) |
| `hawking-experiments/` | 1953 | Legacy frankenstein/superwave/prometheus (source: 166) |
| `tools/` | 1666 | Living source, but `future/` + `headless/` are campaign dumps |
| `crates/` | 1093 | Gravity runtime + HIDE product |
| `lab/` | 464 | Documented as governance; `operators/` is 272 runtime scripts |
| `hcli/` | 198 | AgentOS control plane |
| `app/` | 150 | HIDE UI |
| `ramanujan/` | 110 | Blocked scaffold |
| `docs/` | 28 | Current architecture/nomenclature |
| `civilization/` | 13 | Capability graph + ROADMAP_STATE |

`tools/` second level (campaign names, not organs): `future` (541), `headless` (282), `odyssey` (114), `accelerator` (86), `condense` (130), `haider` (90), `doctor` (9), `theia` (24), `roadmap` (14), plus 27 loose `tools/flash_*.py`.

Nomenclature pipeline (from `docs/HAWKING_NOMENCLATURE.md`, treated as data): Source Specimen → Doctor → Gravity → Noetic IR → Noetic Compiler → Physical Graph Compiler → Hawking Accelerator → candidates → Pareto → Singularity → Resident.

Era I civilizations (from `civilization/ROADMAP_STATE.json`): `I-A_AGENTOS_HCLI`, `I-B_DOCTOR`, `I-C_GRAVITY_NOETIC`, `I-D_ACCELERATOR`, `I-E_ODYSSEY_I`. Constitution unchanged: five eras, three Odysseys, no Era VI, FPGA inside Accelerator/Fusion, Theia is not a civilization, coordinate 0.7.

---

## PROPOSED TREE

**Do not move any file.** This is a map for a later campaign.

Top-level should correspond to durable organs:

| Organ | Keep | Absorb later (still do not move now) |
|---|---|---|
| AgentOS/HCLI | `hcli/` | `tools/haider/hcli/tests` (fossil tests of live hcli). Do not nest `GoalCompiler` under `hcli/agentos/`. |
| Gravity runtime | `crates/hawking-core`, `hawking`, `hawking-serve`, `hawking-bench`, `hawking-speculate` | `gravity_deepseek_v4_*` into a crate-internal package; `model/qwen{30,38,80}_*` into `model/{qwen30,qwen38,qwen80}/`; 233 `examples/` probes; `lab/operators/ascension_*` and `q80_*`; `tools/flash_*.py`; selected `tools/future` + `tools/headless` gravity/noetic modules |
| Doctor | `tools/doctor/`, `lab/operators/doctor6/` | `tools/headless/doctor_*`, `tools/gravity_doctor_*`, `tools/flash_doctor_*`, `tools/doctor_seal.py`. Keep the two generations; stop adding a third. |
| Accelerator | `tools/accelerator/` | `tools/future/fpga_*`, `ane_*`. FPGA stays here. |
| Odyssey | `tools/odyssey/` | `tools/eval/` (support-halo gate is Odyssey evaluation) |
| HIDE product | `crates/hide-*`, `app/` | Do not merge `hide-gateway` / `hawking-perception` / `hawking-comms` until a non-test caller exists |
| Civilization governance | `civilization/`, `tools/roadmap/` | — |
| Evidence | `receipts/` | Reindex by organ, not campaign. Alias duplicates. Never delete. |
| Theia | `tools/theia/` | Not a civilization. Do not promote to Era I rank. |
| Documentation | `docs/` | `workspace/docs/` is a second docs tree |

Historical, not organs: `tools/future/`, `tools/headless/`, `hawking-experiments/`, `ramanujan/`, `workspace/campaign/`, `receipts/future`, `receipts/headless`, `receipts/ascent-*`, `tools/haider/`, `.hcli/delegations/`.

---

## Per-family answers

Each family answers: one file? one package? required by language/build/runtime? authority boundary? independent test/reuse? or accretion?

### EBPW — `ONE_PACKAGE_FEWER_SURFACES` — INTEGRATED

Paths: `tools/future/complete_ebpw.py` (1,125 lines), `tools/future/ebpw_categories.py` (1,550 lines), colocated tests.

**Not one file.** The split protects an authority boundary: `complete_ebpw` bills (`cost`, `mix_report`, `build`, `incumbent_candidate`); `ebpw_categories` types quantities and refuses promotion (`can_promote`, `validate`, `judge_dense_rematerialization`). They already live in `tools.future`. Collapsing them would mix the bill with the judge.

Call sites of the symbols (not imports):

- `mix_report` / `build` — `tools/acceptance/flash/run_gates.py:938`
- `cost` / `incumbent_candidate` — `tools/theia/self_bounty.py:377`
- `can_promote` — `tools/audit/reachability_triage.py:2046`
- `validate` — `tools/doctor/engine.py:438`
- `judge_dense_rematerialization` — `tools/future/mlp_byte_census.py:31`, `deltanet_representation.py:31`
- `CategoryError` — `tools/future/abliteration.py:46`

JSON field `complete_ebpw` in `tools/flash_complete_byte_ledger.py` / `tools/accelerator/scoreboard.py` is **not** a call of this module.

### `organ_bandwidth` name collision — `ONE_PACKAGE_FEWER_SURFACES` — TESTED

`tools/future/organ_bandwidth.py` (schema `hawking.future.organ_bandwidth.v1`, 239 lines) vs `tools/headless/organ_bandwidth.py` (schema `hawking.headless.organ_bandwidth.v1`, 750 lines). Not copies. `tools/headless/organ_roof_ledger.py:33` calls `ORGANS` / `map_ledger_row` from the headless module. `tools/future/physical_compiler_predict.py:154` defines `observations_from_organ_bandwidth`. One canonical module under Gravity; keep both receipts as aliases.

### `hardware_doctor` name collision — `KEEP_SPLIT_AUTHORITY` — TESTED

Not copies. Future file is the FPGA-axis sidecar; accelerator file is the host doctor and **calls** `tools.roadmap.hardware.probe` (`tools/accelerator/hardware_doctor.py:791`). Keep two modules, stop sharing the basename, both under Accelerator.

### Other campaign basename collisions — `ONE_PACKAGE_FEWER_SURFACES` — TESTED

`contamination`, `protected_window`, `tournament` (future vs odyssey); `decoding_gravity`, `state_gravity` (headless vs odyssey). All SHA-256 unequal. Rival campaign science, not an authority split. Canonicalize names; keep receipts.

### `gravity_deepseek_v4_*` — `ONE_PACKAGE_FEWER_SURFACES` — CALLABLE

33 `pub mod` files at hawking-core crate root, 1.70 MB. Largest: `native_token_graph.rs` 218,479 bytes. **One file is impossible** (rustc / editors). The split is required as a *module tree*, not as 33 crate-root public surfaces. Internal `crate::gravity_deepseek_v4*` uses are real (`research_server.rs`, `token_ns/adapt.rs`, the family itself). Outside the crate, 44 examples + 6 `dsv4f_*` tests; other hits are python/docs. Historical stage names (`p0`, `p3a`, `p4b`, `p6`, `p7`).

### Qwen 30/38/80 model runtimes — `ONE_PACKAGE_FEWER_SURFACES` — CALLABLE

`qwen80_complete_runtime.rs` is 663,073 bytes by itself. `model/mod.rs` pub-mods every `qwen80_*` at the model root; `qwen_complete_binary/` is already a subdirectory. Callers outside `model/`: `decode_family.rs`, `metal/mod.rs`, `research_server.rs`, `token_ns/adapt.rs`, `token_ns/served_weight.rs`. Package as `model/{qwen30,qwen38,qwen80}/`. Not one file.

### hawking-core `examples/` — `KEEP_SPLIT_LANGUAGE` — CALLABLE

233 `.rs` + 9 `.metal`. Cargo auto-discovers each as a binary — concatenation is a build constraint. Defect is *location*: these are campaign probes (`ascension_qwen30_*`, `gravity_deepseek_v4_*`, `flash_noetic_*`).

### `tools/future/` dump — `HISTORICAL_ACCRETION` — TESTED

541 files, 538 py, 15.5 MB. 244 of 259 impls have a colocated `test_` sibling — that is independent testing, not one concept split six ways. Shared `_common.py` (`write_receipt`, `seal`) is imported by 358 files in the dump. Do not concatenate. Split later by organ (mlp, deltanet, resident, flash, autonomy).

### MLP family — `ONE_PACKAGE_FEWER_SURFACES` — TESTED

16 impl + 16 tests, 1.14 MB. `mlp_byte_census`, `mlp_teacher_corpus`, `mlp_auxiliary_information`, `mlp_shared_program` are **called** by the other mlp_* modules and by `deltanet_*` (`deltanet_generated_transition.py:783` calls `load_x_f16`). One Gravity `mlp` package with a small public surface (census, corpus, program). Not one file (`mlp_structured_operator.py` is 105,534 bytes).

### `tools/headless` noetic_* — `ONE_PACKAGE_FEWER_SURFACES` — TESTED

48 paths, ~1.9 MB first-token family. Brand-sharing campaign probes, some with tests (`test_noetic_ir`, `test_noetic_scoreboard`). Become a Noetic package under Gravity. Not one file.

### `tools/roadmap/` — `KEEP_SPLIT_REUSE` — INTEGRATED

Already the desired shape (14 files, CLI, tests). Non-test callers: `tools.roadmap.hardware.probe` from `tools/accelerator/hardware_doctor.py` and `tools/acceptance/odyssey/run.py`; `blocked_hardware_wakes` from `tools/lifecycle_events.py`; auditor writes `civilization/CAPABILITY_GRAPH.json`. Do not collapse to one file.

### Thin unintegrated crates — `HISTORICAL_ACCRETION` — TESTED

`hawking-eval` (15 `#[test]`, `score()` called at `lib.rs:241`), `hawking-perception` (6 tests), `hawking-comms` (6), `hide-gateway` (17). **No other crate depends on them.** Bible-section crate cuts before a caller existed. Do not concatenate the four together (unrelated sections). Do not crate-split further. Fold into `hide-core` / `hawking-research` only after a non-test caller is real, or keep as TESTED islands.

`hawking-index-query` is a 4-file CLI over `hawking-index` python-facts. `tools/roadmap/index_client.py` invokes the binary. Cargo.toml comment already says it may later inline into `hawking-index` (which already has two bins). Anticipatory crate boundary.

### `hide-core` `objects_*` — `KEEP_SPLIT_LANGUAGE` — INTEGRATED

13 files behind one `pub mod objects`. `hide-backend` calls `hide_core::objects` from `hcli_sources.rs` and `headless.rs`. Normal rust split, not accretion.

### `hide-backend` — `KEEP_SPLIT_LANGUAGE` — INTEGRATED

120 files, depended on by `hide-serve`, `hide-acp`, `hide-fleet`. Internal `services_*` / `lenses_*` are rust modules. Not a one-file target. Do not merge hide into hawking-core: `Cargo.toml` default-members is a build-time authority boundary.

### `hcli.goal_*` pipeline — `KEEP_SPLIT_AUTHORITY` — INTEGRATED

`goal.py`, `goal_bank.py`, `goal_compile.py`, `goal_graph.py`, `goal_ir.py`, `goal_tokenizer.py` plus colocated tests.

Call sites: `GoalCompiler()` constructed in `hcli/engine.py:593` and `hcli/controller.py:1399`; `.compile()` at `engine.py:1292`; `GoalBank` constructed and `.add` called in `hcli/agentos/resident.py:551,1045`; `compile_worker_context` re-exported by `hcli/context.py:14`.

`docs/CURRENT_ARCHITECTURE.md` forbids nesting these under `hcli/agentos/` (F24). Do not concatenate. Do not create `hcli/goal/` if that shadows `hcli.goal`.

### `hcli/agentos/` nested growth — `KEEP_SPLIT_AUTHORITY` — CALLABLE

50 files. `__init__.py` re-exports `GoalCompiler` **and** ships `flash_*`, `resident.py`, `fpga_preboard.py`. Do not fold flash probes into `goal.py`. Do not move `GoalCompiler` into this directory.

### `hcli/doctor|gravity|vmcp` — `KEEP_SPLIT_AUTHORITY` — CALLABLE

One-file `__init__.py` directories so ownership is importable, not a comment. They should stay one file each. Collapsing them away recreates "ownership is a comment".

### Doctor scattered — `ONE_PACKAGE_FEWER_SURFACES` — INTEGRATED

One organ, many trees. Two live packages already:

- `tools/doctor/` — representation doctor. `tools/doctor/__main__.py` calls `diagnose`/`build`. `tools/future/ebpw_categories.py:969` calls `tools.doctor.zeros`.
- `lab/operators/doctor6/` — physical prescribe/treat/verify. CLI `__main__.py` calls those symbols. Tests: `lab/tests/test_doctor6_*.py`.

Accretion is the third+ copies: `tools/headless/doctor_*`, `noetic_doctor_v2.py`, two `hardware_doctor.py`, `tools/gravity_doctor_*`, `tools/flash_doctor_*`, `hcli/doctor` marker. Keep the two generations; give them one organ home; stop minting more.

### `lab/operators/doctor6` — `KEEP_SPLIT_REUSE` — TESTED

Already the EBPW-good shape (14 files, CLI, tests). Tests live in `lab/tests/` (lab convention). Do not concatenate.

### `lab/operators` ascension/q80 — `HISTORICAL_ACCRETION` — CALLABLE

272 operator files: 127 `ascension_*`, 19 `q80_*`. Independent scripts, 160 without a `lab/tests` sibling. Not one concept. Do not concatenate. Absorb into Gravity runtime operators later; stop using `lab/` as a third `tools/`.

### `lab/tests` vs `lab/operators` — `KEEP_SPLIT_REUSE` — TESTED

109/146 tests pair by basename with an operator. A test package is a valid convention (same as `crates/*/tests`). Not a compression target unless the operators move.

### Receipts campaign trees — `ONE_PACKAGE_FEWER_SURFACES` — INTEGRATED (as evidence)

`receipts/headless` 1,624, `receipts/future` 1,054 (including 592 `nowait-*`), `receipts/ascent-2026-08-16` 716. **42 JSON basenames** exist in both `receipts/future/evidence` and `receipts/headless` (`ACCELERATOR_SCOREBOARD.json`, `APPLE_ANE_ATLAS.json`, `DOCTOR_TRANSFER.json`, `FLASH_META_REPRESENTATION_SUB1.json`, …). `receipts/headless/ORGAN_LIBRARY_CONSOLIDATION.json` already says receipts are never deleted and rivals become aliases. Reindex by organ; alias; do not merge bodies without fixing the producer.

### `workspace/campaign` — `HISTORICAL_ACCRETION` — SCAFFOLDED

5,555 files (records 3,571, evidence 1,433, governance 316). Not source. Keep off the organ source path.

### superwave `.keep` dirs — `HISTORICAL_ACCRETION` — SCAFFOLDED

50 directories whose only file is `.keep` (`hawking-experiments/superwave/g1/sing/{adversarial,auction,causal,…}`). Nested dirs with no architectural boundary.

### frankenstein vs `tools/condense` — `HISTORICAL_ACCRETION` — CALLABLE

frankenstein 1,500 files, 1,440 under `data/`. `tools/condense` is the living tool (130 files). Duplicate location for Condense. Keep `tools/condense` as source.

### ramanujan — `KEEP_SPLIT_AUTHORITY` — SCAFFOLDED

README: non-authorizing scaffold blocked on Hawking completion. `scaffold/` / `container/` / `governance/` / `records/` is an authority boundary. Do not fold into `tools/` or `hcli/`.

### `tools/haider` fossil — `KEEP_SPLIT_AUTHORITY` — SUPERSEDED

`import tools.haider.hcli` raises `ImportError`. Tests of live `hcli/` still sit under `tools/haider/hcli/tests/`. Do not concatenate fossil and live. Remaining defect: two locations for one test suite.

### `tools/flash_*.py` at repo-tools root — `ONE_PACKAGE_FEWER_SURFACES` — CALLABLE

27 loose files (`flash_organ_census.py`, `flash_meta_representation.py`, …). Same brand as `tools/future/flash_*` and headless FLASH receipts. Not a package. Not one file.

### `token_ns` — `KEEP_SPLIT_REUSE` — CALLABLE

Already a package (`mod.rs`, schema, audit, energy, reconcile, adapt, served_weight). Model of a good split.

### Theia — `KEEP_SPLIT_AUTHORITY` — TESTED

Already a package with tests. `self_bounty.py:377` calls `complete_ebpw.cost` and `incumbent_candidate`. Constitution: not a civilization.

### `tools/sovereign` vs `tools/sovereignty` — `HISTORICAL_ACCRETION` — TESTED

Different concepts, colliding names (`g001`–`g014` scripts vs a 2-file package). Rename later; do not concatenate.

### `docs/` vs `workspace/docs/` — `HISTORICAL_ACCRETION` — STALE_ROADMAP_TEXT

Two documentation roots. `docs/CURRENT_ARCHITECTURE.md` is current; it labels `workspace/docs/reference/ARCHITECTURE.md` HISTORICAL. Duplicate location for architecture prose.

### Integrated hawking/hide crates — `KEEP_SPLIT_LANGUAGE` — INTEGRATED

Workspace crate boundaries with a real path-dep graph. `hide-backend` depends on hawking-context/index/orch/research and hide-*. Default `cargo build` excludes hide (96k lines) by design. Do not merge.

---

## Directory audit (checklist)

**One-file directories.** 53 source one-file dirs. Most are required rust (`crates/*/Cargo.toml`, `src/bin`). Identity markers: `hcli/doctor`, `hcli/gravity`, `hcli/vmcp`. Leftovers: `tools/tests/` (one file), `tools/haider/tests/` (one file), `hawking-experiments/frankenstein/__init__.py`.

**Nested dirs with no architectural boundary.** superwave `.keep` trees; frankenstein `data/fullseq_parallelism_probe_20260806/w4/workerN/traces`; `receipts/headless/FLASH_SINGLE_PROCESS_CHAIN_L*_L*/layer-N/`; `receipts/future/nowait-30m-*/no_wait/`; `.hcli/delegations/<id>/.hcli`.

**Legacy experiment trees.** `hawking-experiments/frankenstein`, `superwave`, `prometheus`; `ramanujan/`; `workspace/campaign`.

**Duplicate `future` / `ops` / `runtime` / `core` / `common` / `util`.** `tools/future` + `receipts/future` is the real duplicate brand. `workspace/ops` is ops, not a source organ. `core`/`common`/`util` are not widely duplicated as directory names; campaign brands (`future`, `headless`, `odyssey`, `ascent`) are.

**Multiple locations for one concept.** organ_bandwidth; hardware_doctor; Doctor (many trees); HCLI (`hcli/` vs `tools/haider/hcli` vs `lab/hcli`); docs; EBPW receipts (future / headless / acceptance); support-halo (rust crate / python gate / odyssey seal); Condense (tools vs frankenstein).

**Tests separated from the code they exclusively govern.** rust `crates/*/tests` and `lab/tests` are conventions. Defect: `tools/haider/hcli/tests` after the package moved to `hcli/`. `tools/future` and `hcli/` colocate tests.

**Receipts spread across campaign trees after the concepts became generic.** 42 overlapping basenames future∩headless; date-stamped `ascent-2026-08-16`; 592 `nowait-*` runs; 3,571 `workspace/campaign/records` that never entered `receipts/`.

---

## What this audit is not

- Not an implementation plan. Not a rewrite of `H-ROADMAP.md`.
- Not a measurement of Metal, ANE, FPGA, thermal, or protected benches.
- Not a claim that tests pass. Classification `TESTED` means tests exist and appear to call the symbols.
- Not permission to concatenate `hcli/` modules or to crate-merge hide into hawking-core.

Machine-readable twin: `receipts/audit/aud16-file-directory-compression.json`. Every family record carries `evidence[]` with `evidence_kind` in `{SOURCE_INSPECTION, STATIC_VERIFICATION, INFERRED}` and `refs`.
