# aud17 — Hawking nomenclature lexicon

**Discovery only.** No implementation. No H-ROADMAP rewrite. No refactors.

Companion machine file: `receipts/audit/aud17-nomenclature-lexicon.json`
(`schema: hawking.audit.nomenclature_lexicon.v1`, 118 entries, HEAD `04193ccbc`).

Evidence tier: **SOURCE_INSPECTION / STATIC_VERIFICATION**. Nothing in this lane is `MEASURED` on GPU, ANE, FPGA, HMF, eGPU, or DGX. A module import is not a call site. Tests are not physical measurements. Roadmap prose is not existence.

---

## Verdict

The constitution's expected top-level list is the right *size*. Most of those nouns are real semantic objects. The unearned mass is everywhere else: DNA-map boxes, appendix sciences, empty packages, duplicate class bodies, and one live dual compiler.

**WorkUnit is already the collapsed form the audit asked for.** There is no `PrefillWorkUnit`, `DecodeWorkUnit`, `FpgaWorkUnit`, `GravityWorkUnit`, `WorkUnitFamily`, `WorkUnitGroup`, `WorkUnitBundle`, `WorkUnitCluster`, `EBPWUnit`, `EBPWFamily`, or `EBPWVariant` anywhere under `hcli/`, `tools/`, `crates/`, `lab/`, `src/`, `docs/`, or `civilization/` at HEAD. `tools/future/workunit_species.py` emits *into* the existing WorkUnit field set. Do not invent the subclasses.

The hierarchy inflation that **does** exist is not WorkUnit families. It is:

1. **Two goal compilers** (`GoalIR` / `GoalNode` vs live `GoalCompiler`)
2. **Duplicate class bodies** for Law, Scar, WorkUnit, MachineGenome, HWIR
3. **Overloaded public names** (Fusion, Doctor, Gravity, Ascension, HIDE, Resident)
4. **Roadmap nouns with zero modules** (TransportGenome, PhysicsIR, AIR/ACORE, HawkFrame, TGFD)

---

## Constitution (held)

| Rule | Evidence |
|---|---|
| Exactly five eras | `H-ROADMAP.md:425`; 25 genes I-A..V-E in `civilization/CAPABILITY_GRAPH.json` |
| No Era VI | `tools/future/workunit_species.py:341` and `tools/future/succession.py:298` **raise** if era ∉ {I,II,III,IV,V} |
| Exactly three Odysseys | `H-ROADMAP.md:997`; `codex_ingest.py:42` "no fourth Odyssey" |
| FPGA lives inside Accelerator/Fusion | DNA map + `FPGA_HWIR` gate; U50 **ABSENT** on this host |
| Theia is one bounty model, not a civilization | `H-ROADMAP.md:2974`; `tools/theia/` is an engine without a model |
| 0.7% coordinate | `H-ROADMAP.md:68` — accounting, not a noun |
| North star | Hawking = self-optimizing physical AI computer |

Era VI mentions in `tools/future/` are **refusals**, not stale claims of a sixth era. That is constitution enforcement in code.

---

## Expected top-level — existence against callers

Keep these as top-level nouns. Existence is the strongest state **this lane can support**.

| Noun | Lexicon | Existence | Earned as a separate object? | What actually runs |
|---|---|---|---|---|
| **HCLI** | CANONICAL_PUBLIC | INTEGRATED | yes | `hcli/` runtime, 198 git paths |
| **AgentOS** | CANONICAL_PUBLIC | INTEGRATED | yes | `class AgentOS` at `hcli/agentos/runtime.py:47`; constructors in `agentos_cli.py`, `autonomy_gate.py`, `recovery.py`, `resident.py`, `charge.py` |
| **WorkUnit** | CANONICAL_PUBLIC | INTEGRATED | yes | `hcli/workunit.py:33`; 62 non-test `WorkUnit(` hits |
| **GoalIR** | CANONICAL_PUBLIC | TESTED | **not yet** | **No class `GoalIR`.** `GoalNode` is the atom. `ingest`/`schedule` are imported only from tests. Live path is `GoalCompiler` |
| **Doctor** | CANONICAL_PUBLIC | CALLABLE | yes | `tools/doctor_seal.seal` called from `odyssey_ctl.py`, `odyssey_patient_runner.py`, `nos_pipeline.py`. `hcli/doctor/` is an **empty ownership marker** |
| **Gravity** | CANONICAL_PUBLIC | INTEGRATED | yes | crates + `tools/gravity_*`; `.gravity` files are historical format, not the process |
| **Noetic** | CANONICAL_PUBLIC | CALLABLE | yes | NR/NX tools + flash_noetic examples. **Complete executable is a refused scaffold** |
| **EBPW** | CANONICAL_PUBLIC | CALLABLE | yes | a **metric** (`complete_ebpw` / `mix_report`). No EBPWUnit type. Gate WIRED, bar not met (3.14 vs ≤1) |
| **PhysicalGraph** | CANONICAL_PUBLIC | CALLABLE | yes | `compile_physical_graph(` in flash campaigns, `architecture.py`, `fusion_bridge.py`. Plan, not board execution |
| **Hawking Accelerator** | CANONICAL_PUBLIC | CALLABLE | yes | `tools/accelerator/*` + Metal. FPGA/eGPU/DGX backends **BLOCKED_HARDWARE** |
| **ModelLake** | CANONICAL_PUBLIC | INTEGRATED | yes | MODELLAKE_* gates BUILT/WIRED. Specimen store, not a runtime |
| **Odyssey** | CANONICAL_PUBLIC | INTEGRATED | yes | `tools/odyssey_ctl.py` + `hcli/odyssey.py` via `tool_registry` (`odyssey.cycle`). Phases I/II/III, not three products |
| **MachineGenome** | CANONICAL_PUBLIC | CALLABLE | yes | **Name collision** — see surprises |
| **Device Ascension** | CANONICAL_PUBLIC | SCAFFOLDED | yes | `tools/accelerator/device_ascension.py` + tests. Not the vestigial Ascension campaign |
| **HMF** | CANONICAL_PUBLIC | CALLABLE | yes | software fabric TESTED; **device BLOCKED_HARDWARE** (`HMF_PRESENT=false`) |
| **HGVAS** | CANONICAL_INTERNAL | SCAFFOLDED | **no** | coherence-ladder level 2 + `HgvasRef`. Keep the paired *name*; it is not a second fabric |
| **Fusion** | CANONICAL_PUBLIC | SCAFFOLDED | yes (one sense) | heterogeneous Fusion is UNREACHABLE (no second physical domain). Kernel fusion is a different sense |
| **Theia** | CONSTITUTIONAL | SCAFFOLDED | yes (as a future model) | `tools/theia/` engine + tests; **no hcli production import**; all THEIA_* gates BLOCKED_EXTERNAL |
| **VMCP** | CANONICAL_PUBLIC | CALLABLE | yes | `call_vmcp` / `inspect_vmcp` from `tool_registry.py` / `connectivity.py`. All-Seeing Eye is an alias. Nine acts are verbs |
| **FPGA/HWIR** | CANONICAL_PUBLIC | CALLABLE | yes (as Accelerator IR) | `class HwirGraph` has production callers. `class HWIR` in `fpga_preboard.py` does not. U50 ABSENT |
| **ResultEnvelope / Receipt** | CANONICAL_PUBLIC | INTEGRATED | yes | `build_result_envelope` from `engine.py` and `agentos/runtime.py`. Receipts are durable files of that payload |
| **Law** | CANONICAL_PUBLIC | CALLABLE | yes | **two** `class Law` bodies, both with `Law(` callers |
| **Scar** | CANONICAL_PUBLIC | CALLABLE | yes | **two** `class Scar` bodies, both with `Scar(` callers |

---

## The live GoalIR defect

This is the loudest finding on an expected top-level noun.

- `hcli/goal_ir.py` is a typed schema. The atom is `class GoalNode` (line 320). There is **no** `class GoalIR` and **no** `GoalIR(`.
- The word `GoalIR` appears in four files, three of them module docstrings.
- `goal_compile.ingest` / `schedule` (GoalIR → WorkUnitDAG) are imported only from `hcli/test_goal_compile.py` and `hcli/test_goal_compiler_acceptance.py`. The module `__main__` demo is not a production caller.
- The engine that actually runs:

```text
hcli/engine.py:593     self.goal_compiler = GoalCompiler()
hcli/engine.py:1292    compiled = self.goal_compiler.compile(prompt)
hcli/controller.py     GoalCompiler().compile(...)   (three sites)
hcli/mission.py:316    GoalCompiler().compile(text)
hcli/agentos/runtime.py:280  GoalCompiler().compile(...)
```

**GoalIR is TESTED. GoalCompiler is INTEGRATED.** The constitution named GoalIR. The process named GoalCompiler. That dual has not earned both names.

What would settle it: a non-test call from `engine.py` / `mission.py` into `goal_compile.ingest`/`schedule`, **or** an explicit decision that GoalCompiler *is* GoalIR and the unused schema is a field of it.

---

## Nouns that have not earned existence

These look like peers of WorkUnit in the DNA map, appendices, or current-tense docs. They should be fields, enums, organs, profiles, aliases, or historical brands.

**Absent as modules (roadmap-only):**
TransportGenome · PhysicsIR · AIR · ACORE · HawkFrame · TGFD · Hawking Train T0 · Perpetual Hawking (DORMANT, no git definition)

**Aliases / instances, not products:**
All-Seeing Eye → VMCP · MAXX → scheduler profile · HAWKGPU-0 → HMF Accelerator instance · HUMF → HMF · Headless → HCLI receipt plane · Ultragoal → `GoalType.ULTRAGOAL`

**Components that should stay under a parent:**
NR, NX, NOS, NVM, Tabula → Noetic/Doctor
Foundry, Condense → Gravity
Kernel Forge, Hardware Doctor, ANE → Hawking Accelerator / Doctor
Tool Doctor, nine eyes → VMCP organs
Fusion Bridge → Fusion
Topology Ascension → Device Ascension
WorkUnitDAG, WorkUnitExecutor, Mission → WorkUnit/AgentOS
Product Sovereignty, Developer Platform, Dominance Scoreboard, Green Machine → era genes, not products

**Patients that leaked into types:**
Flash-Next (`hcli/flash_next.py` is `PINNED_REVISION` **data**) · Qwen3.8 / Qwen27 · GLM52 · DSV4F · SpecialUnit · OneMountain · Loc300k

**Empty or stale producers:**
- `hcli/doctor/__init__.py` and `hcli/gravity/__init__.py` — ownership markers, zero callables
- `tools/nos_pipeline.py` — header claims `DOCTOR → GRAVITY → NR → NX → NVM/HIDE`; `qualify_and_promote` actually runs spawn / timing / doctor / provenance / promote / rebind and **never calls Gravity, NR, NX, or NVM**. No non-test importer. **STALE_ROADMAP_TEXT in the producer.**
- `FLASH_FULL_NOETIC_EXECUTABLE` — explicitly refused scaffold (`native_loader=NOT_IMPLEMENTED`)

**Historical campaign brands (vestigial ≠ reclaimable):**
HIDE, Ramanujan, Haider, Frankenstein, Ascension, Ascent, Strand, Genesis, Fabric, Prometheus, Eco, TG, … — keep the evidence, stop using them as live vocabulary. Census 2026-08-27 already classified these. Do not `git mv` sealed schemas.

---

## Duplicate definitions (same noun, extra class)

| Noun | Bodies | Keep |
|---|---|---|
| WorkUnit | `hcli/workunit.py:33` (canonical) · `tools/future/improvement_metabolism.py:295` (duplicate) | `hcli/workunit.py` |
| Law | `autonomy_scars.py:514` · `odyssey2_law_store.py:315` | unresolved — both have production `Law(` callers |
| Scar | `autonomy_scars.py:564` · `negative_index.py:467` | unresolved — both have production `Scar(` callers |
| MachineGenome | `hcli/machine.py:1018` **self-describes as not the genome** · `tools/accelerator/machine_genome.py` · `RuntimeGenome` | probe + accelerator digest; retire the bag's class name |
| HWIR | `fpga_preboard.py:152 class HWIR` (self-ctor only) · `tools/future/hwir.py:862 class HwirGraph` (callers in fusion_bridge, backend_contract, p6_projection, propagate) | `HwirGraph` |
| PhysicalGraph compiler | `hcli/physical_graph.py::compile_physical_graph` · `tools/odyssey/physical_graph_compiler.py` | unresolved — both have callers |
| HMF | `hmf.py` (canonical) · `humf.py` (still imported by fusion_planner) | HMF |

`hcli/machine.py:1019–1032` is worth quoting: the class is a compatibility bag, not a producer, not admission authority; "Admission must not import this class." The noun MachineGenome has earned existence. **This class name has not.**

---

## Overloaded names

**Fusion** is four execution contracts:

1. Hawking Fusion — heterogeneous machines (`fusion_planner` / ISA / wire / bridge). Canonical public sense. Gate `FUSION_FIRST_HETEROGENEOUS_EXECUTABLE` is **UNREACHABLE** (`NO_SECOND_PHYSICAL_DOMAIN`).
2. Metal / kernel op fusion (`*fusion_parity.rs`, `qwen38_fusion_audit.py`)
3. Fusion ISA/wire as implementation of (1)
4. `representation_decode_fusion` / `noetic_dispatch_fusion`

Counting kernel-fusion files as Hawking Fusion would inflate the noun the same way counting imports inflated an earlier lane.

**Doctor** is one mechanism. Tool Doctor ⊂ VMCP. Hardware Doctor ⊂ Doctor with `domain=fpga`. Doctor6/V5 are instruments. `hcli.doctor` is a sign on an empty door.

**Ascension** the vestigial campaign (4028 owned files) vs **Device Ascension** the constitutional process vs live argv0 `ascension_qwen38_resident` (prior nomenclature audit, 2026-09-01). Three senses, one token.

**HIDE** the deferred IDE (`crates/hide-*`) vs **hide_plan** Noetic kernel-selection (`hawking.nos.hide_plan_swap.v1`). Census already called this a trap.

**Genesis** the vestigial Qwen3.8 campaign vs Odyssey-II's "genesis tournament" word.

---

## Surprises (what would settle them)

1. **GoalIR is not on the live path.** Settle with a production caller or a rename of GoalCompiler.
2. **The WorkUnit subclass explosion is not in the tree.** Settle by *not* adding it.
3. **Gene I-A is SCAFFOLDED** in `CAPABILITY_GRAPH.json` (`no_production_caller` of `hcli/engine.py`) while 18 gates are BUILT and `AgentOS()` / `WorkUnit()` have many non-test constructors. Gene implementing-symbol ≠ gate implementing-symbol.
4. **`class MachineGenome` is documented as not the genome.**
5. **`nos_pipeline.py` stage list is a lie about its own function.** Fix the producer, not a receipt.
6. **TransportGenome** is named in H-ROADMAP §18 and has zero source hits.
7. **PhysicsIR** is a whole appendix and has no module. Theia does not import it.
8. **Fusion is four senses;** heterogeneous Fusion is UNREACHABLE on this host.
9. **Theia engine without a Theia model.** `tools/theia/` tests exist; HCLI does not call them; THEIA_* gates are BLOCKED_EXTERNAL.
10. **HGVAS is not a fabric.** Ladder level + identity type inside HMF. No page directory, no device.
11. **Two Law types, two Scar types, extra WorkUnit.** Callers exist; distinct execution contracts were not shown.
12. **hide_plan / HIDE homonym** still live.
13. **This sandbox did not see the live HCLI daemon** (`pgrep hawkingd|hcli.__main__` empty). The environment pack asserts it. Prior audit observed `ascension_qwen38_resident`. This lane did not signal any process. Process state here is **INFERRED**, not MEASURED.
14. **Census surviving vocabulary is four names** (HCLI, Odyssey, Noetic, Gravity). This constitution expects ~23 semantic objects. Campaign-brand survival ≠ semantic-object survival. Do not mass-rename either list.
15. **Noetic Executable is a refused scaffold.** The noun has not earned completeness.
16. **Era VI is refused in code.** Keep those guards.

---

## Capability-graph cross-check (not re-derived)

`civilization/CAPABILITY_GRAPH.json` (STATIC, generated 2026-09-02): 71 gates — 18 BUILT, 20 SCAFFOLDED, 11 WIRED, 13 BLOCKED_HARDWARE, 7 BLOCKED_EXTERNAL, 1 ABSENT (`AGENTOS_BEHAVIOR_LAB`), 1 UNREACHABLE (`FUSION_FIRST_HETEROGENEOUS_EXECUTABLE`). Hardware probes: U50/DGX/eGPU/HMF/new-M-series all `present: false`.

That graph is a gate/gene auditor. This file is a **noun** auditor. Where they disagree (I-A gene SCAFFOLDED vs AgentOS INTEGRATED), this audit trusts **call sites of the named type** (`AgentOS(`, `WorkUnit(`) over a gene's chosen implementing file.

---

## Advisory collapse (not an implementation plan)

Prefer fewer strong nouns with more explicit fields.

- Keep the expected top-level list. Do not promote AIR, ACORE, HAWKGPU, All-Seeing Eye, MAXX, HawkFrame, TransportGenome, PhysicsIR, TGFD, Hawking Train, Kernel Forge, Hardware Doctor, Tool Doctor, Fusion Bridge, NR/NX/NOS/NVM, Tabula, Foundry, Headless, Ultragoal, Charge, or FrontierScheduler.
- One WorkUnit, one Law, one Scar, one MachineGenome producer, one HWIR graph type (`HwirGraph`), one PhysicalGraph compiler.
- HGVAS is an HMF field. HUMF is an HMF alias.
- Fusion public sense = heterogeneous machines. Kernel fusion is a pass.
- Doctor public sense = measure / prescribe / verify. Nine VMCP acts stay verbs.
- Flash-Next / Qwen* stay specimen **data**.
- Historical brands stay historical. Vestigial ≠ reclaimable.
- FPGA stays inside Hawking Accelerator / Fusion. No Era VI.
- Theia stays one bounty model. Appendix F is a Theia lab.

WorkUnit already obeys this law. GoalIR does not, until the engine calls it.

---

## What this lane did not do

- Did not edit `hcli/`, `tools/`, `crates/`, or `civilization/`.
- Did not rewrite `H-ROADMAP.md`.
- Did not run pytest/cargo (call sites from `git grep HEAD`).
- Did not inventory ModelLake specimen bytes (`/Volumes/corpdrive` is read-only and out of authority).
- Did not measure GPU/ANE/FPGA/HMF.
- Did not kill, signal, or restart any process.
