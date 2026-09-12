# Hawking nomenclature and migration map

**Status:** forward vocabulary and implementation migration authority

**Version:** `HAWKING_NOMENCLATURE_V2`

**Base:** `ce5e1b2a551c6817b6e82f05cf937dbdc77bb00e`

This updates the existing nomenclature contract. It is not another Hawking
constitution or a second registry. Historical receipts, schemas, hashes and
paths retain their original bytes and names.

## One system

Hawking is the complete local physical AI environment. `hawkingd` is its
persistent operating owner. Hawking Web and Hawking CLI are interaction
surfaces. HCLI, HIDE and AgentOS remain implementation or compatibility names
where removing them would break a real interface; they are not peer products.

Gravity is Hawking's science and machinery for discovering the
capability-preserving cognitive, representational, executable and physical
form best suited to an objective and machine.

Collapse is the transformation Gravity performs. It is not a service,
registry, qualification, or artifact kind. Nova is the Gravity transformation
class that changes the learned organism and records parent-to-descendant
lineage.

## Canonical lifecycle

```text
Source Specimen
    -> Gravity
    -> NR revision
    -> physical plan and backend programs
    -> NX candidate
    -> qualification and Pareto comparison
    -> selected NX for an execution profile
    -> resident instance
```

An **NR** is a mutable, inspectable, portable semantic organism with an
identifiable revision. An **NX** is an immutable deployment realization of one
exact NR revision and explicit contract. Its complete dependency closure is
accounted even when immutable shared dependencies are referenced by identity.
Mutable KV, recurrent, routing and session state belongs to an NX instance;
its layout and transition rules belong to the NX contract.

Odyssey is population science and transfer/falsification inside Gravity.
ModelLake owns source/specimen identity, storage, retention and provenance. A
download is not an admitted runtime body. Pulsar, Magnetar and Themis are
machine-relative roles for fast work, deep cognition and independent
adjudication.

## Internal contracts

| Contract | Responsibility |
|---|---|
| MachineGenome | Measured machine identity, capabilities, constraints, freshness and evidence tier. |
| PhysicalGraph | Planning projection over stable semantic identities, placement, state/effects and synchronization. |
| HWIR | Spatial/hardware lowering where a backend needs typed streams and resource constraints. It is not mandatory for CPU, Metal or ANE. |
| MegaKernel | Explicit semantic operations and effects, stage annotations, legal regions, selected schedules and backend programs. It does not require one shader or minimum dispatch count. |

`LOAD`, `DECODE`, `ROUTE`, `PROJECT`, `ACCUMULATE`, `STATE_UPDATE` and
`SAMPLE` are MegaKernel annotations. Exact operator semantics remain below
them. Representation decode and autoregressive token decode must use distinct
operator identities.

## Migration map

The map is the implementation crosswalk. Status is updated only when callers,
compatibility and tests support the claim.

| Class | Old identity | Canonical owner | Actual callers | Behavior contract | External or persistent dependencies | Compatibility strategy | Tests | Status and deletion/removal condition |
|---|---|---|---|---|---|---|---|---|
| SEMANTIC_CHANGE | Product-level HCLI / HIDE | Hawking CLI / Hawking Web | `crates/hawking/src/main.rs`; `hcli/cli.py`, `hcli/use.py`, `hcli/serve.py`, `hcli/web.py`; OpenWebUI | An invocation resolves one admitted artifact revision and one supported action; unknown identities fail closed. | Python module/package entry points, `HCLI_*` and `HAWKING_*` environment names, ports 8011/8080, install stamps, browser integration. | Keep `hcli`, `jhcli`, package and environment spellings as forwarding adapters. | `crates/hawking/src/main.rs::public_surface_tests`; `hcli/tests/test_catalog_and_switching.py`; `test_web_explicit_selection.py`; `test_serve_openai_surface.py`. | IMPLEMENTED_PARTIAL. Delete forwarding names only after clean-prefix packaging, permission parity and Web-source projection pass. |
| OWNER_CONSOLIDATION | AgentOS as a peer product | Hawking internal capability and authority machinery | `hcli/agentos/**`, WorkUnits, resident gates, VMCP adapters and supervisor callers. | Preserve explicit capability checks, mutation authority and bounded work execution. | Serialized WorkUnits, tool registry IDs, receipts and process ownership. | Preserve imports and authority gates while forward-facing product language changes. | Existing `hcli/agentos/**/test_*`, campaign restart and mutation tests. | IN_PROGRESS. The internal namespace may remain; peer-product claims are removed when caller and documentation census is clean. |
| OWNER_CONSOLIDATION | Deep Gravity and Model/Context/State Gravity | Gravity with an explicit target, budget and policy | `hcli/gravity*.py`, foundry and condense tools, plans and historical receipts. | One bounded discovery/transformation function retains cancellation, budget, run history and qualification. | Campaign receipts, schemas, command aliases and protected model permissions. | Preserve historical commands and schemas; emit the canonical definition in new public interfaces. | Gravity gauntlet, foundry lifecycle and nomenclature tests. | IN_PROGRESS. Remove a duplicate live engine only after every production caller uses the canonical owner. |
| RENAME_ONLY | Doctor / Tabula product framing | Gravity discovery and diagnosis | `tools/doctor.py`, `tools/future/tabula.py`, `hcli/tool_registry.py`. | Diagnose or propose; neither path grants execution qualification. | Tool IDs, saved invocations and historical receipts. | Keep exact tool entry points and describe them as Gravity operations. | Existing doctor, Tabula and tool-registry tests. | PLANNED. Remove aliases after an exact caller census reaches zero. |
| SEMANTIC_CHANGE | Transient Noetic pipeline vocabulary | NR revision -> physical plan -> NX candidate | `hcli/nomenclature.py`, native artifact loaders, `hcli/physical_graph.py`, receipts. | Preserve semantic revision identity separately from immutable deployment realization and instance state. | V1 schemas, hashes, artifact paths and sealed receipts. | Read V1 forever; emit V2 only at new serialization boundaries; never rewrite evidence. | `hcli/tests/test_nomenclature_v2.py` and science-boundary tests. | IN_PROGRESS. Old schemas remain historical compatibility; new callers must use the V2 interpretation. |
| RENAME_ONLY | Singularity / Singularity Profile | selected NX / execution profile | `hcli/nomenclature.py`, roadmap prose and historical selection records. | A selection binds an exact NX and machine-relative execution contract. | Persisted historical names and receipt hashes. | Read aliases as `SelectedNX` / `ExecutionProfile`; do not rewrite receipts. | Nomenclature and terminology-guard tests. | IN_PROGRESS. Remove new public use after the bounded terminology check is part of all release verification. |
| OWNER_CONSOLIDATION | Event Horizon as a current subsystem | Gravity repository and tool consolidation | Roadmap lineage and archived campaign records only. | Preserve the campaign as provenance; create no runtime authority under this name. | Historical paths and hashes. | History-only reference. | Terminology guard exceptions are limited to explicit history/migration context. | ACCEPTED. No live subsystem is eligible to be created under the historical name. |
| OWNER_CONSOLIDATION | Multiple machine-genome representations | Machine facts with specialized measured producers and compatibility projections | `hcli/machine.py`, `hcli/genomes.py`, `tools/accelerator/machine_genome.py`, placement callers. | Facts carry measurement source, freshness and evidence tier; projections cannot self-admit. | Host probes, cached machine facts and device/backend capability records. | Preserve producer APIs until differential fixtures prove a single projection. | Existing machine, genome and placement tests. | PLANNED. Consolidate only after fixture parity and freshness gates pass. |
| OWNER_CONSOLIDATION | Multiple graph planning interpretations | PhysicalGraph | `hcli/physical_graph.py`; compatibility logic in `tools/future/physical_primitives.py`; architecture adapters. | Plan over stable semantic operation/state identities, explicit effects and placement. | Machine facts, backend contracts and recorded graph fixtures. | Keep Python as the reference/planning oracle while native callers migrate. | Existing PhysicalGraph unit tests plus native region tests. | IN_PROGRESS. Remove production duplicates only after differential fixtures cover their callers. |
| OWNER_CONSOLIDATION | FPGA-shaped lowering treated as universal | HWIR only where spatial lowering is required | `tools/future/hwir.py`; `hcli/agentos/fpga_preboard.py`. | Typed streams and hardware resource constraints for spatial backends; CPU, Metal and ANE may lower directly. | Backend schemas, simulator/driver inputs and saved plans. | Preserve serialized HWIR and explicit adapters. | Existing HWIR/preboard tests where present. | ACCEPTED_CONTRACT; external driver/simulator implementation remains deferred. |
| RUST_PORT | Python graph/state legality mechanics | `hawking_core::gravity::execution` | `DeepSeekV4Ratio0AttentionDeviceExecutor::prepare`, called by the real full-sequence ratio-zero `execute_position` path. | Validate graph identity, convex legal regions, exact data/control boundaries, effects, aliases, ownership/lifetimes, loop state, numerical policy, synchronization, backend support, checked scratch and ordered program coverage. | Rust graph inputs, Metal backend limits and adapter-supplied schedules; no live state. | Python PhysicalGraph remains an independent planning oracle until differential forwarding is complete. | Ten focused native valid/invalid tests and the ratio-zero template test. | CALLER_WIRED_PLAN_ONLY. Execution and numerical qualification plus broader PhysicalGraph forwarding remain before Python production legality can be deleted. |
| OWNER_CONSOLIDATION | Model-named reusable execution mechanics | Native legal-region and backend-program contracts | DeepSeek-V4 ratio-zero adapter, full-sequence attention caller and Rust kernels. | Model adapters provide genuine geometry/math; shared code owns legality, accounting and schedule invariants. | Metal device capabilities and recorded Flash fixtures. | Leave model-specific math in the adapter; relocate shared mechanics only at a committed Flash boundary. | Native region and template tests; GPU parity deferred. | IMPLEMENTED_PARTIAL. Live Flash kernel relocation waits for conflict-aware integration. |
| OWNER_CONSOLIDATION | Method registry copies | `tools/foundry/GRAVITY_METHOD_REGISTRY.json` and `tools/foundry/query_gravity_methods.py` | Foundry lifecycle, acquisition, post-parent review and automatic-reuse callers. | Match observed traits against scoped applicability, required features and exclusions; carry Laws and Scars distinctly; run a cheap falsifier; invoke one compatible adapter; independently verify and receipt the result. | Potency ledger, negative-transfer atlas, method evidence, adapter inputs and verifier output. | Preserve research imports and sealed generation bytes; add automatic adapters only where a real target-specific invocation and refusal case are proven. | `tools/foundry/tests/test_query_gravity_methods.py`; compatible and incompatible reuse receipts. | IMPLEMENTED_BOUNDED. Flash NR accounting is automatically selected, executed and verified; payload-only accounting is refused before invocation. Broader methods still require their own compatible adapters and falsifiers. |
| OWNER_CONSOLIDATION | Laws copied into method records | Accelerator Law base | `tools/accelerator/akb.py`, qualification gates and receipts. | A Law states a scoped measured relationship and never performs or verifies a method. | Evidence hashes, units and measurement scope. | Reference canonical IDs; preserve receipt copies as evidence. | Existing accelerator law tests. | PLANNED. Delete editable copies only after callers resolve canonical IDs. |
| OWNER_CONSOLIDATION | Failures copied as universal prohibitions | Scar / negative science index | `tools/future/negative_index.py`, sovereign negative-science tools and campaign receipts. | A Scar binds failure scope, evidence and reopening condition. | Sealed negative evidence and applicability contexts. | Preserve historical evidence; route incompatible contexts to explicit research. | Existing negative-index and sovereign-science tests. | PLANNED. No shared method database is implied. |
| OWNER_CONSOLIDATION | Observations and verdicts conflated | Profilers measure; verifiers decide | Runtime profilers, benchmark receipts, capability and numerical verifier pipelines. | Observations retain clocks, units and scope; only an independent verifier grants its declared qualification. | Source/reference fixtures, receipt schemas and protected execution permissions. | Keep independent oracle implementations even when infrastructure is shared. | Existing profiler, receipt and capability-gate tests. | ACCEPTED_CONTRACT; specific duplicate removal requires caller-level parity evidence. |

Migration classes mean:

- `RENAME_ONLY`: interface spelling changes without behavior change.
- `OWNER_CONSOLIDATION`: callers converge on one implementation authority.
- `RUST_PORT`: a native owner replaces Python production mechanics under
  conformance tests.
- `SEMANTIC_CHANGE`: behavior, schema, numerical policy or qualification meaning
  changes and must be reviewed separately.

## Compatibility aliases

New code translates historical language at an explicit boundary:

| Historical phrase | Forward interpretation |
|---|---|
| source model, checkpoint | Source Specimen |
| compressed or compact model | NR or NX candidate after inspecting whether it is runnable |
| quantizer | Specific algorithm, or Gravity operator when used as a transformation class |
| Noetic IR / Program | Concrete internal NR form |
| Noetic Compiler | compilation inside Gravity |
| Hawking Accelerator | Hawking execution backend; the standalone/community release intention remains preserved in `H-ACCELERATOR.md` |
| Singularity | selected NX for an explicit execution profile |
| resident model | resident NX instance when actually loaded |
| Constellation | informal collection of admitted Stars |
| fast genesis | measured lifecycle objective, not a subsystem |

Historical names are evidence, not cleanup debt. Never rename sealed receipts,
hashed paths, schema strings, source manifests or scientific identifiers for
cosmetic consistency.

## Selection law

No name grants qualification. Selection resolves an exact artifact revision
and supported execution path. Pareto comparison uses the declared capability,
machine and execution contract; minimum EBPW alone is not a universal order.

The forward law is:

> ONE HAWKING. ONE GRAVITY. COLLAPSE IS THE VERB.

<!-- DOC_STATUS: CURRENT -->
