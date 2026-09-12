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

| Class | Old identity | Canonical owner | Actual callers/dependencies | Compatibility and tests | Status and removal condition |
|---|---|---|---|---|---|
| SEMANTIC_CHANGE | Product-level HCLI / HIDE | Hawking CLI / Hawking Web | `pyproject.toml`, `hcli.cli`, `hcli.web`, Rust `hawking`, OpenWebUI | Keep `hcli`, `jhcli`, package and environment spellings until isolated install, command and permission parity pass. | IN_PROGRESS; retire public framing after all installed entry points forward to Hawking. |
| OWNER_CONSOLIDATION | AgentOS as peer product | Hawking internal capability/authority machinery | `hcli.agentos`, tools, WorkUnits, resident supervisor | Preserve imports and authority gates; update forward help/docs. | IN_PROGRESS; namespace may remain indefinitely while peer-product language disappears. |
| OWNER_CONSOLIDATION | Deep Gravity and Model/Context/State Gravity | Gravity runs with explicit target, budget and policy | Gravity tools, plans and historical receipts | Preserve command/schema/history aliases; new public definitions use Gravity. | IN_PROGRESS; remove a live duplicate engine only after its callers use the canonical owner. |
| RENAME_ONLY | Doctor / Tabula product framing | Gravity discovery and diagnosis | `tools.doctor`, `tools.future.tabula`, tool registry | Keep exact tool entry points; describe them as Gravity operations. | PLANNED; remove aliases only after caller census reaches zero. |
| SEMANTIC_CHANGE | transient Noetic pipeline vocabulary | NR revision -> physical plan -> NX candidate | `hcli.nomenclature`, artifact loaders, PhysicalGraph and receipts | V1 remains readable; V2 is emitted by new serializers. | IN_PROGRESS; old schemas remain historical compatibility forever. |
| RENAME_ONLY | Singularity / Singularity Profile | selected NX / execution profile | nomenclature module, roadmap prose, historical selection records | Read aliases as `SelectedNX` / `ExecutionProfile`; never rewrite receipts. | IN_PROGRESS; remove public use when bounded terminology check is green. |
| OWNER_CONSOLIDATION | Event Horizon as current subsystem | Gravity repository/tool consolidation | roadmap lineage and archive records | Preserve campaign names and evidence paths. | ACCEPTED; no current runtime owner may be created under this name. |
| OWNER_CONSOLIDATION | Multiple machine-genome representations | Machine facts with specialized measured producers and compatibility projections | `hcli.machine`, `hcli.genomes`, `tools.accelerator.machine_genome`, placement | Preserve freshness and evidence-tier gates; never let a compatibility bag become admission authority. | PLANNED; consolidate only with differential fixture coverage. |
| RUST_PORT | Python graph/state legality and applicability mechanics | Hawking native Gravity core | `hcli.physical_graph`, architecture adapters, backend plans | Native validator requires differential valid/invalid fixtures; Python may remain an independent oracle or forwarding skin. | IN_PROGRESS; delete production duplication after a real caller uses native authority. |
| OWNER_CONSOLIDATION | Model-named reusable execution mechanics | MegaKernel legal-region and backend contracts | Rust kernels, model adapters, PhysicalGraph, recorded Flash fixtures | Model adapters keep genuine math; move only general state/effect/region rules. | IN_PROGRESS; live Flash kernel overlap waits for its committed safe boundary. |
| OWNER_CONSOLIDATION | Method, Law, Scar, observation and verifier copies | Their existing distinct owners, referenced by identity | architecture atlas, Accelerator Law base, negative index, profilers and verifier pipeline | Reject duplicate canonical IDs; retain independent oracles. | PLANNED; no shared database is implied. |

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
