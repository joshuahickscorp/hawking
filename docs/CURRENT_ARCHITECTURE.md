<!-- DOC_STATUS: CURRENT -->
# Current architecture

Updated 2026-09-04 on the Event Horizon refactor branch. This document is the
human-readable architecture source of truth. Census files, capability graphs,
and receipts are point-in-time evidence; they may describe an older commit and
must not be treated as a second architecture map.

## Operating shape

```text
Hawking Web / Hawking CLI
          │
          v
       hawkingd ── objectives, tools, work units, memory, gates, receipts
          │
          v
       Gravity ── discovery, collapse, compilation, planning, qualification
          │
          v
 NR revision -> physical plan -> backend programs -> NX -> resident instance
```

Hawking is the single product/system name. `hawkingd` owns persistent local
operation and may supervise multiple descendants while retaining one sovereign
process tree. Python currently owns comparative-advantage orchestration and
resident supervision; the Rust `hcli` binary in `hide-backend` owns the durable
backend protocol, tools, sessions and runtime-facing composition. These HCLI,
HIDE and AgentOS names remain implementation and compatibility identities while
their public product framing migrates to Hawking. Rust owns model execution and
serving. A receipt or plan never raises the capability ceiling of an unmeasured
artifact.

Gravity is the single discovery, analysis, transformation, compilation,
physical-optimization and qualification function. Doctor and Tabula are
diagnostic/discovery operations within Gravity. Odyssey is its population
science; ModelLake is source/specimen storage and provenance. The implementation
crosswalk and compatibility sunset conditions live in
`docs/HAWKING_NOMENCLATURE.md`.

## Ownership map

| Concern | Canonical home | Boundary |
|---|---|---|
| Hawking CLI and command ingress | Python `hcli.cli`, `hcli.commands`, `hcli.controller` plus Rust `hawking` and compatibility `hide-backend`/`hcli` | One typed Hawking action surface is the target; compatibility names must preserve identical authority |
| Internal capability and lifecycle machinery | `hcli.agentos` plus canonical `hcli.goal`, `hcli.workunit`, `hcli.scheduler`, `hcli.mission`, `hcli.verifier_pipeline` | AgentOS is an internal namespace; scheduling/proposal is not verification |
| Runtime and provider execution | `hcli.runtime`, `hcli.engine`, `hcli.backends`, `hcli.session`, `hcli.models` | Provider output is evidence only after the verifier accepts it |
| Crash-safe persistence | `hcli.persist` | The shared text/bytes/JSON atomic writers; specialized compare-and-swap remains in its owner |
| Doctor diagnosis | `tools.doctor.engine` and `tools.doctor.*` | Metadata/receipt diagnosis; no weight loading or hardware claim |
| Gravity | `hcli.gravity`, `tools/gravity_*.py`, model adapters, Rust runtime crates and verifier owners | Representation search, compilation and physical optimization share one function; evidence scopes remain distinct |
| Status verification | `tools.verify.status_causality` | A status may assert only what its actual probe establishes |
| Roadmap and reachability | `tools.roadmap`, especially `tools.roadmap.capability_reachability` | Definitions/imports are not calls; receipts are citations, not callers |
| Odyssey and ModelLake | `tools.odyssey`, including `tools.odyssey.modellake_promote` | Specimen lifecycle and promotion stay separate from Doctor/Gravity |
| Sensory evidence | `hcli.vmcp` and the `visionmcp/` package | External VMCP surface; no duplicate sensory authority |
| Machine/runtime identity | `hcli.machine`, `hcli.genomes`, `hcli.runtime_iface` | Identity and learned metadata do not authorize physical results |
| Visual/IDE/ACP surface | Deferred; rebuild behind hardened VMCP | No active visual authority is shipped in this phase |

`hcli.agentos` is a public namespace and ownership surface; it is not a second
copy of the core lifecycle classes. Goal compilation, WorkUnit identity,
Mission state, and verifier orchestration each retain their established
canonical modules.

`tools/future/` is now a small retained set of call-path sidecars, not a second
product authority. The uncalled producer/test farm was removed; its fixtures
and receipts remain available for audit and reproducibility. The retained
scientific metabolism and resident-supervisor records use explicit sidecar
names so they cannot be mistaken for `hcli.workunit.WorkUnit` or the live
resident control loop. The dated `tools/hcli/bootstrap/` notes were retired;
their disposition is recorded in
`workspace/docs/plans/ASCENSION_PLAN_ARCHIVE.md`.

## Rust workspace boundary

The default build surface is the Hawking inference/serving workspace plus its
context, index, orchestration, research, event, adapter, and bake tools. HCLI
backend/core/protocol crates are workspace members but are outside
`default-members`. `hawking-serve` serves the runtime; Rust `hcli` composes the
HIDE backend against it. The old visual `hide-serve` transport and ACP server
are intentionally absent from the active workspace.

Cargo metadata currently shows six Hawking-to-HIDE edges through shared errors,
IDs, blobs, and UI event types (`hawking-context`, `hawking-index`,
`hawking-orch`, `hawking-research`, and `hawking-events`). They are documented
debt, not silently reclassified as clean layering. Removing those edges needs a
separate neutral-primitive/UI-vocabulary change and is outside the no-physical-
optimization refactor lane.

The undeclared `crates/hide-backend/src/hcli/` tree, its `hcli-backend` wrapper,
and its integration test were removed in Phase III. They had never built and
were not a second runtime authority. The live Rust HCLI surface is the declared
`hcli` binary plus its `hcli_bridge`, profile, research, source, and swarm
modules. The previous frontend, `hide-serve`, and `hide-acp` are removed from
the product branch; their replacement condition is a hardened VMCP boundary.

## Deferred visual boundary

The desktop/frontend and editor-facing HIDE layer is not a current product
requirement. It was removed from this branch together with its dedicated
localhost transport and ACP server so it cannot form a second command or state
authority. Rebuild it only after VMCP is hardened, using the Rust HCLI backend
and `hide-protocol` as the contract boundary.

## Authority rules

1. `hcli.workunit.WorkUnit` is the live WorkUnit identity. Scientific sidecar
   records must use a distinct name when their fields are not the HCLI shape.
2. `hcli.persist.atomic_write_text`, `atomic_write_bytes`, and
   `atomic_write_json` own ordinary crash-safe writes. Compare-and-swap lineage
   writes remain specialized and are not collapsed into last-writer-wins I/O.
3. `tools.verify.status_causality` owns probe-to-status entailment. Historical
   receipt schema strings remain readable; new executable callers use this path.
4. `tools.roadmap.capability_reachability` owns static reachability analysis.
   Its Rust parity binary is an accelerator for the same facts, not a second
   semantic verdict.
5. `tools.doctor.engine` owns Doctor diagnosis. `doctor.query` dispatches to it;
   `gravity.experiment` dispatches to the bounded Flash representation runner
   only when explicitly requested.
6. `tools.odyssey.modellake_promote` owns ModelLake promotion policy. A sealed
   specimen, registry entry, or benchmark receipt alone is not promotion.
7. Physical performance claims require a protected window, live samples,
   independent verification, negative controls, and reproducible provenance.
   This refactor changes structure and callers only; it does not optimize a
   kernel, acquire weights, or reinterpret a static result as measured speed.

## State and evidence placement

- `civilization/` holds roadmap and launch-goal state.
- `receipts/` holds acceptance, provenance, and preserved historical evidence.
  Sealed bytes are not rewritten merely to make names prettier.
- `workspace/`, `.hcli/`, Cargo targets, Python caches, and campaign runtime
  outputs are generated/local state and belong outside the source authority.
- Model weights and external ModelLake payloads are local inputs, never source
  files in this repository.

## Entry points and verification

```bash
python3 -m hcli --help
python3 -m hcli.agentos.resident --help
python3 -m tools.doctor --selftest
python3 -m tools.roadmap --help
cargo run -p hide-backend --bin hcli -- --help
cargo check --workspace
python3 -m pytest
```

The default pytest target is the complete live `hcli/` package. Current
acceptance and audit tests invoke retained sidecars directly; there is no
separate uncalled `tools/future/` test harness.

For isolated work, set `CARGO_TARGET_DIR`, `PYTHONPYCACHEPREFIX`, and the
pytest cache directory to a temporary refactor-specific location. Compare
before/after census counts, test results, Rust checks, CLI smoke, HCLI
restart/resume, protected verifiers, capability-graph/roadmap checks, VMCP,
ModelLake identity, NR/NX, and negative controls. A failure that predates the
refactor remains classified as pre-existing until its cause changes.
