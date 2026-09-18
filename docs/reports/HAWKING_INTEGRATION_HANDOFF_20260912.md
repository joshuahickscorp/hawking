# Hawking integration closeout and continuation handoff

**Canonical local and remote main:** `7608c53ea95d62cabcccb2381502b45b7ab3f9e7`

**Runtime migration:** none. The resident, OpenWebUI processes and ModelLake
watcher were not restarted.

## Landed history

- `0ebc649b3baf182157544a03f0018df89a28b64c` seals the pre-integration
  Hawking, Flash, Gravity, daemon, ModelLake and migration state.
- `ac81444f177915a2f50888893f502408b52d0584` semantically merges the complete
  `codex/hawking-core-unification` history. The merge keeps the newer admitted
  Gravity registry and adds exact revisions/actions, native execution regions,
  the public Hawking action surface, fail-closed Web/serve selection and the
  nomenclature authority.
- `af6eef4dd` seals the next Flash PLE boundary: source-bound BOS PLE formula
  control, pinned-reference parity, reusable PQ construction and a function-aware
  PQ plus sparse-repair negative screen.
- `7d29d8f33` adds bounded automatic Gravity method reuse.
- `7608c53ea` reconciles the PLE frontier with that unified Gravity path and
  refreshes the method-reuse and census evidence.

The original feature branch, commits and
`docs/reports/HAWKING_CORE_UNIFICATION_HANDOFF.md` remain in history as
provenance.

## Names removed from forward authority

No historical file, receipt, schema or compatibility namespace was deleted.
The names removed as peer product/system authorities are:

- HCLI and HIDE as products: forward surfaces are Hawking CLI and Hawking Web;
  the `hcli` package and environment names remain adapters.
- AgentOS as a peer product: its namespace remains internal Hawking capability,
  WorkUnit and authority machinery.
- Deep Gravity and Model/Context/State Gravity as separate systems: these are
  Gravity runs with explicit targets and policies.
- Event Horizon as a current subsystem: it is historical provenance only.
- Singularity and Singularity Profile as forward artifact terms: use selected
  NX and execution profile.
- Doctor and Tabula as product framing: they are Gravity diagnosis and discovery
  operations; their tool aliases remain until their caller census reaches zero.

Ambiguous `compressed model`, `compact model`, `winner`, `best model`, `final
model`, `production model` and `resident model` claims now require inspection
or resolve to NR/NX/Pareto/resident-instance terms. Noetic IR/Program/Compiler
and Hawking Accelerator survive as explicit compatibility interpretations.

## Current degree of unification

The public authority boundary is unified: one Hawking identity, one normal
admitted-artifact registry, exact identity/revision/action resolution, and one
typed Rust public surface for `gravity`, `models`, `actions` and
`select <artifact> execute|serve|web`.

Implementation ownership is partially unified:

| Area | State |
|---|---|
| Public identity and vocabulary | Landed and guarded |
| Admitted artifact selection | One registry and fail-closed exact binding |
| Native Gravity legal regions | Landed; real ratio-zero caller wired; structural plan qualification only |
| Automatic Gravity method reuse | First compatible invocation and incompatible refusal proven |
| CLI/Web action graph | Typed Rust catalog exists; Clap, Python and Web projections are still duplicated |
| Rust/Python PhysicalGraph | Native legality exists; differential forwarding and Python production deletion remain |
| Machine facts | Conceptual authority accepted; producers and freshness projections remain distributed |
| Native serve/profile contract | Shared identity fields exist; native serve and Python adapter remain two execution paths |
| Compatibility namespaces | Still substantial and intentionally retained until caller and packaging gates close |

This is a working unified boundary, not a single implementation throughout the
tree. The next work should thin adapters by caller migration and differential
evidence rather than rename directories.

## Verification and census

- Combined affected Python suite: **119 passed**.
- Native execution-region suite: **10 passed**.
- Real ratio-zero caller: **1 passed**.
- Rust public action surface: **4 passed**.
- Newest Flash MegaKernel boundary test: **1 passed**.
- `cargo check -p hawking -p hawking-core`: passed.
- Terminology base-diff and selfcheck: passed.
- Changed Rust implementation files pass `rustfmt --check`; the broader package
  check still exposes pre-existing formatting drift outside the integration diff.

The final strict census is
`receipts/future/HAWKING_POST_MERGE_CENSUS_20260912.json`:

| Scope | Rust | Python | Denominator | Rust share |
|---|---:|---:|---:|---:|
| Whole active, test-inclusive | 454,034 | 563,563 | 1,017,597 | 44.618253% |
| Minimum product, test-inclusive | 381,550 | 93,396 | 474,946 | 80.335449% |
| Whole runtime, path and inline tests excluded | 369,789 | 450,079 | 819,868 | 45.103480% |
| Minimum product runtime, path and inline tests excluded | 328,178 | 93,396 | 421,574 | 77.845882% |

The strict minimum-product runtime objective exceeds 75%. The whole-tree Rust
share remains open and must not be presented as complete.

## Automatic Gravity reuse evidence

`receipts/future/GRAVITY_METHOD_REUSE_COMPATIBLE_20260912.json` records observed
Flash traits selecting `M-closed-nr-rate-accounting`, passing its input
falsifier, invoking `tools/flash_complete_nr.py`, validating the NR container
and seal, and emitting
`receipts/headless/FLASH_NR_AUTOMATIC_REUSE_20260912.nr.json`.

`receipts/future/GRAVITY_METHOD_REUSE_INCOMPATIBLE_20260912.json` records a
payload-only candidate being refused before invocation. The observation,
method, Law, matching negative-transfer Scar, adapter, verifier and new evidence
remain separate fields.

## Preserved live working-tree frontier

At closeout, `main` has six uncommitted files written after the final Flash
commit:

```text
receipts/future/MODELLAKE_EVENTS.json
receipts/headless/FLASH_NR_ACCOUNTING_BASELINE_20260911.nr.json
tools/flash_complete_nr.py
tools/foundry/GRAVITY_METHOD_REGISTRY.json
tools/foundry/tests/test_query_gravity_methods.py
tools/test_flash_complete_nr_accounting.py
```

These edits carry the new PQ plus fixed sparse-repair output screen into the NR
baseline, add its bounded negative method to the registry and teach automatic
lookup to select it. Preserve and test them before committing. The measured
best point at or below 0.5 projected table EBPW remains high distortion
(`relative_l2=0.850550`); the proposed Scar rejects repeating that exact uniform
PQ plus local top-error repair rule without rejecting generated,
state-conditioned or learned PLE functions.

The immediate scientific frontier remains a materially different PLE function
representation against the sealed output control, followed by broader source
inputs only for survivors. Do not turn this into a broad cleanup detour or claim
complete NR closure, capability or TPS.

## Cleanup inventory

The two integration-owned worktrees are clean and fully merged and may be
removed while retaining their branch refs. The repository also has clean older
worktrees and dirty baselines; do not delete the latter. A stale linked-worktree
record points to missing `/Users/scammermike/Downloads/hawking-product-compression`
and can be pruned.

Six Git bundles were found. They are preservation/campaign artifacts, including
`/Users/scammermike/Archives/Hawking/hawking-packs-event-horizon.bundle` and
copies of the deep-architecture-foundry and paired-authority histories. None was
deleted because archival intent and unique reachable objects were not yet
proven redundant. Audit bundle heads against canonical history before reaping.

<!-- DOC_STATUS: CURRENT -->
