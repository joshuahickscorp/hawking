# Gravity tool-surface reduction

This is the canonical reduction doctrine for HCLI's operating surface. The
historical phrase "Tool Gravity" may remain as a compatibility label, but the
concept is an internal reduction operation under Gravity—not a second
architecture and not a request to execute destructive cleanup immediately.

## Objective

Minimize interface-selection cost while preserving semantic capability,
authority, evidence, provenance, recovery, and physical usefulness:

```text
minimize schema/name/provider surface
subject to capability parity, authority parity, receipt parity,
caller reachability, and measured execution/recovery improvement
```

The optimization target includes first-call correctness, invalid-call
recovery, catalog/schema tokens, wall latency, supervision rounds, and
resident memory. A lower tool count that loses a producer, hides a required
argument, or merges mutation classes is a failed reduction.

## Primitive vocabulary

The target is a small typed vocabulary, not one god tool:

| Family | Semantic verbs | Boundary that remains explicit |
|---|---|---|
| file | `read`, `search`, `list`, `write` | read roots and reversible write roots |
| source | `search`, `fetch`, `acquire` | public evidence versus costly acquisition |
| run | `readonly`, `exec` | no shell interpreter or unbounded command escape |
| measure | `inspect`, `run` | read-only inspection versus costly execution |
| evidence | `read`, `inspect`, `record` | sealed evidence versus new state |
| campaign | `inspect`, `record`, `decide`, `checkpoint`, `escalate` | HCLI confirmation and WorkUnit accounting |
| gravity | `inspect`, `search`, `transform`, `verify`, `compile`, `measure` | representation, execution, and promotion gates |

These are target names. The existing registry remains the compatibility
authority until the migration receipt proves each replacement. The current
before-state is `receipts/future/GRAVITY_TOOL_MAP.json`.
The measured registry/catalog baseline is
`receipts/future/GRAVITY_TOOL_BENCHMARK_BASELINE.json`; it is the comparator
for every later reduction.
The current benchmark schema is v2 and records integer nanoseconds; the older
v1 receipts retain floating-point seconds as historical comparators only.

## Non-negotiable invariants

1. Every registered spelling is inventoried before mutation: schema, result,
   implementation owner, callers, tests, mutation class, resource use,
   authority, provenance, unique semantic capability, and replacement.
2. A merged surface preserves each member's per-operation schema and required
   arguments. An unknown operation is an actionable refusal, not a guessed
   invocation.
3. A merged surface cannot combine incompatible authority classes. Read-only,
   research, reversible, costly, external, and destructive paths stay
   distinguishable to HCLI and to the receipt.
4. Provider names and historical aliases may disappear from discovery only
   after their callable compatibility and caller migration are proven.
5. `shell.readonly` never becomes `shell.exec`; `source.fetch` never silently
   becomes `source.acquire`; repository inspection never becomes repository
   landing.
6. No generic `shell`, `exec`, `do_anything`, or arbitrary provider proxy is
   admitted as a substitute for typed primitives.
7. The result envelope always preserves operation, invocation identity,
   mutation class, elapsed time, error class, provenance, and artifact
   identity. Reduction cannot hide a failure or credential boundary.
8. Capability reduction and refusal reduction are separate measurements. Tool
   consolidation must reduce unnecessary refusal caused by unreachable names,
   missing arguments, or needless provider distinctions, while retaining
   narrow refusals for unauthorized, destructive, privacy-invasive, or
   epistemically ungrounded actions.

## HCLI and Gravity relationship

HCLI owns authorization, retention, WorkUnits, mutation confirmation, result
envelopes, and the final reachability boundary. Gravity owns the semantic
meaning of scientific operations and their evidence. The HCLI registry is the
single registration authority; an internal implementation may have many
modules, but it must not create peer public verbs for one capability. A
Gravity-facing tool should expose the small `gravity.*` family while selecting
internal diagnostics, codecs, Nova transformations, kernels, or execution
strategies behind the typed operation.

ModelLake is represented as `model.inspect` and `model.acquire` concepts, with
storage/provenance remaining distinct from network acquisition. Odyssey is a
campaign coordinator, not a second scientific tool universe. Gravity absorbs
scientific inspect/search/transform/verify/compile/measure operations while
preserving their resource and authority classes. Vision/perception providers
remain allowlisted typed adapters behind `perception.inspect/query`.

## Migration stages

| Stage | Work | Exit evidence |
|---|---|---|
| TG001 | Inventory all registered names and aliases | complete machine-readable tool map |
| TG002 | Identify duplicate aliases and provider histories | alias graph and caller counts |
| TG003 | Cluster by semantic capability | one proposed owner per capability |
| TG004 | Classify read/write/costly/external/destructive authority | no mixed-class merge |
| TG005 | Define target typed primitives and schemas | schema and result contracts |
| TG006 | Add compatibility dispatch | old names remain callable and tested |
| TG007 | Migrate HCLI/AgentOS routing | production callers use canonical surfaces |
| TG008 | Migrate source/test/receipt callers | zero stale active imports or invocations |
| TG009 | Reduce the model-facing catalog | selection benchmark improves or holds parity |
| TG010 | Run a small resident benchmark | before/after receipt with recovery metrics |
| TG011 | Deprecate aliases | warning/receipt and explicit sunset condition |
| TG012 | Delete only retired aliases and duplicate wrappers | zero callers, parity regressions pass |
| TG013 | Recompute interface budget | catalog, latency, memory, and supervision budget close |

TG001 is complete for the current HCLI registry. TG002–TG005 are checkpointed
as a proposal in the map and migration table. A first non-semantic reduction
pass is also sealed: `ToolRegistry` now lazily indexes canonical specs,
compatibility aliases, and focused search text, invalidating only when a tool is
registered. The pass preserved the 109/43/66 callable/discoverable/alias
counts and all catalog byte counts while improving repeated discovery and
focused lookup; see
`receipts/future/GRAVITY_TOOL_BENCHMARK_AFTER_INDEX_PROMPT_20260910.json`.
The current nanosecond refresh is
`receipts/future/GRAVITY_TOOL_BENCHMARK_NS_20260911.json`.
TG006 onward still requires caller-level validation before alias retirement.

## Interface budget

Every new public operation must answer:

```text
What existing primitive cannot express this operation?
What unique semantic capability does it add?
What authority class and resource does it require?
Which caller, test, receipt, and falsifier prove it is reachable?
What is the measured token/latency/recovery cost?
```

If those answers are absent, the operation remains an internal helper or an
experiment receipt rather than a new public verb.

## Refusal and security posture

This reduction is intended to make HCLI a more capable defensive security and
red-team instrument by removing accidental refusal: a typed, authorized
`source.fetch`, `run.readonly`, `measure.run`, `gravity.verify`, or
`campaign.checkpoint` should be reachable without provider-name trivia. It
does not authorize credential theft, unauthorized access, malware deployment,
privacy invasion, destructive mutation, or unverified claims. Authorization
must be declared at the tool boundary, and destructive work must remain
confirmable, auditable, and recoverable.

The focused-discovery index now treats `authorized`, `defensive`, `security`,
and `red-team` as bounded retrieval terms for the existing audit, capability,
claim, shell, Git, test, and Gravity primitives. This fixes the concrete
zero-result path where the capability existed but ordinary security language
could not find it; it does not merge their mutation classes or widen any
permission.

Copyright handling is provenance-first: public evidence may be fetched with
source attribution; user-owned material may be transformed within scope; and
the system should avoid reproducing unavailable third-party text beyond the
permitted transformation. That boundary is typed evidence/provenance policy,
not a blanket refusal of research, analysis, or user-provided transformation.

## Checkpoint

The current checkpoint is
`receipts/future/GRAVITY_TOOL_GRAVITY_CHECKPOINT.json`. It records the before
state, the active KIMI/Gravity work, and the next safe action. It does not
authorize deletion or an unbounded registry rewrite.

The current measured surface is 110 callable spellings, 44 discoverable
model-facing tools, and 66 compatibility aliases. The focused security query
now returns typed primitives. The pre-index comparator is
`receipts/future/GRAVITY_TOOL_BENCHMARK_CURRENT_20260910.json`; the latest
capability-parity index and compact-prompt pass is sealed in
`receipts/future/GRAVITY_TOOL_BENCHMARK_AFTER_INDEX_PROMPT_20260910.json`.
Its current v2 nanosecond timing receipt is
`receipts/future/GRAVITY_TOOL_BENCHMARK_NS_20260911.json`; the fresh medians
are 760,042 ns for registry construction, 16,291 ns for discovery, and
20,533,958 ns for compiling `hcli/tool_registry.py`. The receipt's report
calculations use integer nanoseconds; any seconds values are compatibility
views only.
