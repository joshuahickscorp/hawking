# Gravity Compatibility Map

This is the migration authority for historical names. Compatibility status is
deliberately explicit so KIMI and existing HCLI callers can cross boundaries
without a blind rename.

## Reduced conceptual surface

The canonical nouns are **Hawking**, **HCLI**, and **Gravity**. Odyssey remains
a campaign and AgentOS remains infrastructure. **Deep Gravity** is the mode
that searches learned cognition, representation, and physical design together;
it is not a peer subsystem. **Nova** is the Gravity operation that renews the
learned organism through training or structural transformation and records
parent-to-descendant lineage. **Noetic Representation** is Gravity's mutable
representation substrate, and NX is Gravity's machine-bound output.

Gravity also owns the search across source discovery, anatomy, capability and
behavior diagnosis, authorized adaptation, complete accounting,
representation, compilation, runtime/kernels, CPU/GPU/ANE/UMA placement,
context/cache economics, receipts, controls, falsification, and transfer.
Tool/repository/build reduction uses the same law. These are operations under
Gravity, not additional products.

The table below maps implementation history into that small surface. It does
not authorize mass moves or deletion: internal folders and compatibility names
may remain until ownership, caller reachability, receipt parity, and recovery
are proven.

| Old concept / surface | Canonical Gravity owner | Compatibility status | Delete or retire when |
|---|---|---|---|
| Doctor | Gravity internal `DISCOVER` / `DIAGNOSE` tags | Keep command/API aliases; route implementation through canonical owners | All callers use the Gravity owner and the Doctor alias has a tested deprecation path |
| Abliteration | Gravity `SHAPE`; Nova when permanent weight surgery changes cognition | KIMI `tools/future` implementation remains live-compatible until receipt migration | Shape/Nova owner reproduces Tabula behavior and KIMI receipt is sealed |
| Refusal analysis | Gravity `DIAGNOSE` | Keep KIMI diagnostic entry point; emit canonical diagnostic receipts | Canonical diagnostic API covers held-out, causal, routing, and controls |
| Tabula projection/write owner | Gravity `SHAPE`; Nova for learned-organism mutation | Existing `tools/future/tabula.py` is the current write owner | A canonical Gravity module imports/adapts it with regression parity |
| Doctor/Tabula as independent science | Ordinary Gravity discovery/diagnosis; Nova only when learned state changes | Preserve historical receipts and implementation aliases, not peer authority | Current canon and callers no longer present them as a separate science |
| PQ | Gravity `REDUCE` / Noetic Representation | Codec parameters remain data/configuration | All PQ callers use the common representation contract |
| `pq_test_v2`, `pq_real`, `pq_new`, `pq_kimi`, `pq_sweep`, `pq_probe`, `pq_fixed`, `pq_d16`, `pq_d32` | Gravity representation experiments | Inventory/receipt only; no new peer architecture | Unique findings become receipts/tests/configs and imports reach zero |
| Noetic | Gravity's Noetic Representation substrate | Preserve public term; consolidate duplicate graph types | One representation graph is authoritative |
| Historical `NE` representation spelling | Noetic Representation (`NR`) compatibility alias | Preserve only in sealed contracts/readers that require it | Active writers and user-facing selectors emit NR; historical evidence remains readable |
| NX construction/search | Gravity execution/output production | Keep artifact readers and compatibility commands | Search logic is owned by Gravity and NX is only a produced artifact |
| NX executable | Gravity-produced machine-bound artifact | Preserve `.nx` readers and provenance | Never delete; retire only duplicate producers |
| PhysicalGraph | Internal Gravity lowering/execution graph | Preserve graph serialization/API while moving ownership | Canonical Gravity lowering owns all active callers |
| MachineGenome | Internal Gravity machine descriptor | Durable descriptor; no migration deletion planned | Not applicable; only duplicate schemas retire |
| CPU/GPU/ANE/UMA/Metal/backend tuning | Gravity physical execution | Keep backend adapters, disclose the actual provider, and preserve regressions | Duplicate benchmark/control paths are consolidated after parity |
| TPS tuning | Gravity physical measurement objective | Preserve measurements; do not treat TPS as capability proof | Every benchmark emits common physical receipt fields |
| Complete EBPW implementations | Gravity accounting/verification | Inventory `complete_ebpw`, category, census, and gauntlet callers before merge | One reconciled accounting API passes all retained tests |
| Model anatomy scripts | Gravity `DISCOVER` anatomy | Keep receipts and compatibility adapters | One anatomy API covers tensor, state, modality, and routing facts |
| Routing/locality probes | Gravity `DIAGNOSE` routing | KIMI locality code remains active and receipt-bound | Common routing statistics owner reproduces KIMI evidence |
| Experiment-only Python probes | Receipt/config/fixture under the relevant Gravity phase | Freeze while useful evidence is migrated | Receipt/fixture/regression exists and importer count is zero |
| H-MATH / representation algebra | Gravity cheap-falsifier machinery | Keep `GRAVITY-ALGEBRA.md` and compatibility references; no peer subsystem | Equations, break-even tests, and numeric falsifiers route through the common experiment owner |
| Repository/tool/build/context reduction | Gravity applied to Hawking itself | Small measured migrations only; preserve compatibility and archaeology | Canonical owner, parity test, rollback, and before/after cost exist |
| Ad hoc receipts | Common Gravity experiment receipt protocol | Preserve sealed historical evidence; add schema adapters | No active reader depends on an ad hoc schema |
| GravityForge/GravityLab/GravityBench naming variants | Relevant Gravity phase + common harness | Compatibility aliases only where callers exist | Canonical phase/harness has migrated callers |
| Odyssey | Scientific campaign over organisms invoking Gravity | Keep as campaign identity | Not a duplicate of Gravity; no deletion planned |
| Pareto | Gravity frontier/history mechanism | Keep frontier data/API | Not a duplicate optimizer |
| Singularity | HCLI selector over qualified Gravity alternatives | Keep selector and boundary | Not a duplicate search engine |

KIMI evidence remains explicitly split during migration: raw-model behavior,
deterministic response repair, the resident HCLI serving contract, and any
learned-weight descendant are different classes. The recorded OPA--OPH family
currently has no promotable transformation; a nearby renamed sweep does not
reopen it. This does not block admission of an unchanged-weight operational
body that passes its own task/runtime/authority gates, and it does not qualify
future changed weights.

## Tool Gravity surface map

| Existing HCLI surface | Target semantic owner | Compatibility status | Delete or retire when |
|---|---|---|---|
| `fs.*`, `filesystem.*` | `file.read/search/list/write` | Keep typed aliases; preserve read roots and reversible write permission | Every caller uses the file primitives and window/list regressions pass |
| `web.*`, `github.*`, public source adapters | `source.search/fetch` | Keep provider aliases; source provenance remains in the result | Provider names have no active caller and source receipts retain URL/time/hash |
| `huggingface.*` metadata and download | `source.fetch` / `source.acquire` | Keep acquisition separate from research and costly | Both paths have independent authority and acquisition receipts |
| `lake.*`, `modellake.*`, `specimens.*` | `model.inspect` / `model.acquire` | Storage/provenance remains distinct from network acquisition | ModelLake callers and sealed registry tests migrate |
| `shell.readonly` / `shell.exec` | `run.readonly` / `run.exec` | Must remain two authority classes; no god shell | Typed runner parity and negative write controls pass |
| `receipt.*`, `roadmap.*`, evidence readers | `evidence.read/inspect/record` | Preserve sealed receipt schemas and write gates | All readers use the common envelope and no stale readers remain |
| `tests.*`, `benchmark.*`, `accelerator.*`, `physical.*` | `measure.inspect` / `measure.run` | Keep costly and read-only paths distinct; unify result contract first | Before/after resident benchmark shows equal/better capability and physical cost |
| `odyssey.*` | `campaign.inspect/record/decide/checkpoint/escalate` | Existing consolidation remains callable; Odyssey identity stays | Campaign callers migrate without losing WorkUnit or confirmation semantics |
| `audit.*`, `claim.attack`, `tool.reachable`, capability gates | `verify.inspect/attack/reachability` | Preserve fail-closed falsification and stale-verdict detection | All verification receipts point to one authority |
| `gravity.*`, `doctor.*`, `nr.*` | `gravity.inspect/search/transform/verify/compile/measure` | Gravity owns science; HCLI owns authorization and accounting boundary | Canonical typed Gravity owner reaches all retained tests |
| `vmcp.*` and provider perception | `perception.inspect/query` | Keep allowlisted adapter boundary; never expose the full provider surface | Adapter result/provenance parity and profile tests pass |
| `processes.*` | `host.inspect` | Observation remains read-only; signal/reap stays outside model reachability | Host inspection contract has one owner and process tests pass |
| `git.*` | `repo.inspect` plus separate `repo.land` | Inspect and mutation remain separate | Landing verifier and recovery rules have one owner |

The complete registered-name inventory, implementation owner, caller/test
references, authority, and proposed replacement is generated in
`receipts/future/GRAVITY_TOOL_MAP.json` by
`tools/audit/gravity_tool_map.py`. The map is a before-state receipt, not a
deletion list.

## KIMI bridge

KIMI is undergoing **Deep Gravity**. The active workstream is deliberately not
renamed in place; its boundary is:

```text
KIMI_BASE (immutable)
  → Gravity: diagnose refusal/routing evidence
  → Gravity: shape KIMI_OPERATOR_CANDIDATE_*
  → Gravity: verify capability/epistemic/authorization/HCLI/physical gates
  → Gravity: represent/execute/physically optimize when a candidate qualifies
  → Nova only if learned cognition or architecture must change
```

No operator is promoted from the static catalog. The live evaluation receipt
must bind the measured direction, layer/expert scope, strength Pareto, parent
hash, candidate hash, code commit, and all retained controls before crossing
into Gravity's downstream artifact pipeline.
