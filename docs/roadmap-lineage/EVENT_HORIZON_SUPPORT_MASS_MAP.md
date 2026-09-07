# Event Horizon support mass map

Measured on `refactor/event-horizon` at HEAD `940b64099`, after the Odyssey
novelty-registry cleanup. This is the support
baseline for the family-reduction campaign; product LOC is frozen unless a
support consolidation requires an ownership correction.

## Accounting

The LOC authority counts tracked source/document lines with known code or
documentation extensions, excluding generated files, vendored files, and
documentation archives. Product is the conservative HCLI plus Rust/shader
implementation boundary used by `tools/loc/hawking_loc.py`. The remainder is
support.

| measure | LOC | files |
|---|---:|---:|
| product | 498,012 | 635 |
| support | 469,985 | 1,385 |
| active total | 968,013 | 2,020 |

The authoritative headline is the value reported by `tools/loc/hawking_loc.py`.

## Top 30 support subtrees

Subtrees are non-overlapping two-component path groups, sorted by counted LOC.
This table accounts for the full support total; the tail outside the top 30 is
also included in the total and must not become an UNKNOWN bucket.

| rank | subtree | LOC | primary disposition |
|---:|---|---:|---|
| 1 | `research/lab` | 85,031 | ACTIVE_RESEARCH / MIGRATE_INTO_OWNER |
| 2 | `crates/hawking-core` | 52,758 | CURRENT_VERIFICATION |
| 3 | `tools/future` | 44,685 | ACTIVE_RESEARCH / SUPERSEDED audit required |
| 4 | `tools/accelerator` | 39,185 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 5 | `hcli/tests` | 19,832 | CURRENT_VERIFICATION |
| 6 | `tools/condense` | 19,211 | CURRENT_CORE_SUPPORT / MIGRATE_INTO_OWNER |
| 7 | `tools/headless` | 14,434 | CURRENT_VERIFICATION / MIGRATE_INTO_OWNER |
| 8 | `tools/odyssey` | 12,659 | CURRENT_CORE_SUPPORT |
| 9 | `tools/verify` | 10,467 | CURRENT_VERIFICATION |
| 10 | `tools/roadmap` | 10,136 | CURRENT_CORE_SUPPORT |
| 11 | `tools/graph` | 9,826 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 12 | `tools/acceptance` | 9,017 | CURRENT_VERIFICATION |
| 13 | `crates/hide-backend` | 7,571 | CURRENT_CORE_SUPPORT |
| 14 | `tools/ascent` | 4,972 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 15 | `research/hawking-experiments` | 4,695 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 16 | `workspace/campaign` | 4,571 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 17 | `tools/theia` | 3,984 | ACTIVE_RESEARCH |
| 18 | `tools/agentos` | 3,981 | CURRENT_CORE_SUPPORT |
| 19 | `docs/ultragoals` | 3,786 | IMPORTANT_ARCHIVE / CURRENT_META |
| 20 | `workspace/docs` | 2,831 | IMPORTANT_ARCHIVE / CURRENT_META |
| 21 | `civilization/build_state.py` | 2,685 | CURRENT_CORE_SUPPORT |
| 22 | `tools/sovereign` | 2,602 | CURRENT_VERIFICATION |
| 23 | `tools/doctor` | 2,361 | CURRENT_VERIFICATION |
| 24 | `tools/foundry` | 2,038 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 25 | `docs/contracts` | 2,013 | CURRENT_META |
| 26 | `tools/ascent_daemon.py` | 1,919 | CURRENT_VERIFICATION |
| 27 | `crates/hide-kernel` | 1,796 | CURRENT_CORE_SUPPORT |
| 28 | `docs/roadmap` | 1,671 | CURRENT_META |
| 29 | `receipts/runtime` | 1,662 | IMPORTANT_ARCHIVE |
| 30 | `tools/qwen38_recon_disc` | 1,465 | IMPORTANT_ARCHIVE / ACTIVE_RESEARCH |

## Disposition rules for the next waves

- `research/lab`: registry-backed operators and their dependency clusters
  survive only when their current scientific owner or verifier is identified;
  unowned campaign chains are candidates for extraction to result/receipt/
  reproducer and deletion. The unrun Qwen30 streamed-source/raw-logit
  preparation chain was compressed to `QWEN30_STREAMED_ORACLE_ARCHIVE.md`; the
  frozen product-side Rust examples were retained.
- `tools/headless`: the concluded N051 sensitivity allocation was already
  fully represented by its sealed receipt and bounded interpretation; its
  one-shot generator was retired.
- `tools/future`: the static CUDA literature adapter was retired after its
  sealed receipt became the canonical record; `science_corpus` retains the
  schema vocabulary needed to read those hypotheses.
  The uncalled Flash NX completeness producer was then retired; its sealed
  audit receipt remains the live causality source.
  The uncalled Q30 null-first reporter was retired; the current activation
  probe/repack and its independent tests remain.
  The uncalled GLM teacher-forced shard merger and producer-only test were
  retired; retained GLM/Frankenstein receipts preserve the prior evidence.
  The uncalled C2M-T3 source-admission census was retired; its sealed project
  receipt and current C2M/AKB validators remain.
  The uncalled fidelity-hierarchy producer was then retired; its sealed
  receipt remains archival evidence.
  The uncalled MAXX resource-pipeline producer was then retired; its sealed
  receipt remains archival evidence.
  The uncalled ExpertFamilyGenome producer was then retired; its sealed
  receipt remains archival evidence.
  The uncalled headless kernel-bottleneck and MLP gate-up producers were then
  retired; their sealed measurement receipts remain archival evidence.
  The broken, uncalled headless dispatch/organ measurement generation was then
  retired; its dispatch, roof, organ, and whole-model receipts remain archival.
  The sealed DeepSeek public-Xet autotune, sustained-transfer, and range-slice
  generators were retired; frozen campaign evidence remains archival and the
  live native header executor remains.
  The closed Odyssey corpus-ingestion sidecar and tests for absent streaming/
  rate-binding producers were retired; its barrier semantics remain in the
  compact disposition note and current ModelLake/HCLI paths remain unchanged.
  Dormant Odyssey T0 tournament/runtime-reproduction and adapter surfaces were
  retired; sealed governance evidence and the accelerator-owned bridge remain.
  The uncalled future specimen-events/workgraph producer was retired; its
  sealed transition receipt remains and Doctor retains only the needed rule.
  The parallel future WorkGraph runtime was then sublated into a bounded
  arrival-payload builder; HCLI `scheduler`/`dag_store` now own scheduling.
  The parallel 22-frontier runtime was subsequently retired; its sealed state
  remains archival while `hcli/frontier_scheduler.py` owns active selection.
- `research/hawking-experiments`: retain decisive findings and minimal
  reproducers; delete superseded executable campaign machinery after checking
  provenance and negative controls. The G1 evidence runners are now retired;
  their result payloads and selected experiment notes remain the archival signal;
  eight unreferenced superseded reports and 22 unreferenced G1 reports were
  compressed into `G1_ARCHIVE_LEDGER.md`; the closed Qwen30 admission-probe
  family was reduced to `QWEN30_CLOSED_ADMISSION_ARCHIVE.md`; eight unreferenced
  Ascension contract/workflow modules were reduced to
  `ASCENSION_ORPHANED_CONTRACTS_ARCHIVE.md`; 41 concluded G1 reports were
  reduced to the complete `G1_ARCHIVE_LEDGER.md` decision record.
  The three former live G1 architecture/Tabula reports were then folded into
  that same ledger and removed as parallel machine-readable authorities; the
  ledger now carries the exact doctrine and closure markers consumed by the
  current readers.
  The duplicate Frankenstein `condense/` wrapper and test tree was then
  retired; live tooling remains in `tools/condense/`, operators remain under
  `frankenstein/operators/`, and all Frankenstein data/evidence is preserved.
  The uncalled Frankenstein prototype operator generation was then retired as
  a 23,915-line cluster; current pipeline, trace/gate, and evidence
  authorities remain,
  and the disposition is recorded in `ASCENSION_PLAN_ARCHIVE.md`.
  The unreferenced numbered audit reports and Odyssey Sub-1 narrative packet
  were then compressed into `receipts/audit/EVENT_HORIZON_ARCHIVE_LEDGER.md`;
  durable findings, evidence classes, negative-science boundaries, and
  retention rules remain while 28 historical narratives leave active HEAD.
  Two dated ascent snapshots and the test-only Q80 capture-coverage auditor
  were then compressed into `receipts/ascent-2026-08-18/ARCHIVE_LEDGER.md` and
  the existing Ascension archive; the live capture index remains authoritative.
  The test-only retired-controller shell harness was then removed; current
  campaign-engine and HCLI verification tests remain.
  Two orphan research-registry/source-verifier helpers and their dedicated
  tests were then retired; current Gravity/Doctor capture authorities remain.
  The optional Grok novelty engine was then retired as disabled-by-default
  external scaffolding; deterministic Odyssey/HCLI lanes remain authoritative.
  The one-shot HCLI persistence audit runner was then retired after its sealed
  receipt became the evidence owner; live HCLI persistence code remains.
- `research/lab/operators`: the blocked Qwen scientific optimizer watcher had
  no live caller or material physical state; its implementation, launcher, and
  dedicated fixture were deleted while its blocked-runtime claim boundary was
  retained in `QWEN_SCIENTIFIC_OPTIMIZER_ARCHIVE.md`.
  The dependent Ascension V3 lifecycle/campaign, physical gatekeeper,
  tournament, notifier, source-admission, sandbox, launch-gate, launchd, and
  test chain was then retired as one broken legacy family: 14,997 active LOC
  disappeared while current manager-protocol surfaces, Qwen30 physical
  research, and receipts remained.
  The uncalled GLM52 Gravity selection/benchmark/fixture and repack-score
  cluster was then removed as a research-only generation: 3,393 operator LOC
  plus its dedicated test harnesses disappeared, while the current
  `gravity_metal` decoder and archival evidence boundaries remained.
  The uncalled Qwen state-KV and Q80 mixed-representation experiments were
  subsequently removed as a second research-only cluster; active Qwen30
  activation-weighted and Q80 capture-index owners remain. The orphaned
  physical-metric audit producer and its self-contained test were retired after
  sealed audit/canary receipts became the evidence owner; the current EBPW
  category gate remains.
  The standalone retirement-gate producer and receipt-only test were retired;
  `tools/odyssey_ctl.py` remains the live retirement authority and the sealed
  QWEN retirement receipt remains archival.
  The obsolete Odyssey Pareto producer and receipt-only test were retired after
  the current HCLI/Flash Pareto frontier ledger became the active selection
  owner; `PARETO_ARCHIVE.json` remains archival.
  The uncalled DeepSeek schedule/oracle, residual-teacher admission, and Qwen
  metadata-preflight cluster was then compressed to
  `STALE_RESEARCH_CLUSTER_ARCHIVE.md`; no live DeepSeek stream or Qwen capture
  owner was changed.
  The duplicate, uncalled GLM52 activation-pack v2 generation was retired;
  v1 remains the active pack owner and the campaign registry was corrected.
  The superseded H-ROADMAP copy was then removed from active HEAD; its digest,
  provenance, and recoverability through Git history remain in PRESERVATION.md,
  while the operator-owned current roadmap remains the only resolver authority.
  The unreferenced per-lane ascent runbooks were then collapsed to the compact
  planning archive; shared density/common references and sealed findings remain.
  The uncalled Python Qwen3.8 genesis-child manager and its CLI shim were then
  deleted as an orphan execution pair; the Rust `ascension_qwen38_shared_sessions`
  example remains the executable authority for that capability. The dead future
  orchestration bindings for retired launch/specimen producers were removed,
  and the orphan Odyssey decoding producer was deleted; its durable G038
  receipt and independent receipt tests remain. The uncalled future
  orchestration connector was then retired as well; its binding receipt remains
  archival while HCLI/AgentOS owns active scheduling.
  The uncalled Odyssey transfer-rehearsal producer and its producer-only tests
  were retired; QWEN/FLASH transfer receipts remain as historical evidence and
  the current Odyssey II harness remains separate.
  The uncalled Odyssey resident-seal generator and its self-contained mutation
  tests were retired; the sealed resident receipt and current HCLI/AgentOS
  resident verification remain authoritative.
  The superseded Odyssey G037 state-gravity producer and its coupled test were
  removed; the current headless N048 state-gravity authority remains.
  Two uncalled Odyssey worker scripts were also retired, and their stale process
  needles were removed from qualification; the matched-bits and teacher-capture
  receipts remain archival.
  The superseded Odyssey G005/G013 performance producer cluster was then
  retired: performance qualification, GPU cleanliness, perf addendum, the
  private protected-window helper, and their coupled tests. Their sealed
  measurements remain archival; HCLI/AgentOS and accelerator qualification own
  current protected benchmarking.
  The uncalled Q80 activation-weighted repack/readiness/null-first research
  cluster and its dedicated coherence test were retired; the active Q80 capture
  index and Q30 activation-weighted owner remain.
- `tools/headless`: three uncalled Noetic/bandwidth producer runtimes were
  retired; their sealed findings remain under `receipts/headless/`, and the
  remaining adversary evidence is receipt-backed after its broken
  Noetic/VisionMCP runners were retired. The uncalled Qwen3.8 native benchmark
  launcher was then retired; its timing, coherence failure, and blocked-run
  evidence remain in the sealed QWEN38 native receipts and performance ledger.
- `tools/condense`: the non-live HCLI product-suite scaffold and its dedicated
  test were retired; current HCLI acceptance and DeepSeek suite evidence remain
  authoritative.
- `tools/headless`: the duplicate test-only GrokBridge harness was retired;
  `hcli/tests/test_grok_bridge.py` remains the live contract test owner.
- `tools`: the uncalled G11 Matryoshka NumPy demonstration was retired; its
  measured result remains in the ascent archive and no runtime owner changed
  (1,043 counted LOC removed).
- `tools`: the superseded Qwen3.8 activation-capture-v2 producer and dedicated
  test were retired; sealed capture findings and current Qwen30/Q80 capture
  authorities remain (2,187 counted LOC removed).
- `tools/headless`: the superseded 32-entry negative-science archaeology/sweep
  producer was removed. Its v2 receipt is already generated by the canonical
  nine-field `negative_science.py` store and remains indexed by
  `tools/future/negative_index.py`; the receipt entries and reopen evidence are
  preserved (2,207 counted LOC removed).
- `tools/condense`: the uncalled full-sequence capture shard merger was retired;
  its measured bit-exact parallelism result and reproduction command remain in
  the research evidence archive. The live GLM shard merger remains because its
  probe and test import it (391 counted LOC removed).
- `tools/condense`: the uncalled GLM teacher-forced parallelism probe was retired;
  its measured result and two-worker command remain in the research evidence
  archive. The underlying shard merger and its test remain live (498 counted
  LOC removed).
- `research/lab/tests`: tests for already-deleted Ascension supervisors and
  absent JSON contract fixtures were removed; current manager-protocol and
  Qwen30 physical tests remain. The uncalled manager-tournament readiness /
  protocol pair and generic parity-ladder scaffold were then retired as one
  family; current Rust paired-cognition and Doctor authorities remain.
- `crates/hawking-core/examples`: twelve uncalled Ascension packing, capture,
  drift, and contract programs were retired as historical executable
  scaffolding; current Flash acceptance examples and runtime modules remain.
  The standalone Qwen80 multi-layer completion assessor and its adversarial
  harness were also removed because no current acceptance or registry path used
  them.
- `crates/hawking-core/tests`: the versioned v030–v2s / Phase-2 / draft /
  prototype parity generation was retired as superseded standalone scaffolding;
  current named verification tests remain. The unreferenced low-level
  predecode/Q4K/RoPE/fusion micro-test generation was then retired; the two
  MHA tests named by the active prefill-KV census remain.
- `crates/hawking-core`, `tools/verify`, `tools/acceptance`, `tools/audit`, and
  `hcli/tests`: current verification is a valid survival reason; consolidate
  duplicated plumbing only when proof independence remains intact.
- `tools/future`, `tools/headless`, `tools/condense`, and `tools/odyssey`: audit
  complete dependency clusters against the current HCLI/Rust owner. A caller
  alone is not sufficient evidence of survival.
- `workspace/*`, `docs/*`, and `receipts/*`: classify information separately
  from executable machinery. Preserve current metadata, decisive evidence,
  Laws/Scars, and necessary provenance; compress or retire stale execution
  scaffolding where a smaller canonical representation carries the signal. The
  superseded TG `matched/` receipt root is now represented by canonical
  `matched-v2/`; its v1 timing caveat remains in the v2 evidence root.
  Generated governance rung snapshots are counted outside the active LOC
  authority; three intermediate before/after pairs were compressed to the
  canonical comparison pair, final verification snapshot, and
  `GOVERNANCE_RUNG_ARCHIVE.md`.

The next checkpoint is support <=500K, then <=250K, with the current product
boundary held at approximately 498K.

The current wave removed 470 counted LOC from the uncalled Qwen3.8 native
benchmark launcher. It also removed an exact duplicate of the pinned Python
requirements freeze from scaffolding; `.txt` is outside the active LOC
authority, so that 459-line physical reduction is tracked as file/namespace
compression rather than target LOC. `workspace/docs/plans/studio_pinned_requirements.txt`
is the sole canonical copy.
