# aud02 — roadmap facet reaudit

Discovery pass only. Nothing in `H-ROADMAP.md`, `civilization/`, `hcli/`, `tools/`, or `crates/` was rewritten. This note is the human summary of `receipts/audit/aud02-roadmap-facet-reaudit.json`.

## Answer

The committed capability graph (`civilization/CAPABILITY_GRAPH.json`, generated from commit `7d642800`, **7 commits behind HEAD `04193ccbc`**) is a better call-site instrument than the lane that counted imports, and it is still not a 13-state truth table.

- **Appendix O** (H-ROADMAP.md:9455–9528) names **71 gates** as an unfilled `NOT_RUN|PASS|FAIL|BLOCKED|INCONCLUSIVE` enum. A gate name is not evidence (line 9529). The graph’s 18 `BUILT` marks are **not** written into the roadmap.
- **Appendix A** (5223–7269) has **25 gene cards**. Every card’s AUTHORITY / PHENOTYPE / MUTATION / … / PROMOTION block is the **same two sentences**, copied **325 times** each. That is `STALE_ROADMAP_TEXT`.
- **Independent 13-state (this lane):** no gate is `PHYSICALLY_MEASURED`, `ADVERSARIALLY_VERIFIED`, or `INTEGRATED`. The live launchd daemon was not observed (`ps` is forbidden in this sandbox). GPU/Metal/ANE/U50 performance was not measured.
- **18/18 graph `BUILT` gates are refuted as `BUILT`.** Most survive as `TESTED` or `END_TO_END` under FUNCTIONAL_SIM / STATIC receipts. Two are capability-level refutations: `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED` and `ODYSSEY_I_DISCOVERY`.
- **`AGENTOS_BEHAVIOR_LAB` is not ABSENT.** `tools/vmcp/behavior_lab.py` implements BHV-01..23 and `tools/future/vmcp.py` **Calls** `run_matrix`. The graph said ABSENT because the catalog listed no paths.
- **`civilization/ROADMAP_STATE.json` is inflated and stale.** I-A / I-C / I-D / I-E are `ADVERSARIALLY_VERIFIED` and I-B is `PHYSICALLY_RUNNING` there; the capability graph says those genes are `SCAFFOLDED`; this audit says `CALLABLE` or `SCAFFOLDED`. Its `roadmap_hash` (`51474151…`) does not match the current H-ROADMAP.md (`d43a6b07…`).

Constitution holds on disk: five eras, three Odysseys, no Era VI, FPGA inside Accelerator/Fusion, Theia is a bounty not a civilization, 0.7% coordinate untouched.

## Independent counts (13-state)

| state | gates | genes |
|---|---|---|
| END_TO_END | 6 | 0 |
| TESTED | 10 | 0 |
| CALLABLE | 15 | 7 |
| SCAFFOLDED | 19 | 14 |
| CONCEPT_ONLY | 7 | 1 |
| BLOCKED_HARDWARE | 14 | 3 |
| ABSENT | 0 | 0 |
| PHYSICALLY_MEASURED / ADVERSARIALLY_VERIFIED / INTEGRATED | 0 | 0 |

Graph vocabulary this 13-state list does not contain: `BUILT`, `WIRED`, `UNREACHABLE`, `BLOCKED_EXTERNAL`, `DORMANT`. Those were mapped, never copied.

## How this was checked (and what was not)

**Done, independently of the graph’s status field:**

- Read H-ROADMAP.md (9645 lines, hash `d43a6b07…`) Appendix O + Appendix A + §5.
- Read `civilization/CAPABILITY_GRAPH.json` and `tools/roadmap/catalog.py` (via `git show`) as *probes*, then re-derived callers.
- `git cat-file` / `git show HEAD:<path>` for definitions (sparse checkout is not absence).
- `git grep` + AST `Call` of every catalog symbol at HEAD. Imports were collected and discarded as wiring.
- Acceptance receipts under `receipts/acceptance/*.json` — verdict / `criterion_altered` / evidence_tier / whether the catalog symbol was actually invoked.
- Numeric bar: `receipts/future/COMPLETE_EBPW.json` `incumbent.complete_ebpw = 3.139300850311054` against `<= 1`.
- Hardware inventory this lane: Apple M3 Ultra (`Mac15,14`); `AppleT6031ANEHAL=1`; `nvidia-smi` missing; no U50 / eGPU / HMF / CXL in PCIe / Thunderbolt / ioreg.

**Not done (and therefore never labelled MEASURED / INTEGRATED):**

- Live daemon process listing (sandbox: `ps: operation not permitted`).
- Metal kernel dispatch, protected Qwen27 benchmark, ModelLake rehash, `tools/vmcp/test_behavior_lab.py`.
- Mutation-check of any acceptance test.

Production vs harness: `tools/acceptance/**` that is not `test_*` was **not** treated as a product caller (the graph often did). Same-file Calls that are **not** the `def` line **were** counted (the graph excludes the whole definition file and therefore dropped `resident.py:1012` `admit_evidence_children` and `delegate.py:1470` `abort`).

## Disagreements that matter

These are the load-bearing deltas vs `civilization/CAPABILITY_GRAPH.json`. Full list: 49 items in the JSON (`disagreements_with_committed_graph`).

### Graph `BUILT` that does not survive as completed capability

| gate | graph | independent | why |
|---|---|---|---|
| `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED` | BUILT | CALLABLE | Acceptance ACCEPTED STATIC **without invoking** `run_qwen27_mlp_diagnostic_ab`. Explanation is identity mismatch (6 DIFFERENT_VERIFIED dimensions), not a bounded regression on the same executable. |
| `ODYSSEY_I_DISCOVERY` | BUILT | CALLABLE | ACCEPTED STATIC = `pick_acquire_candidate(state, mutate=False)` + safetensors-header census. That is not Odyssey I (“first real school of model/device facts”). |
| `AGENTOS_REPAIR_BOUNDED` | BUILT | TESTED | Real `Scheduler(` Calls in `hcli/mission.py` / `goal_compile.py`. Catalog symbol is the whole Scheduler. Acceptance FUNCTIONAL_SIM. HEAD `04193ccbc` changed repair/stop behaviour after the graph commit. |
| `AGENTOS_CANCELLATION` | BUILT | TESTED | Graph’s only cited caller is the acceptance harness. Independent: `hcli abort` → `delegate.cli_main` → `abort()` at `hcli/delegate.py:1470` (same file as def, dropped by the graph). |
| `HCLI_SELF_SUPPLEMENT` | BUILT | END_TO_END | Graph cited only harness. Independent: `admit_evidence_children` at `hcli/agentos/resident.py:1012` (resident loop, same file as def). |
| `FLASH_DENSE_VS_NF_AB` / `FLASH_FIRST_GRAVITY_ORGAN` | BUILT | TESTED | FUNCTIONAL_SIM. Simulation is not board reality. |
| `MODELLAKE_IDENTITY_RESOLVED` / `MODELLAKE_ATOMIC_PROMOTION` / `QWEN27_RUNTIME_IDENTITY_FROZEN` | BUILT | TESTED | STATIC receipts. Not a physical lake/GPU measurement this lane. |
| AgentOS END_TO_END cluster (`ORPHAN_RECONCILIATION`, `PERSISTENCE_SINGLE_AUTHORITY`, `CONTEXT_*`, `BACKEND_FAILURE_ISOLATION`) | BUILT | END_TO_END | Production Calls exist. Still not INTEGRATED (daemon not observed) and not PHYSICALLY_MEASURED. |

### Graph ABSENT / BLOCKED_EXTERNAL / UNREACHABLE / DORMANT that the 13-state rejects

| item | graph | independent | why |
|---|---|---|---|
| `AGENTOS_BEHAVIOR_LAB` | ABSENT | CALLABLE | `tools/vmcp/behavior_lab.py` + Calls of `bhv_run_matrix` at `tools/future/vmcp.py:766,930,1001`. Catalog `paths=()`. Acceptance BLOCKED *because* the catalog was empty. |
| Theia ×7 | BLOCKED_EXTERNAL | CONCEPT_ONLY | Empty catalog paths. Bounty **engine** at `tools/theia/` is a different artifact (`python -m tools.theia` → `engine.main`). No trained generalist. `BLOCKED_EXTERNAL` is not in the 13-state list. |
| `FUSION_FIRST_HETEROGENEOUS_EXECUTABLE` | UNREACHABLE | BLOCKED_HARDWARE | Only dependency is HMF, which is absent. `UNREACHABLE` is not in the 13-state list. Tension with “FPGA lives inside Fusion”: Fusion’s first executable waits on HMF, not FPGA. |
| `V-E_PERPETUAL_HAWKING` | DORMANT | CONCEPT_ONLY | Empty paths. Era I sovereign. `DORMANT` is not in the 13-state list. |
| `FPGA_PREBOARD_SCHEMAS` | SCAFFOLDED | CALLABLE | `hcli/agentos_cli.py:1019` Calls `run_fpga_preboard`, which constructs `FPGADeviceGenome`. Catalog symbol is the dataclass. |
| `FLASH_SOURCE_VERIFIED` | SCAFFOLDED (accepted=True) | CALLABLE | Catalog omitted the symbol; acceptance Called `tools.flash_organ_census.main`. |
| `MODELLAKE_HASH_VERIFIED` | WIRED | SCAFFOLDED | Graph’s only caller is the acceptance harness. Generic name `reconcile`. Acceptance BLOCKED. |
| I-A gene | SCAFFOLDED | CALLABLE | Graph looked for a Call of `hcli.engine`. Independent: `hcli/__main__.py` → `cli.main` → `resident_main` / `App` / `delegate.cli_main`. The control plane is the product. Gene-as-whole is **not** promoted (retry BLOCKED, circuit-breaker SCAFFOLDED, VMCP organs mostly SCAFFOLDED). |

### `ROADMAP_STATE.json` vs both graphs

| program | ROADMAP_STATE | capability graph | this audit |
|---|---|---|---|
| I-A | ADVERSARIALLY_VERIFIED | SCAFFOLDED | CALLABLE (children mixed TESTED/END_TO_END/SCAFFOLDED) |
| I-B | PHYSICALLY_RUNNING | SCAFFOLDED | SCAFFOLDED (import of `hcli.doctor` is not a Call) |
| I-C | ADVERSARIALLY_VERIFIED | SCAFFOLDED | CALLABLE (IR + Flash science; EBPW 3.14; no full Noetic executable) |
| I-D | ADVERSARIALLY_VERIFIED | SCAFFOLDED | CALLABLE (HWIR + identity freeze; U50 BLOCKED_HARDWARE) |
| I-E | ADVERSARIALLY_VERIFIED | SCAFFOLDED | CALLABLE (picker, not a school) |
| II-C | NOT_STARTED | WIRED | CALLABLE (exact subprocess of `physical_graph_compiler.py` from `nr_nx_generic.py:1981`) |

## Surprises (loud)

1. **Gene cards are wallpaper.** 325 copies of “Action: instantiate a bounded WorkUnit…” and 325 copies of “Receipt: bind input identity…”. What would settle it: per-gene authority artifacts. This lane does not rewrite the roadmap.
2. **`AGENTOS_BEHAVIOR_LAB` ABSENT is false.** What would settle it: catalog `tools/vmcp/behavior_lab.py` / `run_matrix`, materialize `hcli/workunit.py`, run the 23-fixture test.
3. **`ROADMAP_STATE` says ADVERSARIALLY_VERIFIED** for Era I programs. No mutation-check, no live daemon, FUNCTIONAL_SIM receipts. What would settle it: regenerate that file from call sites, or show the adversarial evidence it claims.
4. **Metal invisible on an M3 Ultra.** `FLASH_NATIVE_NF_KERNEL` acceptance: `metal: no Metal-capable GPU` in another worktree. This sandbox: `xcrun metal` not on PATH. What would settle it: unsandboxed `flash_noetic_q4_kernel_parity` on this host.
5. **Qwen27 “regression explained” did not run the diagnostic.**
6. **Odyssey I “BUILT” is a picker.**
7. **EBPW is 3.139, not ≤ 1.** The graph correctly refused `accepted` here. Nearby Flash `BUILT` marks must not be read as Gravity done.
8. **Graph is 7 commits stale.** `tools/future/hwir.py` 5444 → 6279 lines (pluggable `LoweringTarget`: HLS-style + Rust-HDL-style emitters).
9. **`verified_absent[].verdict` is `PRESENT`** for theia / transport / placement. The field name is inverted.
10. **Fusion’s first heterogeneous executable is gated on HMF, not FPGA.**
11. **Appendix O has no verdicts.**
12. **Generic catalog symbols** (`resolve`, `promote`, `abort`, `scars`, `reconcile`). Bare `resolve` AST-matched 3124 Calls. Bound-to-`hcli.context_budget.resolve` (imported as `resolve_context_budget`) is the real wiring.
13. **Def-file exclusion drops real Calls** (`admit_evidence_children`, `abort` CLI).
14. **Live daemon is invisible in this sandbox.** No `INTEGRATED` claim is honest here.

## Hardware (inventory, not performance)

| probe | result | how |
|---|---|---|
| SoC | Apple M3 Ultra, `Mac15,14` | `sysctl` this lane |
| ANE | present (`AppleT6031ANEHAL=1`) | ioreg class count; **not** an ANE benchmark |
| U50 / Alveo / XCU50 | absent | PCIe + Thunderbolt inventory |
| DGX / nvidia-smi | absent | `nvidia-smi` not found |
| eGPU | absent | Thunderbolt / displays |
| HMF / HGVAS / CXL | absent | ioreg |
| Metal dispatch | not executed | `xcrun metal` missing in sandbox; prior acceptance reported no Metal GPU |

All U50_* gates and `HMF_DEVICE_VISIBLE_TRUST` stay `BLOCKED_HARDWARE`. This auditor is not authoritative for U50 / HMF / DGX / eGPU / GPU-TPS / ANE / thermal facts it did not measure.

## Roadmap wording accuracy

| surface | accurate? | note |
|---|---|---|
| Appendix O ledger | as an **obligation list**, yes; as a **status ledger**, no | unfilled enum; line 9529 already says a gate name is not evidence |
| Appendix A gene cards | **no** | identical boilerplate; `STALE_ROADMAP_TEXT` |
| §5 era/program missions | as **missions**, yes; as **completion**, no | each program copies the same operating rule / promotion gate / self-refill |
| Constitution block | **yes** | five eras, three Odysseys, no Era VI, FPGA in Accelerator/Fusion, Theia bounty, 0.7% |
| Gate names that encode bars (`FLASH_COMPLETE_EBPW_LE_1`, `FLASH_ACCEPTED_TPS_GE_50`) | **yes as unmet bars** | EBPW 3.139; accepted_tps `None` |
| Catalog Fusion dependency on HMF | **stale vs constitution** | FPGA-inside-Fusion is not this gate’s dependency graph |

## Child-gate snapshot (Era I, strongest independent state)

**END_TO_END (FUNCTIONAL_SIM, production Calls):** `AGENTOS_ORPHAN_RECONCILIATION`, `AGENTOS_PERSISTENCE_SINGLE_AUTHORITY`, `HCLI_CONTEXT_AUTHORITY_UNIFIED`, `HCLI_CONTEXT_FOCUSED_WORKUNITS`, `BACKEND_FAILURE_ISOLATION`, `HCLI_SELF_SUPPLEMENT`.

**TESTED:** `AGENTOS_REPAIR_BOUNDED`, `AGENTOS_CANCELLATION`, `AGENTOS_CHECKPOINT_ATOMICITY`, `AGENTOS_RESTART_COHERENCE`, `VMCP_RECEIPT_LAW`, `MODELLAKE_IDENTITY_RESOLVED`, `MODELLAKE_ATOMIC_PROMOTION`, `QWEN27_RUNTIME_IDENTITY_FROZEN`, `FLASH_FIRST_GRAVITY_ORGAN`, `FLASH_DENSE_VS_NF_AB`.

**CALLABLE, bar unmet or acceptance blocked:** `HCLI_STATUS_PHYSICAL`, `HCLI_MIXED_MAX`, `HCLI_SELF_OPTIMIZATION_BOOTSTRAP`, `AGENTOS_BEHAVIOR_LAB`, `AGENTOS_DETERMINISTIC_OFFLOAD`, `QWEN27_PROTECTED_BASELINE`, `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED`, `FLASH_SOURCE_VERIFIED`, `FLASH_NATIVE_NF_KERNEL`, `FLASH_COMPLETE_EBPW_LE_1`, `FPGA_PREBOARD_SCHEMAS`, `FPGA_HWIR`, `ODYSSEY_I_DISCOVERY`, plus Era II/III `ODYSSEY_II_TRANSFER`, `ODYSSEY_III_ADVERSARIAL_META_SCIENCE`.

**SCAFFOLDED:** retry/circuit-breaker, context invalidation, most VMCP organs, Flash full executable / TPS≥50, FPGA link/partition sim, ModelLake hash.

**BLOCKED_HARDWARE:** 12× U50_* + HMF + Fusion-first-executable.

**CONCEPT_ONLY:** seven Theia model-ladder gates (engine exists separately).

## What would change this audit’s mind

- A launchctl/process listing of the live HCLI daemon **without signalling it** → possibly `INTEGRATED` for I-A surfaces that already have production Calls.
- Unsboxed Metal `flash_noetic_q4_kernel_parity` on this M3 Ultra → `FLASH_NATIVE_NF_KERNEL` could become `PHYSICALLY_MEASURED` if it actually dispatches.
- Invoke `run_qwen27_mlp_diagnostic_ab` on a disposable resident → could restore a honest `TESTED` for the regression gate.
- Catalog + run `tools/vmcp/behavior_lab.py` with `hcli/workunit.py` on disk → `AGENTOS_BEHAVIOR_LAB` could become `TESTED`.
- Re-run `tools/roadmap/auditor.py` at HEAD with (a) def-line exclusion not def-file exclusion, (b) fully qualified symbols, (c) 13-state vocab, (d) acceptance harness not counted as production.

None of that is this lane. This lane’s job was to say what the disk actually supports.

## Files

- `receipts/audit/aud02-roadmap-facet-reaudit.json` — machine-readable; 71 gates + 25 genes; every `CALLABLE`/`TESTED`/`END_TO_END` claim has a call site; every weaker claim has an explicit blocker.
- `receipts/audit/aud02-roadmap-facet-reaudit.md` — this note.
