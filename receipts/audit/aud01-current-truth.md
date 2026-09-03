# aud01 — current project truth

Reconstructed from disk and runtime on 2026-09-02. Not inferred from prior chat or roadmap prose. Not an implementation pass. Classification uses the audit vocabulary only. **PHYSICALLY_MEASURED = 0, ADVERSARIALLY_VERIFIED = 0, END_TO_END = 0** in this audit: those words were not earned this session.

Machine-readable twin: `receipts/audit/aud01-current-truth.json`.

---

## Lead

HEAD is `04193ccbc` (`fix(hcli): a graceful stop, and a successful repair, no longer end the mission`) on `grok/aud01-current-truth-20260902-145104`, same commit as `odyssey-i`. This worktree is a sparse checkout (40 roots, 15,913 files at HEAD, 0 dirty before this write).

The live research computer on this host is **not** a running HCLI daemon. It is:

- an Apple M3 Ultra Mac Studio (28 CPU / 60 GPU / 96 GB / ANE present in ioreg)
- a **failed, evacuated** resident whose last disk heartbeat is 2026-09-02 05:49 local
- a **live ModelLake watch** (`com.hawking.modellake.watch` pid 4183) over 55 specimen bodies totalling **4.353 TB** on `/Volumes/corpdrive/hawking-modellake`
- a capability graph that is **five commits stale** and that counted at least one `__main__` demo as a production caller

Era I is still sovereign. Five eras, three Odysseys, no Era VI. Theia is one bounty model, not a civilization. FPGA belongs inside Accelerator/Fusion. The 0.7% civilizational coordinate was **not rewritten**; producers currently disagree about the number.

---

## Surprises (loud)

These are the findings that would be expensive to smooth over. Each names what would settle it.

### S1 — There is no live hawkingd

Contract: a live HCLI daemon under launchd. Disk: `.hcli/resident/state.json` `state=FAILED`, `worker_live=false`, body `UNLOADED`, mission `phase=cancelled` / `cancel_reason=resident_self_evacuation`, `supervisor_pid=null`, last update **2026-09-02T05:49:23** local. `launchctl print gui/$UID` has **no** `com.hawking.hawkingd` / `hcli` / `resident` service. The only hawking launchd job with a pid is `com.hawking.modellake.watch` **4183**.

`ps`/`lsof` are denied in this sandbox, so a non-launchd hawkingd would be invisible here. Disk state is the authority used.

**Settle with:** unsandboxed `ps`/`lsof` on `.hcli/resident/.resident.lock`. If a pid holds it with a matching start token, the FAILED file is stale. If not, the contract preamble is stale.

### S2 — Not ~110 dirty `hcli/` files

Main checkout `/Users/scammermike/Downloads/hawking` dirty `hcli/` observed this session: `backends.py`, `engine.py`, deleted `test_absent_required_array_repair.py` (3 paths). This worktree was clean. The dirty set on the parent moved during the audit (first sample 18 porcelain / 0 hcli; later sample included those 3 hcli paths) because live writers are mutating the parent tree.

**Settle with:** `git status --porcelain -- hcli` on the daemon `repo_root` at contract-write time versus now.

### S3 — Three different H-ROADMAP hashes

| object | sha256 | lines |
|---|---|---|
| `/Users/scammermike/Downloads/H-ROADMAP.md` and `civilization/CAPABILITY_GRAPH.json` | `d43a6b07…` | 9645 |
| `civilization/ROADMAP_STATE.json` `roadmap_hash` | `51474151…` | — |
| `/Volumes/corpdrive/H-ROADMAP.md` | `5d704ca0…` | 9028 |

`python3 -m pytest civilization/test_validate.py` **fails**: `roadmap_hash does not match the canonical roadmap on disk`. H-ROADMAP.md is **not in git at HEAD**.

**Settle with:** one canonical path, re-hash, regenerate `ROADMAP_STATE` from that producer. Do not treat corpdrive or the stored hash as current.

### S4 — Capability graph is stale and inflated one caller

`civilization/CAPABILITY_GRAPH.json` `generated_from_commit=7d6428006`, five commits behind HEAD. Line drift: `compile_worker_context` graph `hcli/mission.py:983`, HEAD `:1055`. Graph counted `hcli/goal_compile.py:518 Scheduler()` as production; at HEAD that line is inside `if __name__ == "__main__":` with asserts. `AGENTOS_CANCELLATION` is graph-BUILT on a single `tools/acceptance` `abort()` call.

This audit **does not copy graph BUILT as PHYSICALLY_MEASURED**. Graph BUILT count was 18; none became PHYSICALLY_MEASURED.

**Settle with:** regenerate the auditor at HEAD, excluding `__main__` demos and `tools/acceptance` from `wired`.

### S5 — "55 sealed specimens" is a body count, not a hash-seal

55 specimen directories, 55 manifests, `specimens_bytes=4,352,666,459,500` (4.353 TB decimal / 3.959 TiB). Catalog `over_budget` versus 3500 GB tier2 (**+852.666 GB**). Manifests have **no `sealed` field**. 45/55 `provenance=backfill`. sha256 verified on 8 manifests observed. `loaded_weights=false`. Catalog `built_at` advanced live during this audit (18:53Z → 18:56Z → 18:58Z) because the watch is running. One retired manifest: `zai-org--GLM-4.5@cbb2c7cfb52f`.

**Settle with:** `verify_only()` per-file sha256 for every specimen, or rename the claim to acquired-bodies / hash-incomplete.

### S6 — `ADVERSARIALLY_VERIFIED` on civilizations is roadmap prose

`ROADMAP_STATE` and `ERA_I_CHECKPOINT_003` label I-A / I-C / I-D / I-E `ADVERSARIALLY_VERIFIED`. Acceptance receipts are `FUNCTIONAL_SIM` or `STATIC`. Checkpoint `last_verified_commit=a2e3d099b` versus HEAD `04193ccbc`. This audit grants **ADVERSARIALLY_VERIFIED to zero capabilities**.

**Settle with:** an adversarial run at HEAD with negative controls, or a downgrade in the producer (`checkpoint.py` / `ROADMAP_STATE` builder). Do not hand-edit the checkpoint artifact.

### S7 — Theia source is PRESENT while gates say BLOCKED_EXTERNAL

`verified_absent.theia.verdict=PRESENT` (`tools/theia/*` on HEAD) while every `THEIA_*` gate is `BLOCKED_EXTERNAL`. Not ABSENT. Classified `BLOCKED_AUTHORITY` (T0 substrate) / `SCAFFOLDED` (bounty engine CLI).

### S8 — A receipt labeled HARDWARE_MEASURED records a Metal failure

`receipts/acceptance/FLASH_NATIVE_NF_KERNEL.json` `evidence_tier=HARDWARE_MEASURED`, verdict BLOCKED, error `metal: no Metal-capable GPU`, while `system_profiler` reports Apple M3 Ultra / Metal Supported. Receipts do not override a broken producer. This audit did not dispatch Metal.

### S9 — Odyssey T0 checkpoint exists while the launch fence is false

`ODYSSEY_LAUNCH_AUTHORIZED` contains `false`. `ODYSSEY_FENCE.json` `authorized=false`. Checkpoint `ca36e0f962b06cbc` exists (`wall_clock=2026-09-02T18:49:01Z`, `stage=T0`, `step=1`, `eval_result_hash=none`, `parent_sha256=genesis`). Checkpoint existence is not an authorized training run.

### S10 — Civilizational coordinate has three values

Constitution: **0.7%**. `ROADMAP_STATE.civilizational_coordinate`: **0.7**. `ERA_I_CHECKPOINT_003.civilization_progress.value_pct`: **1.0** (heuristic). Not rewritten.

### S11 — Sovereign G002/G003/G004/G005 receipts are absent

`receipts/sovereign/` on disk: G001, G009, G010, G014, VERIFIER_MANIFEST. **No** `G003_self_mutation.json`, `G004_context_runtime.json`, `G005_prefill_pipeline.json`, `G002_overhead.json` at HEAD or on disk. Those red gates cannot pass. Mission evidence rows for G001/G002/G003/G005/G006 are `NO_EVIDENCE` / `pre_mutation_pass_not_run`.

G001 and G009 receipts exist from named producers earlier today; this audit did not re-run them (STATIC_VERIFICATION of the files).

### S12 — Sandbox hides processes

`ps` and `lsof` → `Operation not permitted`. launchd print works. Disk `.hcli` state used for daemon liveness.

---

## Repo

- **HEAD** `04193ccbc8ef9fdd2dfd595d65f656760829dddc` 2026-09-02 14:48:41 -0400
- **Branch** `grok/aud01-current-truth-20260902-145104` = `odyssey-i`
- Parent checkout `odyssey-i` is **ahead 8** of `origin/odyssey-i`
- ~497 local branches, **32 worktrees**, 15,913 files at HEAD, 198 `hcli/` files, 78 `hcli` tests
- Recent landings: hawkingd graceful-stop fix; s2 HWIR pluggable lowering merge; acceptance guards that cannot pass in a tree the daemon lives in; blocked lock acquirer is evidence

---

## Hardware (SOURCE_INSPECTION — not a performance measurement)

| fact | value | how |
|---|---|---|
| SoC | Apple M3 Ultra, Mac15,14 | `sysctl` / `system_profiler` |
| CPU | 28 (20P+8E) | `system_profiler SPHardwareDataType` |
| RAM | 96 GB | `hw.memsize=103079215104` |
| GPU | 60 cores, Metal Supported | `SPDisplaysDataType` |
| ANE | present (`AppleT6031ANEHAL=1`, `ANEScheduler=1`) | `ioreg -l` inventory |
| FPGA/U50 | **absent** | no xilinx/alveo/u50 in ioreg |
| DGX | **absent** | `nvidia-smi` not present |
| eGPU | **absent** | Thunderbolt buses, no enclosure |
| llama-server | **not listening** | 8080/8081/8088/8090/52484 closed |

ANE/Metal/FPGA **physical performance was not measured this session**. Authority limit honored.

---

## Daemon / resident / native runtime

Classification: **CALLABLE** (entry exists, 6 name-tests passed in 0.21s). Live: **false**.

- Entry: `hcli/hawkingd.py:31` → `hcli.agentos.resident.daemon_main`
- `pyproject.toml` script: `hawkingd = "hcli.hawkingd:main"`
- Resident schema `hcli.agentos.resident_daemon.v1`: FAILED, generation 866, cycles 855, `accepted_count=0`
- Body schema `hcli.agentos.resident_body.v1`: UNLOADED, `unload_reason=supervisor_stopped`
- Mission: cancelled, 11 evidence rows, all `NO_EVIDENCE`
- Goal bank: 1 queued toy mission ("count .py files in hcli")
- Protected locks `.hcli/locks/protected-accelerator-bench.lock` and `qwen-protected-bench.lock` exist (0 bytes); this audit did not flock them

---

## ModelLake

Root: `/Volumes/corpdrive/hawking-modellake` (not `/Volumes/corpdrive/specimens`).

| | |
|---|---|
| specimens | 55 dirs = 55 manifests |
| bytes | 4,352,666,459,500 |
| retired | 1 (`GLM-4.5`) |
| families | 39 |
| watch | launchd pid 4183, `--poll-secs 0.10` |
| hash-seal | **INCOMPLETE** |
| weights loaded | false |
| this lane | READ-ONLY on `specimens/` |

Largest body observed: `moonshotai--Kimi-K3@9f62e4e9fffb` ~1454 GiB.

`MODELLAKE_ATOMIC_PROMOTION` is **INTEGRATED** in source: `promote()` is called from the live watch. This audit did not invoke `promote`. `MODELLAKE_HASH_VERIFIED` is **CALLABLE** / blocked on a 4.35 TB rehash.

---

## Odyssey

Exactly three Odysseys. Launch fence **false**.

| stream | classification | acceptance | note |
|---|---|---|---|
| I Discovery | TESTED | ACCEPTED STATIC | `pick_acquire_candidate` called from `hcli/acquisition.py:172` and `tools/odyssey_ctl.py:6106`. 148 `receipts/odyssey-i/` files on disk. |
| II Transfer | CALLABLE | BLOCKED | `evaluations_avoided=-8` (required > 0) |
| III Adversarial | CALLABLE | BLOCKED | synthetic selftest; `physical_arm=not_run` |

T0 checkpoint exists and is not an authorized run (S9).

---

## HWIR / FPGA / U50

`tools/future/hwir.py` is PREHARDWARE by its own law. Production callers of `HwirGraph(` at HEAD: `tools/accelerator/backend_contract.py:1094`, `fusion_bridge.py:1444`, `tools/future/propagate.py:695`, `p6_projection.py:857`. Classification: **CALLABLE**. s2 pluggable lowering **merged at this HEAD** after the capability graph commit.

All `U50_*` gates: **BLOCKED_HARDWARE**. `FPGA_PREBOARD_SCHEMAS` / `FPGA_LINK_SIM` / `FPGA_PARTITION_SIM`: **SCAFFOLDED**. Fusion first heterogeneous executable: **BLOCKED_HARDWARE** (no FPGA/HMF/eGPU/DGX).

---

## Accelerator / Flash / Qwen27

`CODEX_ACCELERATOR_HANDOFF.json` `recorded_at=2026-08-30T05:06:03Z`, `accelerator_complete=false` — three days stale relative to HEAD. Cited as data, not current measurement.

- `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED`: **INTEGRATED** (`protected_benchmark_watcher.py` + CLI). Not re-measured.
- `QWEN27_PROTECTED_BASELINE`: **CALLABLE** / **BLOCKED_AUTHORITY** (protected lock).
- Flash gravity organ + dense-vs-NF A/B: **TESTED** (acceptance ACCEPTED, CLI-only live path — downgraded from graph BUILT).
- Flash TPS ≥ 50: **SCAFFOLDED** (`accepted_tps=None`; one accepted token is not multi-token TPS).
- Complete EBPW ≤ 1: **CALLABLE** (prior receipt 3.1393, not ≤ 1; not remeasured).
- Full Noetic executable: **SCAFFOLDED** (NX metadata-only, loader NOT_IMPLEMENTED).

---

## Prefill, context, self-mutation, landing

| claim | classification | call site or blocker |
|---|---|---|
| Worker context packets (`compile_worker_context`) | INTEGRATED | `hcli/mission.py:1055` (HEAD; graph said 983) |
| Sovereign G004 context runtime receipt | SCAFFOLDED | `receipts/sovereign/G004_context_runtime.json` absent |
| `/status` physical renderer | CALLABLE | missing D.9 fields |
| Mixed-max | CALLABLE | no llama-server |
| Prefill G005 | SCAFFOLDED | G005 receipt absent; rust prefill modules exist, not run |
| LandingService | TESTED | `propose_landing` from `tool_registry.py:1526` and `commands.py:1643`; **23/23 in 6.99s** on scratch repos |
| G003 self-mutation e2e | SCAFFOLDED | G003 receipt absent |

Landing never pushes. Governance paths (`landing.py`, `tool_registry.py`, `verifier_pipeline.py`, `executors.py`, protected tests) are in `_ALWAYS_REFUSED_PREFIXES`.

---

## Tests actually run this session (wall times MEASURED)

| command | result | real |
|---|---|---|
| `pytest civilization/test_validate.py civilization/test_checkpoint.py -o addopts=''` | **4 failed, 44 passed** in 0.45s (roadmap_hash + `ps` denied) | 0.61s |
| `pytest lab/tests --collect-only -o addopts=''` | 85 collected, **139 collection errors** (sparse) | 2.24s |
| `pytest tools/haider/hcli/tests --collect-only` | **545 collected** | 0.49s |
| `pytest tools/haider/hcli/tests --maxfail=20` | 20 failed, 179 passed, 1 skipped in 8.41s — **sparse-contaminated, not a clean suite verdict** | 8.63s |
| frankenstein condense collect-only | 0 collected, 17 errors | 0.65s |
| `cargo test -p hawking-index-query --offline --lib` | **15 passed / 0 failed**, test 0.72s | 6.99s |
| `pytest hcli/test_hawkingd_daemon_name.py` (main checkout) | **6 passed** | 0.42s |
| `pytest hcli/test_landing.py` (main checkout) | **23 passed** | 7.17s |

Not run: 78 hcli tests as a suite, 634 tools tests, 195 crate test files, 160 hawking-core tests on disk. Reason: `hcli/` is not materialized here; full runs risk signalling processes or taking protected GPU locks.

---

## Capability census (this audit, not the graph)

81 capability claims (71 graph gates + 10 subsystem/sovereign additions). Every claim has a call site or an explicit blocker.

| classification | n | meaning here |
|---|---|---|
| INTEGRATED | 9 | production caller on resident/mission/controller/engine/watch path **and** an acceptance receipt. Daemon is down, so this is source-integrated, not a live end-to-end run. |
| TESTED | 12 | symbol called (often harness or CLI) and a receipt/tests exist. Not the resident loop. |
| CALLABLE | 12 | production call of the symbol; acceptance missing or blocked. |
| SCAFFOLDED | 25 | definition exists; no non-test production call, or red-gate receipt absent. |
| BLOCKED_HARDWARE | 14 | U50/HMF/Fusion — hardware not on this host. |
| BLOCKED_AUTHORITY | 8 | Theia T0 substrate; ANE/Metal physical performance this lane must not claim. |
| ABSENT | 1 | `AGENTOS_BEHAVIOR_LAB` (no BHV fixture runner). |
| PHYSICALLY_MEASURED | **0** | nothing this process measured on hardware |
| ADVERSARIALLY_VERIFIED | **0** | not granted |
| END_TO_END | **0** | daemon down; no live mission accepted_count |

### INTEGRATED nine (source path, daemon currently FAILED)

1. `AGENTOS_REPAIR_BOUNDED` — `hcli/mission.py:264,:374 Scheduler(` (HEAD). Discounted `__main__` demo at `goal_compile.py:518`.
2. `AGENTOS_ORPHAN_RECONCILIATION` — `hcli/agentos/resident.py`, `handoff.py`, `protected_benchmark_watcher.py`.
3. `AGENTOS_PERSISTENCE_SINGLE_AUTHORITY` — `hcli/controller.py` `MutationLock`.
4. `HCLI_CONTEXT_AUTHORITY_UNIFIED` — `hcli/engine.py`, `backends.py`, `runtime.py` `resolve`.
5. `HCLI_CONTEXT_FOCUSED_WORKUNITS` — `hcli/mission.py:1055 compile_worker_context`.
6. `BACKEND_FAILURE_ISOLATION` — `hcli/mission.py` / `runtime.py` `terminate_pid`.
7. `HCLI_SELF_SUPPLEMENT` — `hcli/agentos/resident.py:1012 admit_evidence_children`.
8. `MODELLAKE_ATOMIC_PROMOTION` — live `modellake_watch.py` `promote()`.
9. `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED` — `protected_benchmark_watcher.py`.

Graph genes: 20 SCAFFOLDED at gene grain (no gene-level caller), 1 CALLABLE (`II-C_PHYSICAL_GRAPH_COMPILER` via subprocess), 3 BLOCKED_HARDWARE (HMF/DGX/eGPU), 1 CONCEPT_ONLY (`V-E_PERPETUAL_HAWKING` DORMANT). Child gates may be stronger than the gene object.

---

## Roadmap artifacts that exist

- `civilization/CAPABILITY_GRAPH.json` (STATIC, stale vs HEAD)
- `civilization/ROADMAP_STATE.json` (stale hash; ADVERSARIALLY_VERIFIED prose)
- `civilization/ERA_I_CHECKPOINT_{001,002,003}.json` (003 last_verified_commit `a2e3d099b`)
- `receipts/acceptance/*` from acc1–acc6 (FUNCTIONAL_SIM / STATIC; git_head mostly `14ed3eb1f` or older)
- `receipts/future/ROADMAP_SCAFFOLD_SATURATION.json`, `HCLI_CAPABILITY_REACHABILITY.json`
- `receipts/sovereign/G001_verifier_synthesis.json`, `G009_reachability.json`
- `CODEX_ACCELERATOR_HANDOFF.json` (2026-08-30)
- `receipts/audit/` **did not exist** before this lane

STALE_ROADMAP_TEXT (data, not rewritten): civilization ADVERSARIALLY_VERIFIED labels; ERA checkpoint commit; ROADMAP_STATE hash; contract preamble about the live daemon and 110 dirty files.

---

## Constitution check (observed, not proposed)

- Eras in `ROADMAP_STATE.era_statuses`: I, II, III, IV, V. No Era VI.
- Odysseys: I, II, III only.
- FPGA gates live under `I-D_ACCELERATOR` / fusion under `IV-A_FUSION`.
- Theia `era=bounty`, `gene=None`.
- Coordinate: not rewritten (S10).

---

## What this audit is not

It is not a rewrite of H-ROADMAP.md. It is not an implementation campaign. It did not kill, signal, or restart any process. It did not write `specimens/`, `hcli/`, `tools/`, `crates/`, or `civilization/`. It did not take protected GPU locks. It did not dispatch Metal or ANE. It did not hash 4.35 TB.


---

## Commit status

**Not committed.** `git add --sparse` fails with `Unable to create '.../worktrees/aud01-current-truth-20260902-145104/index.lock': Operation not permitted`. The parent git dir `/Users/scammermike/Downloads/hawking/.git` is not writable from this process (`com.apple.macl`; objects and refs likewise). Artifacts are untracked on disk at `receipts/audit/`. Worktree cleanup will destroy them unless a process that can write that `.git` commits them.

What would settle it: `git -C <this worktree> add --sparse receipts/audit && git commit` from a process with write access to the parent git dir.
