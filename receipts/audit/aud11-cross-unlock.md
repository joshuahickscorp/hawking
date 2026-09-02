# aud11-cross-unlock

**Did recent scaffolding materially move the roadmap denominator?**

**Less than it appears.**

Eighteen of seventy-one Appendix O gates are BUILT. Zero of twenty-five genes are BUILT. The civilizational coordinate is still the frozen 0.7 against five eras and twenty-five civilizations, and that number is not a ledger-completion statistic. The work that looks like a foundation-unlocking wave is mostly (1) reclassification of code that already existed, (2) sovereign red-gate tests whose receipts were never written, and (3) simulators and object-models sitting in front of hardware that is not on this machine.

Machine-readable twin: `receipts/audit/aud11-cross-unlock.json`. Every capability claim in that file carries a call site of the symbol itself or an explicit blocker. Imports do not count.

Evidence in this audit is `STATIC_VERIFICATION` / `SOURCE_INSPECTION` unless a receipt is named. Nothing here was re-run on the GPU, ANE, or ModelLake hash path. `PHYSICALLY_MEASURED` is not used.

---

## Two denominators, one frozen coordinate

| Ledger | What it counts | Head-state |
|---|---|---|
| Appendix O capability graph | 71 named gates in H-ROADMAP appendix O | BUILT 18 · WIRED 11 · SCAFFOLDED 20 · ABSENT 1 · BLOCKED_HARDWARE 13 · UNREACHABLE 1 · BLOCKED_EXTERNAL 7 |
| Genes / programs | 25 civilizations | SCAFFOLDED 20 · WIRED 1 (II-C Physical Graph Compiler) · BLOCKED_HARDWARE 3 · DORMANT 1 · **BUILT 0** |
| Sovereign obligations | 83 G-ledger items in `civilization/ROADMAP_STATE.json` | VERIFIED 52 · IN_PROGRESS 20 · OPEN 3 · PENDING 5 · BLOCKED 1 · NOT_STARTED 2 |
| Civilizational coordinate | complete Hawking system | **0.7, frozen, heuristic** |

The capability graph at `civilization/CAPABILITY_GRAPH.json` was generated from commit `7d6428006` at 2026-09-02T18:35:55Z. HEAD is `04193ccbc` (six commits later, including the HWIR pluggable-lowering merge). This audit grepped HEAD; it did not re-run the auditor.

`BUILT` in that graph means *wired AND accepted*. Wired is a non-test call of the implementing symbol. A module import is not a call. That law is the right one. It is also why a lot of recent files do not move the Appendix O score.

---

## What recent scaffolding actually did

The saturation receipt (`receipts/future/ROADMAP_SCAFFOLD_SATURATION.json`) is honest about the first inflation and then inflates a second time:

- It says the campaign's opening ABSENT baseline was **materially wrong** — nearly every subsystem it called absent already existed with tests. `BUILT` 26→0 was a *correction* (call sites are not acceptance), not a loss of capability.
- In the same file, `remaining_highest_ev_gaps` still says "No gate has demonstrated ACCEPTANCE / zero are accepted." That sentence is false against the same file's `final.by_status.BUILT = 18` and against the graph's `accepted_gates = 30`.

So the campaign moved **labels**. It did not move the 0.7 coordinate, it did not BUILT any of the 25 genes, and it did not wake U50, HMF, or Theia.

The sovereign lane (`hcli/test_goal_verifier_synthesis.py` and siblings) added a second scoreboard. Four of those red gates have receipts (G001, G009, G010, G014). Five that this audit was asked to treat as foundations do not: G003, G004, G005, G006, G013. A test that fails until a receipt appears is not a measurement.

---

## Foundations the operator named

For each: previously blocked descendants · now scaffoldable · now testable · still physically blocked.

### 1. Goal Compiler + verifier synthesis — `INTEGRATED` (narrowly)

Live HCLI already calls `GoalCompiler.compile`:

- `hcli/engine.py:1292`
- `hcli/controller.py:1394, 1651, 1783`
- `hcli/mission.py:316`
- `hcli/agentos/runtime.py:280`

`_verify_command` now synthesises a scoped `git grep` for `(def\|class) symbol` when the obligation names no `test*.py`. `command_is_admissible` is called from `executors.py:307`, `landing.py:232`, `ledger.py:689`. G001's producer (`tools/sovereign/g001_verifier_synthesis.py:129`) compiled a four-obligation goal through that path and ran each verifier red/green/red in a scratch git tree. Receipt: `receipts/sovereign/G001_verifier_synthesis.json`.

That is real. It is also smaller than the name.

- `hcli.goal_compile.ingest` / `schedule` — the genesis-scale compiler the acceptance test actually imports — has **no production caller**. Only `hcli/test_goal_compile.py` and `hcli/test_goal_compiler_acceptance.py`.
- Synthesised verifiers are definition greps, not property checks. G001 measured four toy obligations (`atomic_write`, `SwapoutsProbe`, `RECEIPT_SOURCE`, `report_unreachable`).
- There is **no Appendix O gate** named GOAL_COMPILER. This cannot move the 71-gate denominator.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Empty verifier when goal text names no test file | git-grep verifiers for code-shaped names with a scoped path | G001; ingest/schedule acceptance suite | Live-ledger VERIFIED via synthesis (4 cases only); `check_disk_satisfaction` on the live scheduler (`schedule()` is test-only); semantic verifiers |

### 2. Context Compiler + long-context runtime — `INTEGRATED` (Python packet) / undischarged (long-context)

Two compilers, plus a third name collision.

1. `hcli.goal.compile_worker_context` is called from `hcli/mission.py:1055`. Appendix O `HCLI_CONTEXT_FOCUSED_WORKUNITS` is already **BUILT** (acceptance evidence_tier `FUNCTIONAL_SIM`).
2. `hawking_context::ContextCompiler::compile` is called from `hide-kernel/src/lib.rs:87` and hide-backend. **No HCLI or hawking-serve caller.** This is the HIDE product.
3. `lab/lineage/continuity.py:191` defines another `compile_worker_context`. It is not a caller of (1).

Sovereign G004 (GoalIR + context compiler on the live runtime path) and G006 (native long-context at 131072 and 262144) have **no receipts**. Those tests cannot pass. This audit did not measure 131k/262k.

`HCLI_CONTEXT_INVALIDATION` is accepted and not wired (0 runtime callers of the catalog symbol).

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Focused workunit packets | Invalidation wiring; HIDE-side packing | Packet unit/acceptance tests | G006 131k/262k; unifying HCLI packets with Rust manifests; Flash TPS at long context |

This does **not** newly move Appendix O. The packet gate is already BUILT.

### 3. ModelLake sealed specimens — `INTEGRATED` (identity) / `WIRED` (hash)

`run_modellake_census` is called from `hcli/agentos_cli.py:739`. `promote` is called from `tools/odyssey/modellake_watch.py:659,809`. Identity acceptance (`receipts/acceptance/MODELLAKE_IDENTITY_RESOLVED.json`) reports **55 sealed, 55 identity-resolved, 0 unresolved**, lake at `/Volumes/corpdrive/hawking-modellake`. Lifecycle mix: CENSUSED 9, MANIFEST_READY 45, SSD_STAGED 1.

Hash is not done. `receipts/acceptance/MODELLAKE_HASH_VERIFIED.json` is BLOCKED: oid-backed sha256 of **54 specimens / 4.351 TB** remaining after a canonical Qwen3-0.6B hash. Size match is not sha256. `reconcile()` was invoked on a scratch tree because the live call promotes with `go=True`.

Atomic promotion is nevertheless **BUILT**. That is a scratch-tree `os.rename` test, and it depends on HASH_VERIFIED in the catalog. The graph does not enforce that edge. The school can look finished while it is not cryptographically verified.

G010 (`receipts/sovereign/G010_modellake_retained.json`) claims a live 660s retained-bytes measurement (Δ 37.2 GiB) and records **6 restarts** and 6 stale partials. This audit did not re-run it. ModelLake law says not to casually restart a healthy worker; six restarts in eleven minutes is a flagged surprise, not a verdict.

USB fill still owns `corpdrive`. G048 real-weights execute waits. Theia does not wake because specimens are not a training substrate.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Countable identity-resolved school | Experiments that only need repo+revision+body; hash pass (symbol already WIRED) | Census/promote/watch tests; identity acceptance | 4.35 TB sha256; G048; all 7 THEIA_* gates; Flash noetic executable |

### 4. FPGA simulator + HWIR — `CALLABLE` (HwirGraph) / `SCAFFOLDED` (inner sim) / `BLOCKED_HARDWARE` (U50)

`HwirGraph(` is constructed from four non-test files:

- `tools/accelerator/backend_contract.py:1094`
- `tools/accelerator/fusion_bridge.py:1444`
- `tools/future/p6_projection.py:857`
- `tools/future/propagate.py:695`

Appendix O `FPGA_HWIR` is **WIRED, not accepted**. `HWIR_V1.json` (recorded 2026-09-02T18:47:20Z, after the graph) states it is a PREHARDWARE sidecar, `not_an_fpga_backend: true`, no board, no bitstream, no `HARDWARE_MEASURED` number.

`hcli/agentos_cli.py:1017` calls `run_fpga_preboard`. That CLI constructs `FPGADeviceGenome` and `TransportLinkSimulator` and calls `simulate_partition` **inside the defining module**. The catalog demands those inner symbols, so `FPGA_PREBOARD_SCHEMAS`, `FPGA_LINK_SIM`, and `FPGA_PARTITION_SIM` stay SCAFFOLDED. That is auditor-narrowness, not missing code — and reclassifying them still would not wake U50.

Pluggable `LoweringTarget` (`hls_style`, `rust_hdl_style`) is tested in `tools/future/test_hwir_lowering.py`. `lower_hwir(` has **no caller** outside `tools/future/hwir.py` and that test file. The four HwirGraph production callers do not lower.

`U50_PRESENT` is false. Twelve `U50_*` gates stay `BLOCKED_HARDWARE`. H-ROADMAP I.23 forbids skipping simulation → cheap FPGA → serious FPGA.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| HWIR graph construction | Inner preboard symbols (or catalog retarget); PREHARDWARE HLS/Rust-HDL emission; sealed predictions | `test_hwir.py`, `test_hwir_lowering.py`, fusion_bridge tests | All 12 U50_* gates; any measured HBM/link/timing number |

**This is the loudest false fan-out.** One IR plus a simulator does not unlock an FPGA civilization.

### 5. Native prefill — `CALLABLE` (two families) / not in Appendix O

`Engine::prefill_slot` is implemented for QwenDense (`qwen_dense.rs:2997`) and RWKV7 (`rwkv7.rs:1805`). The serve loop calls it at `crates/hawking-serve/src/lib.rs:875`. The default trait method returns `Unimplemented`.

Parity tests exist (`prefill_slot_into_multiseq_parity.rs`, `p3_batched_prefill_parity.rs`, `rwkv7_prefill_slot_multiseq_parity.rs`). This audit did not run them.

Sovereign G005 (`hcli/test_qwen38_prefill_pipeline.py`) wants measured f32 GEMM coverage, tok/s improvement, and an honest dispatch gate. `receipts/sovereign/G005_prefill_pipeline.json` is **absent**. There is no PREFILL gate in Appendix O. This does not unlock `FLASH_ACCEPTED_TPS_GE_50` or the rank-1 "first complete native Flash token" workunit.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Serve-loop prompt ingest for QwenDense/RWKV7 | Parallel prefill; prefill disk cache | Core parity tests; G005 once a producer exists | G005; Flash native token; other Engine impls |

### 6. Tool call-site reachability — `TESTED`

`ToolRegistry.invoke` is called from `hcli/engine.py:948` and `hcli/executors.py:403`. `default_tool_registry(` is constructed from engine, executors, agentos runtime/checkpoint/recovery/research, and commands.

G009 (`receipts/sovereign/G009_reachability.json`): 65 registered tools, 12 invoked through the registry in that run, **0 with no production call site**, 9 missing from the live compact catalog. Finding G009-F4: the "41 tools with zero call sites" era is over for this registry.

That is hygiene. It does not add Appendix O BUILT gates. The VMCP cluster is still SCAFFOLDED with 0 runtime callers of its catalog symbols. G009-F1: the live goal names processes as authority; the registry has no process tool and both shells refuse `ps`.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Unreachable-but-registered tools | Adding tools without repeating that bug | G009 discharged | VMCP doctor/capture cluster; AGENTOS_BEHAVIOR_LAB; processes capability |

### 7. Self-landing + successor handoff — two different things

**Landing is `CALLABLE`.** `git.land.propose` in `hcli/tool_registry.py:1532` forwards to `propose_landing`, which is the only path to `LandingService.land` (`hcli/landing.py:433`). `commands.py:1643` also calls it. Governance refuses edits to `landing.py`, `tool_registry.py`, `verifier_pipeline.py`, `executors.py`, and the sovereign test files. It never `git push`.

**Successor process handoff is not discharged.** `receipts/sovereign/G013_successor.json` is absent. `retire_incumbent` (`hcli/agentos/resident.py:1992`) is called from `start_resident` only when `replace=True` (`:2050`). `build_handoff` (`hcli/agentos_cli.py:1213`) is an overnight status snapshot whose own claim_boundary says it does not certify quality, FPGA, or sovereignty. G003 self-mutation receipt is also absent.

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Resident-initiated local commit | Governed edits off the always-refused prefixes; `replace=True` retirement | `hcli/test_landing.py`; G013/G003 once producers exist | G013 rollback/no-amnesia successor; G003 e2e mutation; push |

Lumping landing with successor inflates the foundation.

### 8. HMF/HGVAS scaffolds — `BLOCKED_HARDWARE`

`HMF_PRESENT` is false (no HMF/HGVAS/CXL appliance). `HMF_DEVICE_VISIBLE_TRUST` is BLOCKED. `FUSION_FIRST_HETEROGENEOUS_EXECUTABLE` is UNREACHABLE because it depends on that gate. IV-B/C/D genes are BLOCKED_HARDWARE.

Software object semantics exist. `tools/acceptance/odyssey/run.py:684` calls `hmf_objects.establish_clean` and `:695` `device_digest`. The graph reports 0 runtime callers because the catalog lists paths and no symbol — a catalog gap, not missing code. Status would remain BLOCKED_HARDWARE either way.

`receipts/future/HMF_MANAGED_OBJECTS.json` is `STATIC_ONLY`: "Static sidecar artifact. No hardware measurement."

| Previously blocked | Now scaffoldable | Now testable | Still physically blocked |
|---|---|---|---|
| Trust-state vocabulary as software | UMA-fixture objects; UNDECIDABLE migrate/recompute | `test_hmf_objects.py` | Device-visible trust; Fusion executable; eGPU/DGX HMF; IV-B promotion |

Scaffolds do not unlock Era IV. The graph is honest. Prose that treats the sidecar as device trust is not.

---

## A foundation the operator did not name, and should have

**VMCP production wiring** is the actual high-fanout Appendix O move sitting on disk.

Nine VMCP gates are `accepted=True` and `wired=False` (`VMCP_STATE_LATTICE`, `DEEP_DIGEST`, `TRUTH_LEDGER`, `TOOL_DOCTOR`, `FILE_CLASSIFIER`, `VISUAL_DIFF`, `SPATIAL_VALIDATE`, `COMPACT_SURFACE`, `AGENTOS_INTEGRATION`). Two more are SCAFFOLDED with accepted=False (`WEB_CAPTURE`, `PTY_CAPTURE`). `VMCP_RECEIPT_LAW` is already BUILT. `AGENTOS_BEHAVIOR_LAB` is ABSENT.

The acceptance harness (`tools/acceptance/vmcp/gates.py`) calls `vmcp_lattice_disposition` / `vmcp_capability_probe`. The auditor does not treat that as a production call, which is why accepted and wired can split. **One non-test production call of those symbols would, under current law, convert the accepted-but-unwired cluster from SCAFFOLDED to BUILT.** That is more Appendix O movement than Goal Compiler, HWIR lowering, native prefill, or HMF scaffolds can deliver on this machine.

`FLASH_SOURCE_VERIFIED` is the same shape (accepted, unwired, `tools.flash_organ_census`). Wiring it is hygiene: `FLASH_FIRST_GRAVITY_ORGAN` is already BUILT despite depending on it. Flash TPS/EBPW still wait on `FLASH_FULL_NOETIC_EXECUTABLE`.

---

## Surprises (loud on purpose)

1. **Dual ledger.** Appendix O `HCLI_CONTEXT_FOCUSED_WORKUNITS` is BUILT; sovereign G004 has no receipt. G001 is discharged and is not an Appendix O gate. Settle with a Gxxx ↔ Appendix O map.
2. **Dependencies are not closed.** FPGA_HWIR is WIRED while `FPGA_PREBOARD_SCHEMAS` is SCAFFOLDED. ModelLake promotion is BUILT while hash is BLOCKED. Flash first organ is BUILT while `FLASH_SOURCE_VERIFIED` is SCAFFOLDED. Settle by refusing WIRED/BUILT when a declared dependency is below that status.
3. **Two ContextCompilers, two GoalCompilers.** HCLI packet compiler ≠ Rust `ContextCompiler` ≠ `lab/lineage/continuity.py`. `GoalCompiler.compile` ≠ `goal_compile.ingest`. Same names, no calls between them.
4. **FPGA CLI vs catalog symbols.** `run_fpga_preboard` is reachable; inner symbols are SCAFFOLDED. Retargeting the catalog would reclassify three gates and still not wake U50.
5. **Pluggable lowering is unconsumed.** Recent merge added `LoweringTarget`. The HwirGraph callers never call `lower_hwir()`.
6. **Promotion without hash.** 55 identities, 4.35 TB unhashed, ATOMIC_PROMOTION BUILT.
7. **Saturation prose vs its own counts.** "Zero accepted" in `remaining_highest_ev_gaps` vs BUILT 18 / accepted 30.
8. **HMF callers vs graph.** `establish_clean` is called from the Odyssey acceptance runner; catalog has no symbol so the graph says 0 callers.
9. **G010 restarts.** Six restarts in 660s against ModelLake law. Needs process-table attribution, not a new scaffold.

---

## What would actually move the denominator

In order of effect on Appendix O, not on feelings:

1. **Wire VMCP** (accepted-but-unwired cluster) — cheap, software, high fan-out.
2. **Accept the eleven WIRED-not-accepted gates** — including `FLASH_COMPLETE_EBPW_LE_1` (historically 3.139 against ≤1), `MODELLAKE_HASH_VERIFIED` (4.35 TB), `FPGA_HWIR` (PREHARDWARE). These need measurements, not more files.
3. **First complete native Flash token / `FLASH_FULL_NOETIC_EXECUTABLE`** — already rank 1 in `ROADMAP_STATE.next_decisive_gates`. Science, not scaffolding.
4. **U50DD arrival** — wakes twelve gates. Absent.
5. **HMF appliance arrival** — wakes trust + Fusion. Absent.
6. **Theia training campaign** — wakes seven BLOCKED_EXTERNAL gates. No T0 substrate in this checkout.

Goal-compiler ingest on the live engine, Rust ContextCompiler on the HCLI daemon, native prefill G005, and G013 successor would move the *sovereign* ledger. They would not, by themselves, change 18/71 or 0/25 or 0.7.

---

## Authority and hardware

Authoritative here: source, call sites, tests, schema, receipts, mount presence.

Not authoritative: GPU/ANE/Metal numbers, FPGA/U50 facts, HMF device trust, thermal/power, protected benchmarks.

Present: Apple M3 Ultra, ANE. Absent: FPGA/U50DD, DGX, eGPU, HMF/HGVAS/CXL.

Constitution unchanged: five eras, three Odysseys, no Era VI, FPGA inside Accelerator/Fusion, Theia one bounty generalist, coordinate 0.7.

Did not: rewrite H-ROADMAP, implement, edit `hcli/` `tools/` `crates/` `civilization/`, signal the live daemon, write into ModelLake, run the auditor or the test suite.
