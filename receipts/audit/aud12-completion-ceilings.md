# aud12 — Honest completion ceilings

Discovery / audit pass. No roadmap rewrite, no implementation campaign.

- HEAD: `04193ccbc8ef9fdd2dfd595d65f656760829dddc`
- Capability graph commit: `7d64280068d7f1e1d239e3ffb783a10af485d037` (STATIC auditor, 2026-09-02T18:35:55Z)
- This document evidence tier: **STATIC_VERIFICATION** (plus SOURCE_INSPECTION / SIMULATED where labelled)
- **Never MEASURED.** This process did not run Metal, ANE compile, U50, DGX, eGPU, or powermetrics.

## The point

90% is a legitimate *software ceiling* on AgentOS, FPGA **preboard**, Doctor-as-planner-on-ModelLake, and the Theia **engine**. 90% is **falsification** on U50 physical runtime, DGX/eGPU/HMF execution, Theia-as-trained-model, Era V graduation, blended I-D, blended Era I, Flash EBPW≤1 from the 3.139 receipt, and Odyssey II transfer-complete from evaluations_avoided=−8.

## Constitution (unchanged)

Exactly five eras. Exactly three Odysseys. No Era VI. FPGA lives inside Hawking Accelerator / Fusion. Theia is one generalist bounty model, not another civilization. The 0.7% civilizational coordinate is a frozen heuristic against the complete five-era system; these ceilings do not recompute it. North star: HAWKING = SELF-OPTIMIZING PHYSICAL AI COMPUTER.

## Methodology

- **Definitions do not count without callers.** A module import is not a call of the implementing symbol. An earlier lane that counted imports was rejected; this audit keeps that bar.
- **BUILT = wired ∧ accepted.** Accepted means the gate's own bar, not a receipt on the topic. `FLASH_COMPLETE_EBPW_LE_1` is the load-bearing example: wired, measured 3.139 against required ≤ 1, therefore not accepted.
- Gene status is **not** a rollup of gates. I-A is SCAFFOLDED at gene level with 11 BUILT gates.
- Appendix O (71 gates) is the quantitative denominator when a program has gates. Programs with zero gates are scored gene-level and must not be given a fake verified %.
- Tests are not physical measurements. Simulation is not board reality. Receipts do not override a broken producer.
- U50 preboard and U50 physical are **different ceilings** and are never blended. Same for DGX contracts vs CUDA execution, Theia engine vs model ladder, ANE atlas vs ANE compile.

## Hardware (this host)

| Probe | Present | Evidence |
|---|---|---|
| Apple M3 Ultra, 96 GB, Mac15,14 | yes | `sysctl` / `system_profiler` |
| ANE / CoreML.framework | yes (silicon + framework) | framework path exists; **no** production `ANEProvider(...)` |
| Metal-capable GPU | silicon yes; this process's acceptance run said no | FLASH_NATIVE_NF_KERNEL blocker; see surprise S05 |
| U50 / Xilinx / Alveo | **no** | capability-graph probe `U50_PRESENT=false` |
| NVIDIA / DGX | **no** | `nvidia-smi` not on PATH |
| eGPU | **no** | no enclosure in Thunderbolt inventory |
| HMF / CXL memory appliance | **no** | `HMF_PRESENT=false` |
| ModelLake specimens | 55 dirs | `/Volumes/corpdrive/hawking-modellake/specimens` (read-only) |

`ps` / `launchctl` were not permitted in this profile. The live HCLI daemon is treated as present per contract and was not signalled.

## Where 90% is falsification

| Facet | 90% verdict | Why |
|---|---|---|
| **I-D blended** (19 Appendix O gates) | FALSIFICATION | 12/19 are U50 physical. Software ceiling 36.8%. |
| **Era I blended** | FALSIFICATION | 12/60 gates are U50. Software ceiling 80.0%. |
| **I-D-U50-PHYSICAL** | FALSIFICATION | 12/12 wake on `U50_PRESENT=false`. Cannot MEASURE board runtime. |
| **IV-C-DGX-EXECUTION** | FALSIFICATION | No NVIDIA device. Ceiling 0%. |
| **IV-D-EGPU-EXECUTION** | FALSIFICATION | No enclosure. Ceiling 0%. |
| **IV-A Fusion complete** | FALSIFICATION | One physical domain (APPLE). Gate UNREACHABLE. |
| **THEIA_MODEL_LADDER** | FALSIFICATION | 7/7 BLOCKED_AUTHORITY. No trained model. |
| **Era V INTEGRATED** | FALSIFICATION | Era I sovereign; V-E DORMANT. |
| **I-D-ANE physical qualification** | FALSIFICATION | Atlas forbids compile/placement/latency claims. |
| **FLASH_COMPLETE_EBPW_LE_1 as done** | FALSIFICATION | 3.139 vs ≤ 1. |
| **Odyssey II transfer-complete as done** | FALSIFICATION | evaluations_avoided = −8. |
| **I-A 90% as current verified** | FALSIFICATION | 11/30 = 36.7% BUILT. ROADMAP_STATE 80% is a different (weaker) methodology. |

## Where 90% is a legitimate ceiling

| Facet | Why 90% can be honest *if the remaining software actually meets the bars* |
|---|---|
| **I-A AgentOS / HCLI** | 29/30 scaffolded-or-better; 1 ABSENT (BEHAVIOR_LAB) is implementable; no missing device. |
| **I-D-FPGA-PREBOARD** | 4/4 gates software; CLI already calls `run_fpga_preboard`; receipt PASSED with board ABSENT. |
| **PB-01..21 compiler/HWIR/sim/HBM planning** | Distinct from Vivado/DriverKit/bitstream and from U50 physical. |
| **I-B Doctor (software planner)** | doctor6.prescribe and doctor_seal have callers; 55 lake specimens can host the unseen-model run. |
| **I-E lake identity / Odyssey I census** | 3/4 gates BUILT; hash-verify of remaining 54 is I/O not hardware. Do not blend with 'real weights executed'. |
| **THEIA_BOUNTY_ENGINE** | `tools/theia` + 11 tests. Not the model ladder. |
| **Era I non-U50 slice** | 48/60 gates; 90% of *that slice* is a software target. |
| **I-D-APPLE-METAL 3-gate Qwen27 slice** | No U50. Protected-baseline is authority-blocked (live daemon / no `ps`), not missing NVIDIA. |

## Era ceilings (Appendix O gates)

| Era | ROADMAP_STATE | n gates | verified % | scaffolded+ % | max software-addressable % | hw-blocked remainder % | 90% |
|---|---|---|---|---|---|---|---|
| I | PHYSICALLY_RUNNING | 60 | 30.0 | 78.3 | 80.0 | 20.0 | FALSIFICATION |
| II | EXPLORING | 1 | 0.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| III | NOT_STARTED | 1 | 0.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| IV | EXPLORING | 2 | 0.0 | 0.0 | 0.0 | 100.0 | FALSIFICATION |
| V | NOT_STARTED | 0 | 0.0 | 0.0 | 0.0 | 0.0 | FALSIFICATION |

Era I non-U50 slice: {"n": 48, "current_verified_pct": 37.5, "current_scaffolded_or_better_pct": 97.9, "max_software_addressable_today_pct": 100.0, "true_hardware_blocked_remainder_pct": 0.0}

Unweighted means across the 5 Era-I programs are **not** the era ceiling (they hide I-D's 19-gate U50 mass).

## Program ceilings

| Program | Era | gene class | n gates | verified % | scaffolded+ % | max SW today % | max w/o U50 % | max w/o DGX/eGPU/add % | hw remainder % | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| `I-A_AGENTOS_HCLI` | I | SCAFFOLDED | 30 | 36.7 | 96.7 | 100.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `I-B_DOCTOR` | I | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `I-C_GRAVITY_NOETIC` | I | SCAFFOLDED | 7 | 28.6 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `I-D_ACCELERATOR` | I | SCAFFOLDED | 19 | 10.5 | 36.8 | 36.8 | 36.8 | 36.8 | 63.2 | FALSIFICATION |
| `I-E_ODYSSEY_I` | I | SCAFFOLDED | 4 | 75.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `II-A_ODYSSEY_II` | II | SCAFFOLDED | 1 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `II-B_NOETIC_COMPILER_V1` | II | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `II-C_PHYSICAL_GRAPH_COMPILER` | II | CALLABLE | 0 | 0.0 | 100.0 | 80.0 | 80.0 | 80.0 | 0.0 | FALSIFICATION |
| `II-D_STATE_TOKENIZER_DECODING` | II | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `II-E_GREEN_MACHINE` | II | SCAFFOLDED | 0 | 0.0 | 100.0 | 55.0 | 55.0 | 55.0 | 0.0 | FALSIFICATION |
| `III-A_ODYSSEY_III` | III | SCAFFOLDED | 1 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `III-B_LEARNED_PHYSICAL_COMPILER` | III | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `III-C_RESIDENT_OPTIMIZER` | III | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `III-D_BEYOND_DENSE_REPRESENTATION` | III | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `III-E_AUTONOMOUS_REPRODUCIBLE_SCIENCE` | III | SCAFFOLDED | 0 | 0.0 | 100.0 | 90.0 | 90.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `IV-A_FUSION` | IV | SCAFFOLDED | 1 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 | FALSIFICATION |
| `IV-B_HMF_HGVAS` | IV | BLOCKED_HARDWARE | 1 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 | FALSIFICATION |
| `IV-C_DGX_SPARK` | IV | BLOCKED_HARDWARE | 0 | 0.0 | 0.0 | 40.0 | 40.0 | 0.0 | 60.0 | FALSIFICATION |
| `IV-D_EGPU` | IV | BLOCKED_HARDWARE | 0 | 0.0 | 100.0 | 45.0 | 45.0 | 0.0 | 55.0 | FALSIFICATION |
| `IV-E_FUSION_BRIDGE_TOPOLOGY_ASCENSION` | IV | SCAFFOLDED | 0 | 0.0 | 100.0 | 40.0 | 40.0 | 40.0 | 60.0 | FALSIFICATION |
| `V-A_PRODUCT_SOVEREIGNTY` | V | SCAFFOLDED | 0 | 0.0 | 100.0 | 70.0 | 70.0 | 70.0 | 0.0 | FALSIFICATION |
| `V-B_DEVELOPER_PLATFORM` | V | SCAFFOLDED | 0 | 0.0 | 100.0 | 70.0 | 70.0 | 70.0 | 0.0 | FALSIFICATION |
| `V-C_CONTINUOUS_VERIFIED_IMPROVEMENT` | V | SCAFFOLDED | 0 | 0.0 | 100.0 | 70.0 | 70.0 | 70.0 | 0.0 | FALSIFICATION |
| `V-D_DOMINANCE_SCOREBOARD` | V | SCAFFOLDED | 0 | 0.0 | 100.0 | 70.0 | 70.0 | 70.0 | 0.0 | FALSIFICATION |
| `V-E_PERPETUAL_HAWKING` | V | CONCEPT_ONLY | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | FALSIFICATION |

Programs with `n gates = 0` have **no Appendix O denominator**. `verified % = 0` there means 'no BUILT appendix-O gate', not 'zero code'. See gene defs and independent findings.

`max w/o U50` on IV-A/IV-B is 'this Appendix O row is not a U50_* gate', not 'Fusion/HMF can complete without a second physical domain'. Those remain FALSIFICATION at 90%.

## Split facets (do not re-blend)

| Facet | verified % | scaffolded+ % | max SW % | hw remainder % | 90% |
|---|---|---|---|---|---|
| `I-D-APPLE-METAL` | 66.7 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `I-D-FPGA-PREBOARD` | 0.0 | 100.0 | 100.0 | 0.0 | LEGITIMATE_CEILING |
| `I-D-U50-PHYSICAL` | 0.0 | 0.0 | 0.0 | 100.0 | FALSIFICATION |
| `I-D-ANE` | 0.0 | 100.0 | 70.0 | 0.0 | FALSIFICATION |
| `IV-C-DGX-CONTRACTS` | 0.0 | 0.0 | 40.0 | 60.0 | FALSIFICATION |
| `IV-C-DGX-EXECUTION` | 0.0 | 0.0 | 0.0 | 100.0 | FALSIFICATION |
| `IV-D-EGPU-CONTRACTS` | 0.0 | 100.0 | 45.0 | 55.0 | FALSIFICATION |
| `IV-D-EGPU-EXECUTION` | 0.0 | 0.0 | 0.0 | 100.0 | FALSIFICATION |
| `THEIA_BOUNTY_ENGINE` | 0.0 | 100.0 | 90.0 | 0.0 | LEGITIMATE_CEILING |
| `THEIA_MODEL_LADDER` | 0.0 | 0.0 | 0.0 | 0.0 | FALSIFICATION |

### FPGA preboard PB-01..PB-21

SCAFFOLDED-or-better today: **14/21 = 66.7%**.

| ID | Name | Class |
|---|---|---|
| PB-01 | FPGAProvider trait | SCAFFOLDED |
| PB-02 | MockFPGAProvider | SCAFFOLDED |
| PB-03 | SimFPGAProvider | ABSENT |
| PB-04 | TransportGenome schema | SCAFFOLDED |
| PB-05 | FPGAGenome schema | SCAFFOLDED |
| PB-06 | HBMGenome schema | SCAFFOLDED |
| PB-07 | HardwareProfile schema | SCAFFOLDED |
| PB-08 | HardwareModule schema | SCAFFOLDED |
| PB-09 | HawkingHWIR skeleton | CALLABLE |
| PB-10 | physical-graph partitioner | SCAFFOLDED |
| PB-11 | synthetic link simulator | SCAFFOLDED |
| PB-12 | latency/bandwidth sensitivity simulator | SCAFFOLDED |
| PB-13 | FPGA receipt format | TESTED |
| PB-14 | hardware experiment DAG | CONCEPT_ONLY |
| PB-15 | bitstream/module cache | CONCEPT_ONLY |
| PB-16 | DriverKit direct-PCIe research | CONCEPT_ONLY |
| PB-17 | Shared Activation Mailbox hypothesis | CONCEPT_ONLY |
| PB-18 | Vivado/Vitis reproducible build harness | ABSENT |
| PB-19 | RTL/HLS oracle/verifier harness | ABSENT |
| PB-20 | Qwen27 partition simulations | TESTED |
| PB-21 | Flash expert/state/n-gram simulations | TESTED |

`HCLI_FPGA_PREBOARD.json` status=PASSED, `physical_board.status=ABSENT`, simulations labelled SIMULATION_ONLY. That is a preboard ceiling, not a U50 ceiling.

## Odysseys

| Odyssey | Gate | Graph | Class | Bar | 90% |
|---|---|---|---|---|---|
| I Discovery | ODYSSEY_I_DISCOVERY | BUILT | TESTED/END_TO_END (STATIC census, no weight materialisation) | ACCEPTED | LEGITIMATE_CEILING for lake/census; not for 'real weights executed' |
| II Transfer | ODYSSEY_II_TRANSFER | WIRED | CALLABLE | evaluations_avoided=**−8** | 90% *now* = falsification; planner software ceiling is legitimate |
| III Adversarial | ODYSSEY_III_ADVERSARIAL_META_SCIENCE | WIRED | CALLABLE | synthetic REFUTED, physical_arm=not_run | 90% as independent law refutation = falsification until a non-synthetic arm |

There is no Odyssey IV.

## Appendix O gate rollup

From `civilization/CAPABILITY_GRAPH.json`: {'BUILT': 18, 'SCAFFOLDED': 20, 'WIRED': 11, 'ABSENT': 1, 'BLOCKED_HARDWARE': 13, 'UNREACHABLE': 1, 'BLOCKED_EXTERNAL': 7}

| Status | n | This audit class |
|---|---|---|
| BUILT | 18 | TESTED or END_TO_END (not PHYSICALLY_MEASURED here) |
| WIRED | 11 | CALLABLE (bar open or refused) |
| SCAFFOLDED | 20 | SCAFFOLDED (definition; no non-test symbol call) |
| ABSENT | 1 | ABSENT (`AGENTOS_BEHAVIOR_LAB`) |
| BLOCKED_HARDWARE | 13 | BLOCKED_HARDWARE (12× U50 + HMF) |
| BLOCKED_EXTERNAL | 7 | BLOCKED_AUTHORITY (Theia model ladder) |
| UNREACHABLE | 1 | BLOCKED_HARDWARE (`FUSION_FIRST_HETEROGENEOUS_EXECUTABLE`) |

Movable (not hardware/external) scaffolded-or-better: 49/51 = 96.1% (saturation receipt). That number is **not** verified completion and **not** a licence to print 90% on I-D or Era I.

## Surprises (flagged, not smoothed)

### S01_DOCTOR_HAS_ZERO_APPENDIX_O_GATES — high

I-B Doctor has 12 H-ROADMAP subgenes and zero Appendix O gates. Scoring I-B from the 71-gate ledger silently reports nothing.

*Would settle:* Add an Appendix O gate for the unseen-specimen planner, or score I-B only from doctor6/doctor_seal call sites and an explicit unseen-specimen receipt.

### S02_GENE_STATUS_IS_NOT_A_GATE_ROLLUP — high

I-A_AGENTOS_HCLI gene status is SCAFFOLDED while 11 of its 30 gates are BUILT. Using gene.status as program completion under-reports I-A (and similarly mis-ranks every program).

*Would settle:* Treat gene.status as a probe of one implementing-module bundle, never as min/mean of its gates.

### S03_FPGA_PREBOARD_NAMED_SYMBOL — medium

FPGA_PREBOARD_SCHEMAS is SCAFFOLDED because FPGADeviceGenome is only constructed inside fpga_preboard.py. hcli/agentos_cli.py:1019 calls run_fpga_preboard, which constructs the genome. Named-symbol discipline forbids upgrading the gate.

*Would settle:* Point the catalog symbol at run_fpga_preboard and re-run tools/roadmap/auditor.py. Do not hand-edit CAPABILITY_GRAPH.json.

### S04_ANEPROVIDER_TEST_ONLY_CTOR — high

ANE hardware is present (M3 Ultra, CoreML.framework). ANEProvider(...) is constructed only in hcli/test_ane_provider.py. APPLE_ANE_ATLAS.json is ATLAS_SCAFFOLD_COMPILE_BOUNDARY and forbids compilation/placement/latency claims.

*Would settle:* A non-test ANEProvider(...) on the physical_graph/runtime path plus a public Core ML compile/placement receipt. Until then ANE is SCAFFOLDED, not PHYSICALLY_MEASURED.

### S05_METAL_DISPATCH_FAILED_ON_M3_ULTRA — high

FLASH_NATIVE_NF_KERNEL acceptance refused: 'metal: no Metal-capable GPU' while system_profiler reports Apple M3 Ultra / Metal Supported. This audit's process similarly could not treat GPU runtime as MEASURED. hawking-core engine.rs still calls MetalContext::new_with_trace.

*Would settle:* Re-run the kernel-parity gate in an unsandboxed profile that can open Metal, without signalling the live HCLI daemon. Historical DSV4F receipts are not this run.

### S06_ODYSSEY_II_TRANSFER_IS_NEGATIVE — high

ODYSSEY_II_TRANSFER is WIRED. evaluations_avoided measured -8 (cold 2 evals, transfer 10). actual_saved_experiments=null/UNMEASURED.

*Would settle:* A cold-vs-transfer receipt on an unseen specimen with evaluations_avoided > 0. Similarity scores are not that bar.

### S07_EBPW_3_139_VS_BAR_1 — high

FLASH_COMPLETE_EBPW_LE_1 is WIRED. incumbent.complete_ebpw=3.139300850311054 against required <= 1. FLASH_COMPLETE_V0 exact control is 16.0 EBPW (BF16 fallback). evidence_class STATIC_ONLY, gpu_authority false.

*Would settle:* A mix_report on a complete executable whose complete_ebpw <= 1, or a changed bar with a negative control. Do not round 3.14 to a pass.

### S08_ROADMAP_STATE_IA_80_VS_GATES_40 — high

civilization/ROADMAP_STATE.json reports I-A completion_pct=80.0 status=ADVERSARIALLY_VERIFIED from 9 evidence categories × obligation counts. Appendix O says 11/30 BUILT = 36.7% verified. Different methodology, same name.

*Would settle:* Stop publishing ROADMAP_STATE completion_pct as if it were wired∧accepted. The 0.7% civilizational coordinate is constitutionally frozen and is not recomputed here.

### S09_SATURATION_GAP_TEXT_STALE — medium

receipts/future/ROADMAP_SCAFFOLD_SATURATION.json remaining_highest_ev_gaps[0] still says 'No gate has demonstrated ACCEPTANCE' / 'zero are accepted' while final.by_status.BUILT=18 and the graph has 30 accepted facts.

*Would settle:* Fix the producer (tools/roadmap/saturation.py remaining_highest_ev_gaps), regenerate. Do not hand-edit the receipt.

### S10_DGX_CONTRACT_MODULE_MISSING — medium

IV-C_DGX_SPARK catalog probe is paths=() modules=(). Execution is correctly BLOCKED_HARDWARE (no nvidia-smi). The software contract is not even SCAFFOLDED, unlike eGPU which at least has fusion_planner.topology_apple_spark_egpu.

*Would settle:* A DGX provider/device-contract module with tests and no HARDWARE_MEASURED claims. That raises the *contract* ceiling without touching execution.

### S11_SIMFPGA_PROVIDER_ABSENT — low

H-ROADMAP PB-03 SimFPGAProvider: git grep on HEAD is empty. MockFPGAProvider exists.

*Would settle:* Implement SimFPGAProvider or drop PB-03 from the preboard 90% denominator.

### S12_U50_MIXED_WIRED_AND_HARDWARE_BLOCKED — low

U50_MIXED_APPLE_FPGA_GRAPH is wired (subprocess of physical_graph_compiler.py) and still BLOCKED_HARDWARE. Correct: a caller does not instantiate a U50.

*Would settle:* Nothing — keep the override. Citing the caller as U50-complete would be the bug.

### S13_THEIA_ENGINE_PRESENT — medium

verified_absent.theia verdict is PRESENT (tools/theia/*). All 7 THEIA_* gates remain BLOCKED_EXTERNAL. An earlier lane that called Theia absent was wrong about the engine and would still be wrong to call the model present.

*Would settle:* Keep the split: engine vs model ladder. Training-before-Gravity (19.10) is the law.

### S14_HASH_VERIFY_1_OF_55 — medium

ModelLake specimens/ listing is 55. MODELLAKE_HASH_VERIFIED acceptance: canonical Qwen3-0.6B verify_only ok; 54 sealed specimens (~4.351 TB) lack oid-backed sha256.

*Would settle:* reconcile() over remaining oids. This is disk I/O, not U50/DGX.

### S15_PS_BLOCKED_IN_THIS_PROFILE — medium

ps and launchctl failed with 'operation not permitted'. The live HCLI daemon is assumed present per the contract and must not be signalled. QWEN27_PROTECTED_BASELINE and HCLI_SELF_OPTIMIZATION_BOOTSTRAP cannot be accepted in this profile.

*Would settle:* An unsandboxed (gate) profile that can observe process state without sending signals, or a daemon-owned quiescence receipt.

### S16_FLASH_FULL_NX_CALLER_VS_SCAFFOLDED — medium

tools/acceptance/flash/run_gates.py:744 calls noetic_compiler, but FLASH_FULL_NOETIC_EXECUTABLE is SCAFFOLDED (catalog has no symbols=).

*Would settle:* Add symbols=(round_trip,) or equivalent to the catalog and re-audit. Do not hand-promote the gate.

### S17_FUSION_SINGLE_DOMAIN — high

FUSION_FIRST_HETEROGENEOUS_EXECUTABLE UNREACHABLE: present fusion nodes ['APPLE'] (need >=2). U50=False, EGPU=False, DGX=False. simulate_default timing_decidable=False.

*Would settle:* A second physical domain on the inventory, then a non-COST_MODEL hop. Not a more detailed apple-alone sim.

### S18_CIVILIZATIONAL_COORDINATE_NOT_A_LEDGER — low

ROADMAP_STATE civilization_progress.value_pct=0.7 is heuristic against five eras × 25 civilizations and is constitutionally not to be casually rewritten. These ceilings do not replace it.

*Would settle:* Leave 0.7% in place. Do not recompute it from gate counts (that is the inflation S015 forbids).

## Stale roadmap text

- `civilization/ROADMAP_STATE.json` `civilization_status.I-A_AGENTOS_HCLI.completion_pct` — Obligation×category minimum, not wired∧accepted.
- `civilization/ROADMAP_STATE.json` `civilization_status.I-D_ACCELERATOR.evidence 9/9 with completion_pct 13.3` — Category coverage is not gate closure; the file itself warns this. Still easy to quote as 100% evidence.
- `receipts/future/ROADMAP_SCAFFOLD_SATURATION.json` `remaining_highest_ev_gaps[0]` — Fix tools/roadmap/saturation.py; do not edit the receipt by hand.
- `civilization/ROADMAP_STATE.json` `program_statuses.II-A_ODYSSEY_II.status=NOT_STARTED` — NOT_STARTED understates a wired planner whose numeric bar failed.
- `civilization/ROADMAP_STATE.json` `program_statuses.III-A_ODYSSEY_III.status=NOT_STARTED` — NOT_STARTED understates a wired scars() caller whose acceptance is synthetic-only.

## Independent call sites this audit actually grepped

Not imports.

| Symbol | Non-test call | Used as |
|---|---|---|
| `pick_acquire_candidate` | `hcli/acquisition.py:172`, `tools/acceptance/odyssey/run.py:304` | Odyssey I BUILT |
| `HwirGraph` | `tools/accelerator/backend_contract.py:1094`, `fusion_bridge.py:1444`, `tools/future/p6_projection.py:857` | FPGA_HWIR WIRED |
| `run_fpga_preboard` | `hcli/agentos_cli.py:1019` | independent CALLABLE; does **not** upgrade FPGA_PREBOARD_SCHEMAS |
| `FPGADeviceGenome` | only inside `fpga_preboard.py:483` | why the gate stays SCAFFOLDED |
| `MetalContext::new_with_trace` | `crates/hawking-core/src/engine.rs:22` | Apple Metal INTEGRATED at source; not MEASURED here |
| `prescribe` | `lab/operators/doctor6/__main__.py:149` | Doctor CALLABLE |
| `c2m.translate` | `tools/accelerator/c2m_t3_census.py:195` | C2M CALLABLE; T3 still NOT CLAIMED |
| `ANEProvider(` | **only** `hcli/test_ane_provider.py` | ANE SCAFFOLDED |
| `SimFPGAProvider` | none | ABSENT |
| `call_noetic_compiler` | `tools/acceptance/flash/run_gates.py:744` | independent; does **not** upgrade FLASH_FULL_NOETIC_EXECUTABLE |

## What this audit did not do

- Did not re-run `tools/roadmap/auditor.py`.
- Did not dispatch Metal or ANE.
- Did not hash 54 remaining ModelLake specimens.
- Did not touch `hcli/`, `tools/`, `crates/`, `civilization/`, or H-ROADMAP.md.
- Did not kill, signal, or restart any process.
- Did not write MEASURED for GPU/ANE/U50/DGX/eGPU/thermal.

Machine-readable source of truth: `receipts/audit/aud12-completion-ceilings.json`.

