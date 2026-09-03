# aud03 — old vs current completion delta

Discovery / audit only. No roadmap rewrite, no implementation.

- HEAD: `04193ccbc8ef9fdd2dfd595d65f656760829dddc` — 2026-09-02 14:48:41 -0400 fix(hcli): a graceful stop, and a successful repair, no longer end the mission
- Capability graph generated from `7d64280068d7f1e1d239e3ffb783a10af485d037` (stale vs HEAD by the hwir merge and the hcli graceful-stop commit).
- Evidence tiers used: `SOURCE_INSPECTION`, `STATIC_VERIFICATION`. **No `PHYSICALLY_MEASURED` gate.** Hardware inventory commands ran this session; no GPU/ANE/FPGA performance number is claimed.
- Call-site rule: a module import is not a call. A definition is not a call. `tools/acceptance` is a harness (supports TESTED, not INTEGRATED unless `hcli/` also calls the symbol outside the defining file). An ACCEPTED receipt with `symbol_invoked=false` does not produce INTEGRATED/END_TO_END.

## The question

**Has anything built since the last audit actually unlocked more roadmap completion?**

Less than it looks. Built/end-to-end is 18 → 16 under a call-of-the-symbol rule that also refuses acceptance receipts that did not invoke the catalog symbol. Scaffolded-or-better 66.2% → 69.0% is mostly reclassification (Theia out of ABSENT; existing files the old pass called absent). No new physical carrier appeared. The systems that actually moved a gate's strongest honest state are listed below; none of them closes a hardware gate or the EBPW<=1 / TPS>=50 bars.

## OLD → CURRENT

| quantity | old (untrusted baseline) | current (this audit) | delta |
|---|---:|---:|---:|
| total gates | 71 | 71 | +0 |
| built / end-to-end | 18 | 16 | -2 |
| of which INTEGRATED | — | 15 |  |
| of which END_TO_END | — | 1 |  |
| scaffolded / callable / tested | 29 | 33 | +4 |
|   TESTED | — | 13 |  |
|   CALLABLE | — | 12 |  |
|   SCAFFOLDED | — | 8 |  |
| absent / concept-only | 11 | 1 | -10 |
| hardware-blocked | 13 | 14 | +1 |
| authority-blocked (Theia) | 0 | 7 | +7 |
| scaffolded-or-better count | 47 | 49 | +2 |
| scaffolded-or-better % | 66.2% | 69.0% | +2.80 |
| mean completion % (all 71) | 25.7% | 36.06% | +10.36 |
| software-addressable n | 58 | 50 | -8 |
| software-addressable mean % | 31.1% | 51.2% | +20.10 |
| genes mean % | 28.9% | 22.49% | -6.41 |
| genes scaffolded-or-better % | 88.0% | 80.0% | -8.00 |

Old baseline source: `tools/roadmap/saturation.py` BASELINE / `receipts/future/ROADMAP_SCAFFOLD_SATURATION.json`. That pass counted **module imports as call sites** and is retained only so the delta has a stated origin. Per-gate identity of the old 18 BUILT is not recoverable.

Scoring used for current means (documented, not the old unpublished weights): INTEGRATED=80, END_TO_END=70, TESTED=50, CALLABLE=40, SCAFFOLDED=20, ABSENT/CONCEPT_ONLY/BLOCKED_HARDWARE/BLOCKED_AUTHORITY=0. The +10.36 mean and +20.1 software-addressable mean **mix methodology with evidence** and must not be read as “the machine got 10 points better.” TESTED=50 vs old SCAFFOLDED=20 on the same VMCP files is almost entirely acceptance-receipt re-scoring.

Software-addressable n dropped 58 → 50 because the 7 Theia gates left ABSENT (0) for BLOCKED_AUTHORITY (excluded) and Fusion was moved from the movable set to hardware-blocked (HMF dependency).

## What actually moved

### AgentOS / HCLI acceptance receipts (FUNCTIONAL_SIM)

- What changed: receipts/acceptance/AGENTOS_*.json and HCLI_CONTEXT_*.json / BACKEND_FAILURE_ISOLATION.json dated 2026-09-02. Pytest/harness demonstrated criteria for repair, cancel, orphan, persistence, checkpoint, restart, context authority, focused workunits, failure isolation.
- Effect: These gates are INTEGRATED or TESTED on STATIC_VERIFICATION of receipts, not PHYSICALLY_MEASURED. They were already source-present; the campaign added acceptance evidence. That is not a new runtime capability.
- Unlocks a promotion bar: **no**

### VMCP acceptance receipts (FUNCTIONAL_SIM)

- What changed: Most VMCP_* receipts ACCEPTED. Catalog still has no implementing symbols for 11/13 VMCP gates, so they cannot be INTEGRATED under the call-of-the-symbol law.
- Effect: SCAFFOLDED → TESTED. Not daemon-integrated except VMCP_RECEIPT_LAW (run_vmcp_gate in agentos_cli.py).
- Unlocks a promotion bar: **no**

### Flash science CLI + two ACCEPTED flash receipts

- What changed: FLASH_FIRST_GRAVITY_ORGAN and FLASH_DENSE_VS_NF_AB ACCEPTED FUNCTIONAL_SIM with hcli/agentos_cli.py callers. FLASH_COMPLETE_EBPW_LE_1 still measured 3.139 against <=1. FLASH_ACCEPTED_TPS_GE_50 accepted_tps=None.
- Effect: Two flash gates INTEGRATED on sim receipts. The two promotion bars that would actually move I-C did not move.
- Unlocks a promotion bar: **no**

### FPGA_HWIR / s2-hwir-pluggable-lowering-target

- What changed: Commit da3e5c167 (merged f16ec8cdd) added ~1077 lines to tools/future/hwir.py plus test_hwir_lowering.py. HwirGraph( is called from tools/accelerator/backend_contract.py:1094, fusion_bridge.py:1444, tools/future/p6_projection.py:857, propagate.py:695.
- Effect: FPGA_HWIR is CALLABLE. All 12 U50_* gates remain BLOCKED_HARDWARE. Preboard/link/partition stay SCAFFOLDED because catalog symbols are inner types only constructed inside fpga_preboard.py.
- Unlocks a promotion bar: **no**

### hcli graceful-stop (04193ccbc)

- What changed: A graceful stop and a successful repair no longer end the mission.
- Effect: Does not add or close an appendix-O gate. May matter to AGENTOS_CANCELLATION/REPAIR behaviour of the live daemon; this sparse worktree cannot see the live uncommitted hcli tree.
- Unlocks a promotion bar: **no**

### QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED receipt

- What changed: Auditor graph marks BUILT. Receipt verdict ACCEPTED with symbol_invoked=false (run_qwen27_mlp_diagnostic_ab not invoked because it would start a resident next to the live daemon).
- Effect: This audit demotes it to CALLABLE. That is a quality correction, not an unlock.
- Unlocks a promotion bar: **no**

Did not move:

- U50DD/Alveo still absent (ioreg/Thunderbolt/system_profiler this session)
- HMF/HGVAS, DGX, eGPU still absent
- FLASH complete_ebpw 3.139 ≰ 1
- FLASH accepted_tps is None, not >= 50
- ODYSSEY_II transfer reduction not positive
- ODYSSEY_III no independent law refutation
- Theia: engine at tools/theia/, no trained generalist model
- I-B Doctor: still zero appendix-O gates

## Per era

| era | n | mean % | built/e2e | tested | callable | scaffolded | absent | hw | auth | sob % | sw-addr mean % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 60 | 41.33 | 16 | 13 | 10 | 8 | 1 | 12 | 0 | 78.3 | 51.67 |
| II | 1 | 40.0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 100.0 | 40.0 |
| III | 1 | 40.0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 100.0 | 40.0 |
| IV | 2 | 0.0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0.0 | 0.0 |
| bounty | 7 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0.0 | 0.0 |

Era I still holds 60 of 71 gates. Theia is `era=bounty`, not a sixth civilization.

## Per gene

Gene **label** is the strongest non-hardware/non-authority member. Gene **score** is the unweighted mean of member gates (hardware zeros included), or 20/0 for genes with no appendix-O gates.

| gene | era | status | score % | member gates | sob |
|---|---|---|---:|---:|---|
| I-A_AGENTOS_HCLI | I | INTEGRATED | 54.0 | 30 | yes |
| I-B_DOCTOR | I | SCAFFOLDED | 20 | 0 | yes |
| I-C_GRAVITY_NOETIC | I | INTEGRATED | 47.14 | 7 | yes |
| I-D_ACCELERATOR | I | INTEGRATED | 13.68 | 19 | yes |
| I-E_ODYSSEY_I | I | INTEGRATED | 67.5 | 4 | yes |
| II-A_ODYSSEY_II | II | CALLABLE | 40.0 | 1 | yes |
| II-B_NOETIC_COMPILER_V1 | II | SCAFFOLDED | 20 | 0 | yes |
| II-C_PHYSICAL_GRAPH_COMPILER | II | CALLABLE | 40 | 0 | yes |
| II-D_STATE_TOKENIZER_DECODING | II | SCAFFOLDED | 20 | 0 | yes |
| II-E_GREEN_MACHINE | II | SCAFFOLDED | 20 | 0 | yes |
| III-A_ODYSSEY_III | III | CALLABLE | 40.0 | 1 | yes |
| III-B_LEARNED_PHYSICAL_COMPILER | III | SCAFFOLDED | 20 | 0 | yes |
| III-C_RESIDENT_OPTIMIZER | III | SCAFFOLDED | 20 | 0 | yes |
| III-D_BEYOND_DENSE_REPRESENTATION | III | SCAFFOLDED | 20 | 0 | yes |
| III-E_AUTONOMOUS_REPRODUCIBLE_SCIENCE | III | SCAFFOLDED | 20 | 0 | yes |
| IV-A_FUSION | IV | BLOCKED_HARDWARE | 0.0 | 1 | no |
| IV-B_HMF_HGVAS | IV | BLOCKED_HARDWARE | 0.0 | 1 | no |
| IV-C_DGX_SPARK | IV | BLOCKED_HARDWARE | 0 | 0 | no |
| IV-D_EGPU | IV | BLOCKED_HARDWARE | 0 | 0 | no |
| IV-E_FUSION_BRIDGE_TOPOLOGY_ASCENSION | IV | SCAFFOLDED | 20 | 0 | yes |
| V-A_PRODUCT_SOVEREIGNTY | V | SCAFFOLDED | 20 | 0 | yes |
| V-B_DEVELOPER_PLATFORM | V | SCAFFOLDED | 20 | 0 | yes |
| V-C_CONTINUOUS_VERIFIED_IMPROVEMENT | V | SCAFFOLDED | 20 | 0 | yes |
| V-D_DOMINANCE_SCOREBOARD | V | SCAFFOLDED | 20 | 0 | yes |
| V-E_PERPETUAL_HAWKING | V | CONCEPT_ONLY | 0 | 0 | no |

**I-B_DOCTOR has zero of the 71 appendix-O gates.** `civilization/ROADMAP_STATE.json` obligation completion for I-B is a different ledger and was not mixed in.

I-D_ACCELERATOR labels INTEGRATED because `QWEN27_RUNTIME_IDENTITY_FROZEN` is INTEGRATED, but scores 13.68% because 12/19 members are U50 `BLOCKED_HARDWARE`. Do not read the label as “accelerator done.”

## Disagreements with `civilization/CAPABILITY_GRAPH.json`

The graph claims 18 BUILT. This audit keeps 16 in the built/end-to-end bucket (15 INTEGRATED + 1 END_TO_END) and demotes two graph-BUILT gates:

- `HCLI_SELF_SUPPLEMENT`: graph `BUILT` → this audit `TESTED`. Graph BUILT via acceptance harness as the only non-defining-file caller. This audit does not promote a defining-file helper + harness into INTEGRATED.
- `QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED`: graph `BUILT` → this audit `CALLABLE`. Production CLI caller exists, so CALLABLE. Acceptance is a STATIC identity explanation, not a run of the catalog symbol.

TESTED vs graph SCAFFOLDED (VMCP mostly, FLASH_SOURCE_VERIFIED, AGENTOS_CIRCUIT_BREAKER, HCLI_CONTEXT_INVALIDATION) is a refinement, not a disagreement on wiring: those gates still have no production caller of a catalog symbol outside the harness.

## Surprises (loud)

### acceptance_ignores_symbol_invoked (high)

tools/roadmap/auditor.py _accepted_fact treats receipts/acceptance/<GATE>.json as accepted when verdict=ACCEPTED, criterion_altered is false, and command is set. It does not read symbol_invoked. QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED is ACCEPTED with symbol_invoked=false.

What would settle it: Change the auditor to refuse accepted when symbol_invoked is false, or regenerate the receipt after actually calling run_qwen27_mlp_diagnostic_ab on a machine where that is allowed.

### resolve_vs_Path.resolve (high)

Catalog symbol for HCLI_CONTEXT_AUTHORITY_UNIFIED is resolve. A name grep of resolve( is pathlib.Path.resolve across hundreds of files. Production correctly binds via `from .context_budget import resolve as resolve_context_budget` in hcli/backends.py, config.py, engine.py, runtime.py.

What would settle it: Rename the catalog symbol to the alias the daemon actually calls, or require AST binding (which the auditor already does when the index is used). Do not use name grep for this gate.

### graph_line_drift (medium)

CAPABILITY_GRAPH.json generated_from_commit=7d6428; HEAD=04193cc. Several graph runtime_caller lines no longer contain the cited symbol (HCLI_CONTEXT_FOCUSED_WORKUNITS mission.py:983 is `with self._lock:` at HEAD; the real call is mission.py:1055).

What would settle it: Regenerate the graph at HEAD. Citations bound to a commit must be re-resolved on drift, not only past-EOF.

### stale_saturation_prose (high)

receipts/future/ROADMAP_SCAFFOLD_SATURATION.json final.by_status.BUILT=18 but remaining_highest_ev_gaps[0] still says 'No gate has demonstrated ACCEPTANCE' / '26 gates are wired and zero are accepted'. Hardcoded gap text in tools/roadmap/saturation.py was not updated when acceptance receipts started being consumed.

What would settle it: Derive the gap list from the graph (wired_but_not_accepted and built_gates), never from a frozen string. Fix the producer, not the artifact.

### fpga_catalog_inner_types (medium)

hcli/agentos_cli.py:1019 calls run_fpga_preboard, which constructs FPGADeviceGenome, TransportLinkSimulator, and simulate_partition. Catalog symbols are those inner types, so FPGA_PREBOARD_SCHEMAS / FPGA_LINK_SIM / FPGA_PARTITION_SIM stay SCAFFOLDED.

What would settle it: Retarget catalog symbols to run_fpga_preboard, then re-audit. Still would not wake U50_*.

### u50_mixed_false_wire (medium)

U50_MIXED_APPLE_FPGA_GRAPH was listed wired via nr_nx_generic.py:1981 subprocess.run. The argv is tools/odyssey/physical_graph_compiler.py — Odyssey II-C, not a mixed Apple+FPGA executable.

What would settle it: Attribute that subprocess only to II-C_PHYSICAL_GRAPH_COMPILER (already WIRED/CALLABLE). Keep U50_MIXED BLOCKED_HARDWARE.

### doctor_has_zero_appendix_o_gates (medium)

I-B_DOCTOR is a canonical program with civilization/ROADMAP_STATE.json obligations, but zero of the 71 appendix-O gates. Mixing obligation completion (ROADMAP_STATE I-B 66.7%) with gate completion inflates Doctor.

What would settle it: Either add Doctor gates to appendix O or keep gene I-B as SCAFFOLDED-with-no-gates and never average it with the obligation ledger.

### acceptance_harness_is_not_a_test_path (medium)

is_test_path only matches test_* / *_test.py. tools/acceptance/**/harness.py and gates.py count as production callers. Several graph BUILT gates are wired only that way (HCLI_SELF_SUPPLEMENT).

What would settle it: Treat tools/acceptance as test-like for wiring, or require an hcli/ caller outside the defining file for INTEGRATED.

### live_hcli_uncommitted_not_in_this_worktree (medium)

A live HCLI daemon holds ~110 uncommitted files under hcli/. This worktree is sparse and does not materialize hcli/; audit is HEAD 04193cc blobs via git show. Live daemon behaviour may differ.

What would settle it: A read-only census of the daemon's checkout (without signalling it) compared to HEAD, or regenerate the graph against that tree.

### xcrun_metal_absent_ane_kexts_present (low)

xcrun -f metal failed on this PATH. AppleH16ANEInterface.kext / AppleT6041ANEHAL.kext etc. are installed. Inventory only; no ANE performance claimed.

What would settle it: A bounded ANE compile/placement/latency probe, labelled MEASURED, on this M3 Ultra. Out of this audit's authority until run.

## Hardware inventory this session (not performance)

- SoC: Apple M3 Ultra (Mac15,14)
- GPU: Apple M3 Ultra 60-core built-in, Metal supported (system_profiler SPDisplaysDataType)
- ANE kexts present: AppleH11ANEInterface.kext, AppleH16ANEInterface.kext, AppleT6041ANEHAL.kext — **not** an ANE measurement
- U50/Alveo: absent
- DGX / nvidia-smi: absent
- eGPU: absent
- HMF/HGVAS/CXL: absent
- `xcrun metal`: not found on PATH

## Constitution (unchanged, not proposed)

Five eras. Three Odysseys. No Era VI. FPGA lives inside Hawking Accelerator/Fusion. Theia is one generalist bounty model. Civilizational coordinate 0.7. North star: HAWKING = SELF-OPTIMIZING PHYSICAL AI COMPUTER.

## How to read the JSON

Every object in `gates[]` has either `call_sites` (file/line/symbol/kind) or `blocker`. `evidence_tier` is never `MEASURED`. `auditor_graph_status` is the graph’s claim, not this audit’s.


## Commit status

`git add --sparse -A` failed: `index.lock: Operation not permitted` on the worktree git dir. Artifacts are on disk under `receipts/audit/`. They are not on a branch until something with a writable `.git` adds them.
