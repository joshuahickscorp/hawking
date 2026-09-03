# PART II — OPTIMIZED REMAINING ACTION PLAN

Organized by what BLOCKS each action, not by era. The class is derived from the
gate's own evidence -- a hand-assigned class is an opinion about difficulty, and
the whole point of this part is to separate 'I have not connected these two
components yet' from work that genuinely needs an experiment, wall time, or silicon.

## Blocker census

    SOFTWARE_CONNECTION_REMAINING    10
    EXPERIMENTATION_REQUIRED         18
    LONG_RUN_EVIDENCE_REQUIRED       14
    PHYSICAL_HARDWARE_REQUIRED       14
    UNKNOWN_RESEARCH                 1

    TOTAL REMAINING                  57

SOFTWARE_CONNECTION_REMAINING = 10 is the
number this campaign exists to drive toward zero. Every other class is honest
frontier: an experiment that must run, evidence that needs wall time, hardware that
must physically exist, or a question whose answer nobody has.

## SOFTWARE_CONNECTION_REMAINING (10)

### AGENTOS_CIRCUIT_BREAKER
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/test_failed_unit_says_why.py:19 (+14 more)
    implementation      hcli/scheduler.py, hcli/scheduler.py:55 (+1 more)
    unlocks             1 declared dependencies

### AGENTOS_RETRY_CLASSIFIED
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/test_failed_unit_says_why.py:19 (+14 more)
    implementation      hcli/scheduler.py, hcli/scheduler.py:424 (+1 more)
    unlocks             1 declared dependencies

### FPGA_LINK_SIM
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/agentos/test_fpga_preboard.py:5 (+1 more)
    implementation      hcli/agentos/fpga_preboard.py, hcli/agentos/fpga_preboard.py:216
    unlocks             1 declared dependencies

### FPGA_PARTITION_SIM
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/agentos/test_fpga_preboard.py:5 (+3 more)
    implementation      hcli/agentos/fpga_preboard.py, hcli/agentos/fpga_preboard.py:350
    unlocks             1 declared dependencies

### FPGA_PREBOARD_SCHEMAS
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/agentos/test_fpga_preboard.py:5 (+1 more)
    implementation      hcli/agentos/fpga_preboard.py, hcli/agentos/fpga_preboard.py:127
    unlocks             0 declared dependencies

### RUNTIME_CONTEXT_NATIVE
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/test_context_reduction.py:69 (+7 more)
    implementation      hcli/context_budget.py
    unlocks             0 declared dependencies

### RUNTIME_PREFIX_STATE_REUSE
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/test_prefix_and_prefill_instruments.py:15 (+3 more)
    implementation      hcli/prefix_probe.py, hcli/prefix_probe.py:29
    unlocks             0 declared dependencies

### STABLE_PREFIX_CONTEXT_ALIGNMENT
    missing             no non-test call site reaches this capability
    shortest verifier   hcli/test_prefix_and_prefill_instruments.py:15 (+4 more)
    implementation      hcli/prefix_probe.py, hcli/prefix_probe.py:29 (+1 more)
    unlocks             0 declared dependencies

### VMCP_AGENTOS_INTEGRATION
    missing             no non-test call site reaches this capability
    shortest verifier   tools/acceptance/test_accepted_gates_show_their_evidence.py:31
    implementation      hcli/agentos/vmcp_gate.py, tools/headless/hcli_vmcp_integration.py (+1 more)
    unlocks             2 declared dependencies

### VMCP_COMPACT_SURFACE
    missing             no non-test call site reaches this capability
    shortest verifier   must be written
    implementation      hcli/vmcp/__init__.py, hcli/vmcp_adapter.py
    unlocks             0 declared dependencies

## EXPERIMENTATION_REQUIRED (18)

### AGENTOS_DETERMINISTIC_OFFLOAD
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_abort_checkpoint_atomicity.py:18 (+14 more)
    implementation      lab/hcli/claude_offload_bench.py, hcli/delegate.py (+1 more)
    unlocks             0 declared dependencies

### FLASH_COMPLETE_EBPW_LE_1
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/future/test_complete_ebpw.py:14 (+7 more)
    implementation      tools/future/complete_ebpw.py, tools/future/complete_ebpw.py:193
    unlocks             1 declared dependencies

### FLASH_FULL_NOETIC_EXECUTABLE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/tests/test_hcli_flash_science.py:5 (+9 more)
    implementation      tools/odyssey/noetic_compiler.py, hcli/agentos/flash_executable.py
    unlocks             1 declared dependencies

### FLASH_NATIVE_NF_KERNEL
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/tests/test_hcli_flash_science.py:5 (+5 more)
    implementation      hcli/agentos/flash_executable.py, hcli/agentos/flash_graph_component.py (+1 more)
    unlocks             1 declared dependencies

### HCLI_MIXED_MAX
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_context_memory.py:7 (+23 more)
    implementation      hcli/max_policy.py, hcli/controller.py (+1 more)
    unlocks             1 declared dependencies

### HCLI_SELF_OPTIMIZATION_BOOTSTRAP
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/future/test_resident_optimizer.py:9 (+3 more)
    implementation      hcli/agentos/autonomy_gate.py, tools/future/resident_optimizer.py (+1 more)
    unlocks             3 declared dependencies

### HCLI_STATUS_PHYSICAL
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_command_registry.py:17 (+52 more)
    implementation      hcli/commands.py, hcli/processes.py (+2 more)
    unlocks             0 declared dependencies

### MODELLAKE_HASH_VERIFIED
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_acquisition.py:95 (+7 more)
    implementation      hcli/agentos/modellake_gate.py, tools/odyssey/modellake_watch.py (+1 more)
    unlocks             1 declared dependencies

### MODELLAKE_LIFECYCLE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/future/test_modellake_lifecycle.py:6 (+5 more)
    implementation      tools/future/modellake_lifecycle.py, tools/future/modellake_lifecycle.py:39
    unlocks             0 declared dependencies

### ODYSSEY_III_ADVERSARIAL_META_SCIENCE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/future/test_autonomy_scars.py:10 (+15 more)
    implementation      tools/future/repro_science.py, tools/future/autonomy_scars.py (+1 more)
    unlocks             1 declared dependencies

### ODYSSEY_II_TRANSFER
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/future/test_capability_stages.py:25 (+4 more)
    implementation      tools/future/qualification_pipeline.py, tools/future/qualification_pipeline.py:289
    unlocks             1 declared dependencies

### QWEN27_PROTECTED_BASELINE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/agentos/test_protected_accelerator_benchmark.py:3 (+1 more)
    implementation      hcli/agentos/protected_accelerator_benchmark.py, hcli/agentos/accelerator_regression.py (+1 more)
    unlocks             1 declared dependencies

### RUNTIME_COMPLETE_TOKEN_PROFILE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_prefix_and_prefill_instruments.py:14 (+5 more)
    implementation      hcli/prefill_profile.py, hcli/prefill_profile.py:102
    unlocks             0 declared dependencies

### RUNTIME_DECODE_PROTECTED
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_constrained_decoding.py:51 (+6 more)
    implementation      hcli/hawking_native.py
    unlocks             0 declared dependencies

### RUNTIME_DELTANET_STATE_REUSE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/headless/test_prefill_kv.py:15 (+7 more)
    implementation      tools/headless/prefill_kv.py, tools/headless/prefill_kv.py:239
    unlocks             0 declared dependencies

### RUNTIME_NATIVE_PREFILL
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_prefix_and_prefill_instruments.py:14 (+6 more)
    implementation      hcli/prefill_profile.py, hcli/hawking_native.py (+1 more)
    unlocks             0 declared dependencies

### RUNTIME_PREFILL_PHYSICAL_FRONTIER
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   hcli/test_prefix_and_prefill_instruments.py:14 (+5 more)
    implementation      hcli/prefill_profile.py, hcli/prefill_profile.py:102
    unlocks             0 declared dependencies

### VMCP_PTY_CAPTURE
    missing             wired and verified; its acceptance criterion has never been run
    shortest verifier   tools/acceptance/vmcp/test_receipt_law_defining_property.py:24 (+7 more)
    implementation      tools/vmcp/pty_eye.py, tools/headless/vmcp_capability_probe.py (+1 more)
    unlocks             1 declared dependencies

## LONG_RUN_EVIDENCE_REQUIRED (14)

### FLASH_ACCEPTED_TPS_GE_50
    missing             requires a MEASURED accepted capability-preserving TPS >= 50 under the protected window, against complete-system EBPW <= 1.00 (roadmap 13, line 1607). The roadmap calls these research targets, not cur
    shortest verifier   must be written
    implementation      tools/flash_stateful_gate.py
    unlocks             1 declared dependencies

### HAWKING_PUBLIC_MVP
    missing             HCLI_OPERATIONAL and ODYSSEY_DETACHED and KNOWN_GOOD_RUNTIME_COMMIT frozen. Isolated worktree only; merge after parity. Never reorganize underneath an active Odyssey.
    shortest verifier   must be written
    implementation      absent
    unlocks             0 declared dependencies

### REPO_TOPOLOGY_COMPRESSION
    missing             HCLI_OPERATIONAL and ODYSSEY_DETACHED and KNOWN_GOOD_RUNTIME_COMMIT frozen. Isolated worktree only; merge after parity. Never reorganize underneath an active Odyssey.
    shortest verifier   must be written
    implementation      absent
    unlocks             0 declared dependencies

### SEMANTIC_COMPRESSION
    missing             HCLI_OPERATIONAL and ODYSSEY_DETACHED and KNOWN_GOOD_RUNTIME_COMMIT frozen. Isolated worktree only; merge after parity. Never reorganize underneath an active Odyssey.
    shortest verifier   must be written
    implementation      absent
    unlocks             0 declared dependencies

### THEIA_BOUNTY_GENERALIST_QUALIFIED
    missing             the bounty ENGINE exists at tools/theia/ but no qualified generalist MODEL does. Wake: THEIA_RESEARCH=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### THEIA_GRAVITY_EXECUTABLE
    missing             needs a frozen Theia capability baseline to compress (training-before-Gravity law, roadmap 19.10). Wake: THEIA_RESEARCH=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### THEIA_LAB
    missing             needs a trained ~7B-14B student. Wake: THEIA_MICRO=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### THEIA_MICRO
    missing             needs a trained ~1B-3B student. Wake: THEIA_T0_TRAIN_SUBSTRATE=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### THEIA_RESEARCH
    missing             needs a trained ~30B-100B+ flagship. Wake: THEIA_WORKER=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### THEIA_T0_TRAIN_SUBSTRATE
    missing             Hawking Train T0 substrate is not a live campaign here: no teacher registry, data lake, trace store, curriculum or checkpoint authority runs in this checkout. Wake: T0_TEACHER_REGISTRY_LIVE.
    shortest verifier   must be written
    implementation      absent
    unlocks             0 declared dependencies

### THEIA_WORKER
    missing             needs a trained ~20B-40B student. Wake: THEIA_LAB=PASS.
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

### VMCP_SPATIAL_VALIDATE
    missing             OBJ/GLTF parsing, the spatial validator and an independent renderer are PARKED on the visionmcp 3d extra plus Blender CLI. Wake: VISIONMCP_3D_EXTRA_AND_BLENDER.
    shortest verifier   must be written
    implementation      tools/headless/vmcp_capability_probe.py
    unlocks             1 declared dependencies

### VMCP_VISUAL_DIFF
    missing             visual diff is PARKED on the visionmcp compiler residual. Wake: VISIONMCP_COMPILER_RESIDUAL_AVAILABLE.
    shortest verifier   must be written
    implementation      tools/headless/vmcp_capability_probe.py
    unlocks             1 declared dependencies

### VMCP_WEB_CAPTURE
    missing             browser/CDP, HTML/DOM capture and CSS parsing are PARKED on the visionmcp web extra plus a host Chrome; no Hawking code is missing. Wake: VISIONMCP_WEB_EXTRA_INSTALLED.
    shortest verifier   must be written
    implementation      tools/headless/vmcp_capability_probe.py
    unlocks             1 declared dependencies

## PHYSICAL_HARDWARE_REQUIRED (14)

### FUSION_FIRST_HETEROGENEOUS_EXECUTABLE
    missing             dependencies unsatisfied: HMF_DEVICE_VISIBLE_TRUST
    shortest verifier   tools/future/test_fusion_sim.py:15
    implementation      tools/accelerator/fusion_planner.py, tools/future/fusion_sim.py
    unlocks             1 declared dependencies

### HMF_DEVICE_VISIBLE_TRUST
    missing             silicon absent; wakes on HMF_PRESENT
    shortest verifier   tools/future/test_hmf_objects.py:10
    implementation      tools/accelerator/hmf.py, tools/future/hmf_objects.py
    unlocks             0 declared dependencies

### U50_34_TO_40
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_40_TO_50
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_50_TO_60
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_60_TO_70
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_70_TO_80
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_80_TO_90
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+8 more)
    implementation      tools/future/hwir.py
    unlocks             1 declared dependencies

### U50_DEVICE_PROFILE
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/accelerator/test_fusion_bridge.py:372 (+22 more)
    implementation      tools/future/hwir.py, tools/future/hwir.py:2328
    unlocks             1 declared dependencies

### U50_DMA_HBM
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/future/test_hbm_doctor.py:8
    implementation      tools/future/hbm_doctor.py
    unlocks             1 declared dependencies

### U50_FIRST_NATIVE_ENGINE
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   tools/future/test_fpga_engines.py:19 (+2 more)
    implementation      tools/future/fpga_engines.py
    unlocks             1 declared dependencies

### U50_MIXED_APPLE_FPGA_GRAPH
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   hcli/test_ane_provider.py:6 (+4 more)
    implementation      hcli/physical_graph.py, tools/odyssey/physical_graph_compiler.py
    unlocks             1 declared dependencies

### U50_PURCHASE_ACCEPTANCE
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   must be written
    implementation      tools/accelerator/hardware_doctor.py, tools/accelerator/hardware_doctor.py:230
    unlocks             1 declared dependencies

### U50_SAFE_COOLING
    missing             silicon absent; wakes on U50_PRESENT
    shortest verifier   must be written
    implementation      tools/accelerator/hardware_doctor.py, tools/accelerator/hardware_doctor.py:313
    unlocks             1 declared dependencies

## UNKNOWN_RESEARCH (1)

### AGENTOS_BEHAVIOR_LAB
    missing             no implementation and no verifier exist yet
    shortest verifier   must be written
    implementation      absent
    unlocks             1 declared dependencies

