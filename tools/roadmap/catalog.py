"""Where the auditor looks. Never a status.

Each probe names code paths, import modules, optional symbols, receipt globs,
hardware wake ids, era/gene, dependencies, and a roadmap acceptance span.
The auditor greps those locations; it does not trust this file for BUILT.
"""
from __future__ import annotations

from typing import Any

# Acceptance spans point at H-ROADMAP.md sections. Numbers are line ranges.
# Appendix O ledger lines are attached by the parser; these are the proof-obligation
# sections, not a second copy of the prose.

IA = "I-A_AGENTOS_HCLI"
IB = "I-B_DOCTOR"
IC = "I-C_GRAVITY_NOETIC"
ID = "I-D_ACCELERATOR"
IE = "I-E_ODYSSEY_I"
IIA = "II-A_ODYSSEY_II"
IIB = "II-B_NOETIC_COMPILER_V1"
IIC = "II-C_PHYSICAL_GRAPH_COMPILER"
IID = "II-D_STATE_TOKENIZER_DECODING"
IIE = "II-E_GREEN_MACHINE"
IIIA = "III-A_ODYSSEY_III"
IIIB = "III-B_LEARNED_PHYSICAL_COMPILER"
IIIC = "III-C_RESIDENT_OPTIMIZER"
IIID = "III-D_BEYOND_DENSE_REPRESENTATION"
IIIE = "III-E_AUTONOMOUS_REPRODUCIBLE_SCIENCE"
IVA = "IV-A_FUSION"
IVB = "IV-B_HMF_HGVAS"
IVC = "IV-C_DGX_SPARK"
IVD = "IV-D_EGPU"
IVE = "IV-E_FUSION_BRIDGE_TOPOLOGY_ASCENSION"
VA = "V-A_PRODUCT_SOVEREIGNTY"
VB = "V-B_DEVELOPER_PLATFORM"
VC = "V-C_CONTINUOUS_VERIFIED_IMPROVEMENT"
VD = "V-D_DOMINANCE_SCOREBOARD"
VE = "V-E_PERPETUAL_HAWKING"


def _p(
    *,
    era: str,
    gene: str | None,
    paths: tuple[str, ...] = (),
    modules: tuple[str, ...] = (),
    symbols: tuple[tuple[str, str], ...] = (),
    receipts: tuple[str, ...] = (),
    hw: str | None = None,
    ext: str | None = None,
    deps: tuple[str, ...] = (),
    acc: tuple[int, int] = (9455, 9528),
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "era": era,
        "gene": gene,
        "code_paths": list(paths),
        "modules": list(modules),
        "symbols": [{"module": m, "symbol": s} for m, s in symbols],
        "receipt_globs": list(receipts),
        "hardware_wake": hw,
        "software_blocker": ext,
        "dependencies": list(deps),
        "acceptance_span": {"start_line": acc[0], "end_line": acc[1]},
        "acceptance": acceptance,
    }


# ---------------------------------------------------------------------------
# Gates (71). Hardware-requiring: 12× U50_* + HMF_DEVICE_VISIBLE_TRUST.
# ---------------------------------------------------------------------------

GATES: dict[str, dict[str, Any]] = {
    "AGENTOS_REPAIR_BOUNDED": _p(
        era="I", gene=IA, acc=(7332, 7358),
        paths=("hcli/scheduler.py", "hcli/workunit.py"),
        modules=("hcli.scheduler", "hcli.workunit"),
        symbols=(("hcli.scheduler", "Scheduler"),),
        receipts=("receipts/headless/HCLI_REPAIR_*.json",),
    ),
    "AGENTOS_RETRY_CLASSIFIED": _p(
        era="I", gene=IA, acc=(7361, 7379),
        deps=("AGENTOS_REPAIR_BOUNDED",),
        paths=("hcli/scheduler.py",),
        modules=("hcli.scheduler",),
        symbols=(("hcli.scheduler", "_record_fingerprint"),),
    ),
    "AGENTOS_CIRCUIT_BREAKER": _p(
        era="I", gene=IA, acc=(7332, 7358),
        deps=("AGENTOS_RETRY_CLASSIFIED",),
        paths=("hcli/scheduler.py",),
        modules=("hcli.scheduler",),
        symbols=(("hcli.scheduler", "NO_PROGRESS"),),
    ),
    "AGENTOS_CANCELLATION": _p(
        era="I", gene=IA, acc=(7332, 7358),
        paths=("hcli/scheduler.py", "hcli/cli.py", "hcli/delegate.py"),
        modules=("hcli.scheduler", "hcli.cli", "hcli.delegate"),
        symbols=(("hcli.delegate", "abort"),),
    ),
    "AGENTOS_ORPHAN_RECONCILIATION": _p(
        era="I", gene=IA, acc=(7332, 7358),
        deps=("AGENTOS_CANCELLATION",),
        paths=("hcli/scheduler.py", "hcli/agentos/resident.py", "hcli/agentos/background.py"),
        modules=("hcli.scheduler", "hcli.agentos.resident", "hcli.agentos.background"),
        symbols=(("hcli.agentos.background", "BackgroundJobStore"),),
    ),
    "AGENTOS_PERSISTENCE_SINGLE_AUTHORITY": _p(
        era="I", gene=IA, acc=(7382, 7417),
        paths=("hcli/mutation.py", "hcli/resources.py", "hcli/persist.py"),
        modules=("hcli.mutation", "hcli.resources", "hcli.persist"),
        symbols=(("hcli.resources", "MutationLock"),),
    ),
    "AGENTOS_CHECKPOINT_ATOMICITY": _p(
        era="I", gene=IA, acc=(7382, 7417),
        deps=("AGENTOS_PERSISTENCE_SINGLE_AUTHORITY",),
        paths=("hcli/agentos/checkpoint.py", "hcli/persist.py"),
        modules=("hcli.agentos.checkpoint", "hcli.persist"),
        symbols=(("hcli.agentos.checkpoint", "write_program_checkpoint"),),
    ),
    "AGENTOS_RESTART_COHERENCE": _p(
        era="I", gene=IA, acc=(7662, 7664),
        deps=("AGENTOS_CHECKPOINT_ATOMICITY",),
        paths=("hcli/agentos/recovery.py", "hcli/agentos/resident.py"),
        modules=("hcli.agentos.recovery", "hcli.agentos.resident"),
        symbols=(("hcli.agentos.recovery", "run_recovery_gate"),),
        receipts=("receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",),
    ),
    "HCLI_CONTEXT_AUTHORITY_UNIFIED": _p(
        era="I", gene=IA, acc=(7642, 7644),
        paths=("hcli/context_budget.py", "hcli/engine.py"),
        modules=("hcli.context_budget", "hcli.engine"),
        symbols=(("hcli.context_budget", "resolve"),),
    ),
    "HCLI_CONTEXT_FOCUSED_WORKUNITS": _p(
        era="I", gene=IA, acc=(7420, 7447),
        deps=("HCLI_CONTEXT_AUTHORITY_UNIFIED",),
        paths=("hcli/goal.py",),
        modules=("hcli.goal",),
        symbols=(("hcli.goal", "compile_worker_context"),),
    ),
    "HCLI_CONTEXT_INVALIDATION": _p(
        era="I", gene=IA, acc=(7420, 7447),
        deps=("HCLI_CONTEXT_FOCUSED_WORKUNITS",),
        paths=("hcli/goal.py", "hcli/context_budget.py"),
        modules=("hcli.goal", "hcli.context_budget"),
    ),
    "HCLI_STATUS_PHYSICAL": _p(
        era="I", gene=IA, acc=(7537, 7575),
        paths=("hcli/commands.py", "hcli/processes.py", "hcli/controller.py"),
        modules=("hcli.commands", "hcli.processes", "hcli.controller"),
        symbols=(("hcli.commands", "format_status"),),
    ),
    "HCLI_MIXED_MAX": _p(
        era="I", gene=IA, acc=(7670, 7672),
        deps=("BACKEND_FAILURE_ISOLATION",),
        paths=("hcli/max_policy.py", "hcli/controller.py"),
        modules=("hcli.max_policy", "hcli.controller"),
        symbols=(("hcli.max_policy", "grok_pool_snapshot"),),
    ),
    "BACKEND_FAILURE_ISOLATION": _p(
        era="I", gene=IA, acc=(7361, 7379),
        paths=("hcli/backends.py", "hcli/providers.py", "hcli/engine.py"),
        modules=("hcli.backends", "hcli.providers", "hcli.engine"),
        symbols=(("hcli.backends", "terminate_pid"),),
    ),
    "HCLI_SELF_SUPPLEMENT": _p(
        era="I", gene=IA, acc=(7674, 7676),
        deps=("AGENTOS_REPAIR_BOUNDED",),
        paths=("hcli/agentos/resident.py",),
        modules=("hcli.agentos.resident",),
        symbols=(("hcli.agentos.resident", "admit_evidence_children"),),
    ),
    "HCLI_SELF_OPTIMIZATION_BOOTSTRAP": _p(
        era="I", gene=IA, acc=(7678, 7680),
        deps=("HCLI_SELF_SUPPLEMENT", "HCLI_STATUS_PHYSICAL", "AGENTOS_RESTART_COHERENCE"),
        paths=("hcli/agentos/autonomy_gate.py", "tools/future/resident_optimizer.py"),
        modules=("hcli.agentos.autonomy_gate", "tools.future.resident_optimizer"),
        symbols=(("hcli.agentos.autonomy_gate", "run_autonomy_gate"),),
    ),
    "VMCP_STATE_LATTICE": _p(
        era="I", gene=IA, acc=(7706, 7738),
        paths=("tools/headless/vmcp_lattice_disposition.py", "hcli/vmcp_adapter.py"),
        modules=("tools.headless.vmcp_lattice_disposition", "hcli.vmcp_adapter"),
        receipts=("receipts/headless/VMCP_LATTICE_DISPOSITION.json",),
    ),
    "VMCP_DEEP_DIGEST": _p(
        era="I", gene=IA, acc=(7706, 7738),
        deps=("VMCP_STATE_LATTICE",),
        paths=("tools/headless/vmcp_lattice_disposition.py",),
        modules=("tools.headless.vmcp_lattice_disposition",),
    ),
    "VMCP_TRUTH_LEDGER": _p(
        era="I", gene=IA, acc=(7706, 7738),
        deps=("VMCP_DEEP_DIGEST",),
        paths=("tools/headless/vmcp_forgery_canary.py",),
        modules=("tools.headless.vmcp_forgery_canary",),
        receipts=("receipts/headless/VMCP_FORGERY_CANARY.json",),
    ),
    "VMCP_RECEIPT_LAW": _p(
        era="I", gene=IA, acc=(7770, 7790),
        paths=("hcli/agentos/vmcp_gate.py",),
        modules=("hcli.agentos.vmcp_gate",),
        symbols=(("hcli.agentos.vmcp_gate", "run_vmcp_gate"),),
        receipts=("receipts/headless/HCLI_AGENTOS_VMCP_GATE.json",),
    ),
    "VMCP_TOOL_DOCTOR": _p(
        era="I", gene=IA, acc=(7741, 7767),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
        receipts=("receipts/headless/VMCP_CAPABILITY_SURFACE.json",),
    ),
    "VMCP_FILE_CLASSIFIER": _p(
        era="I", gene=IA, acc=(7793, 7806),
        deps=("VMCP_TOOL_DOCTOR",),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
    ),
    "VMCP_WEB_CAPTURE": _p(
        era="I", gene=IA, acc=(7827, 7840),
        deps=("VMCP_TOOL_DOCTOR",),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
    ),
    "VMCP_VISUAL_DIFF": _p(
        era="I", gene=IA, acc=(7841, 7854),
        deps=("VMCP_TOOL_DOCTOR",),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
    ),
    "VMCP_SPATIAL_VALIDATE": _p(
        era="I", gene=IA, acc=(7855, 7865),
        deps=("VMCP_TOOL_DOCTOR",),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
    ),
    "VMCP_PTY_CAPTURE": _p(
        era="I", gene=IA, acc=(7866, 7881),
        deps=("VMCP_TOOL_DOCTOR",),
        paths=("tools/headless/vmcp_capability_probe.py",),
        modules=("tools.headless.vmcp_capability_probe",),
    ),
    "AGENTOS_BEHAVIOR_LAB": _p(
        era="I", gene=IA, acc=(7882, 7910),
        deps=("VMCP_PTY_CAPTURE",),
        paths=(),
        modules=(),
    ),
    "VMCP_COMPACT_SURFACE": _p(
        era="I", gene=IA, acc=(7954, 7980),
        paths=("hcli/vmcp/__init__.py", "hcli/vmcp_adapter.py"),
        modules=("hcli.vmcp", "hcli.vmcp_adapter"),
    ),
    "VMCP_AGENTOS_INTEGRATION": _p(
        era="I", gene=IA, acc=(7628, 7630),
        deps=("VMCP_RECEIPT_LAW", "VMCP_COMPACT_SURFACE"),
        paths=("hcli/agentos/vmcp_gate.py", "tools/headless/hcli_vmcp_integration.py"),
        modules=("hcli.agentos.vmcp_gate", "tools.headless.hcli_vmcp_integration"),
        receipts=("receipts/headless/VMCP_AGENTOS_INTEGRATION.json",),
    ),
    "AGENTOS_DETERMINISTIC_OFFLOAD": _p(
        era="I", gene=IA, acc=(7485, 7507),
        paths=("lab/hcli/claude_offload_bench.py", "hcli/delegate.py"),
        modules=("lab.hcli.claude_offload_bench", "hcli.delegate"),
        symbols=(("lab.hcli.claude_offload_bench", "run_bench"),),
    ),
    "MODELLAKE_IDENTITY_RESOLVED": _p(
        era="I", gene=IE, acc=(531, 553),
        paths=("hcli/agentos/modellake_gate.py", "tools/odyssey/modellake.py"),
        modules=("hcli.agentos.modellake_gate", "tools.odyssey.modellake"),
        symbols=(("hcli.agentos.modellake_gate", "run_modellake_census"),),
        receipts=("receipts/headless/MODELLAKE_*.json",),
    ),
    "MODELLAKE_HASH_VERIFIED": _p(
        era="I", gene=IE, acc=(531, 553),
        deps=("MODELLAKE_IDENTITY_RESOLVED",),
        paths=("hcli/agentos/modellake_gate.py", "tools/odyssey/modellake_watch.py"),
        modules=("hcli.agentos.modellake_gate", "tools.odyssey.modellake_watch"),
        symbols=(("tools.odyssey.modellake_watch", "reconcile"),),
    ),
    "MODELLAKE_ATOMIC_PROMOTION": _p(
        era="I", gene=IE, acc=(531, 553),
        deps=("MODELLAKE_HASH_VERIFIED",),
        paths=("tools/odyssey/modellake_promote.py",),
        modules=("tools.odyssey.modellake_promote",),
        symbols=(("tools.odyssey.modellake_promote", "promote"),),
    ),
    "QWEN27_RUNTIME_IDENTITY_FROZEN": _p(
        era="I", gene=ID, acc=(506, 530),
        paths=("hcli/agentos/qwen27_runtime_identity.py",),
        modules=("hcli.agentos.qwen27_runtime_identity",),
        symbols=(("hcli.agentos.qwen27_runtime_identity", "run_runtime_archaeology"),),
        receipts=("receipts/headless/QWEN27_HISTORICAL_RUNTIME_IDENTITY.json",),
    ),
    "QWEN27_PROTECTED_BASELINE": _p(
        era="I", gene=ID, acc=(506, 530),
        deps=("QWEN27_RUNTIME_IDENTITY_FROZEN",),
        paths=("hcli/agentos/protected_accelerator_benchmark.py", "hcli/agentos/accelerator_regression.py"),
        modules=("hcli.agentos.protected_accelerator_benchmark", "hcli.agentos.accelerator_regression"),
        symbols=(("hcli.agentos.protected_accelerator_benchmark", "run_protected_accelerator_benchmark"),),
    ),
    "QWEN27_REGRESSION_EXPLAINED_OR_BOUNDED": _p(
        era="I", gene=ID, acc=(506, 530),
        deps=("QWEN27_PROTECTED_BASELINE",),
        paths=("hcli/agentos/qwen27_mlp_diagnostic.py", "hcli/agentos/accelerator_regression.py"),
        modules=("hcli.agentos.qwen27_mlp_diagnostic", "hcli.agentos.accelerator_regression"),
        symbols=(("hcli.agentos.qwen27_mlp_diagnostic", "run_qwen27_mlp_diagnostic_ab"),),
    ),
    "FLASH_SOURCE_VERIFIED": _p(
        era="I", gene=IC, acc=(478, 505),
        paths=("tools/flash_organ_census.py", "tools/gravity_verify_source.py"),
        modules=("tools.flash_organ_census", "tools.gravity_verify_source"),
        receipts=("receipts/headless/FLASH_ORGAN_CENSUS.json",),
    ),
    "FLASH_FIRST_GRAVITY_ORGAN": _p(
        era="I", gene=IC, acc=(478, 505),
        deps=("FLASH_SOURCE_VERIFIED",),
        paths=("hcli/agentos/flash_science.py",),
        modules=("hcli.agentos.flash_science",),
        symbols=(("hcli.agentos.flash_science", "run_flash_science_gate"),),
    ),
    "FLASH_NATIVE_NF_KERNEL": _p(
        era="I", gene=IC, acc=(478, 505),
        deps=("FLASH_FIRST_GRAVITY_ORGAN",),
        paths=("hcli/agentos/flash_executable.py", "hcli/agentos/flash_graph_component.py"),
        modules=("hcli.agentos.flash_executable", "hcli.agentos.flash_graph_component"),
        symbols=(("hcli.agentos.flash_graph_component", "run_flash_graph_component"),),
    ),
    "FLASH_DENSE_VS_NF_AB": _p(
        era="I", gene=IC, acc=(478, 505),
        deps=("FLASH_NATIVE_NF_KERNEL",),
        paths=("hcli/agentos/flash_router_representation_ab.py", "hcli/agentos/representation_ab.py"),
        modules=("hcli.agentos.flash_router_representation_ab", "hcli.agentos.representation_ab"),
        symbols=(("hcli.agentos.flash_router_representation_ab", "run_flash_router_representation_ab"),),
    ),
    "FLASH_FULL_NOETIC_EXECUTABLE": _p(
        era="I", gene=IC, acc=(574, 594),
        deps=("FLASH_DENSE_VS_NF_AB",),
        paths=("tools/odyssey/noetic_compiler.py", "hcli/agentos/flash_executable.py"),
        modules=("tools.odyssey.noetic_compiler", "hcli.agentos.flash_executable"),
    ),
    "FLASH_COMPLETE_EBPW_LE_1": _p(
        era="I", gene=IC, acc=(478, 505),
        deps=("FLASH_FULL_NOETIC_EXECUTABLE",),
        paths=("tools/future/complete_ebpw.py",),
        modules=("tools.future.complete_ebpw",),
        symbols=(("tools.future.complete_ebpw", "mix_report"),),
        receipts=(
            "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
            "receipts/future/COMPLETE_EBPW.json",
        ),
        acceptance={
            "kind": "numeric",
            "receipt": "receipts/future/COMPLETE_EBPW.json",
            "field": "incumbent.complete_ebpw",
            "op": "<=",
            "threshold": 1,
        },
    ),
    "FLASH_ACCEPTED_TPS_GE_50": _p(
        era="I", gene=IC, acc=(478, 505),
        deps=("FLASH_FULL_NOETIC_EXECUTABLE",),
        paths=("tools/flash_stateful_gate.py",),
        modules=("tools.flash_stateful_gate",),
        receipts=("receipts/headless/FLASH_STATEFUL_TPS_GATE_V14.json",),
    ),
    "FPGA_PREBOARD_SCHEMAS": _p(
        era="I", gene=ID, acc=(9000, 9061),
        paths=("hcli/agentos/fpga_preboard.py",),
        modules=("hcli.agentos.fpga_preboard",),
        symbols=(("hcli.agentos.fpga_preboard", "FPGADeviceGenome"),),
    ),
    "FPGA_HWIR": _p(
        era="I", gene=ID, acc=(9000, 9061),
        deps=("FPGA_PREBOARD_SCHEMAS",),
        paths=("tools/future/hwir.py", "hcli/agentos/fpga_preboard.py"),
        modules=("tools.future.hwir", "hcli.agentos.fpga_preboard"),
        symbols=(("tools.future.hwir", "HwirGraph"),),
    ),
    "FPGA_LINK_SIM": _p(
        era="I", gene=ID, acc=(9000, 9061),
        deps=("FPGA_HWIR",),
        paths=("hcli/agentos/fpga_preboard.py",),
        modules=("hcli.agentos.fpga_preboard",),
        symbols=(("hcli.agentos.fpga_preboard", "TransportLinkSimulator"),),
    ),
    "FPGA_PARTITION_SIM": _p(
        era="I", gene=ID, acc=(9000, 9061),
        deps=("FPGA_LINK_SIM",),
        paths=("hcli/agentos/fpga_preboard.py",),
        modules=("hcli.agentos.fpga_preboard",),
        symbols=(("hcli.agentos.fpga_preboard", "simulate_partition"),),
    ),
    "U50_PURCHASE_ACCEPTANCE": _p(
        era="I", gene=ID, acc=(8888, 8901), hw="U50_PRESENT",
        deps=("FPGA_PARTITION_SIM",),
        paths=("tools/future/hardware_doctor.py",),
        modules=("tools.future.hardware_doctor",),
    ),
    "U50_SAFE_COOLING": _p(
        era="I", gene=ID, acc=(8903, 8906), hw="U50_PRESENT",
        deps=("U50_PURCHASE_ACCEPTANCE",),
        paths=("tools/future/hardware_doctor.py",),
        modules=("tools.future.hardware_doctor",),
    ),
    "U50_DEVICE_PROFILE": _p(
        era="I", gene=ID, acc=(8925, 8940), hw="U50_PRESENT",
        deps=("U50_SAFE_COOLING",),
        paths=("tools/odyssey/device_profiles.py",),
        modules=("tools.odyssey.device_profiles",),
    ),
    "U50_DMA_HBM": _p(
        era="I", gene=ID, acc=(8942, 8959), hw="U50_PRESENT",
        deps=("U50_DEVICE_PROFILE",),
        paths=("tools/future/hbm_doctor.py",),
        modules=("tools.future.hbm_doctor",),
    ),
    "U50_FIRST_NATIVE_ENGINE": _p(
        era="I", gene=ID, acc=(8962, 8976), hw="U50_PRESENT",
        deps=("U50_DMA_HBM",),
        paths=("tools/future/fpga_engines.py",),
        modules=("tools.future.fpga_engines",),
    ),
    "U50_MIXED_APPLE_FPGA_GRAPH": _p(
        era="I", gene=ID, acc=(8984, 8987), hw="U50_PRESENT",
        deps=("U50_FIRST_NATIVE_ENGINE",),
        paths=("hcli/physical_graph.py", "tools/odyssey/physical_graph_compiler.py"),
        modules=("hcli.physical_graph", "tools.odyssey.physical_graph_compiler"),
    ),
    "U50_34_TO_40": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_MIXED_APPLE_FPGA_GRAPH",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "U50_40_TO_50": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_34_TO_40",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "U50_50_TO_60": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_40_TO_50",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "U50_60_TO_70": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_50_TO_60",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "U50_70_TO_80": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_60_TO_70",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "U50_80_TO_90": _p(
        era="I", gene=ID, acc=(8989, 8997), hw="U50_PRESENT",
        deps=("U50_70_TO_80",),
        paths=("tools/future/hwir.py",),
        modules=("tools.future.hwir",),
    ),
    "HMF_DEVICE_VISIBLE_TRUST": _p(
        era="IV", gene=IVB, acc=(2801, 2866), hw="HMF_PRESENT",
        paths=("tools/accelerator/hmf.py", "tools/future/hmf_objects.py"),
        modules=("tools.accelerator.hmf", "tools.future.hmf_objects"),
        receipts=("receipts/future/HMF_MANAGED_OBJECTS.json",),
    ),
    "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE": _p(
        era="IV", gene=IVA, acc=(2867, 2971),
        deps=("HMF_DEVICE_VISIBLE_TRUST",),
        paths=("tools/accelerator/fusion_planner.py", "tools/future/fusion_sim.py"),
        modules=("tools.accelerator.fusion_planner", "tools.future.fusion_sim"),
    ),
    "THEIA_T0_TRAIN_SUBSTRATE": _p(era="bounty", gene=None, acc=(8551, 8562), paths=(), modules=()),
    "THEIA_MICRO": _p(
        era="bounty", gene=None, acc=(2972, 3064), deps=("THEIA_T0_TRAIN_SUBSTRATE",),
        paths=(), modules=(),
    ),
    "THEIA_LAB": _p(
        era="bounty", gene=None, acc=(2972, 3064), deps=("THEIA_MICRO",),
        paths=(), modules=(),
    ),
    "THEIA_WORKER": _p(
        era="bounty", gene=None, acc=(2972, 3064), deps=("THEIA_LAB",),
        paths=(), modules=(),
    ),
    "THEIA_RESEARCH": _p(
        era="bounty", gene=None, acc=(2972, 3064), deps=("THEIA_WORKER",),
        paths=(), modules=(),
    ),
    "THEIA_GRAVITY_EXECUTABLE": _p(
        era="bounty", gene=None, acc=(8455, 8579), deps=("THEIA_RESEARCH",),
        paths=(), modules=(),
    ),
    "THEIA_BOUNTY_GENERALIST_QUALIFIED": _p(
        era="bounty", gene=None, acc=(8581, 8702), deps=("THEIA_GRAVITY_EXECUTABLE",),
        paths=(), modules=(),
    ),
    "ODYSSEY_I_DISCOVERY": _p(
        era="I", gene=IE, acc=(531, 553),
        paths=("tools/odyssey_ctl.py", "tools/odyssey_census.py", "hcli/odyssey.py"),
        modules=("tools.odyssey_ctl", "tools.odyssey_census", "hcli.odyssey"),
        symbols=(("tools.odyssey_ctl", "pick_acquire_candidate"),),
    ),
    "ODYSSEY_II_TRANSFER": _p(
        era="II", gene=IIA, acc=(554, 573),
        deps=("ODYSSEY_I_DISCOVERY",),
        paths=("tools/future/qualification_pipeline.py",),
        modules=("tools.future.qualification_pipeline",),
        symbols=(("tools.future.qualification_pipeline", "load_qualification_queue"),),
    ),
    "ODYSSEY_III_ADVERSARIAL_META_SCIENCE": _p(
        era="III", gene=IIIA, acc=(663, 683),
        deps=("ODYSSEY_II_TRANSFER",),
        paths=("tools/future/repro_science.py", "tools/future/autonomy_scars.py"),
        modules=("tools.future.repro_science", "tools.future.autonomy_scars"),
        symbols=(("tools.future.autonomy_scars", "scars"),),
    ),
}


# Disk-truth modules a prior audit wrongly called absent. Credited here so the
# auditor must find them; none of these may be reported as a missing definition
# when git still tracks the path.
DISK_TRUTH_MODULES: tuple[str, ...] = (
    "hcli/physical_graph.py",
    "tools/odyssey/physical_graph_compiler.py",
    "tools/future/hwir.py",
    "tools/accelerator/machine_genome.py",
    "tools/odyssey/device_profiles.py",
    "tools/future/autonomy_scars.py",
    "tools/future/scar_reevaluator.py",
    "tools/odyssey/pareto_archive.py",
    "tools/accelerator/device_ascension.py",
    "tools/future/tabula.py",
    "tools/future/green_machine.py",
    "tools/accelerator/fusion_planner.py",
    "tools/accelerator/repatriation_audit.py",
    "tools/odyssey/noetic_compiler.py",
    "tools/future/complete_ebpw.py",
    "tools/future/capability_reachability.py",
)


GENES: dict[str, dict[str, Any]] = {
    IA: _p(era="I", gene=IA, acc=(428, 451),
           paths=("hcli/engine.py", "hcli/scheduler.py", "hcli/agentos/__init__.py", "hcli/cli.py"),
           modules=("hcli.engine", "hcli.scheduler", "hcli.agentos", "hcli.cli")),
    IB: _p(era="I", gene=IB, acc=(453, 476),
           paths=("hcli/doctor/__init__.py", "tools/doctor_seal.py", "tools/gravity_doctor_gate.py"),
           modules=("hcli.doctor", "tools.doctor_seal", "tools.gravity_doctor_gate")),
    IC: _p(era="I", gene=IC, acc=(478, 504),
           paths=("tools/gravity_ir.py", "tools/odyssey/noetic_compiler.py", "tools/future/complete_ebpw.py"),
           modules=("tools.gravity_ir", "tools.odyssey.noetic_compiler", "tools.future.complete_ebpw")),
    ID: _p(era="I", gene=ID, acc=(506, 529),
           paths=("hcli/physical_graph.py", "tools/accelerator/machine_genome.py", "hcli/agentos/fpga_preboard.py"),
           modules=("hcli.physical_graph", "tools.accelerator.machine_genome", "hcli.agentos.fpga_preboard")),
    IE: _p(era="I", gene=IE, acc=(531, 552),
           paths=("tools/odyssey_ctl.py", "tools/odyssey_census.py", "tools/odyssey/modellake_watch.py"),
           modules=("tools.odyssey_ctl", "tools.odyssey_census", "tools.odyssey.modellake_watch")),
    IIA: _p(era="II", gene=IIA, acc=(554, 572),
            paths=("tools/future/qualification_pipeline.py",),
            modules=("tools.future.qualification_pipeline",)),
    IIB: _p(era="II", gene=IIB, acc=(574, 593),
            paths=("tools/odyssey/noetic_compiler.py",),
            modules=("tools.odyssey.noetic_compiler",)),
    IIC: _p(era="II", gene=IIC, acc=(595, 617),
            paths=("tools/odyssey/physical_graph_compiler.py", "hcli/physical_graph.py"),
            modules=("tools.odyssey.physical_graph_compiler", "hcli.physical_graph")),
    IID: _p(era="II", gene=IID, acc=(619, 640),
            paths=("tools/kv_residency.py", "hcli/agentos/flash_executable.py"),
            modules=("tools.kv_residency", "hcli.agentos.flash_executable")),
    IIE: _p(era="II", gene=IIE, acc=(642, 661),
            paths=("tools/future/green_machine.py",),
            modules=("tools.future.green_machine",)),
    IIIA: _p(era="III", gene=IIIA, acc=(663, 682),
             paths=("tools/future/repro_science.py", "tools/future/autonomy_scars.py"),
             modules=("tools.future.repro_science", "tools.future.autonomy_scars")),
    IIIB: _p(era="III", gene=IIIB, acc=(684, 703),
             paths=("tools/future/lpc_dataset.py",),
             modules=("tools.future.lpc_dataset",)),
    IIIC: _p(era="III", gene=IIIC, acc=(705, 724),
             paths=("tools/future/resident_optimizer.py",),
             modules=("tools.future.resident_optimizer",)),
    IIID: _p(era="III", gene=IIID, acc=(726, 747),
             paths=("tools/future/tabula.py", "tools/future/abliteration.py"),
             modules=("tools.future.tabula", "tools.future.abliteration")),
    IIIE: _p(era="III", gene=IIIE, acc=(749, 769),
             paths=("tools/future/repro_science.py", "tools/future/scar_reevaluator.py"),
             modules=("tools.future.repro_science", "tools.future.scar_reevaluator")),
    IVA: _p(era="IV", gene=IVA, acc=(771, 793),
            paths=("tools/accelerator/fusion_planner.py", "tools/future/fusion_sim.py"),
            modules=("tools.accelerator.fusion_planner", "tools.future.fusion_sim")),
    IVB: _p(era="IV", gene=IVB, acc=(795, 818), hw="HMF_PRESENT",
            paths=("tools/accelerator/hmf.py", "tools/future/hmf_objects.py"),
            modules=("tools.accelerator.hmf", "tools.future.hmf_objects")),
    IVC: _p(era="IV", gene=IVC, acc=(820, 839), hw="DGX_PRESENT", paths=(), modules=()),
    IVD: _p(era="IV", gene=IVD, acc=(841, 862), hw="EGPU_PRESENT",
            paths=("tools/accelerator/fusion_planner.py",),
            modules=("tools.accelerator.fusion_planner",)),
    IVE: _p(era="IV", gene=IVE, acc=(864, 884),
            paths=("tools/accelerator/device_ascension.py", "tools/future/device_ascension_pipeline.py"),
            modules=("tools.accelerator.device_ascension", "tools.future.device_ascension_pipeline")),
    VA: _p(era="V", gene=VA, acc=(886, 906),
           paths=("tools/future/devplatform.py",),
           modules=("tools.future.devplatform",)),
    VB: _p(era="V", gene=VB, acc=(908, 930),
           paths=("tools/future/devplatform.py",),
           modules=("tools.future.devplatform",)),
    VC: _p(era="V", gene=VC, acc=(932, 951),
           paths=("tools/future/qualification_pipeline.py",),
           modules=("tools.future.qualification_pipeline",)),
    VD: _p(era="V", gene=VD, acc=(953, 974),
           paths=("tools/odyssey/pareto_archive.py", "tools/future/tournament.py"),
           modules=("tools.odyssey.pareto_archive", "tools.future.tournament")),
    VE: _p(era="V", gene=VE, acc=(976, 996), paths=(), modules=()),
}


ABSENT_CLAIMS: tuple[tuple[str, str], ...] = (
    ("theia", "no path whose basename or directory component is theia"),
    ("transport", "no PhysicalGraph transport-edge compiler (hide-acp transport.rs is ACP, not II-C)"),
    ("placement", "no standalone placement compiler (hide-fleet fabric_placement.rs is fleet, not II-C)"),
)


# ---------------------------------------------------------------------------
# Theia model-ladder external blockers.
#
# These seven gates all depend on a TRAINED Theia model, which does not exist
# and cannot be produced in this checkout. That is not the same fact as ABSENT:
# the blocker and the wake condition are both known. The bounty ENGINE that
# surrounds them DOES exist (tools/theia/, 33 tests) — it is the model ladder,
# not the laboratory, that is blocked.
# ---------------------------------------------------------------------------

_THEIA_EXTERNAL_BLOCKERS: dict[str, str] = {
    "THEIA_T0_TRAIN_SUBSTRATE": (
        "Hawking Train T0 substrate is not a live campaign here: no teacher "
        "registry, data lake, trace store, curriculum or checkpoint authority "
        "runs in this checkout. Wake: T0_TEACHER_REGISTRY_LIVE."
    ),
    "THEIA_MICRO": "needs a trained ~1B-3B student. Wake: THEIA_T0_TRAIN_SUBSTRATE=PASS.",
    "THEIA_LAB": "needs a trained ~7B-14B student. Wake: THEIA_MICRO=PASS.",
    "THEIA_WORKER": "needs a trained ~20B-40B student. Wake: THEIA_LAB=PASS.",
    "THEIA_RESEARCH": "needs a trained ~30B-100B+ flagship. Wake: THEIA_WORKER=PASS.",
    "THEIA_GRAVITY_EXECUTABLE": (
        "needs a frozen Theia capability baseline to compress "
        "(training-before-Gravity law, roadmap 19.10). Wake: THEIA_RESEARCH=PASS."
    ),
    "THEIA_BOUNTY_GENERALIST_QUALIFIED": (
        "the bounty ENGINE exists at tools/theia/ but no qualified generalist "
        "MODEL does. Wake: THEIA_RESEARCH=PASS."
    ),
}

for _gate, _blocker in _THEIA_EXTERNAL_BLOCKERS.items():
    GATES[_gate]["software_blocker"] = _blocker


# ---------------------------------------------------------------------------
# VMCP organs: cite what implements the capability, not what probes a vendor.
#
# Six VMCP gates pointed at tools/headless/vmcp_capability_probe.py. That file
# MEASURES the external visionmcp package -- it is a probe of somebody else's
# tool, run rather than called -- so the auditor correctly found no call site and
# correctly read six capabilities as unwired scaffolding. Hawking's own organs
# live in tools/vmcp/ and ARE reachable: tools/future/vmcp.py calls all four, and
# tools/acceptance/vmcp/gates.py calls that in turn.
#
# tools/vmcp/tool_doctor.report() already probes PTY at runtime rather than
# trusting its own E3_CLASSES literal, and on this host it reports PTY capture
# CONNECTED via tools.vmcp.pty_eye.capture. Verified directly: pty_eye.probe()
# returns used_real_pty=True, method=libutil.openpty, blocker=None, and
# capture(argv=["/bin/echo", ...]) returns argv, cwd, pid, exit_code, signal and
# terminal text with CRLF from a genuine PTY. The module docstring's flat claim
# that "this sandbox has been measured to deny the slave (EPERM)" is STALE for
# this execution context -- it is a per-context fact stated as a permanent one.
_VMCP_ORGANS: dict[str, tuple[str, ...]] = {
    "VMCP_TOOL_DOCTOR": ("tools/vmcp/tool_doctor.py",),
    "VMCP_FILE_CLASSIFIER": ("tools/vmcp/file_eye.py",),
    "VMCP_PTY_CAPTURE": ("tools/vmcp/pty_eye.py",),
}

# The gate must NAME the symbol whose call counts, or no call can ever match it.
# VMCP_RECEIPT_LAW carries symbols= and reads BUILT; these three carried none, so
# `runtime_caller` could never populate and they read as unwired scaffolding no
# matter how many production callers existed. tools/future/vmcp.py calls every one
# of these -- pty_capture(...), file_observe(...), doctor_report(...),
# doctor_profile(...) -- through `from X import Y as Z` aliases.
_VMCP_ORGAN_SYMBOLS: dict[str, tuple[tuple[str, str], ...]] = {
    "VMCP_TOOL_DOCTOR": (
        ("tools.vmcp.tool_doctor", "profile"),
        ("tools.vmcp.tool_doctor", "report"),
    ),
    "VMCP_FILE_CLASSIFIER": (("tools.vmcp.file_eye", "observe"),),
    "VMCP_PTY_CAPTURE": (("tools.vmcp.pty_eye", "capture"),),
}

for _gate, _syms in _VMCP_ORGAN_SYMBOLS.items():
    GATES[_gate]["symbols"] = [
        {"module": _m, "symbol": _y} for _m, _y in _syms
    ] + list(GATES[_gate].get("symbols") or [])

for _gate, _paths in _VMCP_ORGANS.items():
    # `code_paths`, NOT `paths`: _p() stores the probe's file list under
    # code_paths, so assigning to `paths` writes a key nothing reads -- a silent
    # no-op that still looks like a successful repoint.
    GATES[_gate]["code_paths"] = list(_paths) + list(GATES[_gate].get("code_paths") or [])
    GATES[_gate]["modules"] = [
        _q.removesuffix(".py").replace("/", ".") for _q in _paths
    ] + list(GATES[_gate].get("modules") or [])

# The three that remain PARKED are blocked on OPTIONAL EXTERNAL PACKAGES, not on
# unwritten Hawking code, and calling them "software connection remaining" would
# have pointed this campaign at work that does not exist. tool_doctor.report()
# names the host for each on this machine.
_VMCP_EXTERNAL_BLOCKERS: dict[str, str] = {
    "VMCP_WEB_CAPTURE": (
        "browser/CDP, HTML/DOM capture and CSS parsing are PARKED on the "
        "visionmcp web extra plus a host Chrome; no Hawking code is missing. "
        "Wake: VISIONMCP_WEB_EXTRA_INSTALLED."
    ),
    "VMCP_VISUAL_DIFF": (
        "visual diff is PARKED on the visionmcp compiler residual. "
        "Wake: VISIONMCP_COMPILER_RESIDUAL_AVAILABLE."
    ),
    "VMCP_SPATIAL_VALIDATE": (
        "OBJ/GLTF parsing, the spatial validator and an independent renderer are "
        "PARKED on the visionmcp 3d extra plus Blender CLI. "
        "Wake: VISIONMCP_3D_EXTRA_AND_BLENDER."
    ),
}

for _gate, _blocker in _VMCP_EXTERNAL_BLOCKERS.items():
    GATES[_gate]["software_blocker"] = _blocker


# Same defect as the VMCP organs above, three more times: the gate named no
# symbol, so no call could match it and it read as unwired scaffolding.
#
# tools/acceptance/vmcp/gates.py:237-241 calls prove_deep_digest,
# prove_truth_ledger, prove_asset_lattice and prove_decode_lattice directly.
# All four live in tools/headless/vmcp_lattice_disposition.py -- including
# prove_truth_ledger, which is why VMCP_TRUTH_LEDGER also gets that module
# rather than only the forgery canary its catalogue row pointed at.
_VMCP_LATTICE_SYMBOLS: dict[str, tuple[tuple[str, str], ...]] = {
    "VMCP_STATE_LATTICE": (
        ("tools.headless.vmcp_lattice_disposition", "prove_asset_lattice"),
        ("tools.headless.vmcp_lattice_disposition", "prove_decode_lattice"),
    ),
    "VMCP_DEEP_DIGEST": (
        ("tools.headless.vmcp_lattice_disposition", "prove_deep_digest"),
    ),
    "VMCP_TRUTH_LEDGER": (
        ("tools.headless.vmcp_lattice_disposition", "prove_truth_ledger"),
    ),
}

for _gate, _syms in _VMCP_LATTICE_SYMBOLS.items():
    GATES[_gate]["symbols"] = [
        {"module": _m, "symbol": _y} for _m, _y in _syms
    ] + list(GATES[_gate].get("symbols") or [])
    _mods = {_m for _m, _ in _syms}
    GATES[_gate]["modules"] = sorted(_mods | set(GATES[_gate].get("modules") or []))
    if "tools/headless/vmcp_lattice_disposition.py" not in (GATES[_gate].get("code_paths") or []):
        GATES[_gate]["code_paths"] = ["tools/headless/vmcp_lattice_disposition.py"] + list(
            GATES[_gate].get("code_paths") or []
        )


# ---------------------------------------------------------------------------
# Native runtime. The graph tracked three Qwen27 gates and nothing about prefill,
# context or state reuse -- which is most of the physical work actually happening.
# Seven capabilities, not dozens: enough to make the runtime frontier visible
# without inventing a noun per experiment.
#
# Paths point at what EXISTS. Where nothing implements a capability the gate will
# read ABSENT, and that is the useful answer rather than a silent omission.
_RUNTIME_GATES: dict[str, dict[str, Any]] = {
    "RUNTIME_NATIVE_PREFILL": dict(
        paths=("hcli/prefill_profile.py", "hcli/hawking_native.py"),
        modules=("hcli.prefill_profile",),
        note="real batched prefill rather than decode applied token by token",
    ),
    "RUNTIME_PREFILL_PHYSICAL_FRONTIER": dict(
        paths=("hcli/prefill_profile.py",),
        modules=("hcli.prefill_profile",),
        note="projection / f32 / full-attention prefill path measured and optimized",
    ),
    "RUNTIME_CONTEXT_NATIVE": dict(
        paths=("hcli/context_budget.py",),
        modules=("hcli.context_budget",),
        note="native 131K/262K admission and accounting; YaRN only after the native path",
    ),
    "RUNTIME_PREFIX_STATE_REUSE": dict(
        paths=("hcli/prefix_probe.py",),
        modules=("hcli.prefix_probe",),
        symbols=(("hcli.prefix_probe", "longest_common_prefix"),),
        note=(
            "exact-token append-only prefix reuse. INSTRUMENTED: prefix_reused_tokens, "
            "prefill_tokens_stepped, longest_common_prefix_tokens, "
            "reason_for_prefix_divergence, and reusable_fraction kept DISTINCT from "
            "realized_reuse_fraction. NOT PROVEN in production: zero receipts across "
            "800 scanned carry prefix_reused_tokens, so realized reuse has never been "
            "demonstrated by a run. Do not claim a speedup from wall clock until a "
            "receipt counter establishes reuse."
        ),
    ),
    "RUNTIME_DELTANET_STATE_REUSE": dict(
        paths=("tools/headless/prefill_kv.py",),
        modules=("tools.headless.prefill_kv",),
        note="recurrent state checkpoint/restore and prefix-state reuse",
    ),
    "RUNTIME_DECODE_PROTECTED": dict(
        paths=("hcli/hawking_native.py",),
        modules=("hcli.hawking_native",),
        note="current protected decode authority",
    ),
    "RUNTIME_COMPLETE_TOKEN_PROFILE": dict(
        paths=("hcli/prefill_profile.py",),
        modules=("hcli.prefill_profile",),
        note="prefill + decode + host + tools + context accounted together, not separately",
    ),
}

for _gate, _spec in _RUNTIME_GATES.items():
    GATES[_gate] = _p(
        era="I", gene=ID_ACCEL if "ID_ACCEL" in dir() else "I-D_ACCELERATOR",
        paths=tuple(_spec["paths"]),
        modules=tuple(_spec["modules"]),
        symbols=tuple(_spec.get("symbols") or ()),
        acc=(478, 505),
    )
    GATES[_gate]["runtime_note"] = _spec["note"]

# The multiplier the operator named: maximize the useful stable physical prefix,
# subject to reasoning quality and context budget. It is a MEASUREMENT programme,
# so it is declared with what it must measure rather than as a boolean.
GATES["STABLE_PREFIX_CONTEXT_ALIGNMENT"] = _p(
    era="I", gene="I-D_ACCELERATOR",
    paths=("hcli/prefix_probe.py",),
    modules=("hcli.prefix_probe",),
    acc=(478, 505),
)
GATES["STABLE_PREFIX_CONTEXT_ALIGNMENT"]["runtime_note"] = (
    "measure previous/current prompt tokens, longest common prefix, realized "
    "reused tokens, prefill tokens stepped and the divergence reason. Objective: "
    "maximize the useful stable physical prefix subject to reasoning quality and "
    "context budget. A wall-clock improvement is NOT evidence of reuse."
)


# ModelLake operational truth, and the post-Odyssey product milestones.
#
# The product work is REAL roadmap surface now rather than a vague future idea,
# but it must not become today's work: reorganizing the repository underneath a
# running science campaign is how a known-good runtime stops being known-good.
# Its wake condition is recorded on the gate so nobody has to remember it.
GATES["MODELLAKE_LIFECYCLE"] = _p(
    era="I", gene="I-E_ODYSSEY_I",
    paths=("tools/future/modellake_lifecycle.py",),
    modules=("tools.future.modellake_lifecycle",),
    symbols=(("tools.future.modellake_lifecycle", "lifecycle"),),
    receipts=("receipts/future/MODELLAKE_LIFECYCLE.json",),
    acc=(478, 505),
)

_PRODUCT_WAKE = (
    "HCLI_OPERATIONAL and ODYSSEY_DETACHED and KNOWN_GOOD_RUNTIME_COMMIT frozen. "
    "Isolated worktree only; merge after parity. Never reorganize underneath an "
    "active Odyssey."
)
for _g in ("HAWKING_PUBLIC_MVP", "REPO_TOPOLOGY_COMPRESSION", "SEMANTIC_COMPRESSION"):
    GATES[_g] = _p(era="V", gene=None, paths=(), modules=(), acc=(478, 505),
                   ext=_PRODUCT_WAKE)


# Same defect once more: the gate named no symbol, so its real callers could not
# be matched. tools/headless/state_gravity.py imports session_state_bytes from
# prefill_kv and calls it three times; that function accounts the recurrent
# DeltaNet state and GQA KV bytes a session must hold, which IS the capability.
# Both spellings: state_gravity.py manipulates sys.path and imports the SIBLING
# name `prefill_kv`, not the dotted `tools.headless.prefill_kv`. Declaring only
# the dotted form matches nothing, which is how a real caller stays invisible.
GATES["RUNTIME_DELTANET_STATE_REUSE"]["symbols"] = [
    {"module": "tools.headless.prefill_kv", "symbol": "session_state_bytes"},
    {"module": "prefill_kv", "symbol": "session_state_bytes"},
] + list(GATES["RUNTIME_DELTANET_STATE_REUSE"].get("symbols") or [])
GATES["RUNTIME_DELTANET_STATE_REUSE"]["modules"] = sorted(
    {"tools.headless.prefill_kv", "prefill_kv"}
    | set(GATES["RUNTIME_DELTANET_STATE_REUSE"].get("modules") or [])
)


# ---------------------------------------------------------------------------
# The declaration sweep. Twelve gates named no symbol, so no call could ever
# match them however many production callers existed -- the same defect that
# hid VMCP's organs, the lattice probes and DeltaNet state reuse.
#
# Declaring a symbol CANNOT fabricate wiring: the auditor still has to find a
# real non-test call of it. A gate with no caller stays SCAFFOLDED. This only
# lets it look, which it previously could not do at all.
#
# Symbols chosen to match each gate's DEFINING PROPERTY, not to be convenient.
# FLASH_ACCEPTED_TPS_GE_50 is deliberately absent: its acceptance span is a
# shared default cited by six gates, so its criterion is undefined, and it is a
# physical throughput claim that must never read BUILT on STATIC evidence.
_DECLARATION_SWEEP: dict[str, tuple[tuple[str, str], ...]] = {
    # Representation of HCLI truth, per roadmap section 10. Declaring where a
    # capability lives is NOT implementing it; that campaign still owns the code.
    "RUNTIME_NATIVE_PREFILL": (("hcli.prefill_profile", "bucket_profile"),),
    "RUNTIME_PREFILL_PHYSICAL_FRONTIER": (("hcli.prefill_profile", "attribute"),),
    "RUNTIME_COMPLETE_TOKEN_PROFILE": (("hcli.prefill_profile", "attribute"),),
    # RUNTIME_CONTEXT_NATIVE is DELIBERATELY ABSENT. Declaring
    # native_profile_limits / per_seq_context made it read WIRED on a caller
    # inside hcli/context_budget.py itself -- the module calling its own helper,
    # which the self-call guard correctly refused. No external caller exists, so
    # the honest status is SCAFFOLDED and it stays there until one does.
    "RUNTIME_DECODE_PROTECTED": (("hcli.hawking_native", "config_for_model_path"),),
    "STABLE_PREFIX_CONTEXT_ALIGNMENT": (
        ("hcli.prefix_probe", "longest_common_prefix"),
        ("hcli.prefix_probe", "divergence_reason"),
    ),
    # NOT run_vmcp_gate: that is VMCP_RECEIPT_LAW's symbol, and declaring it here
    # gave two distinct capabilities a byte-identical caller list -- one call
    # cannot be evidence for two different gates. causality_payload is the
    # integration-specific symbol, called from hcli/agentos/recovery.py:413.
    "VMCP_AGENTOS_INTEGRATION": (("hcli.agentos.vmcp_gate", "causality_payload"),),
    "VMCP_COMPACT_SURFACE": (
        ("hcli.vmcp", "inspect_vmcp"),
        ("hcli.vmcp", "call_vmcp"),
    ),
    "HCLI_CONTEXT_INVALIDATION": (("hcli.goal", "assert_evidence_fresh"),),
    # This lane's own.
    "FLASH_SOURCE_VERIFIED": (("tools.flash_organ_census", "census"),),
    "FLASH_FULL_NOETIC_EXECUTABLE": (
        ("tools.odyssey.noetic_compiler", "chain_status"),
        ("tools.odyssey.noetic_compiler", "family_inventory"),
    ),
}

for _gate, _syms in _DECLARATION_SWEEP.items():
    if _gate not in GATES:
        continue
    GATES[_gate]["symbols"] = [
        {"module": _m, "symbol": _y} for _m, _y in _syms
    ] + list(GATES[_gate].get("symbols") or [])
    GATES[_gate]["modules"] = sorted(
        {_m for _m, _ in _syms}
        # sibling spelling too: sys.path-manipulating modules import the bare name
        | {_m.rsplit(".", 1)[-1] for _m, _ in _syms}
        | set(GATES[_gate].get("modules") or [])
    )


# The FLASH gates' acceptance spans pointed at a SHARED DEFAULT section cited by
# six gates, so their criteria read as undefined and Claude refused to wire them.
# That was the wrong conclusion from the right evidence: the span was wrong, not
# the criterion. Both exist and are quotable.
#
#   1607  FLASH HARD GATE: promotion requires BOTH complete-system EBPW <= 1.00
#         AND accepted capability-preserving TPS >= 50. "These are research
#         targets, not current claims."
#   1640  the promotion ladder, 50/70/90/120 TPS against 1.00/0.85/0.75/0.60 EBPW
#   1610  SOURCE / MANIFEST -> EXACT TENSOR CENSUS -> ORGAN GRAPH
#
# Pointing a gate at the obligation that already governs it is not inventing a
# criterion. FLASH_ACCEPTED_TPS_GE_50 still must never read BUILT without a
# MEASURED TPS: the roadmap says in the same breath that these are targets.
for _gate, _start, _end in (
    ("FLASH_ACCEPTED_TPS_GE_50", 1607, 1645),
    ("FLASH_SOURCE_VERIFIED", 1610, 1634),
    ("FLASH_FULL_NOETIC_EXECUTABLE", 1610, 1634),
):
    GATES[_gate]["acceptance_span"] = {"start_line": _start, "end_line": _end}


# With a real criterion, FLASH_SOURCE_VERIFIED can be declared honestly: its
# obligation IS the tensor census, and tools/flash_organ_census.census performs it.
GATES["FLASH_SOURCE_VERIFIED"]["symbols"] = [
    {"module": "tools.flash_organ_census", "symbol": "census"},
] + list(GATES["FLASH_SOURCE_VERIFIED"].get("symbols") or [])

# FLASH_ACCEPTED_TPS_GE_50 is NOT a software connection and must never be filed
# as one. Its criterion is a MEASURED accepted capability-preserving TPS >= 50,
# and the roadmap says in the same sentence that this is a research target, not a
# current claim. No amount of wiring satisfies it; a protected measurement does.
GATES["FLASH_ACCEPTED_TPS_GE_50"]["software_blocker"] = (
    "requires a MEASURED accepted capability-preserving TPS >= 50 under the "
    "protected window, against complete-system EBPW <= 1.00 (roadmap 13, line "
    "1607). The roadmap calls these research targets, not current claims. This "
    "gate is satisfied by a measurement, never by a call site, and must never "
    "read BUILT on STATIC evidence. Wake: PROTECTED_TPS_CAMPAIGN_MEASURED."
)
