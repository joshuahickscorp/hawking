"""STATUS CAUSALITY — challenge a blocker on what its probe actually established.

One failure mode produced five blockers in a day, each accurate about the
check and wrong about the cause. The law, already recorded as scar
STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM, is:

    A STATUS MAY ASSERT ONLY WHAT ITS ACTUAL PROBE ESTABLISHES.

This module makes that challenge a resident call. It never names the true
world state and it never returns "wrong": it reports whether a causal claim
is entailed by the probe that was run. Absence of a recorded probe is
UNTESTED, not evidence the claim is unjustified.

`challenge()` looks up the historical catalog (scan-time / forensic).
`emit()` is the emit-time entry point a gate calls as it stamps a status:
no disk, no catalog, no subprocess. A gate that does not call it is a
named coverage gap, not silent success.

    python3 tools/future/status_causality.py --build
    python3 tools/future/status_causality.py --challenge BLOCKED_NO_METAL_GPU
    python3 tools/future/status_causality.py --scan
    python3 tools/future/status_causality.py --coverage
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, git, load_json, write_receipt

RECEIPT = "STATUS_CAUSALITY_CHALLENGE.json"
SCHEMA = "hawking.future.status_causality.v1"

SUPPORTED = "SUPPORTED"
OVERREACHING = "OVERREACHING"
UNTESTED = "UNTESTED"
UNDERDETERMINED = "UNDERDETERMINED"
CONTRADICTED = "CONTRADICTED"
VERDICTS = (SUPPORTED, OVERREACHING, UNTESTED)
# Generic (observation, conclusion) checker. UNTESTED is a status-challenge
# verdict meaning "no probe recorded"; UNDERDETERMINED is the claim-checker
# counterpart (probe/observation present but the conclusion is not settled).
CLAIM_CHECK_VERDICTS = (SUPPORTED, OVERREACHING, UNDERDETERMINED, CONTRADICTED)

# The detector identifies unjustified claims. "wrong" would adjudicate the world.
FORBIDDEN_VERDICTS = frozenset({"WRONG", "FALSE", "TRUE", "CORRECT", "INVALID"})
WORLD_STATE_KEYS = frozenset(
    {
        "world_state",
        "true_cause",
        "actual_state",
        "is_wrong",
        "correct_status",
        "gpu_present",
        "model_present",
        "weights_present",
        "specimen_present",
        "true_world",
    }
)

PROBE_PROCESS_ERROR = "process_error"
PROBE_PATH_EXISTENCE = "path_existence"
PROBE_LISTING = "listing_membership"
PROBE_METADATA = "metadata_field"
PROBE_LITERAL = "literal_constant"
PROBE_ENUMERATION = "device_enumeration"
PROBE_HASH = "hash_recompute"
PROBE_RECEIPT_FIELD = "receipt_field"
PROBE_MEASURED_FLAGS = "measured_flags"

CLAIM_HOST_HARDWARE_ABSENCE = "host_hardware_absence"
CLAIM_OBJECT_ABSENCE = "object_absence"
CLAIM_CAPABILITY_ABSENCE = "capability_absence"
CLAIM_FIELD_VALUE = "field_value"
CLAIM_PATH_STATE = "path_state"
CLAIM_LISTING_STATE = "listing_state"
CLAIM_PROCESS_FAILURE = "process_failure"
CLAIM_DIGEST_MATCH = "digest_match"
CLAIM_DEVICE_PRESENT = "device_present"
CLAIM_MEASURED_UNMET = "measured_unmet"
PROBE_ONE_LAYER_ROOFLINE = "one_layer_roofline"
CLAIM_GLOBAL_BINDING = "global_organ_binding"

# The five fields every consequential gate must record at emit time.
FIVE_RECORDED_FIELDS: tuple[str, ...] = (
    "probe_performed",
    "direct_observation",
    "interpretation",
    "confidence",
    "alternatives",
)

# A probe uniquely determines only these claims. Anything broader is overreach.
PROBE_ENTAILS: dict[str, frozenset[str]] = {
    PROBE_PROCESS_ERROR: frozenset({CLAIM_PROCESS_FAILURE}),
    PROBE_PATH_EXISTENCE: frozenset({CLAIM_PATH_STATE}),
    PROBE_LISTING: frozenset({CLAIM_LISTING_STATE}),
    PROBE_METADATA: frozenset({CLAIM_FIELD_VALUE}),
    PROBE_LITERAL: frozenset(),
    PROBE_ENUMERATION: frozenset({CLAIM_DEVICE_PRESENT, CLAIM_PROCESS_FAILURE}),
    PROBE_HASH: frozenset({CLAIM_DIGEST_MATCH}),
    PROBE_RECEIPT_FIELD: frozenset({CLAIM_FIELD_VALUE}),
    PROBE_MEASURED_FLAGS: frozenset({CLAIM_MEASURED_UNMET, CLAIM_FIELD_VALUE}),
    # A one-layer MIXED roofline entails the MIXED field, not ALU_BOUND of all organs.
    PROBE_ONE_LAYER_ROOFLINE: frozenset({CLAIM_FIELD_VALUE}),
}

BROAD_ABSENCE_LABELS: dict[str, str] = {
    "BLOCKED_NO_METAL_GPU": CLAIM_HOST_HARDWARE_ABSENCE,
    "MODEL_MISSING": CLAIM_OBJECT_ABSENCE,
    "SPECIMEN_NOT_PRESENT": CLAIM_OBJECT_ABSENCE,
    "WEIGHTS_NOT_PRESENT": CLAIM_OBJECT_ABSENCE,
    "metadata_only_weights_not_present": CLAIM_OBJECT_ABSENCE,
}

NARROW_LABELS: dict[str, str] = {
    "WHOLE_TREE_VERIFIED": CLAIM_DIGEST_MATCH,
    "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION": CLAIM_FIELD_VALUE,
    "declared_path_absent": CLAIM_PATH_STATE,
    "not_in_specimens_listing": CLAIM_LISTING_STATE,
    "HOST_HAS_METAL_GPU": CLAIM_DEVICE_PRESENT,
    "process_failed_at_prefix_initialization": CLAIM_PROCESS_FAILURE,
    "law_store_records_physical_status": CLAIM_FIELD_VALUE,
    "seven_all_met_is_false": CLAIM_FIELD_VALUE,
    "MIXED": CLAIM_FIELD_VALUE,
    "one_layer_mixed": CLAIM_FIELD_VALUE,
}

# Receipts that carried the motivating statuses, or the well-founded counterparts.
KNOWN_RECEIPT_PATHS: tuple[str, ...] = (
    "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
    "receipts/future/METAL_REACHABILITY.json",
    "receipts/future/SPECIMEN_VERIFICATION.json",
    "receipts/future/ODYSSEY2_LAW_STORE.json",
    "receipts/future/ODYSSEY_LAUNCH_GATE.json",
    "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json",
    "receipts/future/AUTONOMY_SCARS.json",
    "receipts/future/CONTAMINATION_SCIENCE.json",
)

# Already-adjudicated receipts: do not treat their `claim` field as a blocker
# they are asserting. Extract only what they actually observed.
ADJUDICATION_SCHEMAS = frozenset(
    {
        "hawking.future.metal_reachability.v1",
        "hawking.future.status_causality.v1",
        "hawking.future.autonomy_scars.v1",
    }
)

_NO_PROBE = frozenset({"", "unrecorded", "absent", "unknown", "not recorded", "none recorded"})

LAW = (
    "A STATUS MAY ASSERT ONLY WHAT ITS ACTUAL PROBE ESTABLISHES. "
    "STATUS LABELS ARE HYPOTHESES UNTIL THEIR CAUSAL CLAIM IS VERIFIED."
)

# Exact strings the IMPROVEMENT_TRIAL misleading_narrow_probe control records.
# emit() of this pair must OVERREACH, and the two receipts must agree.
ALU_BOUND_MIXED_PROBE = "MLP_ALU_ROOFLINE one representative layer, verdict MIXED"
ALU_BOUND_ALL_ORGANS_CLAIM = "all organs are ALU_BOUND"
IMPROVEMENT_TRIAL_REL = "receipts/future/IMPROVEMENT_TRIAL.json"
IMPROVEMENT_TRIAL_CONTROL = "misleading_narrow_probe"
IMPROVEMENT_TRIAL_ALU_BOUND_DETAIL = (
    "claim='all organs are ALU_BOUND' from probe="
    "'MLP_ALU_ROOFLINE one representative layer, verdict MIXED'"
)

# Shape, not identity. A new status name with the same probe-kind/claim-kind
# pair is the next similar label. Catalog lookup of the original five is not
# this regression.
OVERREACH_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "id": "SHAPE.PROCESS_ERROR_AS_HOST_ABSENCE",
        "historical_id": "HC.BLOCKED_NO_METAL_GPU",
        "probe_kind": PROBE_PROCESS_ERROR,
        "claim_kind": CLAIM_HOST_HARDWARE_ABSENCE,
        "example_status": "BLOCKED_NO_METAL_GPU",
        "why": "a process error string is not a host-hardware census",
        "reemit_status": "GPU_ABSENT_REBRANDED",
        "reemit_probe": "error at dense_source_bf16_prefix_initialization: metal: no Metal-capable GPU",
        "reemit_observation": "failure.error='metal: no Metal-capable GPU'",
        "reemit_interpretation": "this host has no Metal-capable GPU",
    },
    {
        "id": "SHAPE.PATH_EXISTENCE_AS_OBJECT_ABSENCE",
        "historical_id": "HC.MODEL_MISSING",
        "probe_kind": PROBE_PATH_EXISTENCE,
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "example_status": "MODEL_MISSING",
        "why": "Path.exists() on one declared path is not object absence",
        "reemit_status": "WEIGHTS_GONE_REBRANDED",
        "reemit_probe": "Path.exists() on a hardcoded parent directory",
        "reemit_observation": "exists=False",
        "reemit_interpretation": "the model is missing",
    },
    {
        "id": "SHAPE.LISTING_AS_OBJECT_ABSENCE",
        "historical_id": "HC.SPECIMEN_NOT_PRESENT",
        "probe_kind": PROBE_LISTING,
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "example_status": "SPECIMEN_NOT_PRESENT",
        "why": "not in one listing is not specimen absence",
        "reemit_status": "SPECIMEN_GONE_REBRANDED",
        "reemit_probe": "membership in ModelLake specimens/",
        "reemit_observation": "present_in_listing=False",
        "reemit_interpretation": "the specimen is not present",
    },
    {
        "id": "SHAPE.METADATA_AS_OBJECT_ABSENCE",
        "historical_id": "HC.WEIGHTS_NOT_PRESENT",
        "probe_kind": PROBE_METADATA,
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "example_status": "WEIGHTS_NOT_PRESENT",
        "why": "a catalog field is not a measurement of the bytes",
        "reemit_status": "FLASH_ABSENT_REBRANDED",
        "reemit_probe": "read schools.Flash.physical_status",
        "reemit_observation": "metadata_only_weights_not_present",
        "reemit_interpretation": "Flash weights are not present",
    },
    {
        "id": "SHAPE.LITERAL_AS_CAPABILITY_ABSENCE",
        "historical_id": "HC.DOCTOR_GRAVITY_LITERAL",
        "probe_kind": PROBE_LITERAL,
        "claim_kind": CLAIM_CAPABILITY_ABSENCE,
        "example_status": "doctor_callable",
        "why": "a hardcoded False is not a capability measurement",
        "reemit_status": "TOOL_UNCALLABLE_REBRANDED",
        "reemit_probe": "schedule=False, frontier=False, refill=False as literals",
        "reemit_observation": "schedule/frontier/refill are false",
        "reemit_interpretation": "Doctor is not resident-callable",
    },
    {
        "id": "SHAPE.MIXED_LAYER_AS_ALU_BOUND",
        "historical_id": None,
        "live_control": {
            "receipt": IMPROVEMENT_TRIAL_REL,
            "control": IMPROVEMENT_TRIAL_CONTROL,
            "claim": ALU_BOUND_ALL_ORGANS_CLAIM,
            "probe": ALU_BOUND_MIXED_PROBE,
            "detail": IMPROVEMENT_TRIAL_ALU_BOUND_DETAIL,
        },
        "probe_kind": PROBE_ONE_LAYER_ROOFLINE,
        "claim_kind": CLAIM_GLOBAL_BINDING,
        "example_status": "ALU_BOUND",
        "why": "a MIXED one-layer roofline does not entail ALU_BOUND of all organs",
        "reemit_status": "ALU_BOUND",
        "reemit_probe": ALU_BOUND_MIXED_PROBE,
        "reemit_observation": "verdict=MIXED",
        "reemit_interpretation": ALU_BOUND_ALL_ORGANS_CLAIM,
    },
)

# Frozen copy of odyssey_launch.CRITERION_IDS. Coverage must not import that
# module (heavy, and it is allowed to be sparse-missing). A test compares
# when the module is importable.
ODYSSEY_LAUNCH_CRITERIA: tuple[str, ...] = (
    "resident_autonomy_trial_pass",
    "specimen_curriculum_ready",
    "modellake_identity",
    "doctor_callable",
    "gravity_callable",
    "nr_nx_path_callable",
    "evidence_hierarchy",
    "negative_science",
    "workgraphs",
    "self_refill",
    "dirty_measurement",
    "protected_scheduling",
    "transfer_substrate",
    "adversary_substrate",
    "crash_recovery",
    "receipts",
)

SELECTION_RULE = (
    "A module is a consequential gate when its emitted status changes what the "
    "campaign does next: whether Odyssey may launch, whether a commit may land, "
    "whether a resident/native/autonomy/modellake/vmcp/recovery/research "
    "qualification is PASSED, whether a specimen is WHOLE_TREE_VERIFIED, "
    "whether a protected window is RUNNABLE, or whether a scanned status was "
    "treated as a blocker. "
    "Included: (1) every path named in the G007 obligation (odyssey_launch "
    "criteria, integration_gate, resident and native gates under hcli/agentos, "
    "specimen verification); "
    "(2) every hcli/agentos module whose filename matches *gate*.py or that "
    "defines run_*_gate (autonomy, modellake, vmcp, native_mission, recovery, "
    "research); "
    "(3) every receipt already reached by status_causality.scan via "
    "KNOWN_RECEIPT_PATHS, except AUTONOMY_SCARS which is an adjudication "
    "schema not a gate; "
    "(4) qualification_pipeline and protected_scheduler, whose RUNNABLE/"
    "REFUSED/BLOCKED verdicts decide whether physical work proceeds. "
    "Excluded, and not as a trick to inflate coverage: Metal kernel names "
    "containing 'gate', crate tests named *_gate.rs, lab/operators hardware "
    "TPS gates, example benches — those do not decide campaign next-step. "
    "This tuple is the list. The rule above is how it was chosen."
)

# Reproducible inventory. Names are the coverage keys.
CONSEQUENTIAL_GATES: tuple[dict[str, Any], ...] = (
    {
        "name": "odyssey_launch",
        "module": "tools/future/odyssey_launch.py",
        "receipt": "receipts/future/ODYSSEY_LAUNCH_GATE.json",
        "emit_fn": "evaluate_launch_criteria",
        "why": "sixteen launch criteria; LAUNCH vs REFUSED decides whether Odyssey proceeds",
        "selection": "named in G007; scan already reaches ODYSSEY_LAUNCH_GATE",
        "named_statuses": list(ODYSSEY_LAUNCH_CRITERIA),
    },
    {
        "name": "integration_gate",
        "module": "tools/future/integration_gate.py",
        "receipt": "receipts/future/INTEGRATION_GATE.json",
        "emit_fn": "check",
        "why": "GREEN vs RED is the door that refuses a canonical commit",
        "selection": "named in G007",
        "named_statuses": ["GREEN", "RED"],
    },
    {
        "name": "resident_gate",
        "module": "hcli/agentos/resident_gate.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_RESIDENT_GATE.json",
        "emit_fn": "run_resident_gate",
        "why": "PASSED/FAILED live residency proof; campaign next-step depends on it",
        "selection": "named in G007 (resident gates under hcli/agentos)",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "native_gate",
        "module": "hcli/agentos/native_gate.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_NATIVE_GATE.json",
        "emit_fn": "run_native_gate",
        "why": "PASSED/FAILED native reproduction ladder",
        "selection": "named in G007 (native gates under hcli/agentos)",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "native_mission_gate",
        "module": "hcli/agentos/native_mission_gate.py",
        "receipt": "receipts/headless/HCLI_NATIVE_MISSION_GATE.json",
        "emit_fn": "run_native_mission_gate",
        "why": "one live native mission PASS/FAIL; qualification next-step",
        "selection": "hcli/agentos run_*_gate; native family",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "autonomy_gate",
        "module": "hcli/agentos/autonomy_gate.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_AUTONOMY_GATE.json",
        "emit_fn": "run_autonomy_gate",
        "why": "odyssey_launch reads all_requested_stages_passed from this receipt",
        "selection": "hcli/agentos *gate*.py; campaign next-step",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "modellake_gate",
        "module": "hcli/agentos/modellake_gate.py",
        "receipt": "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
        "emit_fn": "run_modellake_census",
        "why": "census PASS/FAIL; launch/curriculum identity depends on the lake",
        "selection": "hcli/agentos *gate*.py; campaign next-step",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "vmcp_gate",
        "module": "hcli/agentos/vmcp_gate.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_VMCP_GATE.json",
        "emit_fn": "run_vmcp_gate",
        "why": "HCLI↔VMCP evidence-boundary PASS/FAIL",
        "selection": "hcli/agentos *gate*.py; campaign next-step",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "recovery_gate",
        "module": "hcli/agentos/recovery.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",
        "emit_fn": "run_recovery_gate",
        "why": "crash-recovery PASS/FAIL; autonomy next-step depends on it",
        "selection": "hcli/agentos run_*_gate; campaign next-step",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "research_gate",
        "module": "hcli/agentos/research.py",
        "receipt": "receipts/headless/HCLI_AGENTOS_RESEARCH_GATE.json",
        "emit_fn": "run_research_gate",
        "why": "research-tool evidence-boundary PASS/FAIL",
        "selection": "hcli/agentos run_*_gate; campaign next-step",
        "named_statuses": ["PASSED", "FAILED"],
    },
    {
        "name": "specimen_verify",
        "module": "tools/future/specimen_verify.py",
        "receipt": "receipts/future/SPECIMEN_VERIFICATION.json",
        "emit_fn": "verify_specimen",
        "why": "WHOLE_TREE_VERIFIED vs PARTIAL/CORRUPT/ABSENT; launch curriculum reads it",
        "selection": "named in G007; scan already reaches SPECIMEN_VERIFICATION",
        "named_statuses": [
            "WHOLE_TREE_VERIFIED",
            "CORRUPT_MISMATCH",
            "INCOMPLETE_TIME_BUDGET",
            "PARTIAL_NO_REMOTE_DIGEST",
            "ABSENT",
        ],
    },
    {
        "name": "metal_reachability",
        "module": "tools/future/metal_reachability.py",
        "receipt": "receipts/future/METAL_REACHABILITY.json",
        "emit_fn": "verdict",
        "why": "HOST_HAS_METAL_GPU vs process-side failure; the falsifier of BLOCKED_NO_METAL_GPU",
        "selection": "scan already reaches METAL_REACHABILITY",
        "named_statuses": ["HOST_HAS_METAL_GPU", "FALSIFIED_AS_A_HOST_PROPERTY", "UNTESTED", "CONFIRMED"],
    },
    {
        "name": "flash_nx_audit",
        "module": "tools/future/flash_nx_audit.py",
        "receipt": "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json",
        "emit_fn": "build",
        "why": "seven_all_met is the completeness door for the Flash NX path",
        "selection": "scan already reaches FLASH_NX_COMPLETENESS_AUDIT",
        "named_statuses": ["seven_all_met_is_false"],
    },
    {
        "name": "odyssey2_law_store",
        "module": "tools/future/odyssey2_law_store.py",
        "receipt": "receipts/future/ODYSSEY2_LAW_STORE.json",
        "emit_fn": "build",
        "why": "schools.Flash.physical_status was treated as WEIGHTS_NOT_PRESENT and blocked launch",
        "selection": "scan already reaches ODYSSEY2_LAW_STORE",
        "named_statuses": ["WEIGHTS_NOT_PRESENT", "metadata_only_weights_not_present"],
    },
    {
        "name": "contamination",
        "module": "tools/future/contamination.py",
        "receipt": "receipts/future/CONTAMINATION_SCIENCE.json",
        "emit_fn": "build",
        "why": "contamination_class HEAVY vs QUIESCENT is what protected_scheduling reads",
        "selection": "scan already reaches CONTAMINATION_SCIENCE",
        "named_statuses": ["HEAVY", "QUIESCENT"],
    },
    {
        "name": "qualification_pipeline",
        "module": "tools/future/qualification_pipeline.py",
        "receipt": "receipts/future/QUALIFICATION_PIPELINE.json",
        "emit_fn": "run_static_preflight",
        "why": "qualification status decides whether a candidate may proceed to a protected bench",
        "selection": "verdict decides whether physical work proceeds",
        "named_statuses": ["READY", "BLOCKED", "STATIC_ONLY"],
    },
    {
        "name": "protected_scheduler",
        "module": "tools/future/protected_scheduler.py",
        "receipt": "receipts/future/PROTECTED_SCHEDULER.json",
        "emit_fn": "decide",
        "why": "RUNNABLE vs REFUSED vs BLOCKED_ON_PROTECTED_WINDOW is the physical-work door",
        "selection": "verdict decides whether physical work proceeds",
        "named_statuses": ["RUNNABLE", "REFUSED", "BLOCKED_ON_PROTECTED_WINDOW"],
    },
    {
        "name": "flash_meta_teacher_capture_boundary",
        "module": None,
        "receipt": "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
        "emit_fn": None,
        "why": "stamped BLOCKED_NO_METAL_GPU and the campaign treated that as a host fact",
        "selection": "scan already reaches FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY",
        "named_statuses": ["BLOCKED_NO_METAL_GPU"],
        "rust_emit_point": "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
        "rust_emit_fn": "write_blocked_capture_boundary",
    },
)

# S015 (2026-08-30) dissolved the Codex sidecar write partition: crates/, hcli/,
# tools/ and receipts/ are writable. mutation_surface.CODEX_OWNED is history,
# not a gate. This lane's write set is the eight hcli/agentos emit points plus
# the coverage consumer. flash_meta_teacher_capture_boundary remains unwired
# because its emit point is Rust, not because of the withdrawn partition.
THIS_LANE_WRITE_SCOPE: tuple[str, ...] = (
    "hcli/agentos/resident_gate.py",
    "hcli/agentos/native_gate.py",
    "hcli/agentos/native_mission_gate.py",
    "hcli/agentos/autonomy_gate.py",
    "hcli/agentos/modellake_gate.py",
    "hcli/agentos/vmcp_gate.py",
    "hcli/agentos/recovery.py",
    "hcli/agentos/research.py",
    "tools/future/status_causality.py",
    "tools/future/test_status_causality_gates.py",
    "receipts/future/STATUS_CAUSALITY_COVERAGE.json",
)

FLASH_META_RUST_EMIT_POINT = (
    "crates/hawking-core/examples/flash_meta_teacher_trace.rs"
)
FLASH_META_WIRING_NOTE = (
    "Rust emit point, not Python and not the dissolved S015 partition. "
    "crates/hawking-core/examples/flash_meta_teacher_trace.rs "
    "(present in receipts/future/patches/dirty-crate-work.patch; not a tracked "
    "HEAD file) writes FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json from "
    "write_blocked_capture_boundary(). That function stamps status "
    "BLOCKED_NO_METAL_GPU on ANY dense_source_bf16_prefix_initialization error "
    "and claims 'this host has no Metal-capable GPU'. Wiring would mean "
    "recording probe_performed, direct_observation, interpretation, confidence, "
    "and alternatives in that Rust function at the moment it writes the "
    "receipt — a Python shim would not be the emit point. The language/runtime "
    "boundary is the named remainder."
)

COVERAGE_RECEIPT = "STATUS_CAUSALITY_COVERAGE.json"
COVERAGE_SCHEMA = "hawking.future.status_causality_coverage.v1"

G007_NAMED_GATES: tuple[str, ...] = (
    "odyssey_launch",
    "integration_gate",
    "resident_gate",
    "native_gate",
    "specimen_verify",
)

ODYSSEY_III_CALL = {
    "inherits": True,
    "caller": "tools.future.odyssey3_adversary.emit_for_law and apply_result",
    "calls": "tools.future.status_causality.emit",
    "arguments": {
        "status": "law['law_id']",
        "probe_performed": (
            "the probe named by law['evidence_refs'] — the check that actually "
            "ran, not the law statement"
        ),
        "direct_observation": (
            "the recorded evidence payload those refs actually contain"
        ),
        "interpretation": "law['statement']",
        "probe_kind": "classified from the evidence, never from law['scope']",
        "source": "receipts/future/ODYSSEY2_LAW_STORE.json::<law_id>",
    },
    "when": (
        "before emit_for_law publishes a law, and before apply_result accepts a "
        "scope label as a causal finding"
    ),
    "rule": (
        "OVERREACHING: do not publish the law at its current scope; the statement "
        "exceeds the probe. UNTESTED: refuse the same way NoAttackError refuses a "
        "law with no attack. SUPPORTED: the statement is what the evidence_refs "
        "probe established."
    ),
    "write_scope_note": (
        "odyssey3_adversary.py is outside this obligation's write set; the call "
        "contract is the inheritance. III calls emit() with the arguments above."
    ),
}

S015_PARTITION_NOTE = (
    "AMENDMENT S015 (2026-08-30): Codex is stood down. The sidecar write "
    "partition is dissolved: crates/, hcli/, tools/ and receipts/ are writable, "
    "and 'is this Codex's surface?' no longer gates work. mutation_surface."
    "CODEX_OWNED remains as history and does not gate this lane. What S015 "
    "does not change: no fake completion, no hardware measurement without "
    "hardware, no self-certification, evidence still decides promotion, and "
    "every gate must still measure what it claims."
)


# ---------------------------------------------------------------------------
# Historical cases. Drawn from the receipts and source that produced them,
# not from a reconstruction of the world. The detector must fire on these
# without being a lookup table: each carries the probe the original check
# actually ran, and the claim that was taken from it.
# ---------------------------------------------------------------------------

HISTORICAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "HC.BLOCKED_NO_METAL_GPU",
        "status": "BLOCKED_NO_METAL_GPU",
        "source": "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
        "recovered_from": [
            "tools/future/autonomy_scars.py::STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
            "tools/future/metal_reachability.py",
            "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
        ],
        "probe_kind": PROBE_PROCESS_ERROR,
        "probe_performed": (
            "flash_meta_teacher_trace stamps this status on ANY "
            "dense_source_bf16_prefix_initialization error; this run's error "
            "string was 'metal: no Metal-capable GPU'"
        ),
        "direct_observation": (
            "failure.stage=dense_source_bf16_prefix_initialization; "
            "failure.error='metal: no Metal-capable GPU'; "
            "teacher_rows_written=0; "
            "claim_boundary asserts 'this host has no Metal-capable GPU'"
        ),
        "interpretation": "this host has no Metal-capable GPU",
        "claim_kind": CLAIM_HOST_HARDWARE_ABSENCE,
        "falsifier": (
            "MTLCreateSystemDefaultDevice() and metal::Device::system_default() "
            "from an ordinary process (tools.future.metal_reachability.probe)"
        ),
        "receipt_excerpt": {
            "schema": "hawking.flash.meta_teacher_trace_boundary.v1",
            "status": "BLOCKED_NO_METAL_GPU",
            "failure": {
                "error": "metal: no Metal-capable GPU",
                "stage": "dense_source_bf16_prefix_initialization",
            },
            "teacher_rows_written": 0,
            "claim_boundary": (
                "The required dense source-BF16 teacher capture could not start "
                "because this host has no Metal-capable GPU."
            ),
        },
    },
    {
        "id": "HC.MODEL_MISSING",
        "status": "MODEL_MISSING",
        "source": "tools/odyssey/doctor_tournament.py declared Path (via tools/future/odyssey_launch.py)",
        "recovered_from": [
            "tools/future/autonomy_scars.py::SISTER_SYMPTOMS",
            "tools/future/odyssey_launch.py::_resolve_stale_input",
            "tools/future/external_specimen_seal.py",
        ],
        "probe_kind": PROBE_PATH_EXISTENCE,
        "probe_performed": (
            "Path.exists() on the hardcoded parent "
            "/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"
        ),
        "direct_observation": (
            "declared path is absent; the same directory name is present at "
            "/Volumes/corpdrive/personalmodel/correspondent/qwen3.8-27b-abliterated-bf16"
        ),
        "interpretation": "the model is missing",
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "falsifier": (
            "look for the same directory name under known model roots "
            "(tools.future.odyssey_launch._resolve_stale_input)"
        ),
        "receipt_excerpt": {
            "status": "MODEL_MISSING",
            "probe": {
                "kind": "path_existence",
                "path": "/Users/scammermike/models/qwen3.8-27b-abliterated-bf16",
                "exists": False,
            },
            "claim_boundary": "the model is missing",
        },
    },
    {
        "id": "HC.SPECIMEN_NOT_PRESENT",
        "status": "SPECIMEN_NOT_PRESENT",
        "source": "tools/future/odyssey_launch.py::_ready (specimens listing)",
        "recovered_from": [
            "tools/future/odyssey_launch.py::propose_specimen_curriculum",
            "tools/future/specimen_verify.py::EXTRA_SPECIMENS",
            "receipts/future/SPECIMEN_VERIFICATION.json",
        ],
        "probe_kind": PROBE_LISTING,
        "probe_performed": (
            "membership of Qwen/Qwen3-0.6B in the ModelLake specimens/ listing"
        ),
        "direct_observation": (
            "not in specimens/; complete specimen sits at "
            "/Volumes/corpdrive/hawking-modellake/partial/Qwen--Qwen3-0.6B@c1899de289a0"
        ),
        "interpretation": "the specimen is not present",
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "falsifier": (
            "look under modellake/partial/ and the lake root, not only specimens/"
        ),
        "receipt_excerpt": {
            "status": "SPECIMEN_NOT_PRESENT",
            "probe": {
                "kind": "listing_membership",
                "listing": "ModelLake specimens/",
                "name": "Qwen--Qwen3-0.6B@c1899de289a0",
                "present_in_listing": False,
            },
            "in_specimens_listing": False,
            "identity_known": True,
            "claim_boundary": (
                "identity known but specimen is not in the ModelLake specimens listing"
            ),
        },
    },
    {
        "id": "HC.WEIGHTS_NOT_PRESENT",
        "status": "WEIGHTS_NOT_PRESENT",
        "source": "receipts/future/ODYSSEY2_LAW_STORE.json",
        "recovered_from": [
            "tools/future/odyssey2_law_store.py::SCHOOLS['Flash']['physical_status']",
            "tools/future/odyssey_launch.py::_ready",
            "receipts/future/SPECIMEN_VERIFICATION.json",
        ],
        "probe_kind": PROBE_METADATA,
        "probe_performed": (
            "read schools.Flash.physical_status from the Odyssey II law store"
        ),
        "direct_observation": "metadata_only_weights_not_present",
        "interpretation": "Flash weights are not present",
        "claim_kind": CLAIM_OBJECT_ABSENCE,
        "falsifier": (
            "recompute every published digest for "
            "Qwen--Qwen3.8-Flash-Next@34567a4712bc (tools.future.specimen_verify)"
        ),
        "receipt_excerpt": {
            "schema": "hawking.future.odyssey2_law_store.v1",
            "schools": {
                "Flash": {
                    "school": "Flash",
                    "source_model": "Qwen/Qwen3.8-Flash-Next",
                    "physical_status": "metadata_only_weights_not_present",
                }
            },
        },
    },
    {
        "id": "HC.DOCTOR_GRAVITY_LITERAL",
        "status": "doctor_callable",
        "source": "tools/future/odyssey_launch.py::_eval_callable_tool (historical)",
        "recovered_from": [
            "tools/future/odyssey_launch.py::_eval_callable_tool at 9d12ebf12",
            "tools/future/test_odyssey_launch.py::test_protected_scheduling_is_measured_not_a_constant",
        ],
        "probe_kind": PROBE_LITERAL,
        "probe_performed": (
            "operational_bar(schedule=False, frontier=False, refill=False) as "
            "literals, so doctor_callable and gravity_callable could not pass "
            "on any machine"
        ),
        "direct_observation": (
            "schedule=False, frontier=False, refill=False; reason: "
            "'schedule/frontier/refill are false. A CLI a human can run is not enough.'"
        ),
        "interpretation": (
            "Doctor/Gravity is not resident-callable (a capability absence)"
        ),
        "claim_kind": CLAIM_CAPABILITY_ABSENCE,
        "falsifier": (
            "measure schedule/frontier/refill against the orchestration connector "
            "and the declared parent path, instead of asserting them"
        ),
        "receipt_excerpt": {
            "schema": "hawking.future.odyssey_launch.v1",
            "criteria": [
                {
                    "id": "doctor_callable",
                    "met": False,
                    "reason": (
                        "Doctor is recovered as a human-callable tool and/or prior "
                        "Odyssey I seals but is not resident-operational: "
                        "schedule/frontier/refill are false. "
                        "A CLI a human can run is not enough."
                    ),
                    "operational": {
                        "flags": {
                            "discover": True,
                            "invoke": True,
                            "schedule": False,
                            "verify": True,
                            "frontier": False,
                            "persist": True,
                            "refill": False,
                        }
                    },
                },
                {
                    "id": "gravity_callable",
                    "met": False,
                    "reason": (
                        "Gravity is recovered as a human-callable tool and/or prior "
                        "Odyssey I seals but is not resident-operational: "
                        "schedule/frontier/refill are false. "
                        "A CLI a human can run is not enough."
                    ),
                    "operational": {
                        "flags": {
                            "discover": True,
                            "invoke": True,
                            "schedule": False,
                            "verify": True,
                            "frontier": False,
                            "persist": True,
                            "refill": False,
                        }
                    },
                },
            ],
        },
    },
)


# Well-founded counterparts, also drawn from real receipts. A detector that
# cannot return SUPPORTED on these will cry wolf and be ignored — which has
# already happened once in this partition with a regex-based attacker.
SUPPORTED_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "SF.FLASH_WHOLE_TREE",
        "status": "WHOLE_TREE_VERIFIED",
        "source": "receipts/future/SPECIMEN_VERIFICATION.json",
        "probe_kind": PROBE_HASH,
        "probe_performed": (
            "recompute every published HuggingFace digest for "
            "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
        ),
        "direct_observation": (
            "specimen=Qwen--Qwen3.8-Flash-Next@34567a4712bc; "
            "n_files=144; verified=144; mismatched=0; no_remote_digest=0; "
            "bytes_hashed=360023286454"
        ),
        "interpretation": (
            "this specimen's published digests match the hashes recomputed here"
        ),
        "claim_kind": CLAIM_DIGEST_MATCH,
        "falsifier": "recompute one file and observe a mismatch",
        "receipt_excerpt": {
            "schema": "hawking.future.specimen_verify.v1",
            "results": [
                {
                    "specimen": "Qwen--Qwen3.8-Flash-Next@34567a4712bc",
                    "status": "WHOLE_TREE_VERIFIED",
                    "n_files": 144,
                    "verified": 144,
                    "mismatched": 0,
                    "no_remote_digest": 0,
                    "bytes_hashed": 360023286454,
                    "whole_tree_verified": True,
                }
            ],
        },
    },
    {
        "id": "SF.METAL_DEVICE_PRESENT",
        "status": "HOST_HAS_METAL_GPU",
        "source": "receipts/future/METAL_REACHABILITY.json",
        "probe_kind": PROBE_ENUMERATION,
        "probe_performed": (
            "MTLCreateSystemDefaultDevice() and MTLCopyAllDevices() from an "
            "ordinary command-line process; metal crate Device::system_default() "
            "with the version Cargo.lock resolves"
        ),
        "direct_observation": "system_default='Apple M3 Ultra'; n_devices=1",
        "interpretation": (
            "this process saw a Metal device, so the host has a Metal-capable GPU"
        ),
        "claim_kind": CLAIM_DEVICE_PRESENT,
        "falsifier": "the same enumeration returning no device",
        "receipt_excerpt": {
            "schema": "hawking.future.metal_reachability.v1",
            "observed": {
                "system_default": "Apple M3 Ultra",
                "n_devices": 1,
                "devices": ["Apple M3 Ultra"],
            },
        },
    },
    {
        "id": "SF.NX_SEVEN_ALL_MET_FIELD",
        "status": "seven_all_met_is_false",
        "source": "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json",
        "probe_kind": PROBE_RECEIPT_FIELD,
        "probe_performed": (
            "read seven_all_met from receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
        ),
        "direct_observation": "seven_all_met=False",
        "interpretation": (
            "the completeness audit currently records seven_all_met=False"
        ),
        "claim_kind": CLAIM_FIELD_VALUE,
        "falsifier": "the same field reading True",
        "receipt_excerpt": {
            "schema": "hawking.future.flash_nx_audit.v1",
            "seven_all_met": False,
            "status": "seven_all_met_is_false",
        },
    },
)


# ---------------------------------------------------------------------------
# Loaders. Sparse checkout: missing here is not project absence.
# ---------------------------------------------------------------------------


def _load_receipt(rel: str) -> dict[str, Any] | None:
    """Load JSON from the worktree, then git HEAD. None is a refusal, not empty success."""
    rel = rel.replace("\\", "/").lstrip("./")
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path)
        except (OSError, json.JSONDecodeError):
            return None
    alt = RECEIPTS / Path(rel).name
    if alt.is_file():
        try:
            return load_json(alt)
        except (OSError, json.JSONDecodeError):
            return None
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None
    return None


def historical_cases() -> list[dict[str, Any]]:
    return [dict(c) for c in HISTORICAL_CASES]


def supported_fixtures() -> list[dict[str, Any]]:
    return [dict(c) for c in SUPPORTED_FIXTURES]


# ---------------------------------------------------------------------------
# Classification. Unknown claim kinds stay unknown: accusing a status we
# could not classify is how a regex attacker cried wolf in this partition.
# ---------------------------------------------------------------------------


def _claim_kind_of(row: Mapping[str, Any]) -> str | None:
    if row.get("claim_kind"):
        return str(row["claim_kind"])
    status = str(row.get("status") or "").strip()
    if status in NARROW_LABELS:
        return NARROW_LABELS[status]
    if status in BROAD_ABSENCE_LABELS:
        return BROAD_ABSENCE_LABELS[status]
    interp = str(row.get("interpretation") or row.get("claim_boundary") or "")
    lowered = interp.lower()
    if "this host has no metal" in lowered or "no metal-capable gpu" in lowered:
        return CLAIM_HOST_HARDWARE_ABSENCE
    if "the model is missing" in lowered or "model missing" in lowered:
        return CLAIM_OBJECT_ABSENCE
    if "specimen is not present" in lowered or "weights are not present" in lowered:
        return CLAIM_OBJECT_ABSENCE
    if "not resident-callable" in lowered and row.get("probe_kind") == PROBE_LITERAL:
        return CLAIM_CAPABILITY_ABSENCE
    if _is_broad_alu_bound_claim(row):
        return CLAIM_GLOBAL_BINDING
    return None


def _blob(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def _is_mixed_one_layer_probe(row: Mapping[str, Any]) -> bool:
    probe = str(row.get("probe_kind") or "")
    text = _blob(probe, row.get("probe_performed"), row.get("direct_observation"))
    one_layer = (
        probe == PROBE_ONE_LAYER_ROOFLINE
        or "one representative layer" in text
        or "one layer" in text
    )
    return one_layer and "mixed" in text


def _is_broad_alu_bound_claim(row: Mapping[str, Any]) -> bool:
    if row.get("claim_kind") == CLAIM_GLOBAL_BINDING:
        return True
    text = _blob(row.get("status"), row.get("interpretation"), row.get("claim_boundary"))
    if "alu_bound" in text and ("all organs" in text or "every organ" in text):
        return True
    if str(row.get("status") or "") == "ALU_BOUND" and _is_mixed_one_layer_probe(row):
        return True
    return False


def _infer_probe_kind(row: Mapping[str, Any]) -> str:
    existing = str(row.get("probe_kind") or "").strip()
    if existing and existing.lower() not in _NO_PROBE:
        return existing
    text = _blob(row.get("probe_performed"), row.get("direct_observation"))
    if not text.strip():
        return ""
    if ("mlp_alu_roofline" in text or "roofline" in text) and (
        "one representative layer" in text or "one layer" in text
    ):
        return PROBE_ONE_LAYER_ROOFLINE
    return ""


def _has_probe(row: Mapping[str, Any]) -> bool:
    kind = str(row.get("probe_kind") or "").strip()
    performed = row.get("probe_performed")
    if isinstance(performed, str):
        performed_s = performed.strip().lower()
    elif performed is None:
        performed_s = ""
    else:
        performed_s = str(performed).strip().lower()
    if kind in _NO_PROBE and performed_s in _NO_PROBE:
        return False
    if kind and kind not in _NO_PROBE:
        return True
    return performed_s not in _NO_PROBE


def _observation_shows_device(observation: Any) -> bool:
    if isinstance(observation, Mapping):
        device = observation.get("system_default") or observation.get("device")
        n = observation.get("n_devices")
        return bool(device) or (isinstance(n, int) and n > 0)
    text = str(observation or "")
    return "Apple M3" in text or "system_default=" in text and "NONE" not in text


def _hash_matched(observation: Any) -> bool:
    if isinstance(observation, Mapping):
        n = observation.get("n_files")
        v = observation.get("verified")
        mismatched = observation.get("mismatched") or 0
        no_digest = observation.get("no_remote_digest") or 0
        hashed = observation.get("bytes_hashed") or 0
        return (
            isinstance(n, int)
            and isinstance(v, int)
            and n > 0
            and v == n
            and not mismatched
            and not no_digest
            and isinstance(hashed, int)
            and hashed > 0
        )
    text = str(observation or "")
    return "verified=" in text and "mismatched=0" in text and "n_files=" in text


def _entailed(probe_kind: str, claim_kind: str, observation: Any) -> bool:
    allowed = PROBE_ENTAILS.get(probe_kind)
    if allowed is None:
        return False
    if claim_kind not in allowed:
        return False
    if probe_kind == PROBE_ENUMERATION and claim_kind == CLAIM_DEVICE_PRESENT:
        return _observation_shows_device(observation)
    if probe_kind == PROBE_ENUMERATION and claim_kind == CLAIM_PROCESS_FAILURE:
        return not _observation_shows_device(observation)
    if probe_kind == PROBE_HASH and claim_kind == CLAIM_DIGEST_MATCH:
        return _hash_matched(observation) or _hash_matched_text(observation)
    return True


def _hash_matched_text(observation: Any) -> bool:
    """Direct-observation strings from the Flash fixture use n_files=N; verified=N."""
    text = str(observation or "")
    if "mismatched=0" not in text or "no_remote_digest=0" not in text:
        return False
    n = _after(text, "n_files=")
    v = _after(text, "verified=")
    hashed = _after(text, "bytes_hashed=")
    return bool(n) and n == v and hashed not in {"", "0"}


def _after(text: str, key: str) -> str:
    if key not in text:
        return ""
    rest = text.split(key, 1)[1]
    token = rest.split(";", 1)[0].split(",", 1)[0].strip()
    return token


def _alternatives(
    probe_kind: str, claim_kind: str, observation: Any
) -> list[dict[str, Any]]:
    """World-states consistent with the observation. Hypotheticals, not findings."""
    if probe_kind == PROBE_PROCESS_ERROR and claim_kind == CLAIM_HOST_HARDWARE_ABSENCE:
        return [
            _alt("this process cannot see a present GPU (sandbox, launch, slice)", True, False),
            _alt("the error string names Metal but the failure is elsewhere in prefix init", True, False),
            _alt("the host has no Metal-capable GPU", True, True),
        ]
    if probe_kind == PROBE_PATH_EXISTENCE and claim_kind == CLAIM_OBJECT_ABSENCE:
        return [
            _alt("the directory moved to another volume under the same name", True, False),
            _alt("the hardcoded path is stale; the object is present elsewhere", True, False),
            _alt("the object was never on this host", True, True),
        ]
    if probe_kind == PROBE_LISTING and claim_kind == CLAIM_OBJECT_ABSENCE:
        return [
            _alt("the specimen lives under partial/ (or another lake root), complete", True, False),
            _alt("the census is stale and the directory is in the listing on disk", True, False),
            _alt("the specimen is not on this host at all", True, True),
        ]
    if probe_kind == PROBE_METADATA and claim_kind == CLAIM_OBJECT_ABSENCE:
        return [
            _alt("the field was true when written and the bytes have since landed", True, False),
            _alt("the field is a catalog declaration, never a measurement of the bytes", True, False),
            _alt("the weights are in fact absent", True, True),
        ]
    if probe_kind == PROBE_LITERAL and claim_kind == CLAIM_CAPABILITY_ABSENCE:
        return [
            _alt("the tool is callable; the constant hid it on every machine", True, False),
            _alt("the tool is not callable for a reason the constant does not name", True, False),
            _alt("the tool is not callable", True, True),
        ]
    if probe_kind == PROBE_PROCESS_ERROR and claim_kind == CLAIM_PROCESS_FAILURE:
        return [_alt("the process failed at the named stage with the named error", True, True)]
    if probe_kind == PROBE_PATH_EXISTENCE and claim_kind == CLAIM_PATH_STATE:
        return [_alt("the declared path is absent at that exact location", True, True)]
    if probe_kind == PROBE_LISTING and claim_kind == CLAIM_LISTING_STATE:
        return [_alt("the name is not in that listing", True, True)]
    if probe_kind == PROBE_METADATA and claim_kind == CLAIM_FIELD_VALUE:
        return [_alt("the stored field currently has this value", True, True)]
    if probe_kind == PROBE_RECEIPT_FIELD and claim_kind == CLAIM_FIELD_VALUE:
        return [_alt("the receipt field currently has this value", True, True)]
    if probe_kind == PROBE_HASH and claim_kind == CLAIM_DIGEST_MATCH:
        if _entailed(probe_kind, claim_kind, observation):
            return [_alt("every published digest matched the hash recomputed here", True, True)]
        return [
            _alt("some files were not hashed or did not match", True, False),
            _alt("the tree is fully verified", True, True),
        ]
    if probe_kind == PROBE_ENUMERATION and claim_kind == CLAIM_DEVICE_PRESENT:
        if _observation_shows_device(observation):
            return [_alt("this process saw a Metal device, so the host has one", True, True)]
        return [_alt("this process saw no device; the host may still have one", True, False)]
    if probe_kind == PROBE_ONE_LAYER_ROOFLINE and claim_kind == CLAIM_GLOBAL_BINDING:
        return [
            _alt("the measured layer is MIXED; other organs or layers are not ALU_BOUND", True, False),
            _alt("one layer is not a sample of all organs", True, False),
            _alt("all organs are in fact ALU_BOUND", True, True),
        ]
    if probe_kind == PROBE_ONE_LAYER_ROOFLINE and claim_kind == CLAIM_FIELD_VALUE:
        return [_alt("the named layer's roofline verdict is this field value", True, True)]
    if probe_kind == PROBE_MEASURED_FLAGS and claim_kind in {
        CLAIM_MEASURED_UNMET,
        CLAIM_FIELD_VALUE,
    }:
        return [_alt("the named flags were measured and have these values", True, True)]
    # Probe and claim are classified but we have no specific pair: the
    # entailment table decides, and one hypothetical keeps the negative visible.
    if not _entailed(probe_kind, claim_kind, observation):
        return [
            _alt("a world in which the observation holds and the claim does not", True, False),
            _alt("a world in which both hold", True, True),
        ]
    return [_alt("the observation is inconsistent with the claim being false", True, True)]


def _alt(world: str, with_obs: bool, with_claim: bool) -> dict[str, Any]:
    return {
        "hypothetical": world,
        "consistent_with_observation": with_obs,
        "consistent_with_claim": with_claim,
    }


def _default_falsifier(probe_kind: str, claim_kind: str, row: Mapping[str, Any]) -> str:
    if row.get("falsifier"):
        return str(row["falsifier"])
    if claim_kind == CLAIM_HOST_HARDWARE_ABSENCE:
        return "Metal device enumeration from an ordinary process (metal_reachability.probe)"
    if probe_kind == PROBE_PATH_EXISTENCE:
        return "resolve the same directory name under known model roots"
    if probe_kind == PROBE_LISTING:
        return "look under partial/ and the lake root, not only the listing that was checked"
    if probe_kind == PROBE_METADATA and claim_kind == CLAIM_OBJECT_ABSENCE:
        return "recompute published digests of the named specimen"
    if probe_kind == PROBE_LITERAL:
        return "measure the flags against the connector that would have to be true for the claim"
    if probe_kind == PROBE_ENUMERATION:
        return "repeat the enumeration; a device appearing or disappearing settles presence"
    if probe_kind == PROBE_HASH:
        return "recompute one file and compare"
    if probe_kind == PROBE_ONE_LAYER_ROOFLINE or claim_kind == CLAIM_GLOBAL_BINDING:
        return (
            "run the matched-pair roofline on every named organ; a MIXED layer "
            "does not entail ALU_BOUND of all organs "
            "(IMPROVEMENT_TRIAL negative control misleading_narrow_probe)"
        )
    return "an independent probe of the same object the status names, not a reread of the status"


def _confidence(verdict: str, falsifier: str) -> dict[str, str]:
    if verdict == UNTESTED:
        return {
            "level": "NONE",
            "about": "whether the probe entails the claim; no probe was recorded",
            "would_raise": "record the actual probe that produced the status",
            "would_lower": "not applicable; there is no claim-to-probe link to weaken",
        }
    if verdict == OVERREACHING:
        return {
            "level": "LOW",
            "about": "whether the recorded probe entails the causal claim",
            "would_raise": falsifier,
            "would_lower": "any additional world-state consistent with the same observation",
        }
    return {
        "level": "HIGH",
        "about": (
            "whether the recorded probe entails the claim — not whether the "
            "world is that way for reasons this probe did not test"
        ),
        "would_raise": "independent replication of the same probe",
        "would_lower": "a demonstration that the observation is consistent with the claim being false",
    }


def _verdict_from_parts(row: Mapping[str, Any]) -> str:
    if not _has_probe(row):
        return UNTESTED
    claim_kind = row.get("claim_kind")
    if not claim_kind:
        return UNTESTED
    alts = list(row.get("alternatives") or [])
    if any(
        a.get("consistent_with_observation") and not a.get("consistent_with_claim")
        for a in alts
        if isinstance(a, Mapping)
    ):
        return OVERREACHING
    probe_kind = str(row.get("probe_kind") or "")
    if not _entailed(probe_kind, str(claim_kind), row.get("direct_observation")):
        return OVERREACHING
    return SUPPORTED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def challenge(status: str | Mapping[str, Any]) -> dict[str, Any]:
    """For a consequential blocker, separate the probe from the causal claim.

    Accepts a status string (looked up in the historical catalog, otherwise
    UNTESTED) or a mapping that already names the probe. Never asserts what
    the world is.
    """
    row = _normalize(status)
    for key in WORLD_STATE_KEYS:
        row.pop(key, None)
    inferred = _infer_probe_kind(row)
    if inferred:
        row["probe_kind"] = inferred
    claim_kind = _claim_kind_of(row)
    if claim_kind:
        row["claim_kind"] = claim_kind
    if not row.get("interpretation"):
        row["interpretation"] = str(row.get("status") or "")
    if not _has_probe(row) or not claim_kind:
        out = _challenge_record(row, UNTESTED, [], _default_falsifier("", "", row))
        return out
    alts = _alternatives(
        str(row.get("probe_kind") or ""),
        str(claim_kind),
        row.get("direct_observation"),
    )
    row["alternatives"] = alts
    v = _verdict_from_parts(row)
    falsifier = _default_falsifier(str(row.get("probe_kind") or ""), str(claim_kind), row)
    return _challenge_record(row, v, alts, falsifier)


def _challenge_record(
    row: Mapping[str, Any],
    verdict: str,
    alternatives: Sequence[Mapping[str, Any]],
    falsifier: str,
) -> dict[str, Any]:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not in {VERDICTS}")
    if verdict in FORBIDDEN_VERDICTS:
        raise ValueError(f"the routine does not adjudicate the world: {verdict!r}")
    out = {
        "status": row.get("status"),
        "id": row.get("id"),
        "source": row.get("source"),
        "probe_performed": row.get("probe_performed") or "",
        "direct_observation": row.get("direct_observation") or "",
        "interpretation": row.get("interpretation") or "",
        "probe_kind": row.get("probe_kind") or "",
        "claim_kind": row.get("claim_kind"),
        "confidence": _confidence(verdict, falsifier),
        "alternatives": [dict(a) for a in alternatives],
        "falsifier": falsifier,
        "verdict": verdict,
        "law": LAW,
    }
    if row.get("recovered_from"):
        out["recovered_from"] = list(row["recovered_from"])
    for key in WORLD_STATE_KEYS:
        if key in out:
            raise RuntimeError(f"challenge leaked a world-state key: {key}")
    return out


def emit(
    status: str,
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str = "",
    probe_kind: str = "",
    claim_kind: str | None = None,
    falsifier: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Emit-time challenge. No disk, no catalog, no subprocess.

    A gate calls this as it stamps a status. Passing only a status name
    (no probe) is UNTESTED: the catalog of historical overreaches is not
    a substitute for recording this emission's actual probe.

    Returns SUPPORTED / OVERREACHING / UNTESTED plus the five recorded
    fields. Never 'wrong'. Never asserts the world.
    """
    row: dict[str, Any] = {
        "status": status,
        "probe_performed": probe_performed,
        "direct_observation": direct_observation,
        "interpretation": interpretation or status,
        "probe_kind": probe_kind,
        "use_catalog": False,
        "source": source or "<emit>",
    }
    if claim_kind:
        row["claim_kind"] = claim_kind
    if falsifier:
        row["falsifier"] = falsifier
    out = challenge(row)
    out["entry"] = "emit"
    return out


def stamp(status_row: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Write the five recorded fields onto a gate's status dict as it emits.

    Mutates `status_row` and returns the challenge record. The gate is
    expected to persist the stamped row; calling this and dropping the
    result is not coverage.
    """
    if not isinstance(status_row, dict):
        raise TypeError(f"stamp requires a mutable status dict, not {type(status_row).__name__}")
    status = str(
        kwargs.pop("status", None)
        or status_row.get("status")
        or status_row.get("id")
        or ""
    )
    probe_performed = kwargs.pop(
        "probe_performed", status_row.get("probe_performed") or ""
    )
    direct_observation = kwargs.pop(
        "direct_observation",
        status_row.get("direct_observation")
        if "direct_observation" in status_row
        else "",
    )
    interpretation = kwargs.pop(
        "interpretation",
        status_row.get("interpretation") or status_row.get("reason") or status,
    )
    rec = emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation=direct_observation,
        interpretation=str(interpretation or ""),
        **kwargs,
    )
    for key in FIVE_RECORDED_FIELDS:
        status_row[key] = rec[key]
    status_row["causality_verdict"] = rec["verdict"]
    status_row["falsifier"] = rec["falsifier"]
    return rec


def matching_overreach_shape(row: Mapping[str, Any] | str) -> str | None:
    """Shape id if this emission matches a known overreach, else None.

    Matches probe-kind/claim-kind pairs, not historical ids. A re-emitted
    BLOCKED_NO_METAL_GPU under a new name still matches
    SHAPE.PROCESS_ERROR_AS_HOST_ABSENCE. A MIXED one-layer roofline
    reported as ALU_BOUND matches SHAPE.MIXED_LAYER_AS_ALU_BOUND.
    """
    if isinstance(row, str):
        return matching_overreach_shape(challenge(row))
    probe = str(row.get("probe_kind") or _infer_probe_kind(row) or "")
    claim = str(row.get("claim_kind") or _claim_kind_of(row) or "")
    if _is_mixed_one_layer_probe(row) and (
        claim == CLAIM_GLOBAL_BINDING
        or _is_broad_alu_bound_claim(row)
        or str(row.get("status") or "") == "ALU_BOUND"
    ):
        return "SHAPE.MIXED_LAYER_AS_ALU_BOUND"
    for shape in OVERREACH_SHAPES:
        if shape["id"] == "SHAPE.MIXED_LAYER_AS_ALU_BOUND":
            continue
        if probe == shape["probe_kind"] and claim == shape["claim_kind"]:
            return shape["id"]
    return None


def overreach_shapes() -> list[dict[str, Any]]:
    return [dict(s) for s in OVERREACH_SHAPES]


def consequential_gates() -> list[dict[str, Any]]:
    return [dict(g) for g in CONSEQUENTIAL_GATES]


def records_five_fields(node: Any) -> bool:
    """True iff this mapping itself carries the five recorded fields."""
    if not isinstance(node, Mapping):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, Mapping):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    if not isinstance(alts, list) or not alts:
        return False
    return True


def _any_five_field_record(node: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if records_five_fields(node):
        return True
    if isinstance(node, Mapping):
        for value in node.values():
            if _any_five_field_record(value, depth + 1):
                return True
    elif isinstance(node, list):
        for value in node:
            if _any_five_field_record(value, depth + 1):
                return True
    return False


def _read_text(rel: str) -> str | None:
    """Source bytes from the worktree or git HEAD. None is unreadability."""
    rel = rel.replace("\\", "/").lstrip("./")
    path = REPO / rel
    if path.is_file():
        try:
            return path.read_text()
        except OSError:
            return None
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob
    return None


def _calls_emit_time(src: str) -> bool:
    if not src:
        return False
    markers = (
        "status_causality.emit(",
        "status_causality.stamp(",
        "sc.emit(",
        "sc.stamp(",
    )
    return any(m in src for m in markers)


def _gate_coverage_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec["name"])
    module = spec.get("module")
    receipt_rel = spec.get("receipt")
    src: str | None = None
    src_state = "absent"
    if module:
        src = _read_text(str(module))
        src_state = "readable" if src is not None else "unreadable"
    else:
        src_state = "no_module"
    doc = _load_receipt(str(receipt_rel)) if receipt_rel else None
    receipt_state = "readable" if doc is not None else "unreadable"
    calls_emit = _calls_emit_time(src or "")
    receipt_has_five = _any_five_field_record(doc) if doc is not None else False
    unreadable = (
        (module is not None and src_state == "unreadable")
        and receipt_state == "unreadable"
    )
    records = bool(receipt_has_five)
    gap = None
    if unreadable:
        gap = "source and receipt unreadable in this checkout; not scored as a silent miss"
    elif name == "flash_meta_teacher_capture_boundary" and not records:
        gap = FLASH_META_WIRING_NOTE
    elif not records:
        gap = (
            "emits a campaign-consequential status without probe_performed, "
            "direct_observation, interpretation, confidence, and alternatives "
            "on the stamped row"
        )
        if not calls_emit:
            gap += "; does not call status_causality.emit/stamp at emit time"
    in_scope = bool(module) and str(module) in THIS_LANE_WRITE_SCOPE
    row = {
        "name": name,
        "module": module,
        "receipt": receipt_rel,
        "emit_fn": spec.get("emit_fn"),
        "why": spec.get("why"),
        "selection": spec.get("selection"),
        "named_statuses": list(spec.get("named_statuses") or []),
        "source_state": src_state,
        "receipt_state": receipt_state,
        "calls_emit": calls_emit,
        "receipt_has_five_fields": receipt_has_five,
        "records_five_fields": records,
        "unreadable": unreadable,
        "in_this_lane_write_scope": in_scope,
        "gap": gap,
    }
    if spec.get("rust_emit_point"):
        row["rust_emit_point"] = spec.get("rust_emit_point")
        row["rust_emit_fn"] = spec.get("rust_emit_fn")
    return row


def coverage() -> dict[str, Any]:
    """Per-gate five-field coverage, by name. Not a percentage.

    A gate that does not record the five fields is a named gap. Unreadable
    source+receipt is a third bucket, not a fabricated miss.
    """
    rows = [_gate_coverage_row(g) for g in CONSEQUENTIAL_GATES]
    recording = [r["name"] for r in rows if r["records_five_fields"] and not r["unreadable"]]
    missing = [
        r["name"]
        for r in rows
        if not r["records_five_fields"] and not r["unreadable"]
    ]
    unread = [r["name"] for r in rows if r["unreadable"]]
    criterion_rows = []
    launch = next((r for r in rows if r["name"] == "odyssey_launch"), None)
    if launch is not None:
        doc = _load_receipt("receipts/future/ODYSSEY_LAUNCH_GATE.json")
        criteria = []
        if isinstance(doc, Mapping) and isinstance(doc.get("criteria"), list):
            criteria = [c for c in doc["criteria"] if isinstance(c, Mapping)]
        by_id = {str(c.get("id")): c for c in criteria}
        for cid in ODYSSEY_LAUNCH_CRITERIA:
            crit = by_id.get(cid)
            records = records_five_fields(crit) if crit is not None else False
            criterion_rows.append(
                {
                    "name": cid,
                    "gate": "odyssey_launch",
                    "records_five_fields": records,
                    "present_on_receipt": crit is not None,
                    "gap": None
                    if records
                    else "criterion row does not carry the five recorded fields",
                }
            )
    return {
        "selection_rule": SELECTION_RULE,
        "gates": rows,
        "recording_five_fields": recording,
        "not_recording_five_fields": missing,
        "unreadable": unread,
        "odyssey_launch_criteria": criterion_rows,
        "odyssey_launch_criteria_recording_five_fields": [
            c["name"] for c in criterion_rows if c["records_five_fields"]
        ],
        "odyssey_launch_criteria_not_recording_five_fields": [
            c["name"] for c in criterion_rows if not c["records_five_fields"]
        ],
        "g007_named": list(G007_NAMED_GATES),
        "n_gates": len(rows),
        "n_recording": len(recording),
        "n_not_recording": len(missing),
        "n_unreadable": len(unread),
        "remainder": [
            {
                "name": r["name"],
                "module": r.get("module") or r.get("rust_emit_point"),
                "why_not_wired": r.get("gap") or FLASH_META_WIRING_NOTE,
            }
            for r in rows
            if not r["records_five_fields"] and not r["unreadable"]
        ],
        "write_scope": list(THIS_LANE_WRITE_SCOPE),
        "write_scope_note": (
            "S015 dissolved the Codex write partition. This lane wired the eight "
            "hcli/agentos emit points (resident, native, native_mission, autonomy, "
            "modellake, vmcp, recovery, research). odyssey_launch, integration_gate, "
            "specimen_verify, metal_reachability, flash_nx_audit, odyssey2_law_store, "
            "contamination, qualification_pipeline and protected_scheduler were "
            "wired by prior lanes; their receipts are regenerated so the scanner "
            "and the coverage doc agree. flash_meta_teacher_capture_boundary stays "
            "named: its writer is Rust, and a Python shim would not be the emit point."
        ),
        "s015": S015_PARTITION_NOTE,
    }


def build_coverage() -> Path:
    """Write STATUS_CAUSALITY_COVERAGE.json from a live scan of sources and receipts."""
    cov = coverage()
    criterion_rows = []
    for row in cov.get("odyssey_launch_criteria") or []:
        criterion_rows.append(
            {
                "causality_verdict": None,
                "gap": row.get("gap"),
                "gate": row.get("gate"),
                "met": None,
                "name": row.get("name"),
                "present_on_receipt": row.get("present_on_receipt"),
                "records_five_fields": row.get("records_five_fields"),
            }
        )
    launch = _load_receipt("receipts/future/ODYSSEY_LAUNCH_GATE.json")
    if isinstance(launch, Mapping) and isinstance(launch.get("criteria"), list):
        by_id = {
            str(c.get("id")): c
            for c in launch["criteria"]
            if isinstance(c, Mapping)
        }
        for item in criterion_rows:
            crit = by_id.get(str(item["name"]))
            if crit is not None:
                item["met"] = crit.get("met")
                item["causality_verdict"] = crit.get("causality_verdict")
    doc = {
        "schema": COVERAGE_SCHEMA,
        "version": 1,
        "purpose": (
            "G007 consumer coverage: which consequential gates record "
            "probe_performed, direct_observation, interpretation, confidence, "
            "and alternatives at emit time. Names, not a percentage."
        ),
        "obligation": "G007",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "does_not_assert_world_state": True,
        "emit_entry_point": "tools.future.status_causality.emit()",
        "five_recorded_fields": list(FIVE_RECORDED_FIELDS),
        "law": LAW,
        "selection_rule": cov["selection_rule"],
        "g007_named": cov["g007_named"],
        "gates": cov["gates"],
        "recording_five_fields": cov["recording_five_fields"],
        "not_recording_five_fields": cov["not_recording_five_fields"],
        "unreadable": cov["unreadable"],
        "remainder": cov["remainder"],
        "n_gates": cov["n_gates"],
        "n_recording": cov["n_recording"],
        "n_not_recording": cov["n_not_recording"],
        "n_unreadable": cov["n_unreadable"],
        "odyssey_launch_criteria": criterion_rows,
        "odyssey_launch_criteria_recording_five_fields": cov[
            "odyssey_launch_criteria_recording_five_fields"
        ],
        "odyssey_launch_criteria_not_recording_five_fields": cov[
            "odyssey_launch_criteria_not_recording_five_fields"
        ],
        "write_scope": cov["write_scope"],
        "write_scope_note": cov["write_scope_note"],
        "s015": S015_PARTITION_NOTE,
        "sibling_coverage_available": False,
        "flash_meta_teacher_capture_boundary": {
            "wired": "flash_meta_teacher_capture_boundary"
            not in cov["not_recording_five_fields"],
            "language": "Rust",
            "emit_point": FLASH_META_RUST_EMIT_POINT,
            "emit_fn": "write_blocked_capture_boundary",
            "why_not_a_python_shim": FLASH_META_WIRING_NOTE,
        },
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. This receipt names "
            "which gates record the five causality fields. It does not assert that "
            "a SUPPORTED or OVERREACHING verdict is the true world state. S015 "
            "dissolved the Codex write partition; a remaining gap is named for a "
            "reason that is not that partition."
        ),
    }
    return write_receipt(COVERAGE_RECEIPT, doc, "tools/future/status_causality.py")


def improvement_trial_alu_bound_control() -> dict[str, Any]:
    """The live MIXED→ALU_BOUND negative control, or a recorded refusal."""
    doc = _load_receipt(IMPROVEMENT_TRIAL_REL)
    if doc is None:
        return {
            "readable": False,
            "reason": f"{IMPROVEMENT_TRIAL_REL} absent or unreadable in this checkout",
            "control": IMPROVEMENT_TRIAL_CONTROL,
            "agrees": False,
        }
    controls = ((doc.get("negative_controls") or {}) if isinstance(doc.get("negative_controls"), Mapping) else {}).get("controls") or []
    hit = None
    for item in controls:
        if isinstance(item, Mapping) and item.get("control") == IMPROVEMENT_TRIAL_CONTROL:
            hit = dict(item)
            break
    if hit is None:
        return {
            "readable": True,
            "reason": f"control {IMPROVEMENT_TRIAL_CONTROL!r} not in {IMPROVEMENT_TRIAL_REL}",
            "control": IMPROVEMENT_TRIAL_CONTROL,
            "agrees": False,
        }
    failures = hit.get("automatic_failures") or []
    detail = ""
    if failures and isinstance(failures[0], Mapping):
        detail = str(failures[0].get("detail") or "")
    agrees = (
        hit.get("failed") is True
        and hit.get("verdict") == "FAIL"
        and "ALU_BOUND" in detail
        and "MIXED" in detail
    )
    return {
        "readable": True,
        "control": IMPROVEMENT_TRIAL_CONTROL,
        "verdict": hit.get("verdict"),
        "failed": hit.get("failed"),
        "detail": detail,
        "expected_detail": IMPROVEMENT_TRIAL_ALU_BOUND_DETAIL,
        "agrees": agrees,
        "receipt": IMPROVEMENT_TRIAL_REL,
    }


def verdict(row: Mapping[str, Any] | str) -> str:
    """SUPPORTED | OVERREACHING | UNTESTED. Never 'wrong'."""
    challenged = challenge(row) if not _looks_challenged(row) else None
    if challenged is not None:
        v = challenged["verdict"]
    else:
        v = _verdict_from_parts(row if isinstance(row, Mapping) else {})
    if v not in VERDICTS:
        raise ValueError(f"verdict {v!r} is not in {VERDICTS}")
    return v


def _looks_challenged(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and "probe_performed" in row
        and "alternatives" in row
        and "claim_kind" in row
    )


def _normalize(status: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(status, str):
        name = status.strip()
        for hc in HISTORICAL_CASES:
            if hc["status"] == name or hc["id"] == name:
                return dict(hc)
        for sf in SUPPORTED_FIXTURES:
            if sf["status"] == name or sf["id"] == name:
                return dict(sf)
        return {"status": name, "probe_kind": "", "probe_performed": "", "direct_observation": ""}
    if not isinstance(status, Mapping):
        raise TypeError(f"status must be str or mapping, not {type(status).__name__}")
    row = dict(status)
    if row.get("probe") and isinstance(row.get("probe"), Mapping) and not row.get("probe_kind"):
        return _from_embedded_probe(row, str(row.get("source") or "<dict>"))
    # A receipt-shaped mapping is a document to extract from, not a row,
    # unless it already names a probe or is a single status with failure.
    if _is_document(row) and not row.get("probe_kind") and not row.get("probe_performed"):
        extracted = list(_iter_status_rows(row, str(row.get("source") or "<dict>")))
        if len(extracted) == 1:
            return extracted[0]
        if extracted:
            # Prefer a broad historical label if one is in the document.
            for item in extracted:
                if str(item.get("status") or "") in BROAD_ABSENCE_LABELS:
                    return item
            return extracted[0]
    if not row.get("probe_kind") and not row.get("probe_performed"):
        # Historical catalog may still document the probe for this status name.
        name = str(row.get("status") or "").strip()
        for hc in HISTORICAL_CASES:
            if hc["status"] == name and row.get("use_catalog") is not False:
                # Only fill from the catalog when the caller did not already
                # pass a document that recorded no probe. A bare name may.
                if "probe" not in row and "failure" not in row and "criteria" not in row:
                    filled = dict(hc)
                    filled.update({k: v for k, v in row.items() if k not in filled or k == "status"})
                    return filled
    return row


def _is_document(row: Mapping[str, Any]) -> bool:
    return any(
        k in row
        for k in ("failure", "results", "schools", "criteria", "observed", "seven_all_met")
    )


def scan(
    receipts: Sequence[Mapping[str, Any] | str | Path] | None = None,
    *,
    include_historical: bool | None = None,
) -> list[dict[str, Any]]:
    """Find status strings whose interpretation exceeds their probe.

    `receipts=None` scans the known motivating receipts plus the historical
    catalog, so a sparse checkout still challenges the five cases. Passing
    an explicit list scans only those documents. A document that records a
    status and no probe yields UNTESTED, never OVERREACHING.
    """
    if include_historical is None:
        include_historical = receipts is None
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    refusals: list[dict[str, str]] = []
    for doc, source in _documents(receipts):
        if isinstance(doc, Mapping) and doc.get("_load_refused"):
            refusals.append({"source": source, "reason": str(doc.get("reason") or "unreadable")})
            continue
        for raw in _iter_status_rows(doc, source):
            challenged = challenge(raw)
            key = (challenged.get("status"), challenged.get("source"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(challenged)
    if include_historical:
        for hc in HISTORICAL_CASES:
            key = (hc["status"], hc["source"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(challenge(hc))
    if refusals:
        for item in rows:
            item.setdefault("load_refusals", refusals)
    return rows


def _documents(
    receipts: Sequence[Mapping[str, Any] | str | Path] | None,
) -> Iterable[tuple[dict[str, Any], str]]:
    if receipts is None:
        items: Sequence[Mapping[str, Any] | str | Path] = KNOWN_RECEIPT_PATHS
    else:
        items = receipts
    for item in items:
        if isinstance(item, Mapping):
            source = str(item.get("source") or item.get("rel") or item.get("id") or "<dict>")
            yield dict(item), source
            continue
        rel = str(item)
        doc = _load_receipt(rel)
        if doc is None:
            yield {"_load_refused": True, "rel": rel, "reason": "absent or unreadable in this checkout"}, rel
        else:
            yield doc, rel


def _iter_status_rows(doc: Mapping[str, Any], source: str) -> Iterable[dict[str, Any]]:
    if doc.get("_load_refused"):
        return
    schema = str(doc.get("schema") or "")
    if schema in ADJUDICATION_SCHEMAS or schema == "hawking.future.metal_reachability.v1":
        observed = doc.get("observed") if isinstance(doc.get("observed"), Mapping) else None
        if observed and observed.get("system_default"):
            yield {
                "id": "EX.METAL_OBSERVED_DEVICE",
                "status": "HOST_HAS_METAL_GPU",
                "source": source,
                "probe_kind": PROBE_ENUMERATION,
                "probe_performed": "MTLCreateSystemDefaultDevice() recorded on this receipt",
                "direct_observation": observed,
                "interpretation": (
                    "this process saw a Metal device, so the host has a Metal-capable GPU"
                ),
                "claim_kind": CLAIM_DEVICE_PRESENT,
            }
        return

    # Already a challenge-shaped or fixture-shaped row.
    if doc.get("status") and (doc.get("probe_kind") or doc.get("probe_performed")):
        yield dict(doc)
        return
    if doc.get("status") and doc.get("probe") and isinstance(doc.get("probe"), Mapping):
        yield _from_embedded_probe(doc, source)
        # Continue: a document may also carry criteria/results.

    failure = doc.get("failure") if isinstance(doc.get("failure"), Mapping) else None
    if doc.get("status") and failure:
        yield {
            "status": doc["status"],
            "source": source,
            "probe_kind": PROBE_PROCESS_ERROR,
            "probe_performed": (
                f"error at stage {failure.get('stage')!r}; "
                "the status label is applied to this process failure"
            ),
            "direct_observation": (
                f"failure.stage={failure.get('stage')}; "
                f"failure.error={failure.get('error')!r}"
            ),
            "interpretation": _interpretation_from_boundary(doc, str(doc["status"])),
        }

    if doc.get("status") and not failure and not doc.get("probe") and not doc.get("probe_kind"):
        # A status with nothing that records the probe. Emit UNTESTED.
        if not any(k in doc for k in ("results", "schools", "criteria", "seven_all_met")):
            yield {
                "status": doc["status"],
                "source": source,
                "probe_kind": "",
                "probe_performed": "",
                "direct_observation": "",
                "interpretation": str(doc.get("claim_boundary") or doc["status"]),
                # This document recorded a label and not a probe. Do not
                # backfill the historical catalog: absence of a probe is
                # UNTESTED, even when the label is one we have seen before.
                "use_catalog": False,
            }

    for result in doc.get("results") or []:
        if isinstance(result, Mapping) and result.get("status"):
            yield _from_specimen_row(result, source)

    schools = doc.get("schools") if isinstance(doc.get("schools"), Mapping) else {}
    for name, school in schools.items():
        if not isinstance(school, Mapping):
            continue
        phys = school.get("physical_status")
        if not phys:
            continue
        yield {
            "status": "WEIGHTS_NOT_PRESENT"
            if phys == "metadata_only_weights_not_present"
            else str(phys),
            "source": source,
            "probe_kind": PROBE_METADATA,
            "probe_performed": f"read schools.{name}.physical_status from {source}",
            "direct_observation": str(phys),
            "interpretation": (
                f"{name} weights are not present"
                if phys == "metadata_only_weights_not_present"
                else f"schools.{name}.physical_status={phys}"
            ),
        }

    for crit in doc.get("criteria") or []:
        if isinstance(crit, Mapping):
            extracted = _from_criterion(crit, source)
            if extracted:
                yield extracted

    if "seven_all_met" in doc and doc.get("status") in {None, "seven_all_met_is_false"}:
        yield {
            "status": "seven_all_met_is_false",
            "source": source,
            "probe_kind": PROBE_RECEIPT_FIELD,
            "probe_performed": f"read seven_all_met from {source}",
            "direct_observation": f"seven_all_met={doc.get('seven_all_met')!r}",
            "interpretation": f"the completeness audit currently records seven_all_met={doc.get('seven_all_met')!r}",
            "claim_kind": CLAIM_FIELD_VALUE,
        }


def _interpretation_from_boundary(doc: Mapping[str, Any], status: str) -> str:
    boundary = str(doc.get("claim_boundary") or "")
    if "this host has no Metal-capable GPU" in boundary:
        return "this host has no Metal-capable GPU"
    if "the model is missing" in boundary.lower():
        return "the model is missing"
    return status


def _from_embedded_probe(doc: Mapping[str, Any], source: str) -> dict[str, Any]:
    probe = doc["probe"] if isinstance(doc.get("probe"), Mapping) else {}
    kind = str(probe.get("kind") or "")
    kind_map = {
        "path_existence": PROBE_PATH_EXISTENCE,
        "listing_membership": PROBE_LISTING,
        "process_error": PROBE_PROCESS_ERROR,
        "metadata_field": PROBE_METADATA,
        "literal_constant": PROBE_LITERAL,
        "device_enumeration": PROBE_ENUMERATION,
        "hash_recompute": PROBE_HASH,
        "receipt_field": PROBE_RECEIPT_FIELD,
        "measured_flags": PROBE_MEASURED_FLAGS,
        "one_layer_roofline": PROBE_ONE_LAYER_ROOFLINE,
    }
    return {
        "status": doc.get("status"),
        "source": source,
        "probe_kind": kind_map.get(kind, kind),
        "probe_performed": (
            f"{kind} path={probe.get('path')!r} exists={probe.get('exists')!r}"
            if kind == "path_existence"
            else f"{kind} listing={probe.get('listing')!r} present={probe.get('present_in_listing')!r}"
            if kind == "listing_membership"
            else json.dumps(probe, sort_keys=True)
        ),
        "direct_observation": json.dumps(probe, sort_keys=True),
        "interpretation": _interpretation_from_boundary(doc, str(doc.get("status") or "")),
    }


def _from_specimen_row(result: Mapping[str, Any], source: str) -> dict[str, Any]:
    status = str(result.get("status") or "")
    obs = {
        "specimen": result.get("specimen"),
        "n_files": result.get("n_files"),
        "verified": result.get("verified"),
        "mismatched": result.get("mismatched"),
        "no_remote_digest": result.get("no_remote_digest"),
        "bytes_hashed": result.get("bytes_hashed"),
    }
    return {
        "status": status,
        "source": source,
        "probe_kind": PROBE_HASH,
        "probe_performed": (
            f"recompute published digests for {result.get('specimen')}"
        ),
        "direct_observation": obs,
        "interpretation": (
            "this specimen's published digests match the hashes recomputed here"
            if status == "WHOLE_TREE_VERIFIED"
            else status
        ),
        "claim_kind": CLAIM_DIGEST_MATCH if status == "WHOLE_TREE_VERIFIED" else None,
    }


def _from_criterion(crit: Mapping[str, Any], source: str) -> dict[str, Any] | None:
    cid = str(crit.get("id") or "")
    if not cid:
        return None
    reason = str(crit.get("reason") or "")
    flags = {}
    op = crit.get("operational")
    if isinstance(op, Mapping):
        flags = dict(op.get("flags") or {})
    bundled_literal = "schedule/frontier/refill are false" in reason
    if bundled_literal:
        return {
            "status": cid,
            "source": source,
            "probe_kind": PROBE_LITERAL,
            "probe_performed": (
                "operational_bar schedule/frontier/refill asserted False as a bundle"
            ),
            "direct_observation": f"flags={flags!r}; reason={reason}",
            "interpretation": f"{cid} is not resident-callable",
            "claim_kind": CLAIM_CAPABILITY_ABSENCE,
        }
    # Measured criterion: the claim is the flags themselves, not a world fact.
    unmet = [k for k, v in flags.items() if v is False]
    return {
        "status": cid,
        "source": source,
        "probe_kind": PROBE_MEASURED_FLAGS,
        "probe_performed": f"evaluate launch criterion {cid} against disk evidence",
        "direct_observation": f"met={crit.get('met')!r} unmet_flags={unmet} reason={reason}",
        "interpretation": reason or f"{cid} met={crit.get('met')!r}",
        "claim_kind": CLAIM_MEASURED_UNMET if crit.get("met") is False else CLAIM_FIELD_VALUE,
    }


def challenge_historical() -> list[dict[str, Any]]:
    return [challenge(c) for c in HISTORICAL_CASES]


def challenge_supported() -> list[dict[str, Any]]:
    return [challenge(c) for c in SUPPORTED_FIXTURES]


# ---------------------------------------------------------------------------
# Generic causal claim checker. Narrow probe, broad conclusion.
#
# challenge() keeps its three verdicts (a status with no probe is UNTESTED).
# This path classifies an (observation, conclusion) pair:
#   SUPPORTED        observation entails the conclusion
#   OVERREACHING     observation is consistent with the conclusion being false
#   UNDERDETERMINED  observation neither entails nor contradicts the conclusion
#   CONTRADICTED     observation is inconsistent with the conclusion
# ---------------------------------------------------------------------------

CAMPAIGN_CLAIM_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "CC.PREFILL_AS_PER_TOKEN",
        "scar_id": "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR",
        "observation": (
            "active_bytes_per_token and dispatches_per_generated_token divide "
            "prefill+decode totals by generated tokens; P=12 N=127 G=128 so "
            "the factor is 139/128; reported dispatches 1046.84375 = 964 * 139/128"
        ),
        "conclusion": (
            "those figures are the per-token production costs, so the clean-GEMV "
            "roof is 65.58 TPS and 71 TPS is above it"
        ),
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "the observation is an accounting identity about a denominator; "
            "it does not measure per-forward-pass bytes or production dispatch. "
            "A world in which the true per-pass cost is 9.878 GB and 628 fused "
            "dispatches is fully consistent with the same reported fields."
        ),
    },
    {
        "id": "CC.964_AS_PRODUCTION",
        "scar_id": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "observation": (
            "a probe of the resident with the fusion env unset returned "
            "964 dispatches per decode step"
        ),
        "conclusion": "964 is the production dispatch count for sealed-3.14",
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "the probe measured the unfused default graph. Production sealed-3.14 "
            "sets the fusion env and issues 628. The same observation is exactly "
            "what the unfused arm produces."
        ),
    },
    {
        "id": "CC.SOURCE_FIELDS_AS_RUNNING",
        "scar_id": "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
        "observation": (
            "ascension_qwen38_resident.rs emits dispatches, "
            "dispatches_per_generated_token, active_bytes_per_token and kin"
        ),
        "conclusion": "the running resident can report those fields",
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "source containing a field does not entail the serving binary "
            "containing it. The binary was built 2026-08-26; the instrumentation "
            "landed 2026-08-27 in 8b6f50270."
        ),
    },
    {
        "id": "CC.PRIORITY_ZERO_VIA_OR",
        "scar_id": "PRIORITY_ZERO_FALSY_OR_DEFAULT",
        "observation": (
            "rank used `_detach_priority(j) or 99`; long detached jobs have "
            "priority 0"
        ),
        "conclusion": "priority 0 ranks first, so the long jobs start first",
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": False,
        "contradicts": True,
        "reason": (
            "0 is falsy, so `0 or 99` is 99. The observation is inconsistent "
            "with the conclusion that those jobs rank first."
        ),
    },
    {
        "id": "CC.MIXED_UNITS_NO_OVERLAP",
        "scar_id": "EVENT_TIMESTAMP_UNIT_MISMATCH",
        "observation": (
            "fallback mixed started_at=1788141337.2 (epoch) with t_s=35.0 "
            "(trial-relative) and reported a negative interval; same-unit "
            "stamps on the same events give +2.34s overlap"
        ),
        "conclusion": "the two jobs did not overlap",
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": False,
        "contradicts": True,
        "reason": (
            "the mixed-unit subtraction is not a measurement of overlap. The "
            "same-unit arithmetic on the same events is +2.34s, which is "
            "inconsistent with 'did not overlap'."
        ),
    },
    {
        "id": "CC.ADJACENCY_AS_OVERLAP",
        "scar_id": "ADJACENCY_IS_NOT_OVERLAP",
        "observation": (
            "two detached_started events appear with no detached_completed "
            "between them in the log"
        ),
        "conclusion": "two jobs were live at once",
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "adjacency in a log is consistent with sequential jobs whose "
            "completion event is late or unattributable. Two live pids, or "
            "same-unit intervals with positive overlap, would entail the claim."
        ),
    },
    {
        "id": "CC.BARE_COMMIT_INTENDED_PATHS",
        "scar_id": "SHARED_INDEX_BARE_COMMIT_SWEEPS_FOREIGN_STAGE",
        "observation": (
            "`git add <supervisor paths> && git commit` succeeded after "
            "`git apply --3way` of another lane"
        ),
        "conclusion": "the commit contains only the supervisor paths named in the add",
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "`git commit` without a pathspec commits the index. apply --3way "
            "stages as a side effect, so the observation is exactly what a "
            "mixed commit looks like. fb4240dad carried autonomy_run.py."
        ),
    },
    {
        "id": "CC.CATALOG_CENSUS_INFLATION",
        "scar_id": "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR",
        "observation": (
            "independent HQ38M20 per-tensor census sums to 9,878,901,136; "
            "resident active_bytes_per_token is 10,727,793,881.75; "
            "10,727,793,881.75 / (139/128) restores the census to 7 ppm"
        ),
        "conclusion": (
            "the resident's per_generated_token byte field is inflated by "
            "(P+N)/G for this run"
        ),
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": True,
        "consistent_with_negation": False,
        "contradicts": False,
        "reason": (
            "two independent numerators (catalog sum, reported field) related "
            "by the exact predicted factor leave no world in which the field "
            "is a true per-pass cost."
        ),
    },
    {
        "id": "CC.STATIC_AND_LIVE_DISPATCH",
        "scar_id": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "observation": (
            "tps_budget.py encode-path walk predicted 964 unfused and 628 fused; "
            "the live probe returned 964.00 and 628.00 under those envs"
        ),
        "conclusion": (
            "the unfused graph issues 964 dispatches per decode step and the "
            "sealed fusion graph issues 628"
        ),
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": True,
        "consistent_with_negation": False,
        "contradicts": False,
        "reason": (
            "two independent methods, one static and one dynamic, landing on "
            "the same integers under named envs. That is what those counts are."
        ),
    },
    {
        "id": "CC.MISSING_ENV_WHICH_GRAPH",
        "scar_id": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "observation": "a dispatch count of 964 was recorded; no env hash is attached",
        "conclusion": "this is the production sealed-3.14 dispatch count",
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "without the env actually in effect, 964 is consistent with the "
            "unfused graph, with a different parent, and with a misread of "
            "production. The conclusion is not settled by the number alone; "
            "this is the underdetermined form of the 964-as-production error."
        ),
        "force_underdetermined": True,
    },
    {
        "id": "CC.STRINGS_ABSENT_MAY_BE_STRIPPED",
        "scar_id": "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
        "observation": (
            "strings(binary) does not contain 'dispatches_per_generated_token'"
        ),
        "conclusion": "the serving binary predates its instrumentation",
        "author_was_the_one_who_concluded": False,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "absence of a string in a stripped or optimized binary is strong "
            "but not absolute. A world in which the field is compiled in and "
            "the string table dropped is consistent with the observation. "
            "mtime vs introducing commit, or a live probe, would settle it."
        ),
        "force_underdetermined": True,
    },
    {
        "id": "CC.UNFUSED_INHERITS_INCUMBENT",
        "scar_id": "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
        "observation": (
            "benchmark env_hash is the empty HAWKING_* set; sealed env_hash "
            "includes HAWKING_QWEN38_FUSE_*"
        ),
        "conclusion": "the result may inherit the sealed-3.14-production label",
        "author_was_the_one_who_concluded": True,
        "has_observation": True,
        "entails": False,
        "consistent_with_negation": True,
        "contradicts": False,
        "reason": (
            "environment is part of experiment identity. A mismatched env hash "
            "must be labelled with its own environment, not the incumbent."
        ),
    },
)


def campaign_claim_cases() -> list[dict[str, Any]]:
    return [dict(c) for c in CAMPAIGN_CLAIM_CASES]


def _classify_claim(
    *,
    has_observation: bool,
    entails: bool,
    consistent_with_negation: bool,
    contradicts: bool,
    force_underdetermined: bool = False,
) -> tuple[str, str]:
    """Map structured flags onto a claim-check verdict. Flags, not a lookup."""
    if not has_observation:
        return (
            UNDERDETERMINED,
            "no observation was given; the conclusion is unconstrained",
        )
    if contradicts:
        return (
            CONTRADICTED,
            "the observation is inconsistent with the conclusion",
        )
    if force_underdetermined or (not entails and not consistent_with_negation):
        return (
            UNDERDETERMINED,
            "the observation neither entails nor contradicts the conclusion",
        )
    if consistent_with_negation:
        return (
            OVERREACHING,
            "the observation is consistent with the conclusion being false "
            "(narrow probe, broad conclusion)",
        )
    if entails:
        return (
            SUPPORTED,
            "the observation entails the conclusion; no recorded world keeps "
            "the observation and drops the conclusion",
        )
    return (
        UNDERDETERMINED,
        "the observation neither entails nor contradicts the conclusion",
    )


def _seed_by_id_or_text(
    observation: Any, conclusion: Any, case_id: str | None
) -> dict[str, Any] | None:
    if case_id:
        for c in CAMPAIGN_CLAIM_CASES:
            if c["id"] == case_id or c.get("scar_id") == case_id:
                return dict(c)
    obs_s = str(observation or "").strip()
    con_s = str(conclusion or "").strip()
    if not obs_s and not con_s:
        return None
    for c in CAMPAIGN_CLAIM_CASES:
        if obs_s and obs_s in {c["id"], c.get("scar_id", "")}:
            return dict(c)
        if obs_s == str(c.get("observation") or "") and (
            not con_s or con_s == str(c.get("conclusion") or "")
        ):
            return dict(c)
    return None


def check_claim(
    observation: Any = "",
    conclusion: Any = "",
    *,
    case_id: str | None = None,
    has_observation: bool | None = None,
    entails: bool | None = None,
    consistent_with_negation: bool | None = None,
    contradicts: bool | None = None,
    force_underdetermined: bool | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Classify (observation, conclusion) as a causal claim.

    Structured flags win. A seeded campaign case supplies flags when the
    caller passes its id (or its exact observation text) without flags.
    Unknown free text with no flags is UNDERDETERMINED — we do not guess.
    Never returns 'wrong'.
    """
    if isinstance(observation, Mapping) and (
        "observation" in observation or "conclusion" in observation or "id" in observation
    ):
        row = dict(observation)
        case_id = case_id or (str(row.get("id")) if row.get("id") else None)
        conclusion = conclusion or row.get("conclusion") or ""
        observation = row.get("observation") or observation
        if has_observation is None and "has_observation" in row:
            has_observation = bool(row.get("has_observation"))
        if entails is None and "entails" in row:
            entails = bool(row.get("entails"))
        if consistent_with_negation is None and "consistent_with_negation" in row:
            consistent_with_negation = bool(row.get("consistent_with_negation"))
        if contradicts is None and "contradicts" in row:
            contradicts = bool(row.get("contradicts"))
        if force_underdetermined is None and "force_underdetermined" in row:
            force_underdetermined = bool(row.get("force_underdetermined"))
        if reason is None:
            reason = row.get("reason")

    seed = _seed_by_id_or_text(observation, conclusion, case_id)
    flags_given = any(
        x is not None
        for x in (entails, consistent_with_negation, contradicts, force_underdetermined)
    )
    if seed is not None and not flags_given:
        has_observation = bool(seed.get("has_observation", True)) if has_observation is None else has_observation
        entails = bool(seed.get("entails"))
        consistent_with_negation = bool(seed.get("consistent_with_negation"))
        contradicts = bool(seed.get("contradicts"))
        force_underdetermined = bool(seed.get("force_underdetermined", False))
        observation = seed.get("observation") if not str(observation or "").strip() or str(observation) in {seed["id"], seed.get("scar_id")} else observation
        conclusion = conclusion or seed.get("conclusion") or ""
        reason = reason or seed.get("reason")
        case_id = case_id or seed.get("id")

    obs_present = bool(str(observation or "").strip()) if has_observation is None else bool(has_observation)
    # Free text with no flags and no seed: we cannot tell. That is
    # UNDERDETERMINED, not OVERREACHING — accusing in the dark is how a
    # regex attacker cried wolf in this partition.
    if seed is None and not flags_given:
        v, default_reason = _classify_claim(
            has_observation=obs_present,
            entails=False,
            consistent_with_negation=False,
            contradicts=False,
            force_underdetermined=True,
        )
        out = {
            "observation": observation,
            "conclusion": conclusion,
            "verdict": v,
            "reason": reason or default_reason,
            "law": LAW,
        }
        if case_id:
            out["id"] = case_id
        return out

    v, default_reason = _classify_claim(
        has_observation=obs_present,
        entails=bool(entails),
        consistent_with_negation=bool(consistent_with_negation),
        contradicts=bool(contradicts),
        force_underdetermined=bool(force_underdetermined),
    )
    if v not in CLAIM_CHECK_VERDICTS:
        raise ValueError(f"claim-check verdict {v!r} is not in {CLAIM_CHECK_VERDICTS}")
    if v in FORBIDDEN_VERDICTS:
        raise ValueError(f"the routine does not adjudicate the world: {v!r}")
    out = {
        "observation": observation,
        "conclusion": conclusion,
        "verdict": v,
        "reason": reason or default_reason,
        "has_observation": obs_present,
        "entails": bool(entails),
        "consistent_with_negation": bool(consistent_with_negation),
        "contradicts": bool(contradicts),
        "law": LAW,
    }
    if case_id:
        out["id"] = case_id
    if seed is not None:
        out["scar_id"] = seed.get("scar_id")
        out["author_was_the_one_who_concluded"] = bool(
            seed.get("author_was_the_one_who_concluded")
        )
        out["seeded"] = seed.get("id")
    for key in WORLD_STATE_KEYS:
        if key in out:
            raise RuntimeError(f"check_claim leaked a world-state key: {key}")
    return out


def check_campaign_claims() -> list[dict[str, Any]]:
    """Run the seeded campaign cases through the classifier, not a table lookup."""
    return [check_claim(c) for c in CAMPAIGN_CLAIM_CASES]


def build() -> Path:
    historical = challenge_historical()
    well_founded = challenge_supported()
    scanned = scan(None, include_historical=True)
    loadable: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for rel in KNOWN_RECEIPT_PATHS:
        doc = _load_receipt(rel)
        if doc is None:
            missing.append({"rel": rel, "reason": "absent or unreadable in this checkout"})
        else:
            loadable.append({"rel": rel, "schema": doc.get("schema"), "n_keys": len(doc)})

    overreaching = [r for r in historical if r["verdict"] == OVERREACHING]
    supported = [r for r in well_founded if r["verdict"] == SUPPORTED]
    untested_control = challenge(
        {"status": "SOME_NOVEL_BLOCKER", "interpretation": "the GPU is missing"}
    )
    cov = coverage()
    alu_control = improvement_trial_alu_bound_control()
    named_gaps = [
        f"{name} does not record the five fields at emit time"
        for name in cov["not_recording_five_fields"]
    ]

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Make challenging a status a resident routine: a status may assert "
            "only what its actual probe establishes."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "law": LAW,
        "scar": "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
        "verdicts_emitted": list(VERDICTS),
        "verdicts_refused": sorted(FORBIDDEN_VERDICTS),
        "historical_cases": historical,
        "supported_fixtures": well_founded,
        "n_historical_overreaching": sum(1 for r in historical if r["verdict"] == OVERREACHING),
        "n_supported_fixtures": sum(1 for r in well_founded if r["verdict"] == SUPPORTED),
        "untested_control": untested_control,
        "scan": {
            "n_rows": len(scanned),
            "n_overreaching": sum(1 for r in scanned if r["verdict"] == OVERREACHING),
            "n_supported": sum(1 for r in scanned if r["verdict"] == SUPPORTED),
            "n_untested": sum(1 for r in scanned if r["verdict"] == UNTESTED),
            "statuses": [
                {"status": r.get("status"), "verdict": r["verdict"], "source": r.get("source")}
                for r in scanned
            ],
        },
        "receipts_consulted": loadable,
        "receipts_unreadable": missing,
        "does_not_assert_world_state": True,
        "recovered_implementation": [
            "tools/future/autonomy_scars.py records the law and four instances, including STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
            "tools/future/metal_reachability.py is the worked falsifier of BLOCKED_NO_METAL_GPU as a host property",
            "tools/future/odyssey_launch.py::_eval_callable_tool historically hardcoded schedule/frontier/refill False",
            "tools/future/odyssey_launch.py::_resolve_stale_input distinguishes a moved path from a missing model",
            "tools/future/odyssey_launch.py::_ready deferred to physical_status=metadata_only_weights_not_present",
            "tools/future/specimen_verify.py::EXTRA_SPECIMENS names the partial/ Qwen3-0.6B that the listing missed",
            "tools/future/odyssey2_law_store.py::SCHOOLS['Flash'] still carries physical_status=metadata_only_weights_not_present",
            "tools/future/global_frontier.py F019 already separates the stall (corroborated) from the stated cause (not)",
            "tools/future/odyssey3_adversary.py refuses a law that emits no attack; this routine refuses a status with no probe",
        ],
        "gaps_closed": [
            "challenging a status was something Claude noticed five times, not a resident call",
            "no routine produced probe/observation/interpretation/alternatives/falsifier for a blocker",
            "the five historical overreaches had no regression that would fire on the next similar label — emit() of the same shape is now OVERREACHING, including MIXED one-layer roofline as ALU_BOUND",
            "the routine only scanned receipts on demand; emit() is the emit-time entry point a gate calls as it stamps a status",
        ],
        "negative_findings": [
            "this routine does not establish why the original Metal process saw no device",
            "this routine does not establish that Doctor or Gravity can run today; it challenges the historical constant",
            "this routine does not re-verify 360GB of Flash weights; it challenges the stale physical_status field",
            f"known receipts unreadable in this checkout: {len(missing)}",
            "a status whose probe is not recorded is UNTESTED, not OVERREACHING",
            "coverage is a list of names, not a percentage; gates that do not record the five fields are named gaps",
            *named_gaps,
        ],
        "resident_callable": {
            "entry_point": "tools.future.status_causality.challenge()",
            "emit_entry_point": "tools.future.status_causality.emit()",
            "stamp_entry_point": "tools.future.status_causality.stamp()",
            "workunit": (
                "one CPU_ANALYSIS unit; challenge a consequential blocker before "
                "acting on its label"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.VERIFICATION.negative-index",
            "fails_closed": (
                "an absent probe is UNTESTED never OVERREACHING; an unreadable "
                "receipt is a recorded refusal; the routine never returns 'wrong' "
                "and never asserts the true world state; emit() does not catalog-lookup "
                "a bare status name"
            ),
        },
        "emit_time": {
            "entry_point": "tools.future.status_causality.emit()",
            "stamp": "tools.future.status_causality.stamp()",
            "cheap": "no disk, no catalog, no subprocess",
            "returns": list(VERDICTS),
            "records": list(FIVE_RECORDED_FIELDS),
            "intended_call": (
                "status_causality.emit(status, probe_performed=..., "
                "direct_observation=..., interpretation=..., probe_kind=...)"
            ),
        },
        "overreach_shapes": [dict(s) for s in OVERREACH_SHAPES],
        "n_overreach_shapes": len(OVERREACH_SHAPES),
        "alu_bound_mixed_agreement": alu_control,
        "coverage": cov,
        "gates_recording_five_fields": cov["recording_five_fields"],
        "gates_not_recording_five_fields": cov["not_recording_five_fields"],
        "gates_unreadable": cov["unreadable"],
        "odyssey_iii": dict(ODYSSEY_III_CALL),
    }
    if len(overreaching) != len(HISTORICAL_CASES):
        doc["negative_findings"].append(
            "the historical catalog did not all verdict OVERREACHING; the detector is incomplete"
        )
    if len(supported) < 3:
        doc["negative_findings"].append(
            "fewer than three well-founded fixtures verdict SUPPORTED; the detector will cry wolf"
        )
    if untested_control["verdict"] != UNTESTED:
        doc["negative_findings"].append(
            "a status with no recorded probe was not UNTESTED; the detector is accusing in the dark"
        )
    written = write_receipt(RECEIPT, doc, "tools/future/status_causality.py")
    build_coverage()
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--challenge", metavar="STATUS")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--emit", metavar="STATUS")
    ap.add_argument("--probe", default="", help="probe_performed for --emit")
    ap.add_argument("--observation", default="", help="direct_observation for --emit")
    ap.add_argument("--interpretation", default="", help="interpretation for --emit")
    ap.add_argument("--probe-kind", default="", dest="probe_kind")
    a = ap.parse_args()
    if a.challenge:
        print(json.dumps(challenge(a.challenge), indent=1, sort_keys=True, default=str))
        return 0
    if a.emit:
        print(
            json.dumps(
                emit(
                    a.emit,
                    probe_performed=a.probe,
                    direct_observation=a.observation,
                    interpretation=a.interpretation,
                    probe_kind=a.probe_kind,
                ),
                indent=1,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    if a.scan:
        print(json.dumps(scan(), indent=1, sort_keys=True, default=str))
        return 0
    if a.coverage:
        cov = coverage()
        print(
            json.dumps(
                {
                    "selection_rule": cov["selection_rule"],
                    "recording_five_fields": cov["recording_five_fields"],
                    "not_recording_five_fields": cov["not_recording_five_fields"],
                    "unreadable": cov["unreadable"],
                    "odyssey_launch_criteria_not_recording_five_fields": cov[
                        "odyssey_launch_criteria_not_recording_five_fields"
                    ],
                },
                indent=1,
            )
        )
        return 0
    out = build()
    print(out)
    doc = json.loads(out.read_text())
    cov_path = RECEIPTS / COVERAGE_RECEIPT
    cov_doc = json.loads(cov_path.read_text()) if cov_path.is_file() else {}
    print(
        json.dumps(
            {
                "n_historical_overreaching": doc["n_historical_overreaching"],
                "n_supported_fixtures": doc["n_supported_fixtures"],
                "untested_control": doc["untested_control"]["verdict"],
                "scan": {
                    k: doc["scan"][k]
                    for k in ("n_rows", "n_overreaching", "n_supported", "n_untested")
                },
                "gates_recording_five_fields": doc["gates_recording_five_fields"],
                "gates_not_recording_five_fields": doc["gates_not_recording_five_fields"],
                "coverage_receipt": {
                    "recording_five_fields": cov_doc.get("recording_five_fields"),
                    "not_recording_five_fields": cov_doc.get("not_recording_five_fields"),
                    "n_recording": cov_doc.get("n_recording"),
                    "n_gates": cov_doc.get("n_gates"),
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
