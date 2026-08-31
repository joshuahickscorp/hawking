"""ACCELERATOR_WORKUNITS — schedule Codex's profile→cut→AB→reprofile loop as HCLI species.

codex_behaviors.py already froze the thirty species and the GPU-host DAG.
This module does not redefine them. It is the sidecar scheduler: emit each
real loop stage as an HCLI unit, force every GPU-authority species SLEEPING
(this partition has no GPU and never will), refuse a species whose input
receipt is absent (named, never a default), refuse PROTECTED_AB as runnable
under LIGHT or worse contamination, and answer next_species() from the live
qualification queue.

It also freezes today's hand-composed procedure as reusable WorkUnit
species — ROOF_PROBE, PRODUCTION_GAP_ATTRIBUTION, ADDRESSING_AUDIT,
GEOMETRY_SWEEP, CEREMONY_AUDIT, PARITY_PROOF, COMPLETE_TOKEN_REPROFILE —
so a future Odyssey model receives that treatment without a human composing
it. Scars are refusals. After any large win, reprofile immediately: a
removed denominator exposes another, and the old decomposition stops being
valid.

This library EMITS WorkUnits. It does not execute them. A species that
pretends it can run here is worse than an absent one. next_species() never
returns an empty list: when nothing is runnable it returns a reason.

    python3 tools/future/accelerator_workunits.py --build
    python3 tools/future/accelerator_workunits.py --next
    python3 tools/future/accelerator_workunits.py --chain
    python3 tools/future/accelerator_workunits.py --trigger
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, git, write_receipt
from tools.future import candidate_planner as cp
from tools.future import codex_behaviors as cb
from tools.future import contamination as C
from tools.future import protected_window as pw
from tools.future import qwen27_profile_schema as qps
from tools.future import workunit_species as ws

RECEIPT = "ACCELERATOR_WORKUNITS.json"
SCHEMA = "hawking.future.accelerator_workunits.v1"
RECORDED_BY = "tools/future/accelerator_workunits.py"

HANDOFF_REL = cb.HANDOFF_REL
QUEUE_REL = cp.QUEUE_REL.as_posix()
SCHEMA_REL = f"receipts/future/{qps.RECEIPT}"
STAGED_REL = f"receipts/future/{cp.RECEIPT}"
PREFLIGHT_REL = "receipts/future/STATIC_KERNEL_PREFLIGHT.json"
SCOREBOARD_REL = "receipts/headless/ACCELERATOR_SCOREBOARD.json"
LAW_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"
SCAR_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
ATTACK_REL = "receipts/future/ODYSSEY3_ADVERSARY.json"

# Procedure species. Today's hand-composed treatment, frozen as HCLI units.
ROOF_ANCHOR_REL = "receipts/future/ROOF_ANCHOR.json"
ATTRIBUTION_REL = "receipts/future/ADDRESSING_ATTRIBUTION.json"
STREAM_COUNT_REL = "receipts/future/MLP_STREAM_COUNT.json"
ISSUE_LADDER_REL = "receipts/future/MLP_ISSUE_RATE_LADDER.json"
WALL_GPU_REL = "receipts/future/WALL_GPU_RECONCILIATION.json"
FOLD_ADDQX_REL = "receipts/future/FOLD_ADDQX_AB.json"
DELTANET_WIDEN_REL = "receipts/future/DELTANET_WIDEN_AB.json"
PATH_TO_71_REL = "receipts/future/PATH_TO_71.json"

PROCEDURE_SPECIES: tuple[str, ...] = (
    "ROOF_PROBE",
    "PRODUCTION_GAP_ATTRIBUTION",
    "ADDRESSING_AUDIT",
    "GEOMETRY_SWEEP",
    "CEREMONY_AUDIT",
    "PARITY_PROOF",
    "COMPLETE_TOKEN_REPROFILE",
)

# Ordering a future Odyssey model walks without a human composing it.
PROCEDURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ROOF_PROBE": (),
    "PRODUCTION_GAP_ATTRIBUTION": ("ROOF_PROBE",),
    "ADDRESSING_AUDIT": ("PRODUCTION_GAP_ATTRIBUTION",),
    "GEOMETRY_SWEEP": ("ADDRESSING_AUDIT",),
    "CEREMONY_AUDIT": ("ROOF_PROBE",),
    "PARITY_PROOF": ("GEOMETRY_SWEEP", "CEREMONY_AUDIT"),
    "COMPLETE_TOKEN_REPROFILE": ("PARITY_PROOF",),
}

GEOMETRY_DISCRIMINATORS: tuple[str, ...] = (
    "dependency",
    "register_pressure",
    "occupancy",
)

BOUNDED_TOO_SMALL = "BOUNDED_TOO_SMALL"
LARGE_WIN_THRESHOLD_MS = 1.0
CEREMONY_HOST_GAP_BOUND_MS = 1.0
PROCEDURE_UNIT_PREFIX = "hcli.metabolism."

# Cited from landed receipts. This module does not re-measure.
# FOLD_ADDQX_AB.saving.complete_token_saving_ms / complete_token.incumbent_ms
CITED_FOLD_ADDQX_COMPLETE_SAVING_MS = 3.9833
CITED_FOLD_ADDQX_INCUMBENT_MS = 26.3026
CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS = 1.745
# PATH_TO_71.baseline.token_ms — stale against the 26.3026 incumbent
CITED_PATH_TO_71_TOKEN_MS = 28.722
# DELTANET_WIDEN_AB.saving.{isolated_fair_cut_ms, complete_token_saving_ms}
CITED_WIDEN_ISOLATED_MS = 0.7046
CITED_WIDEN_COMPLETE_MS = 1.0245
# WALL_GPU_RECONCILIATION.derived.host_gap_ms_per_token
CITED_HOST_GAP_MS = 0.9894
# ROOF_ANCHOR.recommended_anchor.value_gb_s (two independent legs WITH activation)
CITED_ARM_A_ROOF_GB_S = 497.4
# ADDRESSING_ATTRIBUTION: addr-probe without an input-vector load
CITED_ADDR_PROBE_GB_S = 703.5
# MLP_ISSUE_RATE_LADDER.judgement.{ilp.ratio_8_over_1, register_pressure.ratio_ws32_over_ws0}
CITED_DEPENDENCY_ILP_RATIO = 1.062
CITED_REGISTER_PRESSURE_RATIO = 1.078
# FOLD_ADDQX_AB.layer0_byte_compare.gate — token ids identical, intermediates not
CITED_GATE_MISMATCH_BYTES = 22309
CITED_GATE_COMPARED_BYTES = 69632

# Specs the scheduler emits. Science lives here; scars are the refusals, not comments.
PROCEDURE_SPECS: dict[str, dict[str, Any]] = {
    "ROOF_PROBE": {
        "id": "ROOF_PROBE",
        "title": "Probe the production-shaped roof WITH the activation load",
        "description": (
            "Anchor the streaming roof at the production-shaped kernel that "
            "loads the activation. Two independent legs (MLP ARM A and the LM "
            f"head) corroborate {CITED_ARM_A_ROOF_GB_S} GB/s. A ceiling must "
            "name its roof_id. Landed example: ROOF_ANCHOR.json."
        ),
        "resource_class": "GPU_DIRTY_OK",
        "verifier": "future.accelerator.roof_probe",
        "input_receipts": (
            "receipts/future/MLP_ALU_ROOFLINE.json",
            "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json",
        ),
        "output_receipt": ROOF_ANCHOR_REL,
        "evidence_parents": (ROOF_ANCHOR_REL, "receipts/future/MLP_ALU_ROOFLINE.json"),
        "acceptance": (
            "a ceiling that names roof_id from the registry and loads the "
            "activation; two independent legs agree"
        ),
        "refusals": (
            "unstated_roof",
            "ceiling_without_named_roof_id",
            "promoted_family_scoring_reference_as_machine_roof",
        ),
        "scar": (
            "a ceiling with an unstated roof produced 595.9, and 589.73 was a "
            "second instance"
        ),
        "example_receipts": (ROOF_ANCHOR_REL,),
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "emit_static_plan",
            "compile_experiment_spec",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when a named production-shaped-with-activation roof is "
            "anchored. This species does not promote. A ceiling with no "
            "roof_id is refused (unstated-roof scar: 595.9, then 589.73)."
        ),
    },
    "PRODUCTION_GAP_ATTRIBUTION": {
        "id": "PRODUCTION_GAP_ATTRIBUTION",
        "title": "Attribute the production gap, each rung sourced",
        "description": (
            "Attribute where the addr-probe figure becomes production-effective, "
            "each rung named with source receipt, statistic, and whether it "
            "loads the activation. Landed example: ADDRESSING_ATTRIBUTION.json "
            f"(703-to-337 chain). The {CITED_ADDR_PROBE_GB_S} GB/s addr-probe "
            "never loads the activation and is not a production-decode ceiling."
        ),
        "resource_class": "STATIC_ANALYSIS",
        "verifier": "future.accelerator.production_gap_attribution",
        "input_receipts": (ROOF_ANCHOR_REL, ATTRIBUTION_REL),
        "output_receipt": ATTRIBUTION_REL,
        "evidence_parents": (ROOF_ANCHOR_REL, ATTRIBUTION_REL),
        "acceptance": (
            "a sourced rung chain; every rung used as a production-decode "
            "comparable loads the activation"
        ),
        "refusals": (
            "addr_probe_as_production_ceiling",
            "rung_without_source_receipt",
            "activation_not_loaded_compared_as_decode",
        ),
        "scar": f"{CITED_ADDR_PROBE_GB_S} never loads the activation",
        "example_receipts": (ATTRIBUTION_REL,),
        "bounded_authority": (
            "read_receipts",
            "run_static_analysis",
            "audit_dependency_chain",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when every rung of the gap is sourced and no addr-probe is "
            "treated as a production-decode ceiling. This species does not promote."
        ),
    },
    "ADDRESSING_AUDIT": {
        "id": "ADDRESSING_AUDIT",
        "title": "Audit stream count at fixed bytes/thread",
        "description": (
            "Hold bytes/thread-iteration fixed and vary stream count. Stream "
            "count is REFUTED at that hold; merging further HURTS. Landed "
            "example: MLP_STREAM_COUNT.json (38 B/iter held; pack_6_32 and "
            "pack_38 slower than mlp_2_2_2_32)."
        ),
        "resource_class": "GPU_DIRTY_OK",
        "verifier": "future.accelerator.addressing_audit",
        "input_receipts": (ATTRIBUTION_REL, STREAM_COUNT_REL),
        "output_receipt": STREAM_COUNT_REL,
        "evidence_parents": (STREAM_COUNT_REL, ATTRIBUTION_REL),
        "acceptance": (
            "a stream-count ladder at fixed bytes/thread whose verdict does "
            "not promote merging"
        ),
        "refusals": (
            "promote_stream_merge",
            "stream_count_bound_after_merge_hurts",
            "bytes_per_thread_not_held",
        ),
        "scar": "stream count REFUTED at fixed bytes/thread; merging further HURTS",
        "example_receipts": (STREAM_COUNT_REL,),
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "compile_experiment_spec",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when stream count is audited at fixed bytes/thread. Merging "
            "further is refused (it HURTS). This species does not promote."
        ),
    },
    "GEOMETRY_SWEEP": {
        "id": "GEOMETRY_SWEEP",
        "title": "Sweep issue-rate geometry and carry its discriminators",
        "description": (
            "Sweep ops/byte (two slopes with a stall) and carry the "
            "constant-op-count discriminators: NOT dependency "
            f"(ILP 8/1 = {CITED_DEPENDENCY_ILP_RATIO}), NOT register pressure "
            f"(ws32/ws0 = {CITED_REGISTER_PRESSURE_RATIO}), NOT occupancy "
            "(raising it is worse). A sweep without those discriminators is "
            "refused. Landed example: MLP_ISSUE_RATE_LADDER.json."
        ),
        "resource_class": "GPU_DIRTY_OK",
        "verifier": "future.accelerator.geometry_sweep",
        "input_receipts": (STREAM_COUNT_REL, ISSUE_LADDER_REL),
        "output_receipt": ISSUE_LADDER_REL,
        "evidence_parents": (ISSUE_LADDER_REL,),
        "acceptance": (
            "a geometry sweep that includes dependency, register_pressure, "
            "and occupancy discriminators"
        ),
        "refusals": (
            "sweep_without_discriminators",
            "missing_dependency_discriminator",
            "missing_register_pressure_discriminator",
            "missing_occupancy_discriminator",
        ),
        "scar": (
            f"two slopes with a stall; NOT dependency ({CITED_DEPENDENCY_ILP_RATIO}), "
            f"NOT register pressure ({CITED_REGISTER_PRESSURE_RATIO}), NOT occupancy "
            "(raising it is worse)"
        ),
        "example_receipts": (ISSUE_LADDER_REL,),
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "compile_experiment_spec",
            "rank_falsifiable_experiments",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when the sweep AND its three discriminators are on the "
            "receipt. A sweep-only result is refused. This species does not promote."
        ),
    },
    "CEREMONY_AUDIT": {
        "id": "CEREMONY_AUDIT",
        "title": "Bound the host ceremony class and stop if too small",
        "description": (
            "Bound the whole host class (CPU submission, command buffers, "
            "readbacks, sync, state movement, host transforms, reporting, "
            "allocation, lock contention). Landed example: "
            "WALL_GPU_RECONCILIATION.json bounds it at "
            f"{CITED_HOST_GAP_MS} ms. The species returns {BOUNDED_TOO_SMALL} "
            "and stops; continuing to hunt host ceremony is refused."
        ),
        "resource_class": "GPU_DIRTY_OK",
        "verifier": "future.accelerator.ceremony_audit",
        "input_receipts": (WALL_GPU_REL,),
        "output_receipt": WALL_GPU_REL,
        "evidence_parents": (WALL_GPU_REL,),
        "acceptance": (
            f"a named host-gap bound; verdict {BOUNDED_TOO_SMALL} is a legal stop"
        ),
        "refusals": (
            "continue_after_bounded_too_small",
            "unnamed_host_gap",
            "host_ceremony_as_unlock_after_bound",
        ),
        "scar": (
            f"the whole host class is bounded at {CITED_HOST_GAP_MS} ms; "
            f"deleting all of it is not the unlock"
        ),
        "example_receipts": (WALL_GPU_REL,),
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "compile_experiment_spec",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            f"stop when the host class is {BOUNDED_TOO_SMALL}. Continuing to "
            "hunt host ceremony after that bound is refused. This species "
            "does not promote."
        ),
    },
    "PARITY_PROOF": {
        "id": "PARITY_PROOF",
        "title": "Prove parity: token ids AND intermediate bytes",
        "description": (
            "Token-id equality is necessary and not sufficient. Parity is "
            "token ids AND a byte comparison of intermediates. Landed "
            "example: FOLD_ADDQX_AB.json — token ids identical, "
            f"{CITED_GATE_MISMATCH_BYTES} of {CITED_GATE_COMPARED_BYTES} "
            "intermediate bytes not. A result that reports token ids alone "
            "is refused. This species blocks promotion until accepted."
        ),
        "resource_class": "GPU_EXCLUSIVE",
        "verifier": "future.accelerator.parity_proof",
        "input_receipts": (FOLD_ADDQX_REL,),
        "output_receipt": FOLD_ADDQX_REL,
        "evidence_parents": (FOLD_ADDQX_REL, DELTANET_WIDEN_REL),
        "acceptance": (
            "token_ids_identical AND an intermediate-byte comparison; "
            "argmax agreement is not parity"
        ),
        "refusals": (
            "token_id_equality_alone",
            "argmax_as_parity",
            "cited_probe_as_identity_proof",
        ),
        "scar": (
            f"token ids identical, {CITED_GATE_MISMATCH_BYTES} of "
            f"{CITED_GATE_COMPARED_BYTES} intermediate bytes NOT"
        ),
        "example_receipts": (FOLD_ADDQX_REL,),
        "blocks_promotion_until_accepted": True,
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "compile_experiment_spec",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when token-id equality AND intermediate-byte identity are "
            "both reported. Token ids alone are refused. This species does "
            "not promote."
        ),
    },
    "COMPLETE_TOKEN_REPROFILE": {
        "id": "COMPLETE_TOKEN_REPROFILE",
        "title": "Reprofile the complete token after a win",
        "description": (
            "A probe is not a token. After any large win, reprofile the "
            "complete token immediately: a removed denominator exposes "
            "another, and the old decomposition stops being valid. Landed "
            "examples: DELTANET_WIDEN_AB.json "
            f"({CITED_WIDEN_ISOLATED_MS} became {CITED_WIDEN_COMPLETE_MS}) "
            "and FOLD_ADDQX_AB.json "
            f"({CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS} became "
            f"{CITED_FOLD_ADDQX_COMPLETE_SAVING_MS}). An isolated measurement "
            "is refused. PATH_TO_71 TOKEN_MS "
            f"{CITED_PATH_TO_71_TOKEN_MS} is stale against incumbent "
            f"{CITED_FOLD_ADDQX_INCUMBENT_MS}."
        ),
        "resource_class": "GPU_EXCLUSIVE",
        "verifier": "future.accelerator.complete_token_reprofile",
        "input_receipts": (FOLD_ADDQX_REL, PATH_TO_71_REL, DELTANET_WIDEN_REL),
        "output_receipt": FOLD_ADDQX_REL,
        "evidence_parents": (FOLD_ADDQX_REL, DELTANET_WIDEN_REL, PATH_TO_71_REL),
        "acceptance": (
            "a complete-token measurement of the new incumbent; isolated "
            "probes and projections are not this species"
        ),
        "refusals": (
            "isolated_measurement_only",
            "projection_as_token",
            "probe_as_token",
            "stale_baseline_after_large_win",
        ),
        "scar": (
            f"{CITED_WIDEN_ISOLATED_MS} became {CITED_WIDEN_COMPLETE_MS}, and "
            f"{CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS} became "
            f"{CITED_FOLD_ADDQX_COMPLETE_SAVING_MS}"
        ),
        "example_receipts": (DELTANET_WIDEN_REL, FOLD_ADDQX_REL),
        "bounded_authority": (
            "read_receipts",
            "propose_workunit",
            "compile_experiment_spec",
            "write_sidecar_receipt",
        ),
        "stop_condition": (
            "stop when the complete token has been re-profiled after the win. "
            "An isolated probe is refused (a probe is not a token). This "
            "species does not promote."
        ),
    },
}

# Closed loop the resident walks. Science for each id lives in codex_behaviors.
REQUIRED_SPECIES: tuple[str, ...] = (
    "PROFILE_COMPLETE_TOKEN",
    "PROFILE_REGION",
    "PROFILE_HOST_CEREMONY",
    "PROFILE_ACTIVE_BYTES",
    "PROFILE_DISPATCH",
    "PROFILE_SYNC",
    "FIND_TALLEST_COST",
    "GENERATE_KERNEL_CANDIDATE",
    "GENERATE_FUSION_CANDIDATE",
    "GENERATE_LAYOUT_CANDIDATE",
    "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
    "STATIC_KERNEL_VERIFY",
    "HOST_SHADER_ABI_VERIFY",
    "STRUCTURAL_COST_COMPARE",
    "DIAGNOSTIC_AB",
    "PROTECTED_AB",
    "REPROFILE_AFTER_WIN",
    "UPDATE_SCOREBOARD",
    "UPDATE_LAW",
    "UPDATE_SCAR",
    "TRANSFER_LAW",
    "ATTACK_LAW",
)

LOOP_WAVES: tuple[tuple[str, ...], ...] = (
    (
        "PROFILE_COMPLETE_TOKEN",
        "PROFILE_REGION",
        "PROFILE_HOST_CEREMONY",
        "PROFILE_ACTIVE_BYTES",
        "PROFILE_DISPATCH",
        "PROFILE_SYNC",
    ),
    ("FIND_TALLEST_COST",),
    (
        "GENERATE_KERNEL_CANDIDATE",
        "GENERATE_FUSION_CANDIDATE",
        "GENERATE_LAYOUT_CANDIDATE",
        "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
    ),
    ("STATIC_KERNEL_VERIFY", "HOST_SHADER_ABI_VERIFY", "STRUCTURAL_COST_COMPARE"),
    ("DIAGNOSTIC_AB",),
    ("PROTECTED_AB",),
    ("REPROFILE_AFTER_WIN",),
    ("UPDATE_SCOREBOARD", "UPDATE_LAW", "UPDATE_SCAR"),
    ("TRANSFER_LAW",),
    ("ATTACK_LAW",),
)

GPU_LANES = frozenset({"metal_gpu", "metal_compiler", "protected_lease", "diagnostic_ab", "flash_nx"})
GPU_RESOURCE = frozenset({"GPU_EXCLUSIVE", "GPU_DIRTY_OK", "GPU_DECODE"})
BLOCKED_CONTAMINATION = frozenset({"LIGHT", "HEAVY", "UNKNOWN"})
WIN_STATUSES = frozenset({"PROTECTED_PASS", "INTEGRATED"})

INPUT_RECEIPTS: dict[str, tuple[str, ...]] = {
    "PROFILE_COMPLETE_TOKEN": (HANDOFF_REL,),
    "PROFILE_REGION": (HANDOFF_REL,),
    "PROFILE_HOST_CEREMONY": (HANDOFF_REL,),
    "PROFILE_ACTIVE_BYTES": (HANDOFF_REL,),
    "PROFILE_DISPATCH": (HANDOFF_REL,),
    "PROFILE_SYNC": (HANDOFF_REL,),
    "FIND_TALLEST_COST": (HANDOFF_REL,),
    "GENERATE_KERNEL_CANDIDATE": (QUEUE_REL,),
    "GENERATE_FUSION_CANDIDATE": (QUEUE_REL,),
    "GENERATE_LAYOUT_CANDIDATE": (QUEUE_REL,),
    "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE": (QUEUE_REL,),
    "STATIC_KERNEL_VERIFY": (STAGED_REL,),
    "HOST_SHADER_ABI_VERIFY": (PREFLIGHT_REL,),
    "STRUCTURAL_COST_COMPARE": (HANDOFF_REL, STAGED_REL),
    "DIAGNOSTIC_AB": (QUEUE_REL,),
    "PROTECTED_AB": (QUEUE_REL,),
    "REPROFILE_AFTER_WIN": (QUEUE_REL,),
    "UPDATE_SCOREBOARD": (SCOREBOARD_REL,),
    "UPDATE_LAW": (LAW_REL,),
    "UPDATE_SCAR": (SCAR_REL,),
    "TRANSFER_LAW": (LAW_REL,),
    "ATTACK_LAW": (ATTACK_REL,),
}

OUTPUT_RECEIPTS: dict[str, str] = {
    "STATIC_KERNEL_VERIFY": PREFLIGHT_REL,
    "HOST_SHADER_ABI_VERIFY": "receipts/future/CLAUDE_SIDECAR_ABI_ADJUDICATION.json",
    "UPDATE_LAW": LAW_REL,
    "UPDATE_SCAR": SCAR_REL,
    "TRANSFER_LAW": LAW_REL,
    "ATTACK_LAW": ATTACK_REL,
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. GPU-authority species "
    "are emitted SLEEPING; this partition cannot take a lease or write a "
    "protected result. A missing input is a named refusal, never a pass. "
    "The procedure-species library EMITS HCLI WorkUnits; it does not execute "
    "them, take a GPU lease, or write a protected result. Every ms/GB/s "
    "figure below is copied from a named prior receipt (cited), never measured here."
)

DOES_NOT_EXECUTE = True
METABOLISM_CLAIM_BOUNDARY = CLAIM_BOUNDARY

SIDECAR_WAKE = (
    "SLEEPING until a distinct HCLI GPU_PROTECTED lane (not this sidecar) holds "
    "a proven lease on a Metal-capable GPU with the Metal compiler present. "
    "This sidecar partition has no GPU authority and never will. "
    f"protected_window.{pw.acquire_lease.__name__} raises rather than flock. "
    "A synthetic complete-token result is refused."
)


class InputRefused(ValueError):
    """A species was asked to emit without a named input receipt."""


class AcceleratorWorkunitError(ValueError):
    """Scheduler contract violation (GPU unit emitted runnable, empty next, …)."""


class ProcedureRefused(ValueError):
    """A metabolism species refused a result that repeats a campaign scar."""


class UnstatedRoofRefused(ProcedureRefused):
    """ROOF_PROBE scar: a ceiling with no named roof produced 595.9, then 589.73."""


class ActivationNotLoadedRefused(ProcedureRefused):
    """PRODUCTION_GAP_ATTRIBUTION scar: 703.5 never loads the activation."""


class StreamMergePromotionRefused(ProcedureRefused):
    """ADDRESSING_AUDIT scar: stream count REFUTED at fixed bytes/thread; merging further HURTS."""


class SweepWithoutDiscriminatorsRefused(ProcedureRefused):
    """GEOMETRY_SWEEP scar: a sweep without its discriminators."""


class CeremonyContinueRefused(ProcedureRefused):
    """CEREMONY_AUDIT scar: host class is BOUNDED_TOO_SMALL; stop."""


class TokenIdOnlyParityRefused(ProcedureRefused):
    """PARITY_PROOF scar: token ids identical, 22309 of 69632 intermediate bytes not."""


class IsolatedOnlyReprofileRefused(ProcedureRefused):
    """COMPLETE_TOKEN_REPROFILE scar: a probe is not a token."""


# ---------------------------------------------------------------------------
# Visibility. Missing in this sparse tree is not campaign-absence.
# ---------------------------------------------------------------------------


def _roots() -> list[Path]:
    return ws._checkout_roots()


def receipt_visible(rel: str, injected: Mapping[str, bool] | None = None) -> bool:
    """Disk/git presence, with an injected override so tests can watch a refusal."""
    if injected is not None and rel in injected:
        return bool(injected[rel])
    for root in _roots():
        path = root / rel
        if path.is_file():
            return True
    blob = git("show", f"HEAD:{rel}")
    return bool(blob)


def missing_inputs(species_id: str, injected: Mapping[str, bool] | None = None) -> list[str]:
    return [rel for rel in input_receipts_for(species_id) if not receipt_visible(rel, injected)]


def input_receipts_for(species_id: str) -> tuple[str, ...]:
    if species_id in INPUT_RECEIPTS:
        return INPUT_RECEIPTS[species_id]
    spec = PROCEDURE_SPECS.get(species_id)
    if spec is not None:
        return tuple(spec["input_receipts"])
    if species_id in cb.SPECIES_IDS:
        return (HANDOFF_REL,)
    raise InputRefused(f"{species_id} refuses: unknown species")


def output_receipt_for(species_id: str) -> str:
    return OUTPUT_RECEIPTS.get(species_id, f"receipts/future/{RECEIPT}")


# ---------------------------------------------------------------------------
# Catalog overlay. Science stays in codex_behaviors; we add the scheduler contract.
# ---------------------------------------------------------------------------


def needs_gpu_authority(spec: Mapping[str, Any]) -> bool:
    lane = str((spec.get("resources") or {}).get("lane") or spec.get("resource_lane") or "")
    rc = str(spec.get("resource_class") or "")
    return lane in GPU_LANES or rc in GPU_RESOURCE


def _fails_closed(species_id: str, *, gpu: bool) -> str:
    named = ", ".join(input_receipts_for(species_id))
    if species_id == "PROTECTED_AB":
        return (
            f"refuses if {QUEUE_REL} is absent (named); never pending under "
            "LIGHT/HEAVY/UNKNOWN contamination; never pending on this sidecar "
            f"even if QUIESCENT; {pw.acquire_lease.__name__} raises rather than flock"
        )
    if gpu:
        return (
            f"refuses if an input receipt is absent (named: {named}); "
            "emitted SLEEPING with a wake condition, never pending, never FAILED"
        )
    return (
        f"refuses if an input receipt is absent (named: {named}); "
        "does not invent a hardware number; does not promote"
    )


def contracts(*, handoff: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Scheduler contract per required species, grounded in the existing catalog."""
    doc = handoff
    if doc is None:
        doc, _src = cb.load_handoff()
    if doc is None:
        raise InputRefused(
            f"REQUIRED_SPECIES refuse: missing receipt {HANDOFF_REL}"
        )
    cat = cb.catalog_by_id(handoff=doc)
    missing = [sid for sid in REQUIRED_SPECIES if sid not in cat]
    if missing:
        raise AcceleratorWorkunitError(
            f"codex_behaviors catalog is missing required species {missing}"
        )
    out: dict[str, dict[str, Any]] = {}
    for sid in REQUIRED_SPECIES:
        spec = cat[sid]
        gpu = needs_gpu_authority(spec)
        lane = str((spec.get("resources") or {}).get("lane") or "static")
        out[sid] = {
            "id": sid,
            "title": spec.get("title"),
            "lane": lane,
            "resource_class": spec.get("resource_class"),
            "gpu_authority_required": gpu,
            "gpu_authority": False,
            "input_receipts": list(input_receipts_for(sid)),
            "output_receipt": output_receipt_for(sid),
            "verifier": spec.get("verifier"),
            "fails_closed": _fails_closed(sid, gpu=gpu),
            "profile_columns": list(qps.REQUIRED_METRICS) if sid.startswith("PROFILE_") or sid == "FIND_TALLEST_COST" else [],
            "extends": spec.get("extends_parent"),
            "loose_parent": spec.get("loose_parent"),
            "evidence_class": "STATIC_ONLY",
        }
    return out


# ---------------------------------------------------------------------------
# Emit. GPU species are SLEEPING on this sidecar, full stop.
# ---------------------------------------------------------------------------


def sidecar_wake(
    spec: Mapping[str, Any],
    *,
    contamination_class: str,
    blockers: Sequence[str],
) -> str:
    lane = str((spec.get("resources") or {}).get("lane") or "static")
    parts = [SIDECAR_WAKE]
    if blockers:
        parts.append(cb.wake_condition_for(lane, blockers))
    if str(spec.get("id")) == "PROTECTED_AB" and contamination_class in BLOCKED_CONTAMINATION:
        parts.append(
            f"PROTECTED_AB is not runnable under contamination_class={contamination_class} "
            "(LIGHT or worse); wake also requires QUIESCENT."
        )
    return " ".join(parts)


def _contamination_class(injected: str | None) -> tuple[str, str]:
    if injected is not None:
        klass = str(injected).strip().upper()
        if klass not in C.CONTAMINATION_CLASSES:
            raise InputRefused(
                f"PROTECTED_AB refuses: unknown contamination_class {injected!r}"
            )
        return klass, "injected"
    try:
        row = C.classify_contamination(C.snapshot())
        return str(row["contamination_class"]), str(row["contamination_reason"])
    except Exception as exc:
        return (
            "UNKNOWN",
            f"contamination snapshot failed ({type(exc).__name__}: {exc}); "
            "UNKNOWN never QUIESCENT",
        )


def emit_species(
    species_id: str,
    *,
    handoff: Mapping[str, Any] | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    contamination_class: str | None = None,
    blockers: Sequence[str] | None = None,
    cycle: int = 0,
) -> dict[str, Any]:
    """Emit one loop species as an HCLI unit. GPU authority → SLEEPING. Missing input → raise."""
    sid = str(species_id)
    if sid in PROCEDURE_SPECIES:
        return emit_procedure_unit(sid, cycle=int(cycle))
    if sid not in REQUIRED_SPECIES and sid not in cb.SPECIES_IDS:
        raise InputRefused(f"{sid} refuses: unknown species")

    doc = handoff
    src = "argument"
    if doc is None:
        doc, src = cb.load_handoff()
    if doc is None:
        raise InputRefused(f"{sid} refuses: missing receipt {HANDOFF_REL}")

    missing = missing_inputs(sid, receipts_visible)
    if missing:
        raise InputRefused(f"{sid} refuses: missing receipt {missing[0]}")

    klass, klass_src = _contamination_class(contamination_class)
    cat = cb.catalog_by_id(handoff=doc)
    if sid not in cat:
        raise InputRefused(f"{sid} refuses: unknown species")
    spec = cat[sid]
    gpu = needs_gpu_authority(spec)
    observed = list(blockers) if blockers is not None else cb.blockers_from_handoff(doc)

    unit = cb.emit_species_unit(
        spec,
        cycle=int(cycle),
        dependencies=[],
        blockers=observed,
    )
    # Sidecar host constraint: GPU species NEVER pending here, even if the
    # handoff blocker list is empty. codex_behaviors derives sleep from
    # blockers; this module derives it from the partition having no GPU.
    unit["gpu_authority"] = False
    unit["gpu_authority_required"] = gpu
    unit["input_receipts"] = list(input_receipts_for(sid))
    unit["output_receipt"] = output_receipt_for(sid)
    unit["fails_closed"] = _fails_closed(sid, gpu=gpu)
    unit["contamination_class"] = klass
    unit["contamination_source"] = klass_src
    unit["handoff_loaded_from"] = src
    unit["profile_columns"] = list(qps.REQUIRED_METRICS) if sid.startswith("PROFILE_") or sid == "FIND_TALLEST_COST" else []
    unit["claim_boundary"] = CLAIM_BOUNDARY
    unit["evidence_class"] = "STATIC_ONLY"
    unit["measurement_class"] = "STATIC_ONLY"
    unit["bench_state"] = "UNKNOWN"

    if gpu:
        wake = sidecar_wake(spec, contamination_class=klass, blockers=observed)
        unit["status"] = cb.STATUS_SLEEPING
        unit["classification"] = cb.CLASS_SLEEPING
        unit["wake_condition"] = wake
        unit["blocked_reason"] = wake
        unit["runnable"] = False
        unit["requires_quiescence"] = True
    else:
        unit["runnable"] = unit.get("status") == "pending"
        unit.setdefault("wake_condition", None)

    if sid == "PROTECTED_AB":
        # LIGHT or worse cannot be runnable. QUIESCENT still cannot: no GPU.
        unit["status"] = cb.STATUS_SLEEPING
        unit["classification"] = cb.CLASS_SLEEPING
        unit["runnable"] = False
        if not unit.get("wake_condition"):
            unit["wake_condition"] = sidecar_wake(spec, contamination_class=klass, blockers=observed)
            unit["blocked_reason"] = unit["wake_condition"]

    if gpu and unit.get("status") != cb.STATUS_SLEEPING:
        raise AcceleratorWorkunitError(
            f"{sid} needs GPU authority but was emitted status={unit.get('status')!r}"
        )
    if unit.get("runnable") is True and gpu:
        raise AcceleratorWorkunitError(f"{sid} was emitted runnable; sidecar has no GPU")
    if str(unit.get("status") or "").lower() in {"failed", "skipped"}:
        raise AcceleratorWorkunitError(f"{sid} emitted {unit.get('status')}; must be SLEEPING or pending")
    ws.validate_emitted_unit(unit)
    return unit


def emit_loop(
    *,
    handoff: Mapping[str, Any] | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    contamination_class: str | None = None,
    blockers: Sequence[str] | None = None,
    cycle: int = 0,
) -> dict[str, Any]:
    """Emit every required loop species. Per-species input refusal is recorded, not rounded into a pass."""
    units: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for sid in REQUIRED_SPECIES:
        try:
            units.append(
                emit_species(
                    sid,
                    handoff=handoff,
                    receipts_visible=receipts_visible,
                    contamination_class=contamination_class,
                    blockers=blockers,
                    cycle=cycle,
                )
            )
        except InputRefused as exc:
            msg = str(exc)
            named = msg.split("missing receipt ", 1)[-1] if "missing receipt " in msg else None
            refusals.append(
                {
                    "species": sid,
                    "reason": msg,
                    "missing_receipt": named,
                    "runnable": False,
                }
            )
    gpu_runnable = [
        u["species"]
        for u in units
        if u.get("gpu_authority_required") and u.get("runnable") is True
    ]
    if gpu_runnable:
        raise AcceleratorWorkunitError(f"GPU species emitted runnable: {gpu_runnable}")
    return {
        "units": units,
        "refusals": refusals,
        "n_emitted": len(units),
        "n_refused": len(refusals),
        "n_sleeping": sum(1 for u in units if u.get("status") == cb.STATUS_SLEEPING),
        "n_pending": sum(1 for u in units if u.get("status") == "pending"),
        "n_gpu_sleeping": sum(
            1 for u in units if u.get("gpu_authority_required") and u.get("status") == cb.STATUS_SLEEPING
        ),
    }


# ---------------------------------------------------------------------------
# next_species — derived from the real candidate queue, never an empty list.
# ---------------------------------------------------------------------------


def load_queue_or_summary() -> tuple[dict[str, Any] | None, str]:
    """Full physical queue if visible; else the handoff's current_queue summary."""
    try:
        queue = cp.load_queue()
    except cp.QueueNotFoundError as exc:
        not_found = str(exc)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{QUEUE_REL}:{exc}"
    else:
        return queue, str(queue.get("_loaded_from") or QUEUE_REL)
    handoff, src = cb.load_handoff()
    summary = (handoff or {}).get("current_queue") if isinstance(handoff, dict) else None
    if isinstance(summary, dict):
        row = dict(summary)
        row["_loaded_from"] = f"{src}:current_queue"
        return row, row["_loaded_from"]
    return None, f"unseen_in_this_checkout after QueueNotFoundError: {not_found}"


def _ids_with_status(queue: Mapping[str, Any], status: str) -> list[str]:
    key = {
        "READY_PROTECTED": "ready_candidate_ids",
        "READY_DIAGNOSTIC": "ready_diagnostic_ids",
        "STATIC_ONLY": "static_only_candidate_ids",
        "BLOCKED": "blocked_candidate_ids",
    }.get(status)
    named = [str(x) for x in (queue.get(key) or [])] if key else []
    rows = queue.get("candidates")
    if isinstance(rows, list):
        from_rows = [
            cp.cid(r) for r in rows if str(r.get("status") or "") == status and r.get("candidate_id")
        ]
        return from_rows or named
    return named


def queue_census(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and identity sets. Empty is a first-class answer, not a missing key."""
    raw_counts = queue.get("status_counts")
    counts: dict[str, int] = {}
    if isinstance(raw_counts, Mapping):
        counts = {str(k): int(v) for k, v in raw_counts.items()}
    rows = queue.get("candidates")
    if isinstance(rows, list):
        derived: dict[str, int] = {}
        for row in rows:
            derived[str(row.get("status") or "")] = derived.get(str(row.get("status") or ""), 0) + 1
        if derived:
            counts = derived
        n = len(rows)
    elif "total_candidates" in queue:
        n = int(queue["total_candidates"] or 0)
    else:
        n = sum(counts.values())
    ready_p = _ids_with_status(queue, "READY_PROTECTED")
    ready_d = _ids_with_status(queue, "READY_DIAGNOSTIC")
    static = _ids_with_status(queue, "STATIC_ONLY")
    blocked = _ids_with_status(queue, "BLOCKED")
    wins = _ids_with_status(queue, "PROTECTED_PASS") + _ids_with_status(queue, "INTEGRATED")
    if not wins:
        wins = []
        if isinstance(rows, list):
            wins = [cp.cid(r) for r in rows if str(r.get("status") or "") in WIN_STATUSES]
    return {
        "n_candidates": int(n),
        "status_counts": dict(sorted(counts.items())),
        "ready_protected_ids": ready_p,
        "ready_diagnostic_ids": ready_d,
        "static_only_ids": static,
        "blocked_ids": blocked,
        "win_ids": wins,
        "n_ready_protected": len(ready_p) if ready_p else int(counts.get("READY_PROTECTED") or 0),
        "n_ready_diagnostic": len(ready_d) if ready_d else int(counts.get("READY_DIAGNOSTIC") or 0),
        "n_static_only": len(static) if static else int(counts.get("STATIC_ONLY") or 0),
        "n_blocked": len(blocked) if blocked else int(counts.get("BLOCKED") or 0),
        "n_wins": len(wins) if wins else int(counts.get("PROTECTED_PASS") or 0) + int(counts.get("INTEGRATED") or 0),
        "loaded_from": queue.get("_loaded_from"),
    }


def _generate_species_for(candidate_id: str) -> str:
    name = candidate_id.lower()
    if "fusion" in name:
        return "GENERATE_FUSION_CANDIDATE"
    if "pipeline" in name or "commit-timing" in name or "encoder-label" in name:
        return "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE"
    if "splitk" in name or "vecgroup" in name or name.endswith("-vec"):
        return "GENERATE_LAYOUT_CANDIDATE"
    return "GENERATE_KERNEL_CANDIDATE"


def _answer(
    *,
    species: str | None,
    runnable: bool,
    status: str,
    reason: str,
    census: Mapping[str, Any] | None,
    contamination_class: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(reason or "").strip():
        raise AcceleratorWorkunitError("next_species produced no reason")
    body: dict[str, Any] = {
        "species": species,
        "runnable": bool(runnable) and species is not None,
        "status": status,
        "reason": str(reason),
        "gpu_authority": False,
        "contamination_class": contamination_class,
        "queue": dict(census) if census else None,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if extra:
        body.update(dict(extra))
    return body


def next_species(
    queue: Mapping[str, Any] | None = None,
    *,
    contamination_class: str | None = None,
    receipts_visible: Mapping[str, bool] | None = None,
    handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the resident would do next on the Accelerator right now.

    Always a dict with a reason. Never []. An empty queue is a named refusal
    to invent work, not a silent no-op.
    """
    klass, klass_src = _contamination_class(contamination_class)
    src = "argument"
    loaded = queue
    if loaded is None:
        loaded, src = load_queue_or_summary()
    if loaded is None:
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                f"nothing runnable: missing receipt {QUEUE_REL} "
                f"(looked at checkout roots and git HEAD; loaded_from={src})"
            ),
            census=None,
            contamination_class=klass,
            extra={"contamination_source": klass_src, "missing_receipt": QUEUE_REL},
        )
    if not isinstance(loaded, Mapping):
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason="nothing runnable: qualification queue is not a mapping",
            census=None,
            contamination_class=klass,
        )

    census = queue_census(loaded)
    census["loaded_from"] = loaded.get("_loaded_from") or src
    n = int(census["n_candidates"])
    if n <= 0:
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                "nothing runnable: qualification queue has 0 candidates; "
                "the Accelerator loop has no patient. An empty list of species "
                "would hide this."
            ),
            census=census,
            contamination_class=klass,
            extra={"contamination_source": klass_src},
        )

    n_win = int(census["n_wins"])
    n_rp = int(census["n_ready_protected"])
    n_rd = int(census["n_ready_diagnostic"])
    n_st = int(census["n_static_only"])
    n_bl = int(census["n_blocked"])

    def _sleeping(sid: str, reason: str) -> dict[str, Any]:
        missing = missing_inputs(sid, receipts_visible)
        if missing:
            return _answer(
                species=sid,
                runnable=False,
                status="refused",
                reason=f"{sid} refuses: missing receipt {missing[0]}",
                census=census,
                contamination_class=klass,
                extra={"missing_receipt": missing[0], "contamination_source": klass_src},
            )
        return _answer(
            species=sid,
            runnable=False,
            status=cb.STATUS_SLEEPING,
            reason=reason,
            census=census,
            contamination_class=klass,
            extra={
                "wake_condition": SIDECAR_WAKE,
                "gpu_authority_required": True,
                "contamination_source": klass_src,
            },
        )

    def _cpu(sid: str, reason: str) -> dict[str, Any]:
        missing = missing_inputs(sid, receipts_visible)
        if missing:
            return _answer(
                species=sid,
                runnable=False,
                status="refused",
                reason=f"{sid} refuses: missing receipt {missing[0]}",
                census=census,
                contamination_class=klass,
                extra={"missing_receipt": missing[0], "contamination_source": klass_src},
            )
        return _answer(
            species=sid,
            runnable=True,
            status="pending",
            reason=reason,
            census=census,
            contamination_class=klass,
            extra={"gpu_authority_required": False, "contamination_source": klass_src},
        )

    if n_win > 0:
        return _sleeping(
            "REPROFILE_AFTER_WIN",
            (
                f"{n_win} PROTECTED_PASS/INTEGRATED identit"
                f"{'y' if n_win == 1 else 'ies'} on the queue; the loop must "
                "reprofile the new incumbent. REPROFILE_AFTER_WIN needs GPU "
                "authority this sidecar does not have."
            ),
        )
    if n_rp > 0:
        why = (
            f"{n_rp} READY_PROTECTED candidate(s) wait on a protected complete-token "
            f"AB. PROTECTED_AB is the Accelerator next step and is not runnable "
            f"here (gpu_authority=false"
        )
        if klass in BLOCKED_CONTAMINATION:
            why += f", contamination_class={klass} is LIGHT or worse"
        why += ")."
        return _sleeping("PROTECTED_AB", why)
    if n_rd > 0:
        return _sleeping(
            "DIAGNOSTIC_AB",
            (
                f"{n_rd} READY_DIAGNOSTIC candidate(s) wait on a diagnostic AB. "
                "DIAGNOSTIC_AB needs a GPU window this sidecar does not have; "
                "a diagnostic pass would not promote."
            ),
        )
    if n_st > 0:
        ident = (census["static_only_ids"] or ["static-only"])[0]
        sid = _generate_species_for(str(ident))
        return _cpu(
            sid,
            (
                f"{n_st} STATIC_ONLY candidate(s) are not in a GPU rung; "
                f"next CPU species is {sid} for {ident}. Spec only; no timing claim."
            ),
        )
    if n_bl > 0 and n_bl == n:
        sample = (census["blocked_ids"] or ["(unnamed)"])[0]
        rows = loaded.get("candidates") if isinstance(loaded.get("candidates"), list) else []
        blocked_reason = None
        for row in rows or []:
            if str(row.get("candidate_id")) == str(sample):
                blocked_reason = row.get("blocked_reason")
                break
        extra = f" sample blocked_reason={blocked_reason!r}" if blocked_reason else ""
        return _answer(
            species=None,
            runnable=False,
            status="refused",
            reason=(
                f"nothing runnable: all {n} candidates are BLOCKED "
                f"(sample={sample}).{extra} The sidecar will not invent a GPU "
                "cell for a candidate the queue itself refuses to run."
            ),
            census=census,
            contamination_class=klass,
            extra={"contamination_source": klass_src},
        )

    return _cpu(
        "FIND_TALLEST_COST",
        (
            f"queue has {n} candidate(s) and no READY_PROTECTED / READY_DIAGNOSTIC / "
            "STATIC_ONLY cursor; FIND_TALLEST_COST can rank AKB columns without "
            "inventing ns (UNKNOWN is a legal stop)."
        ),
    )


# ---------------------------------------------------------------------------
# Procedure species. EMITS WorkUnits. Does not execute them.
# ---------------------------------------------------------------------------


def _procedure_budget(*, gpu: bool) -> dict[str, Any]:
    return {
        "attempts": 3,
        "max_repair_depth": 3,
        "max_repairs_per_root": 6,
        "gpu_windows_requested": 1 if gpu else 0,
        "gpu_windows_held": 0,
        "wall_clock_s": None,
    }


def procedure_gpu_required(species_id: str) -> bool:
    spec = PROCEDURE_SPECS[species_id]
    return str(spec["resource_class"]) in GPU_RESOURCE


def define_procedure_species(species_id: str) -> dict[str, Any]:
    """Construct one procedure species through workunit_species.define_species."""
    sid = str(species_id)
    if sid not in PROCEDURE_SPECS:
        raise InputRefused(f"{sid} refuses: unknown species")
    spec = PROCEDURE_SPECS[sid]
    gpu = procedure_gpu_required(sid)
    row = ws.define_species(
        id=sid,
        title=str(spec["title"]),
        evidence_parents=tuple(spec["evidence_parents"]),
        bounded_authority=tuple(spec["bounded_authority"]),
        resource_class=str(spec["resource_class"]),
        verifier=str(spec["verifier"]),
        budget=_procedure_budget(gpu=gpu),
        stop_condition=str(spec["stop_condition"]),
        role="science",
        description=str(spec["description"]),
        effect_class="READ_ONLY",
        era="I",
        odyssey="I",
    )
    row["input_receipts"] = list(spec["input_receipts"])
    row["output_receipt"] = spec["output_receipt"]
    row["acceptance"] = spec["acceptance"]
    row["refusals"] = list(spec["refusals"])
    row["scar"] = spec["scar"]
    row["example_receipts"] = list(spec["example_receipts"])
    row["gpu_authority"] = False
    row["gpu_authority_required"] = gpu
    row["evidence_class"] = "STATIC_ONLY"
    row["does_not_execute"] = True
    row["executed"] = False
    row["blocks_promotion_until_accepted"] = bool(spec.get("blocks_promotion_until_accepted"))
    return row


def procedure_catalog() -> dict[str, dict[str, Any]]:
    """All seven species, each passed through the HCLI authority constructor."""
    return {sid: define_procedure_species(sid) for sid in PROCEDURE_SPECIES}


def procedure_unit_id(species_id: str) -> str:
    return f"{PROCEDURE_UNIT_PREFIX}{species_id}"


def emit_procedure_unit(species_id: str, *, cycle: int = 0) -> dict[str, Any]:
    """Emit one procedure species as a real HCLI WorkUnit. Does not run it."""
    sid = str(species_id)
    if sid not in PROCEDURE_SPECS:
        raise InputRefused(f"{sid} refuses: unknown species")
    spec = define_procedure_species(sid)
    gpu = bool(spec["gpu_authority_required"])
    deps = [procedure_unit_id(dep) for dep in PROCEDURE_DEPENDENCIES[sid]]
    status = cb.STATUS_SLEEPING if gpu else "pending"
    classification = cb.CLASS_SLEEPING if gpu else "STATIC_ONLY"
    extras: dict[str, Any] = {
        "species": sid,
        "cycle": int(cycle),
        "input_receipts": list(spec["input_receipts"]),
        "output_receipt": spec["output_receipt"],
        "output_receipt_path": spec["output_receipt"],
        "acceptance": spec["acceptance"],
        "refusals": list(spec["refusals"]),
        "scar": spec["scar"],
        "example_receipts": list(spec["example_receipts"]),
        "gpu_authority": False,
        "gpu_authority_required": gpu,
        "evidence_class": "STATIC_ONLY",
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "does_not_execute": True,
        "executed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "blocks_promotion_until_accepted": bool(spec["blocks_promotion_until_accepted"]),
        "requires_quiescence": gpu,
        "fails_closed": (
            f"refuses {list(spec['refusals'])}; scar: {spec['scar']}; "
            "emitting is not executing"
        ),
    }
    if gpu:
        extras["wake_condition"] = SIDECAR_WAKE
        extras["blocked_reason"] = SIDECAR_WAKE
        extras["runnable"] = False
    else:
        extras["wake_condition"] = None
        extras["blocked_reason"] = None
        extras["runnable"] = True
    row = ws.emit_hcli_workunit(
        id=procedure_unit_id(sid),
        role="science",
        description=str(spec["description"]),
        dependencies=deps,
        resource_class=str(spec["resource_class"]),
        verifier=str(spec["verifier"]),
        provider="future.accelerator_workunits",
        effect_class="READ_ONLY",
        status=status,
        classification=classification,
        extras=extras,
    )
    row["status"] = status
    row["classification"] = classification
    row["executed"] = False
    row["does_not_execute"] = True
    row["gpu_authority"] = False
    row["gpu_authority_required"] = gpu
    row["measurement_class"] = "STATIC_ONLY"
    row["bench_state"] = "UNKNOWN"
    row["claim_boundary"] = CLAIM_BOUNDARY
    row["runnable"] = False if gpu else True
    if gpu:
        row["wake_condition"] = SIDECAR_WAKE
        row["blocked_reason"] = SIDECAR_WAKE
    if gpu and row.get("status") != cb.STATUS_SLEEPING:
        raise AcceleratorWorkunitError(
            f"{sid} needs GPU authority but was emitted status={row.get('status')!r}"
        )
    if row.get("runnable") is True and gpu:
        raise AcceleratorWorkunitError(f"{sid} was emitted runnable; sidecar has no GPU")
    if str(row.get("status") or "").lower() in {"failed", "skipped"}:
        raise AcceleratorWorkunitError(f"{sid} emitted {row.get('status')}; must be SLEEPING or pending")
    ws.validate_emitted_unit(row)
    return row


# ---------------------------------------------------------------------------
# Scars as refusals. A result that repeats today's mistake raises.
# ---------------------------------------------------------------------------


def accept_roof_probe(result: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Refuse a ceiling that does not name its roof. Scar: 595.9, then 589.73."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    roof_id = body.get("roof_id")
    if roof_id is None or str(roof_id).strip() == "":
        raise UnstatedRoofRefused(
            "ROOF_PROBE refuses: a ceiling with an unstated roof — this is the "
            "defect that produced 595.9, and 589.73 was a second instance"
        )
    return {
        "accepted": True,
        "species": "ROOF_PROBE",
        "roof_id": str(roof_id),
        "named_roof": True,
        "cited_anchor_gb_s": CITED_ARM_A_ROOF_GB_S,
        "cited_from": ROOF_ANCHOR_REL,
    }


def accept_production_gap_attribution(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Refuse treating the 703.5 addr-probe as a production-decode ceiling."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    rungs = list(body.get("rungs") or [])
    if not rungs:
        rungs = [body]
    if body.get("treat_addr_probe_as_production") or body.get("production_roof_gb_s") in {
        CITED_ADDR_PROBE_GB_S,
        703.6,
        703,
    }:
        raise ActivationNotLoadedRefused(
            "PRODUCTION_GAP_ATTRIBUTION refuses: 703.5 never loads the "
            "activation; an addr-probe is not a production-decode ceiling"
        )
    for rung in rungs:
        used_as_ceiling = bool(
            rung.get("as_production_ceiling")
            or rung.get("comparable_to_production_decode")
            or rung.get("as_dram_roof")
        )
        loads = rung.get("loads_activation")
        if used_as_ceiling and loads is not True:
            raise ActivationNotLoadedRefused(
                "PRODUCTION_GAP_ATTRIBUTION refuses: 703.5 never loads the "
                "activation; an addr-probe is not a production-decode ceiling"
            )
        if loads is False and rung.get("compare_to_production_effective"):
            raise ActivationNotLoadedRefused(
                "PRODUCTION_GAP_ATTRIBUTION refuses: 703.5 never loads the "
                "activation; comparing it to production-effective 337 is different work"
            )
    return {
        "accepted": True,
        "species": "PRODUCTION_GAP_ATTRIBUTION",
        "rung_count": len(rungs),
        "addr_probe_is_not_production_ceiling": True,
        "cited_addr_probe_gb_s": CITED_ADDR_PROBE_GB_S,
        "cited_from": ATTRIBUTION_REL,
    }


def accept_addressing_audit(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Refuse promoting stream merge. Scar: REFUTED at fixed bytes/thread; merging HURTS."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    verdict = str(body.get("verdict") or "")
    promoting = bool(
        body.get("promote_stream_merge")
        or body.get("promote")
        or verdict in {"STREAM_COUNT_BOUND", "PACK_AND_MERGE"}
    )
    if promoting:
        raise StreamMergePromotionRefused(
            "ADDRESSING_AUDIT refuses: stream count is REFUTED at fixed "
            "bytes/thread; merging further HURTS"
        )
    return {
        "accepted": True,
        "species": "ADDRESSING_AUDIT",
        "stream_count_refuted": True,
        "merging_further_hurts": True,
        "bytes_per_thread_held": body.get("bytes_per_thread_iteration_held"),
        "cited_from": STREAM_COUNT_REL,
    }


def accept_geometry_sweep(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Refuse a sweep that does not carry its discriminators."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    discs = body.get("discriminators")
    if not isinstance(discs, Mapping) or body.get("sweep_only"):
        raise SweepWithoutDiscriminatorsRefused(
            "GEOMETRY_SWEEP refuses: a sweep without its discriminators "
            f"(NOT dependency {CITED_DEPENDENCY_ILP_RATIO}, NOT register "
            f"pressure {CITED_REGISTER_PRESSURE_RATIO}, NOT occupancy — "
            "raising it is worse)"
        )
    missing = [name for name in GEOMETRY_DISCRIMINATORS if name not in discs]
    if missing:
        raise SweepWithoutDiscriminatorsRefused(
            "GEOMETRY_SWEEP refuses: a sweep without its discriminators "
            f"(missing {missing}; NOT dependency {CITED_DEPENDENCY_ILP_RATIO}, "
            f"NOT register pressure {CITED_REGISTER_PRESSURE_RATIO}, NOT "
            "occupancy — raising it is worse)"
        )
    return {
        "accepted": True,
        "species": "GEOMETRY_SWEEP",
        "discriminators": dict(discs),
        "cited_dependency_ilp_ratio": CITED_DEPENDENCY_ILP_RATIO,
        "cited_register_pressure_ratio": CITED_REGISTER_PRESSURE_RATIO,
        "occupancy_raising_is_worse": True,
        "cited_from": ISSUE_LADDER_REL,
    }


def accept_ceremony_audit(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Return BOUNDED_TOO_SMALL and stop. Refuse continuing after that bound."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    gap = body.get("host_gap_ms")
    if gap is None:
        raise CeremonyContinueRefused(
            "CEREMONY_AUDIT refuses: host gap is unnamed; cannot bound the host class"
        )
    bound = float(body.get("bound_ms") if body.get("bound_ms") is not None else CEREMONY_HOST_GAP_BOUND_MS)
    gap_f = float(gap)
    too_small = gap_f <= bound
    if too_small and (
        body.get("continue_after_bound") or str(body.get("unlock") or "") == "host_ceremony"
    ):
        raise CeremonyContinueRefused(
            "CEREMONY_AUDIT refuses: host class is BOUNDED_TOO_SMALL "
            f"({gap_f} ms <= {bound} ms); stop. Deleting all of it is not the unlock."
        )
    if too_small:
        return {
            "accepted": True,
            "species": "CEREMONY_AUDIT",
            "verdict": BOUNDED_TOO_SMALL,
            "stop": True,
            "host_gap_ms": gap_f,
            "bound_ms": bound,
            "cited_host_gap_ms": CITED_HOST_GAP_MS,
            "cited_from": WALL_GPU_REL,
        }
    return {
        "accepted": True,
        "species": "CEREMONY_AUDIT",
        "verdict": "HOST_GAP_MATERIAL",
        "stop": False,
        "host_gap_ms": gap_f,
        "bound_ms": bound,
        "cited_from": WALL_GPU_REL,
    }


def accept_parity_proof(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Refuse token-id equality reported alone. Necessary, not sufficient."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    token_ids = body.get("token_ids_identical")
    has_bytes = any(
        body.get(k) is not None
        for k in (
            "n_mismatch_bytes",
            "n_bytes_compared",
            "arithmetic_exact",
            "byte_identity",
            "layer0_byte_compare",
            "intermediate_bytes_compared",
        )
    )
    basis = str(body.get("parity_basis") or "")
    token_ids_only = bool(
        body.get("token_ids_only")
        or (token_ids is True and not has_bytes)
        or (basis == "token_id_equality" and not has_bytes)
    )
    if token_ids_only:
        raise TokenIdOnlyParityRefused(
            "PARITY_PROOF refuses: token-id equality is necessary and not "
            "sufficient (scar: token ids identical, "
            f"{CITED_GATE_MISMATCH_BYTES} of {CITED_GATE_COMPARED_BYTES} "
            "intermediate bytes NOT)"
        )
    arithmetic_exact = body.get("arithmetic_exact")
    n_mismatch = body.get("n_mismatch_bytes")
    exact = arithmetic_exact is True or (n_mismatch == 0)
    parity = bool(token_ids) and bool(exact)
    return {
        "accepted": True,
        "species": "PARITY_PROOF",
        "parity": parity,
        "token_ids_identical": token_ids,
        "token_id_equality_is_not_sufficient": True,
        "blocks_promotion_until_accepted": True,
        "cited_mismatch_bytes": CITED_GATE_MISMATCH_BYTES,
        "cited_compared_bytes": CITED_GATE_COMPARED_BYTES,
        "cited_from": FOLD_ADDQX_REL,
    }


def accept_complete_token_reprofile(
    result: Mapping[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Refuse an isolated probe handed as a complete-token reprofile."""
    body: dict[str, Any] = dict(result or {})
    body.update(kwargs)
    kind = str(body.get("kind") or "")
    has_complete = any(
        body.get(k) is not None
        for k in ("complete_token_ms", "complete_token_saving_ms")
    ) or kind == "complete_token"
    isolated_only = bool(
        body.get("isolated_only")
        or kind in {"isolated", "probe", "projection"}
        or (
            (body.get("isolated_ms") is not None or body.get("projection_ms") is not None)
            and not has_complete
        )
        or not has_complete
    )
    if isolated_only:
        raise IsolatedOnlyReprofileRefused(
            "COMPLETE_TOKEN_REPROFILE refuses: a probe is not a token "
            f"({CITED_WIDEN_ISOLATED_MS} became {CITED_WIDEN_COMPLETE_MS}; "
            f"{CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS} became "
            f"{CITED_FOLD_ADDQX_COMPLETE_SAVING_MS})"
        )
    return {
        "accepted": True,
        "species": "COMPLETE_TOKEN_REPROFILE",
        "complete_token": True,
        "probe_is_not_a_token": True,
        "cited_widen_isolated_ms": CITED_WIDEN_ISOLATED_MS,
        "cited_widen_complete_ms": CITED_WIDEN_COMPLETE_MS,
        "cited_fold_isolated_projection_ms": CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS,
        "cited_fold_complete_ms": CITED_FOLD_ADDQX_COMPLETE_SAVING_MS,
        "cited_from": (DELTANET_WIDEN_REL, FOLD_ADDQX_REL),
    }


# ---------------------------------------------------------------------------
# Reprofile trigger. After any large win the old decomposition is invalid.
# ---------------------------------------------------------------------------


def cited_live_frontier() -> dict[str, Any]:
    """The live stale-baseline condition, cited from landed receipts."""
    return {
        "win_id": "fold_addqx",
        "win_ms": CITED_FOLD_ADDQX_COMPLETE_SAVING_MS,
        "win_source": {
            "receipt": FOLD_ADDQX_REL,
            "field": "saving.complete_token_saving_ms",
        },
        "baseline_token_ms": CITED_PATH_TO_71_TOKEN_MS,
        "baseline_source": {
            "receipt": PATH_TO_71_REL,
            "field": "baseline.token_ms",
        },
        "incumbent_token_ms": CITED_FOLD_ADDQX_INCUMBENT_MS,
        "incumbent_source": {
            "receipt": FOLD_ADDQX_REL,
            "field": "complete_token.incumbent_ms",
        },
        "isolated_projection_ms": CITED_FOLD_ADDQX_ISOLATED_PROJECTION_MS,
        "cited": True,
        "evidence_class": "STATIC_ONLY",
        "does_not_execute": True,
    }


def reprofile_trigger(
    *,
    win_ms: float,
    baseline_token_ms: float,
    incumbent_token_ms: float,
    threshold_ms: float = LARGE_WIN_THRESHOLD_MS,
    win_id: str | None = None,
) -> dict[str, Any]:
    """Fire when a large win leaves the old token decomposition stale.

    Rule: after any large win, reprofile immediately. A removed denominator
    exposes another, and the old decomposition stops being valid.
    """
    large = float(win_ms) >= float(threshold_ms)
    stale = float(baseline_token_ms) > float(incumbent_token_ms)
    fires = bool(large and stale)
    ident = win_id or "win"
    if fires:
        reason = (
            f"after a large win the old decomposition is invalid: {ident} "
            f"removed {win_ms} ms; baseline TOKEN_MS {baseline_token_ms} is "
            f"stale against incumbent {incumbent_token_ms}"
        )
    elif large:
        reason = (
            f"{ident} removed {win_ms} ms (>= {threshold_ms}) but baseline "
            f"{baseline_token_ms} is not stale against incumbent {incumbent_token_ms}"
        )
    else:
        reason = (
            f"{ident} removed {win_ms} ms, below large-win threshold {threshold_ms}"
        )
    return {
        "fires": fires,
        "reason": reason,
        "threshold_ms": float(threshold_ms),
        "win_id": ident,
        "win_ms": float(win_ms),
        "baseline_token_ms": float(baseline_token_ms),
        "incumbent_token_ms": float(incumbent_token_ms),
        "species_to_emit": "COMPLETE_TOKEN_REPROFILE" if fires else None,
        "old_decomposition_valid": not fires,
        "cited": True,
        "evidence_class": "STATIC_ONLY",
        "does_not_execute": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def live_reprofile_trigger() -> dict[str, Any]:
    """The trigger on what is true right now: fold_addqx vs stale PATH_TO_71."""
    live = cited_live_frontier()
    row = reprofile_trigger(
        win_ms=float(live["win_ms"]),
        baseline_token_ms=float(live["baseline_token_ms"]),
        incumbent_token_ms=float(live["incumbent_token_ms"]),
        win_id=str(live["win_id"]),
    )
    row["live_frontier"] = live
    return row


# ---------------------------------------------------------------------------
# Chain. The whole treatment, emitted, not executed.
# ---------------------------------------------------------------------------


def emit_procedure_chain(*, cycle: int = 0) -> dict[str, Any]:
    """Emit the seven-species chain for the live frontier. Does not run it."""
    trigger = live_reprofile_trigger()
    units = [emit_procedure_unit(sid, cycle=cycle) for sid in PROCEDURE_SPECIES]
    gpu_runnable = [
        u["species"] for u in units if u.get("gpu_authority_required") and u.get("runnable") is True
    ]
    if gpu_runnable:
        raise AcceleratorWorkunitError(f"GPU procedure species emitted runnable: {gpu_runnable}")
    executed = [u["species"] for u in units if u.get("executed") is True]
    if executed:
        raise AcceleratorWorkunitError(f"procedure species claimed execution: {executed}")
    by_id = {u["species"]: u["id"] for u in units}
    roof_id = by_id["ROOF_PROBE"]
    gap = next(u for u in units if u["species"] == "PRODUCTION_GAP_ATTRIBUTION")
    if roof_id not in (gap.get("dependencies") or []):
        raise AcceleratorWorkunitError("ROOF_PROBE must precede PRODUCTION_GAP_ATTRIBUTION")
    parity = next(u for u in units if u["species"] == "PARITY_PROOF")
    if not parity.get("blocks_promotion_until_accepted"):
        raise AcceleratorWorkunitError("PARITY_PROOF must block promotion until accepted")
    reprofile = next(u for u in units if u["species"] == "COMPLETE_TOKEN_REPROFILE")
    if by_id["PARITY_PROOF"] not in (reprofile.get("dependencies") or []):
        raise AcceleratorWorkunitError("COMPLETE_TOKEN_REPROFILE must follow PARITY_PROOF")
    for row in units:
        ws.validate_emitted_unit(row)
    return {
        "schema": "hawking.future.accelerator_workunits.procedure_chain.v1",
        "species": list(PROCEDURE_SPECIES),
        "dependencies": {k: list(v) for k, v in PROCEDURE_DEPENDENCIES.items()},
        "n_units": len(units),
        "units": units,
        "reprofile_trigger": trigger,
        "live_frontier": cited_live_frontier(),
        "does_not_execute": True,
        "executed": False,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
        "note": (
            "These are real HCLI WorkUnits. Emitting them is not executing them. "
            "GPU-class units are SLEEPING; this sidecar has no GPU and never will. "
            "PARITY_PROOF blocks promotion until accepted. COMPLETE_TOKEN_REPROFILE "
            "follows any accepted win; the live trigger fires because fold_addqx "
            f"removed {CITED_FOLD_ADDQX_COMPLETE_SAVING_MS} ms and PATH_TO_71 "
            f"TOKEN_MS {CITED_PATH_TO_71_TOKEN_MS} is stale against incumbent "
            f"{CITED_FOLD_ADDQX_INCUMBENT_MS}."
        ),
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    handoff, handoff_src = cb.load_handoff()
    if handoff is None:
        raise InputRefused(
            f"build refuses: missing receipt {HANDOFF_REL} "
            "(the loop cannot be scheduled without the training trace)"
        )
    klass, klass_src = _contamination_class(None)
    loop = emit_loop(handoff=handoff, contamination_class=klass)
    nxt = next_species(handoff=handoff, contamination_class=klass)
    table = contracts(handoff=handoff)
    gpu_ids = [sid for sid, row in table.items() if row["gpu_authority_required"]]
    gpu_emitted = [u for u in loop["units"] if u.get("gpu_authority_required")]
    if any(u.get("runnable") is True for u in gpu_emitted):
        raise AcceleratorWorkunitError("GPU species emitted runnable")
    if any(u.get("status") != cb.STATUS_SLEEPING for u in gpu_emitted):
        raise AcceleratorWorkunitError("GPU species emitted not SLEEPING")

    chain = emit_procedure_chain()
    trigger = chain["reprofile_trigger"]
    proc_table = procedure_catalog()
    proc_gpu = [u for u in chain["units"] if u.get("gpu_authority_required")]
    if any(u.get("runnable") is True for u in proc_gpu):
        raise AcceleratorWorkunitError("GPU procedure species emitted runnable")
    if any(u.get("executed") is True for u in chain["units"]):
        raise AcceleratorWorkunitError("procedure species claimed execution")

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Make Codex's profile → tallest cost → why → remove (information / "
            "bytes / FLOPs / intermediate / dispatch / sync / copy / ceremony) → "
            "A/B → integrate → reprofile loop schedulable by HCLI, so the next "
            "latency seam is found without a human asking for it. Freeze today's "
            "hand-composed procedure as reusable WorkUnit species so a future "
            "Odyssey model receives the treatment without a human composing it."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "does_not_execute": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "handoff": {
            "path": HANDOFF_REL,
            "loaded_from": handoff_src,
            "present": True,
        },
        "loop": {
            "waves": [list(w) for w in LOOP_WAVES],
            "required_species": list(REQUIRED_SPECIES),
            "rule": (
                "profile, rank the tallest denominator, ask why the cost exists, "
                "generate a candidate that eliminates work, verify statically, "
                "AB, ledger, transfer, attack; a PHYSICAL_WIN enqueues REPROFILE_AFTER_WIN. "
                "GPU stages sleep on this sidecar."
            ),
            "profile_columns": list(qps.REQUIRED_METRICS),
            "protected_window_stages": list(pw.WINDOW_STAGES),
        },
        "species": table,
        "emitted": {
            "n_emitted": loop["n_emitted"],
            "n_refused": loop["n_refused"],
            "n_sleeping": loop["n_sleeping"],
            "n_pending": loop["n_pending"],
            "n_gpu_sleeping": loop["n_gpu_sleeping"],
            "units": [
                {
                    "id": u["id"],
                    "species": u.get("species"),
                    "status": u.get("status"),
                    "runnable": u.get("runnable"),
                    "gpu_authority_required": u.get("gpu_authority_required"),
                    "lane": (u.get("resources") or {}).get("lane"),
                    "verifier": u.get("verifier"),
                    "input_receipts": u.get("input_receipts"),
                    "output_receipt": u.get("output_receipt"),
                    "wake_condition": u.get("wake_condition"),
                    "fails_closed": u.get("fails_closed"),
                    "contamination_class": u.get("contamination_class"),
                }
                for u in loop["units"]
            ],
            "refusals": loop["refusals"],
        },
        "next_species": nxt,
        "contamination": {
            "class": klass,
            "source": klass_src,
            "protected_ab_runnable": False,
            "rule": (
                "PROTECTED_AB is never runnable under LIGHT/HEAVY/UNKNOWN, and is "
                "never runnable on this sidecar even if QUIESCENT."
            ),
        },
        "gpu_species": gpu_ids,
        "procedure": {
            "does_not_execute": True,
            "executed": False,
            "rule": (
                "ROOF_PROBE before PRODUCTION_GAP_ATTRIBUTION; PARITY_PROOF "
                "before any promotion; COMPLETE_TOKEN_REPROFILE after any "
                "accepted win. After any large win, reprofile immediately."
            ),
            "species_ids": list(PROCEDURE_SPECIES),
            "dependencies": {k: list(v) for k, v in PROCEDURE_DEPENDENCIES.items()},
            "species": {
                sid: {
                    "id": row["id"],
                    "title": row["title"],
                    "resource_class": row["resource_class"],
                    "verifier": row["verifier"],
                    "input_receipts": row["input_receipts"],
                    "output_receipt": row["output_receipt"],
                    "acceptance": row["acceptance"],
                    "refusals": row["refusals"],
                    "scar": row["scar"],
                    "gpu_authority_required": row["gpu_authority_required"],
                    "gpu_authority": False,
                    "blocks_promotion_until_accepted": row["blocks_promotion_until_accepted"],
                    "evidence_class": "STATIC_ONLY",
                    "does_not_execute": True,
                }
                for sid, row in proc_table.items()
            },
            "chain": {
                "n_units": chain["n_units"],
                "note": chain["note"],
                "units": [
                    {
                        "id": u["id"],
                        "species": u.get("species"),
                        "status": u.get("status"),
                        "runnable": u.get("runnable"),
                        "executed": u.get("executed"),
                        "dependencies": u.get("dependencies"),
                        "resource_class": u.get("resource_class"),
                        "verifier": u.get("verifier"),
                        "input_receipts": u.get("input_receipts"),
                        "output_receipt": u.get("output_receipt"),
                        "acceptance": u.get("acceptance"),
                        "refusals": u.get("refusals"),
                        "scar": u.get("scar"),
                        "gpu_authority_required": u.get("gpu_authority_required"),
                        "blocks_promotion_until_accepted": u.get(
                            "blocks_promotion_until_accepted"
                        ),
                        "wake_condition": u.get("wake_condition"),
                    }
                    for u in chain["units"]
                ],
            },
            "reprofile_trigger": trigger,
            "live_frontier": chain["live_frontier"],
        },
        "recovered_implementation": [
            "tools/future/codex_behaviors.py — thirty grounded species, emit_species_unit, "
            "emit_cycle, wake_condition_for; this module does not fork the catalog",
            "tools/future/workunit_species.py — HCLI field set, define_species, emit_hcli_workunit",
            "tools/future/candidate_planner.py — live qualification queue + staged factorial plan",
            "tools/future/contamination.py — QUIESCENT/LIGHT/HEAVY/UNKNOWN; assert_promotable",
            "tools/future/protected_window.py — eviction envelope; acquire_lease raises",
            "tools/future/qwen27_profile_schema.py — REQUIRED_METRICS columns FIND_TALLEST ranks",
            f"{HANDOFF_REL} — training trace loaded from {handoff_src}",
            f"{ROOF_ANCHOR_REL} — named roof {CITED_ARM_A_ROOF_GB_S} with activation; unstated-roof scar",
            f"{ATTRIBUTION_REL} — 703-to-337 chain; {CITED_ADDR_PROBE_GB_S} never loads the activation",
            f"{STREAM_COUNT_REL} — stream count REFUTED at fixed bytes/thread; merging HURTS",
            f"{ISSUE_LADDER_REL} — discriminators, not just the sweep",
            f"{WALL_GPU_REL} — host class bounded at {CITED_HOST_GAP_MS} ms",
            f"{FOLD_ADDQX_REL} — token ids necessary not sufficient; complete token {CITED_FOLD_ADDQX_COMPLETE_SAVING_MS} ms",
            f"{DELTANET_WIDEN_REL} — isolated {CITED_WIDEN_ISOLATED_MS} became complete {CITED_WIDEN_COMPLETE_MS}",
            f"{PATH_TO_71_REL} — TOKEN_MS {CITED_PATH_TO_71_TOKEN_MS} stale against incumbent {CITED_FOLD_ADDQX_INCUMBENT_MS}",
        ],
        "gaps_closed": [
            "next_species() answers the Accelerator cursor from real queue state and "
            "returns a reason when nothing is runnable (never an empty list)",
            "GPU-authority species are emitted SLEEPING on this sidecar even if the "
            "handoff blocker list is empty (codex_behaviors derives sleep from blockers)",
            "a species whose input receipt is absent refuses, naming the receipt",
            "PROTECTED_AB is never emitted runnable under LIGHT or worse contamination",
            "seven procedure species validate through workunit_species.validate_emitted_unit",
            "scars are raising refusals (unstated roof, addr-probe without activation, "
            "stream-merge promotion, sweep without discriminators, ceremony continue "
            "after BOUNDED_TOO_SMALL, token-id-only parity, isolated-only reprofile)",
            "reprofile trigger fires on the live stale-baseline condition "
            f"(fold_addqx {CITED_FOLD_ADDQX_COMPLETE_SAVING_MS} ms; PATH_TO_71 "
            f"{CITED_PATH_TO_71_TOKEN_MS} vs incumbent {CITED_FOLD_ADDQX_INCUMBENT_MS})",
            "the procedure chain is emitted for the live frontier; emitting is not executing",
        ],
        "negative_findings": [
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "READY_PROTECTED candidates wait on a GPU window this partition must not seize",
            f"contamination_class={klass} ({klass_src}); PROTECTED_AB stays SLEEPING",
            "codex_behaviors.UPDATE_SCOREBOARD and ATTACK_LAW remain UNGROUNDED_FROM_HANDOFF; "
            "this module still emits them as species with their input-receipt gate",
            "orchestration.py BINDINGS is not writable from this lane; the module is "
            "resident-callable via --build/--next regardless",
            "this species library EMITS work units; it does not run them",
        ],
        "resident_callable": {
            "entry_point": "tools.future.accelerator_workunits.next_species()",
            "workunit": (
                "one CPU_ANALYSIS unit for next_species / emit_loop; GPU species "
                "are emitted SLEEPING and are not a work source on this host"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "InputRefused names the missing receipt; GPU units cannot be pending; "
                "next_species on an empty or absent queue returns a reason, not []"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--trigger", action="store_true")
    a = ap.parse_args()
    if a.next:
        print(json.dumps(next_species(), indent=1, sort_keys=True))
        return 0
    if a.trigger:
        print(json.dumps(live_reprofile_trigger(), indent=1, sort_keys=True))
        return 0
    if a.chain:
        chain = emit_procedure_chain()
        compact = {
            "does_not_execute": True,
            "species": chain["species"],
            "dependencies": chain["dependencies"],
            "reprofile_trigger": chain["reprofile_trigger"],
            "units": [
                {
                    "id": u["id"],
                    "species": u.get("species"),
                    "status": u.get("status"),
                    "runnable": u.get("runnable"),
                    "executed": u.get("executed"),
                    "dependencies": u.get("dependencies"),
                    "resource_class": u.get("resource_class"),
                }
                for u in chain["units"]
            ],
            "claim_boundary": chain["claim_boundary"],
        }
        print(json.dumps(compact, indent=1, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
