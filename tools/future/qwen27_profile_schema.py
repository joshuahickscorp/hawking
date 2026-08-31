"""QWEN27_ACCELERATOR_PROFILE_SCHEMA — one contract for every optimization.

Qwen27 optimizations are incomparable when each one invents its own shape.
This module is the schema: every section required unless marked optional,
hard REJECTED (field named) when a required field is missing, transfer-law
tags on every section, incumbent sealed-3.14 recorded as CONTROL not target.

    python3 tools/future/qwen27_profile_schema.py --validate
    python3 tools/future/qwen27_profile_schema.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from typing import Any, Mapping

from tools.future._common import HARDWARE_FIELDS, REPO, git, load_json, write_receipt

RECEIPT = "QWEN27_ACCELERATOR_PROFILE_SCHEMA.json"
SCHEMA = "hawking.future.qwen27_profile.v1"
VERSION = 1

BUDGET_REL = "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json"
BUDGET_TOOL_REL = "tools/accelerator/qwen27_token_budget.py"
BUDGET_TEST_REL = "tools/accelerator/test_qwen27_token_budget.py"
ATLAS_REL = "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"
SEALED_REL = "hcli/hawking-native.sealed-3.14.json"
QUEUE_REL = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
IDENTITY_PY_REL = "hcli/agentos/qwen27_runtime_identity.py"
MLP_PY_REL = "hcli/agentos/qwen27_mlp_diagnostic.py"
HISTORICAL_REL = "receipts/headless/QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"
ORGAN_MAP_REL = "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json"
CAPABILITY_REL = "receipts/headless/CAPABILITY_noetic-sealed-3.14.json"
CAPABILITY_CONTRACT_REL = "receipts/headless/QWEN_CAPABILITY_QUALIFICATION.json"
TOKEN_ATLAS_REL = "receipts/headless/TOKEN_EXECUTION_ATLAS.json"

TRANSFER_SCOPES = (
    "MODEL_LOCAL",
    "ARCHITECTURE_FAMILY",
    "BACKEND_FAMILY",
    "MACHINE_LOCAL",
)

CLAIM_CLASSES = ("STATIC_ONLY", "DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE")
ROLES = ("CONTROL", "CANDIDATE", "EXPERIMENT")

# Token-ns organs from hawking.accelerator.qwen27_token_ns_budget.v1.
REQUIRED_ORGANS = (
    "representation_access",
    "qkv_and_projection",
    "attention",
    "deltanet_and_recurrent_state",
    "mlp",
    "lm_head_and_sampling",
    "dispatch_encode",
    "command_wait_and_synchronization",
    "host_ceremony",
)

# Measurement slots the budget / qualification queue already name.
# Sidecar values must be null / "UNKNOWN"; filling a number requires hardware.
REQUIRED_METRICS = (
    "total_nx_bytes",
    "resident_bytes",
    "active_representation_bytes_per_token",
    "actual_read_bytes_per_token",
    "transient_bytes_per_token",
    "gpu_ns_per_token",
    "complete_wall_ns_per_accepted_token",
    "dispatches_per_token",
    "sync_ns_per_token",
    "accepted_tps",
    "fallback_count",
)

LIFECYCLE_BUCKETS = (
    "cold_load_ns",
    "warm_start_ns",
    "first_token_ns",
    "warm_decode_token_ns",
    "steady_state_decode_token_ns",
    "accepted_complete_token_ns",
)

MEASUREMENT_PROTOCOL_FLAGS = (
    "same_tokenizer_and_output_contract",
    "cold_warm_first_warm_steady_state_separated",
    "complete_accepted_token_denominator",
    "protected_quiescent_before_and_after",
    "native_kernel_genome_and_dispatch_trace",
    "active_bytes_include_actual_read_and_transient_fields",
    "diagnostic_relative_runs_cannot_promote",
)

REQUIRED_SECTIONS = (
    "identity",
    "token_ns_receipt_decomposition",
    "dispatch_topology",
    "active_byte_model",
    "layout_search_space",
    "candidate_geometry",
    "representation_census",
    "cold_warm_lifecycle",
    "resident_resource_model",
    "regression_fixtures",
    "capability_workload_design",
    "transfer_law_tagging",
)

# Nested required fields. A missing key is REJECTED with the dotted name.
# Present + null/"UNKNOWN" is accepted for measurement slots; absence is not.
SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": (
        "model_family",
        "model_id",
        "resident_identity",
        "baseline_profile_path",
        "optimization_id",
        "role",
        "claim_class",
    ),
    "token_ns_receipt_decomposition": (
        "organs",
        "system_ledger_fields",
        "measurement_protocol",
        "status",
    ),
    "dispatch_topology": (
        "kernels",
        "fusion_env",
        "total_catalog_dispatches_per_token",
        "command_buffers",
    ),
    "active_byte_model": (
        "active_weight_bytes_per_token",
        "active_representation_bytes_per_token",
        "actual_read_bytes_per_token",
        "transient_bytes_per_token",
        "activations_included",
        "kv_cache_included",
        "recurrent_state_included",
        "regions",
        "claim_boundary",
    ),
    "layout_search_space": ("axes",),
    "candidate_geometry": (
        "affine2_geo",
        "q2f_geo",
        "threadgroup_width",
        "threads_per_row",
        "q4_group",
        "q2_group",
        "fusion_env",
    ),
    "representation_census": (
        "kind",
        "complete_ebpw",
        "parent_params",
        "catalog_total_bytes",
        "tensor_count",
        "packing",
        "weights_loaded_once",
    ),
    "cold_warm_lifecycle": ("buckets",),
    "resident_resource_model": (
        "total_nx_bytes",
        "resident_bytes",
        "persistent_state",
        "weights_loaded_once",
        "claim_boundary",
    ),
    "regression_fixtures": ("fixtures",),
    "capability_workload_design": (
        "suite",
        "axes",
        "item_count",
        "threshold_rule",
        "zero_fallback_required",
        "complete_accepted_token_required",
        "incumbent_capability_record",
    ),
    "transfer_law_tagging": ("by_section",),
}

# Fields whose presence-as-slot is enough; value may be null / "UNKNOWN".
MEASUREMENT_SLOTS = frozenset(
    {
        "active_representation_bytes_per_token",
        "actual_read_bytes_per_token",
        "transient_bytes_per_token",
        "total_nx_bytes",
        "resident_bytes",
        "command_buffers",
        "active_weight_bytes_per_token",
    }
)

NONEMPTY_LIST_FIELDS = frozenset(
    {
        "organs",
        "system_ledger_fields",
        "kernels",
        "regions",
        "axes",
        "fixtures",
        "packing",
        "persistent_state",
    }
)
NONEMPTY_DICT_FIELDS = frozenset(
    {
        "fusion_env",
        "measurement_protocol",
        "buckets",
        "by_section",
        "threshold_rule",
        "incumbent_capability_record",
    }
)
NONEMPTY_STR_FIELDS = frozenset(
    {
        "model_family",
        "model_id",
        "resident_identity",
        "baseline_profile_path",
        "optimization_id",
        "role",
        "claim_class",
        "status",
        "claim_boundary",
        "kind",
        "suite",
        "affine2_geo",
        "q2f_geo",
        "q4_group",
        "q2_group",
    }
)

TRANSFER_LAWS: dict[str, dict[str, Any]] = {
    "identity": {
        "scope": "MODEL_LOCAL",
        "secondary_scopes": ["ARCHITECTURE_FAMILY"],
        "rationale": (
            "resident_identity and model_id are this Qwen27 body; "
            "family qwen3.8 is architecture-family."
        ),
    },
    "token_ns_receipt_decomposition": {
        "scope": "ARCHITECTURE_FAMILY",
        "secondary_scopes": ["MODEL_LOCAL"],
        "rationale": (
            "organ/ledger closure is the token-ns law (Odyssey II seed); "
            "which organs fire is Qwen27-local."
        ),
    },
    "dispatch_topology": {
        "scope": "BACKEND_FAMILY",
        "secondary_scopes": ["MODEL_LOCAL"],
        "rationale": (
            "Metal dispatch graph is backend-family; the kernel mix is model-local."
        ),
    },
    "active_byte_model": {
        "scope": "MODEL_LOCAL",
        "secondary_scopes": ["BACKEND_FAMILY", "MACHINE_LOCAL"],
        "rationale": (
            "Which tensors are active is model-local; packing is backend-family; "
            "actual_read_bytes_per_token is machine-local (cache, contention)."
        ),
    },
    "layout_search_space": {
        "scope": "BACKEND_FAMILY",
        "secondary_scopes": [],
        "rationale": "Threadgroup/tpr/split-K legality is Metal geometry.",
    },
    "candidate_geometry": {
        "scope": "BACKEND_FAMILY",
        "secondary_scopes": ["MODEL_LOCAL"],
        "rationale": (
            "Geo flags are Metal; which organs they bind to is model-local."
        ),
    },
    "representation_census": {
        "scope": "MODEL_LOCAL",
        "secondary_scopes": ["ARCHITECTURE_FAMILY"],
        "rationale": (
            "This mix and complete EBPW are this artifact; the packing scheme "
            "is Qwen3.8-family."
        ),
    },
    "cold_warm_lifecycle": {
        "scope": "MACHINE_LOCAL",
        "secondary_scopes": ["BACKEND_FAMILY"],
        "rationale": (
            "Cold load, warm start, first token vs steady-state are machine-local."
        ),
    },
    "resident_resource_model": {
        "scope": "MACHINE_LOCAL",
        "secondary_scopes": ["MODEL_LOCAL"],
        "rationale": (
            "What fits is machine-local; what must stay resident is model-local."
        ),
    },
    "regression_fixtures": {
        "scope": "ARCHITECTURE_FAMILY",
        "secondary_scopes": ["MODEL_LOCAL"],
        "rationale": (
            "Geometry oracles are family-level; the Qwen27 MLP selector is model-local."
        ),
    },
    "capability_workload_design": {
        "scope": "MODEL_LOCAL",
        "secondary_scopes": ["ARCHITECTURE_FAMILY"],
        "rationale": (
            "Prompts and the sealed 30/43 record are this body; axis taxonomy "
            "is family-level."
        ),
    },
    "transfer_law_tagging": {
        "scope": "ARCHITECTURE_FAMILY",
        "secondary_scopes": [],
        "rationale": (
            "The tagging contract is the Odyssey II input shape. Unevidenced "
            "promotion between scopes is refused."
        ),
    },
}

# How the existing token-ns budget projects onto this schema.
# SATISFIED only when every listed path is present; otherwise NOT_SATISFIED.
# Present-but-incomplete is listed, never scored as credit.
BUDGET_COVERAGE: dict[str, dict[str, Any]] = {
    "identity": {
        "present_if": ("model", "baseline.profile"),
        "still_missing": (
            "optimization_id",
            "role",
            "claim_class",
            "resident_identity as a first-class field",
        ),
    },
    "token_ns_receipt_decomposition": {
        "present_if": ("organs", "system_ledger", "measurement_protocol", "status"),
        "still_missing": (),
    },
    "dispatch_topology": {
        "present_if": ("source_byte_denominator.regions",),
        "still_missing": ("fusion_env", "command_buffers", "topology edges"),
    },
    "active_byte_model": {
        "present_if": (
            "source_byte_denominator.active_weight_bytes_per_token",
            "source_byte_denominator.regions",
            "source_byte_denominator.claim_boundary",
        ),
        "still_missing": (
            "activations_included",
            "kv_cache_included",
            "recurrent_state_included",
            "actual_read_bytes_per_token as a catalog field (ledger slot is null)",
            "transient_bytes_per_token as a catalog field (ledger slot is null)",
        ),
    },
    "layout_search_space": {"present_if": (), "still_missing": ("axes",)},
    "candidate_geometry": {
        "present_if": (),
        "still_missing": (
            "affine2_geo",
            "q2f_geo",
            "threadgroup_width",
            "threads_per_row",
        ),
    },
    "representation_census": {
        "present_if": (
            "baseline.representation",
            "source_byte_denominator.complete_ebpw",
        ),
        "still_missing": ("parent_params", "catalog_total_bytes", "tensor_count", "packing"),
    },
    "cold_warm_lifecycle": {
        "present_if": tuple(f"lifecycle_buckets.{b}" for b in LIFECYCLE_BUCKETS),
        "still_missing": (),
    },
    "resident_resource_model": {
        "present_if": ("system_ledger.resident_bytes", "system_ledger.total_nx_bytes"),
        "still_missing": ("persistent_state", "weights_loaded_once", "a resource model"),
    },
    "regression_fixtures": {"present_if": (), "still_missing": ("fixtures",)},
    "capability_workload_design": {
        "present_if": (),
        "still_missing": ("suite", "axes", "threshold_rule"),
    },
    "transfer_law_tagging": {"present_if": (), "still_missing": ("by_section",)},
}

QUEUED_QWEN27_READY_PROTECTED = (
    "qwen27-affine2-splitk4",
    "qwen27-attention-gate-fusion",
    "qwen27-ba-delta-fusion",
    "qwen27-commit-timing-elision",
    "qwen27-deltanet-inproj-fusion",
    "qwen27-encoder-label-elision",
    "qwen27-fast-profile",
    "qwen27-gqa-qkv-fusion",
    "qwen27-pipeline-cache-reuse",
    "qwen27-pipeline-state-elision",
    "qwen27-q2f-splitk4",
    "qwen27-q4-vecgroup-x64",
)

CAPABILITY_AXES = (
    "coding",
    "hygiene",
    "knowledge",
    "mutation",
    "reasoning",
    "self_correction",
    "structured_output",
)


class ProfileRejectedError(ValueError):
    """A profile missing a required field. No partial credit."""

    def __init__(self, missing_fields: tuple[str, ...]) -> None:
        self.missing_fields = missing_fields
        named = ", ".join(missing_fields) if missing_fields else "<unspecified>"
        super().__init__(f"REJECTED: missing required field(s): {named}")


def load_authority(rel: str) -> tuple[str, dict[str, Any] | None]:
    """Load JSON from disk, else git HEAD. Disk state is authority."""
    path = REPO / rel
    if path.is_file():
        try:
            doc = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeError):
            return "ON_DISK_UNPARSEABLE", None
        return "ON_DISK", doc if isinstance(doc, dict) else None
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return "ABSENT", None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return "GIT_HEAD_UNPARSEABLE", None
    return "GIT_HEAD", doc if isinstance(doc, dict) else None


def presence(rel: str) -> dict[str, Any]:
    on_disk = (REPO / rel).is_file()
    in_git = bool(git("show", f"HEAD:{rel}"))
    return {
        "path": rel,
        "on_disk": on_disk,
        "in_git_head": in_git,
        "source": "ON_DISK" if on_disk else ("GIT_HEAD" if in_git else "ABSENT"),
    }


def _has(node: Any, path: str) -> bool:
    cur: Any = node
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    return True


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _field_missing(section: str, field: str, value: Any) -> bool:
    if field in MEASUREMENT_SLOTS:
        return False  # key was present; null / UNKNOWN is a valid slot fill
    if field in NONEMPTY_STR_FIELDS:
        return not isinstance(value, str) or value.strip() == ""
    if field in NONEMPTY_LIST_FIELDS:
        return not isinstance(value, list) or len(value) == 0
    if field in NONEMPTY_DICT_FIELDS:
        return not isinstance(value, dict) or len(value) == 0
    if field in {"complete_ebpw", "parent_params", "catalog_total_bytes", "tensor_count",
                 "item_count", "threadgroup_width", "threads_per_row",
                 "total_catalog_dispatches_per_token", "weights_loaded_once",
                 "zero_fallback_required", "complete_accepted_token_required",
                 "activations_included", "kv_cache_included", "recurrent_state_included"}:
        return value is None
    return _blank(value) and field not in MEASUREMENT_SLOTS


def _collect_missing(profile: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in profile:
            missing.append(section)
            continue
        node = profile[section]
        if not isinstance(node, dict):
            missing.append(section)
            continue
        for field in SECTION_FIELDS[section]:
            if field not in node:
                missing.append(f"{section}.{field}")
                continue
            if _field_missing(section, field, node[field]):
                missing.append(f"{section}.{field}")

    identity = profile.get("identity") if isinstance(profile.get("identity"), dict) else {}
    role = identity.get("role")
    if "identity" in profile and "identity.role" not in missing and role not in ROLES:
        missing.append("identity.role")
    claim = identity.get("claim_class")
    if "identity" in profile and "identity.claim_class" not in missing and claim not in CLAIM_CLASSES:
        missing.append("identity.claim_class")

    decomp = profile.get("token_ns_receipt_decomposition")
    if isinstance(decomp, dict) and "token_ns_receipt_decomposition.organs" not in missing:
        organs = decomp.get("organs")
        if isinstance(organs, list):
            names = [o if isinstance(o, str) else (o.get("organ") if isinstance(o, dict) else None) for o in organs]
            for organ in REQUIRED_ORGANS:
                if organ not in names:
                    missing.append(f"token_ns_receipt_decomposition.organs.{organ}")
        fields = decomp.get("system_ledger_fields")
        if isinstance(fields, list):
            for metric in REQUIRED_METRICS:
                if metric not in fields:
                    missing.append(f"token_ns_receipt_decomposition.system_ledger_fields.{metric}")
        protocol = decomp.get("measurement_protocol")
        if isinstance(protocol, dict):
            for flag in MEASUREMENT_PROTOCOL_FLAGS:
                if flag not in protocol:
                    missing.append(f"token_ns_receipt_decomposition.measurement_protocol.{flag}")

    life = profile.get("cold_warm_lifecycle")
    if isinstance(life, dict) and isinstance(life.get("buckets"), dict):
        for bucket in LIFECYCLE_BUCKETS:
            if bucket not in life["buckets"]:
                missing.append(f"cold_warm_lifecycle.buckets.{bucket}")

    tags = profile.get("transfer_law_tagging")
    if isinstance(tags, dict) and isinstance(tags.get("by_section"), dict):
        by = tags["by_section"]
        for section in REQUIRED_SECTIONS:
            if section not in by:
                missing.append(f"transfer_law_tagging.by_section.{section}")
                continue
            row = by[section]
            if not isinstance(row, dict) or "scope" not in row:
                missing.append(f"transfer_law_tagging.by_section.{section}.scope")
            elif row["scope"] not in TRANSFER_SCOPES:
                missing.append(f"transfer_law_tagging.by_section.{section}.scope")

    # Stable order, unique.
    seen: set[str] = set()
    out: list[str] = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def validate_profile(profile: Any) -> dict[str, Any]:
    """Return ACCEPTED or REJECTED. Never warns-and-continues."""
    if not isinstance(profile, Mapping):
        missing = ("<root>",)
        return {
            "status": "REJECTED",
            "missing_fields": list(missing),
            "named_refusal": f"REJECTED: missing required field(s): {missing[0]}",
        }
    missing = _collect_missing(profile)
    if missing:
        return {
            "status": "REJECTED",
            "missing_fields": missing,
            "named_refusal": "REJECTED: missing required field(s): " + ", ".join(missing),
        }
    return {
        "status": "ACCEPTED",
        "missing_fields": [],
        "named_refusal": None,
    }


def accept_profile(profile: Any) -> dict[str, Any]:
    """Raise ProfileRejectedError naming every missing required field."""
    result = validate_profile(profile)
    if result["status"] == "REJECTED":
        raise ProfileRejectedError(tuple(result["missing_fields"]))
    return result


def profiles_comparable(a: Any, b: Any) -> bool:
    """Two optimizations are comparable only if both profiles are ACCEPTED."""
    return validate_profile(a)["status"] == "ACCEPTED" and validate_profile(b)["status"] == "ACCEPTED"


def assess_token_ns_budget(budget: Any) -> dict[str, Any]:
    """Project a token-ns budget document onto the profile schema.

    SATISFIED only when every coverage path is present. A half-filled section
    is NOT_SATISFIED; present_paths are reported as findings, not credit.
    """
    if not isinstance(budget, Mapping):
        return {
            "status": "RECEIPT_ABSENT",
            "schema": None,
            "satisfied_sections": [],
            "unsatisfied_sections": list(REQUIRED_SECTIONS),
            "by_section": {
                section: {
                    "status": "NOT_SATISFIED",
                    "present_paths": [],
                    "missing_coverage_paths": list(spec["present_if"]),
                    "still_missing": list(spec["still_missing"]) or list(SECTION_FIELDS[section]),
                }
                for section, spec in BUDGET_COVERAGE.items()
            },
        }

    by_section: dict[str, Any] = {}
    satisfied: list[str] = []
    unsatisfied: list[str] = []
    for section in REQUIRED_SECTIONS:
        spec = BUDGET_COVERAGE[section]
        present = [p for p in spec["present_if"] if _has(budget, p)]
        missing_paths = [p for p in spec["present_if"] if not _has(budget, p)]
        # SATISFIED only when every coverage path is present AND the section
        # has no remaining required gaps. Present-but-incomplete is not credit.
        ok = bool(spec["present_if"]) and not missing_paths and not spec["still_missing"]
        status = "SATISFIED" if ok else "NOT_SATISFIED"
        if ok:
            satisfied.append(section)
        else:
            unsatisfied.append(section)
        by_section[section] = {
            "status": status,
            "present_paths": present,
            "missing_coverage_paths": missing_paths,
            "still_missing": list(spec["still_missing"]),
        }
    return {
        "status": "ASSESSED",
        "schema": budget.get("schema"),
        "budget_status": budget.get("status"),
        "model": budget.get("model"),
        "satisfied_sections": satisfied,
        "unsatisfied_sections": unsatisfied,
        "by_section": by_section,
        "findings": _budget_findings(budget, by_section),
    }


def _budget_findings(budget: Mapping[str, Any], by_section: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    organs = budget.get("organs")
    if isinstance(organs, list):
        unnamed = [i for i, row in enumerate(organs) if not (isinstance(row, dict) and row.get("organ"))]
        if unnamed:
            findings.append("budget organs exist but some rows lack an organ name")
        bytes_null = [
            row.get("organ")
            for row in organs
            if isinstance(row, dict) and row.get("source_weight_bytes_per_token") is None
        ]
        if bytes_null and len(bytes_null) == len(organs):
            findings.append(
                "organs[].source_weight_bytes_per_token is null on every organ; "
                "byte attribution lives under source_byte_denominator.regions, "
                "not mapped onto the token-ns organs"
            )
    denom = budget.get("source_byte_denominator")
    if isinstance(denom, dict):
        boundary = str(denom.get("claim_boundary") or "")
        if "activations" in boundary.lower() or "KV" in boundary or "recurrent" in boundary.lower():
            findings.append(
                "source_byte_denominator claim_boundary excludes activations, "
                "KV, and recurrent state — the active-byte MODEL is not complete"
            )
        regions = denom.get("regions")
        if isinstance(regions, list):
            findings.append(
                f"source_byte_denominator.regions has {len(regions)} catalog rows "
                "(STATIC_DERIVATION_FROM_CATALOG)"
            )
    if budget.get("promotion_allowed") is False:
        findings.append("promotion_allowed is false (correct; budget is not a protected measurement)")
    life = budget.get("lifecycle_buckets")
    if isinstance(life, dict) and all(life.get(b) is None for b in LIFECYCLE_BUCKETS if b in life):
        findings.append("lifecycle_buckets keys exist and every value is null (slots, not measurements)")
    for section, row in by_section.items():
        if row["status"] == "NOT_SATISFIED":
            findings.append(f"section {section} is NOT_SATISFIED against the profile schema")
    return findings


def recovered_budget_shape(atlas: Mapping[str, Any] | None) -> dict[str, Any]:
    """Key structure of hawking.accelerator.qwen27_token_ns_budget.v1.

    Reconstructed from recover-before-build of the untracked campaign file
    plus the git-authoritative byte atlas. Physical actuals are not copied.
    """
    kernels, _total, active = _kernels_from_atlas(atlas)
    headline = atlas.get("headline") if isinstance(atlas, Mapping) else {}
    art = atlas.get("artifact") if isinstance(atlas, Mapping) else {}
    regions = []
    for row in kernels:
        regions.append(
            {
                "kernel": row.get("kernel"),
                "dispatches_per_token": row.get("catalog_dispatches_per_token"),
                "weight_bytes_per_token": row.get("weight_bytes_per_token"),
                "bytes_per_dispatch": row.get("bytes_per_dispatch"),
                "roles": list(row.get("roles") or []),
                "label": "STATIC_DERIVATION_FROM_CATALOG",
            }
        )
    return {
        "schema": "hawking.accelerator.qwen27_token_ns_budget.v1",
        "status": "PLANNED_UNTIL_NATIVE_PROTECTED_EXECUTION",
        "model": "qwen3.8-27b-sealed-3.14",
        "baseline": {
            "profile": SEALED_REL,
            "representation": "native-packed sealed control",
            "byte_atlas": ATLAS_REL,
        },
        "organs": [
            {
                "organ": name,
                "source_weight_bytes_per_token": None,
                "source_dispatches_per_token": None,
                "actual": {metric: None for metric in REQUIRED_METRICS},
                "status": "WAITING_FOR_NATIVE_PROTECTED_EXECUTION",
            }
            for name in REQUIRED_ORGANS
        ],
        "system_ledger": {name: None for name in REQUIRED_METRICS},
        "lifecycle_buckets": {name: None for name in LIFECYCLE_BUCKETS},
        "measurement_protocol": {flag: True for flag in MEASUREMENT_PROTOCOL_FLAGS},
        "source_byte_denominator": {
            "active_weight_bytes_per_token": (headline or {}).get("active_weight_bytes_per_token", active),
            "complete_ebpw": (art or {}).get("complete_ebpw"),
            "regions": regions,
            "claim_boundary": (
                "catalog-derived weight traffic only; activations, KV, and "
                "recurrent state are not included"
            ),
        },
        "promotion_allowed": False,
        "control_observation_present_on_recovered_file": True,
        "control_observation_metrics_copied": False,
    }


def _unknown() -> str:
    return "UNKNOWN"


def _transfer_tagging() -> dict[str, Any]:
    by_section = {}
    for section in REQUIRED_SECTIONS:
        law = TRANSFER_LAWS[section]
        by_section[section] = {
            "scope": law["scope"],
            "secondary_scopes": list(law["secondary_scopes"]),
            "odyssey": "II",
            "promotion_rule": (
                "Odyssey II law store must evidence a scope promotion; "
                "unevidenced promotion is refused. There is no Odyssey IV."
            ),
            "rationale": law["rationale"],
        }
    return {
        "by_section": by_section,
        "allowed_scopes": list(TRANSFER_SCOPES),
        "odysseys": ("I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?", "III WHERE IS HAWKING WRONG?"),
        "eras": (
            "I Genesis of the Laboratory",
            "II Compounding Civilization",
            "III Autonomous Science Civilization",
            "IV Synthetic Machine Civilization",
            "V Released Hawking Civilization",
        ),
        "no_era_vi": True,
        "no_odyssey_iv": True,
    }


def _layout_axes() -> list[dict[str, Any]]:
    return [
        {
            "name": "threads_per_row",
            "legal_values": [64],
            "illegal_values": [],
            "control": 64,
            "invariant": "arithmetic must match the float64 oracle; tpr is occupancy, not a new function",
            "evidence": "tools/accelerator/test_native_geometry.py",
        },
        {
            "name": "threadgroup_width",
            "legal_values": [64, 128, 256, 512],
            "illegal_values": [32, 96],
            "control": 128,
            "invariant": "tg must be a whole number of rows; padded grids are refused",
            "evidence": "tools/accelerator/test_threadgroup_width.py",
        },
        {
            "name": "affine2_geo",
            "legal_values": ["tpr64", "splitk4"],
            "illegal_values": [],
            "control": "tpr64",
            "invariant": "packed Q2 bytes unchanged; split-K is concurrency, not representation",
            "evidence": "recovered qualification-queue candidate qwen27-affine2-splitk4; kernel name geo_tpr64_tg128",
        },
        {
            "name": "q2f_geo",
            "legal_values": ["tpr64", "splitk4"],
            "illegal_values": [],
            "control": "tpr64",
            "invariant": "same as affine2_geo for the Q2 family kernel",
            "evidence": "recovered qualification-queue candidate qwen27-q2f-splitk4",
        },
        {
            "name": "q4_vecgroup",
            "legal_values": ["default", "x64"],
            "illegal_values": [],
            "control": "default",
            "invariant": "group-64 Q4 packing stays group-64",
            "evidence": "recovered qualification-queue candidate qwen27-q4-vecgroup-x64",
        },
        {
            "name": "fuse_mlp",
            "legal_values": ["swiglu", "1"],
            "illegal_values": [],
            "control": "swiglu",
            "invariant": "HAWKING_QWEN38_FUSE_MLP=1 is an alias for the strongest swiglu graph, not unfused",
            "evidence": "hcli/agentos/qwen27_mlp_diagnostic.py (DIAGNOSTIC_RELATIVE, not for promotion)",
        },
        {
            "name": "fuse_gqa_qkv",
            "legal_values": ["0", "1"],
            "illegal_values": [],
            "control": "1",
            "invariant": "source-approved fusion flag; output-id parity required",
            "evidence": "hcli/hawking-native.sealed-3.14.json fusion_env",
        },
        {
            "name": "fuse_dn_inproj",
            "legal_values": ["0", "1"],
            "illegal_values": [],
            "control": "1",
            "invariant": "source-approved fusion flag; output-id parity required",
            "evidence": "hcli/hawking-native.sealed-3.14.json fusion_env",
        },
        {
            "name": "fuse_add_rmsnorm",
            "legal_values": ["0", "1"],
            "illegal_values": [],
            "control": "1",
            "invariant": "source-approved fusion flag; output-id parity required",
            "evidence": "hcli/hawking-native.sealed-3.14.json fusion_env",
        },
    ]


def _regression_fixtures() -> list[dict[str, Any]]:
    return [
        {
            "id": "native_geometry_tpr64_oracle",
            "path": "tools/accelerator/test_native_geometry.py",
            "what_it_pins": "tpr64 matches one-thread-per-row oracle; illegal geometry is refused",
        },
        {
            "id": "threadgroup_width_refusals",
            "path": "tools/accelerator/test_threadgroup_width.py",
            "what_it_pins": "tg not a whole number of rows is refused; per-row run count does not move with tg",
        },
        {
            "id": "bytes_atlas_reconciliation",
            "path": "tools/accelerator/test_bytes_atlas.py",
            "what_it_pins": "per-dispatch bytes sum to catalog; catalog equals disk; 8*bytes/params = sealed EBPW",
        },
        {
            "id": "qwen27_mlp_selector",
            "path": "hcli/agentos/qwen27_mlp_diagnostic.py",
            "what_it_pins": "swiglu vs 1 is graph identity, DIAGNOSTIC_RELATIVE, NOT_FOR_PROMOTION",
        },
        {
            "id": "capability_incumbent_threshold",
            "path": "tools/headless/capability_contract.py",
            "what_it_pins": "threshold is incumbent-relative, never the candidate's own score",
        },
        {
            "id": "token_ns_budget_slots",
            "path": "tools/accelerator/qwen27_token_budget.py",
            "what_it_pins": "physical metrics stay null until a protected complete-token receipt",
        },
        {
            "id": "output_id_parity",
            "path": "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "what_it_pins": "parity_contract: identical tokenizer-bound output ids; any divergence rejects",
        },
    ]


def _kernels_from_atlas(atlas: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], Any, Any]:
    if not isinstance(atlas, Mapping):
        return [], _unknown(), _unknown()
    rows = atlas.get("pareto_by_bytes")
    kernels: list[dict[str, Any]] = []
    total = 0
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            disp = row.get("dispatches")
            if isinstance(disp, int):
                total += disp
            kernels.append(
                {
                    "kernel": row.get("kernel"),
                    "catalog_dispatches_per_token": disp,
                    "weight_bytes_per_token": row.get("weight_bytes"),
                    "bytes_per_dispatch": row.get("bytes_per_dispatch"),
                    "roles": list(row.get("roles") or []),
                    "label": "STATIC_DERIVATION_FROM_CATALOG",
                }
            )
    headline = atlas.get("headline") if isinstance(atlas.get("headline"), Mapping) else {}
    active = headline.get("active_weight_bytes_per_token", _unknown())
    return kernels, total if kernels else _unknown(), active


def control_profile(
    *,
    atlas: Mapping[str, Any] | None = None,
    sealed: Mapping[str, Any] | None = None,
    capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete CONTROL profile for the sealed-3.14 incumbent.

    Catalog fields are filled from git-authoritative receipts. Every hardware
    measurement slot is UNKNOWN. Historical TPS/EBPW live in incumbent_control
    on the schema receipt, not in HARDWARE_FIELDS.
    """
    if atlas is None:
        _src, atlas = load_authority(ATLAS_REL)
    if sealed is None:
        _src, sealed = load_authority(SEALED_REL)
    if capability is None:
        _src, capability = load_authority(CAPABILITY_REL)

    kernels, total_disp, active_bytes = _kernels_from_atlas(atlas)
    fusion = {}
    if isinstance(sealed, Mapping) and isinstance(sealed.get("fusion_env"), dict):
        fusion = dict(sealed["fusion_env"])
    artifact = atlas.get("artifact") if isinstance(atlas, Mapping) and isinstance(atlas.get("artifact"), dict) else {}
    packing = [
        {
            "family": "q2_group32",
            "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            "roles": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
        {
            "family": "q4_group64",
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "roles": ["linear_attn in/out", "self_attn qkvo", "lm_head"],
        },
        {
            "family": "embedding_q4_row",
            "kernel": "qwen_uniform_q4_embedding_lookup",
            "roles": ["embed_tokens one row per token, not the table"],
        },
    ]
    cap_overall = capability.get("overall") if isinstance(capability, Mapping) else None
    passed = cap_overall.get("passed") if isinstance(cap_overall, dict) else _unknown()
    total_items = cap_overall.get("total") if isinstance(cap_overall, dict) else 43

    profile = {
        "identity": {
            "model_family": "qwen3.8",
            "model_id": (sealed or {}).get("model_id") if isinstance(sealed, Mapping) else "qwen3.8-27b-sealed-3.14",
            "resident_identity": (sealed or {}).get("resident_identity") if isinstance(sealed, Mapping) else "sealed-3.14",
            "baseline_profile_path": SEALED_REL,
            "optimization_id": "qwen27-incumbent-sealed-3.14-CONTROL",
            "role": "CONTROL",
            "claim_class": "STATIC_ONLY",
        },
        "token_ns_receipt_decomposition": {
            "organs": list(REQUIRED_ORGANS),
            "system_ledger_fields": list(REQUIRED_METRICS),
            "measurement_protocol": {flag: True for flag in MEASUREMENT_PROTOCOL_FLAGS},
            "status": "SLOTS_DECLARED_VALUES_UNKNOWN",
            "actual_values": {name: _unknown() for name in REQUIRED_METRICS},
        },
        "dispatch_topology": {
            "kernels": kernels or [
                {
                    "kernel": "UNKNOWN_ATLAS_ABSENT",
                    "catalog_dispatches_per_token": _unknown(),
                    "weight_bytes_per_token": _unknown(),
                    "bytes_per_dispatch": _unknown(),
                    "roles": ["UNKNOWN"],
                    "label": "ATLAS_ABSENT",
                }
            ],
            "fusion_env": fusion or {
                "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
                "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
                "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
                "HAWKING_QWEN38_FUSE_MLP": "swiglu",
            },
            "total_catalog_dispatches_per_token": total_disp,
            "command_buffers": _unknown(),
        },
        "active_byte_model": {
            "active_weight_bytes_per_token": active_bytes,
            "active_representation_bytes_per_token": _unknown(),
            "actual_read_bytes_per_token": _unknown(),
            "transient_bytes_per_token": _unknown(),
            "activations_included": False,
            "kv_cache_included": False,
            "recurrent_state_included": False,
            "regions": kernels or [
                {
                    "kernel": "UNKNOWN_ATLAS_ABSENT",
                    "catalog_dispatches_per_token": _unknown(),
                    "weight_bytes_per_token": _unknown(),
                    "bytes_per_dispatch": _unknown(),
                    "roles": ["UNKNOWN"],
                    "label": "ATLAS_ABSENT",
                }
            ],
            "claim_boundary": (
                "Catalog-derived WEIGHT traffic only. Activations, KV cache, and "
                "DeltaNet recurrent state are not in the HQ38M20 catalog and are "
                "not counted. actual_read_bytes_per_token and transient_bytes_per_token "
                "remain UNKNOWN until a protected complete-token receipt."
            ),
        },
        "layout_search_space": {"axes": _layout_axes()},
        "candidate_geometry": {
            "affine2_geo": "tpr64",
            "q2f_geo": "tpr64",
            "threadgroup_width": 128,
            "threads_per_row": 64,
            "q4_group": "group64",
            "q2_group": "group32",
            "fusion_env": fusion or {
                "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
                "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
                "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
                "HAWKING_QWEN38_FUSE_MLP": "swiglu",
            },
            "source": "kernel names in ACCELERATOR_TOKEN_BYTES_ATLAS.json (geo_tpr64_tg128) plus sealed fusion_env",
        },
        "representation_census": {
            "kind": (sealed or {}).get("representation", {}).get("kind")
            if isinstance(sealed, Mapping) and isinstance(sealed.get("representation"), dict)
            else "native-packed",
            "complete_ebpw": artifact.get("complete_ebpw", _unknown()),
            "parent_params": artifact.get("parent_params", _unknown()),
            "catalog_total_bytes": artifact.get("catalog_total_bytes", _unknown()),
            "tensor_count": artifact.get("tensors", _unknown()),
            "packing": packing,
            "weights_loaded_once": True,
        },
        "cold_warm_lifecycle": {
            "buckets": {name: _unknown() for name in LIFECYCLE_BUCKETS},
            "claim_boundary": "lifecycle timings require a protected complete-token receipt",
        },
        "resident_resource_model": {
            "total_nx_bytes": _unknown(),
            "resident_bytes": _unknown(),
            "persistent_state": [
                {"id": "resident_weight_shards", "lifetime": "persistent", "per_token_transfer": False},
                {"id": "activations", "lifetime": "token", "per_token_transfer": True},
                {"id": "partial_reductions", "lifetime": "token", "per_token_transfer": True},
                {"id": "persistent_state", "lifetime": "sequence", "per_token_transfer": False},
            ],
            "weights_loaded_once": True,
            "claim_boundary": (
                "Persistent-state names recovered from QWEN27_FPGA_ORGAN_MAP.json "
                "as [D] hypotheses. FPGA is Accelerator/Physical Compiler/Fusion, "
                "not its own civilization. No board, bitstream, or HBM number is claimed."
            ),
        },
        "regression_fixtures": {"fixtures": _regression_fixtures()},
        "capability_workload_design": {
            "suite": "QWEN_CAPABILITY_QUALIFICATION",
            "axes": list(CAPABILITY_AXES),
            "item_count": total_items if isinstance(total_items, int) else 43,
            "threshold_rule": {
                "basis": "incumbent llama.cpp Q5_K production artifact",
                "rule": "per-axis rate >= incumbent rate - 0.1",
                "why_not_self_referential": "a threshold read off the candidate cannot fail",
            },
            "zero_fallback_required": True,
            "complete_accepted_token_required": True,
            "incumbent_capability_record": {
                "role": "CONTROL",
                "not_a_target": True,
                "not_a_ceiling": True,
                "source": CAPABILITY_REL,
                "passed": passed,
                "total": total_items if isinstance(total_items, int) else 43,
                "label": "historical sealed contract 30/43",
            },
        },
        "transfer_law_tagging": _transfer_tagging(),
    }
    return profile


def incumbent_control_block(sealed: Mapping[str, Any] | None, atlas: Mapping[str, Any] | None,
                            capability: Mapping[str, Any] | None) -> dict[str, Any]:
    """Historical ~3.14 EBPW / ~25 accepted TPS as CONTROL, never a hardware field."""
    ebpw = None
    tps_record = None
    hist_qual = None
    if isinstance(sealed, Mapping):
        ebpw = sealed.get("physical_ebpw")
        runtime = sealed.get("current_runtime")
        if isinstance(runtime, dict):
            tps_record = runtime.get("complete_tps_current_measured")
            hist_qual = runtime.get("complete_tps_historical_qualified")
        rep = sealed.get("representation")
        if ebpw is None and isinstance(rep, dict):
            ebpw = rep.get("physical_ebpw")
    if ebpw is None and isinstance(atlas, Mapping):
        art = atlas.get("artifact")
        if isinstance(art, dict):
            ebpw = art.get("complete_ebpw")
    cap = None
    if isinstance(capability, Mapping) and isinstance(capability.get("overall"), dict):
        cap = capability["overall"]
    return {
        "role": "CONTROL",
        "not_a_target": True,
        "not_a_ceiling": True,
        "label": "historical sealed-3.14 resident",
        "provenance": {
            "profile": SEALED_REL,
            "profile_schema": (sealed or {}).get("profile_schema") if isinstance(sealed, Mapping) else "hcli.provider.profile.v1",
            "resident_identity": (sealed or {}).get("resident_identity") if isinstance(sealed, Mapping) else "sealed-3.14",
            "qualification_record": (sealed or {}).get("qualification") if isinstance(sealed, Mapping) else "capability 30/43 historical sealed contract",
            "byte_atlas": ATLAS_REL,
        },
        "historical_physical_ebpw": {
            "approx_label": "~3.14",
            "recorded_complete_ebpw": ebpw,
            "source_field": "physical_ebpw / representation.physical_ebpw / artifact.complete_ebpw",
            "evidence_class": "STATIC_CATALOG_IDENTITY",
            "role": "CONTROL",
        },
        "historical_accepted_tokens_per_second_record": {
            "approx_label": "~25",
            "recorded_complete_tokens_per_second": tps_record,
            "source_field": "current_runtime.complete_tps_current_measured",
            "source_path": SEALED_REL,
            "evidence_class": "HISTORICAL_RECORD_NOT_A_SIDECAR_MEASUREMENT",
            "role": "CONTROL",
            "do_not_promote": True,
            "contamination": "unknown; sidecar did not remeasure; not DIAGNOSTIC_RELATIVE and not PROTECTED_ABSOLUTE",
        },
        "related_historical_records": [
            {
                "label": "complete_tps_historical_qualified",
                "recorded_complete_tokens_per_second": hist_qual,
                "source_field": "current_runtime.complete_tps_historical_qualified",
                "source_path": SEALED_REL,
                "role": "HISTORICAL_RECORD",
                "do_not_promote": True,
            },
            {
                "label": "capability_noetic_sealed_3_14",
                "passed": (cap or {}).get("passed") if isinstance(cap, dict) else 30,
                "total": (cap or {}).get("total") if isinstance(cap, dict) else 43,
                "source_path": CAPABILITY_REL,
                "role": "CONTROL",
                "not_a_target": True,
            },
        ],
        "hardware_field_policy": (
            "These numbers are not written under tps / accepted_tps / token_ns / "
            "gpu_ns / wall_ns. They are historical record with provenance."
        ),
    }


def _hardware_numeric_keys(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                hits.append(here)
            hits.extend(_hardware_numeric_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_hardware_numeric_keys(value, f"{path}[{i}]"))
    return hits


def build_document() -> dict[str, Any]:
    probes = {
        name: presence(rel)
        for name, rel in (
            ("token_ns_budget", BUDGET_REL),
            ("token_budget_tool", BUDGET_TOOL_REL),
            ("token_budget_test", BUDGET_TEST_REL),
            ("byte_atlas", ATLAS_REL),
            ("sealed_profile", SEALED_REL),
            ("qualification_queue", QUEUE_REL),
            ("runtime_identity_py", IDENTITY_PY_REL),
            ("mlp_diagnostic_py", MLP_PY_REL),
            ("historical_identity", HISTORICAL_REL),
            ("fpga_organ_map", ORGAN_MAP_REL),
            ("capability_sealed", CAPABILITY_REL),
            ("capability_contract", CAPABILITY_CONTRACT_REL),
            ("token_execution_atlas", TOKEN_ATLAS_REL),
        )
    }
    budget_src, budget = load_authority(BUDGET_REL)
    atlas_src, atlas = load_authority(ATLAS_REL)
    sealed_src, sealed = load_authority(SEALED_REL)
    cap_src, capability = load_authority(CAPABILITY_REL)

    tagging = _transfer_tagging()
    profile = control_profile(atlas=atlas, sealed=sealed, capability=capability)
    accepted = accept_profile(profile)
    budget_assessment = assess_token_ns_budget(budget)

    stripped = {k: v for k, v in profile.items() if k != "active_byte_model"}
    negative = validate_profile(stripped)

    recovered_paths = [{**row, "role": name} for name, row in probes.items()]

    gaps_closed = [
        "one comparable Qwen27 optimization profile with every section required",
        "hard REJECTED (field named) when a required field is missing; no partial credit",
        "negative control: missing active_byte_model is refused by name",
        "transfer-law tagging MODEL_LOCAL / ARCHITECTURE_FAMILY / BACKEND_FAMILY / MACHINE_LOCAL on every section",
        "incumbent sealed-3.14 recorded as CONTROL with provenance, not as a target or ceiling",
        "projection of hawking.accelerator.qwen27_token_ns_budget.v1 onto the profile sections",
    ]

    negative_findings = [
        (
            "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json is ABSENT from this worktree "
            f"disk and from git HEAD (load={budget_src}). It exists untracked in the "
            "campaign working tree. Coverage below is from live load when present, "
            "else RECEIPT_ABSENT. Structural gaps were also inspected during recover-before-build."
        ),
        "tools/accelerator/qwen27_token_budget.py is not in git HEAD of this worktree (untracked in the campaign tree).",
        "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json is not in git HEAD (untracked in the campaign tree).",
        "hcli/agentos is not materialized in this sparse checkout; identity/mlp modules were read via git show.",
        "This sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE measurements.",
        "Catalog active-byte model excludes activations, KV, and recurrent state (stated on the budget claim_boundary).",
        "Budget organs do not carry per-organ source_weight_bytes_per_token; attribution stays on atlas regions.",
        "No pre-existing tools/future qwen27 profile schema module — this file is the first.",
        "FPGA organ map is a [D] hypothesis; FPGA is not a civilization and no FPGA backend was built.",
    ]

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Single contract under which every Qwen27 optimization, past and future, "
            "is comparable. A profile missing a required field is REJECTED."
        ),
        "nomenclature": {
            "eras": tagging["eras"],
            "odysseys": tagging["odysseys"],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga": "part of Accelerator / Physical Compiler / Fusion; not its own civilization",
            "diagnostic_relative": "contaminated A/B on a busy machine. Guides. Never promotes.",
            "protected_absolute": "measurement under a real protected GPU lease. Decides.",
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "required_sections": list(REQUIRED_SECTIONS),
        "section_fields": {k: list(v) for k, v in SECTION_FIELDS.items()},
        "required_organs": list(REQUIRED_ORGANS),
        "required_metrics": list(REQUIRED_METRICS),
        "lifecycle_buckets": list(LIFECYCLE_BUCKETS),
        "transfer_scopes": list(TRANSFER_SCOPES),
        "transfer_laws": TRANSFER_LAWS,
        "claim_classes": list(CLAIM_CLASSES),
        "roles": list(ROLES),
        "rejection_rule": (
            "A profile missing a required field is REJECTED with the field named. "
            "No partial credit, no warning-and-continue."
        ),
        "incumbent_control": incumbent_control_block(sealed, atlas, capability),
        "control_profile": profile,
        "control_profile_validation": accepted,
        "negative_control": {
            "removed_field": "active_byte_model",
            "result": negative,
            "fires": negative["status"] == "REJECTED" and "active_byte_model" in negative["missing_fields"],
        },
        "token_ns_budget": {
            "path": BUDGET_REL,
            "load": budget_src,
            "presence": probes["token_ns_budget"],
            "assessment": budget_assessment,
            "recovered_campaign_working_tree": {
                "note": (
                    "The live receipt is untracked in the campaign working tree and "
                    "absent from this worktree's git HEAD. Shape below is reconstructed "
                    "from recover-before-build plus the git-authoritative byte atlas. "
                    "Physical actuals and control_observation metrics were not copied."
                ),
                "assessment": assess_token_ns_budget(recovered_budget_shape(atlas)),
            },
        },
        "queued_qwen27_ready_protected": {
            "source": QUEUE_REL,
            "presence": probes["qualification_queue"],
            "count": len(QUEUED_QWEN27_READY_PROTECTED),
            "ids": list(QUEUED_QWEN27_READY_PROTECTED),
            "note": (
                "12 READY_PROTECTED candidates recovered from the untracked "
                "qualification queue. This schema exists so those candidates "
                "become comparable rather than informally ranked."
            ),
        },
        "authority_sources": {
            "atlas": atlas_src,
            "sealed": sealed_src,
            "capability": cap_src,
            "budget": budget_src,
        },
        "recovered_implementation": {
            "summary": (
                "No Qwen27 accelerator profile schema existed. "
                "tools/accelerator/qwen27_token_budget.py (untracked in HEAD) already "
                "emits the token-ns ledger scaffold; this module consumes that shape "
                "as one required section instead of forking it. Byte atlas, sealed-3.14 "
                "identity, capability 30/43, MLP diagnostic, and the qualification queue "
                "are recovered inputs. The gap closed is the comparable-profile contract "
                "around them: hard rejection, transfer-law tags, incumbent-as-control."
            ),
            "paths": recovered_paths,
        },
        "gaps_closed": gaps_closed,
        "negative_findings": negative_findings,
        "downstream": {
            "validate_profile": "validate_profile(profile) -> {status, missing_fields, named_refusal}",
            "accept_profile": "accept_profile(profile) -> result or raise ProfileRejectedError",
            "control_profile": "control_profile() -> complete CONTROL instance",
            "assess_token_ns_budget": "assess_token_ns_budget(budget) -> section coverage",
            "profiles_comparable": "profiles_comparable(a, b) -> bool",
        },
    }
    hardware_hits = _hardware_numeric_keys(doc)
    if hardware_hits:
        raise RuntimeError(f"schema receipt would claim hardware fields: {hardware_hits}")
    return doc


def build() -> Any:
    doc = build_document()
    return write_receipt(RECEIPT, doc, "tools/future/qwen27_profile_schema.py")


def selftest() -> Any:
    profile = control_profile()
    accept_profile(profile)
    stripped = {k: v for k, v in profile.items() if k != "active_byte_model"}
    result = validate_profile(stripped)
    if result["status"] != "REJECTED":
        raise AssertionError("negative control did not REJECT a profile missing active_byte_model")
    if "active_byte_model" not in result["missing_fields"]:
        raise AssertionError(
            "negative control REJECTED but did not name active_byte_model; "
            f"got {result['missing_fields']}"
        )
    try:
        accept_profile(stripped)
    except ProfileRejectedError as exc:
        if "active_byte_model" not in str(exc):
            raise AssertionError(f"refusal did not name the field: {exc}") from exc
    else:
        raise AssertionError("accept_profile did not raise on missing active_byte_model")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        out = selftest()
        doc = json.loads(out.read_text())
        assessment = doc["token_ns_budget"]["assessment"]
        print(f"receipt: {out}")
        print(f"control_profile: {doc['control_profile_validation']['status']}")
        print(
            "negative_control: "
            f"fires={doc['negative_control']['fires']} "
            f"status={doc['negative_control']['result']['status']}"
        )
        print(f"budget_load: {doc['token_ns_budget']['load']}")
        print(f"budget_assessment: {assessment.get('status')}")
        print("satisfied_sections:", assessment.get("satisfied_sections"))
        print("unsatisfied_sections:", assessment.get("unsatisfied_sections"))
        recovered = doc["token_ns_budget"]["recovered_campaign_working_tree"]["assessment"]
        print("recovered_budget_satisfied:", recovered.get("satisfied_sections"))
        print("recovered_budget_unsatisfied:", recovered.get("unsatisfied_sections"))
        print(f"seal: {doc['seal_sha256']}")
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
