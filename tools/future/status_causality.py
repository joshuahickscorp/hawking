"""STATUS CAUSALITY — challenge a blocker on what its probe actually established.

One failure mode produced five blockers in a day, each accurate about the
check and wrong about the cause. The law, already recorded as scar
STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM, is:

    A STATUS MAY ASSERT ONLY WHAT ITS ACTUAL PROBE ESTABLISHES.

This module makes that challenge a resident call. It never names the true
world state and it never returns "wrong": it reports whether a causal claim
is entailed by the probe that was run. Absence of a recorded probe is
UNTESTED, not evidence the claim is unjustified.

    python3 tools/future/status_causality.py --build
    python3 tools/future/status_causality.py --challenge BLOCKED_NO_METAL_GPU
    python3 tools/future/status_causality.py --scan
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
VERDICTS = (SUPPORTED, OVERREACHING, UNTESTED)

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
    return None


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
            "the five historical overreaches had no regression that would fire on the next similar label",
        ],
        "negative_findings": [
            "this routine does not establish why the original Metal process saw no device",
            "this routine does not establish that Doctor or Gravity can run today; it challenges the historical constant",
            "this routine does not re-verify 360GB of Flash weights; it challenges the stale physical_status field",
            f"known receipts unreadable in this checkout: {len(missing)}",
            "a status whose probe is not recorded is UNTESTED, not OVERREACHING",
        ],
        "resident_callable": {
            "entry_point": "tools.future.status_causality.challenge()",
            "workunit": (
                "one CPU_ANALYSIS unit; challenge a consequential blocker before "
                "acting on its label"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.VERIFICATION.negative-index",
            "fails_closed": (
                "an absent probe is UNTESTED never OVERREACHING; an unreadable "
                "receipt is a recorded refusal; the routine never returns 'wrong' "
                "and never asserts the true world state"
            ),
        },
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
    return write_receipt(RECEIPT, doc, "tools/future/status_causality.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--challenge", metavar="STATUS")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    if a.challenge:
        print(json.dumps(challenge(a.challenge), indent=1, sort_keys=True, default=str))
        return 0
    if a.scan:
        print(json.dumps(scan(), indent=1, sort_keys=True, default=str))
        return 0
    out = build()
    print(out)
    doc = json.loads(out.read_text())
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
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
