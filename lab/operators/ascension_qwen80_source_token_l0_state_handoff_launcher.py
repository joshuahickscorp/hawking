"""Fail-closed CPU-only outer preflight for Qwen80 source-token L0→L1 state handoff.

This module binds the earned L0 9+14 component, the sealed incomplete handoff
authority, and the immutable CPU-only child preflight.  It writes planning
evidence only.  It invokes exactly one bounded, reaped CPU child preflight;
that child receives no lease, capture, or Metal-mode arguments.  This module
never creates a Metal context, issues a lease, starts a watcher/server,
measures a token/TPS/TG result, or launches a tournament.

``--prepare-one-shot`` is also planning-only: it consumes (but never issues) a
future lease, reserves its ID with a create-new replay guard, and writes a
receipt-last outer-reaper authority for a separately authorized executor.

The admission-current pointer is versioned mutable control-plane evidence. A
later seal/byte change is accepted only when its canonical path still selects
the exact immutable manifest and admission receipt; every preflight and launch
observation retains its historical raw/seal evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import seal, verify


SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_launcher.v1"
OUTER_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_preflight.v1"
OUTER_PREFLIGHT_STATUS = "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_READY_NOT_LEASED_OR_EXECUTED"
PREFLIGHT_PROOF_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_preflight_proof.v1"
PREFLIGHT_PROOF_STATUS = "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_AND_CHILD_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
CPU_CHILD_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1"
CPU_CHILD_PREFLIGHT_STATUS = "PREPARED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_CHILD_NOT_LEASED_OR_EXECUTED"
TERMINAL_PLAN_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_terminal_plan.v1"
TERMINAL_PLAN_STATUS = "PLANNED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_RECEIPT_LAST_NOT_EXECUTED"
OUTER_LAUNCH_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_launch_authority.v1"
OUTER_LAUNCH_AUTHORITY_STATUS = "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_REAPED_ONE_SHOT_METAL_CHILD"

RUNTIME_DIR = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
COMPLETE_GRAVITY_DIR = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
MANIFEST_PATH = COMPLETE_GRAVITY_DIR / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
ADMISSION_CURRENT_PATH = COMPLETE_GRAVITY_DIR / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"
HANDOFF_AUTHORITY_PATH = RUNTIME_DIR / "QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_AUTHORITY_20260809T063100Z.json"
HANDOFF_AUTHORITY_RAW_SHA256 = "29bb95796a4305658f997104ac891eb59125c63f301c236bbd5d7ae43f1ab3a8"
HANDOFF_AUTHORITY_BYTES = 7_007
HANDOFF_AUTHORITY_SEAL_SHA256 = "ec31baec387a1065692b7fd0c2350f54db689112e77b50046225a37d316c40ca"
HANDOFF_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_l0_to_layer1_handoff_authority.v1"
HANDOFF_AUTHORITY_STATUS = "ASSESSED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_INCOMPLETE_MISSING_RETAINED_DEVICE_OUTPUT_AND_POST_STATE_WITNESSES"

CHILD_PREFLIGHT_PATH = RUNTIME_DIR / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_CHILD_PREFLIGHT_20260809T064500Z.json"
CHILD_PREFLIGHT_RAW_SHA256 = "ad372790dea8fdc5ed5e2f3fddf2ddd62057b73398aed4578f8fe197016970dd"
CHILD_PREFLIGHT_BYTES = 5_544
CHILD_PREFLIGHT_SEAL_SHA256 = "c7c81b8015f946adaf7e217273a9ac6557ef98de7e7a4c08cbff069dbf69f6bf"
CHILD_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l0_to_layer1_state_handoff_device.v1"
CHILD_PREFLIGHT_STATUS = "PREPARED_QWEN80_SOURCE_TOKEN_L0_STATE_COMMIT_ROLLBACK_AND_LAYER1_HANDOFF_CHILD_NOT_EXECUTED"
SOURCE_ALL_TEN_OUTER_PREFLIGHT_PATH = (
    RUNTIME_DIR
    / "QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_CPU_PREFLIGHT_20260809T061742Z"
    / "outer-preflight.json"
)
SOURCE_ALL_TEN_OUTER_PREFLIGHT_RAW_SHA256 = "22194bc685016126f6460f0d78f81cc983885ee677e38b9a47c114ef75d8cf7e"
SOURCE_ALL_TEN_OUTER_PREFLIGHT_BYTES = 5_541
SOURCE_ALL_TEN_OUTER_PREFLIGHT_SEAL_SHA256 = "319ae49f62a84071e3e65221600270a4c6e7cf90ad1d8f983e292cf4af6126d5"
SOURCE_ALL_TEN_OUTER_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1"
SOURCE_ALL_TEN_OUTER_PREFLIGHT_STATUS = "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED"
ADMISSION_CURRENT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
SOURCE_AUDIT_SEAL_SHA256 = "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4"
HANDOFF_WITNESS_SCHEMA = "hawking.ascension.qwen80_l0_to_layer1_device_handoff_witness.v1"
HANDOFF_WITNESS_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_RETAINED_DEVICE_HANDOFF_COMPONENT_ONLY"
PRE_L1_CAPTURE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1"
PRE_L1_CAPTURE_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY"
EXPECTED_PROBE_BASENAME = "ascension_qwen80_source_token_l0_to_layer1_state_handoff_device"
# Rebuilt after the Rust/Python canonical JSON seal compatibility repair.
EXPECTED_PROBE_SHA256 = "d6b37c0083a60f04638cd4a63d8d3477e724141515f29444cf37c584082925e6"

COMPONENT_OUTER_SCHEMA = "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_outer_launcher.v1"
COMPONENT_OUTER_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_TERMINAL_COMPONENT_ONLY"
COMPONENT_INNER_SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_graph_device.v1"
COMPONENT_INNER_STATUS = "EARNED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"

FUTURE_LEASE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease.v1"
FUTURE_LEASE_STATUS = "GRANTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_NON_TIMED_DEVICE_PARITY_LEASE"
OUTER_PREFLIGHT_FILENAME = "outer-preflight.json"
PREFLIGHT_PROOF_FILENAME = "preflight-proof.json"
TERMINAL_PLAN_FILENAME = "outer-terminal-plan.json"
OUTER_LAUNCH_AUTHORITY_FILENAME = "outer-launch-authority.json"
ACTIVE_FILENAME = "active.json"
TERMINAL_RECEIPT_FILENAME = "outer-terminal-receipt.json"
INNER_CAPTURE_DIRNAME = "inner"
MAX_JSON_BYTES = 1_000_000
CPU_CHILD_PREFLIGHT_TIMEOUT_SECONDS = 60
CHILD_PREFLIGHT_STDOUT_FILENAME = "child-preflight.stdout.json"
CHILD_PREFLIGHT_STDERR_FILENAME = "child-preflight.stderr.log"
FORBIDDEN_ARGUMENTS = frozenset({"--execute-one-shot", "--metal", "--router-receipt", "--route-plan", "--device-child"})


class SourceTokenL0StateHandoffLauncherError(RuntimeError):
    """The exact L0→L1 CPU planning chain is incomplete or unsafe."""


@dataclass(frozen=True)
class HandoffContext:
    manifest_evidence: dict[str, Any]
    manifest_seal_sha256: str
    admission_current_evidence: dict[str, Any]
    admission_current_seal_sha256: str
    admission_receipt_evidence: dict[str, Any]
    admission_receipt_seal_sha256: str
    source_audit_seal_sha256: str
    source_revision: str
    source_all_ten_outer_preflight_evidence: dict[str, Any]
    source_all_ten_outer_preflight_seal_sha256: str
    authority: dict[str, Any]
    authority_evidence: dict[str, Any]
    authority_seal_sha256: str
    component_outer: dict[str, Any]
    component_outer_evidence: dict[str, Any]
    component_outer_seal_sha256: str
    component_inner: dict[str, Any]
    component_inner_evidence: dict[str, Any]


@dataclass(frozen=True)
class ChildPreflightContext:
    document: dict[str, Any]
    evidence: dict[str, Any]
    seal_sha256: str
    handoff: HandoffContext


@dataclass(frozen=True)
class PreflightContext:
    capture_dir: Path
    child: ChildPreflightContext
    probe_binary: dict[str, Any]
    outer_preflight: dict[str, Any]
    outer_preflight_evidence: dict[str, Any]
    outer_preflight_seal_sha256: str
    proof: dict[str, Any]
    proof_evidence: dict[str, Any]
    proof_seal_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTokenL0StateHandoffLauncherError(f"{label} must be an object")
    return dict(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise SourceTokenL0StateHandoffLauncherError(f"{label} must be an absolute path")
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} must be one non-symlink regular file")
    if executable and not metadata.st_mode & stat.S_IXUSR:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} must be executable")
    return candidate.resolve(strict=True)


def _evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    canonical = _canonical_regular(path, label, executable=executable)
    try:
        raw = canonical.read_bytes()
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"cannot read {label}: {exc}") from exc
    return {"path": str(canonical), "present": True, "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    canonical = _canonical_regular(path, label)
    raw = canonical.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} exceeds bounded JSON size")
    try:
        document = _mapping(json.loads(raw.decode("utf-8")), label)
        verified = verify(document, label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} is not a valid sealed JSON document: {exc}") from exc
    seal_sha256 = verified.get("seal_sha256")
    if not _is_sha(seal_sha256):
        raise SourceTokenL0StateHandoffLauncherError(f"{label} has an invalid seal")
    return verified, str(seal_sha256)


def _unsealed_json(path: Path, label: str) -> dict[str, Any]:
    canonical = _canonical_regular(path, label)
    raw = canonical.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} exceeds bounded JSON size")
    try:
        return _mapping(json.loads(raw.decode("utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"{label} is not JSON") from exc


def _binding(evidence: Mapping[str, Any], seal_sha256: str) -> dict[str, Any]:
    if not _is_sha(seal_sha256):
        raise SourceTokenL0StateHandoffLauncherError("binding needs a SHA-256 seal")
    return {**dict(evidence), "seal_sha256": seal_sha256}


def _exact(value: object, expected: Mapping[str, Any], label: str) -> None:
    if _mapping(value, label) != dict(expected):
        raise SourceTokenL0StateHandoffLauncherError(f"{label} drifted from exact evidence")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists():
        raise SourceTokenL0StateHandoffLauncherError(f"refusing non-unique output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"cannot create {path}: {exc}") from exc


def _new_capture_dir(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate == candidate.parent or candidate == REPO_ROOT or candidate.exists():
        raise SourceTokenL0StateHandoffLauncherError(f"{label} must be a unique, bounded absolute directory")
    try:
        candidate.mkdir(parents=True, mode=0o700)
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(f"cannot create {label}: {exc}") from exc
    return candidate.resolve(strict=True)


def _validate_source(binding: Mapping[str, Any], label: str) -> None:
    expected = {
        "model_key": "qwen80",
        "source_token_id": 1,
        "source_revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        "manifest_document_sha256": "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10",
        "manifest_seal_sha256": "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b",
        "admission_receipt_seal_sha256": "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628",
    }
    for key, item in expected.items():
        if binding.get(key) != item:
            raise SourceTokenL0StateHandoffLauncherError(f"{label}.{key} drifted")


def _handoff_contract() -> dict[str, Any]:
    """The capture is an L0 post-state handoff, never an L1 execution proof."""
    return {
        "source_token_id": 1,
        "prefix_dispatches": 9,
        "suffix_dispatches": 14,
        "total_dispatches": 23,
        "same_tcb_fence_required": True,
        "l1_prefix_dispatches": 0,
        "l1_binding_not_executed": True,
        "strict_claim_boundary": {
            "l1_executed": False,
            "l1_deltanet_dispatch_performed": False,
            "complete_layer_or_token_claim": False,
            "decoder_generation_hcli_tps_tg_or_tournament_claim": False,
            "may_not_satisfy_next_layer_execution_dependency": True,
        },
    }


def _validate_source_all_ten_outer_preflight(
    *,
    manifest_evidence: Mapping[str, Any],
    manifest_seal_sha256: str,
    admission_receipt_evidence: Mapping[str, Any],
    admission_receipt_seal_sha256: str,
) -> tuple[dict[str, Any], str]:
    canonical = _canonical_regular(
        SOURCE_ALL_TEN_OUTER_PREFLIGHT_PATH,
        "source all-ten outer preflight",
    )
    evidence = _evidence(canonical, "source all-ten outer preflight")
    if (
        evidence["bytes"] != SOURCE_ALL_TEN_OUTER_PREFLIGHT_BYTES
        or evidence["sha256"] != SOURCE_ALL_TEN_OUTER_PREFLIGHT_RAW_SHA256
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "source all-ten outer preflight raw evidence drifted"
        )
    document, seal_sha256 = _sealed_json(canonical, "source all-ten outer preflight")
    if (
        document.get("schema") != SOURCE_ALL_TEN_OUTER_PREFLIGHT_SCHEMA
        or document.get("status") != SOURCE_ALL_TEN_OUTER_PREFLIGHT_STATUS
        or seal_sha256 != SOURCE_ALL_TEN_OUTER_PREFLIGHT_SEAL_SHA256
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "source all-ten outer preflight schema/status/seal drifted"
        )
    binding = _mapping(document.get("source_binding"), "source all-ten binding")
    _exact(binding.get("manifest"), manifest_evidence, "source all-ten manifest")
    if binding.get("manifest_seal_sha256") != manifest_seal_sha256:
        raise SourceTokenL0StateHandoffLauncherError("source all-ten manifest seal drifted")
    _exact(
        binding.get("admission_receipt"),
        admission_receipt_evidence,
        "source all-ten admission receipt",
    )
    if binding.get("admission_receipt_seal_sha256") != admission_receipt_seal_sha256:
        raise SourceTokenL0StateHandoffLauncherError(
            "source all-ten admission receipt seal drifted"
        )
    boundary = _mapping(document.get("claim_boundary"), "source all-ten claim boundary")
    if (
        boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("lease_issued") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim")
        is not True
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "source all-ten outer preflight scope drifted"
        )
    return evidence, seal_sha256


def _source_binding(child: ChildPreflightContext) -> dict[str, Any]:
    """Exact names consumed by the future child; preserve all source lineage."""
    handoff = child.handoff
    return {
        "manifest": handoff.manifest_evidence,
        "manifest_seal_sha256": handoff.manifest_seal_sha256,
        "admission_current": handoff.admission_current_evidence,
        "admission_pointer_seal_sha256": handoff.admission_current_seal_sha256,
        "admission_receipt": handoff.admission_receipt_evidence,
        "admission_receipt_seal_sha256": handoff.admission_receipt_seal_sha256,
        "source_audit_seal_sha256": handoff.source_audit_seal_sha256,
        "source_revision": handoff.source_revision,
        "source_all_ten_outer_preflight": handoff.source_all_ten_outer_preflight_evidence,
        "source_all_ten_outer_preflight_seal_sha256": handoff.source_all_ten_outer_preflight_seal_sha256,
        "l0_state_handoff_child_preflight": child.evidence,
        "l0_state_handoff_child_preflight_seal_sha256": child.seal_sha256,
        "baseline_l0_to_l1_handoff_authority": handoff.authority_evidence,
        "baseline_l0_to_l1_handoff_authority_seal_sha256": handoff.authority_seal_sha256,
    }


def _versioned_current_acceptance() -> dict[str, bool]:
    """The only mutable authority is the versioned-current pointer receipt."""
    return {
        "canonical_pointer_path_required": True,
        "pointer_reseal_allowed_only_when_immutable_authority_is_exact": True,
        "immutable_manifest_sha_and_seal_must_remain_exact": True,
        "immutable_admission_receipt_sha_and_seal_must_remain_exact": True,
        "manifest_or_receipt_substitution_accepted": False,
    }


def _versioned_current_observation(
    child: ChildPreflightContext, *, phase: str
) -> dict[str, Any]:
    """Seal the pointer identity observed for a preflight or future launch."""
    handoff = child.handoff
    return {
        "phase": phase,
        "canonical_pointer_path": str(ADMISSION_CURRENT_PATH.resolve(strict=True)),
        "observed_pointer_evidence": handoff.admission_current_evidence,
        "observed_pointer_seal_sha256": handoff.admission_current_seal_sha256,
        "immutable_manifest": _binding(
            handoff.manifest_evidence, handoff.manifest_seal_sha256
        ),
        "immutable_admission_receipt": _binding(
            handoff.admission_receipt_evidence,
            handoff.admission_receipt_seal_sha256,
        ),
        "acceptance": _versioned_current_acceptance(),
    }


def _terminal_versioned_current_recheck(child: ChildPreflightContext) -> dict[str, Any]:
    return {
        "canonical_pointer_path": str(ADMISSION_CURRENT_PATH.resolve(strict=True)),
        "terminal_pointer_raw_and_seal_evidence_required": True,
        "pointer_reseal_allowed_only_when_immutable_authority_is_exact": True,
        "immutable_manifest": _binding(
            child.handoff.manifest_evidence,
            child.handoff.manifest_seal_sha256,
        ),
        "immutable_admission_receipt": _binding(
            child.handoff.admission_receipt_evidence,
            child.handoff.admission_receipt_seal_sha256,
        ),
    }


def _validate_historical_pointer_evidence(value: object, label: str) -> dict[str, Any]:
    evidence = _mapping(value, label)
    expected_path = str(ADMISSION_CURRENT_PATH.resolve(strict=True))
    if (
        set(evidence) != {"path", "present", "bytes", "sha256"}
        or evidence.get("path") != expected_path
        or evidence.get("present") is not True
        or isinstance(evidence.get("bytes"), bool)
        or not isinstance(evidence.get("bytes"), int)
        or evidence["bytes"] <= 0
        or not _is_sha(evidence.get("sha256"))
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label} is not an exact canonical historical pointer observation"
        )
    return evidence


def _validate_versioned_current_source_binding(
    observed: object, child: ChildPreflightContext, label: str
) -> None:
    """Accept a reseal only; every immutable lineage field remains exact."""
    actual = _mapping(observed, label)
    current = _source_binding(child)
    if set(actual) != set(current):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label} has an unrecognized source-binding field"
        )
    for field, expected in current.items():
        if field in {"admission_current", "admission_pointer_seal_sha256"}:
            continue
        if actual.get(field) != expected:
            raise SourceTokenL0StateHandoffLauncherError(
                f"{label} immutable authority {field} drifted"
            )
    _validate_historical_pointer_evidence(
        actual.get("admission_current"), f"{label}.admission_current"
    )
    if not _is_sha(actual.get("admission_pointer_seal_sha256")):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label}.admission_pointer_seal_sha256 is malformed"
        )


def _validate_versioned_current_observation(
    value: object, child: ChildPreflightContext, *, phase: str, label: str
) -> None:
    observation = _mapping(value, label)
    handoff = child.handoff
    if observation.get("phase") != phase:
        raise SourceTokenL0StateHandoffLauncherError(f"{label}.phase drifted")
    if observation.get("canonical_pointer_path") != str(
        ADMISSION_CURRENT_PATH.resolve(strict=True)
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label}.canonical_pointer_path drifted"
        )
    _validate_historical_pointer_evidence(
        observation.get("observed_pointer_evidence"),
        f"{label}.observed_pointer_evidence",
    )
    if not _is_sha(observation.get("observed_pointer_seal_sha256")):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label}.observed_pointer_seal_sha256 is malformed"
        )
    _exact(
        observation.get("immutable_manifest"),
        _binding(handoff.manifest_evidence, handoff.manifest_seal_sha256),
        f"{label}.immutable_manifest",
    )
    _exact(
        observation.get("immutable_admission_receipt"),
        _binding(
            handoff.admission_receipt_evidence,
            handoff.admission_receipt_seal_sha256,
        ),
        f"{label}.immutable_admission_receipt",
    )
    _exact(observation.get("acceptance"), _versioned_current_acceptance(), f"{label}.acceptance")


def _validate_observation_matches_source_binding(
    observation: object, source_binding: object, label: str
) -> None:
    observed = _mapping(observation, f"{label} observation")
    source = _mapping(source_binding, f"{label} source binding")
    _exact(
        observed.get("observed_pointer_evidence"),
        _mapping(source.get("admission_current"), f"{label} source pointer"),
        f"{label} pointer evidence",
    )
    if observed.get("observed_pointer_seal_sha256") != source.get(
        "admission_pointer_seal_sha256"
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            f"{label} pointer seal differs from source binding"
        )


def validate_handoff_authority(path: Path = HANDOFF_AUTHORITY_PATH) -> HandoffContext:
    canonical = _canonical_regular(path, "--handoff-authority")
    if canonical != HANDOFF_AUTHORITY_PATH.resolve(strict=True):
        raise SourceTokenL0StateHandoffLauncherError("--handoff-authority must be the immutable current authority")
    authority_evidence = _evidence(canonical, "--handoff-authority")
    if authority_evidence["bytes"] != HANDOFF_AUTHORITY_BYTES or authority_evidence["sha256"] != HANDOFF_AUTHORITY_RAW_SHA256:
        raise SourceTokenL0StateHandoffLauncherError("handoff authority raw evidence drifted")
    authority, authority_seal = _sealed_json(canonical, "--handoff-authority")
    if (
        authority.get("schema") != HANDOFF_AUTHORITY_SCHEMA
        or authority.get("status") != HANDOFF_AUTHORITY_STATUS
        or authority_seal != HANDOFF_AUTHORITY_SEAL_SHA256
        or authority.get("ready_for_l1_device_handoff") is not False
        or authority.get("component_only") is not True
    ):
        raise SourceTokenL0StateHandoffLauncherError("handoff authority schema/status/seal/readiness drifted")
    source = _mapping(authority.get("source_binding"), "handoff source")
    _validate_source(source, "handoff source")
    manifest_evidence = _evidence(MANIFEST_PATH, "current Qwen80 manifest")
    if manifest_evidence["sha256"] != source["manifest_document_sha256"]:
        raise SourceTokenL0StateHandoffLauncherError("current Qwen80 manifest SHA drifted")
    admission_current, admission_current_seal = _sealed_json(
        ADMISSION_CURRENT_PATH, "current Qwen80 admission"
    )
    admission_current_evidence = _evidence(ADMISSION_CURRENT_PATH, "current Qwen80 admission")
    selected_manifest = _mapping(admission_current.get("complete_manifest"), "current admission manifest")
    selected_receipt = _mapping(admission_current.get("admission_receipt"), "current admission receipt")
    if (
        admission_current.get("schema") != ADMISSION_CURRENT_SCHEMA
        or selected_manifest.get("path") != str(MANIFEST_PATH)
        or selected_manifest.get("document_sha256") != source["manifest_document_sha256"]
        or selected_manifest.get("seal_sha256") != source["manifest_seal_sha256"]
        or selected_receipt.get("seal_sha256") != source["admission_receipt_seal_sha256"]
        or admission_current.get("status") != "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
    ):
        raise SourceTokenL0StateHandoffLauncherError("current manifest/admission binding drifted")
    admission_receipt_path = _canonical_regular(
        Path(str(selected_receipt.get("path"))), "current admission receipt"
    )
    admission_receipt_evidence = _evidence(
        admission_receipt_path, "current admission receipt"
    )
    if admission_receipt_evidence["sha256"] != selected_receipt.get("document_sha256"):
        raise SourceTokenL0StateHandoffLauncherError("current admission receipt raw evidence drifted")
    admission_receipt, admission_receipt_seal = _sealed_json(
        admission_receipt_path, "current admission receipt"
    )
    if (
        admission_receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA
        or admission_receipt.get("status") != ADMISSION_RECEIPT_STATUS
        or admission_receipt_seal != source["admission_receipt_seal_sha256"]
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "current admission receipt schema/status/seal drifted"
        )
    receipt_manifest = _mapping(
        admission_receipt.get("complete_manifest"), "admission receipt manifest"
    )
    receipt_source = _mapping(
        admission_receipt.get("current_source_revalidation"), "admission receipt source"
    )
    if (
        receipt_manifest.get("path") != str(MANIFEST_PATH)
        or receipt_manifest.get("document_sha256") != manifest_evidence["sha256"]
        or receipt_manifest.get("seal_sha256") != source["manifest_seal_sha256"]
        or receipt_source.get("revision") != source["source_revision"]
        or receipt_source.get("source_audit_seal_sha256") != SOURCE_AUDIT_SEAL_SHA256
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "current admission receipt source lineage drifted"
        )
    source_all_ten_evidence, source_all_ten_seal = _validate_source_all_ten_outer_preflight(
        manifest_evidence=manifest_evidence,
        manifest_seal_sha256=str(source["manifest_seal_sha256"]),
        admission_receipt_evidence=admission_receipt_evidence,
        admission_receipt_seal_sha256=admission_receipt_seal,
    )
    consumed = _mapping(authority.get("consumed_component_capture"), "handoff consumed component")
    if (
        consumed.get("layer") != 0
        or consumed.get("linear_state_slot") != 0
        or _mapping(consumed.get("same_command_graph"), "handoff graph").get("prefix_dispatches") != 9
        or _mapping(consumed.get("same_command_graph"), "handoff graph").get("suffix_dispatches") != 14
        or _mapping(consumed.get("same_command_graph"), "handoff graph").get("total_dispatches") != 23
    ):
        raise SourceTokenL0StateHandoffLauncherError("handoff authority L0 9+14 evidence drifted")
    residual = _mapping(consumed.get("second_residual"), "handoff second residual")
    if residual != {"bytes": 8192, "elements": 2048, "f32le_sha256": "dae9e008fc5fb62ff3e10f8c89e2623b4c79c7b5fc44af5e1478923ec748f85c"}:
        raise SourceTokenL0StateHandoffLauncherError("handoff authority second residual baseline drifted")
    outer_evidence = _mapping(consumed.get("outer_terminal"), "handoff outer terminal evidence")
    inner_evidence = _mapping(consumed.get("inner_receipt"), "handoff inner receipt evidence")
    component_outer_path = _canonical_regular(Path(str(outer_evidence.get("path"))), "earned L0 outer terminal")
    component_inner_path = _canonical_regular(Path(str(inner_evidence.get("path"))), "earned L0 inner receipt")
    actual_outer = _evidence(component_outer_path, "earned L0 outer terminal")
    actual_inner = _evidence(component_inner_path, "earned L0 inner receipt")
    if actual_outer != outer_evidence or actual_inner != inner_evidence:
        raise SourceTokenL0StateHandoffLauncherError("earned L0 component raw evidence drifted")
    component_outer, component_outer_seal = _sealed_json(component_outer_path, "earned L0 outer terminal")
    if (
        component_outer.get("schema") != COMPONENT_OUTER_SCHEMA
        or component_outer.get("status") != COMPONENT_OUTER_STATUS
        or component_outer_seal != consumed.get("outer_terminal_seal_sha256")
    ):
        raise SourceTokenL0StateHandoffLauncherError("earned L0 outer terminal schema/status/seal drifted")
    component_inner = _unsealed_json(component_inner_path, "earned L0 inner receipt")
    if (
        component_inner.get("schema") != COMPONENT_INNER_SCHEMA
        or component_inner.get("status") != COMPONENT_INNER_STATUS
        or component_inner.get("component_only") is not True
        or component_inner.get("complete_layer_or_token_performed") is not False
    ):
        raise SourceTokenL0StateHandoffLauncherError("earned L0 inner receipt scope drifted")
    return HandoffContext(
        manifest_evidence,
        str(source["manifest_seal_sha256"]),
        admission_current_evidence,
        admission_current_seal,
        admission_receipt_evidence,
        admission_receipt_seal,
        SOURCE_AUDIT_SEAL_SHA256,
        str(source["source_revision"]),
        source_all_ten_evidence,
        source_all_ten_seal,
        authority,
        authority_evidence,
        authority_seal,
        component_outer,
        actual_outer,
        component_outer_seal,
        component_inner,
        actual_inner,
    )


def validate_child_preflight(path: Path = CHILD_PREFLIGHT_PATH, *, handoff_authority: Path = HANDOFF_AUTHORITY_PATH) -> ChildPreflightContext:
    handoff = validate_handoff_authority(handoff_authority)
    canonical = _canonical_regular(path, "--child-preflight")
    if canonical != CHILD_PREFLIGHT_PATH.resolve(strict=True):
        raise SourceTokenL0StateHandoffLauncherError("--child-preflight must be the immutable current child preflight")
    evidence = _evidence(canonical, "--child-preflight")
    if evidence["bytes"] != CHILD_PREFLIGHT_BYTES or evidence["sha256"] != CHILD_PREFLIGHT_RAW_SHA256:
        raise SourceTokenL0StateHandoffLauncherError("child preflight raw evidence drifted")
    document, seal_sha256 = _sealed_json(canonical, "--child-preflight")
    if document.get("schema") != CHILD_PREFLIGHT_SCHEMA or document.get("status") != CHILD_PREFLIGHT_STATUS or seal_sha256 != CHILD_PREFLIGHT_SEAL_SHA256 or document.get("mode") != "cpu_only_preflight":
        raise SourceTokenL0StateHandoffLauncherError("child preflight schema/status/seal/mode drifted")
    _exact(document.get("baseline_handoff_authority"), handoff.authority_evidence, "child baseline handoff evidence")
    if document.get("baseline_handoff_authority_seal_sha256") != handoff.authority_seal_sha256:
        raise SourceTokenL0StateHandoffLauncherError("child baseline handoff seal drifted")
    _validate_source(_mapping(document.get("source_binding"), "child source"), "child source")
    fresh = _mapping(document.get("fresh_capture_required"), "child fresh capture")
    graph = _mapping(fresh.get("same_token_command_buffer"), "child same-token graph")
    if fresh.get("must_reencode_source_token_l0") is not True or fresh.get("may_not_reuse_historical_component_output_hash_as_a_live_buffer") is not True or graph != {"fence_once_after_prefix_and_suffix": True, "prefix_dispatches": 9, "suffix_dispatches": 14, "total_dispatches": 23}:
        raise SourceTokenL0StateHandoffLauncherError("child fresh L0 9+14 contract drifted")
    planned = _mapping(document.get("planned_pre_l1_handoff_capture"), "child planned pre-L1 capture")
    if planned.get("schema") != PRE_L1_CAPTURE_SCHEMA or planned.get("status") != PRE_L1_CAPTURE_STATUS or planned.get("l1_binding_not_executed") is not True or planned.get("l1_prefix_dispatches") != 0 or planned.get("may_not_satisfy_next_layer_execution_dependency") is not True or planned.get("receipt_must_record_l0_post_state_rollback_and_retained_output") is not True:
        raise SourceTokenL0StateHandoffLauncherError("child planned pre-L1 capture boundary drifted")
    witness = _mapping(document.get("required_next_layer_handoff_witness"), "child remaining next-layer witness")
    if witness.get("schema") != HANDOFF_WITNESS_SCHEMA or witness.get("status") != HANDOFF_WITNESS_STATUS:
        raise SourceTokenL0StateHandoffLauncherError("child next-layer witness schema/status drifted")
    retained = _mapping(witness.get("retained_l0_second_residual"), "child retained residual")
    layer1 = _mapping(witness.get("layer1_input_binding"), "child Layer1 binding")
    if witness.get("remains_required_after_planned_pre_l1_capture") is not True or retained.get("bytes") != 8192 or retained.get("elements") != 2048 or retained.get("device_buffer_id_required") is not True or retained.get("future_layer1_execution_retention_required") is not True or layer1.get("layer") != 1 or layer1.get("linear_state_slot") != 1 or layer1.get("input_device_buffer_id_must_equal_retained_l0_output") is not True or layer1.get("input_f32le_sha256_must_equal_retained_l0_output") is not True or layer1.get("same_command_graph_retained_required") is not True:
        raise SourceTokenL0StateHandoffLauncherError("child retained-output/Layer1 handoff contract drifted")
    boundary = _mapping(document.get("implementation_boundary"), "child implementation boundary")
    claim = _mapping(document.get("claim_boundary"), "child claim boundary")
    if boundary.get("device_context_or_dispatch_performed") is not False or boundary.get("outer_reaped_receipt_last_replay_guard_required_before_future_metal_mode") is not True or claim.get("watcher_or_server_transition_not_authorized") is not True or claim.get("component_only_even_after_a_future_pass") is not True or claim.get("planned_pre_l1_capture_is_not_l1_execution") is not True or claim.get("planned_pre_l1_capture_may_not_promote_the_next_layer_dependency") is not True:
        raise SourceTokenL0StateHandoffLauncherError("child CPU-only/reaper boundary drifted")
    return ChildPreflightContext(document, evidence, seal_sha256, handoff)


def _probe(path: Path) -> dict[str, Any]:
    probe = _canonical_regular(path, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise SourceTokenL0StateHandoffLauncherError(f"--probe-bin must be {EXPECTED_PROBE_BASENAME}")
    evidence = _evidence(probe, "--probe-bin", executable=True)
    if evidence["sha256"] != EXPECTED_PROBE_SHA256:
        raise SourceTokenL0StateHandoffLauncherError("--probe-bin SHA-256 drifted")
    return evidence


def _validate_cpu_child_preflight_document(
    document: object,
    *,
    outer_evidence: Mapping[str, Any],
    outer_seal_sha256: str,
) -> dict[str, Any]:
    parsed = _mapping(document, "CPU child preflight stdout")
    if (
        parsed.get("schema") != CPU_CHILD_PREFLIGHT_SCHEMA
        or parsed.get("status") != CPU_CHILD_PREFLIGHT_STATUS
        or parsed.get("mode") != "preflight"
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "CPU child preflight schema/status/mode drifted"
        )
    outer_binding = _mapping(
        parsed.get("outer_preflight_binding"), "CPU child outer-preflight binding"
    )
    expected_outer_binding = {
        "path": outer_evidence["path"],
        "document_sha256": outer_evidence["sha256"],
        "seal_sha256": outer_seal_sha256,
    }
    _exact(outer_binding, expected_outer_binding, "CPU child outer-preflight binding")
    graph = _mapping(
        parsed.get("same_command_graph_contract"), "CPU child graph contract"
    )
    for field, expected in {
        "source_token_id": 1,
        "prefix_dispatches": 9,
        "suffix_dispatches": 14,
        "total_dispatches": 23,
        "l1_prefix_dispatches": 0,
        "l1_binding_not_executed": True,
        "retained_l0_second_residual_elements": 2048,
        "retained_l0_second_residual_bytes": 8192,
        "l0_slot": 0,
        "l1_slot": 1,
    }.items():
        if graph.get(field) != expected:
            raise SourceTokenL0StateHandoffLauncherError(
                f"CPU child graph contract {field} drifted"
            )
    claim = _mapping(parsed.get("claim_boundary"), "CPU child claim boundary")
    for field in (
        "metal_device_or_dispatch_performed",
        "lease_issued",
        "l1_prefix_executed",
        "complete_layer_or_token_performed",
    ):
        if claim.get(field) is not False:
            raise SourceTokenL0StateHandoffLauncherError(
                f"CPU child claim boundary {field} drifted"
            )
    if (
        claim.get("cannot_satisfy_next_layer_execution_dependency") is not True
        or claim.get("no_decoder_generation_server_hcli_tps_tg_or_tournament_claim")
        is not True
    ):
        raise SourceTokenL0StateHandoffLauncherError("CPU child claim scope drifted")
    return parsed


def _run_cpu_child_preflight(
    *,
    probe_bin: Path,
    outer_preflight_path: Path,
    outer_evidence: Mapping[str, Any],
    outer_seal_sha256: str,
    capture_dir: Path,
) -> dict[str, Any]:
    """Run exactly one bounded CPU-only child preflight and reap it.

    The command intentionally has no lease, capture, Metal, or device-mode
    arguments.  Its persisted streams become sealed proof inputs; a failed or
    noisy child never produces a usable proof.
    """
    executable = _canonical_regular(probe_bin, "--probe-bin", executable=True)
    stdout_path = capture_dir / CHILD_PREFLIGHT_STDOUT_FILENAME
    stderr_path = capture_dir / CHILD_PREFLIGHT_STDERR_FILENAME
    if stdout_path.exists() or stderr_path.exists():
        raise SourceTokenL0StateHandoffLauncherError("CPU child stream path already exists")
    command = [
        str(executable),
        "--outer-preflight",
        str(outer_preflight_path),
        "--mode",
        "preflight",
        "--workers",
        "1",
    ]
    try:
        stdout_descriptor = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        stderr_descriptor = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(
            f"cannot create CPU child stream files: {exc}"
        ) from exc
    try:
        with os.fdopen(stdout_descriptor, "wb") as stdout_handle, os.fdopen(
            stderr_descriptor, "wb"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
            )
            try:
                exit_code = process.wait(timeout=CPU_CHILD_PREFLIGHT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise SourceTokenL0StateHandoffLauncherError(
                    "CPU child preflight timed out and was reaped"
                ) from exc
    except OSError as exc:
        raise SourceTokenL0StateHandoffLauncherError(
            f"cannot run CPU child preflight: {exc}"
        ) from exc
    stdout_evidence = _evidence(stdout_path, "CPU child stdout")
    stderr_evidence = _evidence(stderr_path, "CPU child stderr")
    if (
        stdout_evidence["bytes"] > MAX_JSON_BYTES
        or stderr_evidence["bytes"] > MAX_JSON_BYTES
    ):
        raise SourceTokenL0StateHandoffLauncherError("CPU child stream exceeded bound")
    if exit_code != 0 or stderr_evidence["bytes"] != 0:
        raise SourceTokenL0StateHandoffLauncherError(
            "CPU child preflight did not exit cleanly with empty stderr"
        )
    try:
        parsed = _validate_cpu_child_preflight_document(
            json.loads(stdout_path.read_text(encoding="utf-8")),
            outer_evidence=outer_evidence,
            outer_seal_sha256=outer_seal_sha256,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenL0StateHandoffLauncherError(
            "CPU child preflight stdout is not bounded JSON"
        ) from exc
    return {
        "command": command,
        "exit_code": exit_code,
        "reaped": True,
        "stdout": stdout_evidence,
        "stderr": stderr_evidence,
        "stderr_bytes": stderr_evidence["bytes"],
        "parsed": parsed,
    }


def _outer_preflight(child: ChildPreflightContext, probe_binary: Mapping[str, Any]) -> dict[str, Any]:
    return seal({
        "schema": OUTER_PREFLIGHT_SCHEMA,
        "status": OUTER_PREFLIGHT_STATUS,
        "recorded_at": _utc_now(),
        "source_binding": _source_binding(child),
        "versioned_current_admission": _versioned_current_observation(
            child, phase="preflight"
        ),
        "handoff_contract": _handoff_contract(),
        "handoff_authority": _binding(child.handoff.authority_evidence, child.handoff.authority_seal_sha256),
        "earned_l0_component_outer": _binding(child.handoff.component_outer_evidence, child.handoff.component_outer_seal_sha256),
        "earned_l0_component_inner": child.handoff.component_inner_evidence,
        "child_preflight": _binding(child.evidence, child.seal_sha256),
        "probe_binary": dict(probe_binary),
        "future_execution_contract": {
            "fresh_component_lease_required": True,
            "qwen30_no_device_required": True,
            "watcher_hold_must_remain_active": True,
            "outer_reaped_one_child_required": True,
            "terminal_receipt_written_last_required": True,
            "unique_capture_dirs_required": True,
            "lease_id_binding_and_replay_refusal_required": True,
            "automatic_retry_prohibited": True,
            "child_and_probe_sha256_exactly_bound": True,
        },
        "required_pre_l1_handoff_capture": {
            "schema": PRE_L1_CAPTURE_SCHEMA,
            "status": PRE_L1_CAPTURE_STATUS,
            "source_token_id": 1,
            "prefix_dispatches": 9,
            "suffix_dispatches": 14,
            "total_dispatches": 23,
            "same_tcb_fence_required": True,
            "retained_l0_second_residual_elements": 2048,
            "retained_l0_second_residual_bytes": 8192,
            "retained_l0_second_residual_buffer_id_required": True,
            "l0_active_and_rollback_conv_recurrent_hashes_required": True,
            "l1_slot1_input_buffer_and_sha_equal_retained_l0_output_required": True,
            "l1_slot1_state_allocations_non_aliased_required": True,
            "l1_binding_not_executed": True,
            "l1_prefix_dispatches": 0,
            "may_not_satisfy_next_layer_execution_dependency": True,
            "l1_deltanet_dispatch_allowed": False,
            "pre_l1_component_only": True,
        },
        "next_layer_execution_dependency": {"schema": HANDOFF_WITNESS_SCHEMA, "required_status": HANDOFF_WITNESS_STATUS, "remains_unmet_after_this_capture": True, "may_not_be_created_or_accepted_by_this_outer": True},
        "claim_boundary": {"cpu_only": True, "metal_device_or_dispatch_performed": False, "lease_issued": False, "device_child_spawned": False, "watcher_or_server_started": False, "l1_prefix_executed": False, "complete_layer_or_token_performed": False, "complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": False},
    })


def _terminal_plan(
    capture: Path,
    outer_evidence: Mapping[str, Any],
    outer_seal: str,
    child: ChildPreflightContext,
) -> dict[str, Any]:
    return seal({
        "schema": TERMINAL_PLAN_SCHEMA,
        "status": TERMINAL_PLAN_STATUS,
        "recorded_at": _utc_now(),
        "outer_preflight": _binding(outer_evidence, outer_seal),
        "preflight_versioned_current_admission": _versioned_current_observation(
            child, phase="preflight"
        ),
        "terminal_versioned_current_recheck": _terminal_versioned_current_recheck(child),
        "planned_outer_capture_dir": str(capture),
        "planned_inner_capture_dir": str(capture / INNER_CAPTURE_DIRNAME),
        "planned_terminal_receipt_path": str(capture / TERMINAL_RECEIPT_FILENAME),
        "receipt_last_contract": {"outer_reaps_exactly_one_child_before_terminal_receipt": True, "terminal_receipt_written_last_is_completion_marker": True, "automatic_retry_prohibited": True, "lease_reuse_prohibited_after_terminal": True},
        "terminal_receipt_exists": False,
        "claim_boundary": {"plan_only": True, "device_child_spawned": False, "metal_device_or_dispatch_performed": False, "lease_issued": False, "l1_prefix_executed": False, "complete_layer_or_token_performed": False, "terminal_receipt_written": False},
    })


def _proof(
    child: ChildPreflightContext,
    probe_binary: Mapping[str, Any],
    outer: Mapping[str, Any],
    outer_evidence: Mapping[str, Any],
    terminal: Mapping[str, Any],
    terminal_evidence: Mapping[str, Any],
    cpu_child_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return seal({
        "schema": PREFLIGHT_PROOF_SCHEMA,
        "status": PREFLIGHT_PROOF_STATUS,
        "recorded_at": _utc_now(),
        "source_binding": {
            "probe_binary": dict(probe_binary),
            "outer_preflight": {
                "path": outer_evidence["path"],
                "sha256": outer_evidence["sha256"],
                "seal_sha256": str(outer["seal_sha256"]),
            },
        },
        "outer_source_binding": _source_binding(child),
        "versioned_current_admission": _versioned_current_observation(
            child, phase="preflight"
        ),
        "handoff_contract": _handoff_contract(),
        "handoff_authority": _binding(child.handoff.authority_evidence, child.handoff.authority_seal_sha256),
        "earned_l0_component_outer": _binding(child.handoff.component_outer_evidence, child.handoff.component_outer_seal_sha256),
        "earned_l0_component_inner": child.handoff.component_inner_evidence,
        "l0_state_handoff_child_preflight": _binding(child.evidence, child.seal_sha256),
        "child_preflight": dict(cpu_child_preflight),
        "probe_binary": dict(probe_binary),
        "outer_preflight": _binding(outer_evidence, str(outer["seal_sha256"])),
        "terminal_plan": _binding(terminal_evidence, str(terminal["seal_sha256"])),
        "future_execution_contract": {"receipt_last": True, "fresh_lease_id_required": True, "lease_id_replay_refused": True, "one_child_outer_reaped": True, "automatic_retry_prohibited": True, "no_implicit_device_action": True},
        "claim_boundary": {"cpu_only": True, "cpu_preflight_child_spawned": True, "cpu_preflight_child_reaped": True, "device_child_spawned": False, "metal_device_or_dispatch_performed": False, "lease_issued": False, "l1_prefix_executed": False, "complete_layer_or_token_performed": False, "terminal_receipt_written": False},
    })


def _validate_cpu_child_preflight_proof(
    child_result: object,
    *,
    probe_binary: Mapping[str, Any],
    outer_evidence: Mapping[str, Any],
    outer_seal_sha256: str,
) -> None:
    result = _mapping(child_result, "proof CPU child preflight")
    expected_command = [
        probe_binary["path"],
        "--outer-preflight",
        outer_evidence["path"],
        "--mode",
        "preflight",
        "--workers",
        "1",
    ]
    if (
        result.get("command") != expected_command
        or result.get("exit_code") != 0
        or result.get("reaped") is not True
        or result.get("stderr_bytes") != 0
    ):
        raise SourceTokenL0StateHandoffLauncherError(
            "proof CPU child did not use the clean bounded preflight command"
        )
    stdout_claim = _mapping(result.get("stdout"), "proof CPU child stdout")
    stderr_claim = _mapping(result.get("stderr"), "proof CPU child stderr")
    stdout_path = _canonical_regular(Path(str(stdout_claim.get("path"))), "proof CPU child stdout")
    stderr_path = _canonical_regular(Path(str(stderr_claim.get("path"))), "proof CPU child stderr")
    _exact(stdout_claim, _evidence(stdout_path, "proof CPU child stdout"), "proof CPU child stdout")
    _exact(stderr_claim, _evidence(stderr_path, "proof CPU child stderr"), "proof CPU child stderr")
    if stderr_claim["bytes"] != 0 or stdout_claim["bytes"] > MAX_JSON_BYTES:
        raise SourceTokenL0StateHandoffLauncherError("proof CPU child stream boundary drifted")
    try:
        parsed_from_stdout = json.loads(stdout_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenL0StateHandoffLauncherError(
            "proof CPU child stdout is not JSON"
        ) from exc
    parsed = _validate_cpu_child_preflight_document(
        parsed_from_stdout,
        outer_evidence=outer_evidence,
        outer_seal_sha256=outer_seal_sha256,
    )
    _exact(result.get("parsed"), parsed, "proof CPU child parsed document")


def run_preflight_only(*, child_preflight: Path, handoff_authority: Path, probe_bin: Path, capture_dir: Path) -> dict[str, Any]:
    """Create/revalidate a CPU-only proof; never spawn a device child."""
    if capture_dir.exists():
        proof_path = capture_dir / PREFLIGHT_PROOF_FILENAME
        if not proof_path.is_file():
            raise SourceTokenL0StateHandoffLauncherError("existing preflight capture lacks proof; second plan refused")
        return validate_preflight_proof(proof_path=proof_path, child_preflight=child_preflight, handoff_authority=handoff_authority, probe_bin=probe_bin).proof
    child = validate_child_preflight(child_preflight, handoff_authority=handoff_authority)
    probe_binary = _probe(probe_bin)
    capture = _new_capture_dir(capture_dir, "--preflight-capture-dir")
    outer = _outer_preflight(child, probe_binary)
    outer_path = capture / OUTER_PREFLIGHT_FILENAME
    _write_new(outer_path, outer)
    outer_evidence = _evidence(outer_path, "outer preflight")
    terminal = _terminal_plan(
        capture, outer_evidence, str(outer["seal_sha256"]), child
    )
    terminal_path = capture / TERMINAL_PLAN_FILENAME
    _write_new(terminal_path, terminal)
    terminal_evidence = _evidence(terminal_path, "terminal plan")
    cpu_child_preflight = _run_cpu_child_preflight(
        probe_bin=probe_bin,
        outer_preflight_path=outer_path,
        outer_evidence=outer_evidence,
        outer_seal_sha256=str(outer["seal_sha256"]),
        capture_dir=capture,
    )
    proof = _proof(
        child,
        probe_binary,
        outer,
        outer_evidence,
        terminal,
        terminal_evidence,
        cpu_child_preflight,
    )
    _write_new(capture / PREFLIGHT_PROOF_FILENAME, proof)
    return proof


def validate_preflight_proof(*, proof_path: Path, child_preflight: Path, handoff_authority: Path, probe_bin: Path) -> PreflightContext:
    proof, proof_seal = _sealed_json(proof_path, "--preflight-proof")
    proof_evidence = _evidence(proof_path, "--preflight-proof")
    if proof.get("schema") != PREFLIGHT_PROOF_SCHEMA or proof.get("status") != PREFLIGHT_PROOF_STATUS:
        raise SourceTokenL0StateHandoffLauncherError("preflight proof schema/status drifted")
    child = validate_child_preflight(child_preflight, handoff_authority=handoff_authority)
    probe_binary = _probe(probe_bin)
    expected_handoff_contract = _handoff_contract()
    _validate_versioned_current_source_binding(
        proof.get("outer_source_binding"), child, "proof outer source binding"
    )
    _validate_versioned_current_observation(
        proof.get("versioned_current_admission"),
        child,
        phase="preflight",
        label="proof versioned-current admission",
    )
    _validate_observation_matches_source_binding(
        proof.get("versioned_current_admission"),
        proof.get("outer_source_binding"),
        "proof versioned-current admission",
    )
    _exact(proof.get("handoff_contract"), expected_handoff_contract, "proof handoff contract")
    _exact(proof.get("handoff_authority"), _binding(child.handoff.authority_evidence, child.handoff.authority_seal_sha256), "proof handoff")
    _exact(proof.get("earned_l0_component_outer"), _binding(child.handoff.component_outer_evidence, child.handoff.component_outer_seal_sha256), "proof L0 outer")
    _exact(proof.get("earned_l0_component_inner"), child.handoff.component_inner_evidence, "proof L0 inner")
    _exact(
        proof.get("l0_state_handoff_child_preflight"),
        _binding(child.evidence, child.seal_sha256),
        "proof static child",
    )
    _exact(proof.get("probe_binary"), probe_binary, "proof probe")
    outer_binding = _mapping(proof.get("outer_preflight"), "proof outer")
    outer_path = _canonical_regular(Path(str(outer_binding.get("path"))), "proof outer")
    outer, outer_seal = _sealed_json(outer_path, "proof outer")
    outer_evidence = _evidence(outer_path, "proof outer")
    _exact(outer_binding, _binding(outer_evidence, outer_seal), "proof outer binding")
    if outer.get("schema") != OUTER_PREFLIGHT_SCHEMA or outer.get("status") != OUTER_PREFLIGHT_STATUS:
        raise SourceTokenL0StateHandoffLauncherError("outer preflight schema/status drifted")
    _validate_versioned_current_source_binding(
        outer.get("source_binding"), child, "outer source binding"
    )
    _validate_versioned_current_observation(
        outer.get("versioned_current_admission"),
        child,
        phase="preflight",
        label="outer versioned-current admission",
    )
    _validate_observation_matches_source_binding(
        outer.get("versioned_current_admission"),
        outer.get("source_binding"),
        "outer versioned-current admission",
    )
    _exact(
        proof.get("outer_source_binding"),
        _mapping(outer.get("source_binding"), "outer source binding"),
        "proof/outer historical source binding",
    )
    _exact(
        proof.get("versioned_current_admission"),
        _mapping(outer.get("versioned_current_admission"), "outer versioned-current admission"),
        "proof/outer historical versioned-current observation",
    )
    _exact(outer.get("handoff_contract"), expected_handoff_contract, "outer handoff contract")
    _exact(outer.get("child_preflight"), _binding(child.evidence, child.seal_sha256), "outer child")
    _exact(outer.get("probe_binary"), probe_binary, "outer probe")
    expected_proof_source_binding = {
        "probe_binary": probe_binary,
        "outer_preflight": {
            "path": outer_evidence["path"],
            "sha256": outer_evidence["sha256"],
            "seal_sha256": outer_seal,
        },
    }
    _exact(
        proof.get("source_binding"),
        expected_proof_source_binding,
        "proof child source binding",
    )
    _validate_cpu_child_preflight_proof(
        proof.get("child_preflight"),
        probe_binary=probe_binary,
        outer_evidence=outer_evidence,
        outer_seal_sha256=outer_seal,
    )
    terminal_binding = _mapping(proof.get("terminal_plan"), "proof terminal plan")
    terminal_path = _canonical_regular(Path(str(terminal_binding.get("path"))), "proof terminal plan")
    terminal, terminal_seal = _sealed_json(terminal_path, "proof terminal plan")
    terminal_evidence = _evidence(terminal_path, "proof terminal plan")
    _exact(terminal_binding, _binding(terminal_evidence, terminal_seal), "proof terminal binding")
    capture = outer_path.parent
    if terminal.get("schema") != TERMINAL_PLAN_SCHEMA or terminal.get("status") != TERMINAL_PLAN_STATUS or terminal.get("planned_outer_capture_dir") != str(capture) or terminal.get("planned_terminal_receipt_path") != str(capture / TERMINAL_RECEIPT_FILENAME) or (capture / TERMINAL_RECEIPT_FILENAME).exists():
        raise SourceTokenL0StateHandoffLauncherError("receipt-last terminal plan drifted")
    _validate_versioned_current_observation(
        terminal.get("preflight_versioned_current_admission"),
        child,
        phase="preflight",
        label="terminal preflight versioned-current admission",
    )
    _exact(
        terminal.get("preflight_versioned_current_admission"),
        _mapping(outer.get("versioned_current_admission"), "outer versioned-current admission"),
        "terminal/outer historical versioned-current observation",
    )
    _exact(
        terminal.get("terminal_versioned_current_recheck"),
        _terminal_versioned_current_recheck(child),
        "terminal versioned-current recheck",
    )
    for label, document in (("proof", proof), ("outer", outer), ("terminal", terminal)):
        boundary = _mapping(document.get("claim_boundary"), f"{label} boundary")
        if (
            boundary.get("metal_device_or_dispatch_performed") is not False
            or boundary.get("lease_issued") is not False
            or boundary.get("device_child_spawned") is not False
            or boundary.get("l1_prefix_executed") is not False
            or boundary.get("complete_layer_or_token_performed") is not False
        ):
            raise SourceTokenL0StateHandoffLauncherError(f"{label} wrongly claims device activity")
    return PreflightContext(capture, child, probe_binary, outer, outer_evidence, outer_seal, proof, proof_evidence, proof_seal)


def _bind_future_lease(path: Path, context: PreflightContext) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    evidence = _evidence(path, "--lease-receipt")
    if document.get("schema") != FUTURE_LEASE_SCHEMA or document.get("status") != FUTURE_LEASE_STATUS or not _is_sha(document.get("lease_id")):
        raise SourceTokenL0StateHandoffLauncherError("future lease schema/status/ID drifted")
    _exact(document.get("outer_preflight"), context.outer_preflight_evidence, "lease outer preflight")
    if document.get("outer_preflight_seal_sha256") != context.outer_preflight_seal_sha256:
        raise SourceTokenL0StateHandoffLauncherError("lease outer preflight seal drifted")
    _exact(
        document.get("l0_state_handoff_child_preflight"),
        context.child.evidence,
        "lease child preflight",
    )
    if (
        document.get("l0_state_handoff_child_preflight_seal_sha256")
        != context.child.seal_sha256
    ):
        raise SourceTokenL0StateHandoffLauncherError("lease child preflight seal drifted")
    _exact(
        document.get("baseline_l0_to_l1_handoff_authority"),
        context.child.handoff.authority_evidence,
        "lease baseline handoff authority",
    )
    if (
        document.get("baseline_l0_to_l1_handoff_authority_seal_sha256")
        != context.child.handoff.authority_seal_sha256
    ):
        raise SourceTokenL0StateHandoffLauncherError("lease baseline handoff seal drifted")
    _exact(document.get("handoff_contract"), _handoff_contract(), "lease handoff contract")
    _exact(document.get("probe_binary"), context.probe_binary, "lease probe")
    lifecycle = _mapping(document.get("lifecycle"), "lease lifecycle")
    for field in ("fresh_for_this_exact_launch", "outer_reaped_capture_required", "lease_released_after_first_terminal_child", "automatic_retry_prohibited", "replay_guarded"):
        if lifecycle.get(field) is not True:
            raise SourceTokenL0StateHandoffLauncherError(f"lease lifecycle {field} drifted")
    policy = _mapping(document.get("execution_policy"), "lease policy")
    if (
        policy.get("component") != "qwen80_source_token_l0_state_handoff"
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("l1_prefix_execution_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise SourceTokenL0StateHandoffLauncherError("lease policy drifted")
    watcher = _mapping(document.get("watcher_coordination"), "lease watcher coordination")
    if (
        watcher.get("watcher_hold_must_remain_active") is not True
        or watcher.get("watcher_restart_or_transition_authorized") is not False
    ):
        raise SourceTokenL0StateHandoffLauncherError("lease watcher coordination drifted")
    return document, evidence, lease_seal, str(document["lease_id"])


def prepare_future_one_shot(
    *,
    preflight_proof: Path,
    child_preflight: Path,
    handoff_authority: Path,
    probe_bin: Path,
    lease_receipt: Path,
    capture_dir: Path,
    replay_guard_dir: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Reserve a future reaped one-shot; no child/process/device work occurs."""
    context = validate_preflight_proof(proof_path=preflight_proof, child_preflight=child_preflight, handoff_authority=handoff_authority, probe_bin=probe_bin)
    # Re-read the mutable pointer immediately before authority creation.  The
    # static child/manifest/receipt chain remains exact; only a valid pointer
    # reseal may differ from the preflight observation.
    launch_child = validate_child_preflight(
        child_preflight, handoff_authority=handoff_authority
    )
    if capture_dir.exists() or not replay_guard_dir.is_absolute():
        raise SourceTokenL0StateHandoffLauncherError("future capture must be new and replay guard dir absolute")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise SourceTokenL0StateHandoffLauncherError("future one-shot workers must be 1..4")
    lease, lease_evidence, lease_seal, lease_id = _bind_future_lease(lease_receipt, context)
    launch_observation = _versioned_current_observation(launch_child, phase="launch")
    identity = _sha256_bytes(json.dumps({"proof": _binding(context.proof_evidence, context.proof_seal_sha256), "child": _binding(context.child.evidence, context.child.seal_sha256), "probe": context.probe_binary, "lease": _binding(lease_evidence, lease_seal), "launch_versioned_current": launch_observation, "lease_id": lease_id, "capture": str(capture_dir), "workers": workers}, sort_keys=True, separators=(",", ":")).encode())
    replay_guard_dir.mkdir(parents=True, exist_ok=True)
    guard = replay_guard_dir / f"{lease_id}.json"
    _write_new(guard, seal({"schema": "hawking.ascension.qwen80_source_token_l0_state_handoff_lease_replay_guard.v1", "status": "RESERVED_ONE_SHOT_NOT_EXECUTED", "lease_id": lease_id, "launch_identity_sha256": identity, "replay_refused": True, "device_child_spawned": False}))
    capture = _new_capture_dir(capture_dir, "--capture-dir")
    authority = seal({
        "schema": OUTER_LAUNCH_AUTHORITY_SCHEMA, "status": OUTER_LAUNCH_AUTHORITY_STATUS, "recorded_at": _utc_now(), "launch_identity_sha256": identity, "lease_id": lease_id,
        "source_binding": _source_binding(launch_child), "handoff_contract": _handoff_contract(),
        "preflight_versioned_current_admission": _mapping(context.proof.get("versioned_current_admission"), "proof versioned-current admission"),
        "launch_versioned_current_admission": launch_observation,
        "terminal_versioned_current_recheck": _terminal_versioned_current_recheck(launch_child),
        "lease_receipt": lease_evidence, "lease_receipt_seal_sha256": lease_seal,
        "preflight_proof": context.proof_evidence, "preflight_proof_seal_sha256": context.proof_seal_sha256,
        "child_preflight_proof_binding": {"path": context.proof_evidence["path"], "document_sha256": context.proof_evidence["sha256"], "seal_sha256": context.proof_seal_sha256},
        "outer_preflight": context.outer_preflight_evidence, "outer_preflight_seal_sha256": context.outer_preflight_seal_sha256,
        "l0_state_handoff_child_preflight": context.child.evidence, "l0_state_handoff_child_preflight_seal_sha256": context.child.seal_sha256,
        "baseline_l0_to_l1_handoff_authority": context.child.handoff.authority_evidence, "baseline_l0_to_l1_handoff_authority_seal_sha256": context.child.handoff.authority_seal_sha256,
        "probe_binary": context.probe_binary,
        "planned_outer_capture_dir": str(capture), "planned_inner_capture_dir": str(capture / INNER_CAPTURE_DIRNAME), "planned_terminal_receipt_path": str(capture / TERMINAL_RECEIPT_FILENAME), "workers": workers,
        "execution_policy": {"quiet_qwen80_device_lease": True, "strict_math": True, "outer_reaped_capture_required": True, "timing_or_benchmarking_allowed": False, "l1_prefix_execution_allowed": False, "complete_layer_or_token_allowed": False, "tps_or_tg_claim_allowed": False, "automatic_retry_allowed": False},
        "lifecycle": {"one_shot": True, "receipt_last": True, "replay_guarded": True, "lease_release_required_on_every_terminal_outcome": True},
        "watcher_coordination": {"watcher_hold_must_remain_active": True, "watcher_restart_or_transition_authorized": False},
        "outer_reaper": {"outer_starts_exactly_one_child": True, "outer_reaps_child_before_terminal_receipt": True, "terminal_receipt_written_last": True, "automatic_retry_prohibited": True, "lease_reuse_prohibited_after_first_terminal_child": True},
        "required_pre_l1_handoff_capture": {"schema": PRE_L1_CAPTURE_SCHEMA, "status": PRE_L1_CAPTURE_STATUS, "retained_l0_8192b_buffer_id_required": True, "l0_active_rollback_conv_recurrent_hashes_required": True, "l1_slot1_input_matches_retained_l0_buffer_and_sha": True, "l1_slot1_non_alias_required": True, "l1_binding_not_executed": True, "l1_prefix_dispatches": 0, "may_not_satisfy_next_layer_execution_dependency": True, "l1_deltanet_dispatch_allowed": False},
        "next_layer_execution_dependency": {"schema": HANDOFF_WITNESS_SCHEMA, "required_status": HANDOFF_WITNESS_STATUS, "remains_unmet_after_this_capture": True, "may_not_be_created_or_accepted_by_this_outer": True},
        "claim_boundary": {"planning_only": True, "device_child_spawned": False, "metal_device_or_dispatch_performed": False, "lease_issued_by_this_outer": False, "l1_prefix_executed": False, "complete_layer_or_token_performed": False, "terminal_receipt_written": False},
    })
    _write_new(capture / ACTIVE_FILENAME, seal({"schema": "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_active_plan.v1", "status": "RESERVED_FUTURE_ONE_SHOT_NOT_EXECUTED", "lease_id": lease_id, "launch_identity_sha256": identity, "workers": workers, "terminal_receipt_exists": False, "device_child_spawned": False, "l1_prefix_executed": False, "complete_layer_or_token_performed": False}))
    _write_new(capture / OUTER_LAUNCH_AUTHORITY_FILENAME, authority)
    return authority


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--prepare-one-shot", action="store_true")
    parser.add_argument("--handoff-authority", type=Path, default=HANDOFF_AUTHORITY_PATH)
    parser.add_argument("--child-preflight", type=Path, default=CHILD_PREFLIGHT_PATH)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--preflight-capture-dir", type=Path)
    parser.add_argument("--preflight-proof", type=Path)
    parser.add_argument("--lease-receipt", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--replay-guard-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        for argument in arguments:
            if argument.split("=", 1)[0] in FORBIDDEN_ARGUMENTS:
                raise SourceTokenL0StateHandoffLauncherError(f"{argument.split('=', 1)[0]} is forbidden: no device execution exists here")
        args = build_parser().parse_args(arguments)
        if args.preflight_only:
            if args.preflight_capture_dir is None or any(value is not None for value in (args.preflight_proof, args.lease_receipt, args.capture_dir, args.replay_guard_dir)):
                raise SourceTokenL0StateHandoffLauncherError("--preflight-only requires only --preflight-capture-dir plus immutable inputs")
            print(json.dumps(run_preflight_only(child_preflight=args.child_preflight, handoff_authority=args.handoff_authority, probe_bin=args.probe_bin, capture_dir=args.preflight_capture_dir), sort_keys=True))
            return 0
        if args.preflight_capture_dir is not None or any(value is None for value in (args.preflight_proof, args.lease_receipt, args.capture_dir, args.replay_guard_dir)):
            raise SourceTokenL0StateHandoffLauncherError("--prepare-one-shot requires proof, lease, capture, replay guard and no preflight capture")
        print(json.dumps(prepare_future_one_shot(preflight_proof=args.preflight_proof, child_preflight=args.child_preflight, handoff_authority=args.handoff_authority, probe_bin=args.probe_bin, lease_receipt=args.lease_receipt, capture_dir=args.capture_dir, replay_guard_dir=args.replay_guard_dir), sort_keys=True))
        return 0
    except (SourceTokenL0StateHandoffLauncherError, OSError, ValueError) as exc:
        print(f"ascension_qwen80_source_token_l0_state_handoff_launcher: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
