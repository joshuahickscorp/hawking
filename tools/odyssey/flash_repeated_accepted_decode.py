#!/usr/bin/env python3
"""Run Flash's smallest honest repeated-accepted-decode gate.

The native executor is the candidate under test, so it cannot also create the
token sequence that certifies it.  A non-dry run therefore requires a sealed,
independent full-source greedy-reference contract.  The native body then runs
one fresh 48-layer stateful session over that externally supplied chain and
checks every token at the terminal boundary.

The runner refuses a visibly occupied Hawking native/GPU lane, preserving the
protected measurement requirement.  It does not turn a native-only historical
session or a self-derived next token into a source-reference claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # This script is also invoked directly, where Python otherwise puts only
    # tools/odyssey on sys.path.  The trust-anchor loader remains the existing
    # repository owner, not a local fallback implemented by this runner.
    sys.path.insert(0, str(ROOT))

from tools.verify.restream_guard import (  # noqa: E402
    load_pinned_owner_public_key,
    pinned_owner_public_key_path,
)

DEFAULT_MODEL_ROOT = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
MODEL_LAKE_ROOT = DEFAULT_MODEL_ROOT.parents[1]
MODEL_LAKE_SLUG = DEFAULT_MODEL_ROOT.name
PROMPT_IDS = (5423, 799, 4581, 3817, 13)
REPO_ID = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
PLE_PRE_LAYER_STATE_BANK_SCHEMA = "hawking.flash.ple_pre_layer_state_bank.v1"
PLE_PRE_LAYER_STATE_BANK_STATUS = "EXPORTED_SOURCE_BOUND_NATIVE_PRE_PLE_STATE_BANK__FORMULA_ONLY"
PLE_PRE_LAYER_STATE_WIDTH = 10_240
EXTERNAL_REFERENCE_SCHEMA = "hawking.flash.external_greedy_reference.v2"
EXTERNAL_REFERENCE_STATUS = "ADMITTED_EXTERNAL_COMPLETE_SOURCE_GREEDY_REFERENCE"
SOURCE_SHARD_LEDGER_SCHEMA = "hawking.flash.source_shard_ledger.v1"
SOURCE_SHARD_LEDGER_STATUS = "VERIFIED_COMPLETE_SOURCE_SHARD_LEDGER"
LOGITS_TRACE_SCHEMA = "hawking.flash.external_greedy_logits_trace.v1"
LOGITS_TRACE_STATUS = "SEALED_EXTERNAL_COMPLETE_SOURCE_GREEDY_LOGITS_TRACE"
SOURCE_INPUT_IDENTITY_SCHEMA = "hawking.flash.source_input_identity.v1"
OWNER_AUTHORIZATION_SCHEMA = "hawking.flash.external_greedy_reference_owner_authorization.v1"
OWNER_AUTHORIZATION_SCOPE = "admit_external_greedy_reference_v2"
LOCAL_CANONICAL_ADMISSION_SCHEMA = (
    "hawking.flash.external_greedy_reference_local_canonical_admission.v1"
)
LOCAL_CANONICAL_ADMISSION_STATUS = "ADMITTED_LOCAL_CANONICAL_CAPTURE"
CANONICAL_CAPTURE_PRODUCER_BASENAME = "flash_external_greedy_reference_mlx_worker.py"
CANONICAL_CAPTURE_PACKAGER_BASENAME = "flash_external_greedy_reference_mlx.py"
CANONICAL_CAPTURE_EXECUTION_SCHEDULE = (
    "exact_qwen4_exp_row_addressed_ple_and_experts_with_evaluated_sublayer_boundaries"
)
# The external-reference status describes a sealed complete provider capture,
# not scientific admission.  The admission class below remains the authority.
EXTERNAL_REFERENCE_STATUS_VOCABULARY_NOTE = (
    "EXTERNAL_REFERENCE_STATUS names a sealed-complete source greedy reference; "
    "authority is admission_class, and science_status remains UNEARNED until ordered compare."
)
# Historical Hawking capture roots are narrowly allowlisted for byte-bound
# capture producers.  Foreign roots still require a detached owner signature.
_EXTRA_CANONICAL_CAPTURE_REPO_ROOTS = (
    Path("/Users/scammermike/Downloads/hawking-grok-staging/worker-integration-snapshot"),
)
# The sealed 2026-09-12 local teacher predates the Hawking package move.  Its
# source artifacts remain in the reachable migration snapshot below, but must
# never be executed or silently substituted for current code.  These exact
# blob bindings let the verifier *read* that historical source identity after
# the current files changed imports during consolidation.  Any other receipt,
# path, digest, or unavailable blob fails closed.
_HISTORICAL_LOCAL_CAPTURE_RECEIPT = ROOT / (
    "receipts/headless/FLASH_EXTERNAL_GREEDY_REFERENCE_MLX_ROW_ADDRESSED_"
    "20260912/external-reference.v2.json"
)
_HISTORICAL_LOCAL_CAPTURE_RECEIPT_SHA256 = (
    "79a452fa40b2ded92af90d7aa4eec9cd14101c8ddf7198816b17bfb5182eb47f"
)
_HISTORICAL_LOCAL_CAPTURE_ARTIFACTS = {
    "producer": {
        "path": ROOT / "tools/odyssey/flash_external_greedy_reference_mlx_worker.py",
        "sha256": "3007e4b6d4a65493bf8eb795681d72bbba4195a00cc9b01f8f9a7a7e02d3b418",
        "git_snapshot": "a9c8b603fe2f06311324582ca722c55fd32aefb8",
        "git_path": "tools/odyssey/flash_external_greedy_reference_mlx_worker.py",
    },
    "packager": {
        "path": ROOT / "tools/odyssey/flash_external_greedy_reference_mlx.py",
        "sha256": "034dd2073e8b3aea3d08e5efb1b7a13d352b48eace0dac576580491c54aa26e6",
        "git_snapshot": "a9c8b603fe2f06311324582ca722c55fd32aefb8",
        "git_path": "tools/odyssey/flash_external_greedy_reference_mlx.py",
    },
}
NATIVE_SESSION_SCHEMA = "hawking.flash.stateful_complete_token_session.v1"
NATIVE_SESSION_STATUS = "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
NATIVE_CANDIDATE_ROUTE_OBSERVATION_STATUS = (
    "CANDIDATE_ROUTE_EXACT_TERMINAL_ACCEPTED_STATE_HASH_WITHHELD"
)
RUNNER_SIDECAR_SCHEMA = "hawking.flash.repeated_accepted_decode_runner.v3"
RUNNER_SIDECAR_STATUS = "PASSED_STATEFUL_REPEATED_EXTERNAL_REFERENCE_ACCEPTED_DECODE"
MAX_TRACE_VOCAB_SIZE = 1_000_000


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _json_object(raw: str | bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _json(path: Path) -> dict[str, Any]:
    return _json_object(path.read_bytes(), str(path))


def _flash_binary_command(
    native_binary: Path,
    model_root: Path,
    token_ids: list[int],
    prompt_len: int,
    out: Path,
    *,
    ple_pre_layer_state_bank_out: Path | None = None,
    route_teacher: Path | None = None,
    token_major_resident_banks: bool = False,
    token_major_clean_replay_reps: int = 0,
    candidate_route_observation: bool = False,
) -> list[str]:
    command = [
        str(native_binary),
        "--root", str(model_root),
        "--token-ids", ",".join(str(token) for token in token_ids),
        "--prompt-length", str(prompt_len),
        "--out", str(out),
        # Rust independently requires this wrapper-mode marker before Metal;
        # it is not a secret and does not replace V2 owner authorization.
        "--wrapper-admitted-source-control",
        "--supervising-launcher-pid", str(os.getpid()),
    ]
    if ple_pre_layer_state_bank_out is not None:
        command.extend(("--ple-pre-layer-state-bank-out", str(ple_pre_layer_state_bank_out)))
    if route_teacher is not None:
        command.extend(("--route-teacher", str(route_teacher)))
        command.extend(("--retain-linear-banks", "--device-only-compact-banks"))
    if token_major_resident_banks:
        command.append("--token-major-resident-banks")
    if token_major_clean_replay_reps > 0:
        command.extend(("--token-major-clean-replay-reps", str(token_major_clean_replay_reps)))
    if candidate_route_observation:
        command.append("--token-major-allow-state-drift-candidate")
    return command


def _native_binary_path() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    return target / "release/examples/flash_stateful_complete_token_session"


def _source_file_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"declared source-closure file is absent: {resolved}")
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256_file(resolved)}


def _native_source_closure_paths() -> tuple[Path, ...]:
    """Return the source leaves whose drift invalidates the bounded native seam.

    This is intentionally a small, explicit execution closure rather than a
    claim to reproduce Cargo's complete dependency graph.  It covers the
    example plus the source readers and PLE control semantics it calls.  In
    particular, an auxiliary control dtype change must invalidate a queued
    source-boundary run before raw bytes can be interpreted differently.
    """
    return (
        Path(__file__),
        ROOT / "crates/hawking-core/src/lib.rs",
        ROOT / "crates/hawking-core/examples/flash_stateful_complete_token_session.rs",
        ROOT / "crates/hawking-core/examples/flash_full_attention_layer3.rs",
        ROOT / "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
        ROOT / "crates/hawking-core/examples/flash_source_bf16_terminal.rs",
        ROOT / "crates/hawking-core/src/flash_ple.rs",
        ROOT / "crates/hawking-core/src/model/source_safetensors.rs",
        ROOT / "crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs",
        ROOT / "Cargo.lock",
    )


def _command_text(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _preexecution_build_binding(native_binary: Path) -> dict[str, Any]:
    """Fingerprint an observed prebuilt executable/source closure, not its build."""
    if native_binary.is_symlink() or not native_binary.is_file():
        raise FileNotFoundError(
            "prebuilt native executable is absent or is a symlink; build and inspect it separately before "
            f"a repeated-decode run: {native_binary}"
        )
    git_status = _command_text(["git", "status", "--porcelain=v1"])
    return {
        "captured_before_native_execution": True,
        "attestation": "observed_executable_and_source_closure_only",
        "compiler_provenance_attested": False,
        "promotion_allowed": False,
        "git_head": _command_text(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
        "worktree_dirty": bool(git_status),
        "rustc_version": _command_text(["rustc", "-Vv"]),
        "native_executable": _source_file_binding(native_binary),
        "source_closure": [
            _source_file_binding(path) for path in _native_source_closure_paths()
        ],
    }


def _same_executable_closure(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Reject a run if its executable or declared source body drifted mid-session."""
    return all(
        before.get(key) == after.get(key)
        for key in ("git_head", "rustc_version", "native_executable", "source_closure")
    )


def _native_session_artifact_dir(out: Path) -> Path:
    """Match the Rust executor's owned session-artifact namespace exactly."""
    return out.with_name(f"{out.stem}.session_artifacts")


def _path_entry_exists(path: Path) -> bool:
    """Detect every occupied path, including a dangling symlink."""
    return os.path.lexists(path)


def _reserve_fresh_run_outputs(out: Path, state_bank_out: Path) -> Path:
    """Refuse all pre-existing or structurally colliding native output targets."""
    runner_sidecar = out.with_name(f"{out.stem}.runner.json")
    session_artifacts = _native_session_artifact_dir(out)
    targets = (out, runner_sidecar, session_artifacts, state_bank_out)
    for target in targets:
        parent = target.parent
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"native output parent is not a directory: {parent}")
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "native receipt, runner sidecar, session-artifact, and state-bank targets must not collide or nest"
                )
    existing = [str(target) for target in targets if _path_entry_exists(target)]
    if existing:
        raise ValueError(
            "native receipt, runner sidecar, session-artifact, or state-bank output already exists; choose fresh paths: "
            + ", ".join(existing)
        )
    return runner_sidecar


def _protected_native_lane() -> dict[str, Any]:
    """Observe the protected Flash/native lane before starting a native body.

    This is deliberately only an observed process snapshot.  It refuses a
    visibly occupied lane, but is not a timing witness for a benchmark or a
    proof that no process could start after this preflight.
    """

    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,stat=,command="], check=True, capture_output=True, text=True,
    )
    markers = (
        "flash_stateful_complete_token_session",
        "flash_repeated_accepted_decode.py",
        "flash_route_union_control.py",
        "flash_source_boundary_capture_mlx.py",
        "flash_source_boundary_mlx_worker.py",
        "flash_noetic_complete_layer0",
        "mlx_vlm.server",
        "hawkingd",
    )
    rows: list[tuple[str, str, str, str]] = []
    for row in completed.stdout.splitlines():
        stripped = row.strip()
        parts = stripped.split(None, 3)
        # The live observation includes PPID so a nested protected transaction
        # can exclude its own launcher ancestry.  Retain the old fixture forms
        # because this predicate is also independently regression-tested.
        if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
            raw_pid, raw_ppid, state, command = parts
        elif len(parts) >= 3 and parts[0].isdigit():
            raw_pid, state = parts[:2]
            command = " ".join(parts[2:])
            raw_ppid = ""
        elif len(parts) >= 2:
            raw_pid = ""
            raw_ppid = ""
            state = parts[0]
            command = stripped[len(state):].strip()
        else:
            continue
        rows.append((raw_pid, raw_ppid, state, command))

    # A leased source-boundary transaction is launched through the lane shell:
    # its public parent holds the complete ``--native-binary`` argument, which
    # intentionally contains a protected executable basename.  Exclude only
    # our exact ancestry, never sibling or unrelated protected processes.
    parent_by_pid = {pid: ppid for pid, ppid, _, _ in rows if pid and ppid}
    own_ancestry: set[str] = set()
    cursor = str(os.getpid())
    while cursor and cursor not in own_ancestry:
        own_ancestry.add(cursor)
        parent = parent_by_pid.get(cursor)
        if not parent or parent == "0":
            break
        cursor = parent
    # Old fixture formats do not include PPID, so retain the direct-parent
    # exclusion when ancestry cannot be reconstructed from the snapshot.
    own_ancestry.add(str(os.getppid()))

    matches: list[dict[str, str]] = []
    for raw_pid, _raw_ppid, state, command in rows:
        # The direct launcher (for example ``zsh -c python3 <this-script>``)
        # necessarily contains the protected script name in its command line.
        # It is part of this invocation, not competing lane ownership.
        if raw_pid and raw_pid in own_ancestry:
            continue
        # A terminated child can remain visible briefly until its live
        # hawkingd parent reaps it.  It owns no GPU/UMA working set and must
        # not block a protected lane; every non-zombie matching row still does.
        if state.startswith("Z") or not command:
            continue
        marker = next((candidate for candidate in markers if candidate in command), None)
        if marker is not None:
            match = {"state": state, "command": command, "marker": marker}
            if raw_pid:
                match["pid"] = raw_pid
            matches.append(match)
    return {
        "clean": not matches,
        "matches": matches,
        "observation": "point-in-time ps preflight only; not a timing or exclusivity witness",
    }


def _hawking_gpu_is_busy() -> bool:
    """Compatibility predicate for callers that only need the lane verdict."""
    return not _protected_native_lane()["clean"]


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    parsed = _require_nonnegative_int(value, label)
    if parsed == 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _require_token_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not all(_is_token_id(token) for token in value):
        raise ValueError(f"{label} must be a list of non-negative integer token IDs")
    return list(value)


def _verify_compact_seal(document: dict[str, Any], label: str) -> str:
    seal = _require_sha256(document.get("seal_sha256"), f"{label} seal_sha256")
    body = dict(document)
    body.pop("seal_sha256", None)
    try:
        expected = _compact_utf8_sorted_seal(body)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} cannot be canonical compact UTF-8 JSON") from exc
    if seal != expected:
        raise ValueError(f"{label} has an invalid compact UTF-8 seal")
    return seal


def _verify_native_compact_receipt_seal(doc: dict[str, Any]) -> str:
    """Verify the exact canonical format emitted by the Rust session body."""
    return _verify_compact_seal(doc, "source-bound native session")


def _resolve_regular_file(value: object, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError(f"{label} path must name a local regular file")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else base / raw
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing or inaccessible") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _verify_artifact_binding(
    binding: object,
    *,
    base: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} must be an artifact binding object")
    artifact = _resolve_regular_file(binding.get("path"), base=base, label=label)
    expected_sha256 = _require_sha256(binding.get("sha256"), f"{label} sha256")
    actual_sha256 = _sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 does not match its local artifact")
    return {"path": str(artifact), "sha256": actual_sha256}


def _verify_historical_local_capture_artifact(
    binding: object,
    *,
    receipt: Path,
    field: str,
) -> dict[str, str] | None:
    """Verify one exact historical source blob without executing it.

    The migration left the old sealed teacher and its raw logits intact while
    changing the current package imports.  This escape hatch is intentionally
    narrower than a generic Git fallback: it admits only the named receipt,
    its fixed SHA-256, its two fixed artifact paths, and the corresponding
    reachable snapshot blobs.
    """
    record = _HISTORICAL_LOCAL_CAPTURE_ARTIFACTS.get(field)
    if not isinstance(binding, dict) or not isinstance(record, dict):
        return None
    try:
        if (
            receipt.resolve() != _HISTORICAL_LOCAL_CAPTURE_RECEIPT.resolve()
            or _sha256_file(receipt) != _HISTORICAL_LOCAL_CAPTURE_RECEIPT_SHA256
            or binding.get("path") != str(Path(record["path"]).resolve())
            or binding.get("sha256") != record["sha256"]
        ):
            return None
    except (OSError, TypeError):
        return None

    try:
        completed = subprocess.run(
            ["git", "show", f"{record['git_snapshot']}:{record['git_path']}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ValueError("historical local-capture Git snapshot is unavailable") from exc
    if completed.returncode != 0:
        raise ValueError("historical local-capture Git snapshot is unavailable")
    actual_sha256 = hashlib.sha256(completed.stdout).hexdigest()
    if actual_sha256 != record["sha256"]:
        raise ValueError("historical local-capture Git blob differs from the sealed artifact")
    return {
        "path": str(Path(record["path"]).resolve()),
        "sha256": actual_sha256,
        "historical_git_snapshot": str(record["git_snapshot"]),
        "historical_git_path": str(record["git_path"]),
    }


def _source_file(model_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name a relative source-root file")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} must remain below the selected source root")
    candidate = model_root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(model_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} escapes or is absent from the selected source root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular source file")
    return resolved


def _canonical_model_lake_manifest(model_root: Path) -> Path:
    lake_root = MODEL_LAKE_ROOT.expanduser().resolve()
    expected_root = (lake_root / "specimens" / MODEL_LAKE_SLUG).resolve()
    if model_root != expected_root:
        raise ValueError("selected source root is not the configured canonical ModelLake Flash specimen")
    return _resolve_regular_file(
        str(lake_root / "manifests" / f"{MODEL_LAKE_SLUG}.json"),
        base=lake_root,
        label="canonical ModelLake manifest",
    )


def _verify_model_lake_manifest(
    binding: object,
    *,
    contract_dir: Path,
    model_root: Path,
) -> dict[str, Any]:
    """Bind the source root to its one canonical local ModelLake manifest."""
    canonical_path = _canonical_model_lake_manifest(model_root)
    verified = _verify_artifact_binding(
        binding,
        base=contract_dir,
        label="canonical ModelLake manifest",
    )
    if Path(verified["path"]) != canonical_path:
        raise ValueError("external reference contract does not name the canonical local ModelLake manifest")
    if not isinstance(binding, dict):  # Covered above; narrows the type for field checks.
        raise AssertionError("artifact binding type narrowed")
    if binding.get("repo") != REPO_ID or binding.get("revision") != PINNED_REVISION:
        raise ValueError("ModelLake manifest binding omitted the pinned Flash identity")
    manifest = _json(canonical_path)
    declared_root = manifest.get("path")
    if not isinstance(declared_root, str) or not declared_root:
        raise ValueError("canonical ModelLake manifest omitted its specimen path")
    if (
        manifest.get("repo") != REPO_ID
        or manifest.get("revision") != PINNED_REVISION
        or manifest.get("resolved_sha") != PINNED_REVISION
        or Path(declared_root).expanduser().resolve() != model_root
    ):
        raise ValueError("canonical ModelLake manifest does not bind the selected pinned Flash specimen")
    return {
        "path": str(canonical_path),
        "sha256": verified["sha256"],
        "repo": REPO_ID,
        "revision": PINNED_REVISION,
    }


def _load_safetensors_index(index_path: Path) -> dict[str, Any]:
    index = _json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source safetensors index omitted a non-empty weight map")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ValueError("source safetensors index contains a malformed weight map")
    return index


def _canonical_source_vocab_size(config: dict[str, Any]) -> int:
    """Read Flash-Next's actual nested vocabulary field, not a guessed alias."""
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("source config omitted the canonical text_config object")
    vocabulary = _require_positive_int(text_config.get("vocab_size"), "source text_config vocab_size")
    top_level = config.get("vocab_size")
    if top_level is not None and _require_positive_int(top_level, "source top-level vocab_size") != vocabulary:
        raise ValueError("source config has a contradictory top-level vocabulary size")
    return vocabulary


def _expected_native_source_input_identity(
    *,
    model_root: Path,
    manifest: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    index_path: Path,
    index_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    """Build the exact Rust ``source_input_identity`` the session must emit."""
    return {
        "schema": SOURCE_INPUT_IDENTITY_SCHEMA,
        "model": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "model_root": str(model_root),
        "model_lake_manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "repo": REPO_ID,
            "revision": PINNED_REVISION,
        },
        "config": {
            "file": "config.json",
            "sha256": config_sha256,
            "bytes": config_path.stat().st_size,
        },
        "safetensors_index": {
            "file": "model.safetensors.index.json",
            "sha256": index_sha256,
            "bytes": index_path.stat().st_size,
        },
        "tokenizer": {
            "file": "tokenizer.json",
            "sha256": tokenizer_sha256,
            "bytes": tokenizer_path.stat().st_size,
        },
    }


def _verify_shard_ledger(
    binding: object,
    *,
    contract_dir: Path,
    model_root: Path,
    manifest_sha256: str,
    index_path: Path,
    index_sha256: str,
) -> dict[str, Any]:
    """Accept only a sealed exact-hash ledger covering every indexed shard.

    This deliberately consumes an existing verifier artifact.  Rehashing the
    360 GB specimen while parsing a tiny reference contract would be a new,
    unbounded source experiment; the admitted ledger records that earlier
    exact-byte verification instead, while this reader still rejects a missing
    shard, size drift, broken seal, or mismatched binding.
    """
    verified = _verify_artifact_binding(
        binding,
        base=contract_dir,
        label="verified source shard ledger",
    )
    if not isinstance(binding, dict):  # Covered above; narrows the type for field checks.
        raise AssertionError("artifact binding type narrowed")
    ledger_path = Path(verified["path"])
    ledger = _json(ledger_path)
    ledger_seal = _verify_compact_seal(ledger, "verified source shard ledger")
    if (
        binding.get("schema") != SOURCE_SHARD_LEDGER_SCHEMA
        or binding.get("status") != SOURCE_SHARD_LEDGER_STATUS
        or _require_sha256(binding.get("seal_sha256"), "verified source shard ledger binding seal_sha256")
        != ledger_seal
        or ledger.get("schema") != SOURCE_SHARD_LEDGER_SCHEMA
        or ledger.get("status") != SOURCE_SHARD_LEDGER_STATUS
    ):
        raise ValueError("verified source shard ledger has an incompatible schema, status, or seal binding")
    source = ledger.get("source")
    if not isinstance(source, dict) or (
        source.get("model") != REPO_ID
        or source.get("pinned_revision") != PINNED_REVISION
        or source.get("model_root") != str(model_root)
        or source.get("model_lake_manifest_sha256") != manifest_sha256
        or source.get("safetensors_index_sha256") != index_sha256
    ):
        raise ValueError("verified source shard ledger does not bind the selected ModelLake source")
    verification = ledger.get("verification")
    if not isinstance(verification, dict) or (
        verification.get("method") != "sha256_exact_file_bytes"
        or verification.get("all_indexed_shards_verified") is not True
    ):
        raise ValueError("verified source shard ledger lacks an exact-byte verification declaration")

    index = _load_safetensors_index(index_path)
    indexed_shards = set(index["weight_map"].values())
    records = ledger.get("shards")
    if not isinstance(records, list) or len(records) != len(indexed_shards):
        raise ValueError("verified source shard ledger does not cover every indexed shard exactly once")
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("verified source shard ledger contains a malformed record")
        shard_name = record.get("path")
        if not isinstance(shard_name, str) or shard_name in by_name:
            raise ValueError("verified source shard ledger has a duplicate or invalid shard path")
        by_name[shard_name] = record
    if set(by_name) != indexed_shards:
        raise ValueError("verified source shard ledger shard inventory differs from the source index")
    for shard_name in sorted(indexed_shards):
        source_shard = _source_file(model_root, shard_name, f"indexed source shard {shard_name}")
        record = by_name[shard_name]
        if (
            _require_nonnegative_int(record.get("bytes"), f"ledger bytes for {shard_name}")
            != source_shard.stat().st_size
            or record.get("verification") != "SHA256_EXACT_FILE_BYTES"
        ):
            raise ValueError(f"verified source shard ledger no longer matches {shard_name}")
        _require_sha256(record.get("sha256"), f"ledger SHA-256 for {shard_name}")
    return {
        "path": str(ledger_path),
        "sha256": verified["sha256"],
        "seal_sha256": ledger_seal,
        "indexed_shard_count": len(indexed_shards),
        "verification": "sha256_exact_file_bytes (attested by sealed ledger)",
    }


def _verify_provider_artifacts(
    provider: object,
    *,
    contract_dir: Path,
    receipt: Path,
) -> dict[str, Any]:
    if not isinstance(provider, dict):
        raise ValueError("external reference contract omitted provider provenance")
    if provider.get("independent_from_hawking_native_executor") is not True:
        raise ValueError("external reference contract does not declare an independent provider")
    version = provider.get("implementation_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("external reference contract provider omitted an implementation version")
    artifacts: dict[str, dict[str, str]] = {}
    for field in ("implementation", "runtime_lock", "producer"):
        try:
            artifacts[field] = _verify_artifact_binding(
                provider.get(field),
                base=contract_dir,
                label=f"provider {field}",
            )
        except ValueError:
            # Only the sealed migration-era teacher may resolve its historical
            # producer source through the exact Git blob; every other provider
            # still needs a live, byte-identical local artifact.
            historical = (
                _verify_historical_local_capture_artifact(
                    provider.get(field), receipt=receipt, field=field,
                )
                if field == "producer"
                else None
            )
            if historical is None:
                raise
            artifacts[field] = historical
    paths = {Path(artifact["path"]) for artifact in artifacts.values()}
    if len(paths) != 3 or receipt in paths:
        raise ValueError("provider implementation, runtime-lock, and producer must be distinct existing artifacts")
    if Path(artifacts["implementation"]["path"]) == Path(__file__).resolve():
        raise ValueError("Hawking's native repeated-decode runner cannot certify itself as a source provider")
    return {
        "implementation_version": version,
        "independent_from_hawking_native_executor": True,
        **artifacts,
    }


def _f32_argmax(raw: bytes, *, elements: int, label: str) -> int:
    if len(raw) != elements * 4:
        raise ValueError(f"{label} does not contain the declared F32 payload")
    best_index = -1
    best_value = -math.inf
    for index in range(elements):
        value = struct.unpack_from("<f", raw, index * 4)[0]
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite F32 logit")
        # This is the conventional deterministic greedy tie rule: the first
        # maximum (therefore lowest token ID) wins.
        if best_index < 0 or value > best_value:
            best_index = index
            best_value = value
    return best_index


def _verify_logits_trace(
    binding: object,
    *,
    contract_dir: Path,
    prompt: list[int],
    references: list[int],
    vocab_size: int,
    source_hashes: dict[str, str],
    provider: dict[str, Any],
) -> dict[str, Any]:
    """Verify raw per-step F32 logits rather than trusting declared token IDs."""
    verified = _verify_artifact_binding(
        binding,
        base=contract_dir,
        label="sealed per-step logits trace",
    )
    if not isinstance(binding, dict):  # Covered above; narrows the type for field checks.
        raise AssertionError("artifact binding type narrowed")
    trace_path = Path(verified["path"])
    trace = _json(trace_path)
    trace_seal = _verify_compact_seal(trace, "sealed per-step logits trace")
    if (
        binding.get("schema") != LOGITS_TRACE_SCHEMA
        or binding.get("status") != LOGITS_TRACE_STATUS
        or _require_sha256(binding.get("seal_sha256"), "logits trace binding seal_sha256") != trace_seal
        or trace.get("schema") != LOGITS_TRACE_SCHEMA
        or trace.get("status") != LOGITS_TRACE_STATUS
    ):
        raise ValueError("sealed per-step logits trace has an incompatible schema, status, or seal binding")
    trace_source = trace.get("source")
    if not isinstance(trace_source, dict) or any(
        trace_source.get(key) != value
        for key, value in {
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            **source_hashes,
        }.items()
    ):
        raise ValueError("sealed per-step logits trace does not bind the admitted Flash source")
    trace_provider = trace.get("provider_artifacts")
    if not isinstance(trace_provider, dict) or any(
        trace_provider.get(f"{field}_sha256") != provider[field]["sha256"]
        for field in ("implementation", "runtime_lock", "producer")
    ):
        raise ValueError("sealed per-step logits trace does not bind the admitted provider artifacts")
    trace_prompt = _require_token_list(trace.get("prompt_token_ids"), "logits trace prompt_token_ids")
    trace_references = _require_token_list(trace.get("generated_token_ids"), "logits trace generated_token_ids")
    if trace_prompt != prompt or trace_references != references:
        raise ValueError("sealed per-step logits trace token sequence differs from the reference contract")
    sampling = trace.get("sampling")
    if (
        not isinstance(sampling, dict)
        or sampling.get("method") != "greedy_argmax"
        or sampling.get("do_sample") is not False
        or isinstance(sampling.get("temperature"), bool)
        or sampling.get("temperature") != 0
    ):
        raise ValueError("sealed per-step logits trace is not a deterministic greedy-argmax run")
    if (
        trace.get("dtype") != "F32_LE"
        or trace.get("argmax_tie_break") != "lowest_token_id"
        or _require_positive_int(trace.get("vocab_size"), "logits trace vocab_size") != vocab_size
    ):
        raise ValueError("sealed per-step logits trace has an incompatible F32 vocabulary contract")
    if vocab_size > MAX_TRACE_VOCAB_SIZE:
        raise ValueError("source vocabulary exceeds the bounded logits-trace verifier limit")
    if any(token >= vocab_size for token in [*prompt, *references]):
        raise ValueError("reference contract token ID exceeds the source vocabulary")

    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) != len(references):
        raise ValueError("sealed per-step logits trace does not cover every generated token")
    seen_payloads: set[Path] = set()
    computed_argmax: list[int] = []
    for generation_index, (expected_token, step) in enumerate(zip(references, steps, strict=True)):
        if not isinstance(step, dict):
            raise ValueError("sealed per-step logits trace contains a malformed step")
        expected_input = [*prompt, *references[:generation_index]]
        input_tokens = _require_token_list(
            step.get("input_token_ids"),
            f"logits trace input_token_ids at step {generation_index}",
        )
        if (
            _require_nonnegative_int(step.get("generation_index"), "logits trace generation_index")
            != generation_index
            or input_tokens != expected_input
            or step.get("expected_token_id") != expected_token
            or step.get("argmax_token_id") != expected_token
        ):
            raise ValueError("sealed per-step logits trace has a shifted token/input boundary")
        payload = _verify_artifact_binding(
            step.get("logits_f32"),
            base=trace_path.parent,
            label=f"F32 logits payload for generation step {generation_index}",
        )
        if not isinstance(step.get("logits_f32"), dict):  # Covered above; narrows the type.
            raise AssertionError("logits artifact binding type narrowed")
        payload_metadata = step["logits_f32"]
        payload_path = Path(payload["path"])
        if payload_path == trace_path or payload_path in seen_payloads:
            raise ValueError("sealed per-step logits trace reuses or embeds a logits payload")
        seen_payloads.add(payload_path)
        if (
            payload_metadata.get("dtype") != "F32_LE"
            or _require_nonnegative_int(
                payload_metadata.get("elements"),
                f"F32 logits elements at generation step {generation_index}",
            )
            != vocab_size
            or _require_nonnegative_int(
                payload_metadata.get("bytes"),
                f"F32 logits bytes at generation step {generation_index}",
            )
            != vocab_size * 4
        ):
            raise ValueError("sealed per-step logits trace has an invalid F32 payload declaration")
        observed_argmax = _f32_argmax(
            payload_path.read_bytes(),
            elements=vocab_size,
            label=f"F32 logits payload for generation step {generation_index}",
        )
        if observed_argmax != expected_token:
            raise ValueError("actual F32 logits argmax does not match the reference generated token")
        computed_argmax.append(observed_argmax)
    return {
        "path": str(trace_path),
        "sha256": verified["sha256"],
        "seal_sha256": trace_seal,
        "step_count": len(steps),
        "vocab_size": vocab_size,
        "computed_argmax_token_ids": computed_argmax,
    }


def load_external_reference_contract(
    path: Path,
    model_root: Path,
    *,
    owner_authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Load only an admitted V2 source reference, never a handwritten V1 claim.

    The V2 admission binds the local canonical ModelLake identity, a prior
    exact-hash shard ledger, the actual reference-provider files, and raw F32
    logits for every generated step.  Provenance-complete captures produced by
    the canonical Hawking MLX capture owner may admit as
    ``ADMITTED_LOCAL_CANONICAL_CAPTURE`` without a detached Ed25519 signature.
    Imported, foreign, or noncanonical captures still require the optional
    machine-admin Ed25519 authorization path.  Neither route proves native
    correctness, capability, EBPW, TPS, or Pulsar promotion.
    """
    receipt = path.expanduser().resolve()
    receipt_bytes = receipt.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    document = _json_object(receipt_bytes, str(receipt))
    if document.get("schema") != EXTERNAL_REFERENCE_SCHEMA:
        if document.get("schema") == "hawking.flash.external_greedy_reference.v1":
            raise ValueError("handwritten external greedy-reference V1 contracts are withheld; an admitted V2 contract is required")
        raise ValueError("external reference contract has an incompatible schema")
    if document.get("status") != EXTERNAL_REFERENCE_STATUS:
        raise ValueError("external reference contract is not admitted")
    seal = _verify_compact_seal(document, "external reference contract")
    model_root = model_root.expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"Flash source root does not exist: {model_root}")

    source = document.get("source")
    if not isinstance(source, dict):
        raise ValueError("external reference contract omitted source binding")
    config_path = _source_file(model_root, "config.json", "source config")
    index_path = _source_file(model_root, "model.safetensors.index.json", "source safetensors index")
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    if (
        source.get("model") != REPO_ID
        or source.get("pinned_revision") != PINNED_REVISION
        or source.get("config_sha256") != config_sha256
        or source.get("safetensors_index_sha256") != index_sha256
        or source.get("ple_inclusive") is not True
    ):
        raise ValueError("external reference contract does not bind the exact PLE-inclusive Flash source")
    config = _json(config_path)
    vocab_size = _canonical_source_vocab_size(config)
    manifest = _verify_model_lake_manifest(
        source.get("model_lake_manifest"),
        contract_dir=receipt.parent,
        model_root=model_root,
    )

    tokenizer = document.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("external reference contract omitted tokenizer binding")
    if tokenizer.get("file") != "tokenizer.json":
        raise ValueError("external reference contract must bind the native source tokenizer.json")
    tokenizer_path = _source_file(model_root, tokenizer.get("file"), "external reference tokenizer")
    tokenizer_sha256 = _require_sha256(tokenizer.get("sha256"), "external reference tokenizer sha256")
    if tokenizer_sha256 != _sha256_file(tokenizer_path):
        raise ValueError("external reference contract tokenizer hash does not match the source root")
    source_input_identity = _expected_native_source_input_identity(
        model_root=model_root,
        manifest=manifest,
        config_path=config_path,
        config_sha256=config_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
    )

    prompt = _require_token_list(document.get("prompt_token_ids"), "external reference prompt_token_ids")
    references = _require_token_list(document.get("generated_token_ids"), "external reference generated_token_ids")
    if prompt != list(PROMPT_IDS):
        raise ValueError("external reference contract prompt token sequence does not match the canonical Flash gate")
    if len(references) < 2:
        raise ValueError("external reference contract needs at least two generated token IDs")
    sampling = document.get("sampling")
    if (
        not isinstance(sampling, dict)
        or sampling.get("method") != "greedy_argmax"
        or sampling.get("do_sample") is not False
        or isinstance(sampling.get("temperature"), bool)
        or sampling.get("temperature") != 0
    ):
        raise ValueError("external reference contract is not a deterministic greedy-argmax run")

    shard_ledger = _verify_shard_ledger(
        source.get("shard_ledger"),
        contract_dir=receipt.parent,
        model_root=model_root,
        manifest_sha256=manifest["sha256"],
        index_path=index_path,
        index_sha256=index_sha256,
    )
    provider = _verify_provider_artifacts(
        document.get("provider"),
        contract_dir=receipt.parent,
        receipt=receipt,
    )
    logits_trace = _verify_logits_trace(
        document.get("logits_trace"),
        contract_dir=receipt.parent,
        prompt=prompt,
        references=references,
        vocab_size=vocab_size,
        source_hashes={
            "config_sha256": config_sha256,
            "safetensors_index_sha256": index_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "model_lake_manifest_sha256": manifest["sha256"],
            "shard_ledger_sha256": shard_ledger["sha256"],
        },
        provider=provider,
    )
    if _sha256_file(receipt) != receipt_sha256:
        raise ValueError("external reference contract changed during admission")
    admission_authority = _resolve_v2_admission_authority(
        owner_authorization_path,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        receipt_seal_sha256=seal,
        provider=provider,
        logits_trace=logits_trace,
        shard_ledger=shard_ledger,
        manifest=manifest,
        document=document,
    )
    return {
        "mode": "admitted_external_complete_source_greedy_reference_v2",
        "receipt": str(receipt),
        "sha256": receipt_sha256,
        "seal_sha256": seal,
        "prompt_token_ids": prompt,
        "generated_token_ids": references,
        "source": {
            "model_lake_manifest": manifest,
            "shard_ledger": shard_ledger,
            "config_sha256": config_sha256,
            "safetensors_index_sha256": index_sha256,
            "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_sha256},
            "source_input_identity": source_input_identity,
        },
        "source_input_identity": source_input_identity,
        "provider": provider,
        "logits_trace": logits_trace,
        "owner_authorization": admission_authority,
        "claim_boundary": admission_authority.get(
            "claim_boundary",
            (
                "This V2 contract is admitted only for the stated local source identity and raw F32 greedy "
                "token chain. Admission authenticates provenance bindings, not provider semantics or "
                "execution. It does not establish native state continuity, TPS, EBPW, capability, or promotion."
            ),
        ),
        "admission_class": admission_authority.get("admission_class"),
        "science_status": admission_authority.get("science_status", "UNEARNED"),
        "science_unearned": True,
        "contract_status_vocabulary_note": EXTERNAL_REFERENCE_STATUS_VOCABULARY_NOTE,
    }


def admission_authority_binding_sha256(authority: dict[str, Any]) -> str:
    """Bind boundary receipts to a signed or local-canonical admission decision."""
    owner = authority.get("owner_authorization")
    if not isinstance(owner, dict):
        raise ValueError("admission authority omitted owner_authorization binding")
    admission_class = authority.get("admission_class") or owner.get("admission_class")
    file_sha = owner.get("sha256")
    if (
        isinstance(file_sha, str)
        and len(file_sha) == 64
        and all(character in "0123456789abcdef" for character in file_sha)
    ):
        return file_sha
    subject = {
        "admission_class": admission_class,
        "science_status": authority.get("science_status") or owner.get("science_status") or "UNEARNED",
        "reference_contract_sha256": owner.get("reference_contract_sha256"),
        "reference_contract_seal_sha256": owner.get("reference_contract_seal_sha256"),
        "producer_sha256": owner.get("producer_sha256"),
        "provider_implementation_sha256": owner.get("provider_implementation_sha256"),
        "provider_runtime_lock_sha256": owner.get("provider_runtime_lock_sha256"),
        "logits_trace_sha256": owner.get("logits_trace_sha256"),
        "logits_trace_seal_sha256": owner.get("logits_trace_seal_sha256"),
        "shard_ledger_sha256": owner.get("shard_ledger_sha256"),
        "shard_ledger_seal_sha256": owner.get("shard_ledger_seal_sha256"),
        "model_lake_manifest_sha256": owner.get("model_lake_manifest_sha256"),
    }
    return _compact_utf8_sorted_seal(subject)


def load_admitted_reference_for_boundary(
    reference: Path,
    model_root: Path,
    *,
    owner_authorization_path: Path | None,
) -> dict[str, Any]:
    """Load a V2 reference and attach the digest used by bounded diagnostics."""
    authority = load_external_reference_contract(
        reference,
        model_root,
        owner_authorization_path=owner_authorization_path,
    )
    authority = dict(authority)
    authority["authorization_binding_sha256"] = admission_authority_binding_sha256(authority)
    authority["science_unearned"] = True
    if authority.get("science_status") != "UNEARNED":
        authority["science_status"] = "UNEARNED"
    authority["promotion_allowed"] = False
    return authority


# Kept as a compatibility alias for existing callers while other Flash controls
# use the public name above.  A route-union replay must consume exactly the
# same external-reference contract as the repeated-decode gate, rather than
# implementing a weaker parallel parser.
_load_external_reference_contract = load_external_reference_contract


def _verify_repeated(
    doc: dict[str, Any],
    references: list[int],
    *,
    model_root: Path,
    source_input_identity: dict[str, Any],
    allow_candidate_route_observation: bool = False,
) -> None:
    expected_references = _require_token_list(references, "expected reference_generated_token_ids")
    if len(expected_references) < 2:
        raise ValueError("repeated session requires at least two external reference tokens")
    if doc.get("schema") != NATIVE_SESSION_SCHEMA:
        raise ValueError("repeated session has an incompatible native session schema")
    _verify_native_compact_receipt_seal(doc)
    _verify_source_identity(doc)
    model_root = model_root.expanduser().resolve()
    if doc.get("root") != str(model_root):
        raise ValueError("repeated session root differs from the selected canonical ModelLake source")
    if doc.get("source_input_identity") != source_input_identity:
        raise ValueError(
            "repeated session source input identity differs from the selected admitted V2 ModelLake source"
        )
    allowed_statuses = {NATIVE_SESSION_STATUS}
    if allow_candidate_route_observation:
        allowed_statuses.add(NATIVE_CANDIDATE_ROUTE_OBSERVATION_STATUS)
    if doc.get("status") not in allowed_statuses:
        raise ValueError(f"repeated session status was {doc.get('status')!r}")
    token_ids = _require_token_list(doc.get("token_ids"), "native session token_ids")
    prompt_token_ids = _require_token_list(doc.get("prompt_token_ids"), "native session prompt_token_ids")
    reference_token_ids = _require_token_list(
        doc.get("reference_generated_token_ids"),
        "native session reference_generated_token_ids",
    )
    expected_token_ids = [*PROMPT_IDS, *expected_references]
    if (
        token_ids != expected_token_ids
        or prompt_token_ids != list(PROMPT_IDS)
        or reference_token_ids != expected_references
        or doc.get("candidate_token_id") != expected_references[0]
    ):
        raise ValueError("repeated session token IDs or prompt/reference boundary drifted")
    if doc.get("accepted_generation_tokens") != len(expected_references):
        raise ValueError("repeated session did not accept every reference token")
    execution = doc.get("execution")
    if not isinstance(execution, dict) or execution.get("source_reset_or_reprefill") is not False:
        raise ValueError("verification session did not prove reset/re-prefill boundary")
    if allow_candidate_route_observation:
        if execution.get("expert_bank_mode") != "route_union_compact_teacher_bound":
            raise ValueError("candidate route observation did not retain the compact route-union mode")
        if execution.get("source_independent") is not False:
            raise ValueError("candidate route observation did not remain source-bound")
        if not str(execution.get("per_layer_state_hash_contract", "")).startswith("withheld"):
            raise ValueError("candidate route observation did not disclose its withheld state-hash contract")
    memory = execution.get("state_memory")
    if not isinstance(memory, dict) or _require_positive_int(
        memory.get("total_persistent_bytes"),
        "native session total_persistent_bytes",
    ) <= 0:
        raise ValueError("verification session omitted persistent-state census")
    if _require_positive_int(
        memory.get("growth_bytes_per_additional_token"),
        "native session growth_bytes_per_additional_token",
    ) <= 0:
        raise ValueError("verification session omitted state-growth census")
    if _require_positive_int(
        execution.get("source_payload_bytes_read"),
        "native session source_payload_bytes_read",
    ) <= 0:
        raise ValueError("verification session omitted source-byte census")
    terminal = doc.get("terminal")
    if not isinstance(terminal, dict) or terminal.get("candidate_accepted") is not True:
        raise ValueError("verification session omitted terminal acceptance boundary")
    checks = terminal.get("reference_checks")
    if not isinstance(checks, list) or len(checks) != len(expected_references):
        raise ValueError("verification session omitted terminal reference checks")
    for generation_index, (expected, check) in enumerate(zip(expected_references, checks, strict=True)):
        if not isinstance(check, dict):
            raise ValueError("verification session contains a malformed terminal reference check")
        if (
            _require_nonnegative_int(check.get("generation_index"), "terminal generation_index")
            != generation_index
            or _require_nonnegative_int(check.get("input_state_index"), "terminal input_state_index")
            != len(PROMPT_IDS) - 1 + generation_index
            or check.get("expected_token_id") != expected
        ):
            raise ValueError("verification reference token/input boundary drifted")
        if check.get("predicted_token_id") != expected or check.get("accepted") is not True:
            raise ValueError("verification terminal check failed")


def _verify_source_identity(doc: dict[str, Any]) -> None:
    """Reject a plausible session receipt from an unpinned or wrong body."""

    if doc.get("model") != REPO_ID:
        raise ValueError(f"source model mismatch: {doc.get('model')!r}")
    if doc.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("source revision mismatch")
    execution = doc.get("execution")
    if not isinstance(execution, dict) or execution.get("process_boundary") != "one native process":
        raise ValueError("source receipt did not establish one native process")


def _canonical_compact_utf8_sorted(document: dict[str, Any]) -> bytes:
    """Serialize the one canonical JSON form shared by seals and signatures."""
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _compact_utf8_sorted_seal(document: dict[str, Any]) -> str:
    """Match native ``serde_json``'s compact UTF-8, sorted-key receipt seal."""
    return hashlib.sha256(_canonical_compact_utf8_sorted(document)).hexdigest()


def _load_machine_admin_owner_public_key() -> bytes | None:
    """Use the established root-owned machine-admin trust anchor only."""
    return load_pinned_owner_public_key()


def _canonical_capture_repo_roots() -> tuple[Path, ...]:
    """Return only Hawking roots that may own a local canonical capture."""
    roots: list[Path] = [ROOT.resolve()]
    for extra in _EXTRA_CANONICAL_CAPTURE_REPO_ROOTS:
        try:
            resolved = extra.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_dir():
            roots.append(resolved)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return tuple(deduped)


def _path_is_allowlisted_odyssey_tool(path: Path, basename: str) -> bool:
    """Require exactly <allowlisted-root>/tools/odyssey/<basename>."""
    resolved = path.expanduser().resolve()
    if resolved.name != basename or not resolved.is_file():
        return False
    for root in _canonical_capture_repo_roots():
        expected = (root / "tools" / "odyssey" / basename).resolve()
        if resolved == expected:
            return True
    return False


def _verify_canonical_capture_artifact(
    binding: object,
    *,
    receipt: Path,
    field: str,
    basename: str,
    label: str,
) -> dict[str, str]:
    """Verify a current canonical tool or one exact read-only historic blob."""
    try:
        verified = _verify_artifact_binding(binding, base=receipt.parent, label=label)
    except ValueError as exc:
        historical = _verify_historical_local_capture_artifact(
            binding, receipt=receipt, field=field,
        )
        if historical is None:
            raise exc
        return historical
    path = Path(verified["path"])
    if not _path_is_allowlisted_odyssey_tool(path, basename):
        if field == "producer":
            raise ValueError(
                "noncanonical or foreign capture producer requires --owner-authorization "
                "signed by the machine-admin owner key; local canonical admission refused"
            )
        raise ValueError(f"{label} is not an allowlisted Hawking capture artifact")
    return verified


def _require_sibling_capture_document(receipt: Path) -> tuple[Path, dict[str, Any]]:
    """Require the original sibling capture evidence for local admission."""
    capture_path = (receipt.parent / "capture.json").resolve()
    if not capture_path.is_file():
        raise ValueError("local canonical admission requires sibling capture.json beside the reference contract")
    return capture_path, _json(capture_path)


def _admit_local_canonical_capture(
    *,
    receipt: Path,
    receipt_sha256: str,
    receipt_seal_sha256: str,
    provider: dict[str, Any],
    logits_trace: dict[str, Any],
    shard_ledger: dict[str, Any],
    manifest: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Admit a byte-bound local Hawking capture without an owner signature.

    This is the sealed TASK 029 route.  All source, provider, logits, and
    ledger validation has already completed before this provenance decision.
    """
    producer = _verify_canonical_capture_artifact(
        provider["producer"],
        receipt=receipt,
        field="producer",
        basename=CANONICAL_CAPTURE_PRODUCER_BASENAME,
        label="canonical capture producer",
    )
    producer_path = Path(producer["path"])
    if document.get("provider", {}).get("execution_schedule") != CANONICAL_CAPTURE_EXECUTION_SCHEDULE:
        raise ValueError("local canonical admission requires the sealed row-addressed PLE execution schedule")

    capture_path, capture = _require_sibling_capture_document(receipt)
    packager = capture.get("provider", {}).get("hawking_packager")
    if not isinstance(packager, dict):
        raise ValueError("sibling capture.json omitted hawking_packager provenance")
    packager = _verify_canonical_capture_artifact(
        packager,
        receipt=receipt,
        field="packager",
        basename=CANONICAL_CAPTURE_PACKAGER_BASENAME,
        label="canonical capture packager",
    )
    packager_path = Path(packager["path"])
    historical_producer = producer.get("historical_git_snapshot")
    historical_packager = packager.get("historical_git_snapshot")
    if bool(historical_producer) != bool(historical_packager):
        raise ValueError("historical local capture must bind both producer and packager snapshots")

    status = str(capture.get("status") or "")
    if not status.startswith("CAPTURED_EXTERNAL_COMPLETE_SOURCE_GREEDY_REFERENCE"):
        raise ValueError("sibling capture.json status is not a complete external greedy capture")
    historical_owner_auth_suffix = status.endswith("__OWNER_AUTHORIZATION_REQUIRED")
    capture_binding = {
        "path": str(capture_path),
        "sha256": _sha256_file(capture_path),
        "status": status,
        "hawking_packager_path": str(packager_path),
        "hawking_packager_sha256": packager["sha256"],
        "historical_owner_authorization_required_suffix": historical_owner_auth_suffix,
        "historical_suffix_policy": (
            "CAPTURED_...__OWNER_AUTHORIZATION_REQUIRED remains a sealed historical label; "
            "local-canonical admission supersedes the Ed25519 mandate without mutating capture.json."
            if historical_owner_auth_suffix
            else "capture status does not assert a pending Ed25519 mandate"
        ),
    }
    return {
        "admission_class": LOCAL_CANONICAL_ADMISSION_STATUS,
        "schema": LOCAL_CANONICAL_ADMISSION_SCHEMA,
        "authorization_scope": OWNER_AUTHORIZATION_SCOPE,
        "path": None,
        "sha256": None,
        "owner_signature_required": False,
        "owner_signature_present": False,
        "science_status": "UNEARNED",
        "science_unearned": True,
        "producer_path": str(producer_path),
        "producer_sha256": producer["sha256"],
        "reference_contract_sha256": receipt_sha256,
        "reference_contract_seal_sha256": receipt_seal_sha256,
        "provider_implementation_sha256": provider["implementation"]["sha256"],
        "provider_runtime_lock_sha256": provider["runtime_lock"]["sha256"],
        "logits_trace_sha256": logits_trace["sha256"],
        "logits_trace_seal_sha256": logits_trace["seal_sha256"],
        "shard_ledger_sha256": shard_ledger["sha256"],
        "shard_ledger_seal_sha256": shard_ledger["seal_sha256"],
        "model_lake_manifest_sha256": manifest["sha256"],
        "sibling_capture": capture_binding,
        "historical_git_snapshot": historical_producer,
        "contract_status_vocabulary_note": EXTERNAL_REFERENCE_STATUS_VOCABULARY_NOTE,
        "policy_note": (
            "Detached machine-owner Ed25519 is not required for provenance-complete canonical local source "
            "captures. Scientific correctness remains unearned and is established only by later source/native "
            "comparison."
        ),
        "claim_boundary": (
            "ADMITTED_LOCAL_CANONICAL_CAPTURE grants permission to use this provenance-bound external source "
            "capture as the teacher for bounded diagnostics only. It does not prove native Flash correctness, "
            "PLE parity, capability, complete NR, EBPW, TPS, or Pulsar promotion."
        ),
    }


def _resolve_v2_admission_authority(
    authorization_path: Path | None,
    *,
    receipt: Path,
    receipt_sha256: str,
    receipt_seal_sha256: str,
    provider: dict[str, Any],
    logits_trace: dict[str, Any],
    shard_ledger: dict[str, Any],
    manifest: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Use a supplied Ed25519 authorization, otherwise TASK 029 local admission."""
    if authorization_path is not None:
        verified = dict(_verify_owner_authorization(
            authorization_path,
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            receipt_seal_sha256=receipt_seal_sha256,
            provider=provider,
            logits_trace=logits_trace,
            shard_ledger=shard_ledger,
            manifest=manifest,
        ))
        verified.update({
            "admission_class": "ADMITTED_OWNER_AUTHORIZED",
            "owner_signature_required": True,
            "owner_signature_present": True,
            "science_status": "UNEARNED",
            "science_unearned": True,
            "contract_status_vocabulary_note": EXTERNAL_REFERENCE_STATUS_VOCABULARY_NOTE,
            "claim_boundary": (
                "This V2 contract is owner-authorized only for the stated local source identity and raw F32 "
                "greedy token chain. The signature authenticates the bound admission inputs, not provider "
                "semantics or execution. It does not establish native state continuity, TPS, EBPW, capability, "
                "or promotion."
            ),
        })
        return verified
    return _admit_local_canonical_capture(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        receipt_seal_sha256=receipt_seal_sha256,
        provider=provider,
        logits_trace=logits_trace,
        shard_ledger=shard_ledger,
        manifest=manifest,
        document=document,
    )


def _verify_owner_authorization(
    authorization_path: Path | None,
    *,
    receipt: Path,
    receipt_sha256: str,
    receipt_seal_sha256: str,
    provider: dict[str, Any],
    logits_trace: dict[str, Any],
    shard_ledger: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify an owner signature for imported, foreign, or explicit signed V2 admission."""
    if authorization_path is None:
        raise ValueError(
            "admitted V2 external reference requires --owner-authorization signed by the machine-admin owner key"
        )
    authorization_file = _resolve_regular_file(
        str(authorization_path),
        base=receipt.parent,
        label="owner-signed external-reference authorization",
    )
    if authorization_file == receipt:
        raise ValueError("owner-signed external-reference authorization may not be the reference contract itself")
    public_key = _load_machine_admin_owner_public_key()
    if public_key is None:
        raise ValueError(
            "machine-admin owner trust anchor is unavailable or fails ownership checks: "
            f"{pinned_owner_public_key_path()}"
        )
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("machine-admin owner trust anchor is not a 32-byte Ed25519 public key")
    authorization = _json(authorization_file)
    if set(authorization) != {
        "schema",
        "owner_public_key_sha256",
        "payload",
        "signature_ed25519_hex",
    } or authorization.get("schema") != OWNER_AUTHORIZATION_SCHEMA:
        raise ValueError("owner-signed external-reference authorization schema or fields differ")
    expected_key_sha256 = hashlib.sha256(public_key).hexdigest()
    if authorization.get("owner_public_key_sha256") != expected_key_sha256:
        raise ValueError("owner-signed external-reference authorization public-key fingerprint differs")
    payload = authorization.get("payload")
    required_payload = {
        "authorization_scope",
        "reference_contract_sha256",
        "reference_contract_seal_sha256",
        "provider_implementation_sha256",
        "provider_runtime_lock_sha256",
        "provider_producer_sha256",
        "logits_trace_sha256",
        "logits_trace_seal_sha256",
        "shard_ledger_sha256",
        "shard_ledger_seal_sha256",
        "model_lake_manifest_sha256",
        "issued_unix_ns",
        "expires_unix_ns",
        "nonce",
    }
    if not isinstance(payload, dict) or set(payload) != required_payload:
        raise ValueError("owner-signed external-reference authorization payload fields differ")
    expected_payload = {
        "authorization_scope": OWNER_AUTHORIZATION_SCOPE,
        "reference_contract_sha256": receipt_sha256,
        "reference_contract_seal_sha256": receipt_seal_sha256,
        "provider_implementation_sha256": provider["implementation"]["sha256"],
        "provider_runtime_lock_sha256": provider["runtime_lock"]["sha256"],
        "provider_producer_sha256": provider["producer"]["sha256"],
        "logits_trace_sha256": logits_trace["sha256"],
        "logits_trace_seal_sha256": logits_trace["seal_sha256"],
        "shard_ledger_sha256": shard_ledger["sha256"],
        "shard_ledger_seal_sha256": shard_ledger["seal_sha256"],
        "model_lake_manifest_sha256": manifest["sha256"],
    }
    if any(payload.get(field) != expected for field, expected in expected_payload.items()):
        raise ValueError("owner-signed external-reference authorization does not bind the exact V2 contract inputs")
    issued = _require_nonnegative_int(payload.get("issued_unix_ns"), "owner authorization issued_unix_ns")
    expires = _require_nonnegative_int(payload.get("expires_unix_ns"), "owner authorization expires_unix_ns")
    if expires < issued or not issued <= time.time_ns() <= expires:
        raise ValueError("owner-signed external-reference authorization is expired or not yet valid")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or len(nonce) < 32 or not nonce.strip():
        raise ValueError("owner-signed external-reference authorization nonce is absent or too short")
    signature_hex = authorization.get("signature_ed25519_hex")
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) != 128
        or any(character not in "0123456789abcdef" for character in signature_hex)
    ):
        raise ValueError("owner-signed external-reference authorization signature is not lowercase 64-byte hex")
    try:
        signature = bytes.fromhex(signature_hex)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            _canonical_compact_utf8_sorted(payload),
        )
    except (ValueError, InvalidSignature) as exc:
        raise ValueError("owner Ed25519 authorization signature verification failed") from exc
    return {
        "path": str(authorization_file),
        "sha256": _sha256_file(authorization_file),
        "owner_public_key_path": str(pinned_owner_public_key_path()),
        "owner_public_key_sha256": expected_key_sha256,
        "signature_ed25519_sha256": hashlib.sha256(signature_hex.encode("ascii")).hexdigest(),
        "issued_unix_ns": issued,
        "expires_unix_ns": expires,
    }


def _verify_runner_sidecar(
    sidecar: dict[str, Any],
    *,
    session_path: Path,
    session_doc: dict[str, Any],
    model_root: Path,
    references: list[int],
    source_input_identity: dict[str, Any],
    allow_candidate_route_observation: bool = False,
) -> None:
    """Check the sealed runner summary against the exact native session bytes."""
    model_root = model_root.expanduser().resolve()
    if (
        sidecar.get("schema") != RUNNER_SIDECAR_SCHEMA
        or sidecar.get("status") != RUNNER_SIDECAR_STATUS
    ):
        raise ValueError("runner sidecar has an incompatible schema or status")
    _verify_compact_seal(sidecar, "runner sidecar")
    expected_references = _require_token_list(references, "runner expected reference_generated_token_ids")
    reference_authority = sidecar.get("reference_authority")
    if (
        not isinstance(reference_authority, dict)
        or reference_authority.get("mode") != "admitted_external_complete_source_greedy_reference_v2"
        or reference_authority.get("source_input_identity") != source_input_identity
        or not isinstance(reference_authority.get("owner_authorization"), dict)
    ):
        raise ValueError("runner sidecar does not bind an owner-authorized V2 source authority")
    if (
        sidecar.get("model_root") != str(model_root)
        or sidecar.get("source_input_identity") != source_input_identity
        or sidecar.get("promotion_allowed") is not False
        or _require_token_list(sidecar.get("prompt_token_ids"), "runner prompt_token_ids") != list(PROMPT_IDS)
        or _require_token_list(
            sidecar.get("reference_generated_token_ids"),
            "runner reference_generated_token_ids",
        )
        != expected_references
    ):
        raise ValueError("runner sidecar token/source boundary differs from the verified session")
    session_path = session_path.resolve()
    native_session = sidecar.get("native_session")
    if not isinstance(native_session, dict):
        raise ValueError("runner sidecar omitted its native session binding")
    session_seal = _verify_native_compact_receipt_seal(session_doc)
    if (
        sidecar.get("verification_receipt") != str(session_path)
        or native_session.get("path") != str(session_path)
        or native_session.get("sha256") != _sha256_file(session_path)
        or native_session.get("seal_sha256") != session_seal
        or native_session.get("schema") != NATIVE_SESSION_SCHEMA
        or native_session.get("status")
        not in (
            {NATIVE_SESSION_STATUS, NATIVE_CANDIDATE_ROUTE_OBSERVATION_STATUS}
            if allow_candidate_route_observation
            else {NATIVE_SESSION_STATUS}
        )
        or native_session.get("root") != str(model_root)
        or native_session.get("source_input_identity") != source_input_identity
    ):
        raise ValueError("runner sidecar does not bind the exact native session path, bytes, seal, schema, and status")
    _verify_repeated(
        session_doc,
        expected_references,
        model_root=model_root,
        source_input_identity=source_input_identity,
        allow_candidate_route_observation=allow_candidate_route_observation,
    )


def _write_new_runner_sidecar(path: Path, sidecar: dict[str, Any]) -> None:
    """Create the wrapper-owned sidecar once; never replace a racing writer."""
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise ValueError(
            f"runner sidecar path became occupied after native execution; refusing to overwrite: {path}"
        ) from exc


def _verify_ple_pre_layer_state_bank(
    manifest_path: Path,
    *,
    session_path: Path,
    session_doc: dict[str, Any],
    token_ids: list[int],
    prompt_len: int,
) -> dict[str, Any]:
    """Require every exported state to bind the exact accepted session.

    This is deliberately stricter than a directory-exists check.  The bank is
    an input to a future PLE function experiment, so accidental reuse from a
    different route/session, stale raw state, or a rewritten manifest must be
    rejected before any learned representation consumes it.
    """
    manifest_path = manifest_path.resolve()
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != PLE_PRE_LAYER_STATE_BANK_SCHEMA
        or manifest.get("status") != PLE_PRE_LAYER_STATE_BANK_STATUS
        or manifest.get("model") != REPO_ID
        or manifest.get("pinned_revision") != PINNED_REVISION
    ):
        raise ValueError("pre-PLE state-bank manifest has an incompatible source identity")
    seal = manifest.get("seal_sha256")
    body = dict(manifest)
    body.pop("seal_sha256", None)
    if not isinstance(seal, str) or seal != _compact_utf8_sorted_seal(body):
        raise ValueError("pre-PLE state-bank manifest has an invalid native seal")
    session = manifest.get("session_receipt")
    if not isinstance(session, dict):
        raise ValueError("pre-PLE state-bank manifest omitted its session binding")
    session_bytes = session_path.read_bytes()
    if (
        session.get("path") != str(session_path)
        or session.get("sha256") != hashlib.sha256(session_bytes).hexdigest()
        or session.get("seal_sha256") != session_doc.get("seal_sha256")
        or session.get("schema") != session_doc.get("schema")
        or session.get("status") != session_doc.get("status")
    ):
        raise ValueError("pre-PLE state-bank manifest does not bind the exact session receipt")
    if manifest.get("token_ids") != token_ids or manifest.get("prompt_length") != prompt_len:
        raise ValueError("pre-PLE state-bank token sequence differs from the accepted session")
    capture = manifest.get("capture")
    if not isinstance(capture, dict) or capture.get("layer") != 0:
        raise ValueError("pre-PLE state-bank did not capture the layer-0 PLE boundary")
    if capture.get("state_width") != PLE_PRE_LAYER_STATE_WIDTH:
        raise ValueError("pre-PLE state-bank state width differs from Flash PLE geometry")
    if capture.get("upstream_PLE_inclusive_source_trajectory") is not False:
        raise ValueError("pre-PLE state-bank did not declare the native PLE trajectory boundary")
    records = manifest.get("states")
    if not isinstance(records, list) or len(records) != len(token_ids):
        raise ValueError("pre-PLE state-bank has incomplete token-state coverage")
    payload_bytes = 0
    for step, (record, token_id) in enumerate(zip(records, token_ids, strict=True)):
        if not isinstance(record, dict):
            raise ValueError("pre-PLE state-bank record is not an object")
        state_path = Path(str(record.get("path", ""))).expanduser()
        if not state_path.is_absolute():
            state_path = (manifest_path.parent / state_path).resolve()
        raw = state_path.read_bytes()
        if (
            record.get("step") != step
            or record.get("token_id") != token_id
            or record.get("layer") != 0
            or record.get("dtype") != "F32_LE"
            or record.get("elements") != PLE_PRE_LAYER_STATE_WIDTH
            or record.get("bytes") != PLE_PRE_LAYER_STATE_WIDTH * 4
            or len(raw) != PLE_PRE_LAYER_STATE_WIDTH * 4
            or record.get("sha256") != hashlib.sha256(raw).hexdigest()
            or record.get("finite") is not True
        ):
            raise ValueError(f"pre-PLE state-bank record {step} does not bind its F32 payload")
        payload_bytes += len(raw)
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "seal_sha256": seal,
        "state_count": len(records),
        "state_width": PLE_PRE_LAYER_STATE_WIDTH,
        "payload_bytes": payload_bytes,
        "claim_boundary": (
            "The bank contains source-bound native layer-0 boundary snapshots from an accepted stateful session. "
            "Its current native trajectory does not embed PLE, so it is only an input control for subsequent "
            "PLE function experiments, not PLE/model parity, "
            "a representation, EBPW, capability, TPS, or promotion evidence."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json",
    )
    parser.add_argument(
        "--native-binary",
        type=Path,
        default=_native_binary_path(),
        help="prebuilt, inspected stateful-session executable; cargo run is intentionally not used here",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="deprecated compatibility flag; it cannot bypass the protected native-lane preflight",
    )
    parser.add_argument(
        "--reference-contract",
        type=Path,
        help=(
            "sealed PLE-inclusive full-source greedy-reference receipt; required for a non-dry repeated-"
            "acceptance claim. Admission is via --owner-authorization or local-canonical provenance"
        ),
    )
    parser.add_argument(
        "--owner-authorization",
        type=Path,
        help=(
            "optional detached Ed25519 authorization for --reference-contract; required for foreign/noncanonical "
            "captures, while a provenance-complete local-canonical capture may omit it"
        ),
    )
    parser.add_argument(
        "--ple-pre-layer-state-bank-out",
        type=Path,
        help=(
            "empty directory for exact layer-0 source states from phase 2; defaults to a "
            "new sibling of --out"
        ),
    )
    parser.add_argument(
        "--token-major-resident-banks",
        action="store_true",
        help=(
            "run the accepted session with all 48 token-major banks resident; this is a bounded body/replay "
            "control and does not qualify capability TPS"
        ),
    )
    parser.add_argument(
        "--route-teacher",
        type=Path,
        help=(
            "sealed token-major route observation consumed for the resident benchmark; a candidate observation "
            "is permitted only with --candidate-route-observation"
        ),
    )
    parser.add_argument(
        "--candidate-route-observation",
        action="store_true",
        help=(
            "run the explicitly non-promotable compact candidate route/state observation and withhold its "
            "per-layer state-hash comparison"
        ),
    )
    parser.add_argument(
        "--token-major-clean-replay-reps",
        type=int,
        default=0,
        help=(
            "after accepted token-major setup, replay the known chain this many times without diagnostics; "
            "requires --token-major-resident-banks"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.token_major_clean_replay_reps < 0:
        parser.error("--token-major-clean-replay-reps must be non-negative")
    if args.token_major_clean_replay_reps > 0 and not args.token_major_resident_banks:
        parser.error("--token-major-clean-replay-reps requires --token-major-resident-banks")
    if args.candidate_route_observation and args.route_teacher is None:
        parser.error("--candidate-route-observation requires --route-teacher")
    if args.route_teacher is not None and not args.token_major_resident_banks:
        parser.error("--route-teacher requires --token-major-resident-banks")
    model_root = args.root.resolve()
    out = args.out.resolve()
    native_binary = args.native_binary.expanduser()
    if not native_binary.is_absolute():
        native_binary = ROOT / native_binary
    route_teacher = (
        args.route_teacher.expanduser().resolve()
        if args.route_teacher is not None
        else None
    )
    state_bank_out = (
        args.ple_pre_layer_state_bank_out.resolve()
        if args.ple_pre_layer_state_bank_out is not None
        else out.with_name(f"{out.stem}.ple_pre_layer_state_bank")
    )
    runner_sidecar = out.with_name(f"{out.stem}.runner.json")
    if not args.dry_run:
        runner_sidecar = _reserve_fresh_run_outputs(out, state_bank_out)
    reference_authority: dict[str, Any] | None = None
    references: list[int] | None = None
    if args.reference_contract is not None:
        reference_authority = load_external_reference_contract(
            args.reference_contract,
            model_root,
            owner_authorization_path=args.owner_authorization,
        )
        references = list(reference_authority["generated_token_ids"])

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "reference_contract": reference_authority or {
                "required": True,
                "reopen_condition": (
                    "supply --reference-contract with either a detached --owner-authorization or complete "
                    "local-canonical provenance, plus canonical ModelLake, shard-ledger, provider-artifact, "
                    "and F32-logits bindings"
                ),
            },
            "phase_1": "no Hawking-native reference discovery is permitted",
            "phase_2": "fresh native verification over the externally sealed reference chain",
            "phase_2_ple_pre_layer_state_bank": str(state_bank_out),
            "native_binary": {
                "path": str(native_binary),
                "exists": native_binary.is_file() and not native_binary.is_symlink(),
                "reopen_condition": "prebuild and inspect the exact native executable before a non-dry run",
                "claim_boundary": "an observed executable/source closure is not compiler provenance or a build attestation",
            },
            "claim_boundary": "no GPU work, no repeated-decode result",
        }, indent=2))
        return 0
    if references is None or reference_authority is None:
        raise ValueError(
            "non-dry repeated acceptance requires an admitted --reference-contract; native discovery and prior "
            "native sessions are self-controls, not source reference authority"
        )
    source_input_identity = reference_authority.get("source_input_identity")
    if not isinstance(source_input_identity, dict):
        raise ValueError("admitted V2 reference authority omitted the expected native source_input_identity")
    lane_preflight = _protected_native_lane()
    if not lane_preflight["clean"]:
        print(json.dumps({
            "status": "BLOCKED_CLEAN_GPU_LANE",
            "reason": "a protected Hawking native/provider lane is visible; refusing a concurrent Flash run",
            "observed_native_lane_preflight": lane_preflight,
            "allow_busy_gpu_ignored": args.allow_busy_gpu,
            "reopen_condition": "quiesce every listed non-zombie protected native/provider process, then rerun this exact command",
        }, indent=2))
        return 3
    if not model_root.is_dir():
        raise FileNotFoundError(f"Flash source root does not exist: {model_root}")

    started_ns = time.time_ns()
    build = _preexecution_build_binding(native_binary)
    verification_tokens = [*PROMPT_IDS, *references]
    verification_command = _flash_binary_command(
        native_binary,
        model_root,
        verification_tokens,
        len(PROMPT_IDS),
        out,
        ple_pre_layer_state_bank_out=state_bank_out,
        route_teacher=route_teacher,
        token_major_resident_banks=args.token_major_resident_banks,
        token_major_clean_replay_reps=args.token_major_clean_replay_reps,
        candidate_route_observation=args.candidate_route_observation,
    )
    _run(verification_command)
    if not _same_executable_closure(build, _preexecution_build_binding(native_binary)):
        raise ValueError(
            "native executable or its declared source closure changed during repeated-decode execution; "
            "withholding runner sidecar"
        )
    final_doc = _json(out)
    _verify_repeated(
        final_doc,
        references,
        model_root=model_root,
        source_input_identity=source_input_identity,
        allow_candidate_route_observation=args.candidate_route_observation,
    )
    state_bank = _verify_ple_pre_layer_state_bank(
        state_bank_out / "manifest.json",
        session_path=out,
        session_doc=final_doc,
        token_ids=verification_tokens,
        prompt_len=len(PROMPT_IDS),
    )
    run_receipt = {
        "schema": RUNNER_SIDECAR_SCHEMA,
        "status": RUNNER_SIDECAR_STATUS,
        "model_root": str(model_root),
        "source_input_identity": source_input_identity,
        "prompt_token_ids": list(PROMPT_IDS),
        "reference_generated_token_ids": references,
        "reference_authority": reference_authority,
        "verification_receipt": str(out),
        "native_session": {
            "path": str(out),
            "sha256": _sha256_file(out),
            "seal_sha256": _verify_native_compact_receipt_seal(final_doc),
            "schema": final_doc.get("schema"),
            "status": final_doc.get("status"),
            "root": final_doc.get("root"),
            "source_input_identity": final_doc.get("source_input_identity"),
        },
        "build": build,
        "native_lane_preflight": lane_preflight,
        "ple_pre_layer_state_bank": state_bank,
        "verification_session": (
            "one source-bound session; prompt prefill occurs once at its start and no token-boundary "
            "reset or re-prefill is claimed"
        ),
        "elapsed_ns": time.time_ns() - started_ns,
        "claim_boundary": (
            "This runner proves only stateful native terminal agreement with an owner-authorized source-reference "
            "chain and an observed pre/post executable/source closure. That closure is not compiler provenance or "
            "a build attestation. The lane record is a point-in-time preflight, not a timing witness. This does not "
            "establish TPS, capability, EBPW, direct Noetic execution, residency, or promotion."
        ),
        "promotion_allowed": False,
    }
    run_receipt["seal_sha256"] = _compact_utf8_sorted_seal(run_receipt)
    _verify_runner_sidecar(
        run_receipt,
        session_path=out,
        session_doc=final_doc,
        model_root=model_root,
        references=references,
        source_input_identity=source_input_identity,
        allow_candidate_route_observation=args.candidate_route_observation,
    )
    _write_new_runner_sidecar(runner_sidecar, run_receipt)
    print(json.dumps(run_receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "FAILED_PRECISE_DIAGNOSIS", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
